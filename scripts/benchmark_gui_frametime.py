"""
scripts/benchmark_gui_frametime.py — 量測 GUI 實際跑起來的 frametime 組成，
用來判斷 FPS 偏低到底卡在 physics、render 還是 CPU 側。

方法論照 Isaac Sim 官方 benchmark 的判讀方式（見
https://docs.isaacsim.omniverse.nvidia.com/6.0.0/reference_material/benchmarks.html）：

- Physics frametime 貼近 App_Update frametime → 物理是瓶頸
- GPU frametime 貼近 App_Update frametime     → 算圖是瓶頸
- GPU frametime 遠低於 App_Update frametime   → CPU 側才是瓶頸

四個 recorder 的取數方式直接照抄 `isaacsim.benchmark.services` 的
datarecorders（app_frametime.py／physics_frametime.py／render_frametime.py／
gpu_frametime.py），但不引入整套 benchmark 框架——那套要配 metrics backend、
phase 管理，對「找瓶頸」這件事是多餘的。

額外多做一件官方 recorder 沒做的事：`subscribe_profile_stats_events()` 其實
會送**所有** PhysX zone（官方只挑 "PhysX Update" 一個），這裡全部收下來累計，
可以直接看出物理時間花在哪個 zone。

環境變數：
    BENCH_HEADLESS=1        無視窗（預設 0，開真的視窗才能重現 GUI 的 FPS）
    BENCH_FRAMES=600        取樣幀數（預設 600）
    BENCH_WARMUP=180        暖機幀數，不計入統計（預設 180）
    BENCH_TRAINING=0/1      是否建立 Training 球檯（預設照 extension 自己的預設）
    BENCH_LABEL=xxx         輸出標籤，方便多次比較
    BENCH_RENDERER=...      覆寫 renderer（RaytracedLighting／RealTimePathTracing…）

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    PYTHONIOENCODING=utf-8 \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/benchmark_gui_frametime.py
"""

import json
import os
import statistics
import sys
import time
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HEADLESS = os.environ.get("BENCH_HEADLESS", "0") == "1"
_FRAMES = int(os.environ.get("BENCH_FRAMES", "600"))
_WARMUP = int(os.environ.get("BENCH_WARMUP", "180"))
_LABEL = os.environ.get("BENCH_LABEL", "baseline")
_RENDERER = os.environ.get("BENCH_RENDERER", "")
_RESULT_DIR = os.path.join(_PROJECT_ROOT, "_bench")


class _Collector:
    """四路取樣器合一。每一路的訂閱方式都跟官方 recorder 相同，差別只在
    這裡是同一個物件同時收，且 physics 那一路收下全部 zone。"""

    def __init__(self) -> None:
        self.app_samples: list[float] = []
        self.render_samples: list[float] = []
        self.physx_update_samples: list[float] = []
        self.gpu_samples: list[float] = []
        # zone_name -> 累計 ms 與次數，用來看物理時間的內部組成
        self.physx_zones: dict[str, list[float]] = {}
        self._collecting = False
        self._app_last_ns = 0
        self._render_last_ns = 0
        self._subs: list = []
        self._physics_sub = None
        self._hydra_stats = None

    def attach(self) -> None:
        import carb.eventdispatcher
        import omni.kit.app

        dispatcher = carb.eventdispatcher.get_eventdispatcher()
        self._app_last_ns = time.perf_counter_ns()
        self._render_last_ns = self._app_last_ns
        self._subs.append(
            dispatcher.observe_event(
                event_name=omni.kit.app.GLOBAL_EVENT_PRE_UPDATE,
                on_event=self._on_app_update,
                observer_name="bench_app_frametime",
            )
        )
        self._subs.append(
            dispatcher.observe_event(
                event_name="runloop:rendering_0:update",
                on_event=self._on_render_update,
                observer_name="bench_render_frametime",
            )
        )

        try:
            import omni.physics.core

            iface = omni.physics.core.get_physics_benchmarks_interface()
            self._physics_sub = iface.subscribe_profile_stats_events(self._on_physics_stats)
        except Exception as exc:  # noqa: BLE001 - 量測工具，缺了就少一路資料，不該中斷
            print(f"[bench] 取不到 physics benchmarks interface：{exc}")

        try:
            from omni.hydra.engine.stats import HydraEngineStats

            self._hydra_stats = HydraEngineStats()
        except Exception as exc:  # noqa: BLE001
            print(f"[bench] 取不到 HydraEngineStats（GPU frametime 會缺）：{exc}")

    def detach(self) -> None:
        self._subs = []
        self._physics_sub = None

    def set_collecting(self, value: bool) -> None:
        self._collecting = value

    def _on_app_update(self, event) -> None:
        now = time.perf_counter_ns()
        dt_ms = (now - self._app_last_ns) / 1_000_000
        self._app_last_ns = now
        if self._collecting:
            self.app_samples.append(dt_ms)
            self._sample_gpu()

    def _on_render_update(self, event) -> None:
        now = time.perf_counter_ns()
        dt_ms = (now - self._render_last_ns) / 1_000_000
        self._render_last_ns = now
        if self._collecting:
            self.render_samples.append(dt_ms)

    def _on_physics_stats(self, profile_stats) -> None:
        if not self._collecting:
            return
        for stat in profile_stats:
            self.physx_zones.setdefault(stat.zone_name, []).append(stat.ms)
            if stat.zone_name == "PhysX Update":
                self.physx_update_samples.append(stat.ms)

    def _sample_gpu(self) -> None:
        if not self._hydra_stats:
            return
        try:
            nodes = self._hydra_stats.get_gpu_profiler_result()
        except Exception:  # noqa: BLE001
            return
        if not nodes:
            return
        # get_gpu_profiler_result() 回傳每張 GPU 的 node 樹，root 的
        # duration 就是該幀的 GPU 總時間（跟官方 gpu_frametime.py 取法相同）
        try:
            total = sum(node[0]["duration"] for node in nodes if node)
        except Exception:  # noqa: BLE001
            return
        self.gpu_samples.append(total)


def _stats(name: str, samples: list[float]) -> dict:
    if not samples:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": len(samples),
        "mean": round(statistics.fmean(samples), 3),
        "median": round(statistics.median(samples), 3),
        "min": round(min(samples), 3),
        "max": round(max(samples), 3),
        "p95": round(sorted(samples)[int(len(samples) * 0.95) - 1], 3),
    }


def _print_row(label: str, s: dict) -> None:
    if s.get("n", 0) == 0:
        print(f"  {label:<28} 無資料")
        return
    print(
        f"  {label:<28} mean={s['mean']:>8.3f}ms  median={s['median']:>8.3f}  "
        f"p95={s['p95']:>8.3f}  max={s['max']:>9.3f}  n={s['n']}"
    )


_TICK_NS: list[int] = []


def _instrument_extension_tick() -> bool:
    """把 extension 的 `_on_tick` 換成計時版本。

    這件事很重要：`BilliardExtension` 是用
    `SimulationManager.register_callback(..., event=SimulationEvent.PHYSICS_POST_STEP)`
    註冊 tick 的，也就是我們自己的 Python 邏輯（RMPflow 計算、observation
    組裝、每顆球的速度讀取）是在**物理步進裡面**被呼叫的，因此會整包算進
    "PhysX Update" 這個 zone。不分開量的話，會把自己的成本誤判成 PhysX
    解算太慢。

    用 gc 找出活著的 extension 實例，而不是改 production 程式碼——量測工具
    不該為了量測去污染被量測的對象。"""
    import gc

    from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

    # 不靠模組名稱（Kit 的 extension loader 用什麼名字載入不保證），直接用
    # 類別名稱在物件堆裡找
    instances = [
        o for o in gc.get_objects()
        if type(o).__name__ == "BilliardExtension" and hasattr(o, "_tick_callback_id")
    ]
    if not instances:
        print("[bench] 找不到 BilliardExtension 實例，跳過 tick 計時")
        return False

    extension = instances[0]
    original = extension._on_tick
    if extension._tick_callback_id is not None:
        SimulationManager.deregister_callback(extension._tick_callback_id)

    def _timed_tick(step_dt, context):
        started = time.perf_counter_ns()
        try:
            original(step_dt, context)
        finally:
            _TICK_NS.append(time.perf_counter_ns() - started)

    extension._tick_callback_id = SimulationManager.register_callback(
        _timed_tick, event=SimulationEvent.PHYSICS_POST_STEP
    )
    print("[bench] 已掛上 _on_tick 計時器（分離『我們的 Python 邏輯』與『PhysX 解算』）")
    return True


_API_NS: dict[str, list[int]] = {}


def _instrument_rigid_body_api() -> bool:
    """統計 `_on_tick` 裡對 RigidBodyAPI／ArticulationAPI 的呼叫次數與耗時。

    懷疑點：`RigidBodyAPIImpl` 是每個 prim path 各建一個單一 prim 的
    `RigidPrim` view，`get_position()`／`get_linear_velocity()` 每次呼叫都是
    一次獨立的 tensor 讀取（`.list()` 會強制 GPU→CPU 同步）。
    ObservationBuilder 每 tick 讀 10 顆球的位置、BallMotionMonitor 再讀 10 次
    速度、RollingResistanceService 又讀 10 次——如果單次成本是 0.4ms 等級，
    光這裡就吃掉整個 tick。先量出來再決定要不要改成批次讀取。"""
    import gc

    targets = ("RigidBodyAPIImpl", "ArticulationAPIImpl")
    instances = [o for o in gc.get_objects() if type(o).__name__ in targets]
    if not instances:
        print("[bench] 找不到 API 實作實例，跳過 API 計時")
        return False

    patched = 0
    for instance in instances:
        cls_name = type(instance).__name__
        for method_name in dir(instance):
            if method_name.startswith("_"):
                continue
            attribute = getattr(instance, method_name, None)
            if not callable(attribute) or not hasattr(attribute, "__self__"):
                continue

            key = f"{cls_name}.{method_name}"

            def _make(bound, bucket):
                def _wrapper(*args, **kwargs):
                    started = time.perf_counter_ns()
                    try:
                        return bound(*args, **kwargs)
                    finally:
                        _API_NS.setdefault(bucket, []).append(time.perf_counter_ns() - started)

                return _wrapper

            try:
                setattr(instance, method_name, _make(attribute, key))
                patched += 1
            except Exception:  # noqa: BLE001 - 有些屬性不可覆寫，跳過即可
                continue

    print(f"[bench] 已對 {len(instances)} 個 API 實例的 {patched} 個方法掛上計時")
    return True


def _apply_ablations() -> list[str]:
    """依環境變數逐項關掉可疑成本，用來一次驗證一個假設（跟本次對話稍早
    `profile_ur10e_tick_ablation.py` 同一套做法）。必須在 timeline.play()
    之前呼叫——PhysX 是在 play 的時候才去讀 /PhysicsScene 的屬性。"""
    import carb
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    applied: list[str] = []
    stage = omni.usd.get_context().get_stage()
    settings = carb.settings.get_settings()

    gpu_dynamics = os.environ.get("BENCH_GPU_DYNAMICS", "")
    if gpu_dynamics:
        want = gpu_dynamics == "1"
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Scene):
                continue
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
            physx_scene.CreateEnableGPUDynamicsAttr().Set(want)
            physx_scene.CreateBroadphaseTypeAttr().Set("GPU" if want else "MBP")
            applied.append(f"enableGPUDynamics={want}")
            break

    if os.environ.get("BENCH_ROOM_COLLIDERS", "") == "0":
        count = 0
        for prim in stage.Traverse():
            if "SimpleRoom" not in str(prim.GetPath()):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            count += 1
        applied.append(f"SimpleRoom collider 關閉 {count} 個")

    if os.environ.get("BENCH_ROOM_VISIBLE", "") == "0":
        count = 0
        for prim in stage.Traverse():
            if str(prim.GetPath()).endswith("SimpleRoom"):
                UsdGeom.Imageable(prim).MakeInvisible()
                count += 1
        applied.append(f"SimpleRoom 隱藏 {count} 個")

    if os.environ.get("BENCH_UPDATE_VELOCITIES", "") == "0":
        settings.set("/physics/updateVelocitiesToUsd", False)
        applied.append("updateVelocitiesToUsd=False")

    if os.environ.get("BENCH_UPDATE_TO_USD", "") == "0":
        settings.set("/physics/updateToUsd", False)
        applied.append("updateToUsd=False")

    min_frame_rate = os.environ.get("BENCH_MIN_FRAME_RATE", "")
    if min_frame_rate:
        settings.set("/persistent/simulation/minFrameRate", int(min_frame_rate))
        applied.append(f"minFrameRate={min_frame_rate}")

    if os.environ.get("BENCH_ASYNC_RENDER", "") == "1":
        settings.set("/app/asyncRendering", True)
        settings.set("/app/asyncRenderingLowLatency", True)
        applied.append("asyncRendering=True")

    if applied:
        print(f"[bench] 已套用 ablation：{'；'.join(applied)}")
    return applied


def _dump_physics_scene() -> dict:
    """列出物理場景的實際組成。61ms 的 PhysX Update 對 20 顆球來說完全不合理，
    要先知道場景裡到底有多少 collider、是什麼近似型別、以及 scene 本身的
    solver/substep 設定，才有辦法判斷時間花在哪。"""
    import carb
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    rigid_body_count = 0
    collider_count = 0
    approximations: dict[str, int] = {}
    articulation_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_count += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collider_count += 1
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
                key = str(attr.Get()) if attr and attr.Get() else "(未設定)"
            else:
                key = prim.GetTypeName() or "(非 Mesh)"
            approximations[key] = approximations.get(key, 0) + 1

    scene_settings: dict[str, object] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        physx_scene = PhysxSchema.PhysxSceneAPI(prim)
        for name, attr in (
            ("timeStepsPerSecond", physx_scene.GetTimeStepsPerSecondAttr()),
            ("solverType", physx_scene.GetSolverTypeAttr()),
            ("broadphaseType", physx_scene.GetBroadphaseTypeAttr()),
            ("enableGPUDynamics", physx_scene.GetEnableGPUDynamicsAttr()),
            ("enableCCD", physx_scene.GetEnableCCDAttr()),
            ("enableStabilization", physx_scene.GetEnableStabilizationAttr()),
        ):
            if attr:
                scene_settings[f"{prim.GetPath()}:{name}"] = attr.Get()
        break

    carb_settings = carb.settings.get_settings()
    for key in (
        "/persistent/simulation/minFrameRate",
        "/physics/updateToUsd",
        "/physics/updateVelocitiesToUsd",
        "/physics/updateParticlesToUsd",
        "/physics/updateForceSensorsToUsd",
        "/physics/outputVelocitiesLocalSpace",
        "/physics/fabricUpdateTransformations",
        "/physics/disableContactProcessing",
        "/app/asyncRendering",
        "/app/runLoops/main/rateLimitEnabled",
        "/app/runLoops/main/rateLimitFrequency",
    ):
        scene_settings[key] = carb_settings.get(key)

    info = {
        "rigid_bodies": rigid_body_count,
        "colliders": collider_count,
        "articulation_roots": articulation_count,
        "collider_approximations": approximations,
        "scene_settings": scene_settings,
    }

    print("-" * 78)
    print(f"[bench] 物理場景組成：rigid_body={rigid_body_count} collider={collider_count} "
          f"articulation_root={articulation_count}")
    print("[bench] collider 近似型別分佈：")
    for key, count in sorted(approximations.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {key:<32} {count}")
    print("[bench] 場景/設定：")
    for key, value in scene_settings.items():
        print(f"    {key:<52} {value}")
    return info


def _run(simulation_app) -> None:
    import omni.kit.app
    import omni.timeline

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.add_path(_EXT_DIR)
    for _ in range(10):
        simulation_app.update()
    manager.set_extension_enabled_immediate("billiard_digital_twin", True)
    print("[bench] billiard_digital_twin 已啟用，等待場景建立…")
    for _ in range(120):
        simulation_app.update()

    _instrument_extension_tick()
    _instrument_rigid_body_api()
    ablations = _apply_ablations()
    physics_info = _dump_physics_scene()
    physics_info["ablations"] = ablations

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("[bench] timeline.play()，暖機中…")

    collector = _Collector()
    collector.attach()

    for _ in range(_WARMUP):
        simulation_app.update()

    _TICK_NS.clear()
    _API_NS.clear()
    collector.set_collecting(True)
    wall_start = time.perf_counter()
    for _ in range(_FRAMES):
        simulation_app.update()
    wall_elapsed = time.perf_counter() - wall_start
    collector.set_collecting(False)
    collector.detach()

    app_stats = _stats("app_update", collector.app_samples)
    physics_stats = _stats("physx_update", collector.physx_update_samples)
    render_stats = _stats("render_thread", collector.render_samples)
    gpu_stats = _stats("gpu", collector.gpu_samples)

    print()
    print("=" * 78)
    print(f"[bench] label={_LABEL}  headless={_HEADLESS}  frames={_FRAMES}  "
          f"renderer={_RENDERER or '(預設)'}")
    print(f"[bench] 牆鐘 {wall_elapsed:.2f}s / {_FRAMES} 幀 = "
          f"{_FRAMES / wall_elapsed:.2f} FPS（外圈實測）")
    print("-" * 78)
    _print_row("App_Update frametime", app_stats)
    _print_row("Physics (PhysX Update)", physics_stats)
    _print_row("Render thread", render_stats)
    _print_row("GPU (Hydra)", gpu_stats)

    tick_ms = [ns / 1_000_000 for ns in _TICK_NS]
    tick_stats = _stats("extension_on_tick", tick_ms)
    _print_row("我們的 _on_tick（含在物理裡）", tick_stats)
    if tick_stats.get("n") and physics_stats.get("n"):
        substeps = tick_stats["n"] / len(collector.app_samples)
        tick_per_frame = tick_stats["mean"] * substeps
        print(f"  → 每個 app frame 跑 {substeps:.2f} 個物理 substep，"
              f"_on_tick 每 frame 合計 {tick_per_frame:.3f}ms "
              f"（PhysX Update 的 {tick_per_frame / physics_stats['mean'] * 100:.1f}%）")
    print("-" * 78)

    if app_stats.get("n"):
        app_mean = app_stats["mean"]
        print(f"[bench] Mean FPS（App_Update 推算）= {1000 / app_mean:.2f}")
        for label, s in (("Physics", physics_stats), ("GPU", gpu_stats)):
            if s.get("n"):
                ratio = s["mean"] / app_mean * 100
                print(f"[bench] {label} 佔 App_Update 的 {ratio:.1f}%")

    if _API_NS:
        print("-" * 78)
        frame_count = len(collector.app_samples) or 1
        print(f"[bench] API 呼叫熱點（依每 frame 總耗時排序，共 {frame_count} frame）：")
        api_rows = sorted(
            (
                (name, sum(v) / 1_000_000 / frame_count, len(v) / frame_count,
                 statistics.fmean(v) / 1_000_000)
                for name, v in _API_NS.items()
            ),
            key=lambda r: r[1],
            reverse=True,
        )
        for name, per_frame_ms, calls_per_frame, mean_ms in api_rows[:12]:
            if per_frame_ms < 0.01:
                continue
            print(f"    {name:<44} {per_frame_ms:>7.3f}ms/frame  "
                  f"{calls_per_frame:>5.1f} 次/frame  單次 {mean_ms:.4f}ms")

    if collector.physx_zones:
        print("-" * 78)
        print("[bench] PhysX zone 分解（依平均耗時排序，前 15 名）：")
        rows = sorted(
            ((n, statistics.fmean(v), len(v)) for n, v in collector.physx_zones.items()),
            key=lambda r: r[1],
            reverse=True,
        )
        for zone_name, mean_ms, count in rows[:15]:
            print(f"    {zone_name:<44} mean={mean_ms:>8.3f}ms  n={count}")
    print("=" * 78)

    os.makedirs(_RESULT_DIR, exist_ok=True)
    out_path = os.path.join(_RESULT_DIR, f"{_LABEL}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "label": _LABEL,
                "headless": _HEADLESS,
                "frames": _FRAMES,
                "renderer": _RENDERER,
                "wall_fps": round(_FRAMES / wall_elapsed, 3),
                "app_update": app_stats,
                "physics": physics_stats,
                "render_thread": render_stats,
                "gpu": gpu_stats,
                "physics_scene": physics_info,
                "extension_on_tick": tick_stats,
                "physx_zones": {
                    n: {"mean": round(statistics.fmean(v), 4), "n": len(v)}
                    for n, v in collector.physx_zones.items()
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[bench] 結果已寫入 {out_path}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    config = {"headless": _HEADLESS}
    if _RENDERER:
        config["renderer"] = _RENDERER

    simulation_app = SimulationApp(config)
    try:
        _run(simulation_app)
    except Exception:
        print("[bench] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
