import math
from typing import Callable, Optional

import omni.usd
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import (
    ContactEventHeaderVector,
    ContactDataVector,
    ContactEventType,
    TriggerEventData,
    TriggerEventType,
)
from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools

from core.ports.physics_api import PhysicsAPI
from core.models.contact_event import ContactEvent


class PhysicsAPIImpl(PhysicsAPI):
    """
    混合實作：動態剛體（球）用 PhysX Contact Report，靜態感測區（球袋）用
    PhysX Trigger Volume，兩者是完全不同的底層訂閱機制，這裡統一轉譯成
    同一種 ContactEvent 交給呼叫端（見 core/services/pocket_event_handler.py
    的用法：pocket 跟 ball 的 prim path 都呼叫同一個 enable_contact_reporting()，
    共用同一個 subscribe_contact_events() callback）。

    球袋不能用 Contact Report——那會產生真正的物理碰撞反應，球會被彈開/擋住，
    不符合「球要真的掉進去，但要偵測到」的需求；Trigger 只做幾何相交測試，
    不會有碰撞反應，但也因此沒有 impulse/position 等資訊，Trigger 事件的
    impulse 只能填 0.0 佔位。
    """

    def __init__(self) -> None:
        self._contact_sub = None  # carb.Subscription，RAII，設 None 即取消訂閱
        self._trigger_sub_id: Optional[int] = None  # int id，需手動 unsubscribe

    def enable_contact_reporting(self, prim_path: str) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # 動態剛體（球）：threshold=0 代表任何碰撞都回報（預設是 1.0）
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(0.0)
        else:
            # 靜態感測區（球袋）：CollisionAPI+PhysxTriggerAPI 讓 PhysX 把這個
            # shape 排除在碰撞回應之外，只偵測 overlap，球可以真的掉進去。
            UsdPhysics.CollisionAPI.Apply(prim)
            PhysxSchema.PhysxTriggerAPI.Apply(prim)

    def subscribe_contact_events(
        self, callback: Callable[[ContactEvent], None]
    ) -> None:
        def _on_contact(
            headers: ContactEventHeaderVector, data: ContactDataVector
        ) -> None:
            for header in headers:
                if header.type != ContactEventType.CONTACT_FOUND:
                    continue
                max_impulse = 0.0
                for i in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data,
                ):
                    imp = data[i].impulse
                    max_impulse = max(
                        max_impulse, math.sqrt(imp.x**2 + imp.y**2 + imp.z**2)
                    )
                callback(
                    ContactEvent(
                        actor_path_a=str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                        actor_path_b=str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                        collider_path_a=str(
                            PhysicsSchemaTools.intToSdfPath(header.collider0)
                        ),
                        collider_path_b=str(
                            PhysicsSchemaTools.intToSdfPath(header.collider1)
                        ),
                        impulse=max_impulse,
                    )
                )

        def _on_trigger(trigger_data: TriggerEventData) -> None:
            if trigger_data.event_type != TriggerEventType.TRIGGER_ON_ENTER:
                return
            callback(
                ContactEvent(
                    actor_path_a=str(
                        PhysicsSchemaTools.intToSdfPath(trigger_data.trigger_body_prim_id)
                    ),
                    actor_path_b=str(
                        PhysicsSchemaTools.intToSdfPath(trigger_data.other_body_prim_id)
                    ),
                    collider_path_a=str(
                        PhysicsSchemaTools.intToSdfPath(
                            trigger_data.trigger_collider_prim_id
                        )
                    ),
                    collider_path_b=str(
                        PhysicsSchemaTools.intToSdfPath(
                            trigger_data.other_collider_prim_id
                        )
                    ),
                    impulse=0.0,  # Trigger 沒有物理衝量資訊，只能填佔位值
                )
            )

        sim_iface = get_physx_simulation_interface()
        self._contact_sub = sim_iface.subscribe_contact_report_events(_on_contact)
        self._trigger_sub_id = sim_iface.subscribe_physics_trigger_report_events(
            trigger_report_fn=_on_trigger, stage_id=0, prim_id=0
        )

    def unsubscribe_contact_events(self) -> None:
        self._contact_sub = None  # RAII：設 None 即自動取消訂閱
        if self._trigger_sub_id is not None:
            get_physx_simulation_interface().unsubscribe_physics_trigger_report_events(
                self._trigger_sub_id
            )
            self._trigger_sub_id = None
