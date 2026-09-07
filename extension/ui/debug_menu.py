import asyncio
from typing import Callable
from .table_combo_box_model import TableComboBoxModel
from .ui_style import UiStyle

import omni.kit.app
import omni.ui


class DebugMenu:
    """
    Debug 用UI, 放在Viewport 右側
    """

    def __init__(
        self,
        on_training_toggle: Callable[[bool], None],
        on_demo_toggle: Callable[[bool], None],
        get_table_ids: Callable[[], list[str]],
        get_table_debug_info: Callable[[str], str],
        on_controller_mode_changed: Callable[[str, bool], None],
        get_joint_state_text: Callable[[str], str],
    ) -> None:
        self._window = omni.ui.Window(
            "Billiard Debug",
            width=300,
            height=400,
            visible=True,
            dockPreference=omni.ui.DockPreference.RIGHT_TOP,
        )
        self._on_training_toggle = on_training_toggle
        self._on_demo_toggle = on_demo_toggle
        self._get_table_ids = get_table_ids
        self._get_table_debug_info = get_table_debug_info
        self._on_controller_mode_changed = on_controller_mode_changed
        self._get_joint_state_text = get_joint_state_text
        self._table_combo_model = TableComboBoxModel()
        self._build_ui()
        asyncio.ensure_future(self._dock_to_viewport())
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="billiard_debug_menu_refresh")
        )

    def _build_ui(self) -> None:
        toggle_style = UiStyle.get_toggle_style()
        with self._window.frame:
            with omni.ui.VStack(spacing=5):
                with omni.ui.HStack(height=24):
                    omni.ui.Label("Training")
                    training_model = omni.ui.SimpleBoolModel(True)
                    omni.ui.ToolButton(
                        text="",
                        model=training_model,
                        width=50,
                        height=24,
                        style=toggle_style,
                    )
                    training_model.add_value_changed_fn(
                        lambda m: self._on_training_toggle(m.get_value_as_bool())
                    )
                with omni.ui.HStack(height=24):
                    omni.ui.Label("Break shot demo")
                    demo_model = omni.ui.SimpleBoolModel(True)
                    omni.ui.ToolButton(
                        text="",
                        model=demo_model,
                        width=50,
                        height=24,
                        style=toggle_style,
                    )
                    demo_model.add_value_changed_fn(
                        lambda m: self._on_demo_toggle(m.get_value_as_bool())
                    )

                with omni.ui.HStack(height=24):
                    omni.ui.Label("Table")
                    omni.ui.ComboBox(self._table_combo_model, width=180, height=24)

                with omni.ui.HStack(height=24):
                    # 對「Table」選中的那張桌子換操作策略；true=AI
                    # （ModelController，訓練好的 policy 決定擊球）、
                    # false=Manual（ManualController，由 #115 的 HUD 面板
                    # 手動決定擊球參數，按「擊球」鈕才出一桿，不自動循環）。
                    # 沒選桌子時按下無效果，是刻意的 no-op。
                    omni.ui.Label("Controller: AI / Manual")
                    controller_mode_model = omni.ui.SimpleBoolModel(True)
                    omni.ui.ToolButton(
                        text="",
                        model=controller_mode_model,
                        width=50,
                        height=24,
                        style=toggle_style,
                    )
                    controller_mode_model.add_value_changed_fn(
                        self._on_controller_mode_toggle
                    )

                self._status_label = omni.ui.Label("", word_wrap=True)
                self._joint_state_label = omni.ui.Label("", word_wrap=True)

    def _on_controller_mode_toggle(self, model: omni.ui.SimpleBoolModel) -> None:
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            return
        self._on_controller_mode_changed(table_id, model.get_value_as_bool())

    def set_available_tables(self, table_ids: list[str]) -> None:
        self._table_combo_model.set_items(table_ids)

    def _on_update(self, event) -> None:
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            self._status_label.text = ""
            self._joint_state_label.text = ""
            return

        self._status_label.text = self._get_table_debug_info(table_id)
        self._joint_state_label.text = self._get_joint_state_text(table_id)

    async def _dock_to_viewport(self) -> None:
        target_window = None
        for _ in range(5):
            target_window = omni.ui.Workspace.get_window("Viewport")
            if omni.ui.Workspace.get_window("Viewport"):
                break
            await omni.kit.app.get_app().next_update_async()
        if target_window:
            self._window.dock_in(target_window, omni.ui.DockPosition.RIGHT, ratio=0.25)

    def show(self) -> None:
        if self._window:
            self._window.visible = True

    def hide(self) -> None:
        if self._window:
            self._window.visible = False

    def destroy(self) -> None:
        self._update_sub = None
        if self._window:
            self._window.destroy()
            self._window = None
