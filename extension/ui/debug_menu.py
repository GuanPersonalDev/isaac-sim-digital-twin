import asyncio
from typing import Callable
from .ui_style import UiStyle

import omni.kit.app
import omni.ui


class _TableComboItem(omni.ui.AbstractItem):
    """單一選桌選項，對應一個 table_id（沿用 prim path）"""

    def __init__(self, table_id: str) -> None:
        super().__init__()
        self.table_id = table_id
        self.model = omni.ui.SimpleStringModel(table_id)


class _TableComboBoxModel(omni.ui.AbstractItemModel):
    """
    可動態增刪選項的 ComboBox model，對外一律用字串（table_id）溝通，不暴露
    index。ComboBox 底層沒有原生「無選擇」狀態，用 -1 代表「尚未選擇 / 選項
    已被移除」。
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_index = omni.ui.SimpleIntModel(-1)
        self._items: list[_TableComboItem] = []

    def get_item_children(self, item=None):
        return self._items

    def get_item_value_model(self, item=None, column_id=0):
        if item is None:
            return self._current_index
        return item.model

    def set_items(self, table_ids: list[str]) -> None:
        """
        整批更新選項清單。若原本選中的 table_id 已不在新清單中（該桌被
        Toggle 關閉刪除），自動清空選擇；面板回到空白，不自動切換到其他桌。
        """
        previous_selected = self.get_selected_table_id()

        self._items = [_TableComboItem(table_id) for table_id in table_ids]

        if previous_selected is not None and previous_selected in table_ids:
            self._current_index.set_value(table_ids.index(previous_selected))
        else:
            self._current_index.set_value(-1)

        self._item_changed(None)

    def get_selected_table_id(self) -> str | None:
        idx = self._current_index.as_int
        if 0 <= idx < len(self._items):
            return self._items[idx].table_id
        return None


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
        get_ball_velocities_text: Callable[[str], str],
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
        self._get_ball_velocities_text = get_ball_velocities_text
        self._show_ball_velocities = False
        self._table_combo_model = _TableComboBoxModel()
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

                self._status_label = omni.ui.Label("", word_wrap=True)

                with omni.ui.HStack(height=24):
                    omni.ui.Label("Show Ball Velocities")
                    velocity_toggle_model = omni.ui.SimpleBoolModel(False)
                    omni.ui.ToolButton(
                        text="",
                        model=velocity_toggle_model,
                        width=50,
                        height=24,
                        style=toggle_style,
                    )
                    velocity_toggle_model.add_value_changed_fn(
                        lambda m: setattr(self, "_show_ball_velocities", m.get_value_as_bool())
                    )

                self._velocity_label = omni.ui.Label("", word_wrap=True)

    def set_available_tables(self, table_ids: list[str]) -> None:
        self._table_combo_model.set_items(table_ids)

    def _on_update(self, event) -> None:
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            self._status_label.text = ""
            self._velocity_label.text = ""
            return

        self._status_label.text = self._get_table_debug_info(table_id)
        self._velocity_label.text = (
            self._get_ball_velocities_text(table_id) if self._show_ball_velocities else ""
        )

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
