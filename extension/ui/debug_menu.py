import asyncio
from typing import Callable
from ui_style import UiStyle

import omni.kit.app
import omni.ui


class DebugMenu:
    """
    Debug 用UI, 放在Viewport 右側
    """

    def __init__(self, on_training_toggle: Callable[[bool], None], on_demo_toggle: Callable[[bool], None]) -> None:
        self._window = omni.ui.Window(
            "Billiard Debug",
            width=300,
            height=400,
            visible=True,
            dockPreference=omni.ui.DockPreference.RIGHT_TOP,
        )
        self._on_training_toggle = on_training_toggle
        self._on_demo_toggle = on_demo_toggle
        self._build_ui()
        asyncio.ensure_future(self._dock_to_viewport())

    def _build_ui(self) -> None:
        toggle_style = UiStyle.get_toggle_style()
        with self._window.frame:
            with omni.ui.VStack(spacing=5):
                with omni.ui.HStack(height=24):
                    omni.ui.Label("Training")
                    training_model = omni.ui.SimpleBoolModel(False)
                    omni.ui.ToolButton(
                        text="", model=training_model,width=50,height=24, style=toggle_style
                    )
                    training_model.add_value_changed_fn(
                        lambda m: self._on_training_toggle(m.get_value_as_bool())
                    )
                with omni.ui.HStack(height=24):
                    omni.ui.Label("Break shot demo")
                    demo_model = omni.ui.SimpleBoolModel(False)
                    omni.ui.ToolButton(
                        text="",model=demo_model, width=50, height=24, style=toggle_style
                    )
                    demo_model.add_value_changed_fn(
                        lambda m: self._on_demo_toggle(m.get_value_as_bool())
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
        if self._window:
            self._window.destroy()
            self._window = None
