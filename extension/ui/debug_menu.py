import asyncio

import omni.kit.app
import omni.ui


class DebugMenu:
    """
    Debug 用UI, 放在Viewport 右側
    """

    def __init__(self) -> None:
        self._window = omni.ui.Window(
            "Billiard Debug",
            width=300,
            height=400,
            visible=True,
            dockPreference=omni.ui.DockPreference.RIGHT_TOP,
        )
        self._build_ui()
        asyncio.ensure_future(self._dock_to_viewport())

    def _build_ui(self) -> None:
        with self._window.frame:
            with omni.ui.VStack(spacing=5):
                # add buttons here
                pass

    async def _dock_to_viewport(self) -> None:
        for _ in range(5):
            if omni.ui.Workspace.get_window("Viewport"):
                break
            await omni.kit.app.get_app().next_update_async()
        self._window.dock_in_window("Viewport", omni.ui.DockPosition.RIGHT, ratio=0.25)

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
