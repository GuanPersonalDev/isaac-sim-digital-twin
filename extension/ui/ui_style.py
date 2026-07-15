
class UiStyle:
    _BACKGROUND_COLOR = "background_color"
    _BORDER_RADIUS = "border_radius"

    @staticmethod
    def get_toggle_style():
        return {
            "ToolButton":{
                UiStyle._BACKGROUND_COLOR: 0xFF555555,
                UiStyle._BORDER_RADIUS: 12,
            },
            "ToolButton:checked":{
                UiStyle._BACKGROUND_COLOR: 0xFF2ECC71,
                UiStyle._BORDER_RADIUS: 12,
            }
        }