import omni.ui


class _TableComboItem(omni.ui.AbstractItem):
    """單一選桌選項，對應一個 table_id（沿用 prim path）"""

    def __init__(self, table_id: str) -> None:
        super().__init__()
        self.table_id = table_id
        self.model = omni.ui.SimpleStringModel(table_id)


class TableComboBoxModel(omni.ui.AbstractItemModel):
    """
    可動態增刪選項的 ComboBox model，對外一律用字串（table_id）溝通，不暴露
    index。ComboBox 底層沒有原生「無選擇」狀態，用 -1 代表「尚未選擇 / 選項
    已被移除」。

    從 `debug_menu.py` 搬出（#115）：HUD 面板（階段 4）也需要一份自己的選桌
    下拉，不該複製第二份相同邏輯。
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_index = omni.ui.SimpleIntModel(-1)
        self._current_index.add_value_changed_fn(
            lambda _: self._item_changed(None)
        )
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
