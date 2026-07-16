import importlib.util
import pathlib
import sys
from typing import Callable

import omni.kit.menu.utils
from omni.kit.menu.utils import MenuItemDescription

_REGISTERED_TOOLS: list[tuple[str, Callable]] = []


def tool_menu_item(menu_path: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        _REGISTERED_TOOLS.append((menu_path, func))
        return func

    return decorator


def _wrap(menu_path: str, func: Callable) -> Callable:
    def onclick_fn() -> None:
        try:
            func()
        except Exception as exc:
            print(f"[ToolMenu] {menu_path} failed: {exc}")
            raise

    return onclick_fn


def _scan_scripts_dir(scripts_dir: str) -> None:
    for py_file in pathlib.Path(scripts_dir).rglob("*.py"):
        if py_file.stem.startswith("_"):
            continue
        module_name = f"_tool_menu_scan.{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def discover_and_register(scripts_dir: str, top_menu_name: str) -> list[MenuItemDescription]:
    _REGISTERED_TOOLS.clear()
    _scan_scripts_dir(scripts_dir)

    seen_paths: set[str] = set()
    items: list[MenuItemDescription] = []
    for menu_path, func in _REGISTERED_TOOLS:
        if menu_path in seen_paths:
            raise ValueError(f"重複的 menu_path：{menu_path}")
        seen_paths.add(menu_path)
        items.append(MenuItemDescription(name=menu_path, onclick_fn=_wrap(menu_path, func)))

    omni.kit.menu.utils.add_menu_items(items, top_menu_name)
    return items


def unregister(menu_items: list[MenuItemDescription], top_menu_name: str) -> None:
    omni.kit.menu.utils.remove_menu_items(menu_items, top_menu_name)
