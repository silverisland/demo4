# -*- coding: utf-8 -*-
"""paths.py —— 让本目录脚本能找到仓库里的 forecast_backbones / pipelines / interfaces。

搜索顺序（先加入的优先级低，后加入的插到最前面）：
  1. 本目录的上两级（默认：本目录放在仓库 experiments/nwp_covariate 下时就是仓库根）；
  2. 环境变量 NWP_COV_SYS_PATH（多个路径用 ':' 分隔）；
  3. config.yaml 的 run.sys_path 列表（相对路径按仓库根解析）。
部署机上 forecast_backbones 可能不在这个仓库里（线上 multi_infer 把 /home/nfdw/nuoya/algo/nuoya_algo 等加进 sys.path），
这时在 run.sys_path 写上"包含 forecast_backbones 目录的那一层"即可。
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1]
NEEDED = ("forecast_backbones", "pipelines", "interfaces")


def _insert(p) -> None:
    p = str(Path(p).expanduser().resolve())
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


def setup(extra: Optional[Iterable[str]] = None) -> None:
    for p in (str(HERE), str(DEFAULT_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for p in filter(None, os.environ.get("NWP_COV_SYS_PATH", "").split(os.pathsep)):
        _insert(p)
    for p in extra or ():
        q = Path(str(p)).expanduser()
        _insert(q if q.is_absolute() else DEFAULT_ROOT / q)


def require(module: str):
    """import 一个仓库模块；找不到时说明去哪找、怎么配。"""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        top = (e.name or "").split(".")[0]
        if top in NEEDED:
            found = [p for p in sys.path if p and (Path(p) / top).is_dir()]
            raise ModuleNotFoundError(
                f"找不到 {e.name!r}（导入 {module} 时）。需要一个目录，它下面直接有 "
                f"{'/'.join(NEEDED)} 这几个子目录。\n"
                f"  当前搜索路径里含 {top}/ 的目录：{found or '无'}\n"
                f"  解决：在 config.yaml 写 run.sys_path: [/那个目录]，或 export NWP_COV_SYS_PATH=/那个目录；\n"
                f"  本目录默认认为仓库根是 {DEFAULT_ROOT}，请确认 {DEFAULT_ROOT / top} 存在。"
            ) from e
        raise ModuleNotFoundError(f"缺 Python 包 {e.name!r}（导入 {module} 时），请 pip install 或换到装好它的环境") from e


setup()
