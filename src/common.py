"""Tiện ích dùng chung cho toàn bộ dự án.

Mục tiêu chính:
- Chuẩn hóa đường dẫn tương đối theo thư mục gốc dự án.
- Thiết lập seed nhất quán cho Python, NumPy và PyTorch.
- Cung cấp ánh xạ nhãn và các hàm đánh giá chung.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "depression_severity_preprocessed.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

LABEL_MAP: Dict[str, int] = {
    "minimum": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}
ID_TO_LABEL: Dict[int, str] = {value: key for key, value in LABEL_MAP.items()}
LABEL_ORDER = ["minimum", "mild", "moderate", "severe"]


def ensure_output_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Thiết lập seed cho Python, NumPy và PyTorch.

    ``deterministic=True`` ưu tiên khả năng tái lập. Một vài phép toán có thể
    chậm hơn, đặc biệt khi chạy GPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except ImportError:
        # Các script EDA/classical vẫn chạy được khi chưa cài PyTorch.
        pass


def dataloader_generator(seed: int):
    """Tạo generator có seed cho DataLoader của PyTorch."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_int_list(values: Sequence[str] | Iterable[int]) -> list[int]:
    """Chuyển danh sách seed từ CLI thành list[int]."""
    return [int(value) for value in values]
