"""Module ASR Cache: Bộ nhớ đệm lưu trữ kết quả nhận diện âm thanh.
Kế thừa cơ chế tối ưu từ Zalo_AI: Lưu trữ kết quả gỡ âm theo đường dẫn hoặc mã băm SHA256 của file,
tránh lãng phí tài nguyên GPU và thời gian suy luận khi chạy lại các vòng đánh giá.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ASRCache:
    """Quản lý bộ nhớ đệm ASR lưu trữ trên đĩa JSON."""

    def __init__(self, cache_file_path: Optional[Union[str, Path]] = None):
        self.cache_file = Path(cache_file_path).resolve() if cache_file_path else None
        self._cache: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Nạp dữ liệu cache từ file JSON nếu tồn tại."""
        if self.cache_file and self.cache_file.is_file():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception as e:
                print(f"[ASRCache] Cảnh báo: Không thể nạp cache từ {self.cache_file} ({e})")
                self._cache = {}

    def save(self) -> None:
        """Ghi dữ liệu cache ra đĩa an toàn."""
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ASRCache] Lỗi khi lưu cache ({e})")

    @staticmethod
    def compute_file_hash(file_path: Union[str, Path]) -> str:
        """Tính mã băm MD5/SHA256 của file âm thanh."""
        path = Path(file_path)
        if not path.is_file():
            return str(file_path)
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Lấy bản ghi từ cache."""
        return self._cache.get(key)

    def set(self, key: str, value: Any, auto_save: bool = True) -> None:
        """Lưu bản ghi vào cache và tự động ghi đĩa."""
        self._cache[key] = value
        if auto_save:
            self.save()

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)
