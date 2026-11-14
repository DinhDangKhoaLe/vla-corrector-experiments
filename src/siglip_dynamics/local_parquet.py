"""Bounded local Parquet access without building another Arrow image dataset."""
from __future__ import annotations

import io
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


class LocalParquetDataset:
    """Keep at most one row group resident, decoding only the requested image.

    Numeric metadata can be streamed separately without reading image columns.
    No Hugging Face dataset conversion, download, or cache write occurs.
    """

    def __init__(self, dataset_path: str):
        self._root = Path(dataset_path)
        data = self._root / "data" if (self._root / "data").is_dir() else self._root
        self._files = sorted(data.rglob("*.parquet"))
        if not self._files:
            raise FileNotFoundError(f"No Parquet data below {data}")
        self._groups = []
        self._ends = []
        count = 0
        for path in self._files:
            with pq.ParquetFile(path) as reader:
                for group in range(reader.num_row_groups):
                    size = reader.metadata.row_group(group).num_rows
                    if size:
                        self._groups.append((path, group))
                        count += size
                        self._ends.append(count)
        self._table = None
        self._cached_group = None

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = bisect_right(self._ends, index)
        if self._cached_group != group:
            self._table = None  # Release the previous group before allocating another.
            path, row_group = self._groups[group]
            with pq.ParquetFile(path) as reader:
                self._table = reader.read_row_group(row_group, use_threads=False)
            self._cached_group = group
        start = self._ends[group - 1] if group else 0
        item = self._table.slice(index - start, 1).to_pylist()[0]
        for key, value in item.items():
            if not isinstance(value, dict) or not ({"bytes", "path"} & value.keys()):
                continue
            source = io.BytesIO(value["bytes"]) if value.get("bytes") is not None else value.get("path")
            if isinstance(source, str) and not Path(source).is_absolute():
                source = self._root / source
            if source is not None:
                with Image.open(source) as image:
                    item[key] = np.array(image.convert("RGB"))
        return item

    def iter_columns(self, columns: list[str]):
        """Read only present scalar/action columns, never loading encoded images."""
        for path in self._files:
            with pq.ParquetFile(path) as reader:
                selected = [key for key in columns if key in reader.schema_arrow.names]
                for batch in reader.iter_batches(batch_size=4096, columns=selected, use_threads=False):
                    yield from batch.to_pylist()
