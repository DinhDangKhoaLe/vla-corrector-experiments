import io
from argparse import Namespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from siglip_dynamics.local_parquet import LocalParquetDataset
from laco.extract import _frame_metadata


def test_local_parquet_indexes_row_groups_and_decodes_only_requested_frame(tmp_path):
    (tmp_path / "data").mkdir()
    image = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), 37, dtype=np.uint8)).save(image, format="PNG")
    for file in range(2):
        pq.write_table(pa.table({"episode_index": [file] * 3, "task_index": [file + 5] * 3,
                                 "action": [[float(i), 0, 0, 1] for i in range(3)],
                                 "observation.image": [{"bytes": image.getvalue(), "path": None}] * 3}),
                       tmp_path / "data" / f"file-{file}.parquet", row_group_size=2)
    ds = LocalParquetDataset(str(tmp_path))
    assert len(ds) == 6
    assert ds._table is None  # Constructor reads only Parquet footers.
    episodes, tasks, actions, key = _frame_metadata(ds, Namespace(action_key="action"))
    assert ds._table is None  # Metadata pass must not touch image row groups.
    np.testing.assert_array_equal(episodes, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(tasks, [5, 5, 5, 6, 6, 6])
    assert actions.shape == (6, 4) and key == "action"
    for index in [0, 1, 2, 3, 5, -1]:
        item = ds[index]
        assert item["episode_index"] == ((index % 6) // 3)
        np.testing.assert_array_equal(item["observation.image"], np.full((8, 8, 3), 37, dtype=np.uint8))
    assert ds._table.num_rows <= 2
    with pytest.raises(IndexError):
        ds[6]
