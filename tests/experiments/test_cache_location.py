"""Cache residency must preserve temporal alignment, batches and trained weights."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DistributedSampler, Subset

from siglip_dynamics.dataset import SiglipDynamicsDataset, cache_dataloader
from siglip_dynamics.experiments.config import from_dict
from siglip_dynamics.experiments.data import PrefixDataset
from siglip_dynamics.experiments.run import legacy_train_command, resolve
from siglip_dynamics.experiments.train import train_experiment
from test_corrector_experiments import tiny_config
from test_spatial_pool import make_pooled_cache


class CacheLocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        make_pooled_cache(self.cache)

    def dataset(self, location, **kwargs):
        return SiglipDynamicsDataset(str(self.cache), k_step=4, h_window=3,
                                     cache_location=location, **kwargs)

    def assert_batches_equal(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            self.assertEqual(expected[key].dtype, actual[key].dtype, key)
            torch.testing.assert_close(expected[key], actual[key].cpu(), rtol=0, atol=0, msg=key)

    def check_loader_equivalence(self, location):
        disk, resident = self.dataset("disk"), self.dataset(location)
        self.assertEqual(disk.pairs, resident.pairs)
        # Exercise nested Subsets, a shuffled loader, distributed sampling, and a partial last batch.
        ids = list(range(len(disk) - 1, 0, -3))
        datasets = [Subset(Subset(ds, ids), list(range(1, len(ids)))) for ds in (disk, resident)]
        for distributed in (False, True):
            batches = []
            for ds in datasets:
                torch.manual_seed(123)
                sampler = DistributedSampler(ds, num_replicas=2, rank=1, seed=7) if distributed else None
                loader = cache_dataloader(ds, batch_size=7, shuffle=not distributed, sampler=sampler)
                rows = list(loader)
                self.assertEqual(len(loader), len(rows))
                batches.append(rows)
            self.assertEqual(len(batches[0]), len(batches[1]))
            for expected, actual in zip(*batches):
                self.assert_batches_equal(expected, actual)

    def test_ram_owns_packed_arrays_and_matches_disk_loaders(self):
        ram = self.dataset("ram")
        self.assertIs(type(ram.z_q), np.ndarray)
        self.assertEqual(ram.z_q.dtype, np.int8)
        self.assertEqual(ram.z_scale.dtype, np.float16)
        self.check_loader_equivalence("ram")
        before = ram.z_q.copy()
        mapped = np.load(self.cache / "z_q.npy", mmap_mode="r+")
        mapped[:] = 0
        mapped.flush()
        np.testing.assert_array_equal(ram.z_q, before)

    def check_prefix_equivalence(self, location):
        bases = [SiglipDynamicsDataset(str(self.cache), k_step=4, cache_location=where)
                 for where in ("disk", location)]
        datasets = [PrefixDataset(self.cache, [1, 2, 4], [0, 2, 4], base=b) for b in bases]
        validation = PrefixDataset(self.cache, [1, 2, 4], [1, 3, 5], base=bases[1])
        self.assertIs(validation.base, datasets[1].base)
        rows = [list(cache_dataloader(ds, batch_size=11)) for ds in datasets]
        for expected, actual in zip(*rows):
            self.assert_batches_equal(expected, actual)
            for h, actions in zip(actual["horizon"], actual["actions"]):
                self.assertTrue(torch.all(actions[int(h):] == 0))

    def test_prefix_ram_masks_and_shared_validation_cache(self):
        self.check_prefix_equivalence("ram")

    def test_cli_configuration_and_invalid_locations(self):
        cfg = resolve("configs/corrector/metaworld_original_pool4.yaml", ["training.cache_location=gpu"])
        self.assertIn("--cache-location=gpu", legacy_train_command(cfg))
        self.assertEqual(resolve("configs/corrector/metaworld_prefix_pool4.yaml", []).training.cache_location, "disk")
        with self.assertRaisesRegex(ValueError, "cache_location"):
            from_dict({"training": {"cache_location": "invalid"}})
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.dataset("gpu", cache_device="cpu")

    def check_training_equivalence(self, location, device):
        for variant in ("original", "prefix", "uncertainty"):
            weights = []
            metrics = []
            splits = []
            for where in ("disk", location):
                out = self.root / f"{variant}_{device}_{where}"
                if variant == "original":
                    from siglip_dynamics.config import TrainConfig, SiglipMLPConfig, ModelScale
                    from siglip_dynamics.train import train
                    cfg = TrainConfig(dataset_path=str(self.cache), checkpoint_dir=str(out),
                                      device=device, cache_location=where, epochs=1, batch_size=17,
                                      k_step=4, h_window=1, num_workers=0,
                                      model=SiglipMLPConfig(token_dim=4, action_dim=2, action_embed_dim=4,
                                                           scale=ModelScale.CUSTOM, custom_widths=(8, 8)))
                    with patch("siglip_dynamics.train.wandb"):
                        train(cfg)
                    payload = torch.load(out / "best_model.pt", map_location="cpu", weights_only=False)
                    weights.append(payload["state_dict"])
                else:
                    cfg = tiny_config()
                    cfg.model.image_token_grid_size = 4
                    cfg.model.horizons = [4]
                    cfg.features.uncertainty_monitor = variant == "uncertainty"
                    cfg.training.device = device
                    cfg.training.cache_location = where
                    cfg.training.cache = str(self.cache)
                    cfg.training.output = str(out)
                    cfg.training.batch_size = 17
                    cfg.training.microbatch_size = 5
                    train_experiment(cfg)
                    payload = torch.load(out / "last_model.pt", map_location="cpu", weights_only=False)
                    weights.append(payload["state_dict"])
                    metrics.append(json.loads((out / "metrics.jsonl").read_text()))
                    splits.append(json.loads((out / "split_manifest.json").read_text()))
            self.assertEqual(weights[0].keys(), weights[1].keys())
            for key in weights[0]:
                torch.testing.assert_close(weights[0][key], weights[1][key], rtol=0, atol=0, msg=f"{variant}: {key}")
            self.assertEqual(metrics[:1], metrics[1:])
            self.assertEqual(splits[:1], splits[1:])

    def test_ram_training_matches_disk_all_three_variants(self):
        self.check_training_equivalence("ram", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_batches_packed_storage_and_errors(self):
        self.check_loader_equivalence("gpu")
        self.check_prefix_equivalence("gpu")
        ds = self.dataset("gpu")
        self.assertEqual(ds._gpu_arrays["z_q"].dtype, torch.int8)
        self.assertEqual(ds._gpu_arrays["z_scale"].dtype, torch.float16)
        self.assertTrue(all(t.is_cuda for t in ds._gpu_arrays.values()))
        loader = cache_dataloader(ds, batch_size=7, pin_memory=True)
        self.assertFalse(loader.pin_memory)
        self.assertTrue(next(iter(loader))["z_t"].is_cuda)
        with self.assertRaisesRegex(ValueError, "num_workers=0"):
            cache_dataloader(ds, batch_size=7, num_workers=1)
        with patch("torch.cuda.mem_get_info", return_value=(0, 0)):
            with self.assertRaisesRegex(MemoryError, "additional VRAM"):
                self.dataset("gpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_training_matches_disk_all_three_variants(self):
        self.check_training_equivalence("gpu", "cuda")
