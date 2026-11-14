from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class TinyPolicy(PI05Policy):
    constructed_on_meta = False

    def __init__(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = torch.nn.Linear(3, 2)
        self.model.register_buffer("rotary", torch.ones(2), persistent=False)
        type(self).constructed_on_meta = self.model.weight.is_meta

    def _fix_pytorch_state_dict_keys(self, state, config):
        return state


def test_pi05_assigns_weights_without_random_model_allocation(tmp_path):
    save_file({"weight": torch.full((2, 3), 7., dtype=torch.float16), "bias": torch.ones(2)},
              tmp_path / "model.safetensors")
    policy = TinyPolicy.from_pretrained(tmp_path, config=SimpleNamespace(device="cpu"), local_files_only=True)
    assert TinyPolicy.constructed_on_meta
    assert not any(value.is_meta for value in policy.parameters())
    assert policy.model.weight.dtype == torch.float32
    assert not policy.model.rotary.is_meta
    assert not policy.training
    torch.testing.assert_close(policy.model.weight, torch.full((2, 3), 7.))


def test_pi05_missing_weights_fail_instead_of_returning_uninitialized_model(tmp_path):
    with pytest.raises(RuntimeError, match="Failed to load pretrained PI05"):
        TinyPolicy.from_pretrained(tmp_path, config=SimpleNamespace(device="cpu"), local_files_only=True)


def test_modified_pi05_constructor_defers_device_move_during_checkpoint_load(tmp_path, monkeypatch):
    import lerobot.policies.pi05_modified.modeling_pi05_modified as modified
    from lerobot.policies.pi05_modified.configuration_pi05_modified import PI05ModifiedConfig
    monkeypatch.setattr(modified, "PI05PytorchModified", lambda *a, **kw: torch.nn.Linear(3, 2))
    save_file({"weight": torch.full((2, 3), 5.), "bias": torch.ones(2)}, tmp_path / "model.safetensors")
    cfg = PI05ModifiedConfig(device="cpu")
    policy = modified.PI05ModifiedPolicy.from_pretrained(tmp_path, config=cfg, local_files_only=True)
    assert not any(value.is_meta for value in policy.parameters())
    torch.testing.assert_close(policy.model.weight, torch.full((2, 3), 5.))
