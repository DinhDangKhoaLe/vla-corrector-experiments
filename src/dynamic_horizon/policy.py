"""Frozen local π0.5 adapter shared by the five execution methods.

Keeps one observation-prefix KV cache per decision, while each candidate or
verification uses fresh noise. No changes to the existing corrector or policy.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .types import Prediction


def load_checkpoint_config(checkpoint):
    """Accept typed exports and legacy compatibility views without a type tag."""
    import draccus
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    raw = json.loads((Path(checkpoint) / "config.json").read_text())
    if raw.pop("type", "pi05") != "pi05":
        raise ValueError("Expected a π0.5 checkpoint configuration")
    # Decode the concrete class so legacy views need no on-disk repair. Unknown
    # fields still fail; the strict tensor-key/shape check follows before inference.
    return draccus.decode(PI05Config, raw)


class SentencePieceTokenizer:
    """OpenPI PaliGemma token IDs, exposed through LeRobot's tokenizer interface."""

    def __init__(self, path):
        import sentencepiece

        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(Path(path).expanduser()))
        if self.processor.vocab_size() != 257152 or self.processor.bos_id() != 2:
            raise ValueError("Expected the OpenPI PaliGemma tokenizer (257152 tokens, BOS=2)")

    def __call__(self, text, max_length, truncation, padding, padding_side, return_tensors):
        if padding != "max_length" or padding_side != "right" or return_tensors != "pt":
            raise ValueError("Only the saved π0.5 right-padded tokenizer format is supported")
        texts = [text] if isinstance(text, str) else text
        ids, masks = [], []
        for prompt in texts:
            tokens = self.processor.encode(prompt, add_bos=True)
            if len(tokens) > max_length and not truncation:
                raise ValueError("Prompt exceeds tokenizer_max_length")
            tokens = tokens[:max_length]
            masks.append([True] * len(tokens) + [False] * (max_length - len(tokens)))
            ids.append(tokens + [0] * (max_length - len(tokens)))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class Pi05Sampler:
    def __init__(self, settings):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors

        checkpoint = Path(settings["checkpoint"]).expanduser().resolve()
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        if settings["device"].startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the active PyTorch/CUDA environment")
        print(f"Loading frozen π0.5 from {checkpoint} ({settings['dtype']})", flush=True)
        config = load_checkpoint_config(checkpoint)
        if config.chunk_size != 50 or config.output_features["action"].shape[0] != 4:
            raise ValueError("Expected the downloaded MetaWorld π0.5 checkpoint: H=50, action_dim=4")
        config.device = "cpu"
        config.dtype = settings["dtype"]
        config.n_action_steps = config.chunk_size = 50
        config.num_inference_steps = settings["denoising_steps"]
        config.gradient_checkpointing = config.compile_model = False
        config.rtc_config = None
        self.policy = PI05Policy(config)
        # The upstream loader can return random weights after a load error.
        # Use a strict streaming local load instead and fail on any mismatch.
        destination = self.policy.state_dict()
        with safe_open(str(weights), framework="pt", device="cpu") as source:
            expected, actual = set(destination), set(source.keys())
            if expected != actual:
                raise ValueError(f"Checkpoint keys differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
            with torch.no_grad():
                for key, tensor in destination.items():
                    value = source.get_tensor(key)
                    if tensor.shape != value.shape:
                        raise ValueError(f"Checkpoint shape mismatch: {key}")
                    tensor.copy_(value)
        self.device = torch.device(settings["device"])
        self.policy.to(self.device).eval().requires_grad_(False)
        # The local torch/CUDA combination's SigLIP SDPA path produced different
        # features on identical inputs, even in eval mode. Eager vision attention
        # is repeatable here and keeps cached/native inference exactly aligned.
        self.policy.model.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation = "eager"
        config.device = str(self.device)
        tokenizer = SentencePieceTokenizer(settings["tokenizer_model"])
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config, pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "tokenizer_processor": {"tokenizer": tokenizer},
            },
        )
        self.model = self.policy.model
        self.steps = settings["denoising_steps"]
        self.cache = None
        self.samples = self.denoising_calls = self.prefix_calls = 0

    @torch.no_grad()
    def prepare(self, observation, instruction):
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

        batch = self.preprocessor({
            "observation.image": torch.from_numpy(np.asarray(observation["pixels"]).copy()).permute(2, 0, 1).float() / 255,
            "observation.state": torch.as_tensor(np.asarray(observation["agent_pos"]).copy(), dtype=torch.float32),
            "task": instruction,
        })
        self.batch = batch  # Also used by the native-sampler parity integration check.
        images, image_masks = self.policy._preprocess_images(batch)
        embs, masks, attention = self.model.embed_prefix(
            images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
        )
        combined = self.model.paligemma_with_expert
        combined.paligemma.model.language_model.config._attn_implementation = "eager"
        outputs, self.cache = combined.forward(
            attention_mask=self.model._prepare_attention_masks_4d(make_att_2d_masks(masks, attention)),
            position_ids=masks.cumsum(1) - 1, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True,
        )
        self.prefix_masks = masks
        self.context = outputs[0][0, masks[0]].detach()
        self.prefix_calls += 1

    @torch.no_grad()
    def sample(self, *, attention_step=None, features=False, fixed_actions=None, fixed_mask=None, noise=None):
        if self.cache is None:
            raise RuntimeError("prepare(observation, instruction) must precede sampling")
        if attention_step is not None and not 0 <= attention_step < self.steps:
            raise ValueError("attention_step is a zero-based denoising index")
        if (fixed_actions is None) != (fixed_mask is None):
            raise ValueError("Conditional decoding needs both fixed actions and a mask")
        if noise is None:
            noise = self.model.sample_noise((1, 50, self.policy.config.max_action_dim), self.device)
        if noise.shape != (1, 50, self.policy.config.max_action_dim):
            raise ValueError("Noise must match [1, 50, padded_action_dim]")
        x = noise.clone()
        attention_blocks, expert_features = [], []

        def record_attention(module, args, output):
            probabilities = output[1]
            if probabilities is None:
                raise RuntimeError("AutoHorizon requires eager attention probabilities")
            attention_blocks.append(probabilities[0, :, -50:, -50:].float().mean(0))

        def record_features(module, args):
            expert_features.append(args[0][0].detach())

        dt = -1.0 / self.steps
        for step in range(self.steps):
            # Match native Euler arithmetic exactly: division versus multiply
            # can cross bfloat16 rounding boundaries in the following step.
            time = 1.0 + step * dt
            if fixed_mask is not None:
                fixed = fixed_actions.to(self.device)[None]
                mask = fixed_mask.to(self.device)[None, :, None]
                x = torch.where(mask, time * noise + (1 - time) * fixed, x)
            handles = []
            try:
                if step == attention_step:
                    for layer in self.model.paligemma_with_expert.gemma_expert.model.layers:
                        handles.append(layer.self_attn.register_forward_hook(record_attention))
                if features and step == self.steps - 1:
                    handles.append(self.model.action_out_proj.register_forward_pre_hook(record_features))
                velocity = self.model.denoise_step(
                    self.prefix_masks, self.cache, x,
                    torch.tensor([time], dtype=torch.float32, device=self.device),
                )
            finally:
                for handle in handles:
                    handle.remove()
            x = x + dt * velocity
            if fixed_mask is not None:
                next_time = 1.0 + (step + 1) * dt
                x = torch.where(mask, next_time * noise + (1 - next_time) * fixed, x)
            self.denoising_calls += 1
        self.samples += 1
        actions = self.postprocessor(x[..., :4].clone())[0]
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Non-finite π0.5 action prediction")
        if attention_step is not None and len(attention_blocks) != len(self.model.paligemma_with_expert.gemma_expert.model.layers):
            raise RuntimeError("Missing AutoHorizon attention layers")
        return Prediction(
            normalized=x[0], actions=actions,
            attention=torch.stack(attention_blocks).mean(0) if attention_blocks else None,
            context=self.context if features else None,
            velocity_features=expert_features[-1] if features else None,
        )
