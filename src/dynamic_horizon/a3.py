"""A3 consensus draft, conditional re-decoding and prefix verification.

Port of the execution rules in INCEPTIONwang/A3 at 96a85c6. Verification
is sequential here to fit a single GPU; both verification orders are retained.
"""
import numpy as np
import torch

from .types import Decision


def select_draft(actions, epsilon=1e-8):
    # Official MetaWorld comparison excludes the gripper (three XYZ channels).
    trajectories = np.cumsum(np.asarray(actions)[..., :3], axis=1)
    mean, variance = trajectories.mean(0), trajectories.var(0)
    confidence = np.exp(-0.5 * (((trajectories - mean) ** 2) / (variance + epsilon)).sum(-1))
    high = confidence >= np.median(confidence, axis=0)
    count = high.sum(-1)
    candidates = np.flatnonzero(count == count.max())
    best = int(candidates[np.argmax((confidence * high).sum(-1)[candidates])])
    return best, confidence[best]


def verify_order(sampler, draft, order, thresholds):
    accepted = {0}
    fixed_mask = torch.zeros(len(draft.actions), dtype=torch.bool, device=draft.normalized.device)
    fixed_mask[0] = True
    failure = None
    for position in order:
        position = int(position)
        if position == 0:
            continue
        checked = sampler.sample(fixed_actions=draft.normalized, fixed_mask=fixed_mask)
        delta = (checked.actions[position, :3] - draft.actions[position, :3]).abs()
        if bool((delta < delta.new_tensor(thresholds)).all()):
            accepted.add(position)
            fixed_mask[position] = True
        else:
            failure = (position, checked.actions[position])
            break
    horizon = 1
    while horizon in accepted:
        horizon += 1
    return horizon, failure


class A3:
    def __init__(self, settings):
        self.settings = settings

    def decide(self, sampler, task, **kwargs):
        candidates = [sampler.sample() for _ in range(self.settings.get("samples", 8))]
        best, confidence = select_draft(
            np.stack([p.actions.cpu().numpy() for p in candidates]),
            self.settings.get("confidence_epsilon", 1e-8),
        )
        draft = candidates[best]
        positions = np.arange(len(draft.actions))
        order = np.lexsort((positions, -confidence))
        thresholds = self.settings.get("delta_thresholds", [2.0, 2.0, 3.4])
        hc, fc = verify_order(sampler, draft, order, thresholds)
        hs, fs = verify_order(sampler, draft, positions, thresholds)
        horizon = min(hc, hs)
        result = draft.actions[:horizon]
        # Released A3 appends the first rejection's newly decoded action when
        # it is exactly at the accepted-prefix boundary (sequence wins ties).
        failures = [fc] if hc < hs else [fs] if hs < hc else [fs, fc]
        replaced = False
        for failure in failures:
            if failure is not None and failure[0] == horizon:
                result = torch.cat([result, failure[1][None]], dim=0)
                replaced = True
                break
        return Decision(result, {"draft": best, "consensus_prefix": hc,
                                 "sequential_prefix": hs, "replacement_appended": replaced})
