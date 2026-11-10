"""COAP-controlled initial noise standard deviation for PI0.5 flow sampling."""

import math


def temperature_from_score(score, settings):
    """Linear, bounded controller: lower compatibility gives higher noise scale."""
    if score is None:
        return float(settings.warmup_temperature)
    if not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError("Generation controller requires a finite cosine in [-1,1]")
    fraction = min(1.0, max(0.0, (score - settings.score_low) / (settings.score_high - settings.score_low)))
    return float(settings.max_temperature - fraction * (settings.max_temperature - settings.min_temperature))


def scale_initial_noise(noise, temperature):
    """Scale noise, never actions or denoising updates; T=1 preserves the original tensor."""
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Generation temperature must be a positive finite noise standard-deviation scale")
    return noise if temperature == 1 else noise * temperature
