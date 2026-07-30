"""`--pid-degrade-sigma`: noise the LQ latent to match the sigma~U[0, 0.8] degradation
PiD's LQ gate was distilled against.

The failure this guards is silent: the gates are *told* sigma while the latent they gate on
stays pristine, which is a distribution mismatch rather than a weaker degradation. So the
tests assert the latent actually moves, by the right amount, on an RNG stream independent of
the sampler's.
"""

import mlx.core as mx
import pytest

from mflux.models.common.pid_decoder.pid_decoder import (
    PID_MAX_DEGRADE_SIGMA,
    _degrade_latent,
)


def _latent(seed: int = 0) -> mx.array:
    return mx.random.normal((1, 16, 8, 8), key=mx.random.key(seed))


def test_zero_sigma_is_the_identity():
    latent = _latent()
    assert mx.array_equal(_degrade_latent(latent=latent, degrade_sigma=0.0, seed=7), latent)


def test_nonzero_sigma_moves_the_latent():
    latent = _latent()
    out = _degrade_latent(latent=latent, degrade_sigma=0.2, seed=7)
    assert not mx.array_equal(out, latent)
    assert out.shape == latent.shape
    assert out.dtype == latent.dtype


def test_interpolation_matches_pids_flow_matching_frame():
    # x_t = (1 - s) * x_0 + s * eps, so recovering eps from the output must reproduce the
    # same noise draw. Guards against a drifting convention (e.g. additive sigma * eps).
    latent, sigma, seed = _latent(), 0.25, 11
    out = _degrade_latent(latent=latent, degrade_sigma=sigma, seed=seed)
    recovered = (out - (1 - sigma) * latent) / sigma
    expected = mx.random.normal(latent.shape, key=mx.random.key(seed ^ 0x91D)).astype(latent.dtype)
    assert float(mx.max(mx.abs(recovered - expected))) < 1e-4


def test_degradation_is_reproducible_for_a_seed():
    a = _degrade_latent(latent=_latent(), degrade_sigma=0.3, seed=42)
    b = _degrade_latent(latent=_latent(), degrade_sigma=0.3, seed=42)
    assert mx.array_equal(a, b)


def test_degradation_stream_is_independent_of_the_samplers():
    # The sampler seeds the global RNG (mx.random.seed) for its pixel noise. Degradation uses
    # a keyed stream, so drawing it must not depend on -- or disturb -- global state, or
    # raising sigma would reshuffle the whole image instead of only softening the latent.
    mx.random.seed(1234)
    first = _degrade_latent(latent=_latent(), degrade_sigma=0.2, seed=5)
    mx.random.seed(9999)
    second = _degrade_latent(latent=_latent(), degrade_sigma=0.2, seed=5)
    assert mx.array_equal(first, second)


def test_larger_sigma_moves_the_latent_further():
    latent = _latent()
    deltas = [
        float(mx.mean(mx.abs(_degrade_latent(latent=latent, degrade_sigma=s, seed=3) - latent)))
        for s in (0.1, 0.2, 0.4)
    ]
    assert deltas == sorted(deltas), deltas


@pytest.mark.parametrize("sigma", [-0.1, PID_MAX_DEGRADE_SIGMA + 0.01, 1.0])
def test_out_of_range_sigma_is_rejected(sigma):
    with pytest.raises(ValueError, match="pid_degrade_sigma"):
        _degrade_latent(latent=_latent(), degrade_sigma=sigma, seed=0)


def test_the_trained_ceiling_itself_is_allowed():
    out = _degrade_latent(latent=_latent(), degrade_sigma=PID_MAX_DEGRADE_SIGMA, seed=0)
    assert out.shape == (1, 16, 8, 8)
