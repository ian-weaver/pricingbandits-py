"""Fallback/error-handling diagnostics.

Mirrors the R package's RNG-neutral counters: they never draw random numbers,
so instrumented runs behave identically to uninstrumented ones.
"""

from __future__ import annotations

_COUNTER_NAMES = [
    "HyperoptPrior",        # hyperparameter optimization failed -> priors used
    "GPErrorPrevAS",        # GP policy errored -> previous action scores reused
    "MonoTimeout1",         # basis-mono: first truncated-MVN attempt timed out
    "MonoRetryError",       # basis-mono: retry errored -> last resort
    "PolicyUpdates",        # number of policy evaluations performed
]

_state: dict = {}


def reset_diagnostics():
    """Reset all counters."""
    for name in _COUNTER_NAMES:
        _state[name] = 0


def bump(name):
    """Increment a counter (creating it if needed)."""
    _state[name] = _state.get(name, 0) + 1


def get_diagnostics():
    """Return a copy of the current counters."""
    return dict(_state)


reset_diagnostics()
