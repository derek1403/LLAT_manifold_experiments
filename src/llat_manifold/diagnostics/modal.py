"""Modal analysis of perturbation fields — RESERVED.

Reserved space for EOF / higher-moment modal decomposition of the δ ensembles
(e.g. the snapshot power-iteration sequence, which converges toward the leading
finite-time singular/normal mode). Intended capabilities:

* stack a sequence of δ bundles into a (samples × space) matrix;
* EOF/SVD to extract dominant spatial modes and their amplitude time series;
* higher moments (skewness/kurtosis) of the δ distribution to flag nonlinearity.

Not implemented in the first slice — see ``docs/perturbation_method.md`` for why the
snapshot iteration relates to the leading mode.
"""
from __future__ import annotations


def eof(*args, **kwargs):
    raise NotImplementedError("modal.eof is reserved; not implemented in the first slice.")
