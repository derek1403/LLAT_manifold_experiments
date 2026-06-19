"""TC-centre tracking on the FCNv2 global MSL field.

Ported from the reference scripts (``find_tc_center_msl``). The coupled driver uses
this each step to recentre TC-following perturbations (e.g. the continuous heating
Gaussian) on the storm.
"""
from __future__ import annotations

import numpy as np

# FCNv2 global grid: 721 lats (90 -> -90), 1440 lons (0 -> 359.75).
_LATS = np.linspace(90, -90, 721)
_LONS = np.linspace(0, 360 - 0.25, 1440)


def find_tc_center_msl(msl: np.ndarray, prev_lat: float, prev_lon: float,
                       search_radius_deg: float = 5.0):
    """Locate the MSL minimum within ``search_radius_deg`` of the previous centre.

    ``msl`` is the FCNv2 MSL field, shape (721, 1440). Returns
    ``(lat, lon, min_value)``. Does not handle the longitude wrap (matches the
    reference; fine for WPAC cases away from the date line).
    """
    lat_idx = np.where((_LATS >= prev_lat - search_radius_deg) &
                       (_LATS <= prev_lat + search_radius_deg))[0]
    lon_idx = np.where((_LONS >= prev_lon - search_radius_deg) &
                       (_LONS <= prev_lon + search_radius_deg))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError("TC search window out of bounds; check centre/radius.")

    sub = msl[np.ix_(lat_idx, lon_idx)]
    si, sj = np.unravel_index(np.argmin(sub), sub.shape)
    i_lat = lat_idx[si]
    i_lon = lon_idx[sj]
    return float(np.round(_LATS[i_lat], 2)), float(np.round(_LONS[i_lon], 2)), float(np.min(sub))
