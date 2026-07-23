"""Large-scale 2D field panels — reuses Part 1's field-comparison plotter.

Rather than reinvent the 5-row field grid (10 m wind / precip / 850 hPa vorticity /
700 hPa ω / TCWV), this builds a DLAMPty xarray from a bundle's upper/surface arrays
using the **vendored** ``to_xarray`` and renders it with Part 1's
``regional_couple.plotting.fields.plot_fields_comparison`` (same colormaps, layout and
labels as ``outputs/.../fields/fields_048h.png``).
"""
from __future__ import annotations

from pathlib import Path

from .. import io
from . import experiment_title

# Importing this puts the frozen vendor code (utils.data_processor) on sys.path.
from regional_couple import _vendor  # noqa: F401
from regional_couple.plotting import fields as rc_fields


def bundle_to_xarray(upper, sfc):
    """DLAMPty (upper, sfc) arrays -> plottable xarray with derived fields."""
    from utils.data_processor import to_xarray
    uv, sv, uu, su, pl = rc_fields._var_spec()
    ds = to_xarray(upper, sfc, uv, sv, uu, su, pl)
    try:
        from utils.data_processor import calc_additional_vars
        return calc_additional_vars(ds).squeeze()
    except Exception:
        # Fall back to the lighter derived-field set (ws10 + vort via metpy).
        return rc_fields._add_plot_fields(ds)


def plot_fields(delta_path, out_png=None, label="LLAT"):
    """Render the 5-row field grid for one bundle (one column)."""
    delta_path = Path(delta_path)
    if out_png is None:
        # Default into the run's plots/ — never hand None down to the Part-1
        # saver (that once produced a stray "None.png" at the CWD).
        out_png = delta_path.parent.parent / "plots" / "fields.png"
    up, sfc = io.load_delta_bundle(delta_path)
    ds = bundle_to_xarray(up, sfc)
    title = experiment_title(delta_path.parent.parent, prefix="fields:")
    out = rc_fields.plot_fields_comparison({label: ds}, out_png, title=title)
    print(f"[fields] wrote {out_png}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="2D field panels for a bundle")
    ap.add_argument("delta_npz")
    ap.add_argument("--out", default="fields.png")
    a = ap.parse_args()
    plot_fields(a.delta_npz, a.out)
