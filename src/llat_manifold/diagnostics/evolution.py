"""Per-iteration *evolution* views of a run — frames, GIFs, and growth curves.

The standard suite (:mod:`llat_manifold.diagnostics.suite`) renders only the **last**
bundle of a run. For a ``snapshot`` run that bundle is the converged end-state of the
LLAT semi-linear power iteration; for ``continuous``/``forward`` it is the final lead.
This module instead walks **every** δ bundle and reuses the existing per-bundle
plotters to build, into ``plots/evolution/``:

  * one PNG frame per iteration for each chosen diagnostic, stamped ``iter k/N``;
  * an animated GIF per diagnostic stitched from those frames (Pillow);
  * a ``growth_curves.png`` summary of scalar δ-amplitude metrics vs iteration —
    the quantitative companion that shows the power iteration growing / saturating.

For ``snapshot`` the GIF axis is the power-iteration index (frozen valid time), not
physical time. Bundles are ordered by the same ``iter``/``lead`` key the suite uses, so
the module is mode-agnostic.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import io, layout
from . import experiment_title, load_stamp
from . import pv, divergence, fields, wind_profile, hydrostatic
# Reuse the suite's bundle helpers rather than re-deriving them.
from .suite import _key, _resolve_baseline, _absolute_bundle

# diagnostic key -> (subfolder name, needs absolute ū+δ bundle?)
_DIAGS = {
    "pv": ("PV_Theta", False),
    "div": ("div_Theta", False),
    "wind_circ": ("wind_circulation", True),
    "wind_bal": ("wind_balance", True),
    "fields": ("fields", True),
    # Non-hydrostatic checks: animate the spatially-rich r–z map (the profile is the
    # static end-state figure). Both need the absolute ū+δ state.
    "hydro_eps": ("hydrostatic_eps", True),
    "hydro_thermo": ("hydrostatic_thermo", True),
}
_DEFAULT = ("pv", "div", "wind_circ", "wind_bal", "fields", "hydro_eps", "hydro_thermo")


def _iter_label(path: Path) -> str:
    """Short ``iterNNN`` / ``leadNNNhr`` token for a frame filename."""
    m = re.search(r"(iter\d+|lead\d+hr)", path.name)
    return m.group(1) if m else path.stem


def _pil_stamp(frame_path: Path, k: int, n: int) -> None:
    """Overlay an ``iter k/N`` label onto an already-saved PNG (for plotters that own
    their own save/close, e.g. the vendored 2D-field grid)."""
    from PIL import Image, ImageDraw

    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    txt = f"iter {k:02d}/{n}"
    w, h = img.size
    draw.text((w - 110, h - 26), txt, fill=(0, 0, 0))
    img.save(frame_path)


def _stamp_and_save(fig, frame_path: Path, k: int, n: int) -> None:
    """Annotate a returned figure with the iteration index, save, and release it."""
    import matplotlib.pyplot as plt

    fig.text(0.99, 0.01, f"iter {k:02d}/{n}", ha="right", va="bottom",
             fontsize=11, weight="bold")
    fig.savefig(frame_path, dpi=160, bbox_inches="tight")
    # Memory hygiene: clear the figure's Agg buffers *then* drop the pyplot ref. Over
    # 20×5 high-res frames, plt.close alone can let the canvas backing arrays linger.
    fig.clf()
    plt.close(fig)


def _render_frames(diag: str, bundles, evo: Path, baseline, add, data: Path) -> list[str]:
    """Render one diagnostic's frames over all bundles; return frame paths in order."""
    subdir, needs_abs = _DIAGS[diag]
    out_dir = evo / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(bundles)
    frames: list[str] = []

    for k, b in enumerate(bundles, start=1):
        frame = out_dir / f"{_iter_label(b)}.png"
        try:
            if diag == "pv":
                fig = pv.plot_pv_theta_cross_section(
                    b, out_png=None, baseline_path=baseline, add_to_baseline=add)
            elif diag == "div":
                fig = divergence.plot_div_theta_cross_section(
                    b, out_png=None, baseline_path=baseline, add_to_baseline=add)
            elif diag == "fields":
                # The vendored 5-row grid owns its own save/close and does not return a
                # usable Figure, so save straight to the frame and stamp via PIL.
                ab = _absolute_bundle(b, baseline, data)
                try:
                    fields.plot_fields(ab, out_png=str(frame))
                finally:
                    if Path(ab) != Path(b):
                        Path(ab).unlink(missing_ok=True)
                _pil_stamp(frame, k, n)
                frames.append(str(frame))
                continue
            else:
                # Absolute-state plots: build ū+δ once and share across them.
                ab = _absolute_bundle(b, baseline, data)
                try:
                    if diag == "wind_circ":
                        fig = pv.plot_wind_circulation_cross_section(ab, out_png=None)
                    elif diag == "hydro_eps":
                        fig = hydrostatic.plot_nonhydrostatic_epsilon(
                            ab, out_png=None, which="xsection")
                    elif diag == "hydro_thermo":
                        fig = hydrostatic.plot_hydrostatic_thermo(
                            ab, out_png=None, which="xsection")
                    else:  # wind_bal
                        fig = wind_profile.plot_wind_balance_profile(ab, out_png=None)
                finally:
                    # Keep data/ clean: drop the temp _abs bundle (never the real δ).
                    if Path(ab) != Path(b):
                        Path(ab).unlink(missing_ok=True)
            _stamp_and_save(fig, frame, k, n)
            frames.append(str(frame))
        except Exception as e:  # noqa: BLE001
            print(f"[evolution] {diag} {_iter_label(b)} skipped: {type(e).__name__}: {e}")
    return frames


def _make_gif(frames: list[str], gif_path: Path, fps: int) -> str | None:
    """Stitch ordered PNG frames into a looping animated GIF (Pillow)."""
    if len(frames) < 2:
        return None
    from PIL import Image

    imgs = [Image.open(f).convert("RGB") for f in frames]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0)
    print(f"[evolution] wrote {gif_path} ({len(frames)} frames)")
    return str(gif_path)


def _growth_curves(bundles, run_dir: Path, out_png: Path) -> str:
    """Scalar δ-amplitude metrics vs iteration: max & 95th-pct of |δT| and |δ wind|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ti, ui, vi = (layout.upper_index("t"), layout.upper_index("u"),
                  layout.upper_index("v"))
    it, dt_max, dt_p95, dt_rms, w_max, w_p95 = [], [], [], [], [], []
    for k, b in enumerate(bundles, start=1):
        up, _ = io.load_delta_bundle(b)
        dt = np.abs(up[..., ti])
        wind = np.sqrt(up[..., ui] ** 2 + up[..., vi] ** 2)
        it.append(k)
        dt_max.append(float(dt.max()));   dt_p95.append(float(np.percentile(dt, 95)))
        dt_rms.append(float(np.sqrt(np.mean(dt ** 2))))
        w_max.append(float(wind.max()));  w_p95.append(float(np.percentile(wind, 95)))

    fig, axL = plt.subplots(figsize=(9, 6))
    axR = axL.twinx()
    # δT (K) on the left axis: max solid, 95th-pct dashed (same colour); RMS thin.
    axL.plot(it, dt_max, color="firebrick", lw=2.2, marker="o", ms=4, label="max |δT|")
    axL.plot(it, dt_p95, color="firebrick", lw=2.0, ls="--", marker="^", ms=4,
             label="p95 |δT|")
    axL.plot(it, dt_rms, color="darkorange", lw=1.6, ls=":", label="RMS δT")
    # δ wind (m/s) on the right axis.
    axR.plot(it, w_max, color="navy", lw=2.2, marker="s", ms=4, label="max |δ wind|")
    axR.plot(it, w_p95, color="navy", lw=2.0, ls="--", marker="v", ms=4,
             label="p95 |δ wind|")

    axL.set_xlabel("Iteration", fontsize=13, weight="bold")
    axL.set_ylabel("δ Temperature [K]", fontsize=13, weight="bold", color="firebrick")
    axR.set_ylabel("δ Wind speed [m s$^{-1}$]", fontsize=13, weight="bold", color="navy")
    axL.tick_params(axis="y", colors="firebrick")
    axR.tick_params(axis="y", colors="navy")
    axL.set_xlim(min(it), max(it))
    axL.grid(True, ls="--", alpha=0.5)
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10, ncol=2)
    axL.set_title(experiment_title(run_dir, prefix="growth:"), fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.clf()
    plt.close(fig)
    print(f"[evolution] wrote {out_png}")
    return str(out_png)


def evolution_plots(run_dir, *, diagnostics=_DEFAULT, make_gif=True, fps=2,
                    stride=1) -> list[str]:
    """Render per-iteration frames + GIFs + a growth-curve summary for ``run_dir``."""
    run_dir = Path(run_dir)
    data = run_dir / "data"
    plots = run_dir / "plots"

    bundles = sorted(data.glob("delta_*.npz"), key=_key)
    if stride > 1:
        bundles = bundles[::stride]
    if len(bundles) < 2:
        print(f"[evolution] need ≥2 δ bundles in {data} (found {len(bundles)})")
        return []

    mode = load_stamp(run_dir).get("resolved_config", {}).get("mode", "")
    baseline = _resolve_baseline(data, bundles[-1], mode)
    add = baseline is not None

    evo = plots / "evolution"
    evo.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []

    for diag in diagnostics:
        if diag not in _DIAGS:
            print(f"[evolution] unknown diagnostic {diag!r}, skipping")
            continue
        frames = _render_frames(diag, bundles, evo, baseline, add, data)
        produced.extend(frames)
        if make_gif:
            gif = _make_gif(frames, evo / f"{_DIAGS[diag][0]}.gif", fps)
            if gif:
                produced.append(gif)

    try:
        produced.append(_growth_curves(bundles, run_dir, evo / "growth_curves.png"))
    except Exception as e:  # noqa: BLE001
        print(f"[evolution] growth_curves skipped: {type(e).__name__}: {e}")

    print(f"[evolution] produced {len(produced)} file(s) under {evo}")
    return produced


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="per-iteration evolution frames/GIFs/curves")
    ap.add_argument("run_dir", help="experiment run dir (contains data/ and plots/)")
    ap.add_argument("--diag", default=",".join(_DEFAULT),
                    help="comma-separated subset of: " + ",".join(_DIAGS))
    ap.add_argument("--stride", type=int, default=1, help="use every Nth bundle")
    ap.add_argument("--fps", type=int, default=2, help="GIF frames per second")
    ap.add_argument("--no-gif", action="store_true", help="frames only, skip GIFs")
    a = ap.parse_args()
    evolution_plots(a.run_dir, diagnostics=tuple(d.strip() for d in a.diag.split(",")),
                    make_gif=not a.no_gif, fps=a.fps, stride=a.stride)
