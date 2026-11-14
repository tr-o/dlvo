"""Run with python -m dlvo --help."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

from . import E_CHARGE, Parameters


def main():
    parser = argparse.ArgumentParser(description="DLVO energy of two identical spheres")
    parser.add_argument("--radius-nm", type=float, default=100)
    parser.add_argument("--potential-mv", type=float, default=20)
    parser.add_argument("--ionic-strength-m", type=float, default=1e-3, help="mol/L")
    parser.add_argument("--hamaker-j", type=float, default=1e-20)
    parser.add_argument("--temperature-k", type=float, default=298.15)
    parser.add_argument("--relative-permittivity", type=float, default=78.5)
    parser.add_argument("--h-min-nm", type=float, default=0.3)
    parser.add_argument("--h-max-nm", type=float, default=10)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    try:
        params = Parameters(
            radius_m=args.radius_nm * 1e-9,
            potential_v=args.potential_mv * 1e-3,
            ionic_strength_mol_l=args.ionic_strength_m,
            hamaker_j=args.hamaker_j,
            temperature_k=args.temperature_k,
            relative_permittivity=args.relative_permittivity,
        )
        if not (math.isfinite(args.h_min_nm) and math.isfinite(args.h_max_nm)
                and 0 < args.h_min_nm < args.h_max_nm):
            raise ValueError("require finite 0 < h-min-nm < h-max-nm")
        if args.points < 2:
            raise ValueError("points must be at least 2")
    except ValueError as exc:
        parser.error(str(exc))

    notes = []
    if E_CHARGE * abs(params.potential_v) / params.thermal_energy_j > 0.5:
        notes.append("Linear PB is approximate: |e psi/kBT| > 0.5.")
    if params.debye_length_m / params.radius_m > 0.1:
        notes.append("Derjaguin approximation is weak: Debye length/radius > 0.1.")
    if args.h_max_nm / args.radius_nm > 0.1:
        notes.append("Derjaguin approximation is weak: maximum gap/radius > 0.1.")
    for note in notes:
        print(f"Model note: {note}", file=sys.stderr)

    rows = []
    for i in range(args.points):
        h_nm = args.h_min_nm * (args.h_max_nm / args.h_min_nm)**(i / (args.points - 1))
        energies = params.energy_j(h_nm * 1e-9)
        rows.append((h_nm, *energies, *(u / params.thermal_energy_j for u in energies)))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "dlvo.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["h_nm", "vdw_j", "double_layer_j", "total_j",
                         "vdw_kbt", "double_layer_kbt", "total_kbt"])
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for column, label in ((4, "van der Waals"), (5, "Double layer"), (6, "Total DLVO")):
        ax.plot([r[0] for r in rows], [r[column] for r in rows], label=label)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set(xscale="log", xlabel="Surface separation h (nm)", ylabel="Interaction energy (kBT)",
           title=f"DLVO: a={args.radius_nm:g} nm, psi={args.potential_mv:g} mV, I={args.ionic_strength_m:g} M")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(args.output / "dlvo.png", dpi=180)
    plt.close(fig)

    maximum = max(rows, key=lambda row: row[6])
    summary = {
        "parameters_si": vars(params),
        "debye_length_nm": params.debye_length_m * 1e9,
        "sampled_maximum": {"h_nm": maximum[0], "energy_kbt": maximum[6],
                            "at_grid_boundary": maximum in (rows[0], rows[-1])},
        "grid": {"h_min_nm": args.h_min_nm, "h_max_nm": args.h_max_nm, "points": args.points},
        "notes": notes,
        "maximum_definition": "Sampled maximum in requested range, relative to infinite separation; not a fitted barrier or activation energy.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Debye length: {summary['debye_length_nm']:.4f} nm")
    print(f"Sampled maximum: {maximum[6]:.4f} kBT at h={maximum[0]:.4f} nm")
    print(f"CSV, plot and parameters: {args.output.resolve()}")


if __name__ == "__main__":
    main()
