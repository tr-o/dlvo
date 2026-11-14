"""Reproduce with python -m studies.batio3_ethanol (install .[research])."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dlvo import E_CHARGE, Parameters
from dlvo.nonlinear import dimensionless_edl, energies

OUT = Path(__file__).resolve().parent / "batio3_ethanol"
DIAMETERS = [50, 100, 200, 500]
POTENTIALS = np.arange(10, 101, 5)
STRENGTHS = np.geomspace(1e-6, 1e-2, 81)
HAMAKERS = [4e-20, 8e-20, 12e-20]  # Sensitivity assumptions, NOT ethanol measurements.


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Dimensionless PB solution depends on potential alone, reused across I/a/A.
    gaps = np.geomspace(1e-4, 30, 700)
    reference = Parameters(relative_permittivity=24.3)
    solutions = {}
    for potential in POTENTIALS:
        solutions[int(potential)] = dimensionless_edl(
            potential * 1e-3 * E_CHARGE / reference.thermal_energy_j, gaps)
        print(f"Solved nonlinear PB at {potential} mV", flush=True)
    records = []
    curves = []
    for diameter in DIAMETERS:
        for hamaker in HAMAKERS:
            for potential in POTENTIALS:
                for strength in STRENGTHS:
                    p = Parameters(radius_m=diameter * 0.5e-9, potential_v=potential * 1e-3,
                                   ionic_strength_mol_l=float(strength), hamaker_j=hamaker,
                                   relative_permittivity=24.3)
                    h, vdw, edl, total = energies(p, gaps, solutions[int(potential)])
                    # Only nonmolecular gaps, far endpoint >=20 screening lengths.
                    mask = (h >= 0.3e-9) & (gaps <= 20)
                    hh, uu = h[mask], total[mask] / p.thermal_energy_j
                    j = int(np.argmax(uu))
                    interior = j not in (0, len(uu) - 1)
                    barrier = max(0, float(uu[j])) if interior else 0.0
                    thin = p.debye_length_m / p.radius_m <= 0.1
                    small_gap = hh[j] / p.radius_m <= 0.1
                    eligible = interior and thin and small_gap
                    records.append({
                        "diameter_nm": diameter, "hamaker_j": hamaker,
                        "surface_potential_mv": int(potential), "free_ionic_strength_m": float(strength),
                        "debye_length_nm": p.debye_length_m * 1e9,
                        "positive_peak_kbt": barrier, "peak_h_nm": float(hh[j] * 1e9),
                        "interior_peak": interior, "thin_double_layer": thin,
                        "small_peak_gap": bool(small_gap), "geometry_eligible": bool(eligible),
                        "candidate_20kbt": bool(eligible and barrier >= 20),
                    })
                    if diameter == 100 and hamaker == 8e-20 and potential in (30, 50, 70, 90) and abs(strength-1e-3) < 1e-12:
                        for k in np.flatnonzero(mask):
                            curves.append([potential, h[k]*1e9, vdw[k]/p.thermal_energy_j,
                                           edl[k]/p.thermal_energy_j, total[k]/p.thermal_energy_j])
    with (OUT / "scan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    with (OUT / "curves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["potential_mv", "h_nm", "vdw_kbt", "edl_kbt", "total_kbt"])
        writer.writerows(curves)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for diameter, ax in zip(DIAMETERS, axes.flat):
        subset = [r for r in records if r["diameter_nm"] == diameter and r["hamaker_j"] == 8e-20]
        values = np.array([r["positive_peak_kbt"] for r in subset]).reshape(len(POTENTIALS), -1)
        allowed = np.array([r["geometry_eligible"] for r in subset]).reshape(values.shape)
        mesh = ax.pcolormesh(STRENGTHS, POTENTIALS, values, shading="auto", vmin=0, vmax=60, cmap="viridis")
        ax.contour(STRENGTHS, POTENTIALS, values, levels=[10, 20, 40], colors="white", linewidths=0.8)
        ax.contourf(STRENGTHS, POTENTIALS, (~allowed).astype(float), levels=[0.5, 1.5], colors="none", hatches=["///"])
        ax.set(xscale="log", xlabel="Free ionic strength (mol/L)", ylabel="Surface potential magnitude (mV)", title=f"Diameter {diameter} nm")
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Positive interior maximum relative to infinity (kBT)")
    fig.suptitle("BaTiO3 / ethanol, 25 C, assumed A=8e-20 J\nHatched: geometry criteria not met or no resolved interior peak")
    fig.savefig(OUT / "stability_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for potential in (30, 50, 70, 90):
        rows = np.array([r for r in curves if r[0] == potential])
        ax.plot(rows[:, 1], rows[:, 4], label=f"{potential} mV")
    ax.set(xscale="log", xlim=(0.3, 100), ylim=(-60, 70), xlabel="Surface gap (nm)", ylabel="Total energy (kBT)",
           title="Diameter 100 nm, I=1 mM, assumed A=8e-20 J\nIndicative: Debye length/radius=0.107 exceeds strict 0.1 criterion")
    ax.axhline(20, linestyle="--", color="gray", label="20 kBT screening target")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "energy_curves.png", dpi=180)
    plt.close(fig)

    thresholds = []
    for diameter in DIAMETERS:
        for hamaker in HAMAKERS:
            candidates = [r for r in records if r["diameter_nm"] == diameter and r["hamaker_j"] == hamaker and r["candidate_20kbt"]]
            best = min(candidates, key=lambda r: (r["surface_potential_mv"], r["free_ionic_strength_m"])) if candidates else None
            thresholds.append({"diameter_nm": diameter, "hamaker_j": hamaker, "lowest_sampled_potential_candidate": best})
    metadata = {
        "model": "Nonlinear constant-potential planar PB + Derjaguin for identical spheres; nonretarded Derjaguin vdW",
        "temperature_k": 298.15, "relative_permittivity": 24.3,
        "hamaker_status": "4, 8, 12 e-20 J are sensitivity assumptions; not measured for ethanol",
        "diameters_nm": DIAMETERS, "potentials_mv": POTENTIALS.tolist(),
        "ionic_strength_range_m": [1e-6, 1e-2], "ionic_strength_points": 81,
        "gap_grid": {"dimensionless_min": 1e-4, "dimensionless_max": 30, "points": 700,
                     "analysis_min_h_nm": 0.3, "analysis_max_kappa_h": 20},
        "geometry_criteria": "Debye length/a <= 0.1 AND peak gap/a <= 0.1 AND interior positive maximum",
        "criterion": "20 kBT is a screening target only; not a validated stability time",
        "thresholds": thresholds,
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for row in thresholds:
        best = row["lowest_sampled_potential_candidate"]
        print(row["diameter_nm"], row["hamaker_j"], "none" if best is None else
              (best["surface_potential_mv"], best["free_ionic_strength_m"], best["positive_peak_kbt"]))


if __name__ == "__main__":
    main()
