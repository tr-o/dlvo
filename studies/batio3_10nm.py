"""Run python -m studies.batio3_10nm for curved-surface 10 nm DLVO study."""

from dataclasses import replace
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from dlvo import E_CHARGE, Parameters
from dlvo.nonlinear import dimensionless_edl, energies
from dlvo.spheres import SphereGeometry, sphere_vdw_j

OUT = Path(__file__).resolve().parent / "batio3_10nm"
POTENTIALS = [20, 34, 50, 75, 100]
STRENGTHS = [1e-6, 1e-4, 1e-3, 1e-2]
HAMAKERS = [4e-20, 8e-20, 12e-20]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = Parameters(radius_m=5e-9, relative_permittivity=24.3, hamaker_j=8e-20)
    records, summaries, validations = [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for strength, ax in zip(STRENGTHS, axes.flat):
        base = replace(ref, ionic_strength_mol_l=strength)
        # Extend beyond 12 screening lengths, including the 2a centre offset.
        h = np.geomspace(.3e-9, max(100e-9, 12*base.debye_length_m), 65)
        forces = {psi: [] for psi in POTENTIALS}
        for j, gap in enumerate(h):
            geometry = SphereGeometry(base.radius_m, base.debye_length_m, float(gap), resolution=20)
            for psi in POTENTIALS:
                p = replace(base, potential_v=psi*1e-3)
                solution = geometry.solve(p)
                forces[psi].append(solution.force_n)
            if j % 16 == 0:
                print(f"I={strength:g} M, gap {j+1}/{len(h)}", flush=True)
            # Representative potential field at 1 mM, closest gap to 1 nm.
            if strength == 1e-3 and j == np.argmin(abs(h-1e-9)):
                field = geometry.solve(replace(base, potential_v=.034))
                np.savez_compressed(OUT / "potential_field.npz", r_m=field.r_m, z_m=field.z_m,
                                    potential_v=field.potential_v, gap_m=gap)
                draw_field(field, float(gap))
        for psi in POTENTIALS:
            force = np.asarray(forces[psi])
            primitive = PchipInterpolator(h, force).antiderivative()
            edl = primitive(h[-1]) - primitive(h)
            trapezoid = -cumulative_trapezoid(force[::-1], h[::-1], initial=0)[::-1]
            # Save force integration discrepancy near the resulting primary maximum.
            for ah in HAMAKERS:
                vdw = sphere_vdw_j(base.radius_m, h, ah)
                total = (edl + vdw) / base.thermal_energy_j
                j = int(np.argmax(total))
                summaries.append({"diameter_nm": 10, "surface_potential_mv": psi,
                                  "free_ionic_strength_m": strength, "hamaker_j": ah,
                                  "positive_maximum_kbt": max(0, float(total[j])),
                                  "peak_h_nm": float(h[j]*1e9), "at_boundary": j in (0, len(h)-1),
                                  "force_integration_difference_kbt": float(abs(edl[j]-trapezoid[j])/base.thermal_energy_j),
                                  "last_force_n": float(force[-1])})
                for index in range(len(h)):
                    records.append([strength, psi, ah, h[index]*1e9, force[index],
                                    edl[index]/base.thermal_energy_j, vdw[index]/base.thermal_energy_j,
                                    total[index]])
                if ah == 8e-20:
                    ax.plot(h*1e9, total, label=f"{psi} mV")
        ax.set(xscale="log", xlim=(.3, 200), ylim=(-12, 24), xlabel="Surface gap (nm)",
               ylabel="Interaction energy (kBT)", title=f"Free ionic strength = {strength:g} M")
        ax.axhline(0, color="gray", linewidth=.7)
        ax.axhline(20, color="gray", linestyle="--", linewidth=.8)
        ax.grid(alpha=.2)
        ax.legend(ncol=2, fontsize=8)
    fig.suptitle("10 nm BaTiO3 in ethanol: nonlinear spherical PB + exact-sphere Hamaker\n25 C, constant equal surface potentials, assumed A=8e-20 J")
    fig.savefig(OUT / "energy_curves.png", dpi=180)
    plt.close(fig)

    with (OUT / "curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["free_ionic_strength_m", "surface_potential_mv", "hamaker_j", "h_nm",
                         "edl_force_n", "edl_kbt", "vdw_kbt", "total_kbt"])
        writer.writerows(records)
    with (OUT / "peaks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    # Check the hardest overlap case and both ends of the ionic-strength range.
    for strength, psi, gap_nm in [(1e-6, 100, 1), (1e-3, 34, 1), (1e-2, 100, 1)]:
        p = replace(ref, ionic_strength_mol_l=strength, potential_v=psi*1e-3)
        cases = []
        for resolution, padding in [(12, 12), (20, 12), (32, 12), (20, 16)]:
            s = SphereGeometry(p.radius_m, p.debye_length_m, gap_nm*1e-9,
                               resolution=resolution, padding=padding).solve(p)
            cases.append({"resolution": resolution, "padding": padding, "force_n": s.force_n,
                          "scaled_residual": s.scaled_residual, "unknowns": s.unknowns})
        validations.append({"ionic_strength_m": strength, "potential_mv": psi,
                            "gap_nm": gap_nm, "cases": cases})

    # Separate curvature effects in EDL and vdW at literature-like 34 mV.
    compare_approximations(ref, records)
    metadata = {"radius_m": ref.radius_m, "temperature_k": ref.temperature_k,
                "relative_permittivity": ref.relative_permittivity,
                "boundary": "Equal fixed surface potentials; mirror plane and axis symmetry; zero far-field potential",
                "model": "Axisymmetric nonlinear 1:1 PB solved around curved sphere boundaries; midpoint osmotic + Maxwell stress; exact nonretarded Hamaker spheres",
                "resolution": 20, "outer_padding_debye_lengths": 12, "gap_points": 65,
                "potentials_mv": POTENTIALS, "ionic_strengths_m": STRENGTHS,
                "hamakers_j": HAMAKERS, "hamaker_status": "Sensitivity assumptions, not measured ethanol values",
                "maxima": summaries, "validation": validations,
                "omitted": ["charge regulation", "ion pairing", "charge discreteness", "steric and solvation forces", "retardation", "ferroelectric interactions"],
                "criterion": "20 kBT is a screening heuristic, not a validated storage time"}
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Study complete", flush=True)
    for psi in POTENTIALS:
        best = max((s for s in summaries if s["surface_potential_mv"] == psi), key=lambda s:s["positive_maximum_kbt"])
        print(psi, best, flush=True)


def draw_field(field, gap):
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    mask_r, mask_z = field.r_m <= 25e-9, field.z_m <= 25e-9
    r, z = field.r_m[mask_r]*1e9, field.z_m[mask_z]*1e9
    v = field.potential_v[np.ix_(mask_r, mask_z)]*1e3
    mesh = ax.pcolormesh(r, z, v.T, shading="auto", cmap="magma", vmin=0, vmax=34)
    angle = np.linspace(0, np.pi, 200)
    ax.plot(5*np.sin(angle), 5+gap*0.5e9+5*np.cos(angle), color="cyan")
    ax.set(aspect="equal", xlabel="r (nm)", ylabel="z (nm)",
           title=f"Half-domain potential: 10 nm sphere, 34 mV, 1 mM\nMirror plane z=0; surface gap={gap*1e9:.3g} nm")
    fig.colorbar(mesh, ax=ax, label="Potential (mV)")
    fig.savefig(OUT / "potential_field.png", dpi=180)
    plt.close(fig)


def compare_approximations(ref, records):
    gaps = np.geomspace(1e-4, 30, 700)
    scaled = dimensionless_edl(.034*E_CHARGE/ref.thermal_energy_j, gaps)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    comparison = []
    for strength, color in [(1e-6, "tab:blue"), (1e-3, "tab:orange")]:
        p = replace(ref, ionic_strength_mol_l=strength, potential_v=.034)
        h, vdw, edl, total = energies(p, gaps, scaled)
        rows = [r for r in records if r[0] == strength and r[1] == 34 and r[2] == 8e-20]
        arr = np.asarray(rows)
        ax.plot(arr[:, 3], arr[:, 7], color=color, label=f"Curved PB + sphere vdW, I={strength:g} M")
        ax.plot(h*1e9, total/p.thermal_energy_j, color=color, linestyle="--", label=f"Derjaguin both, I={strength:g} M")
        for row in rows:
            gap = row[3]*1e-9
            planar_edl = np.interp(gap/p.debye_length_m, gaps, edl) / p.thermal_energy_j
            planar_vdw = -p.hamaker_j*p.radius_m/(12*gap*p.thermal_energy_j)
            comparison.append([strength, row[3], row[5], planar_edl, row[6], planar_vdw])
    ax.set(xscale="log", xlim=(.3, 200), ylim=(-15, 5), xlabel="Surface gap (nm)",
           ylabel="Interaction energy (kBT)", title="10 nm, 34 mV: curvature changes both EDL and vdW")
    ax.axhline(0, color="gray", linewidth=.7)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(OUT / "approximation_comparison.png", dpi=180)
    plt.close(fig)
    with (OUT / "approximation_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["free_ionic_strength_m", "h_nm", "curved_edl_kbt", "derjaguin_edl_kbt", "sphere_vdw_kbt", "derjaguin_vdw_kbt"])
        writer.writerows(comparison)


if __name__ == "__main__":
    main()
