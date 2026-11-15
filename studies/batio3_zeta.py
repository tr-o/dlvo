"""Conditional 34 mV zeta / 1 nm shear-plane correction, separate force cache."""

from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from dlvo import Parameters
from dlvo.electrokinetics import surface_potential_for_zeta
from dlvo.regulation import RegulatedGeometry, isolated_regulation_data
from dlvo.speciation import free_ionic_strength_m
from dlvo.spheres import SphereGeometry, sphere_vdw_j
from .batio3_extended import OUT, STRENGTHS, HAMAKERS, write_dict_csv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    solver_hash = hashlib.sha256(b"".join((root/"dlvo"/name).read_bytes()
                                         for name in ("spheres.py", "regulation.py", "electrokinetics.py"))).hexdigest()
    cache_inputs = {"solver_hash": solver_hash, "zeta_v": .034, "shear_offset_m": 1e-9,
                    "radius_m": 5e-9, "temperature_k": 298.15, "solvent_permittivity": 24.3,
                    "particle_permittivity": 150, "strengths": STRENGTHS, "regulations": [0, .5, 1],
                    "resolution": 20, "gap_points": 8, "gap_count": 65, "padding": 12}
    signature = OUT/"zeta_force_inputs.json"
    if signature.exists() and json.loads(signature.read_text()) != cache_inputs:
        raise RuntimeError("Zeta force cache uses different solver/inputs. Move the old output directory before rerunning.")
    signature.write_text(json.dumps(cache_inputs, indent=2)+"\n")
    ref = Parameters(radius_m=5e-9, potential_v=.034, relative_permittivity=24.3)
    mappings = []
    for strength in STRENGTHS:
        p = replace(ref, ionic_strength_mol_l=strength)
        for offset in [0, .3, .5, 1, 2]:
            try:
                surface = surface_potential_for_zeta(p, .034, offset*1e-9)*1e3
                status = "computed"
            except ValueError:
                surface, status = None, "no solution in surface potential bracket up to 10 kBT/e"
            mappings.append({"ionic_strength_m": strength, "assumed_zeta_mv": 34,
                             "assumed_shear_offset_nm": offset,
                             "inferred_surface_potential_mv": surface, "status": status})
    write_dict_csv(OUT/"zeta_mapping.csv", mappings)
    association = [{"nominal_salt_m": nominal, "assumed_K_l_mol": k,
                    "free_ionic_strength_m": free_ionic_strength_m(nominal, k)}
                   for nominal in [1e-4, 1e-3, 1e-2] for k in [0, 1e2, 1e4, 1e6]]
    write_dict_csv(OUT/"ion_pairing_sensitivity.csv", association)

    cache_path = OUT/"zeta_forces.csv"
    fields = ["ionic_strength_m", "regulation", "gap_index", "h_nm", "inferred_surface_mv", "force_n"]
    cache = {}
    if cache_path.exists():
        with cache_path.open() as f:
            cache = {tuple(float(r[k]) for k in fields[:3]): {k:float(v) for k,v in r.items()} for r in csv.DictReader(f)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    peaks = []
    with cache_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        if f.tell() == 0:
            writer.writeheader()
        for strength, ax in zip(STRENGTHS, axes):
            p = replace(ref, ionic_strength_mol_l=strength)
            surface = surface_potential_for_zeta(p, .034, 1e-9)
            p = replace(p, potential_v=surface)
            calibration = isolated_regulation_data(p)
            h = np.geomspace(.3e-9, max(100e-9, 12*p.debye_length_m), 65)
            for j, gap in enumerate(h):
                needed = [mode for mode in (0, .5, 1) if (strength, mode, j) not in cache]
                if not needed:
                    continue
                g = SphereGeometry(p.radius_m, p.debye_length_m, float(gap), resolution=20, gap_points=8)
                r = RegulatedGeometry(g)
                for mode in needed:
                    s = r.solve(p, regulation=mode, particle_permittivity=150, calibration=calibration)
                    row = dict(zip(fields, [strength, mode, j, gap*1e9, surface*1e3, s.force_n]))
                    cache[(strength, mode, j)] = row
                    writer.writerow(row)
                f.flush()
                if j%16 == 0:
                    print(f"Zeta correction: I={strength:g} M, gap {j+1}/65", flush=True)
            dense = np.geomspace(.3e-9, 100e-9, 4000)
            for mode, label in [(0, "CP"), (.5, "p=0.5"), (1, "Fixed free charge")]:
                force = [cache[(strength, mode, j)]["force_n"] for j in range(len(h))]
                primitive = PchipInterpolator(h, force).antiderivative()
                edl = (primitive(h[-1])-primitive(dense))/p.thermal_energy_j
                for ah in HAMAKERS:
                    total = edl+sphere_vdw_j(p.radius_m, dense, ah)/p.thermal_energy_j
                    j = int(np.argmax(total))
                    peaks.append({"ionic_strength_m": strength, "assumed_zeta_mv": 34,
                                  "assumed_shear_offset_nm": 1, "inferred_surface_potential_mv": surface*1e3,
                                  "regulation": mode, "particle_permittivity": 150,
                                  "hamaker_j": ah, "positive_maximum_kbt": max(0, float(total[j])),
                                  "peak_h_nm": float(dense[j]*1e9)})
                    if ah == 8e-20:
                        ax.plot(dense*1e9, total, label=label)
            ax.set(xscale="log", xlim=(.3, 100), ylim=(-10, 22), xlabel="Core surface gap (nm)",
                   ylabel="Energy (kBT)", title=f"I={strength:g} M, inferred surface {surface*1e3:.1f} mV")
            ax.axhline(20, color="gray", linestyle="--")
            ax.axhline(0, color="gray", linewidth=.7)
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Conditional zeta correction: 34 mV at a shear plane 1 nm beyond the core\n10 nm BaTiO3 / ethanol, assumed core permittivity 150, A=8e-20 J")
    fig.savefig(OUT/"zeta_corrected_curves.png", dpi=180)
    plt.close(fig)
    write_dict_csv(OUT/"zeta_corrected_peaks.csv", peaks)
    (OUT/"zeta_summary.json").write_text(json.dumps({"mappings": mappings, "peaks": peaks,
                                                    "assumptions": "Isolated spherical PB; zeta equals radial potential at an assumed shear plane. No electrokinetic mobility inversion."}, indent=2)+"\n")
    print("Zeta correction complete; maximum:", max(r["positive_maximum_kbt"] for r in peaks), flush=True)


if __name__ == "__main__":
    main()
