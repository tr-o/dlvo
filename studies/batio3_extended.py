"""10 nm curved PB with charge regulation and conditional extra repulsions."""

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

from dlvo import E_CHARGE, Parameters
from dlvo.regulation import RegulatedGeometry, isolated_regulation_data
from dlvo.short_range import corona_kbt, solvation_kbt
from dlvo.spheres import SphereGeometry, sphere_vdw_j

OUT = Path(__file__).resolve().parent/"batio3_extended"
STRENGTHS = [1e-6, 1e-3, 1e-2]
POTENTIALS = [34, 50]
HAMAKERS = [4e-20, 8e-20, 12e-20]
FORCE_FIELDS = ["ionic_strength_m", "isolated_potential_mv", "regulation", "particle_permittivity",
                "gap_index", "h_nm", "force_n", "surface_min_mv", "surface_max_mv", "residual"]


def models(strength, psi):
    result = [(0, 150), (.5, 150), (1, 150)]
    if strength == 1e-3 and psi == 34:
        result += [(p, eps) for p in (.5, 1) for eps in (30, 1000)]
    return result


def force_key(row):
    return tuple(float(row[k]) for k in FORCE_FIELDS[:5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    solver_hash = hashlib.sha256(b"".join((root/"dlvo"/name).read_bytes()
                                         for name in ("spheres.py", "regulation.py"))).hexdigest()
    config = {"solver_hash": solver_hash, "strengths": STRENGTHS, "potentials": POTENTIALS,
              "resolution": 20, "gap_points": 8, "gap_count": 65, "padding": 12,
              "radius_m": 5e-9, "temperature_k": 298.15, "solvent_permittivity": 24.3}
    config_path, force_path = OUT/"force_inputs.json", OUT/"forces.csv"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Saved forces use different solver/inputs. Move the old output directory before rerunning.")
    config_path.write_text(json.dumps(config, indent=2)+"\n")
    saved = {}
    if force_path.exists():
        with force_path.open(newline="") as f:
            saved = {force_key(row): {k:float(v) for k,v in row.items()} for row in csv.DictReader(f)}
    ref = Parameters(radius_m=5e-9, potential_v=.034, relative_permittivity=24.3, hamaker_j=8e-20)
    charges = []
    with force_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FORCE_FIELDS, lineterminator="\n")
        if f.tell() == 0:
            writer.writeheader()
        for strength in STRENGTHS:
            base = replace(ref, ionic_strength_mol_l=strength)
            gaps = np.geomspace(.3e-9, max(100e-9, 12*base.debye_length_m), 65)
            calibrations = {}
            for psi in POTENTIALS:
                p = replace(base, potential_v=psi*1e-3)
                calibrations[psi] = isolated_regulation_data(p)
                _, q, cd = calibrations[psi]
                sigma = q*p.permittivity*p.thermal_energy_j/E_CHARGE/p.radius_m
                charges.append({"ionic_strength_m": strength, "isolated_potential_mv": psi,
                                "isolated_charge_c_m2": sigma,
                                "isolated_charge_e": sigma*4*np.pi*p.radius_m**2/E_CHARGE,
                                "dimensionless_diffuse_capacitance": cd})
            for j, gap in enumerate(gaps):
                needed = [(psi, mode, eps) for psi in POTENTIALS for mode,eps in models(strength, psi)
                          if (strength, psi, mode, eps, j) not in saved]
                if not needed:
                    continue
                g = SphereGeometry(base.radius_m, base.debye_length_m, float(gap), resolution=20, gap_points=8)
                r = RegulatedGeometry(g)
                for psi, mode, eps in needed:
                    p = replace(base, potential_v=psi*1e-3)
                    s = r.solve(p, regulation=mode, particle_permittivity=eps, calibration=calibrations[psi])
                    row = dict(zip(FORCE_FIELDS, [strength, psi, mode, eps, j, gap*1e9, s.force_n,
                                                  min(s.surface_potential_v)*1e3,
                                                  max(s.surface_potential_v)*1e3, s.scaled_residual]))
                    writer.writerow(row)
                    saved[force_key(row)] = row
                f.flush()
                if j % 16 == 0:
                    print(f"Regulation: I={strength:g} M, gap {j+1}/65", flush=True)
    postprocess(ref, list(saved.values()), charges, config)


def postprocess(ref, rows, charges, config):
    curves, maxima, thresholds = [], [], []
    dense_h = np.geomspace(.3e-9, 100e-9, 4000)
    bases = {}
    for strength in STRENGTHS:
        for psi in POTENTIALS:
            for mode, eps in models(strength, psi):
                group = sorted((r for r in rows if r["ionic_strength_m"] == strength
                                and r["isolated_potential_mv"] == psi and r["regulation"] == mode
                                and r["particle_permittivity"] == eps), key=lambda r:r["gap_index"])
                h = np.array([r["h_nm"]*1e-9 for r in group])
                primitive = PchipInterpolator(h, [r["force_n"] for r in group]).antiderivative()
                edl = (primitive(h[-1])-primitive(dense_h))/ref.thermal_energy_j
                for ah in HAMAKERS:
                    vdw = sphere_vdw_j(ref.radius_m, dense_h, ah)/ref.thermal_energy_j
                    total = edl+vdw
                    key = (strength, psi, mode, eps, ah)
                    bases[key] = total
                    j = int(np.argmax(total))
                    info = {"ionic_strength_m": strength, "isolated_potential_mv": psi,
                            "regulation": mode, "particle_permittivity": eps, "hamaker_j": ah}
                    maxima.append({**info, "positive_maximum_kbt": max(0, float(total[j])),
                                   "peak_h_nm": float(dense_h[j]*1e9)})
                    # Keep enough points in exported curves without duplicating the dense scan.
                    for t in range(0, len(dense_h), 10):
                        curves.append([strength, psi, mode, eps, ah, dense_h[t]*1e9,
                                       edl[t], vdw[t], total[t]])
                    if eps == 150:
                        for kind, scales in [("solvation", [.2, .3, .5, 1]), ("corona", [.5, 1, 2, 3])]:
                            for scale_nm in scales:
                                unit = (solvation_kbt(dense_h, 1, scale_nm*1e-9) if kind == "solvation"
                                        else corona_kbt(ref.radius_m, scale_nm*1e-9, dense_h, 1e6, ref.thermal_energy_j))
                                mask = unit > 1e-100
                                amplitudes = np.full(len(dense_h), np.inf)
                                amplitudes[mask] = np.maximum(0, 20-total[mask])/unit[mask]
                                t = int(np.argmin(amplitudes))
                                thresholds.append({**info, "mechanism": kind, "range_nm": scale_nm,
                                                   "required_amplitude": float(amplitudes[t]),
                                                   "amplitude_unit": "kBT at h=0" if kind == "solvation" else "MPa effective energy density",
                                                   "target_kbt": 20, "peak_h_nm": float(dense_h[t]*1e9),
                                                   "minimum_distance_boundary": t == 0})
    write_csv(OUT/"curves.csv", ["ionic_strength_m", "isolated_potential_mv", "regulation",
                                "particle_permittivity", "hamaker_j", "h_nm", "edl_kbt", "vdw_kbt", "total_kbt"], curves)
    write_dict_csv(OUT/"peaks.csv", maxima)
    write_dict_csv(OUT/"required_repulsion.csv", thresholds)
    write_dict_csv(OUT/"isolated_charges.csv", charges)
    examples = []
    for key, base in bases.items():
        strength, psi, mode, eps, ah = key
        if eps != 150:
            continue
        for layer_nm, penalty_mpa in [(2, 1.5), (3, .5)]:
            extra = corona_kbt(ref.radius_m, layer_nm*1e-9, dense_h, penalty_mpa*1e6, ref.thermal_energy_j)
            total = base+extra
            j = int(np.argmax(total))
            examples.append({"ionic_strength_m": strength, "isolated_potential_mv": psi,
                             "regulation": mode, "particle_permittivity": eps, "hamaker_j": ah,
                             "assumed_layer_nm": layer_nm, "assumed_penalty_mpa": penalty_mpa,
                             "positive_maximum_kbt": max(0, float(total[j])),
                             "peak_h_nm": float(dense_h[j]*1e9), "minimum_distance_boundary": j == 0})
    write_dict_csv(OUT/"conditional_corona_examples.csv", examples)
    validations = validate(ref)
    data = {"inputs": config, "charge_regulation": "q=q_inf-C_in*(u-u_inf); C_in=C_diff*(1-p)/p; p=0 CP, p=1 local fixed free charge",
            "particle_interior": "Laplace equation represented by harmonic Legendre normal-derivative map",
            "hamakers_j": HAMAKERS, "hamaker_status": "Sensitivity assumptions, not measured in ethanol",
            "short_range_status": "Unmeasured phenomenological pair-energy sensitivity; inverse thresholds, not material predictions",
            "peaks": maxima, "thresholds": thresholds, "conditional_corona_examples": examples,
            "validation": validations,
            "isolated_charges": charges}
    (OUT/"summary.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    draw(ref, dense_h, bases, thresholds)
    for strength in STRENGTHS:
        for mode in (0, .5, 1):
            r = next(x for x in maxima if x["ionic_strength_m"] == strength and x["isolated_potential_mv"] == 34
                     and x["regulation"] == mode and x["particle_permittivity"] == 150 and x["hamaker_j"] == 8e-20)
            print("Bare peak:", r, flush=True)
    print("Extended study complete", flush=True)


def validate(ref):
    results = []
    for strength, mode in [(1e-6, 1), (1e-3, .5), (1e-2, 1)]:
        p = replace(ref, ionic_strength_mol_l=strength)
        calibration = isolated_regulation_data(p)
        cases = []
        for resolution, padding in [(20, 12), (32, 12), (48, 12), (32, 16)]:
            g = SphereGeometry(p.radius_m, p.debye_length_m, 1e-9, resolution=resolution, padding=padding, gap_points=8)
            s = RegulatedGeometry(g).solve(p, regulation=mode, calibration=calibration)
            cases.append({"resolution": resolution, "padding": padding, "force_n": s.force_n,
                          "surface_max_mv": s.maximum_surface_potential_v*1e3, "residual": s.scaled_residual})
        results.append({"ionic_strength_m": strength, "regulation": mode, "isolated_potential_mv": 34,
                        "particle_permittivity": 150, "h_nm": 1, "cases": cases})
    return results


def draw(ref, h, bases, thresholds):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for strength, ax in zip(STRENGTHS, axes):
        for mode, label in [(0, "Constant potential"), (.5, "Regulation p=0.5"), (1, "Fixed free charge")]:
            ax.plot(h*1e9, bases[(strength, 34, mode, 150, 8e-20)], label=label)
        ax.set(xscale="log", xlim=(.3, 100), ylim=(-6, 5), xlabel="Core surface gap (nm)",
               ylabel="Energy (kBT)", title=f"Free ionic strength {strength:g} M")
        ax.axhline(0, color="gray", linewidth=.7)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("10 nm BaTiO3 / ethanol: local charge regulation + dielectric sphere interiors\nIsolated potential 34 mV, assumed core permittivity 150, A=8e-20 J")
    fig.savefig(OUT/"regulation_curves.png", dpi=180)
    plt.close(fig)

    # Representative unfavorable condition from the tested bare-particle set.
    base = bases[(1e-2, 34, 0, 150, 12e-20)]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(h*1e9, base, label="Bare CP model")
    examples = [("Corona L=2 nm, w=1.5 MPa", corona_kbt(ref.radius_m, 2e-9, h, 1.5e6, ref.thermal_energy_j)),
                ("Corona L=3 nm, w=0.5 MPa", corona_kbt(ref.radius_m, 3e-9, h, .5e6, ref.thermal_energy_j)),
                ("Assumed solvation B=100 kBT, lambda=0.5 nm", solvation_kbt(h, 100, .5e-9))]
    for label, extra in examples:
        ax.plot(h*1e9, base+extra, label=label)
    ax.axhline(20, color="gray", linestyle="--", label="20 kBT screening target")
    ax.set(xscale="log", xlim=(.3, 20), ylim=(-20, 70), xlabel="Core surface gap (nm)",
           ylabel="Energy (kBT)", title="Conditional extra repulsion: NOT measured material parameters\n34 mV, 10 mM, A=12e-20 J, CP baseline")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(OUT/"short_range_examples.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for mode, label in [(0, "CP"), (.5, "p=0.5"), (1, "Fixed free charge")]:
        subset = [x for x in thresholds if x["mechanism"] == "corona" and x["isolated_potential_mv"] == 34
                  and x["ionic_strength_m"] == 1e-3 and x["hamaker_j"] == 12e-20 and x["regulation"] == mode]
        ax.plot([x["range_nm"] for x in subset], [x["required_amplitude"] for x in subset], marker="o", label=label)
    ax.set(yscale="log", xlabel="Assumed corona thickness per particle (nm)",
           ylabel="Required effective penalty w (MPa)", title="Inverse design for a 20 kBT maximum\n10 nm, 34 mV, 1 mM, A=12e-20 J; model assumptions")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(OUT/"required_corona_penalty.png", dpi=180)
    plt.close(fig)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def write_dict_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
