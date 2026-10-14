"""Advanced level without QPU access: the 100-qubit kicked-Ising experiment on a noise model
built from the calibration of the chosen 100-qubit line (FakeFez = IBM Heron r2 snapshot).

  part 'z0'      full 100-qubit noisy stabiliser simulation, theta_h = 0 (exact answer M_z = 1)
  part 'stab'    full 100-qubit noisy stabiliser simulation, theta_h = pi/2 Clifford stabilisers (exact +1)
  part 'windows' theta_h = pi/8, pi/4, 3pi/8: exact noisy bulk <Z> on light-cone windows at 5 positions
Each part: raw / TREX / ZNE (CZ folding lambda = 1, 3, 5).  Output: results/emu_<part>.json
usage: python scripts/emulate_advanced.py <part>
"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import ising_lab as il
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_aer import AerSimulator

PART = sys.argv[1]
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
b = FakeFez(); N = 100
line, cost = il.best_line(b.target, N, cz_per_bond=8, sx_per_qubit=16)
FACTORS = (1, 3, 5)
out = {"backend": b.name, "line": line, "factors": FACTORS}
rng = np.random.default_rng(7)
K = 4
masks = [rng.integers(0, 2, N) for _ in range(K)]

if PART in ("z0", "stab"):
    nm = il.compact_noise_model(b, line)
    sim = AerSimulator(method="stabilizer", noise_model=nm)
    cal = []
    for m in masks:                               # TREX calibration: |0..0> through the twirled readout
        qc = il.kicked_ising_clifford_native(N, 0, 0.0, mask=m)
        cal.append(il.z_from_counts(sim.run(qc, shots=1000, seed_simulator=int(rng.integers(1e9))).result().get_counts(), N, m)[0])
    f = np.mean(cal, axis=0); out["readout_factor"] = f.tolist()
    log(f"readout factors: mean {f.mean():.4f}")

def spins(counts, mask=None):
    keys = list(counts); w = np.array([counts[k] for k in keys])
    bits = np.array([[int(k[N - 1 - i]) for i in range(N)] for k in keys], np.int8)
    if mask is not None:
        bits ^= np.asarray(mask, np.int8)
    return 1 - 2 * bits, w

if PART == "z0":
    SHOTS = 400; out["t"] = [2, 4, 6, 8, 12, 16, 20]; out["raw"] = []; out["trex"] = {str(x): [] for x in FACTORS}
    for t in out["t"]:
        base = il.kicked_ising_clifford_native(N, t, 0.0)
        s, w = spins(sim.run(base, shots=SHOTS, seed_simulator=int(rng.integers(1e9))).result().get_counts())
        out["raw"].append(float((w @ s).mean() / w.sum()))
        for fac in FACTORS:
            vals = []
            for m in masks:
                qc = il.kicked_ising_clifford_native(N, t, 0.0, mask=m)
                qc = il.fold_cz_logical(qc, fac) if fac > 1 else qc
                s, w = spins(sim.run(qc, shots=SHOTS // K, seed_simulator=int(rng.integers(1e9))).result().get_counts(), m)
                vals.append(w @ s / w.sum())
            out["trex"][str(fac)].append(float((np.mean(vals, axis=0) / f).mean()))
        log(f"z0 t={t}: raw {out['raw'][-1]:.4f}  trex λ=1,3,5 " + " ".join(f"{out['trex'][str(x)][-1]:.4f}" for x in FACTORS))
        (ROOT / "results" / "emu_z0.json").write_text(json.dumps(out, indent=1))

if PART == "stab":
    SHOTS = 400; out["t"] = [2, 4, 6, 8, 12, 16, 20]; centers = [N // 4, 3 * N // 4]; out["centers"] = centers
    out["weights"] = []; out["raw"] = []; out["trex"] = {str(x): [] for x in FACTORS}
    for t in out["t"]:
        ops, signs, supps = {}, [], []
        for c in centers:
            S = il.clifford_stabilizer(N, t, c); lab = S.paulis[0].to_label()[::-1]
            sup = [q for q, p in enumerate(lab) if p != "I"]; supps.append(sup); signs.append(S.coeffs[0].real)
            ops.update({q: lab[q] for q in sup})
        out["weights"].append([len(sp) for sp in supps])
        def est(counts, mask, rescale):
            s, w = spins(counts, mask)
            return [sg * float(w @ np.prod(s[:, sp], axis=1) / w.sum()) / (np.prod(f[sp]) if rescale else 1)
                    for sg, sp in zip(signs, supps)]
        base = il.kicked_ising_clifford_native(N, t, np.pi / 2, basis_ops=ops)
        out["raw"].append(est(sim.run(base, shots=SHOTS, seed_simulator=int(rng.integers(1e9))).result().get_counts(), None, False))
        for fac in FACTORS:
            vals = []
            for m in masks:
                qc = il.kicked_ising_clifford_native(N, t, np.pi / 2, basis_ops=ops, mask=m)
                qc = il.fold_cz_logical(qc, fac) if fac > 1 else qc
                vals.append(est(sim.run(qc, shots=SHOTS // K, seed_simulator=int(rng.integers(1e9))).result().get_counts(), m, True))
            out["trex"][str(fac)].append(np.mean(vals, axis=0).tolist())
        log(f"stab t={t} w={out['weights'][-1]}: raw {np.round(out['raw'][-1],3)} trex λ=1 {np.round(out['trex']['1'][-1],3)} λ=5 {np.round(out['trex']['5'][-1],3)}")
        (ROOT / "results" / "emu_stab.json").write_text(json.dumps(out, indent=1))

if PART == "windows":
    thetas = [np.pi / 8, np.pi / 4, 3 * np.pi / 8]; centers = [15, 32, 50, 67, 84]
    out.update({"thetas": thetas, "centers": centers, "t": list(range(1, 8)), "raw": [], "trex": {str(x): [] for x in FACTORS}})
    for t in out["t"]:
        raw = [il.noisy_bulk_z(b, line, t, thetas, center=c, trajectories=200, readout=True) for c in centers]
        out["raw"].append(np.mean(raw, axis=0).tolist())
        for fac in FACTORS:
            v = [il.noisy_bulk_z(b, line, t, thetas, center=c, trajectories=200, readout=False, factor=fac) for c in centers]
            out["trex"][str(fac)].append(np.mean(v, axis=0).tolist())
        log(f"windows t={t}: raw {np.round(out['raw'][-1],3)}  λ=1 {np.round(out['trex']['1'][-1],3)}")
        (ROOT / "results" / "emu_windows.json").write_text(json.dumps(out, indent=1))
log("done")
