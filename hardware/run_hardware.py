"""Submit / fetch the IBM Quantum hardware experiments for Track 3.

Usage (from the repo root):
    python hardware/run_hardware.py dry-run      # build every PUB against FakeFez, no QPU
    python hardware/run_hardware.py submit       # pick a QPU, submit 3 jobs, save job IDs
    python hardware/run_hardware.py submit --quick   # minimal Advanced run (~1-2 QPU min) if time is short
    python hardware/run_hardware.py status
    python hardware/run_hardware.py fetch        # download results to results/hardware_*.npz

Jobs (Open Plan friendly: plain jobs, no sessions):
  A  EstimatorV2  kicked Ising, 100-qubit line, t in T_KI, theta_h in THETAS  +  Clifford checks
  B  EstimatorV2  TFIM N=12 (Intermediate circuit) for the noisy-sim vs QPU comparison
  C  SamplerV2    raw (unmitigated) kicked-Ising bitstrings + readout calibration circuits
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qiskit import QuantumCircuit                                      # noqa: E402
from qiskit.circuit import Parameter                                   # noqa: E402
from qiskit.quantum_info import SparsePauliOp                          # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager            # noqa: E402

from ising_lab import (best_line, clifford_stabilizer, kicked_ising_circuit,   # noqa: E402
                       pauli_label, tfim_trotter_circuit)

RESULTS = ROOT / "results"
JOBS_FILE = RESULTS / "hardware_jobs.json"

# ---------------------------------------------------------------- experiment plan
N_KI = 100
T_KI = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]
THETAS = [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8]
T_CLIFF = [2, 4, 6, 8, 12, 16, 20]
CENTERS = [25, 75]
N_TF, DT_TF, SCHEME_TF = 12, 0.25, "suzuki"
STEPS_TF = list(range(1, 9))                    # t = 0.25 ... 2.0
SAMPLER_THETAS = [np.pi / 4, 3 * np.pi / 8]
PREC_KI, PREC_CLIFF, PREC_TF = 0.03, 0.015, 0.02
SAMPLER_SHOTS = 2000


def estimator_options(max_time: int):
    from qiskit_ibm_runtime import EstimatorOptions
    o = EstimatorOptions()
    o.resilience.measure_mitigation = True              # TREX (twirled readout)
    o.resilience.zne_mitigation = True                  # zero-noise extrapolation
    o.resilience.zne.amplifier = "gate_folding"
    o.resilience.zne.noise_factors = (1, 3, 5)
    o.resilience.zne.extrapolator = ("exponential", "linear")
    o.twirling.enable_gates = True                      # Pauli twirling of CZ layers
    o.twirling.enable_measure = True
    o.twirling.num_randomizations = 16
    o.dynamical_decoupling.enable = True
    o.dynamical_decoupling.sequence_type = "XpXm"
    o.max_execution_time = max_time
    o.environment.job_tags = ["qff2026", "track3"]
    return o


def isa_stats(isa: QuantumCircuit, ideal_cz: int | None = None) -> dict:
    """Compiled-circuit statistics. The transpiler decomposes SWAPs into CZs, so
    routing overhead shows up as extra CZs: with ``ideal_cz`` (the CZ count of a
    perfectly mapped circuit) we report swaps = extra/3 and assert it is zero."""
    ops = dict(isa.count_ops())
    cz = int(ops.get("cz", 0) + ops.get("ecr", 0) + ops.get("cx", 0))
    st = {"ops": {k: int(v) for k, v in ops.items()},
          "depth": int(isa.depth()),
          "depth_2q": int(isa.depth(lambda i: i.operation.num_qubits == 2)),
          "two_q": cz, "ideal_two_q": ideal_cz}
    st["swaps"] = None if ideal_cz is None else (cz - ideal_cz) / 3
    return st


def build_all(backend, n_ki=N_KI, t_ki=T_KI, t_cliff=T_CLIFF, centers=None, n_tf=N_TF):
    centers = centers or [n_ki // 4, 3 * n_ki // 4]
    target = backend.target
    # line cost = -log ESP for the average circuit: ~8 CZ per bond, ~16 SX per qubit (KI);
    # 16 CZ per bond, ~30 SX per qubit (12-spin TFIM, 8 steps)
    path_ki, cost_ki = best_line(target, n_ki, cz_per_bond=8, sx_per_qubit=16)
    path_tf, cost_tf = best_line(target, n_tf, cz_per_bond=16, sx_per_qubit=30)
    if path_ki is None:
        raise RuntimeError(f"no {n_ki}-qubit line on {backend.name}")
    pm_ki = generate_preset_pass_manager(optimization_level=2, backend=backend, initial_layout=path_ki)
    pm_tf = generate_preset_pass_manager(optimization_level=2, backend=backend, initial_layout=path_tf)
    th = Parameter("theta_h")

    z_obs = [SparsePauliOp(pauli_label(n_ki, {i: "Z"})) for i in range(n_ki)]
    zz_obs = [SparsePauliOp(pauli_label(n_ki, {i: "Z", i + 1: "Z"})) for i in range(n_ki - 1)]
    pubs_a, meta_a = [], []
    theta_vals = np.array(THETAS).reshape(-1, 1, 1)          # (n_theta, 1 obs-broadcast, 1 param)
    for t in t_ki:
        isa = pm_ki.run(kicked_ising_circuit(n_ki, t, th))
        st = isa_stats(isa, ideal_cz=(t - 1) * (n_ki - 1))      # one CZ per RZZ(-pi/2), first layer dropped
        assert st["swaps"] == 0, "routing added SWAPs: layout is not a line"
        obs = [o.apply_layout(isa.layout) for o in z_obs + zz_obs]
        pubs_a.append((isa, obs, theta_vals, PREC_KI))
        meta_a.append({"kind": "ki", "t": t, "thetas": THETAS, **st})
    for t in t_cliff:
        isa = pm_ki.run(kicked_ising_circuit(n_ki, t, np.pi / 2))
        st = isa_stats(isa, ideal_cz=(t - 1) * (n_ki - 1))
        assert st["swaps"] == 0
        obs = [clifford_stabilizer(n_ki, t, c).apply_layout(isa.layout) for c in centers]
        weights = [sum(ch != "I" for ch in clifford_stabilizer(n_ki, t, c).paulis[0].to_label()) for c in centers]
        pubs_a.append((isa, obs, None, PREC_CLIFF))
        meta_a.append({"kind": "clifford", "t": t, "centers": centers, "weights": weights, **st})

    tz = [SparsePauliOp(pauli_label(n_tf, {i: "Z"})) for i in range(n_tf)]
    tx = [SparsePauliOp(pauli_label(n_tf, {i: "X"})) for i in range(n_tf)]
    pubs_b, meta_b = [], []
    for s in STEPS_TF:
        isa = pm_tf.run(tfim_trotter_circuit(n_tf, DT_TF, s, SCHEME_TF))
        st = isa_stats(isa, ideal_cz=2 * s * (n_tf - 1))          # s RZZ layers x (n-1) bonds x 2 CZ
        assert st["swaps"] == 0
        obs = [o.apply_layout(isa.layout) for o in tz + tx]
        pubs_b.append((isa, obs, None, PREC_TF))
        meta_b.append({"kind": "tfim", "steps": s, "t": s * DT_TF, **st})

    pubs_c, meta_c = [], []
    for thv in SAMPLER_THETAS:
        for t in t_ki:
            isa = pm_ki.run(kicked_ising_circuit(n_ki, t, thv, measure=True))
            pubs_c.append((isa,))
            meta_c.append({"kind": "raw", "t": t, "theta": thv, **isa_stats(isa, ideal_cz=(t - 1) * (n_ki - 1))})
    for label in ("cal0", "cal1"):
        qc = QuantumCircuit(n_ki)
        if label == "cal1":
            qc.x(range(n_ki))
        qc.measure_all()
        pubs_c.append((pm_ki.run(qc),))
        meta_c.append({"kind": label})

    layout = {"path_ki": [int(q) for q in path_ki], "cost_ki": float(cost_ki),
              "path_tf": [int(q) for q in path_tf], "cost_tf": float(cost_tf)}
    return (pubs_a, meta_a), (pubs_b, meta_b), (pubs_c, meta_c), layout


def calibration_snapshot(backend, qubits):
    t = backend.target
    g2 = "cz" if "cz" in t.operation_names else "ecr"
    snap = {"readout_error": {}, "sx_error": {}, "two_q_error": {}}
    for q in qubits:
        m = t["measure"].get((q,))
        snap["readout_error"][q] = None if m is None else m.error
        s = t["sx"].get((q,))
        snap["sx_error"][q] = None if s is None else s.error
    for a, b in zip(qubits[:-1], qubits[1:]):
        p = t[g2].get((a, b)) or t[g2].get((b, a))
        snap["two_q_error"][f"{a}-{b}"] = None if p is None else p.error
    return snap


# ---------------------------------------------------------------- commands

def dry_run():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from qiskit_ibm_runtime import EstimatorV2, SamplerV2
    b = FakeFez()
    t0 = time.time()
    (pa, ma), (pb, mb), (pc, mc), lay = build_all(b)
    print(f"built full plan in {time.time()-t0:.1f}s on {b.name}: A={len(pa)} B={len(pb)} C={len(pc)} PUBs")
    print("layout cost (-log F, line of 100):", round(lay["cost_ki"], 3))
    for m in ma[:3] + ma[-2:]:
        print("  A", {k: m[k] for k in ("kind", "t", "depth_2q", "swaps")}, "cz=", m["ops"].get("cz"))
    for m in mb[-1:]:
        print("  B", {k: m[k] for k in ("kind", "t", "depth_2q", "swaps")}, "cz=", m["ops"].get("cz"))
    # client-side validation of every PUB (shapes, parameter binding, precision)
    from qiskit.primitives.containers.estimator_pub import EstimatorPub
    from qiskit.primitives.containers.sampler_pub import SamplerPub
    shapes = sorted({EstimatorPub.coerce(p).shape for p in pa + pb})
    print("Estimator PUB shapes:", shapes, "| Sampler PUBs:", len([SamplerPub.coerce(p) for p in pc]))
    print("all CZ counts equal the ideal (no routing SWAPs):",
          all(m.get("swaps") in (0, None) for m in ma + mb + mc))


QUICK = "--quick" in sys.argv
if QUICK:
    # minimal Advanced experiment, ~1-2 QPU minutes: real 100-qubit data at shallow, medium and deep
    # circuits, a high-signal angle (0), a non-trivial angle (pi/4) and two Clifford checks
    T_KI = [2, 4, 6, 10, 20]
    THETAS = [0.0, np.pi / 8, np.pi / 4]
    T_CLIFF = [4, 8]
    STEPS_TF = [2, 4, 8]
    SAMPLER_THETAS = [np.pi / 4]


def submit():
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, SamplerV2
    service = QiskitRuntimeService()
    # Safety: only ever spend free Open Plan minutes. Refuse if the active instance is a paid plan.
    active = (service.active_account() or {}).get("instance")
    plans = {i.get("crn"): str(i.get("plan", "")).lower() for i in service.instances()}
    plan = plans.get(active, "unknown")
    print("active instance plan:", plan)
    if plan != "open" and "--allow-paid" not in sys.argv:
        raise SystemExit(f"Refusing to submit: instance plan is '{plan}', not 'open' (pass --allow-paid to override).")
    name = next((a for a in sys.argv[2:] if not a.startswith("--")), None)   # optional: force a backend name
    backend = service.backend(name) if name else service.least_busy(operational=True, simulator=False,
                                                                      min_num_qubits=N_KI + 20)
    print("backend:", backend.name, "| pending jobs:", backend.status().pending_jobs)
    for n_ki in (N_KI, 90, 80, 64):                              # fall back if no 100-qubit line exists
        try:
            (pa, ma), (pb, mb), (pc, mc), lay = build_all(backend, n_ki=n_ki, t_ki=T_KI, t_cliff=T_CLIFF)
            break
        except RuntimeError as err:
            print(err)
    print("chain length:", n_ki, "| line cost:", round(lay["cost_ki"], 3), "| path head:", lay["path_ki"][:8])
    rec = {"backend": backend.name, "submitted": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "layout": lay, "meta": {"A": ma, "B": mb, "C": mc},
           "calibration": calibration_snapshot(backend, lay["path_ki"]),
           "plan": {"N_KI": n_ki, "T_KI": T_KI, "THETAS": THETAS, "T_CLIFF": T_CLIFF, "CENTERS": [n_ki // 4, 3 * n_ki // 4],
                    "N_TF": N_TF, "DT_TF": DT_TF, "SCHEME_TF": SCHEME_TF, "STEPS_TF": STEPS_TF,
                    "SAMPLER_THETAS": SAMPLER_THETAS, "SAMPLER_SHOTS": SAMPLER_SHOTS}, "jobs": {}}
    RESULTS.mkdir(exist_ok=True)
    ja = EstimatorV2(mode=backend, options=estimator_options(480)).run(pa)
    rec["jobs"]["A"] = ja.job_id(); print("A submitted:", ja.job_id())
    JOBS_FILE.write_text(json.dumps(rec, indent=1))
    jc = SamplerV2(mode=backend).run(pc, shots=SAMPLER_SHOTS)
    rec["jobs"]["C"] = jc.job_id(); print("C submitted:", jc.job_id())
    JOBS_FILE.write_text(json.dumps(rec, indent=1))
    jb = EstimatorV2(mode=backend, options=estimator_options(240)).run(pb)
    rec["jobs"]["B"] = jb.job_id(); print("B submitted:", jb.job_id())
    JOBS_FILE.write_text(json.dumps(rec, indent=1))
    print("saved", JOBS_FILE)


def status():
    from qiskit_ibm_runtime import QiskitRuntimeService
    rec = json.loads(JOBS_FILE.read_text())
    service = QiskitRuntimeService()
    for k, jid in rec["jobs"].items():
        j = service.job(jid)
        extra = ""
        try:
            extra = f"usage={j.usage()}s" if str(j.status()) in ("DONE", "JobStatus.DONE") else ""
        except Exception:
            pass
        print(k, jid, j.status(), extra)


def _data_to_dict(data) -> dict:
    out = {}
    for key in data.keys() if hasattr(data, "keys") else []:
        v = data[key]
        try:
            out[key] = np.asarray(v)
        except Exception:
            pass
    return out


def fetch():
    from qiskit_ibm_runtime import QiskitRuntimeService
    rec = json.loads(JOBS_FILE.read_text())
    service = QiskitRuntimeService()
    for k, jid in rec["jobs"].items():
        j = service.job(jid)
        st = str(j.status())
        if "DONE" not in st:
            print(k, jid, st, "(not done)"); continue
        res = j.result()
        arrays = {}
        for i, pr in enumerate(res):
            if k == "C":
                ba = pr.data.meas
                arrays[f"p{i}_bits"] = ba.array            # (shots, ceil(n/8)) packed, big-endian bytes
                arrays[f"p{i}_nbits"] = np.array(ba.num_bits)
            else:
                for key, v in _data_to_dict(pr.data).items():
                    arrays[f"p{i}_{key}"] = v
        np.savez_compressed(RESULTS / f"hardware_{k}.npz", **arrays)
        rec.setdefault("usage_s", {})[k] = j.usage() if hasattr(j, "usage") else None
        try:
            rec.setdefault("result_metadata", {})[k] = json.loads(json.dumps(res.metadata, default=str))
        except Exception:
            pass
        print(k, jid, "saved", len(res), "pub results; usage", rec["usage_s"][k])
    JOBS_FILE.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dry-run"
    {"dry-run": dry_run, "submit": submit, "status": status, "fetch": fetch}[cmd]()
