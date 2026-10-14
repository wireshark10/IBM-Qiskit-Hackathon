"""Pre-compute the Intermediate-level noisy simulations (cached in results/).

The notebook loads results/intermediate_sim.npz, results/tradeoff_sim.json and
results/noise_model_check.json, or re-runs this script when RECOMPUTE=True
(~10-15 min on a laptop).   usage: python scripts/intermediate_sim.py [aer_threads]
"""
import sys, time, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ising_lab import (best_line, tfim_trotter_circuit, physical_line, CompactSim, mitigation_ladder,
                       measured_variant, z_from_counts)
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.transpiler import generate_preset_pass_manager

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

b = FakeFez(); n = 12; dt = 0.25; STEPS = list(range(1, 9))
path, cost = best_line(b.target, n, cz_per_bond=16, sx_per_qubit=30)
pm = generate_preset_pass_manager(optimization_level=2, backend=b, initial_layout=path)
bodies = [pm.run(tfim_trotter_circuit(n, dt, s, "suzuki")) for s in STEPS]
phys = physical_line(bodies[0], n)
sim = CompactSim(b, phys, threads=THREADS)
out = {"steps": np.array(STEPS), "dt": dt, "path": np.array(phys), "line_cost": cost}
for basis, seed in (("Z", 11), ("X", 12)):
    r = mitigation_ladder(sim, bodies, n, basis, K=8, shots=2000, target=b.target, seed=seed)
    for k, v in r.items():
        out[f"{basis}_{k}"] = v
    log(f"ladder {basis}: t=2 raw {r['raw'][-1].mean():.3f} trex {r['trex'][-1].mean():.3f} zne {r['zne_exp'][-1].mean():.3f}")
    np.savez_compressed(ROOT / "results" / "intermediate_sim.npz", **out)

# Pauli twirling leaves a Pauli channel unchanged: demonstrate once (t = 2, Z basis)
r = mitigation_ladder(sim, bodies[-1:], n, "Z", K=8, shots=4000, target=b.target, seed=21,
                      twirl_demo=True, factors=(1,))
out["twirl_demo_trex"] = r["trex"][0]; out["twirl_demo_twirl"] = r["twirl"][0]
np.savez_compressed(ROOT / "results" / "intermediate_sim.npz", **out)
log(f"twirl demo: trex {r['trex'][0].mean():.4f}  twirled {r['twirl'][0].mean():.4f}")

# Trotter-step vs noise trade-off at fixed final time t = 2
tradeoff = []
for ns in (2, 3, 4, 5, 6, 8, 10, 12):
    body = pm.run(tfim_trotter_circuit(n, 2 / ns, ns, "suzuki"))
    row = {"ns": ns, "dt": 2 / ns, "cz": int(body.count_ops().get("cz", 0)),
           "depth_2q": int(body.depth(lambda i: i.operation.num_qubits == 2))}
    for basis in ("Z", "X"):
        r = mitigation_ladder(sim, [body], n, basis, K=4, shots=1000, target=b.target, seed=100 + ns)
        for k in ("raw", "trex", "zne_exp", "zne_lin"):
            row[f"{basis}_{k}"] = float(r[k][0].mean())
    tradeoff.append(row); log(f"tradeoff ns={ns}")
    (ROOT / "results" / "tradeoff_sim.json").write_text(json.dumps(tradeoff, indent=1))

# cross-check the compact Pauli model against the full FakeFez noise model (raw values)
from qiskit_aer import AerSimulator
full = AerSimulator.from_backend(b, max_parallel_threads=THREADS)
check = []
for s in (2, 4, 8):
    c = measured_variant(bodies[s - 1], phys, "Z", np.zeros(n, int))
    zf = z_from_counts(full.run(c, shots=1000, seed_simulator=5).result().get_counts(), n)[0]
    zc = z_from_counts(sim.run(c, shots=4000, seed_simulator=5).result().get_counts(), n)[0]
    check.append({"steps": s, "t": s * dt, "full_fakefez_mz": float(zf.mean()), "compact_mz": float(zc.mean())})
    log(f"model check steps={s}: full {zf.mean():.4f} compact {zc.mean():.4f}")
    (ROOT / "results" / "noise_model_check.json").write_text(json.dumps(check, indent=1))
log("all done")
