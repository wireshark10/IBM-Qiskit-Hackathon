"""Noise-model prediction of the bulk <Z> of the 100-qubit kicked-Ising runs, using the
calibration of the device and line that were actually used (falls back to FakeFez).
Writes results/noisy_emulation.json.   usage: python scripts/emulate_hw.py [tmax]"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import ising_lab as il

TMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 7
rec_path = ROOT / "results" / "hardware_jobs.json"
rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
backend = None
if rec:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService().backend(rec["backend"])
        line = rec["layout"]["path_ki"]
    except Exception as err:
        print("could not load the real backend:", err)
if backend is None:
    from qiskit_ibm_runtime.fake_provider import FakeFez
    backend = FakeFez(); line, _ = il.best_line(backend.target, 100, cz_per_bond=8, sx_per_qubit=16)
thetas = [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8]
out = {"backend": backend.name, "thetas": thetas, "t": [], "with_readout": [], "gates_only": []}
for t in range(1, TMAX + 1):
    t0 = time.time()
    out["t"].append(t)
    out["with_readout"].append(il.noisy_bulk_z(backend, line, t, thetas, trajectories=300).tolist())
    out["gates_only"].append(il.noisy_bulk_z(backend, line, t, thetas, trajectories=300, readout=False).tolist())
    print(f"t={t} done in {time.time()-t0:.1f}s", flush=True)
    (ROOT / "results" / "noisy_emulation.json").write_text(json.dumps(out, indent=1))
