"""Notebook cells: analysis, limitations, credits."""
from nbcells import md, code

CELLS = []
C = CELLS.append

C(md(r"""
## 4. What worked, what didn't, and next steps

Where each method works and where it stops:

| Method | Works well | Breaks down | Where we show it |
|---|---|---|---|
| Lie-Trotter | Z-basis observables from a Z-product state (effectively second order) | X-basis observables: first order, 12× more CX than merged Suzuki for 0.01 accuracy (1834 vs 154) | 1.3–1.4, slopes 2.00 vs 1.02 |
| Merged Suzuki-Trotter | both observables at first-order CX cost | — (the error constant depends on which half-steps are on the outside) | 1.3 table |
| Trotter step chosen from Trotter error alone | noiseless simulation | on hardware, a finer step adds CZ errors faster than it removes Trotter error | 2.4 |
| TREX | removes readout bias for any observable | assumes readout is the only measurement error; does nothing for gate noise | 2.3 |
| ZNE (gate folding) | moderate depth, signal well above shot noise | once the λ=5 circuit has decayed to ≈ 0, extrapolation amplifies shot noise and the exponential and linear fits disagree | 2.3, 3.3 |
| Free-fermion check | any N and any depth for the integrable kicked transverse-field Ising chain | a longitudinal field $h_z$ or 2D couplings break integrability | 3.1 |
| Light-cone check | any local circuit, any N | cost $2^{2t+2}$, so about $t\le 11$ on a laptop | 3.1 |

Limitations
* No real-QPU data. We could not complete IBM Quantum account verification (it required a card check) before the deadline, so the 100-qubit results come from a noise model built from IBM Heron r2 (*ibm_fez* snapshot) calibration. The model has depolarising gate errors and readout errors, but no coherent errors, crosstalk, leakage or drift. Real hardware would decay faster and the ZNE horizon would be shorter. The submission script, PUBs and options are validated against the FakeFez target and ready to run (we estimate about 4 QPU minutes).
* Classical simulability. The 100-qubit circuits are easy to simulate classically because we chose an integrable 1D model, and that is also what lets us check them exactly. The same pipeline (layout search, mitigation, trust horizon) works unchanged for a non-integrable version, but then only the light cone ($t\lesssim 11$), the Clifford point and the internal consistency checks would remain.
* The noise model. Our fast simulator uses a Pauli approximation of FakeFez's calibration. It has no coherent errors or crosstalk, so it can't show what twirling or dynamical decoupling add. Both are part of the plan for the real device.
* ZNE by gate folding makes circuits 3–5× deeper. For the deepest circuits ($t\ge 16$, about 1500–1900 CZ) the λ=5 data is close to the noise floor. Probabilistic error amplification (PEA, as in Kim et al.) or PEC would be the next step, but they need more QPU time than the 10-minute Open Plan allows.
* Statistics. The emulator's Clifford-point runs use 400 shots per circuit, split over 4 TREX masks, which matches the planned QPU budget (Job A: about 1100 shots per circuit per noise factor). A single site has σ ≈ 0.03–0.05 per noise factor and ≈ 0.07–0.1 after extrapolation. The 100-site $M_z$ has σ ≈ 0.01 after extrapolation. A single stabiliser has σ ≈ 0.06 (weight 7) to 0.17 (weight 39) per noise factor, because TREX rescaling divides by $\prod_i f_i \approx 0.3$ at weight 39. The light-cone windows average 200 noise trajectories and have no measurement shot noise.

Next steps
* Run the prepared jobs on a real Heron QPU (`hardware/run_hardware.py submit`, about 4 Open Plan minutes). The notebook picks up `results/hardware_*.npz` automatically, and comparing with the emulator would show how much noise the calibration data misses (coherent errors, crosstalk, drift).
* Break integrability with a small longitudinal field, and check it with light-cone windows and Clifford-point checks.
* Use PEA-based ZNE with learned layer noise. Our circuits only have two distinct CZ layers (even and odd bonds), so learning them would be cheap.
* Use fractional `RZZ` gates on Heron for the TFIM: one native gate per bond instead of two CZs, which halves the depth.
"""))

C(md(r"""
### Summary numbers (saved to `results/summary.json`, used for the README and slides)
"""))

C(code(r"""
summary = {}
summary["thresholds_rows"] = "".join(
    f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[3]:.4f} ({r[2]} steps)</td><td>{r[5]}</td><td>{r[7]}</td></tr>" for r in table)
summary["compile_rows"] = "".join(
    f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]:.3f}</td></tr>" for r in rows21)
summary["ladder_rows"] = "".join(
    f"<tr><td>{lab}</td><td>{lad_err[('Z', k)][1]:.3f}</td><td>{lad_err[('X', k)][1]:.3f}</td></tr>"
    for k, lab in (("raw", "raw"), ("trex", "TREX"), ("zne_lin", "TREX + ZNE (linear)"), ("zne_exp", "TREX + ZNE (exp)")))
ki_meta = [m for m in meta_A if m["kind"] == "ki"]
summary["max_cz"] = max(m["ops"].get("cz", 0) for m in ki_meta)
summary["max_depth2q"] = max(m["depth_2q"] for m in ki_meta)
summary["n_ki"] = N_KI
emu_h = json.load(open(RESULTS / "emu_horizon.json"))["emu_horizon"] if (RESULTS / "emu_horizon.json").exists() else None
if (RESULTS / "emu_horizon.json").exists():
    summary["emu_takeaway"] = json.load(open(RESULTS / "emu_horizon.json")).get("emu_takeaway", "")
if emu_h and not HAVE_HW:
    summary["trust_headline"] = f"t = {emu_h.get('0', 0)} recovered"
    summary["trust_sub"] = (f"θ_h=0 M_z within 0.05 of exact at 1881 CZ (100-qubit noisy emulation); "
                            f"stabilisers only to t = {emu_h.get('π/2 (stabiliser)', 0)} (weight ≤ 7)")
    summary["horizon_text"] = ", ".join(f"{k}: t={v}" for k, v in emu_h.items())
if HAVE_HW:
    dev = {lab: np.abs(hw[lab]["zne_z"].mean(1) - mz_exact_ki[lab]) for lab in TH}
    summary["horizon_text"] = ", ".join(f"θ_h={k}: t={v}" for k, v in horizon.items())
    best = max(horizon.values())
    summary["trust_headline"] = f"t = {best} certified"
    summary["trust_sub"] = f"mitigated M_z within max(2σ, 0.05) of exact, {N_KI} qubits"
    summary["hw_takeaway"] = ("Mean |QPU+ZNE − exact| over all depths: " +
                              ", ".join(f"θ_h={lab}: {dev[lab].mean():.3f}" for lab in TH) + ".")
    summary["hw_jobs"] = f"{jobs_meta['backend']} · jobs {', '.join(jobs_meta['jobs'].values())}"
    use = jobs_meta.get("usage_s", {})
    names = {"A": "A · kicked Ising + Clifford", "B": "B · 12-spin TFIM", "C": "C · raw bitstrings"}
    prim = {"A": "EstimatorV2 (TREX, twirl, DD, ZNE)", "B": "EstimatorV2 (TREX, twirl, DD, ZNE)", "C": "SamplerV2 (raw)"}
    summary["usage_rows"] = "".join(f"<tr><td>{names[k]}</td><td>{prim[k]}</td><td>{use.get(k, '—')}</td></tr>" for k in ("A", "B", "C"))
json.dump(summary, open(RESULTS / "summary.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_rows")}, indent=1, default=str))
"""))

C(md(r"""
## 5. Credits and references

* Tutorial: adapted from the BasQ Qiskit Fall Fest 2026 Track 3 tutorial by Benjamin Tirado (text CC BY 4.0, code Apache 2.0). Challenge prompt by Angel Rodriguez, Benjamin Tirado and Miguel Angel Jimenez, as adapted by Quantica, IIIT-Delhi.
* Software: Qiskit 2.5, Qiskit Aer 0.17, Qiskit IBM Runtime 0.50 (EstimatorV2 and SamplerV2 with TREX, ZNE, twirling and DD), NumPy, SciPy and Matplotlib.
* References:
  * Y. Kim *et al.*, *Evidence for the utility of quantum computing before fault tolerance*, Nature **618**, 500 (2023)
  * N. Hatano and M. Suzuki, *Finding exponential product formulas of higher orders* (2005)
  * A. Childs *et al.*, *Theory of Trotter error with commutator scaling*, PRX **11**, 011020 (2021)
  * E. van den Berg *et al.*, *Model-free readout-error mitigation (TREX)*, PRA **105**, 032620 (2022)
  * K. Temme, S. Bravyi and J. Gambetta, *Error mitigation for short-depth quantum circuits*, PRL **119**, 180509 (2017)
  * Bertini, Kos and Prosen, *Exact correlation functions for dual-unitary lattice models* (kicked Ising), PRL **123**, 210601 (2019)
  * M. Wimmer, *Efficient numerical computation of the Pfaffian*, ACM TOMS **38**, 30 (2012)
* AI tools: we used Claude (Anthropic) as a coding and writing assistant, for drafting helper functions and notebook text and for debugging. Every result was checked numerically in this notebook (cross-validation to about 1e-14 and comparison with exact solvers). The team chose the methods and reviewed all of the analysis.
"""))
