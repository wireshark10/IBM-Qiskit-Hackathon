"""Notebook cells: analysis, limitations, credits."""
from nbcells import md, code

CELLS = []
C = CELLS.append

C(md(r"""
## 4 · Performance, limitations and what we would do next

**Where each method works, and where it stops**

| Method | Works well | Breaks down | Evidence in this notebook |
|---|---|---|---|
| Lie-Trotter | Z-basis observables from a Z-product state (secretly 2nd order) | X-basis observables: first order, ~10× more CX for 0.01 accuracy | §1.3–1.4, slopes 2.00 vs 1.02 |
| Merged Suzuki-Trotter | both observables at 1st-order CX cost | — (the error constant depends on which half-steps sit outside) | §1.3 table |
| Trotter step chosen by Trotter error alone | noiseless simulation | on hardware, over-refining adds CZ errors faster than it removes Trotter error | §2.4 trade-off |
| TREX | removes readout bias for any observable | assumes readout is the only measurement error; does nothing for gate noise | §2.3 |
| ZNE (gate folding) | moderate depth, signal well above the shot noise | once the λ=5 circuit has decayed to ≈ 0, extrapolation amplifies shot noise; exponential and linear fits disagree | §2.3, §3.6 |
| Free-fermion certification | any N and any depth for the integrable kicked transverse-field Ising chain | adding a longitudinal field $h_z$ or 2D couplings breaks integrability | §3.1 |
| Light-cone certification | any local circuit, any N | cost $2^{2t+2}$: about $t\le 11$ on a laptop | §3.1 |

**Honest limitations**
* **Classical simulability.** The 100-qubit circuits are classically easy *because* we chose an integrable 1D model. That is what makes the hardware data certifiable. The same pipeline (layout search, mitigation, trust horizon) applies unchanged to a non-integrable variant, but then only the light cone ($t\lesssim 11$), the Clifford point and internal consistency checks remain.
* **The noise model.** Our fast simulator uses a Pauli approximation built from FakeFez calibration. It ignores coherent errors and crosstalk, so it cannot show the benefit of twirling or dynamical decoupling. Both are applied on the real device.
* **ZNE by gate folding** triples or quintuples circuit depth. At the deepest circuits ($t\ge 16$: ~1500–1900 CZ) the λ=5 data sits near the noise floor. Probabilistic error amplification (PEA, as in Kim *et al.*) or PEC would be the next step, at a higher QPU-time cost than our 10-minute Open Plan budget allows.
* **Statistics.** We used ~1000 shots per circuit per noise factor on hardware, so per-site values have σ ≈ 0.03–0.1 after extrapolation. Averages over 100 sites are much tighter.

**Next steps**
* Break integrability with a small longitudinal field, and certify with light-cone windows plus Clifford-perturbation checks.
* Use PEA-based ZNE with learned layer noise. Our circuits contain only two distinct CZ layers (even and odd bonds), so learning them is cheap.
* Use fractional `RZZ` gates on Heron for the TFIM: one native gate per bond instead of two CZs, halving depth.
"""))

C(md(r"""
### Headline numbers (exported to `results/summary.json` for the README and slides)
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
## 5 · Credits and references

* **Tutorial:** adapted from the BasQ Qiskit Fall Fest 2026 Track 3 tutorial by Benjamin Tirado (text CC BY 4.0, code Apache 2.0). Challenge prompt by Angel Rodriguez, Benjamin Tirado and Miguel Angel Jimenez, as adapted by Quantica, IIIT-Delhi.
* **Software:** Qiskit 2.5, Qiskit Aer 0.17, Qiskit IBM Runtime 0.50 (EstimatorV2 and SamplerV2 with TREX, ZNE, twirling and DD), NumPy, SciPy and Matplotlib.
* **References:**
  * Y. Kim *et al.*, *Evidence for the utility of quantum computing before fault tolerance*, Nature **618**, 500 (2023)
  * N. Hatano and M. Suzuki, *Finding exponential product formulas of higher orders* (2005)
  * A. Childs *et al.*, *Theory of Trotter error with commutator scaling*, PRX **11**, 011020 (2021)
  * E. van den Berg *et al.*, *Model-free readout-error mitigation (TREX)*, PRA **105**, 032620 (2022)
  * K. Temme, S. Bravyi and J. Gambetta, *Error mitigation for short-depth quantum circuits*, PRL **119**, 180509 (2017)
  * Bertini, Kos and Prosen, *Exact correlation functions for dual-unitary lattice models* (kicked Ising), PRL **123**, 210601 (2019)
  * M. Wimmer, *Efficient numerical computation of the Pfaffian*, ACM TOMS **38**, 30 (2012)
* **AI tools:** Claude (Anthropic), used as a coding and writing assistant: drafting helper functions and notebook text, and debugging. Every result was checked numerically in this notebook (cross-validation to ~1e-14 and comparisons with exact solvers). The team chose the methods and reviewed all analysis.
"""))
