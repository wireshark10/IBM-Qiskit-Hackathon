"""Notebook cells: Intermediate level."""
from nbcells import md, code

CELLS = []
C = CELLS.append

C(md(r"""
## 2 · 🟠 Intermediate: recovering the physics that noise hides (12 spins)

Same model, now with $N=12$ spins, $J=h=1$, $0\le t\le 2$, and realistic noise. The plan follows what a hardware run actually needs:

1. **Cut the circuit first** (§2.1). Every CZ we remove is error we never have to mitigate.
2. **Model the noise** from FakeFez (IBM Heron r2, 156 qubits) calibration data (§2.2).
3. **Mitigate**, one rung at a time, with code we wrote ourselves (§2.3): readout twirling (TREX), Pauli twirling, and zero-noise extrapolation (ZNE) by CZ folding.
4. **Choose the Trotter step under noise** (§2.4). A finer step lowers Trotter error but adds CZ errors, so refining stops paying off.
5. **Prepare the same circuit for a real IBM QPU** (§2.5): the job is built and validated; we had no QPU access before the deadline.

We use merged 2nd-order Suzuki-Trotter with $\Delta t = 0.25$ (8 steps to $t=2$). For 12 spins its Trotter error is 0.024 in $M_z$ and 0.018 in $M_x$. §2.4 shows that on noisy hardware the finer $\Delta t \approx 0.15$ that the 0.01 tolerance would need buys nothing: the error is noise-limited for $\Delta t \lesssim 0.4$.
"""))

C(code(r"""
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeFez

N12, DT12, STEPS12 = 12, 0.25, list(range(1, 9))
fez = FakeFez()
t12 = np.array([0] + [s * DT12 for s in STEPS12])

# exact reference (dense, 4096-dim) and the noiseless Trotter circuit the hardware will run
eng12 = il.TFIMEngine(N12)
H12 = il.tfim_hamiltonian(N12).to_matrix()
w12, V12 = np.linalg.eigh(H12); c12 = V12.conj().T[:, 0]
exact12 = [V12 @ (np.exp(-1j * w12 * t) * c12) for t in t12]
mz12_exact = np.array([eng12.mz(p) for p in exact12]); mx12_exact = np.array([eng12.mx(p) for p in exact12])
tro12 = eng12.trajectory(DT12, STEPS12[-1], "suzuki")
mz12_trot = np.array([eng12.mz(p) for p in tro12]); mx12_trot = np.array([eng12.mx(p) for p in tro12])
print("Trotter error of the noiseless circuit:  Mz %.4f   Mx %.4f" %
      (np.abs(mz12_trot - mz12_exact).max(), np.abs(mx12_trot - mx12_exact).max()))
"""))

C(md(r"""
### 2.1 Cutting depth before mitigating: circuit structure and layout

Four ways to compile the same $t=2$ evolution for FakeFez. We score each by **CZ count**, **two-qubit depth**, **SWAPs added by routing**, and the **estimated success probability** $\text{ESP}=\prod_{\text{gates}}(1-\epsilon_g)\prod_{\text{measured qubits}}(1-\epsilon_{\text{ro}})$, computed from FakeFez's calibration. ESP is the standard first-order figure of merit for comparing compiled circuits.

* **(a) Qiskit `SuzukiTrotter`, default layout.** `PauliEvolutionGate`, transpiled at optimisation level 2.
* **(b) Our merged brickwork circuit, default layout.** Half-steps merged, trivial first layer dropped. Even bonds, then odd bonds, so the two-qubit depth per step is 2·2 = 4 CZ layers whatever N is.
* **(c) Our circuit on an error-aware line.** `il.best_line` runs a beam search over the heavy-hex coupling map for the 12-qubit path with the lowest summed CZ, readout and SX infidelity.
* **(d) Our circuit on a careless layout.** 12 scattered qubits, so the router has to insert SWAPs.
"""))

C(code(r"""
def esp(isa, backend, n_meas):
    t = backend.target
    logf = 0.0
    for inst in isa.data:
        name = inst.operation.name
        if name in ("barrier", "measure", "delay") or name not in t.operation_names:
            continue
        qs = tuple(isa.find_bit(q).index for q in inst.qubits)
        props = t[name].get(qs)
        if props is not None and props.error is not None:
            logf += math.log(max(1e-12, 1 - props.error))
    phys = list(isa.layout.final_index_layout(filter_ancillas=True))[:n_meas]
    for q in phys:
        logf += math.log(1 - (t["measure"][(q,)].error or 0))
    return math.exp(logf)

def summarize(label, isa, cz_perfect):
    # the transpiler decomposes SWAPs into 3 CZs, so routing overhead = extra CZs / 3
    cz = int(isa.count_ops().get("cz", 0))
    return (label, cz, int(isa.depth(lambda i: i.operation.num_qubits == 2)),
            round((cz - cz_perfect) / 3), esp(isa, fez, N12))

steps_T = STEPS12[-1]
qiskit_circ = qiskit_trotter_circuit(il.tfim_hamiltonian(N12), steps_T, DT12, order=2)
ours = il.tfim_trotter_circuit(N12, DT12, steps_T, "suzuki")
# line cost = -log(ESP) of this circuit: 16 CZ per bond, ~30 SX per qubit
line12, line12_cost = il.best_line(fez.target, N12, cz_per_bond=16, sx_per_qubit=30)
rng = np.random.default_rng(3)
scattered = [int(q) for q in rng.choice(fez.num_qubits, N12, replace=False)]

pm_default = generate_preset_pass_manager(optimization_level=2, backend=fez, seed_transpiler=1)
pm_line = generate_preset_pass_manager(optimization_level=2, backend=fez, initial_layout=line12)
pm_bad = generate_preset_pass_manager(optimization_level=2, backend=fez, initial_layout=scattered, seed_transpiler=1)
cz_q = int(pm_line.run(qiskit_circ).count_ops().get("cz", 0))      # CZ count with no routing at all
cz_o = int(pm_line.run(ours).count_ops().get("cz", 0))
rows21 = [summarize("(a) Qiskit SuzukiTrotter, default layout", pm_default.run(qiskit_circ), cz_q),
          summarize("(b) merged brickwork, default layout", pm_default.run(ours), cz_o),
          summarize("(c) merged brickwork, error-aware line", pm_line.run(ours), cz_o),
          summarize("(d) merged brickwork, scattered qubits", pm_bad.run(ours), cz_o)]
print(f"{'circuit':44s} {'CZ':>5s} {'2q depth':>8s} {'SWAPs':>5s} {'ESP':>7s}")
for r in rows21:
    print(f"{r[0]:44s} {r[1]:5d} {r[2]:8d} {r[3]:5d} {r[4]:7.3f}")
print("\nerror-aware line on FakeFez:", line12)
"""))

C(md(r"""
**What the table says.** Merging and brickwork ordering (a → b) cut the CZ count by 45% (322 → 176) and the two-qubit depth 10× (322 → 32): Qiskit's default synthesis applies the ZZ terms one after another, while the brickwork applies all even bonds, then all odd bonds, in parallel. The scattered layout (d) shows what routing costs: the router's SWAPs (3 CZ each, counted as extra CZs because the transpiler decomposes them) multiply the gate count and ESP collapses. The VF2 layout pass in Qiskit's default compilation already finds a perfect line for a 12-qubit chain, so (b) and (c) are close here. Our line search optimises the same −log ESP directly (weights = 16 CZ per bond, ~30 SX per qubit). For the 100-qubit chain in §3 a default layout is no longer reliable, which is why we search for the line ourselves using calibration data.

### 2.2 The noise model, and a check that it is faithful

Simulating the full 156-qubit FakeFez noise model costs ~40 s per circuit, which is too slow for hundreds of mitigation circuits. So we build a **compact noise model** (`il.compact_noise_model`) for just the 12 chosen qubits, from FakeFez's calibration data:
* depolarising CZ, SX and X errors whose strength matches each gate's reported average error ($\lambda = r\,d/(d-1)$);
* each qubit's own **asymmetric** readout error.

Pauli twirling turns hardware noise into this kind of Pauli channel anyway. Below we compare it with the complete `AerSimulator.from_backend(FakeFez())` model, which adds thermal relaxation, on the raw $M_z$ at three depths.
"""))

C(code(r"""
if RECOMPUTE:
    import subprocess, sys
    subprocess.run([sys.executable, "scripts/intermediate_sim.py"], check=True)   # ~10-15 min

sim12 = dict(np.load(RESULTS / "intermediate_sim.npz"))
tradeoff = json.load(open(RESULTS / "tradeoff_sim.json"))
model_check = json.load(open(RESULTS / "noise_model_check.json")) if (RESULTS / "noise_model_check.json").exists() else []
print("simulated line:", sim12["path"].tolist())
print(f"{'t':>5s} {'full FakeFez model':>19s} {'compact model':>14s}   (raw M_z, 1000 / 4000 shots)")
for r in model_check:
    print(f"{r['t']:5.2f} {r['full_fakefez_mz']:19.4f} {r['compact_mz']:14.4f}")
"""))

C(md(r"""
### 2.3 Error mitigation, one rung at a time

Each rung is implemented in `ising_lab.py`:

| Rung | What it does | Why it works |
|---|---|---|
| raw | measure, average $\langle Z_i\rangle$ or $\langle X_i\rangle$ (X basis via $H=R_Z S_X R_Z$) | — |
| **TREX** | random X before each readout, undone classically (K = 8 masks); divide by the same twirled readout of $\lvert0\cdots0\rangle$ | twirling makes readout noise a pure scale factor $f_i$, which the calibration measures |
| **Pauli twirling** | random Paulis around every CZ, cancelled after it | turns coherent CZ errors into stochastic Pauli noise that ZNE can extrapolate |
| **ZNE** | fold every CZ into CZ·CZ·CZ (λ=3) or 5 CZs (λ=5); extrapolate λ→0 (linear, exponential, Richardson) | for Pauli noise, ⟨O⟩(λ) decays smoothly with λ |

One honest observation. In a Pauli noise model, twirling the CZ gates changes nothing on average, because the channel is already Pauli-symmetric. We verify that once below. Twirling earns its keep on real hardware, where coherent errors (ZZ crosstalk, miscalibrated phases) are present. There we let IBM's Estimator twirl, together with dynamical decoupling, which suppresses errors on idle qubits.
"""))

C(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
lad_err = {}
for ax, basis, ex, tr, name in ((axes[0], "Z", mz12_exact, mz12_trot, "M_z"), (axes[1], "X", mx12_exact, mx12_trot, "M_x")):
    ax.plot(t12, ex, color=COL["exact"], lw=2, label="exact")
    ax.plot(t12, tr, ":", color="grey", lw=1.5, label="noiseless Trotter circuit")
    for key, lab, c, mk in (("raw", "noisy, raw", COL["raw"], "o"), ("trex", "+ TREX", COL["trex"], "s"),
                            ("zne_lin", "+ ZNE (linear)", "#e377c2", "v"), ("zne_exp", "+ ZNE (exponential)", COL["zne"], "D")):
        y = np.concatenate([[1.0 if basis == "Z" else 0.0], sim12[f"{basis}_{key}"].mean(axis=1)])
        ax.plot(t12, y, mk + "-", ms=4, color=c, label=lab)
        lad_err[(basis, key)] = (np.abs(y[1:] - tr[1:]).mean(), np.abs(y[1:] - ex[1:]).mean())
    ax.set_xlabel("t"); ax.set_ylabel(f"${name}(t)$"); ax.set_title(f"12-spin TFIM on FakeFez noise: ${name}$")
axes[0].legend(fontsize=8)
plt.tight_layout(); savefig("i1_mitigation_ladder"); plt.show()

print(f"{'rung':22s} {'mean|err| vs noiseless circuit':>32s} {'mean|err| vs exact':>20s}")
for basis in ("Z", "X"):
    for key in ("raw", "trex", "zne_lin", "zne_exp", "zne_rich"):
        if (basis, key) not in lad_err:
            y = sim12[f"{basis}_{key}"].mean(axis=1)
            tr, ex = (mz12_trot, mz12_exact) if basis == "Z" else (mx12_trot, mx12_exact)
            lad_err[(basis, key)] = (np.abs(y - tr[1:]).mean(), np.abs(y - ex[1:]).mean())
        a, b = lad_err[(basis, key)]
        print(f"M{basis.lower()}  {key:18s} {a:32.4f} {b:20.4f}")
print("\nreadout calibration factors f_i:", np.round(sim12["Z_cal"], 3))
print("Pauli-twirling check at t=2 (Pauli noise):  TREX only %.4f   TREX + twirled CZs %.4f"
      % (sim12["twirl_demo_trex"].mean(), sim12["twirl_demo_twirl"].mean()))
"""))

C(md(r"""
**Reading the ladder.**
* **TREX on its own can look slightly *worse* than raw.** FakeFez readout is asymmetric, with $P(0|1) > P(1|0)$, so raw readout adds a positive bias $b=P(0|1)-P(1|0)$ to every $\langle Z\rangle$. Here that bias happens to offset part of the gate-noise damping. TREX removes the bias, as it should, and leaves the pure gate-noise attenuation, which ZNE then removes.
* **The order matters.** A readout bias does not scale with the folding factor λ, so ZNE on biased data extrapolates to the wrong point. TREX first, ZNE second, is the correct order.
* **Result.** With TREX + ZNE (exponential or Richardson), the $M_x$ error relative to the *noiseless circuit* drops from 0.040 to ≈0.013, about 3×. The remaining 0.013 is residual noise plus shot noise: σ ≈ 0.01–0.015 after extrapolation with 2000/1000 shots. Against the *exact* dynamics the mitigated error is 0.018–0.019, because the Trotter error of the circuit itself (≤ 0.018 for $M_x$) adds on top. Mitigation cannot remove Trotter error by design: it can only recover the noiseless circuit.
"""))

C(md(r"""
### 2.4 Choosing the Trotter step when the hardware is noisy

The Beginner analysis asked for the step that keeps *Trotter* error below 0.01. On a noisy device that is the wrong target. The total error is roughly

$$\varepsilon_{\text{total}}(\Delta t) \approx \underbrace{c\,\Delta t^2}_{\text{Trotter}} + \underbrace{1-e^{-\gamma\,n_{CZ}(\Delta t)}}_{\text{noise},\ n_{CZ}\propto 1/\Delta t},$$

so refining the step stops paying off once the noise term dominates. Below we sweep $\Delta t = 2/n$ at the final time $t=2$: noiseless Trotter error, raw noisy error, and ZNE-mitigated error.
"""))

C(code(r"""
def exact_at(t):
    return V12 @ (np.exp(-1j * w12 * t) * c12)
ex2 = exact_at(2.0); mz_ex2, mx_ex2 = eng12.mz(ex2), eng12.mx(ex2)
ns = np.array([r["ns"] for r in tradeoff]); dts = 2 / ns
trot_err = []
for n_ in ns:
    psi = eng12.trajectory(2 / n_, int(n_), "suzuki")[-1]
    trot_err.append((abs(eng12.mz(psi) - mz_ex2), abs(eng12.mx(psi) - mx_ex2)))
trot_err = np.array(trot_err)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
for ax, b, j, exv, name in ((axes[0], "Z", 0, mz_ex2, "M_z"), (axes[1], "X", 1, mx_ex2, "M_x")):
    ax.plot(dts, trot_err[:, j], "o:", color="grey", label="Trotter error only (noiseless)")
    ax.plot(dts, [abs(r[f"{b}_raw"] - exv) for r in tradeoff], "o-", color=COL["raw"], label="noisy, raw")
    ax.plot(dts, [abs(r[f"{b}_zne_exp"] - exv) for r in tradeoff], "D-", color=COL["zne"], label="noisy + TREX + ZNE")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Trotter step Δt  (n = 2/Δt steps)")
    ax.set_title(f"error in ${name}(t=2)$"); ax.axvline(0.25, color="k", lw=0.8, ls="--")
axes[0].set_ylabel("|error| at t = 2"); axes[0].legend(fontsize=8)
axes[0].text(0.17, 0.0016, "M_z(2) ≈ 0.05: a small signal, so\nshot noise (σ≈0.01) dominates this panel", fontsize=8, color="grey")
for ax in axes: ax.text(0.252, ax.get_ylim()[1] * 0.6, "chosen Δt = 0.25", fontsize=8, rotation=90, va="top")
for r in tradeoff:
    axes[1].annotate(f"{r['cz']} CZ", (r["dt"], abs(r["X_raw"] - mx_ex2)), fontsize=7, xytext=(2, 4), textcoords="offset points")
plt.tight_layout(); savefig("i2_tradeoff"); plt.show()
"""))

C(md(r"""
The $M_x$ panel is the informative one.
* **Raw (unmitigated) error: a noise-limited plateau.** For $0.25 \le \Delta t \le 0.4$ the raw error is flat at 0.065–0.07, even though the Trotter error underneath it falls 4×. Going finer makes it *worse*: 0.087 at $\Delta t=0.2$ and 0.105 at $\Delta t=0.167$, because each extra CZ layer adds more noise than it removes Trotter error.
* **After TREX + ZNE**, every $\Delta t \le 0.4$ lands at 0.002–0.036. These differences are within this sweep's shot noise (σ ≈ 0.02 with 1000/500 shots), so they do **not** single out one optimal step.
* **Conclusion.** Any $\Delta t$ on the plateau 0.25–0.4 is as good as the finer steps, at 38–62% fewer CZs (110–176 vs 286) than the $\Delta t = 2/13 \approx 0.15$ that the Trotter-only criterion requires for 12 spins. We keep $\Delta t = 0.25$ (176 CZ): it sits on the plateau and keeps the Trotter error below 0.025. $\Delta t = 1/3$ (132 CZ) would have been a defensible cheaper choice.
* The $M_z(2)$ signal is only about 0.05, so that panel is dominated by shot noise and we draw no conclusions from it.

### 2.5 The same 12-spin circuit on a real IBM QPU (prepared, not run)

Job B in `hardware/run_hardware.py` submits these eight circuits through IBM's `EstimatorV2`, with TREX, Pauli-twirled CZs, XpXm dynamical decoupling and gate-folding ZNE ($\lambda = 1, 3, 5$). When results exist, the cell below plots the device result at $\lambda=1$ (twirled, readout-mitigated, before extrapolation) and after ZNE, against our noisy simulation and the exact curve. We did not obtain QPU access before the deadline.
"""))

C(code(r"""
hwB_path = RESULTS / "hardware_B.npz"
jobs_meta = json.load(open(RESULTS / "hardware_jobs.json")) if (RESULTS / "hardware_jobs.json").exists() else None

def hw_noise_factor_slice(d, i, which=0):
    # EstimatorV2 with ZNE returns evs (extrapolated) and evs_noise_factors (..., n_factors)
    for key in (f"p{i}_evs_noise_factors",):
        if key in d:
            return np.asarray(d[key])[..., which]
    return None

if hwB_path.exists():
    dB = dict(np.load(hwB_path))
    mB = jobs_meta["meta"]["B"]; kB = np.array([m["steps"] for m in mB]); tB = t12[kB]   # steps actually run
    hz, hx, hz1, hx1 = [], [], [], []
    for i in range(len(mB)):
        ev = dB[f"p{i}_evs"]; hz.append(ev[:N12].mean()); hx.append(ev[N12:].mean())
        nf = hw_noise_factor_slice(dB, i)
        if nf is not None:
            hz1.append(nf[:N12].mean()); hx1.append(nf[N12:].mean())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for ax, ex, sraw, szne, hwz, hw1, name in ((axes[0], mz12_exact, sim12["Z_raw"], sim12["Z_zne_exp"], hz, hz1, "M_z"),
                                              (axes[1], mx12_exact, sim12["X_raw"], sim12["X_zne_exp"], hx, hx1, "M_x")):
        ax.plot(t12, ex, color=COL["exact"], lw=2, label="exact")
        ax.plot(t12[1:], sraw.mean(1), "o--", color=COL["raw"], ms=4, label="FakeFez sim, raw")
        ax.plot(t12[1:], szne.mean(1), "D--", color="#ff9896", ms=4, label="FakeFez sim, TREX+ZNE")
        if hw1:
            ax.plot(tB, hw1, "s-", color="#1f77b4", ms=5, label=f"{jobs_meta['backend']}: λ=1 (TREX+twirl+DD)")
        ax.plot(tB, hwz, "D-", color=COL["hw"], ms=5, label=f"{jobs_meta['backend']}: + ZNE")
        ax.set_xlabel("t"); ax.set_ylabel(f"${name}$"); ax.set_title(f"12 spins on real hardware: ${name}$")
    axes[0].legend(fontsize=7.5)
    plt.tight_layout(); savefig("i3_hardware_tfim"); plt.show()
    print("mean |QPU+ZNE - exact|:  Mz %.4f   Mx %.4f" % (np.abs(np.array(hz) - mz12_exact[kB]).mean(),
                                                         np.abs(np.array(hx) - mx12_exact[kB]).mean()))
else:
    print("results/hardware_B.npz not found: run hardware/run_hardware.py submit / fetch to add the QPU curve.")
"""))
