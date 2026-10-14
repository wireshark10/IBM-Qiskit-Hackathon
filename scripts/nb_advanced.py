"""Notebook cells: Advanced level."""
from nbcells import md, code

CELLS = []
C = CELLS.append

C(md(r"""
## 3. Advanced: a 100-qubit kicked Ising chain, checked without brute force

We use the kicked (Floquet) Ising model from Kim et al., Nature 618, 500 (2023), but on a 1D chain. One Floquet step is

$$U_F(\theta_h) = \prod_j R_X^{(j)}(\theta_h)\;\prod_{\langle j,k\rangle} R_{ZZ}^{(jk)}(\theta_J),\qquad \theta_J=-\pi/2,$$

applied $t$ times to $|0\rangle^{\otimes N}$ on an open chain of N = 100 qubits. The experiment (prepared for the QPU, run here on a noise model) measures every $\langle Z_i\rangle$ (magnetisation $M_z$) and every nearest-neighbour $\langle Z_iZ_{i+1}\rangle$, for $\theta_h\in\{0,\pi/8,\pi/4,3\pi/8\}$ and $t\in\{1,\dots,8,10,12,14,16,20\}$. That makes 52 circuits, the deepest with 1881 CZ gates at two-qubit depth 38.

This circuit suits the hardware well. At $\theta_J=-\pi/2$ each $R_{ZZ}$ is exactly one CZ plus single-qubit phases, and on a line of physically connected qubits it needs no SWAPs at all. On $|0\cdots0\rangle$ the first ZZ layer is only a global phase, so we leave it out.

### 3.1 How can we know the answer for 100 qubits?

The challenge asks us to argue that the results are right without brute force. We ended up with three independent exact methods. Each of them works at 100 qubits, and we checked each one against the others and against statevector simulation:

| Method | Gives | Cost | Works for |
|---|---|---|---|
| Light-cone window (`il.kicked_ising_exact_profile`) | every $\langle Z_i\rangle$, $\langle Z_iZ_{i+1}\rangle$ | statevector on $2t+2$ qubits, independent of N | any model with local gates, up to $t\approx 11$ |
| Free fermions + Pfaffians (`il.free_fermion_profile`) | every $\langle Z_i\rangle$ and any $\langle Z_iZ_j\rangle$ | $O(N^3)$, about 0.3 s for N = 100 at $t=20$ | the integrable 1D kicked transverse-field Ising chain, at any depth |
| Clifford point $\theta_h=\pi/2$ (`il.clifford_stabilizer`) | evolved stabilisers $S=UZ_cU^\dagger$, $\langle S\rangle=+1$ | stabiliser tableau, polynomial | any N, any depth |

How the free-fermion method works: with a Jordan-Wigner transformation in the X basis, $X_j=-i\gamma_{2j}\gamma_{2j+1}$ and $Z_jZ_{j+1}=-i\gamma_{2j+1}\gamma_{2j+2}$. Every gate then just rotates the vector of Majorana operators, and the covariance matrix evolves as $M\to RMR^T$. The start state $|0\cdots0\rangle=(|\text{GHZ}_+\rangle+|\text{GHZ}_-\rangle)/\sqrt2$ is a sum of two Gaussian states, and for parity-even observables the two parity sectors don't interfere. The difficulty is that $Z_i$ itself is parity-odd (in fermion language it is a long string). We got around this by adding an extra "ghost" spin that never evolves, so that $\langle Z_i\rangle = \langle Z_gZ_i\rangle$, which is an even product of Majoranas. Wick's theorem then gives it as a Pfaffian.

One caveat: integrability is exactly what lets us check 100 qubits at depth 20, and it also means this model is easy for a classical computer. We use it as a test bench where the right answer is known, not as an example of quantum advantage.
"""))

C(code(r"""
_jm = json.load(open(RESULTS / "hardware_jobs.json")) if (RESULTS / "hardware_jobs.json").exists() else None
N_KI = _jm["plan"]["N_KI"] if _jm else 100          # chain length actually run on the QPU
TH_ALL = {"0": 0.0, "π/8": np.pi / 8, "π/4": np.pi / 4, "3π/8": 3 * np.pi / 8}
T_KI = _jm["plan"]["T_KI"] if _jm else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]
TH = {k: v for k, v in TH_ALL.items() if (not _jm) or any(abs(v - x) < 1e-9 for x in _jm["plan"]["THETAS"])}

# (1) all three exact methods vs full statevector on a 12-qubit chain
n_small, worst = 12, 0.0
for t in (1, 3, 5):
    for th in TH.values():
        sv = Statevector(il.kicked_ising_circuit(n_small, t, th))
        z_sv = np.array([np.real(sv.expectation_value(o)) for o in il.single_site_ops(n_small)])
        z_ff, _ = il.free_fermion_profile(n_small, t, th)
        z_lc, _ = il.kicked_ising_exact_profile(n_small, t, [th])
        worst = max(worst, np.abs(z_ff - z_sv).max(), np.abs(z_lc[0] - z_sv).max())
    S = il.clifford_stabilizer(n_small, t, n_small // 2)
    print(f"t={t}: Clifford stabiliser weight {sum(c != 'I' for c in S.paulis[0].to_label()):2d}, "
          f"<S> statevector = {np.real(Statevector(il.kicked_ising_circuit(n_small, t, np.pi/2)).expectation_value(S)):.12f}")
print(f"12 qubits: max |free-fermion or light-cone - statevector| = {worst:.1e}")

# (2) at N = 100: light-cone window vs free fermions (two unrelated algorithms)
t0 = time.time(); worst = 0.0
for t in (2, 5, 8):
    zlc, zzlc = il.kicked_ising_exact_profile(N_KI, t, [np.pi / 4])
    zff, zzff = il.free_fermion_profile(N_KI, t, np.pi / 4, zz_pairs=[(i, i + 1) for i in range(N_KI - 1)])
    worst = max(worst, np.abs(zlc[0] - zff).max(), np.abs(zzlc[0] - np.array([zzff[(i, i + 1)] for i in range(N_KI - 1)])).max())
print(f"N=100: max |light-cone - free-fermion| over all sites/bonds, t=2,5,8: {worst:.1e}   ({time.time()-t0:.0f}s)")
"""))

C(code(r"""
# Exact reference for every hardware circuit: site-resolved <Z_i>, <Z_i Z_i+1> and a few long-range <Z_50 Z_50+r>
LR = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
exact_ki = {}
t0 = time.time()
for lab, th in TH.items():
    for t in T_KI:
        pairs = [(i, i + 1) for i in range(N_KI - 1)] + [(N_KI // 2 - r // 2, N_KI // 2 - r // 2 + r) for r in LR]
        z, zz = il.free_fermion_profile(N_KI, t, th, zz_pairs=pairs)
        exact_ki[(lab, t)] = {"z": z, "zz": np.array([zz[(i, i + 1)] for i in range(N_KI - 1)]),
                              "lr": np.array([zz[(N_KI // 2 - r // 2, N_KI // 2 - r // 2 + r)] for r in LR])}
print(f"exact references for {len(exact_ki)} (θ_h, t) pairs at N=100 in {time.time()-t0:.0f}s")
mz_exact_ki = {lab: np.array([exact_ki[(lab, t)]["z"].mean() for t in T_KI]) for lab in TH}
for lab in TH:
    print(f"θ_h={lab:5s} exact M_z(t):", np.round(mz_exact_ki[lab], 3))
"""))

C(md(r"""
### 3.2 Hardware: choosing 100 qubits and compiling without SWAPs

`il.best_line` runs a beam search over the device's coupling map. It looks for the path of 100 qubits that minimises $-\log$ ESP for an average circuit in our plan (8 CZ per bond, 16 SX and one readout per qubit), using that day's calibration data. A 1D chain on this path needs no routing. The submission script saves the path, a calibration snapshot and the compiled-circuit statistics with every job.

Error suppression and mitigation planned for the QPU (IBM Runtime `EstimatorV2`):
* Pauli twirling of every CZ layer (16 randomisations);
* TREX readout mitigation;
* XpXm dynamical decoupling on idle qubits;
* zero-noise extrapolation by gate folding ($\lambda = 1,3,5$), with an exponential fit and a linear fallback.

The plan also includes a `SamplerV2` job that returns raw, unmitigated bitstrings for two angles, plus $|0\rangle^{\otimes100}$ and $|1\rangle^{\otimes100}$ calibration shots. This gives a baseline without any mitigation and lets us look at long-range correlations.
"""))

C(code(r"""
jobs_meta = json.load(open(RESULTS / "hardware_jobs.json")) if (RESULTS / "hardware_jobs.json").exists() else None
if jobs_meta is None:
    print("No hardware record yet: showing the compile plan on FakeFez instead.")
    import sys; sys.path.insert(0, "hardware")
    import run_hardware as rh
    (pa, ma), (pb, mb), (pc, mc), lay = rh.build_all(fez)
    meta_A, backend_name, layout = ma, "fake_fez (plan)", lay
else:
    meta_A, backend_name, layout = jobs_meta["meta"]["A"], jobs_meta["backend"], jobs_meta["layout"]
    print("backend:", backend_name, "| submitted:", jobs_meta["submitted"])
    print("job IDs:", jobs_meta["jobs"])
    cal = jobs_meta["calibration"]
    ro = np.array([v for v in cal["readout_error"].values() if v is not None])
    cz = np.array([v for v in cal["two_q_error"].values() if v is not None])
    print(f"chosen line: median CZ error {np.median(cz):.2e} (max {cz.max():.2e}), median readout error {np.median(ro):.2e} (max {ro.max():.2e})")
print("line (first 20 physical qubits):", layout["path_ki"][:20], "...")
print(f"\n{'circuit':>16s} {'CZ':>6s} {'ideal CZ':>9s} {'2q depth':>9s} {'depth':>6s} {'SWAPs':>6s}")
for m in meta_A:
    lab = f"KI t={m['t']}" if m["kind"] == "ki" else f"Clifford t={m['t']}"
    print(f"{lab:>16s} {m['two_q']:6d} {m['ideal_two_q']:9d} {m['depth_2q']:9d} {m['depth']:6d} {m['swaps']:6.0f}")
print("CZ count == (t-1)(N-1) for every circuit: the router inserted no SWAPs.")
"""))

C(md(r"""
### 3.3 The 100-qubit experiment on a noise model

QPU status: we could not finish IBM Quantum account verification before the deadline, so there is no real-QPU data. The submission script is complete and validated against the FakeFez target. For the rest of the Advanced part we use a noise model built from the calibration of the same 100-qubit line we would run on. FakeFez is a snapshot of IBM Heron r2 *ibm_fez*, and the model has per-gate CZ / SX / X depolarising errors and per-qubit asymmetric readout errors. Section 4 lists what this does and doesn't capture.

How we simulate 100 noisy qubits without brute force:
* Clifford points on the full chain. At $\theta_h=0$ and $\theta_h=\pi/2$ the circuit is Clifford and the noise is Pauli, so a stabiliser simulation of the whole noisy 100-qubit circuit is exact and fast. This includes the CZ-folded circuits for ZNE and real shot noise (400 shots per circuit, split over 4 TREX masks). We compiled these two points by hand using $R_{ZZ}(-\pi/2)\simeq CZ\,(S^\dagger\!\otimes S^\dagger)$ and $R_X(\pi/2)\simeq SX$, which gives the same CZ count, (t−1)·99, as the transpiled circuits.
* Light-cone windows for the other angles. For $\theta_h\in\{\pi/8,\pi/4,3\pi/8\}$, Pauli noise outside the backward light cone can't affect $\langle Z_c\rangle$. So `il.noisy_bulk_z` simulates only the $(2t+1)$-qubit window of the real line and averages 200 noise trajectories. This is exact apart from small differences in how the transpiler compiles single-qubit gates at the edge of the window; the next cell compares the window with a full-chain density-matrix simulation. We repeat it at 5 positions along the chain to sample how the noise varies across the device. Readout is handled analytically here: "raw" applies each qubit's readout channel, and "TREX" removes it exactly, so it is an ideal TREX without calibration shot noise.

The mitigation is the same as in the QPU plan: TREX readout twirling, then ZNE by CZ folding ($\lambda=1,3,5$, exponential and linear fits).
"""))

C(code(r"""
# Validation of the light-cone noisy window: exact density-matrix simulation of the FULL noisy chain
# vs the (2t+1)-qubit window, same calibrated noise model, no sampling noise.
from qiskit_aer import AerSimulator
def noisy_dm_z(nq, qubits, centre, t, th):
    pm = generate_preset_pass_manager(optimization_level=2, backend=fez, initial_layout=qubits)
    comp = il.compact_circuit(pm.run(il.kicked_ising_circuit(nq, t, th)), qubits)
    comp.save_expectation_value(SparsePauliOp(il.pauli_label(nq, {centre: "Z"})), range(nq), label="z")
    sim = AerSimulator(method="density_matrix", noise_model=il.compact_noise_model(fez, qubits))
    return float(np.real(sim.run(comp).result().data(0)["z"]))
line11, _ = il.best_line(fez.target, 11, cz_per_bond=8, sx_per_qubit=16)
for t_, th_ in ((2, np.pi / 4), (3, np.pi / 8), (3, np.pi / 4)):
    c_ = 5
    full_ = noisy_dm_z(11, line11, c_, t_, th_)
    win_ = noisy_dm_z(2 * t_ + 1, line11[c_ - t_: c_ + t_ + 1], t_, t_, th_)
    ideal_ = il.free_fermion_profile(11, t_, th_)[0][c_]
    print(f"t={t_}, θ_h={th_:.3f}: full noisy chain {full_:.6f} | window {win_:.6f} | "
          f"|diff| = {abs(full_ - win_):.1e} = {abs(full_ - win_) / abs(full_ - ideal_):.1%} of the noise effect")
"""))

C(code(r"""
if RECOMPUTE:                                   # ~25 min per part on a laptop; parts are independent
    import subprocess, sys
    for part in ("z0", "stab", "windows"):
        subprocess.run([sys.executable, "scripts/emulate_advanced.py", part], check=True)
emu = {k: json.load(open(RESULTS / f"emu_{k}.json")) for k in ("z0", "stab", "windows") if (RESULTS / f"emu_{k}.json").exists()}
F = np.array([1, 3, 5])
def zne(stack, kind="exponential"):
    return il.extrapolate_zero(F, np.asarray(stack, float), kind)

fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
emu_dev = {}
if "z0" in emu:
    e = emu["z0"]; nt = min(len(e["trex"][str(f)]) for f in F); t = np.array(e["t"][:nt])
    st = np.array([e["trex"][str(f)][:nt] for f in F]); e = {**e, "raw": e["raw"][:nt]}
    ax = axes[0, 0]
    ax.axhline(1, color=COL["exact"], lw=2, label="exact (= 1)")
    ax.plot(t, e["raw"], "o:", color=COL["raw"], label="noisy, raw")
    ax.plot(t, st[0], "s--", color=COL["trex"], label="+ TREX (λ=1)")
    ax.plot(t, zne(st, "linear"), "v-", color="#e377c2", label="+ ZNE linear")
    ax.plot(t, zne(st), "D-", color=COL["zne"], label="+ ZNE exponential")
    ax.set_title("θ_h = 0: full 100-qubit noisy simulation"); ax.set_xlabel("Floquet steps t"); ax.set_ylabel("$M_z$")
    ax.legend(fontsize=7.5); emu_dev["0"] = (t, np.abs(zne(st) - 1))
if "stab" in emu:
    e = emu["stab"]; nt = min(len(e["trex"][str(f)]) for f in F)
    w = np.array(e["weights"][:nt]).mean(1); st = np.array([e["trex"][str(f)][:nt] for f in F]).mean(-1)
    e = {**e, "raw": e["raw"][:nt], "t": e["t"][:nt]}
    ax = axes[0, 1]
    ax.axhline(1, color=COL["exact"], lw=2, label="exact (+1)")
    ax.plot(w, np.array(e["raw"]).mean(1), "o:", color=COL["raw"], label="noisy, raw")
    ax.plot(w, st[0], "s--", color=COL["trex"], label="+ TREX")
    ax.plot(w, zne(st, "linear"), "v-", color="#e377c2", label="+ ZNE linear")
    ax.plot(w, zne(st), "D-", color=COL["zne"], label="+ ZNE exponential")
    ax.set_title("θ_h = π/2: Clifford stabilisers, 100 qubits"); ax.set_xlabel("stabiliser weight 2t−1")
    ax.set_ylabel(r"$\langle S_c\rangle$ (2 centres)"); ax.legend(fontsize=7.5)
    emu_dev["π/2 (stabiliser)"] = (np.array(e["t"]), np.abs(zne(st) - 1))
if "windows" in emu:
    e = emu["windows"]; nt = min(len(e["trex"][str(f)]) for f in F); t = np.array(e["t"][:nt]); labs = ["π/8", "π/4", "3π/8"]
    raw = np.array(e["raw"][:nt]); st = np.array([e["trex"][str(f)][:nt] for f in F])   # (F, t, theta)
    for a, (ax, lab) in enumerate(zip(axes[1], labs)):
        ex = np.array([il.free_fermion_profile(N_KI, int(tt), TH_ALL[lab])[0][N_KI // 2] for tt in t])
        ax.plot(t, ex, "-", color=COL["exact"], lw=2, label="exact")
        ax.plot(t, raw[:, a], "o:", color=COL["raw"], label="noisy, raw")
        ax.plot(t, st[0, :, a], "s--", color=COL["trex"], label="+ TREX")
        ax.plot(t, zne(st[:, :, a], "linear"), "v-", color="#e377c2", label="+ ZNE linear")
        ax.plot(t, zne(st[:, :, a]), "D-", color=COL["zne"], label="+ ZNE exponential")
        ax.set_title(f"θ_h = {lab}: bulk ⟨Z⟩ (5 positions on the line)"); ax.set_xlabel("Floquet steps t")
        emu_dev[lab] = (t, np.abs(zne(st[:, :, a]) - ex))
    axes[1, 0].set_ylabel(r"bulk $\langle Z_i\rangle$"); axes[1, 0].legend(fontsize=7.5)
ax = axes[0, 2]                                   # residuals: the trust horizon at a glance
for lab, (t_, d_) in emu_dev.items():
    ax.semilogy(t_, np.maximum(d_, 1e-4), "o-", ms=4, label=f"θ_h = {lab}")
ax.axhline(0.05, color="k", ls="--", lw=1, label="trust threshold 0.05")
ax.set_xlabel("Floquet steps t"); ax.set_ylabel("|TREX+ZNE − exact|"); ax.set_title("mitigated error vs depth")
ax.legend(fontsize=7)
plt.suptitle(f"100-qubit kicked Ising on the calibrated {emu.get('z0', emu.get('windows', {})).get('backend', 'FakeFez')} noise model", y=1.0)
plt.tight_layout(); savefig("a6_emulation"); plt.show()
"""))

C(md(r"""
Trust horizon on the emulator: for each observable we list $|\text{ZNE}-\text{exact}|$ by depth, and the deepest $t$ that stays within 0.05. Depolarising noise pulls values towards 0, so the observables with a large signal are the ones that set the horizon: $\theta_h=0$ (exact value 1), the stabilisers (exact value +1, weight up to 39) and $\theta_h=\pi/8$. At $3\pi/8$ even the raw data "agrees" at late times, simply because the exact answer there is close to 0.
"""))

C(code(r"""
emu_horizon = {}
for lab, (t, d) in emu_dev.items():
    ok = d <= 0.05
    passing = [int(tt) for tt, o in zip(t, ok) if o]
    emu_horizon[lab] = max(passing) if passing else 0
    print(f"{lab:>18s}: " + "  ".join(f"t={int(tt)}:{dd:.3f}{'✓' if o else '✗'}" for tt, dd, o in zip(t, d, ok)))
print("\ntrust horizon on the 100-qubit emulator (deepest t with |ZNE − exact| ≤ 0.05):", emu_horizon)
if "z0" in emu:
    nt = min(len(emu["z0"]["trex"][str(f)]) for f in F)
    st = np.array([emu["z0"]["trex"][str(f)][:nt] for f in F])
    print("θ_h=0  |exp − linear ZNE| by depth:", np.round(np.abs(zne(st) - zne(st, "linear")), 3),
          " (internal consistency, needs no exact answer)")
parts = []
if "z0" in emu:
    nt = min(len(emu["z0"]["trex"][str(f)]) for f in F); tl = emu["z0"]["t"][nt - 1]
    st = np.array([emu["z0"]["trex"][str(f)][:nt] for f in F])
    parts.append(f"θ_h=0, t={tl} ({(tl-1)*99} CZ): raw {emu['z0']['raw'][nt-1]:.2f} → TREX+ZNE {zne(st)[-1]:.2f} (exact 1)")
if "stab" in emu:
    nt = min(len(emu["stab"]["trex"][str(f)]) for f in F)
    st = np.array([emu["stab"]["trex"][str(f)][:nt] for f in F]).mean(-1)
    parts.append(f"weight-{max(emu['stab']['weights'][nt-1])} stabiliser: raw {np.mean(emu['stab']['raw'][nt-1]):.2f} → {zne(st)[-1]:.2f} (exact 1)")
emu_takeaway = "; ".join(parts) + "."
print(emu_takeaway)
json.dump({"emu_horizon": emu_horizon, "emu_takeaway": emu_takeaway}, open(RESULTS / "emu_horizon.json", "w"))
"""))

C(md(r"""
What we learn from the 100-qubit emulation:
* Single-site observables hold up in deep circuits. At $\theta_h=0$, $M_z$ after 20 Floquet steps (1881 CZ) is 0.78 raw, and TREX + ZNE brings it back to 0.96 against an exact value of 1, so it stays within 0.05 at every depth. The remaining 0.035 at $t=20$ is 3–5σ (propagating binomial shot noise through the exponential fit gives σ ≈ 0.007–0.01 for the 100-site average with 400 shots per noise factor). It is therefore a systematic bias of the exponential fit rather than shot noise, and the gap between the exponential and linear fits (0.08 at $t=20$) shows it even without the exact answer. At the non-Clifford angles the bulk $\langle Z\rangle$ comes back to within 0.043 up to $t=7$, the deepest window we emulated.
* How far we can trust a result depends on the weight of the observable. A Clifford stabiliser of weight $w=2t-1$ picks up an error from any CZ inside its light cone, and there are roughly $t^2$ of those. Its signal therefore decays much faster than a single $\langle Z_i\rangle$, which at $\theta_h=0$ only sees errors on its own two bonds. At weight 7 ($t=4$) ZNE still lands within 0.04 of +1, and at weight 11 it misses by 0.11. From weight 15 on, the $\lambda=5$ data has sunk into the shot noise (and the TREX rescaling by $\prod_i f_i$ makes that worse), so the extrapolation fails: we get −0.03 instead of +1 at weight 39. Getting high-weight correlators right at this depth would need PEC/PEA-type mitigation or many more shots.
* The gap between the exponential and linear ZNE fits grows from 0.001 to 0.08 with depth and follows the true error. This is the check we would rely on where no exact answer exists.
"""))

C(md(r"""
### 3.4 Real-QPU results (these cells fill in when `results/hardware_*.npz` exists)

For each $\theta_h$: raw device data (Sampler, no mitigation), the twirled and readout-mitigated value at $\lambda=1$, the ZNE value, and the exact free-fermion curve.
"""))

C(code(r"""
hwA_path, hwC_path = RESULTS / "hardware_A.npz", RESULTS / "hardware_C.npz"
HAVE_HW = hwA_path.exists()
if HAVE_HW:
    dA = dict(np.load(hwA_path))
    nki = [i for i, m in enumerate(meta_A) if m["kind"] == "ki"]
    # evs shape: (n_theta, 199) -> first 100 are Z_i, then 99 Z_i Z_i+1
    def get(i, key):
        return np.asarray(dA[f"p{i}_{key}"])
    hw = {}
    for lab_i, lab in enumerate(TH):
        hw[lab] = {"zne_z": np.array([get(i, "evs")[lab_i, :N_KI] for i in nki]),
                   "zne_zz": np.array([get(i, "evs")[lab_i, N_KI:] for i in nki]),
                   "std_z": np.array([get(i, "stds")[lab_i, :N_KI] for i in nki])}
        if f"p{nki[0]}_evs_noise_factors" in dA:
            nf = [get(i, "evs_noise_factors") for i in nki]           # (..., n_factors)
            hw[lab]["nf_z"] = np.array([x[lab_i, :N_KI, :] for x in nf])
            hw[lab]["nf_zz"] = np.array([x[lab_i, N_KI:, :] for x in nf])
    print("hardware arrays loaded:", {k: v.shape for k, v in hw["π/4"].items()})
else:
    print("results/hardware_A.npz not found yet: the plots below show only the exact reference.")

raw = {}
if hwC_path.exists():
    dC = dict(np.load(hwC_path))
    meta_C = jobs_meta["meta"]["C"]
    def unpack(i):
        bits = np.unpackbits(dC[f"p{i}_bits"], axis=-1)[:, ::-1][:, :N_KI]   # column j = clbit j = virtual qubit j
        return bits.astype(np.int8)
    i0 = [i for i, m in enumerate(meta_C) if m["kind"] == "cal0"][0]
    i1 = [i for i, m in enumerate(meta_C) if m["kind"] == "cal1"][0]
    p01 = unpack(i0).mean(0); p10 = 1 - unpack(i1).mean(0)                     # P(1|0), P(0|1) per qubit
    print(f"readout from calibration shots: mean P(1|0)={p01.mean():.3f}, mean P(0|1)={p10.mean():.3f}")
    for i, m in enumerate(meta_C):
        if m["kind"] != "raw":
            continue
        s = 1 - 2 * unpack(i)                                                   # ±1 spins
        zr = s.mean(0)
        z_ro = (zr - (p10 - p01)) / (1 - p01 - p10)                             # invert local readout channel
        lab = min(TH, key=lambda k: abs(TH[k] - m["theta"]))
        raw[(lab, m["t"])] = {"z": zr, "z_ro": z_ro, "spins": s}
    print("raw sampler circuits:", len(raw))
"""))

C(code(r"""
fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True)
tt = np.array(T_KI)
for ax, lab in zip(axes.flat, TH):
    tf = np.arange(1, 21)
    ex_fine = [il.free_fermion_profile(N_KI, t, TH[lab])[0].mean() for t in tf] if lab != "0" else np.ones_like(tf, float)
    ax.plot(tf, ex_fine, "-", color=COL["exact"], lw=2, label="exact (free fermions)")
    if raw and (lab, T_KI[0]) in raw:
        ax.plot(tt, [raw[(lab, t)]["z"].mean() for t in tt], "o:", color=COL["raw"], ms=4, label="raw QPU (no mitigation)")
    if HAVE_HW:
        if "nf_z" in hw[lab]:
            ax.plot(tt, hw[lab]["nf_z"][:, :, 0].mean(1), "s--", color="#1f77b4", ms=4, label="QPU λ=1: twirl + TREX + DD")
        mz = hw[lab]["zne_z"].mean(1); err = np.sqrt((hw[lab]["std_z"]**2).sum(1)) / N_KI
        ax.errorbar(tt, mz, yerr=err, fmt="D-", color=COL["hw"], ms=5, capsize=2, label="QPU + ZNE")
    ax.set_title(f"θ_h = {lab}"); ax.set_ylabel(r"$M_z$ (100 qubits)")
for ax in axes[1]:
    ax.set_xlabel("Floquet steps t")
for ax in axes.flat:
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16, 20])
axes[0, 0].legend(fontsize=8); axes[0, 0].set_ylim(0.0, 1.08)
fig.suptitle(f"Kicked Ising, N = {N_KI}: exact free-fermion references" + (f" vs {backend_name}" if HAVE_HW else ""), y=1.0)
plt.tight_layout(); savefig("a1_mz_100q"); plt.show()
"""))

C(code(r"""
if HAVE_HW:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for lab, c in zip(TH, ("#7f7f7f", "#2ca02c", "#1f77b4", "#d62728")):
        ex = [exact_ki[(lab, t)]["zz"].mean() for t in T_KI]
        axes[0].plot(tt, ex, "-", color=c, lw=1.5)
        axes[0].plot(tt, hw[lab]["zne_zz"].mean(1), "D", color=c, ms=5, label=f"θ_h={lab}")
        if "nf_zz" in hw[lab]:
            axes[0].plot(tt, hw[lab]["nf_zz"][:, :, 0].mean(1), "x", color=c, ms=5)
    axes[0].set_xlabel("Floquet steps t"); axes[0].set_ylabel(r"$\overline{\langle Z_iZ_{i+1}\rangle}$")
    axes[0].set_title("nearest-neighbour correlator: lines exact, ◆ ZNE, × λ=1"); axes[0].legend(fontsize=8)
    # site-resolved profile at one depth: does the device resolve the boundary physics?
    t_show = 6 if 6 in T_KI else T_KI[len(T_KI) // 2]; k = T_KI.index(t_show)
    axes[1].plot(range(N_KI), exact_ki[("π/4", t_show)]["z"], "-", color=COL["exact"], lw=2, label="exact")
    axes[1].plot(range(N_KI), hw["π/4"]["zne_z"][k], ".", color=COL["hw"], label="QPU + ZNE")
    if raw:
        axes[1].plot(range(N_KI), raw[("π/4", t_show)]["z"], ".", color=COL["raw"], alpha=0.6, label="raw QPU")
    axes[1].set_xlabel("site i"); axes[1].set_ylabel(r"$\langle Z_i\rangle$")
    axes[1].set_title(f"site-resolved profile, θ_h=π/4, t={t_show} (edges differ from the bulk)")
    axes[1].legend(fontsize=8)
    plt.tight_layout(); savefig("a2_zz_profile"); plt.show()
"""))

C(md(r"""
Noiseless, noise model and hardware at 100 qubits. Pauli noise outside the backward light cone $[c-t,c+t]$ can't change $\langle Z_c\rangle$, so the noisy value at a bulk site can be simulated on a $(2t+1)$-qubit window. The window is transpiled onto the matching part of the hardware line and gets that part's calibrated gate and readout errors (`il.noisy_bulk_z`, cached by `scripts/emulate_hw.py`). Comparing this prediction with the QPU would show how much of the hardware noise the calibration data explains. On the hardware side we average over the bulk sites $t<i<N-1-t$, which all have the same exact value.
"""))

C(code(r"""
emu_path = RESULTS / "noisy_emulation.json"
if emu_path.exists():
    emu = json.load(open(emu_path)); te = emu["t"]
    labs = ["π/8", "π/4", "3π/8"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, lab in zip(axes, labs):
        a = list(TH_ALL).index(lab)
        ex = [il.free_fermion_profile(N_KI, t, TH_ALL[lab])[0][N_KI // 2] for t in te]
        ax.plot(te, ex, "-", color=COL["exact"], lw=2, label="exact")
        ax.plot(te, [r[a] for r in emu["gates_only"]], "--", color="#1f77b4", label="noise model: gate errors")
        ax.plot(te, [r[a] for r in emu["with_readout"]], ":", color="grey", label="noise model: gates + readout")
        if HAVE_HW and lab in hw:
            te_hw = [t for t in te if t in T_KI]
            ks = [T_KI.index(t) for t in te_hw]
            bulk = lambda t: slice(t + 1, N_KI - 1 - t)
            if "nf_z" in hw[lab]:
                ax.plot(te_hw, [hw[lab]["nf_z"][k, bulk(t), 0].mean() for k, t in zip(ks, te_hw)], "s", color="#1f77b4", label="QPU λ=1 (twirl+TREX+DD)")
            ax.plot(te_hw, [hw[lab]["zne_z"][k, bulk(t)].mean() for k, t in zip(ks, te_hw)], "D", color=COL["hw"], label="QPU + ZNE")
            if raw and all((lab, t) in raw for t in te_hw):
                ax.plot(te_hw, [raw[(lab, t)]["z"][bulk(t)].mean() for t in te_hw], "o", color=COL["raw"], label="QPU raw")
        ax.set_title(f"bulk ⟨Z⟩, θ_h = {lab}"); ax.set_xlabel("Floquet steps t")
    axes[0].set_ylabel(r"bulk $\langle Z_i\rangle$"); axes[0].legend(fontsize=7)
    plt.suptitle(f"noise-model prediction from {emu['backend']} calibration" + (" vs the device" if HAVE_HW else " (no QPU data)"), y=1.02)
    plt.tight_layout(); savefig("a5_noise_model_vs_qpu"); plt.show()
else:
    print("results/noisy_emulation.json not found (run scripts/emulate_hw.py)")
"""))

C(md(r"""
### 3.5 Clifford-point check on the QPU

At $\theta_h=\pi/2$ the circuit is Clifford. The evolved stabiliser $S_c(t)=U Z_c U^\dagger$ is a Pauli string of weight $2t-1$ with $\langle S_c\rangle=+1$ exactly, for any system size. It is the hardest observable in our set, because an error anywhere in its growing support affects it. How quickly it decays with weight tells us the device's error per light-cone layer, and ZNE should bring it back towards 1.
"""))

C(code(r"""
if HAVE_HW:
    ncl = [i for i, m in enumerate(meta_A) if m["kind"] == "clifford"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for j, c in enumerate(meta_A[ncl[0]]["centers"]):
        wts = [meta_A[i]["weights"][j] for i in ncl]
        ev = [get(i, "evs")[j] for i in ncl]; sd = [get(i, "stds")[j] for i in ncl]
        ax.errorbar(wts, ev, yerr=sd, fmt="D-", capsize=2, color=COL["hw"] if j == 0 else "#ff9896",
                    label=f"QPU + ZNE, centre {c}")
        if f"p{ncl[0]}_evs_noise_factors" in dA:
            ax.plot(wts, [get(i, "evs_noise_factors")[j][0] for i in ncl], "s--", color="#1f77b4" if j == 0 else "#aec7e8",
                    label=f"QPU λ=1, centre {c}")
    ax.axhline(1, color="k", lw=1.5, label="exact (+1)")
    ax.set_xlabel("stabiliser weight 2t−1"); ax.set_ylabel(r"$\langle S_c(t)\rangle$")
    ax.set_title("Clifford-point check on 100 qubits"); ax.legend(fontsize=7.5)
    plt.tight_layout(); savefig("a3_clifford"); plt.show()
    print("t      :", [meta_A[i]["t"] for i in ncl])
    for j, c in enumerate(meta_A[ncl[0]]["centers"]):
        print(f"<S_{c}> :", np.round([get(i, 'evs')[j] for i in ncl], 3))
"""))

C(md(r"""
### 3.6 Long-range correlations from raw QPU bitstrings

The Sampler shots give a full 100-bit snapshot for every run, so we can read off $\langle Z_iZ_j\rangle$ at any distance (after inverting each qubit's readout channel) and compare with the exact Pfaffians. At short distances this probes local error rates. The decay with distance is the physics: correlations spreading at a finite speed after a free-fermion quench.
"""))

C(code(r"""
if raw:
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for lab, t, c in [(l, t, c) for (l, t, c) in (("π/4", 3, "#1f77b4"), ("π/4", 8, "#2ca02c"), ("3π/8", 3, "#d62728"),
                                                     ("π/4", 2, "#1f77b4"), ("π/4", 6, "#2ca02c"), ("π/4", 20, "#d62728"))]:
        if (lab, t) not in raw:
            continue
        s = raw[(lab, t)]["spins"].astype(float)
        # measured spin = m * true spin + b, per qubit, with m = 1 - P(1|0) - P(0|1), b = P(0|1) - P(1|0)
        m_ro, b_ro = 1 - p01 - p10, p10 - p01
        z_true = (s.mean(0) - b_ro) / m_ro
        vals = []
        for r in LR:
            i, j = N_KI // 2 - r // 2, N_KI // 2 - r // 2 + r
            two = np.mean(s[:, i] * s[:, j])
            vals.append((two - m_ro[i] * b_ro[j] * z_true[i] - b_ro[i] * m_ro[j] * z_true[j] - b_ro[i] * b_ro[j])
                        / (m_ro[i] * m_ro[j]))
        ax.plot(LR, exact_ki[(lab, t)]["lr"], "-", color=c, lw=1.5)
        ax.plot(LR, vals, "o", color=c, label=f"θ_h={lab}, t={t}: raw QPU (readout-corrected)")
    ax.set_xscale("log"); ax.set_xlabel("distance r"); ax.set_ylabel(r"$\langle Z_iZ_{i+r}\rangle$")
    ax.set_title("two-point function across the 100-qubit chain (lines exact)"); ax.legend(fontsize=7.5)
    plt.tight_layout(); savefig("a4_longrange"); plt.show()
"""))

C(md(r"""
### 3.7 Trust horizon on the QPU

We call a result trustworthy at depth $t$ when the mitigated $M_z$ is within $\max(2\sigma, 0.05)$ of the exact value, where $\sigma$ is the reported standard error. For each angle we report the deepest circuit that passes. We also include a check that doesn't need the exact answer: the gap between the exponential and linear ZNE fits. Once that gap is larger than the statistical error, the extrapolation can't be trusted. Without an exact solution, this check and the Clifford-point decay (Section 3.3 on the emulator, 3.5 on the QPU) are what we would rely on.

Which agreements count: depolarising-type noise pulls every $\langle Z\rangle$ towards 0. Where the exact answer is itself close to 0 ($\theta_h=3\pi/8$ after $t\approx5$), even an unmitigated device would "agree", so agreement there means little, as the noise-model emulation above shows. The real tests are the observables with a large signal: $\theta_h=\pi/8$ ($M_z\approx0.8$–$0.9$), $\theta_h=0$ (exactly 1) and the Clifford stabilisers (exactly +1, weight up to 39). The trust horizon we quote is based on those.
"""))

C(code(r"""
if HAVE_HW:
    print(f"{'θ_h':>6s} " + " ".join(f"t={t:<4d}" for t in T_KI))
    horizon = {}
    for lab in TH:
        mz = hw[lab]["zne_z"].mean(1); sd = np.sqrt((hw[lab]["std_z"]**2).sum(1)) / N_KI
        dev = np.abs(mz - mz_exact_ki[lab]); ok = dev <= np.maximum(2 * sd, 0.05)
        horizon[lab] = max([t for t, o in zip(T_KI, ok) if o] or [0])
        print(f"{lab:>6s} " + " ".join(f"{d:6.3f}{'✓' if o else '✗'}" for d, o in zip(dev, ok)))
    print("\ntrust horizon (deepest passing t):", horizon)
    json.dump({"horizon": horizon}, open(RESULTS / "trust_horizon.json", "w"))
    if "nf_z" in hw["π/4"]:
        f = np.array([1, 3, 5]); y = hw["π/4"]["nf_z"].mean(1)                # (t, 3)
        lin = np.array([np.polyfit(f, yy, 1)[1] for yy in y])
        expo = np.array([math.exp(np.polyfit(f, np.log(np.clip(yy, 1e-3, None)), 1)[1]) for yy in y])
        print("\nθ_h=π/4  |exp − linear ZNE| by depth:", np.round(np.abs(expo - lin), 3))
"""))
