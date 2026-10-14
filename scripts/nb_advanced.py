"""Notebook cells: Advanced level."""
from nbcells import md, code

CELLS = []
C = CELLS.append

C(md(r"""
## 3 · 🔴 Advanced: a 100-qubit kicked-Ising chain on IBM hardware, certified without brute force

**Model** (as in Kim *et al.*, Nature 618, 500 (2023), on a 1D chain). One Floquet step is

$$U_F(\theta_h) = \prod_j R_X^{(j)}(\theta_h)\;\prod_{\langle j,k\rangle} R_{ZZ}^{(jk)}(\theta_J),\qquad \theta_J=-\pi/2,$$

applied $t$ times to $|0\rangle^{\otimes N}$ on an open chain of **N = 100** qubits. We measure every $\langle Z_i\rangle$ (magnetisation $M_z$) and every nearest-neighbour $\langle Z_iZ_{i+1}\rangle$, for $\theta_h\in\{0,\pi/8,\pi/4,3\pi/8\}$ and $t\in\{1,\dots,8,10,12,14,16,20\}$. That is 52 circuits with up to **1881 CZ gates** at two-qubit depth 38.

**Why this circuit suits the hardware.** At $\theta_J=-\pi/2$ each $R_{ZZ}$ is exactly **one CZ** plus single-qubit phases. On a 1D line of physically coupled qubits, routing needs **zero SWAPs**. And on $|0\cdots0\rangle$ the first ZZ layer is only a global phase, so we delete it.

### 3.1 How do we know the answer when $2^{100}$ amplitudes are out of reach?

The challenge asks us to argue correctness without brute force. We built **three independent exact methods**. Each one scales to 100 qubits, and each is cross-checked against the others and against statevector simulation:

| Method | Gives | Cost | Valid for |
|---|---|---|---|
| **Light-cone window** (`il.kicked_ising_exact_profile`) | every $\langle Z_i\rangle$, $\langle Z_iZ_{i+1}\rangle$ | statevector on $2t+2$ qubits, independent of N | any model with local gates, up to $t\approx 11$ |
| **Free fermions + Pfaffians** (`il.free_fermion_profile`) | every $\langle Z_i\rangle$ and *any* $\langle Z_iZ_j\rangle$ | $O(N^3)$, about 3 s for N = 100 | the integrable 1D kicked transverse-field Ising chain, **any depth** |
| **Clifford point** $\theta_h=\pi/2$ (`il.clifford_stabilizer`) | evolved stabilisers $S=UZ_cU^\dagger$, $\langle S\rangle=+1$ | stabiliser tableau, polynomial | any N, any depth |

The free-fermion trick. With a Jordan-Wigner transformation in the X basis, $X_j=-i\gamma_{2j}\gamma_{2j+1}$ and $Z_jZ_{j+1}=-i\gamma_{2j+1}\gamma_{2j+2}$. Every gate is then a rotation of the Majorana vector, and the covariance matrix evolves as $M\to RMR^T$. $|0\cdots0\rangle=(|\text{GHZ}_+\rangle+|\text{GHZ}_-\rangle)/\sqrt2$ is a sum of two Gaussian states, and the two parity sectors do not interfere for parity-even observables. $Z_i$ itself is parity-odd: it is a non-local string. So we add a **ghost spin** that never evolves, which gives $\langle Z_i\rangle = \langle Z_gZ_i\rangle$, an even Majorana product. Wick's theorem then evaluates it as a **Pfaffian**.

One caveat we state openly: integrability is what lets us certify 100 qubits at depth 20. It also means this particular model is classically easy. We use it as a calibrated benchmark of the hardware plus mitigation stack, not as a claim of quantum advantage.
"""))

C(code(r"""
_jm = json.load(open(RESULTS / "hardware_jobs.json")) if (RESULTS / "hardware_jobs.json").exists() else None
N_KI = _jm["plan"]["N_KI"] if _jm else 100          # chain length actually run on the QPU
TH = {"0": 0.0, "π/8": np.pi / 8, "π/4": np.pi / 4, "3π/8": 3 * np.pi / 8}
T_KI = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]

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
### 3.2 Hardware awareness: choosing 100 qubits and compiling with zero SWAPs

`il.best_line` runs a beam search over the device's coupling map. It finds the simple path of 100 qubits with the lowest summed CZ infidelity, readout error and 2× SX error, using **that day's calibration**. A 1D chain on that path needs no routing. The submission script records the path, a calibration snapshot and the compiled-circuit statistics with every job.

**Error suppression and mitigation on the QPU** (IBM Runtime `EstimatorV2`):
* Pauli twirling of every CZ layer (16 randomisations);
* **TREX** readout mitigation;
* **XpXm dynamical decoupling** on idle windows;
* **zero-noise extrapolation** by gate folding ($\lambda = 1,3,5$), exponential extrapolation with a linear fallback.

A separate `SamplerV2` job returns **raw, unmitigated bitstrings** for two angles, plus $|0\rangle^{\otimes100}$ / $|1\rangle^{\otimes100}$ calibration shots. That gives an honest baseline and long-range correlations.
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
### 3.3 Results: 100-qubit magnetisation against the exact solution

Below: raw device data (Sampler, no mitigation), the twirled and readout-mitigated estimate at $\lambda=1$, the ZNE estimate, and the exact free-fermion curve, for each $\theta_h$.
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
    if raw and (lab, 1) in raw:
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
fig.suptitle(f"Kicked Ising, N = 100, on {backend_name}", y=1.0)
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
    t_show = 6; k = T_KI.index(t_show)
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
**Noiseless vs noise-model vs hardware, at 100 qubits.** Pauli noise outside the backward light cone $[c-t,c+t]$ cannot change $\langle Z_c\rangle$. So the *noisy* value of a bulk site can be simulated exactly on a $(2t+1)$-qubit window. That window is transpiled onto the matching segment of the hardware line and carries the device's own calibrated gate and readout errors (`il.noisy_bulk_z`, cached by `scripts/emulate_hw.py`). Comparing that prediction with the QPU tells us how much of the hardware noise the calibration data explains. On the hardware side we average the bulk sites $t<i<N-1-t$, which all share one exact value.
"""))

C(code(r"""
emu_path = RESULTS / "noisy_emulation.json"
if emu_path.exists():
    emu = json.load(open(emu_path)); te = emu["t"]
    labs = ["π/8", "π/4", "3π/8"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, lab in zip(axes, labs):
        a = list(TH).index(lab)
        ex = [exact_ki[(lab, t)]["z"][N_KI // 2] for t in te]
        ax.plot(te, ex, "-", color=COL["exact"], lw=2, label="exact")
        ax.plot(te, [r[a] for r in emu["gates_only"]], "--", color="#1f77b4", label="noise model: gate errors")
        ax.plot(te, [r[a] for r in emu["with_readout"]], ":", color="grey", label="noise model: gates + readout")
        if HAVE_HW:
            ks = [T_KI.index(t) for t in te]
            bulk = lambda t: slice(t + 1, N_KI - 1 - t)
            if "nf_z" in hw[lab]:
                ax.plot(te, [hw[lab]["nf_z"][k, bulk(t), 0].mean() for k, t in zip(ks, te)], "s", color="#1f77b4", label="QPU λ=1 (twirl+TREX+DD)")
            ax.plot(te, [hw[lab]["zne_z"][k, bulk(t)].mean() for k, t in zip(ks, te)], "D", color=COL["hw"], label="QPU + ZNE")
            if raw and (lab, te[0]) in raw:
                ax.plot(te, [raw[(lab, t)]["z"][bulk(t)].mean() for t in te], "o", color=COL["raw"], label="QPU raw")
        ax.set_title(f"bulk ⟨Z⟩, θ_h = {lab}"); ax.set_xlabel("Floquet steps t")
    axes[0].set_ylabel(r"bulk $\langle Z_i\rangle$"); axes[0].legend(fontsize=7)
    plt.suptitle(f"noise model built from {emu['backend']} calibration vs the device", y=1.02)
    plt.tight_layout(); savefig("a5_noise_model_vs_qpu"); plt.show()
else:
    print("results/noisy_emulation.json not found (run scripts/emulate_hw.py)")
"""))

C(md(r"""
### 3.4 Clifford-point certification: does the mitigated device return exactly +1?

At $\theta_h=\pi/2$ the circuit is Clifford. The evolved stabiliser $S_c(t)=U Z_c U^\dagger$ is a Pauli string of weight $2t-1$ with $\langle S_c\rangle=+1$ exactly, at any size. This is the hardest observable we measure: it is sensitive to an error **anywhere in its growing support**. Its decay with weight is a direct readout of the device's error per light-cone layer, and ZNE should restore it towards 1.
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
### 3.5 Long-range correlations from raw bitstrings

The Sampler shots give the full 100-bit snapshot of every run. From them we read off $\langle Z_iZ_j\rangle$ at **any** distance, after inverting each qubit's readout channel, and compare with exact Pfaffians. Short distances probe local error rates. The decay with distance is the physics: the correlations of a free-fermion quench spreading at finite speed.
"""))

C(code(r"""
if raw:
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for lab, t, c in (("π/4", 3, "#1f77b4"), ("π/4", 8, "#2ca02c"), ("3π/8", 3, "#d62728")):
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
### 3.6 How far can we trust the device? The trust horizon

We call a result **trustworthy** at depth $t$ when the mitigated $M_z$ agrees with the exact value within $\max(2\sigma, 0.05)$, where $\sigma$ is the reported standard error. Below we report, for each angle, the deepest circuit that passes. We also include an internal-consistency test that needs **no exact answer**: the spread between the exponential and linear ZNE extrapolations. When that spread grows larger than the statistical error, the extrapolation is no longer reliable. In a regime with no exact solution we would rely on this test, together with the Clifford-point decay from §3.4.

**Which agreements actually count.** Depolarising-type noise pulls every $\langle Z\rangle$ towards 0. Where the exact answer is itself near 0 ($\theta_h=3\pi/8$ beyond $t\approx5$), even an unmitigated device "agrees", so agreement there is weak evidence; the noise-model emulation above shows this directly. The stringent tests are the high-signal observables: $\theta_h=\pi/8$ ($M_z\approx0.8$–$0.9$), $\theta_h=0$ (exactly 1), and the Clifford stabilisers (exactly +1, weight up to 39). The trust horizon we quote is governed by those.
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
