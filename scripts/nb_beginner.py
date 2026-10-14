"""Notebook cells: setup + Beginner level."""
from nbcells import md, code

CELLS = []
C = CELLS.append

C(md(r"""
# Dynamics Under Noise: from an 8-spin Trotter study to a certified 100-qubit kicked-Ising run on IBM Quantum

**Qiskit Fall Fest 2026 · IIIT-Delhi · Track 3: Quantum Simulation** · level targeted: **Advanced**

This notebook covers all three levels of the track. It runs top to bottom. The heavy noisy-simulation sweeps and every real-QPU result load from `results/` by default, so the whole notebook finishes in a few minutes. Set `RECOMPUTE = True` to regenerate the simulations, or `RUN_ON_HARDWARE = True` to resubmit to IBM Quantum.

| Level | What we do | Headline |
|---|---|---|
| 🟢 Beginner | 8-spin TFIM, Lie vs Suzuki Trotter, exact `expm` reference | For Z observables from \|0…0⟩, Lie-Trotter **equals** Suzuki-Trotter exactly. The "why it depends on what you measure" question, proven |
| 🟠 Intermediate | 12-spin TFIM, FakeFez noise, our own TREX + twirling + ZNE, depth cutting | 2nd-order accuracy at 1st-order CZ cost. An error-aware line layout with zero SWAPs |
| 🔴 Advanced | Kicked (Floquet) Ising chain, **100 qubits**, on an IBM Heron QPU | Certified against **exact** solutions at all depths: free fermions + Pfaffians, light cones and Clifford stabilisers |

**Contents**
0. Setup
1. Beginner: getting the simulation right (8 spins)
2. Intermediate: recovering the physics that noise hides (12 spins)
3. Advanced: a 100-qubit kicked-Ising chain on IBM hardware, certified without brute force
4. Performance, limitations and what we would do next
5. Credits and references
"""))

C(md(r"""
## 0 · Setup

Every helper lives in [`ising_lab.py`](ising_lab.py), next to this notebook. That way the analysis and the hardware-submission script ([`hardware/run_hardware.py`](hardware/run_hardware.py)) use exactly the same circuits. On Colab, the first cell clones the repository so the module and the cached results are available.
"""))

C(code(r"""
# On Google Colab, uncomment the next two lines first:
# !pip install -q "qiskit[visualization]==2.5.2" "qiskit-aer==0.17.2" "qiskit-ibm-runtime==0.50.0"
# !git clone -q --depth 1 https://github.com/REPO_OWNER/REPO_NAME && %cd REPO_NAME

RECOMPUTE = False         # True: re-run the noisy-simulator sweeps (~15 min) instead of loading results/
RUN_ON_HARDWARE = False   # True: resubmit the QPU jobs (needs a saved IBM Quantum account and QPU minutes)
"""))

C(code(r"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: avoid BLAS thread oversubscription
import json, math, time, warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

import qiskit, qiskit_aer, qiskit_ibm_runtime
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp, Statevector

import ising_lab as il

warnings.filterwarnings("ignore", category=DeprecationWarning)
RESULTS = Path("results"); FIGS = Path("figures"); FIGS.mkdir(exist_ok=True)
print("qiskit", qiskit.__version__, "| qiskit-aer", qiskit_aer.__version__, "| qiskit-ibm-runtime", qiskit_ibm_runtime.__version__)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
COL = {"exact": "black", "lie": "#1f77b4", "suzuki": "#ff7f0e", "strang": "#2ca02c",
       "raw": "#9e9e9e", "trex": "#8c564b", "twirl": "#9467bd", "zne": "#d62728", "hw": "#d62728"}
def savefig(name):
    plt.savefig(FIGS / f"{name}.png", dpi=160, bbox_inches="tight")
"""))

# ----------------------------------------------------------------------------- Beginner
C(md(r"""
## 1 · 🟢 Beginner: getting the simulation right (8 spins)

**Model.** Open transverse-field Ising chain, $H = -J\sum_i Z_iZ_{i+1} - h\sum_i X_i$, with $N=8$, $J=h=1$, starting from $|00000000\rangle$ and evolving to $t=2$.

**Exact reference.** We use `scipy.linalg.expm` as the challenge asks: $|\psi(t)\rangle = e^{-iHt}|\psi_0\rangle$ on the $2^8$-dimensional space.

**Product formulas.** Write $H = H_{ZZ} + H_X$. Each part is a sum of *commuting* terms, so $e^{-iH_{ZZ}\tau}$ and $e^{-iH_X\tau}$ are exact products of `RZZ` and `RX` gates. All of the Trotter error comes from splitting between the two groups:

* Lie-Trotter (1st order): $U_1(\Delta t) = e^{-iH_X\Delta t}\,e^{-iH_{ZZ}\Delta t}$
* Suzuki-Trotter (2nd order): $U_2(\Delta t) = e^{-iH_{ZZ}\Delta t/2}\,e^{-iH_X\Delta t}\,e^{-iH_{ZZ}\Delta t/2}$

These are the orderings Qiskit's `LieTrotter` / `SuzukiTrotter` build from our `SparsePauliOp`, which lists the ZZ terms first.
"""))

C(code(r"""
N, J, h, T_FINAL = 8, 1.0, 1.0, 2.0
H = il.tfim_hamiltonian(N, J, h)                     # ZZ terms first, then X terms
Mz_op, Mx_op = il.average_op(N, "Z"), il.average_op(N, "X")
Hm, Mz_m, Mx_m = H.to_matrix(), Mz_op.to_matrix(), Mx_op.to_matrix()
psi0 = np.zeros(2**N, complex); psi0[0] = 1

def exact_state(t):
    return expm(-1j * Hm * t) @ psi0

t_grid = np.linspace(0, T_FINAL, 81)
psi_exact = [exact_state(t) for t in t_grid]
mz_exact = np.array([np.real(np.vdot(p, Mz_m @ p)) for p in psi_exact])
mx_exact = np.array([np.real(np.vdot(p, Mx_m @ p)) for p in psi_exact])

# the same reference via one eigendecomposition (used later for fast dt scans) agrees with expm:
w, V = np.linalg.eigh(Hm); c0 = V.conj().T @ psi0
eig_state = lambda t: V @ (np.exp(-1j * w * t) * c0)
print("max |expm - eigh| over the grid:", max(np.abs(exact_state(t) - eig_state(t)).max() for t in t_grid))
"""))

C(md(r"""
### 1.1 Qiskit Trotter circuits, and a fast engine that reproduces them exactly

We build the circuits with `PauliEvolutionGate` and keep `.decompose(reps=2)`, so the Trotter error stays visible, as the README warns. Scanning hundreds of step sizes with circuits is slow, so `il.TFIMEngine` applies the same factors $e^{-iH_{ZZ}\tau}$ (diagonal) and $e^{-iH_X\tau}$ (one rotation per qubit) directly to a statevector. We check below that it matches the Qiskit circuits to machine precision.
"""))

C(code(r"""
def qiskit_trotter_circuit(H, n_steps, dt, order):
    synth = LieTrotter(reps=1) if order == 1 else SuzukiTrotter(order=2, reps=1)
    qc = QuantumCircuit(H.num_qubits)
    step = PauliEvolutionGate(H, time=dt, synthesis=synth)
    for _ in range(n_steps):
        qc.append(step, range(H.num_qubits))
    return qc.decompose(reps=2)          # keep the Trotter error (see README warning)

engine = il.TFIMEngine(N, J, h)
for order, scheme in ((1, "lie"), (2, "suzuki")):
    for dt, k in ((0.3, 4), (0.17, 9)):
        sv = Statevector(qiskit_trotter_circuit(H, k, dt, order)).data
        me = engine.trajectory(dt, k, scheme)[-1]
        print(f"order {order}, dt={dt}, {k} steps: 1-|<qiskit|engine>| = {1-abs(np.vdot(sv, me)):.1e}")
"""))

C(code(r"""
dt_plot = 0.2
k_plot = int(round(T_FINAL / dt_plot))
traj = {s: engine.trajectory(dt_plot, k_plot, s) for s in ("lie", "suzuki")}
t_k = dt_plot * np.arange(k_plot + 1)

fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
for col, (name, ex, f) in enumerate((("M_z", mz_exact, engine.mz), ("M_x", mx_exact, engine.mx))):
    ax, axe = axes[0, col], axes[1, col]
    ax.plot(t_grid, ex, color=COL["exact"], lw=2, label="exact (expm)")
    ref = np.interp(t_k, t_grid, ex)
    for s, lab in (("lie", "Lie-Trotter (1st)"), ("suzuki", "Suzuki-Trotter (2nd)")):
        vals = np.array([f(p) for p in traj[s]])
        ax.plot(t_k, vals, "o--", ms=4, color=COL[s], label=f"{lab}, Δt={dt_plot}")
        exact_k = np.array([f(eig_state(t)) for t in t_k])
        axe.semilogy(t_k[1:], np.abs(vals - exact_k)[1:] + 1e-16, color=COL[s])
    ax.set_ylabel(f"${name}(t)$"); ax.legend(fontsize=8)
    axe.axhline(0.01, color="grey", ls=":"); axe.set_ylabel("|error|"); axe.set_xlabel("t")
axes[0, 0].set_title("Longitudinal magnetisation  $M_z=\\frac{1}{N}\\sum\\langle Z_i\\rangle$")
axes[0, 1].set_title("Transverse magnetisation  $M_x=\\frac{1}{N}\\sum\\langle X_i\\rangle$")
plt.tight_layout(); savefig("b1_magnetisation"); plt.show()
"""))

C(md(r"""
At the same $\Delta t$ the two formulas give **identical** $M_z$ points, but very different $M_x$ points. This is not a coincidence. We explain it in §1.4.

### 1.2 Spatial correlations at $t=2$

We compute $\langle Z_0 Z_r\rangle$ (site 0 is the chain end) and the connected part $\langle Z_0Z_r\rangle-\langle Z_0\rangle\langle Z_r\rangle$, which removes the contribution of the magnetisation itself.
"""))

C(code(r"""
psi_T = exact_state(T_FINAL)
zz = np.array([engine.zz_corr(psi_T, 0, r) for r in range(N)])
z_site = np.array([float(np.sum(np.abs(psi_T)**2 * engine.z[:, i])) for i in range(N)])   # <Z_i>
conn = zz - z_site[0] * z_site
psi_T_suz = engine.trajectory(T_FINAL / 12, 12, "suzuki")[-1]
zz_suz = np.array([engine.zz_corr(psi_T_suz, 0, r) for r in range(N)])

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(range(N), zz, "o-", color=COL["exact"], label=r"$\langle Z_0Z_r\rangle$ exact")
ax.plot(range(N), zz_suz, "x", ms=9, color=COL["suzuki"], label=r"Suzuki, Δt = 1/6")
ax.plot(range(N), conn, "s--", color="#17becf", label=r"connected $\langle Z_0Z_r\rangle_c$ exact")
ax.set_xlabel("distance r"); ax.set_ylabel("correlation at t = 2"); ax.legend(fontsize=8)
ax.set_title("Correlations spread from the edge spin"); plt.tight_layout(); savefig("b2_correlations"); plt.show()
print("r      :", list(range(N)))
print("<Z0Zr> :", np.round(zz, 4))
print("conn.  :", np.round(conn, 4))
"""))

C(md(r"""
### 1.3 The largest step that keeps the error below 0.01 over the whole run

For $\Delta t = 2/n$, $n=1,2,\dots$, we compare $M_z$ and $M_x$ with the exact values at **every** step time $k\Delta t$ ($k=0..n$) and record the worst error. "Largest $\Delta t$" means the smallest $n$ whose run stays below 0.01. We also checked that every finer step stays below the threshold. We include a third ordering, field terms first ("Strang"), because it changes the answer.
"""))

C(code(r"""
# For 8 spins every factor is a 256x256 matrix, so a Trotter step is a single matrix product.
Hzz_m = SparsePauliOp.from_list([(l, c) for l, c in H.to_list() if "X" not in l]).to_matrix()
Hx_m = Hm - Hzz_m
zbar = engine.z.mean(axis=1)                                   # diagonal of M_z

def step_matrix(dt, scheme):
    Uzz = lambda tau: np.diag(np.exp(-1j * tau * np.diag(Hzz_m)))
    Ux = lambda tau: expm(-1j * tau * Hx_m)
    return {"lie": Ux(dt) @ Uzz(dt),                           # matrix order: rightmost acts first
            "suzuki": Uzz(dt / 2) @ Ux(dt) @ Uzz(dt / 2),
            "strang": Ux(dt / 2) @ Uzz(dt) @ Ux(dt / 2)}[scheme]

def worst_errors(n, scheme, T=2.0):
    dt = T / n
    U, Uex = step_matrix(dt, scheme), expm(-1j * Hm * dt)
    a, b = psi0.copy(), psi0.copy()
    ez = ex = 0.0
    for _ in range(n):
        a, b = U @ a, Uex @ b
        ez = max(ez, abs(np.sum((np.abs(a)**2 - np.abs(b)**2) * zbar)))
        ex = max(ex, abs(np.real(np.vdot(a, Mx_m @ a) - np.vdot(b, Mx_m @ b))))
    return ez, ex

N_LIST = list(range(1, 61)) + list(range(65, 205, 5))
t0 = time.time()
scan8 = {s: np.array([(n, 2.0 / n, *worst_errors(n, s)) for n in N_LIST]) for s in ("lie", "suzuki", "strang")}
print(f"scanned {3*len(N_LIST)} step sizes in {time.time()-t0:.1f}s")

def threshold(rows, col, scheme, tol=0.01):
    ok = rows[:, col] < tol
    for i in range(len(rows)):                      # first scanned n after which it stays below tol
        if ok[i:].all():
            break
    lo = int(rows[i - 1, 0]) if i > 0 else 0        # refine every n between the coarse grid points
    for n in range(lo + 1, int(rows[i, 0]) + 1):
        e = worst_errors(n, scheme)[col - 2]
        if e < tol:
            return n, 2.0 / n, e
    return int(rows[i, 0]), rows[i, 1], rows[i, col]

# two-qubit gate cost of a full run to t=2, as Qiskit builds it, and with our merged circuits
def cx_qiskit(n_steps, dt, scheme):
    HH = il.tfim_hamiltonian(N, J, h, x_first=(scheme == "strang"))
    return qiskit_trotter_circuit(HH, n_steps, dt, 1 if scheme == "lie" else 2).num_nonlocal_gates()
def cx_merged(n_steps, dt, scheme, keep_final_zz=True):
    qc = il.tfim_trotter_circuit(N, dt, n_steps, scheme)
    return 2 * qc.num_nonlocal_gates()           # each RZZ = 2 CX/CZ on hardware without fractional gates

table = []
for s in ("lie", "suzuki", "strang"):
    for col, obs in ((2, "Mz"), (3, "Mx")):
        n_, dt_, err_ = threshold(scan8[s], col, s)
        table.append((s, obs, n_, dt_, err_, cx_qiskit(n_, dt_, s), cx_qiskit(n_, dt_, s) // n_, cx_merged(n_, dt_, s)))
print(f"{'formula':8s} {'obs':3s} {'steps':>5s} {'largest dt':>10s} {'max err':>8s} {'CX (Qiskit)':>11s} {'CX/step':>7s} {'CX (merged)':>11s}")
for r in table:
    print(f"{r[0]:8s} {r[1]:3s} {r[2]:5d} {r[3]:10.4f} {r[4]:8.4f} {r[5]:11d} {r[6]:7d} {r[7]:11d}")
json.dump([dict(zip(["formula", "obs", "steps", "dt", "err", "cx_qiskit", "cx_per_step", "cx_merged"],
                    [r[0], r[1], int(r[2]), float(r[3]), float(r[4]), int(r[5]), int(r[6]), int(r[7])])) for r in table],
          open(RESULTS / "beginner_thresholds.json", "w"), indent=1)
"""))

C(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, col, obs in ((axes[0], 2, "M_z"), (axes[1], 3, "M_x")):
    for s, lab in (("lie", "Lie (1st)"), ("suzuki", "Suzuki (2nd, ZZ outside)"), ("strang", "Suzuki (2nd, X outside)")):
        rows = scan8[s]
        m = rows[:, 1] < 0.08                       # fit the convergence order on the small-dt tail
        p = np.polyfit(np.log(rows[m, 1]), np.log(rows[m, col]), 1)[0]
        ax.loglog(rows[:, 1], rows[:, col], ".-", ms=3, color=COL[s], label=f"{lab}: slope {p:.2f}",
                  lw=2.5 if s == "lie" else 1.2, zorder=1 if s == "lie" else 2)
    ax.axhline(0.01, color="grey", ls=":", label="0.01 tolerance")
    ax.set_xlabel("time step Δt"); ax.set_title(f"worst error in ${obs}$ over 0 ≤ t ≤ 2")
axes[0].set_ylabel("max |Trotter − exact|")
for ax in axes: ax.legend(fontsize=8, loc="lower right")
axes[0].text(0.012, 0.3, "Lie (thick blue) lies exactly\nunder Suzuki for $M_z$", fontsize=8, color=COL["lie"])
plt.tight_layout(); savefig("b3_dt_scan"); plt.show()
"""))

C(md(r"""
### 1.4 Which formula is cheaper, and why the answer depends on what you measure

Read the gate sequences in time order, starting from $|0\cdots0\rangle$:

* Lie: $\;ZZ(\Delta t)\,X\,ZZ\,X\cdots ZZ\,X$
* Suzuki: $\;ZZ(\tfrac{\Delta t}{2})\,X\,ZZ(\Delta t)\,X\cdots X\,ZZ(\tfrac{\Delta t}{2})$

Two boundary facts decide everything:

1. **The first ZZ layer does nothing.** $|0\cdots0\rangle$ is an eigenstate of every $Z_iZ_{i+1}$, so that layer only adds a global phase.
2. **A final ZZ layer cannot change any Z-basis observable**, since it commutes with $Z_i$ and $Z_iZ_j$.

Strip those two pieces and **Lie and Suzuki apply exactly the same operator sequence for $M_z$ and $\langle Z_0Z_r\rangle$**. So Lie-Trotter is effectively second order for Z observables: the fitted slope is 2, not 1. For $M_x$ the final half-step $ZZ(\Delta t/2)$ *does* matter. It is precisely the correction that cancels the leading error term, so Lie stays first order (slope 1) and needs a ~10× smaller step. Ordering the field terms first ("Strang") moves the half-steps onto the X layers, which flips the roles: $M_x$ becomes the easy observable and $M_z$ the hard one.

**Cost.** Qiskit's `SuzukiTrotter` emits two ZZ half-layers per step, so it costs 2× the CX gates of Lie per step (column "CX/step"). Neighbouring half-steps of consecutive Suzuki steps merge, though. Our `tfim_trotter_circuit` merges them and drops the trivial first layer, so a 2nd-order run costs the **same CX count as a 1st-order run** (column "CX (merged)"). The cheapest correct choice:

* Z-only observables: Lie-Trotter (identical to Suzuki here, with fewer gates).
* $M_x$, or both observables at once: merged Suzuki-Trotter with $\Delta t = 2/13$, about 13× fewer CX gates than Lie at $\Delta t = 2/132$.
"""))

C(code(r"""
# the claim "Lie == Suzuki for every Z-basis observable" holds to machine precision:
for dt, k in ((0.5, 4), (0.2, 10), (0.1, 20)):
    a = engine.trajectory(dt, k, "lie"); b = engine.trajectory(dt, k, "suzuki")
    dz = max(abs(engine.mz(x) - engine.mz(y)) for x, y in zip(a, b))
    dzz = max(abs(engine.zz_corr(x, 0, 5) - engine.zz_corr(y, 0, 5)) for x, y in zip(a, b))
    dx = max(abs(engine.mx(x) - engine.mx(y)) for x, y in zip(a, b))
    print(f"dt={dt}:  max|ΔMz| = {dz:.1e}   max|Δ<Z0Z5>| = {dzz:.1e}   but max|ΔMx| = {dx:.3f}")
"""))
