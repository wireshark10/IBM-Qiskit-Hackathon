"""Notebook cells: setup + Beginner level."""
from nbcells import md, code

CELLS = []
C = CELLS.append

C(md(r"""
# Dynamics under noise: Trotter errors, error mitigation and a 100-qubit kicked Ising chain

Team GitBoosters (Pinaksh Kumar, Keyur Desai), Qiskit Fall Fest 2026 at IIIT-Delhi, Track 3 (Quantum Simulation).

We completed the Beginner and Intermediate levels. For the Advanced level we built and checked the full 100-qubit experiment, but we could not get IBM Quantum access before the deadline, so it runs on a noise model made from IBM Heron calibration data rather than on a real QPU (details in Section 3.3).

The notebook runs top to bottom in about five minutes. The slow noisy-simulation sweeps are cached in `results/`; set `RECOMPUTE = True` to regenerate them. The hardware scripts are in `hardware/run_hardware.py`, and if hardware results are ever added to `results/`, the plots in Sections 2.5 and 3.4 to 3.7 fill in automatically.

| Level | What we did | Main result |
|---|---|---|
| Beginner | 8-spin TFIM, Lie vs Suzuki Trotter, exact reference from `expm` | For Z observables starting from \|0…0⟩, Lie-Trotter and Suzuki-Trotter give exactly the same numbers, which is why the answer depends on what you measure |
| Intermediate | 12-spin TFIM with FakeFez noise, our own TREX and ZNE code, shorter circuits | Second-order accuracy for the CZ cost of first order, on a qubit line that needs no SWAPs |
| Advanced | Kicked Ising chain on 100 qubits, compiled for IBM Heron with no SWAPs and run on a noise model from that line's calibration | Results compared with exact answers at every depth (free fermions with Pfaffians, light cones, Clifford stabilisers), including a full noisy 100-qubit simulation at the Clifford points |

Contents
0. Setup
1. Beginner: getting the simulation right (8 spins)
2. Intermediate: getting the physics back from noisy data (12 spins)
3. Advanced: a 100-qubit kicked Ising chain, checked without brute force
4. What worked, what didn't, and next steps
5. Credits and references
"""))

C(md(r"""
## 0. Setup

All helper functions are in [`ising_lab.py`](ising_lab.py), next to this notebook. We kept them in one module so the analysis here and the hardware script ([`hardware/run_hardware.py`](hardware/run_hardware.py)) build exactly the same circuits. On Colab, uncomment the lines in the next cell so the module and the cached results are available.
"""))

C(code(r"""
# On Google Colab, uncomment the next three lines first:
# !pip install -q "qiskit[visualization]==2.5.2" "qiskit-aer==0.17.2" "qiskit-ibm-runtime==0.50.0"
# !git clone -q --depth 1 https://github.com/wireshark10/IBM-Qiskit-Hackathon
# %cd IBM-Qiskit-Hackathon

RECOMPUTE = False         # True: re-run every noisy-simulator sweep instead of loading results/
                          #       (12-spin ladder ~20 min + 100-qubit emulation ~25 min x 3 parts)
# QPU runs: python hardware/run_hardware.py submit | fetch   (needs a saved IBM Quantum account)
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
## 1. Beginner: getting the simulation right (8 spins)

The model is the open transverse-field Ising chain $H = -J\sum_i Z_iZ_{i+1} - h\sum_i X_i$ with $N=8$ and $J=h=1$. We start in $|00000000\rangle$ and evolve to $t=2$.

As the challenge asks, the exact reference comes from `scipy.linalg.expm`: $|\psi(t)\rangle = e^{-iHt}|\psi_0\rangle$ on the full $2^8$-dimensional space.

For the product formulas we split $H = H_{ZZ} + H_X$. The terms inside each group commute, so $e^{-iH_{ZZ}\tau}$ and $e^{-iH_X\tau}$ are exactly one layer of `RZZ` gates and one layer of `RX` gates. All of the Trotter error therefore comes from splitting between the two groups:

* Lie-Trotter (first order): $U_1(\Delta t) = e^{-iH_X\Delta t}\,e^{-iH_{ZZ}\Delta t}$
* Suzuki-Trotter (second order): $U_2(\Delta t) = e^{-iH_{ZZ}\Delta t/2}\,e^{-iH_X\Delta t}\,e^{-iH_{ZZ}\Delta t/2}$

These are the orderings that Qiskit's `LieTrotter` and `SuzukiTrotter` produce from our `SparsePauliOp`, because we list the ZZ terms first.
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
### 1.1 Qiskit Trotter circuits and a faster engine

We build the circuits with `PauliEvolutionGate` and keep the `.decompose(reps=2)` step from the tutorial; without it the simulator applies the evolution exactly and the Trotter error disappears. Simulating circuits for hundreds of step sizes is slow, so we also wrote `il.TFIMEngine`, which applies the same two factors directly to a statevector ($e^{-iH_{ZZ}\tau}$ is diagonal and $e^{-iH_X\tau}$ is one rotation per qubit). The next cell checks that it agrees with the Qiskit circuits to machine precision.
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
With the same $\Delta t$, the two formulas give the same $M_z$ points but clearly different $M_x$ points. Section 1.4 explains why.

### 1.2 Spatial correlations at $t=2$

We look at $\langle Z_0 Z_r\rangle$, where site 0 is the end of the chain, and at the connected part $\langle Z_0Z_r\rangle-\langle Z_0\rangle\langle Z_r\rangle$, which removes the contribution of the magnetisation itself.
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
### 1.3 Largest step that keeps the error below 0.01 for the whole run

We use $\Delta t = 2/n$ for $n=1,2,\dots$ and compare $M_z$ and $M_x$ with the exact values at every step time $k\Delta t$ ($k=0,\dots,n$), keeping the worst error. The largest $\Delta t$ is then the smallest $n$ whose whole run stays below 0.01. We also checked that the finer steps on our scan grid (every $n \le 60$, then every 5th $n$ up to 200) stay below the threshold, and we go through $n$ one by one between grid points. We added a third ordering with the field terms first ("Strang") because it changes which observable is the easy one.
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
### 1.4 Which formula is cheaper, and why it depends on what you measure

Write the gate sequences in time order, starting from $|0\cdots0\rangle$:

* Lie: $\;ZZ(\Delta t)\,X\,ZZ\,X\cdots ZZ\,X$
* Suzuki: $\;ZZ(\tfrac{\Delta t}{2})\,X\,ZZ(\Delta t)\,X\cdots X\,ZZ(\tfrac{\Delta t}{2})$

Two things happen at the ends of these sequences:

1. The first ZZ layer does nothing physical. $|0\cdots0\rangle$ is an eigenstate of every $Z_iZ_{i+1}$, so this layer only adds a global phase.
2. A ZZ layer at the very end cannot change anything measured in the Z basis, because it commutes with $Z_i$ and $Z_iZ_j$.

Once those two pieces are removed, Lie and Suzuki apply exactly the same operators as far as $M_z$ and $\langle Z_0Z_r\rangle$ are concerned. So for Z observables Lie-Trotter is effectively second order, and the fitted slope is 2 rather than 1. For $M_x$ the final half-step $ZZ(\Delta t/2)$ does matter, since it is the piece that cancels the leading error term. Without it Lie stays first order (slope 1) and needs a 12× smaller step (0.0152 vs 0.182). Putting the field terms first ("Strang") moves the half-steps onto the X layers and swaps the roles: $M_x$ becomes the easy observable and $M_z$ the hard one.

On cost: Qiskit's `SuzukiTrotter` writes two ZZ half-layers per step, so per step it uses twice the CX gates of Lie (column "CX/step"). However, the half-steps of neighbouring Suzuki steps can be merged. Our `tfim_trotter_circuit` merges them and drops the trivial first layer, so a second-order run ends up with the same CX count as a first-order one (column "CX (merged)"). The cheapest choices that meet the 0.01 target are:

* Z-only observables: Lie-Trotter, which here is identical to Suzuki and uses fewer gates.
* $M_x$: merged Suzuki-Trotter with $\Delta t = 2/11$. It needs 154 CX, compared with 1834 for merged Lie at $\Delta t = 2/132$, so about 12× fewer. With Qiskit's own circuits it is 308 vs 1848, or 6× fewer.
* Both observables at once: merged Suzuki-Trotter with $\Delta t = 2/12$ (168 CX).
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
