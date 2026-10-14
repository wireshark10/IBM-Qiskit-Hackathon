"""ising_lab: helpers for Qiskit Fall Fest 2026 Track 3 (Dynamics Under Noise).

Everything the notebook needs that is not a one-off plot lives here, so the
notebook stays readable and the hardware-submission script uses exactly the
same circuits as the analysis.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector

# ----------------------------------------------------------------------------
# Pauli-string helpers (Qiskit is little-endian: qubit 0 is the rightmost char)
# ----------------------------------------------------------------------------

def pauli_label(n: int, ops: dict[int, str]) -> str:
    s = ["I"] * n
    for q, p in ops.items():
        s[q] = p
    return "".join(reversed(s))


def single_site_ops(n: int, p: str = "Z") -> list[SparsePauliOp]:
    return [SparsePauliOp(pauli_label(n, {i: p})) for i in range(n)]


def nn_zz_ops(n: int) -> list[SparsePauliOp]:
    return [SparsePauliOp(pauli_label(n, {i: "Z", i + 1: "Z"})) for i in range(n - 1)]


def average_op(n: int, p: str = "Z") -> SparsePauliOp:
    return SparsePauliOp.from_list([(pauli_label(n, {i: p}), 1.0 / n) for i in range(n)])


# ----------------------------------------------------------------------------
# Transverse-field Ising model  H = -J sum Z_i Z_{i+1} - h sum X_i  (open chain)
# ----------------------------------------------------------------------------

def tfim_hamiltonian(n: int, J: float = 1.0, h: float = 1.0, x_first: bool = False) -> SparsePauliOp:
    """TFIM as a SparsePauliOp. ``x_first`` puts the field terms first, which
    changes the Trotter ordering (and the cost of Suzuki-Trotter, see notebook)."""
    zz = [(pauli_label(n, {i: "Z", i + 1: "Z"}), -J) for i in range(n - 1)]
    xx = [(pauli_label(n, {i: "X"}), -h) for i in range(n)]
    return SparsePauliOp.from_list(xx + zz if x_first else zz + xx)


def exact_trajectory(H: SparsePauliOp, observables: dict[str, SparsePauliOp], times, psi0=None):
    """Exact <O>(t) via dense diagonalisation (fine up to ~14 qubits)."""
    n = H.num_qubits
    Hm = H.to_matrix()
    evals, evecs = np.linalg.eigh(Hm)
    psi0 = np.zeros(2**n, complex) if psi0 is None else psi0
    if not np.any(psi0):
        psi0[0] = 1.0
    c0 = evecs.conj().T @ psi0
    mats = {k: (o.to_matrix(sparse=True)) for k, o in observables.items()}
    out = {k: [] for k in observables}
    for t in times:
        psi = evecs @ (np.exp(-1j * evals * t) * c0)
        for k, m in mats.items():
            out[k].append(float(np.real(np.vdot(psi, m @ psi))))
    return {k: np.array(v) for k, v in out.items()}


class TFIMEngine:
    """Fast exact-in-statevector application of Trotter product formulas for the
    TFIM (no Trotter error is hidden: each factor e^{-i H_ZZ tau}, e^{-i H_X tau}
    is applied exactly, as the decomposed Qiskit circuits do). Verified against
    Qiskit's LieTrotter / SuzukiTrotter circuits in the notebook.

    schemes (time order, left to right):
      'lie'      : ZZ(dt) X(dt)                  == Qiskit LieTrotter, ZZ terms first
      'suzuki'   : ZZ(dt/2) X(dt) ZZ(dt/2)       == Qiskit SuzukiTrotter(order=2), ZZ first
      'strang'   : X(dt/2) ZZ(dt) X(dt/2)        == SuzukiTrotter with the field terms first
    """

    def __init__(self, n, J=1.0, h=1.0):
        self.n, self.J, self.h = n, J, h
        idx = np.arange(2**n)
        bits = (idx[:, None] >> np.arange(n)) & 1
        z = 1 - 2 * bits                                      # z[k, i] = eigenvalue of Z_i
        self.zz_diag = -J * np.sum(z[:, :-1] * z[:, 1:], axis=1)   # H_ZZ eigenvalues
        self.z = z

    def apply_zz(self, psi, tau):
        return np.exp(-1j * tau * self.zz_diag) * psi

    def apply_x(self, psi, tau):
        # exp(-i tau (-h) X) = cos(h tau) I + i sin(h tau) X on every qubit
        c, s = math.cos(self.h * tau), 1j * math.sin(self.h * tau)
        psi = psi.reshape([2] * self.n)
        for ax in range(self.n):
            a = np.take(psi, 0, axis=ax); b = np.take(psi, 1, axis=ax)
            psi = np.stack([c * a + s * b, s * a + c * b], axis=ax)
        return psi.reshape(-1)

    def step(self, psi, dt, scheme):
        if scheme == "lie":
            return self.apply_x(self.apply_zz(psi, dt), dt)
        if scheme == "suzuki":
            return self.apply_zz(self.apply_x(self.apply_zz(psi, dt / 2), dt), dt / 2)
        if scheme == "strang":
            return self.apply_x(self.apply_zz(self.apply_x(psi, dt / 2), dt), dt / 2)
        raise ValueError(scheme)

    def trajectory(self, dt, steps, scheme, psi0=None):
        psi = np.zeros(2**self.n, complex); psi[0] = 1.0
        if psi0 is not None:
            psi = psi0.astype(complex)
        out = [psi]
        for _ in range(steps):
            psi = self.step(psi, dt, scheme)
            out.append(psi)
        return out

    def mz(self, psi):
        return float(np.sum(np.abs(psi) ** 2 * self.z.mean(axis=1)))

    def mx(self, psi):
        # <X_i> = 2 Re sum_k conj(psi_k) psi_{k xor 2^i}
        n = self.n
        idx = np.arange(2**n)
        return float(np.mean([np.real(np.vdot(psi, psi[idx ^ (1 << i)])) for i in range(n)]))

    def zz_corr(self, psi, i, j):
        return float(np.sum(np.abs(psi) ** 2 * self.z[:, i] * self.z[:, j]))


def tfim_trotter_circuit(n: int, dt: float, steps: int, scheme: str = "suzuki",
                         J: float = 1.0, h: float = 1.0, measure_basis: str | None = None,
                         drop_trivial: bool = True) -> QuantumCircuit:
    """Depth-optimised TFIM Trotter circuit, gate-for-gate equal to TFIMEngine.

    * brickwork ZZ layers (even bonds, then odd bonds): two-qubit depth 2 per
      layer, independent of n;
    * adjacent half-steps between Suzuki steps are merged, so second order
      costs the same number of RZZ layers as first order;
    * ``drop_trivial``: a leading ZZ layer acting on |0...0> is a global phase
      and is removed.
    Conventions: e^{+i h tau X} = RX(-2 h tau), e^{+i J tau ZZ} = RZZ(-2 J tau).
    """
    seq: list[tuple[str, float]] = []           # (layer type, tau) in time order
    for _ in range(steps):
        if scheme == "lie":
            seq += [("zz", dt), ("x", dt)]
        elif scheme == "suzuki":
            seq += [("zz", dt / 2), ("x", dt), ("zz", dt / 2)]
        elif scheme == "strang":
            seq += [("x", dt / 2), ("zz", dt), ("x", dt / 2)]
        else:
            raise ValueError(scheme)
    merged: list[list] = []
    for kind, tau in seq:                        # merge equal neighbouring layer types
        if merged and merged[-1][0] == kind:
            merged[-1][1] += tau
        else:
            merged.append([kind, tau])
    if drop_trivial and merged and merged[0][0] == "zz":
        merged = merged[1:]
    qc = QuantumCircuit(n)
    for kind, tau in merged:
        if kind == "zz":
            for parity in (0, 1):
                for i in range(parity, n - 1, 2):
                    qc.rzz(-2 * J * tau, i, i + 1)
        else:
            for q in range(n):
                qc.rx(-2 * h * tau, q)
    if measure_basis == "X":
        qc.h(range(n))
    if measure_basis is not None:
        qc.measure_all()
    return qc


def strang_tfim_circuit(n: int, dt: float, steps: int, J: float = 1.0, h: float = 1.0,
                        measure_basis: str | None = None) -> QuantumCircuit:
    """Second-order (Strang) Trotter circuit with the X half-steps on the outside
    and merged between steps. Each step costs ONE brickwork layer of RZZ gates,
    i.e. exactly the two-qubit cost of first-order Lie-Trotter.

    exp(-iH dt) ~ e^{-i H_X dt/2} e^{-i H_ZZ dt} e^{-i H_X dt/2}
    RX(theta)=exp(-i theta X/2): e^{+i h tau X} = RX(-2 h tau)
    RZZ(theta)=exp(-i theta ZZ/2): e^{+i J tau ZZ} = RZZ(-2 J tau)
    """
    qc = QuantumCircuit(n)
    if steps == 0:
        pass
    else:
        for q in range(n):
            qc.rx(-h * dt, q)                      # first half step
        for s in range(steps):
            for parity in (0, 1):                  # brickwork: even bonds, then odd bonds
                for i in range(parity, n - 1, 2):
                    qc.rzz(-2 * J * dt, i, i + 1)
            last = s == steps - 1
            for q in range(n):
                qc.rx(-h * dt if last else -2 * h * dt, q)
    if measure_basis == "X":
        qc.h(range(n))
    if measure_basis is not None:
        qc.measure_all()
    return qc


# ----------------------------------------------------------------------------
# Kicked (Floquet) Ising chain, as in Kim et al., Nature 618, 500 (2023)
#   U_step = prod_j RX_j(theta_h) . prod_<jk> RZZ_jk(theta_J),  theta_J = -pi/2
# ----------------------------------------------------------------------------

THETA_J = -np.pi / 2


def kicked_ising_circuit(n: int, steps: int, theta_h, theta_j: float = THETA_J,
                         drop_first_zz: bool = True, measure: bool = False) -> QuantumCircuit:
    """Brickwork kicked-Ising circuit on an open chain of ``n`` qubits.

    ``drop_first_zz``: on |0...0> the first RZZ layer only adds a global phase
    (|0..0> is an eigenstate of every Z_i Z_j), so we remove it for free.
    ``theta_h`` may be a float or a qiskit Parameter.
    """
    qc = QuantumCircuit(n)
    for s in range(steps):
        if not (drop_first_zz and s == 0):
            for parity in (0, 1):
                for i in range(parity, n - 1, 2):
                    qc.rzz(theta_j, i, i + 1)
        for q in range(n):
            qc.rx(theta_h, q)
    if measure:
        qc.measure_all()
    return qc


def kicked_ising_lightcone_z(n: int, steps: int, theta_h: float, theta_j: float = THETA_J,
                             sites=None) -> np.ndarray:
    """EXACT <Z_i> after ``steps`` Floquet steps for ANY chain length ``n``.

    Light-cone argument: in the Heisenberg picture one Floquet step grows the
    support of an operator by at most one site on each side, so <Z_i>(t)
    depends only on qubits [i-t, i+t]. We simulate exactly that window (clipped
    at the physical chain ends) with a statevector. Cost ~ 2^(2t+1), independent
    of n. By translation symmetry all bulk sites (distance >= t from both ends)
    share one value, so only ~t+2 windows are ever simulated.
    """
    sites = range(n) if sites is None else sites
    cache: dict[tuple, float] = {}
    out = []
    for i in sites:
        lo, hi = max(0, i - steps), min(n - 1, i + steps)
        key = (i - lo, hi - lo + 1)          # (position in window, window length)
        if key not in cache:
            w = key[1]
            qc = kicked_ising_circuit(w, steps, theta_h, theta_j)
            sv = Statevector(qc)
            cache[key] = float(np.real(sv.expectation_value(SparsePauliOp(pauli_label(w, {key[0]: "Z"})))))
        out.append(cache[key])
    return np.array(out)


def kicked_ising_exact_profile(n: int, steps: int, thetas, theta_j: float = THETA_J,
                               precision: str = "double"):
    """EXACT site-resolved <Z_i> and bond-resolved <Z_i Z_{i+1}> of an n-site
    kicked-Ising chain for every theta_h in ``thetas``, from ONE statevector
    simulation of a (2t+2)-qubit edge window per theta, for any n.

    The backward light cone of Z_k after t steps is [k-t, k+t]. In the
    left-edge window [0, 2t+1], every site k <= t+1 (and bond k <= t) has its
    whole light cone inside the window, so it takes its true value. Sites with
    k >= t never see the left edge: they carry the bulk value (translation
    symmetry). The circuit is reflection symmetric, so the right edge mirrors
    the left. Returns arrays of shape (len(thetas), n) and (len(thetas), n-1).
    """
    from qiskit_aer import AerSimulator
    thetas = np.atleast_1d(np.asarray(thetas, float))
    full = n <= 2 * steps + 2
    w = n if full else 2 * steps + 2
    nz = w if full else steps + 2
    nzz = w - 1 if full else steps + 1
    th = Parameter("th")
    qc = kicked_ising_circuit(w, steps, th, theta_j)
    for k in range(nz):
        qc.save_expectation_value(SparsePauliOp(pauli_label(w, {k: "Z"})), range(w), label=f"z{k}")
    for k in range(nzz):
        qc.save_expectation_value(SparsePauliOp(pauli_label(w, {k: "Z", k + 1: "Z"})), range(w), label=f"zz{k}")
    sim = AerSimulator(method="statevector", precision=precision)
    res = sim.run([qc.assign_parameters({th: x}) for x in thetas]).result()
    Z = np.zeros((len(thetas), n)); ZZ = np.zeros((len(thetas), n - 1))
    for a in range(len(thetas)):
        d = res.data(a)
        left = np.array([d[f"z{k}"] for k in range(nz)]).real
        lzz = np.array([d[f"zz{k}"] for k in range(nzz)]).real
        if full:
            Z[a], ZZ[a] = left, lzz
            continue
        Z[a] = left[-1]; Z[a, :nz] = left; Z[a, n - nz:] = left[::-1]
        ZZ[a] = lzz[-1]; ZZ[a, :nzz] = lzz; ZZ[a, n - 1 - nzz:] = lzz[::-1]
    return Z, ZZ


def kicked_ising_lightcone_zz(n: int, steps: int, theta_h: float, theta_j: float = THETA_J) -> np.ndarray:
    """EXACT nearest-neighbour <Z_i Z_{i+1}> via the same light-cone window [i-t, i+1+t]."""
    cache: dict[tuple, float] = {}
    out = []
    for i in range(n - 1):
        lo, hi = max(0, i - steps), min(n - 1, i + 1 + steps)
        key = (i - lo, hi - lo + 1)
        if key not in cache:
            w = key[1]
            qc = kicked_ising_circuit(w, steps, theta_h, theta_j)
            sv = Statevector(qc)
            cache[key] = float(np.real(sv.expectation_value(
                SparsePauliOp(pauli_label(w, {key[0]: "Z", key[0] + 1: "Z"})))))
        out.append(cache[key])
    return np.array(out)


# ----------------------------------------------------------------------------
# Free-fermion (Jordan-Wigner) exact solution for <Z_i Z_{i+1}> at ANY depth.
# The 1D kicked transverse-field Ising chain is integrable. With JW strings in
# the X basis:  X_j = -i g_{2j} g_{2j+1},  Z_j Z_{j+1} = -i g_{2j+1} g_{2j+2}.
# Every gate is a Gaussian unitary (a rotation of the Majorana vector), so the
# Majorana covariance matrix evolves as M -> R M R^T: poly(n) cost, exact.
# |0..0> = (|GHZ+> + |GHZ->)/sqrt2 mixes two parity sectors; for parity-even
# observables such as Z_j Z_{j+1} the sectors do not interfere, so
# <O> = (<O>_+ + <O>_-)/2 with both GHZ states Gaussian.
# ----------------------------------------------------------------------------

def _rot(M2n: int, a: int, b: int, phi: float) -> np.ndarray:
    R = np.eye(M2n)
    c, s = math.cos(phi), math.sin(phi)
    R[a, a] = c; R[b, b] = c; R[a, b] = -s; R[b, a] = s
    return R


def free_fermion_zz(n: int, steps: int, theta_h: float, theta_j: float = THETA_J,
                    sign_conv: tuple[int, int] = (1, 1)) -> np.ndarray:
    """Exact <Z_i Z_{i+1}> for i = 0..n-2 after ``steps`` Floquet steps.

    Validated against statevector/light-cone results in the notebook (the sign
    conventions in ``sign_conv`` are fixed by that check).
    """
    m = 2 * n
    sz, sx = sign_conv
    res = np.zeros(n - 1)
    for parity in (+1, -1):
        # covariance M_ab = (i/2) <[g_a, g_b]>; <-i g_a g_b> = M_ab for a != b
        M = np.zeros((m, m))
        for j in range(n - 1):              # Z_j Z_{j+1} = +1 on GHZ states
            a, b = 2 * j + 1, 2 * j + 2
            M[a, b] = 1.0; M[b, a] = -1.0
        a, b = 0, m - 1                     # zero-mode pair fixed by parity
        M[a, b] = parity; M[b, a] = -parity
        for s in range(steps):
            if s > 0:
                R = np.eye(m)
                for j in range(n - 1):
                    R = _rot(m, 2 * j + 1, 2 * j + 2, sz * theta_j) @ R
                M = R @ M @ R.T
            R = np.eye(m)
            for j in range(n):
                R = _rot(m, 2 * j, 2 * j + 1, sx * theta_h) @ R
            M = R @ M @ R.T
        res += 0.5 * np.array([M[2 * j + 1, 2 * j + 2] for j in range(n - 1)])
    return res


def pfaffian(A: np.ndarray) -> float:
    """Pfaffian of a real antisymmetric matrix (Parlett-Reid with pivoting)."""
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    if n % 2:
        return 0.0
    pf = 1.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + int(np.argmax(np.abs(A[k + 1:, k])))
        if kp != k + 1:                       # swap rows/cols k+1 <-> kp
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            pf = -pf
        if A[k + 1, k] == 0.0:
            return 0.0
        pf *= A[k, k + 1]
        if k + 2 < n:
            tau = A[k, k + 2:] / A[k, k + 1]
            A[k + 2:, k + 2:] += np.outer(tau, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + 1], tau)
    return pf


def leading_pfaffians(B: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Pf(B[:2m, :2m]) for m = 1..n/2 from ONE skew-symmetric elimination.

    The first m elimination steps only touch the leading 2m x 2m block, so the
    running product of pivots is the leading sub-Pfaffian. No pivoting is
    possible here, so if a pivot is ~0 we fall back to ``pfaffian`` (with
    pivoting) for the remaining blocks.
    """
    A = np.array(B, dtype=float, copy=True)
    n = A.shape[0]
    out = np.zeros(n // 2)
    pf = 1.0
    for m, k in enumerate(range(0, n - 1, 2)):
        piv = A[k, k + 1]
        if abs(piv) < tol:
            for mm in range(m, n // 2):
                out[mm] = pfaffian(B[: 2 * (mm + 1), : 2 * (mm + 1)])
            return out
        pf *= piv
        out[m] = pf
        if k + 2 < n:
            tau = A[k, k + 2:] / piv
            A[k + 2:, k + 2:] += np.outer(tau, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + 1], tau)
    return out


def _givens_layer(M: np.ndarray, planes, phi: float) -> np.ndarray:
    """M -> R M R^T for a product of commuting Givens rotations on disjoint planes."""
    c, s = math.cos(phi), math.sin(phi)
    for a, b in planes:
        ra, rb = M[a].copy(), M[b].copy()
        M[a], M[b] = c * ra - s * rb, s * ra + c * rb
        ca, cb = M[:, a].copy(), M[:, b].copy()
        M[:, a], M[:, b] = c * ca - s * cb, s * ca + c * cb
    return M


def free_fermion_profile(n: int, steps: int, theta_h: float, theta_j: float = THETA_J,
                         zz_pairs=None):
    """EXACT <Z_i> for every site and <Z_i Z_j> for requested pairs, any n, any depth.

    Z_i is parity-odd (a Jordan-Wigner string), so we add one GHOST spin g at
    the left end that never evolves: <Z_g Z_i> = <Z_i> because g stays in |0>.
    Z_g Z_i = prod of nearest-neighbour bonds = an even Majorana product, whose
    Gaussian expectation is a Pfaffian of the covariance submatrix (Wick). The
    two GHZ parity sectors are averaged as in ``free_fermion_zz``.
    Returns (Z of shape (n,), dict pair->value).
    """
    ne = n + 1                                  # site 0 = ghost, physical i -> i+1
    m = 2 * ne
    zz_pairs = list(zz_pairs or [])
    Z = np.zeros(n)
    ZZ = {p: 0.0 for p in zz_pairs}
    zz_planes = [(2 * s + 1, 2 * s + 2) for s in range(1, ne - 1)]   # physical bonds only
    x_planes = [(2 * s, 2 * s + 1) for s in range(1, ne)]            # physical sites only
    for parity in (+1, -1):
        M = np.zeros((m, m))
        for s in range(ne - 1):
            a, b = 2 * s + 1, 2 * s + 2
            M[a, b], M[b, a] = 1.0, -1.0
        M[0, m - 1], M[m - 1, 0] = parity, -parity
        for st in range(steps):
            if st > 0:
                M = _givens_layer(M, zz_planes, theta_j)
            M = _givens_layer(M, x_planes, theta_h)
        # <Z_g Z_i> uses Majoranas 1 .. 2(i+1): the leading sub-Pfaffians of M[1:, 1:]
        Z += 0.5 * leading_pfaffians(M[1:2 * n + 1, 1:2 * n + 1])
        for (i, j) in ZZ:                        # unique pairs; <Z_i Z_j>, i<j: Majoranas 2(i+1)+1 .. 2(j+1)
            idx = np.arange(2 * (i + 1) + 1, 2 * (j + 1) + 1)
            ZZ[(i, j)] += 0.5 * pfaffian(M[np.ix_(idx, idx)])
    return Z, ZZ


# ----------------------------------------------------------------------------
# Clifford point theta_h = pi/2: evolved stabilisers S = U Z_c U^dagger have
# <S> = +1 exactly for any n (stabiliser simulation), a perfect hardware check.
# ----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _floquet_cliffords(n: int, theta_j: float):
    from qiskit.quantum_info import Clifford
    first = Clifford(kicked_ising_circuit(n, 1, np.pi / 2, theta_j))                     # RX layer only
    step = Clifford(kicked_ising_circuit(n, 2, np.pi / 2, theta_j, drop_first_zz=False)
                    .copy_empty_like().compose(_one_full_step(n, theta_j)))
    return first, step


def _one_full_step(n: int, theta_j: float) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for parity in (0, 1):
        for i in range(parity, n - 1, 2):
            qc.rzz(theta_j, i, i + 1)
    qc.rx(np.pi / 2, range(n))
    return qc


@lru_cache(maxsize=None)
def clifford_stabilizer(n: int, steps: int, center: int, theta_j: float = THETA_J) -> SparsePauliOp:
    """S = U Z_c U^dagger at the Clifford point theta_h = pi/2, so <S> = +1 exactly.
    Built by pushing Z_c through one Floquet-step Clifford ``steps`` times."""
    from qiskit.quantum_info import Pauli
    first, step = _floquet_cliffords(n, theta_j)
    P = Pauli(pauli_label(n, {center: "Z"})).evolve(first, frame="s")
    for _ in range(steps - 1):
        P = P.evolve(step, frame="s")
    phase = (-1j) ** P.phase
    P.phase = 0
    return SparsePauliOp(P, coeffs=[phase.real])


# ----------------------------------------------------------------------------
# Hardware awareness: choose the best 1D line of physical qubits
# ----------------------------------------------------------------------------

def _two_qubit_gate_name(target) -> str:
    for name in ("cz", "ecr", "cx"):
        if name in target.operation_names:
            return name
    raise ValueError("no 2q gate found")


def qubit_line_costs(target):
    """Per-edge 2q-gate and per-qubit readout / SX infidelities (-log fidelity) from calibration."""
    g2 = _two_qubit_gate_name(target)
    edge = {}
    for qargs, props in target[g2].items():
        if props is None or props.error is None:
            continue
        a, b = qargs
        e = min(props.error, 0.5)
        key = (min(a, b), max(a, b))
        edge[key] = min(edge.get(key, 50.0), -math.log(1 - e))
    ro, sx = {}, {}
    for (q,), props in target["measure"].items():
        e = props.error if (props is not None and props.error is not None) else 0.05
        ro[q] = -math.log(1 - min(e, 0.5))
        p1 = target["sx"].get((q,)) if "sx" in target.operation_names else None
        e1 = p1.error if (p1 is not None and p1.error is not None) else 1e-3
        sx[q] = -math.log(1 - min(e1, 0.5))
    return edge, ro, sx


def best_line(target, n: int, beam: int = 4000, cz_per_bond: float = 1.0, sx_per_qubit: float = 2.0,
              measure_weight: float = 1.0, bad=frozenset()):
    """Beam search for a simple path of ``n`` physical qubits minimising

        cost = cz_per_bond * sum_bonds -log(1-e_2q) + sum_qubits [ -log(1-e_ro) + sx_per_qubit * -log(1-e_sx) ],

    i.e. -log(ESP) of a circuit with that many gates per bond / qubit.
    Returns (path, cost)."""
    edge, ro, sx = qubit_line_costs(target)
    node = {q: measure_weight * ro[q] + sx_per_qubit * sx.get(q, 1e-3) for q in ro}
    edge = {k: cz_per_bond * v for k, v in edge.items()}
    adj: dict[int, list[int]] = {}
    for (a, b) in edge:
        if a in bad or b in bad:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    ecost = lambda a, b: edge[(min(a, b), max(a, b))]
    # state = (cost, path tuple, visited bitmask); (mask, last) identifies equivalent states
    states = [(node.get(q, 0.1), (q,), 1 << q) for q in adj]
    for _ in range(n - 1):
        new = []
        for c, path, mask in states:
            last = path[-1]
            for nb in adj[last]:
                if not (mask >> nb) & 1:
                    new.append((c + ecost(last, nb) + node.get(nb, 0.1),
                                path + (nb,), mask | (1 << nb)))
        if not new:
            return None, math.inf
        new.sort(key=lambda s: s[0])
        uniq, seen_keys = [], set()
        for st in new:
            k = (st[2], st[1][-1])
            if k in seen_keys:
                continue
            seen_keys.add(k)
            uniq.append(st)
            if len(uniq) >= beam:
                break
        states = uniq
    cost, path, _ = states[0]
    return list(path), cost


# ----------------------------------------------------------------------------
# Error mitigation building blocks we implement ourselves (used on simulators)
# ----------------------------------------------------------------------------

def fold_two_qubit_gates(isa: QuantumCircuit, factor: int, gate: str = "cz") -> QuantumCircuit:
    """Local unitary folding G -> G (G^dag G)^k for every 2q gate; factor = 2k+1.
    CZ is self-inverse, so G^dag = G. Barriers stop any later optimiser from
    cancelling the folded pairs."""
    assert factor % 2 == 1
    k = (factor - 1) // 2
    out = isa.copy_empty_like()
    for inst in isa.data:
        out.append(inst)
        if inst.operation.name == gate and k:
            for _ in range(k):
                out.barrier(inst.qubits)
                out.append(inst.operation, inst.qubits)
                out.append(inst.operation, inst.qubits)
            out.barrier(inst.qubits)
    return out


def extrapolate_zero(factors, values, kind: str = "exponential"):
    """Zero-noise extrapolation of ``values`` (shape: len(factors) x ...)."""
    f = np.asarray(factors, float)
    v = np.asarray(values, float)
    if kind == "linear":
        A = np.vstack([np.ones_like(f), f]).T
        coef, *_ = np.linalg.lstsq(A, v.reshape(len(f), -1), rcond=None)
        return coef[0].reshape(v.shape[1:])
    if kind == "richardson":
        # Lagrange polynomial through all points evaluated at 0
        out = np.zeros(v.shape[1:])
        for i, fi in enumerate(f):
            w = np.prod([fj / (fj - fi) for j, fj in enumerate(f) if j != i])
            out = out + w * v[i]
        return out
    if kind == "exponential":
        # fit v = a * exp(-b f) per entry (sign-safe: fall back to linear if signs flip)
        flat = v.reshape(len(f), -1)
        res = np.empty(flat.shape[1])
        lin = extrapolate_zero(f, flat, "linear")
        for j in range(flat.shape[1]):
            y = flat[:, j]
            if np.all(y > 1e-3) or np.all(y < -1e-3):
                s = np.sign(y[0])
                b, loga = np.polyfit(f, np.log(np.abs(y)), 1)
                res[j] = s * math.exp(loga)
                if not np.isfinite(res[j]) or abs(res[j]) > 1.5:
                    res[j] = lin[j]
            else:
                res[j] = lin[j]
        return res.reshape(v.shape[1:])
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# Our own mitigation pipeline for the noisy simulator (Intermediate level)
# ----------------------------------------------------------------------------

def physical_line(isa: QuantumCircuit, n: int) -> list[int]:
    """Physical qubit of each virtual qubit 0..n-1 after transpilation."""
    return list(isa.layout.final_index_layout(filter_ancillas=True))[:n]


def measured_variant(body: QuantumCircuit, phys: list[int], basis: str, mask) -> QuantumCircuit:
    """Append a basis change (X basis: H = RZ.SX.RZ, ISA-native), the TREX bit-flip
    mask (X before measurement, undone classically) and measurements."""
    from qiskit import ClassicalRegister
    n = len(phys)
    qc = body.copy()
    cr = ClassicalRegister(n, "meas")
    qc.add_register(cr)
    for i, p in enumerate(phys):
        if basis == "X":
            qc.rz(np.pi / 2, p); qc.sx(p); qc.rz(np.pi / 2, p)
        if mask[i]:
            qc.x(p)
        qc.measure(p, cr[i])
    return qc


def z_from_counts(counts: dict, n: int, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-qubit <Z_i> and nearest-neighbour <Z_i Z_{i+1}> from a counts dict
    (bitstring little-endian: rightmost char = virtual qubit 0), undoing a TREX mask."""
    keys = list(counts)
    bits = np.array([[int(k[n - 1 - i]) for i in range(n)] for k in keys], dtype=np.int8)
    if mask is not None:
        bits ^= np.asarray(mask, dtype=np.int8)
    w = np.array([counts[k] for k in keys], float)
    w /= w.sum()
    s = 1 - 2 * bits
    return w @ s, w @ (s[:, :-1] * s[:, 1:])


def mitigation_ladder(sim, bodies: list[QuantumCircuit], n: int, basis: str, *, K: int = 8,
                      shots: int = 2000, factors=(1, 3, 5), seed: int = 7, target=None,
                      twirl_demo: bool = False):
    """Every rung of our own mitigation ladder on a noisy simulator.

    Returns per-body arrays of <P_i> (P = the measured ``basis``):
      raw              no mitigation (``shots`` shots, no readout twirl)
      trex             K random readout bit-flip masks (TREX) + rescaling by the
                       |0..0> calibration of the same twirled measurement
      zne_<f>          TREX at CZ-folding noise factor f (f=1 is ``trex``)
      zne_lin/exp/rich extrapolated to zero noise
      twirl            (if twirl_demo) TREX + K Pauli-twirled copies of the CZ gates
    Shots: raw ``shots``; trex ``shots`` split over K masks; each f>1 ``shots//2``.
    """
    from qiskit.circuit import pauli_twirl_2q_gates
    rng = np.random.default_rng(seed)
    phys = physical_line(bodies[0], n)
    masks = [rng.integers(0, 2, n) for _ in range(K)]
    zeros = np.zeros(n, int)
    batches = []                               # (circuits, shots, tag)
    empty = bodies[0].copy_empty_like()
    batches.append(([measured_variant(empty, phys, "Z", m) for m in masks], max(1, shots // K), ("cal", None, None)))
    for b_i, body in enumerate(bodies):
        batches.append(([measured_variant(body, phys, basis, zeros)], shots, ("raw", b_i, None)))
        for f in factors:
            circ = fold_two_qubit_gates(body, f) if f > 1 else body
            sh = max(1, (shots if f == 1 else shots // 2) // K)
            batches.append(([measured_variant(circ, phys, basis, m) for m in masks], sh, ("zne", b_i, f)))
        if twirl_demo:
            tw = pauli_twirl_2q_gates(body, num_twirls=K, seed=int(rng.integers(1 << 30)), target=target)
            batches.append(([measured_variant(c, phys, basis, m) for c, m in zip(tw, masks)],
                            max(1, shots // K), ("twirl", b_i, None)))
    vals = {}
    for circs, sh, tag in batches:
        res = sim.run(circs, shots=sh, seed_simulator=int(rng.integers(1 << 30))).result()
        if tag[0] == "raw":
            vals[tag] = z_from_counts(res.get_counts(0), n)[0]
        else:
            vals[tag] = np.mean([z_from_counts(res.get_counts(i), n, masks[i])[0] for i in range(len(circs))], axis=0)
    cal = vals[("cal", None, None)]
    out = {"cal": cal, "raw": [], "trex": []}
    for f in factors:
        out[f"zne_{f}"] = []
    if twirl_demo:
        out["twirl"] = []
    for b_i in range(len(bodies)):
        out["raw"].append(vals[("raw", b_i, None)])
        for f in factors:
            out[f"zne_{f}"].append(vals[("zne", b_i, f)] / cal)
        out["trex"].append(out[f"zne_{factors[0]}"][-1])
        if twirl_demo:
            out["twirl"].append(vals[("twirl", b_i, None)] / cal)
    for k in list(out):
        out[k] = np.array(out[k])
    if len(factors) > 1:
        stack = np.array([out[f"zne_{f}"] for f in factors])        # (F, bodies, n)
        out["zne_lin"] = extrapolate_zero(factors, stack, "linear")
        out["zne_exp"] = extrapolate_zero(factors, stack, "exponential")
        out["zne_rich"] = extrapolate_zero(factors, stack, "richardson")
    return out


def compact_noise_model(backend, path: list[int]):
    """Pauli (depolarising + readout) noise model for the qubits in ``path``,
    relabelled 0..len(path)-1, built from the backend's calibration data:
    CZ / SX / X depolarising strength matched to the reported average gate
    error (lambda = r d/(d-1)), asymmetric readout errors from the properties.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    t = backend.target
    g2 = _two_qubit_gate_name(t)
    nm = NoiseModel(basis_gates=["cz", "sx", "x", "rz", "measure"] if g2 == "cz" else None)
    props = backend.properties() if hasattr(backend, "properties") else None
    for v, p in enumerate(path):
        for g in ("sx", "x"):
            e = t[g][(p,)].error or 0.0
            if e > 0:
                nm.add_quantum_error(depolarizing_error(min(e * 2, 0.75), 1), g, [v])
        p01 = p10 = t["measure"][(p,)].error or 0.0
        if props is not None:
            try:
                p10 = props.qubit_property(p, "prob_meas0_prep1")[0]   # P(0 | 1)
                p01 = props.qubit_property(p, "prob_meas1_prep0")[0]   # P(1 | 0)
            except Exception:
                pass
        nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [v])
    pos = {p: v for v, p in enumerate(path)}
    for (a, b), pr in t[g2].items():                 # every coupler inside the qubit set
        if a in pos and b in pos:
            e = (pr.error if pr is not None and pr.error is not None else 5e-3)
            nm.add_quantum_error(depolarizing_error(min(e * 4 / 3, 1.0), 2), g2, [pos[a], pos[b]])
    return nm


def compact_circuit(isa: QuantumCircuit, path: list[int]) -> QuantumCircuit:
    """Re-express an ISA circuit that only touches ``path`` on len(path) qubits."""
    pos = {p: i for i, p in enumerate(path)}
    out = QuantumCircuit(len(path))
    for creg in isa.cregs:
        out.add_register(creg)
    for inst in isa.data:
        qs = [isa.find_bit(q).index for q in inst.qubits]
        if not qs and inst.operation.name == "barrier":
            continue
        if any(q not in pos for q in qs):
            if inst.operation.name == "barrier":
                qs = [q for q in qs if q in pos]
                if not qs:
                    continue
            else:
                raise ValueError(f"{inst.operation.name} acts outside the line: {qs}")
        out.append(inst.operation, [out.qubits[pos[q]] for q in qs], inst.clbits)
    return out


class CompactSim:
    """AerSimulator wrapper: compacts each ISA circuit onto the chosen line and
    simulates it with ``compact_noise_model`` (fast Pauli trajectories)."""

    def __init__(self, backend, path, method="statevector", threads=0):
        from qiskit_aer import AerSimulator
        self.path = list(path)
        self.noise_model = compact_noise_model(backend, self.path)
        # NB: no shot branching. In Aer 0.17.2, statevector + shot_branching silently drops
        # readout errors (we caught this: P(0|1) 0.0004 measured vs 0.0108 in the model).
        self.sim = AerSimulator(noise_model=self.noise_model, method=method, max_parallel_threads=threads)

    def run(self, circuits, shots, seed_simulator=None):
        circuits = circuits if isinstance(circuits, list) else [circuits]
        return self.sim.run([compact_circuit(c, self.path) for c in circuits], shots=shots,
                            seed_simulator=seed_simulator)


def noisy_bulk_z(backend, line: list[int], steps: int, thetas, center: int | None = None,
                 trajectories: int = 400, seed: int = 1, threads: int = 0, readout: bool = True,
                 factor: int = 1):
    """Noise-model prediction of a BULK <Z_c> of the kicked-Ising chain on ``line``.

    Pauli noise outside the backward light cone [c-t, c+t] of Z_c cannot change
    <Z_c>, so we simulate only that (2t+1)-qubit window, transpiled onto the
    matching segment of the real line, with ``compact_noise_model`` (gate errors)
    and the qubit's readout channel applied analytically. Returns <Z_c> per theta
    (noise trajectories averaged; no measurement shot noise).
    """
    from qiskit_aer import AerSimulator
    from qiskit.transpiler import generate_preset_pass_manager
    c = len(line) // 2 if center is None else center
    seg = list(line[c - steps: c + steps + 1])
    w = len(seg)
    th = Parameter("th")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend, initial_layout=seg)
    isa = pm.run(kicked_ising_circuit(w, steps, th))
    if factor > 1:
        isa = fold_two_qubit_gates(isa, factor)          # ZNE noise amplification
    comp = compact_circuit(isa, seg)
    comp.save_expectation_value(SparsePauliOp(pauli_label(w, {steps: "Z"})), range(w), label="z")
    nm = compact_noise_model(backend, seg)
    sim = AerSimulator(noise_model=nm, method="statevector", max_parallel_threads=threads)
    vals = []
    for x in np.atleast_1d(thetas):
        r = sim.run(comp.assign_parameters({comp.parameters[0]: float(x)}), shots=trajectories,
                    seed_simulator=seed).result()
        vals.append(float(np.real(r.data(0)["z"])))
    ro = nm._local_readout_errors.get((steps,)) if readout else None
    if ro is not None:                                  # measured Z = m * Z + b
        P = ro.probabilities
        m, b = 1 - P[0][1] - P[1][0], P[1][0] - P[0][1]
        vals = [m * v + b for v in vals]
    return np.array(vals)


def cliffordize(circ: QuantumCircuit, tol: float = 1e-8) -> QuantumCircuit:
    """Replace every rz(k*pi/2) by its exact Clifford (I, S, Z, Sdg) so Aer's stabilizer
    method accepts the circuit. RZ is a virtual, noiseless frame change on IBM hardware,
    so this changes nothing physically. Raises if an angle is not Clifford."""
    out = circ.copy_empty_like()
    for inst in circ.data:
        op = inst.operation
        if op.name == "rz":
            th = float(op.params[0])
            k = round(th / (np.pi / 2))
            if abs(th - k * np.pi / 2) > tol:
                raise ValueError(f"non-Clifford rz({th})")
            g = {0: None, 1: "s", 2: "z", 3: "sdg"}[k % 4]
            if g:
                getattr(out, g)(inst.qubits[0])
        else:
            out.append(inst)
    return out


def kicked_ising_clifford_native(n: int, steps: int, theta_h: float, basis_ops=None,
                                 mask=None) -> QuantumCircuit:
    """Kicked Ising at a Clifford point (theta_h in {0, pi/2}) compiled by hand to native
    Clifford gates: RZZ(-pi/2) = CZ (Sdg x Sdg) and RX(pi/2) = SX, up to global phases.
    Same CZ count as the transpiled circuit, (t-1)(n-1). ``basis_ops`` maps qubit ->
    'X'/'Y'/'Z' measurement basis (default Z on all); ``mask`` is a TREX bit-flip mask."""
    assert min(abs(theta_h), abs(theta_h - np.pi / 2)) < 1e-12
    qc = QuantumCircuit(n, n)
    for s in range(steps):
        if s > 0:
            for parity in (0, 1):
                for i in range(parity, n - 1, 2):
                    qc.cz(i, i + 1); qc.sdg(i); qc.sdg(i + 1)
        if theta_h > 0:
            qc.sx(range(n))
    for q in range(n):
        b = (basis_ops or {}).get(q, "Z")
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q); qc.h(q)
        if mask is not None and mask[q]:
            qc.x(q)
        qc.measure(q, q)
    return qc


def fold_cz_logical(qc: QuantumCircuit, factor: int) -> QuantumCircuit:
    """CZ folding (CZ -> CZ (CZ CZ)^k) on a logical circuit; barriers keep the copies."""
    return fold_two_qubit_gates(qc, factor, gate="cz")
