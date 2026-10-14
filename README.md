# GitBoosters · Qiskit Fall Fest 2026 · Track 3: Quantum Simulation

| | |
|---|---|
| **Team** | GitBoosters |
| **Members** | Pinaksh Kumar, pinakshkr@gmail.com · Keyur Desai, Keyurkumar25071@iiitd.ac.in |
| **Track** | 3 · Quantum Simulation (*Dynamics Under Noise*) |
| **Level reached** | Intermediate (complete), plus the Advanced analysis on a 100-qubit noise model built from IBM Heron calibration data. We did not get to run on a real QPU; see [Hardware runs](#hardware-runs). |
| **Code files** | [`track3_dynamics_under_noise.ipynb`](track3_dynamics_under_noise.ipynb) (main notebook, runs top to bottom in about 5 min), [`ising_lab.py`](ising_lab.py) (helper functions), [`hardware/run_hardware.py`](hardware/run_hardware.py) (QPU dry-run / submit / fetch), [`scripts/`](scripts/) (cached sweeps, notebook and slide builders) |
| **Slides** | [`slides.pdf`](slides.pdf) (8 slides) |

## Summary

We simulated transverse-field and kicked Ising chains at three sizes.

With 8 spins we found that, for Z-basis observables measured from |0…0⟩, Lie-Trotter and Suzuki-Trotter give exactly the same circuit, so the cheapest "accurate enough" product formula depends on what you measure. We show why and confirm it numerically: the error converges with slope 2.00 for M<sub>z</sub> and 1.02 for M<sub>x</sub> with Lie-Trotter.

With 12 spins and FakeFez noise, we first made the circuit shorter (second-order accuracy for the CZ cost of first order, on a qubit line that needs no SWAPs). Then our own TREX and zero-noise extrapolation code brought the M<sub>x</sub> error down about 3×. We also found that on a noisy device a finer Trotter step stops helping, because the error is set by the noise.

For 100 qubits we compiled a kicked (Floquet) Ising chain with up to 1881 CZ gates onto a 100-qubit line of IBM's 156-qubit Heron r2 layout (the FakeFez target), with no SWAPs. We ran it on a noise model built from the calibration of that line and mitigated it. Every result is compared with an exact answer that does not need brute force: free-fermion Pfaffians at any depth, light-cone windows, and Clifford-point stabilisers.

## Approach

**Beginner (8 spins, J = h = 1, t ≤ 2)**
- Exact reference from `scipy.linalg.expm`.
- Circuits from `PauliEvolutionGate` with `LieTrotter` and `SuzukiTrotter(order=2)`, kept decomposed so that the Trotter error stays visible.
- To scan step sizes quickly we wrote a small matrix engine, checked against the Qiskit circuits to 1e-14, and scanned 264 step sizes. For each formula and observable we found the largest Δt that keeps the error below 0.01 for the whole run, and its CX cost.
- The observable dependence comes from the two ends of the gate sequence: the first ZZ layer is only a phase on |0…0⟩, and a ZZ layer at the end commutes with Z.

**Intermediate (12 spins, FakeFez, a snapshot of IBM Heron r2)**
- Shorter circuits: brickwork ZZ ordering (two-qubit depth per step does not grow with N), merged Suzuki half-steps, the trivial first layer removed, and an error-aware qubit line from a beam search that minimises −log ESP using calibration data. This gives 176 CZ and two-qubit depth 32, compared with 322 and 322 for Qiskit's default `SuzukiTrotter`.
- Noise model: a compact Pauli model from FakeFez calibration (gate errors and asymmetric readout), compared with the full `AerSimulator.from_backend(FakeFez())` model. While doing this we noticed that in Aer 0.17.2, statevector simulation with shot branching silently drops readout errors, so we turned shot branching off.
- Mitigation we wrote ourselves: TREX (readout bit-flip twirling plus calibration), Pauli twirling (which, as we show, has no effect on Pauli noise), and ZNE by CZ folding (λ = 1, 3, 5) with linear, exponential and Richardson extrapolation.
- Trotter step under noise: we swept the step size and found that the raw error levels off for Δt ≈ 0.25–0.4, and finer steps make it worse (0.105 at Δt = 0.167 vs 0.069 at 0.25). After ZNE, all Δt ≤ 0.4 agree within shot noise. So the step the Trotter-only criterion asks for, Δt = 2/13 ≈ 0.15 (286 CZ), costs 62% more CZs than Δt = 0.25 (176 CZ) and gains nothing.

**Advanced (kicked Ising, 100 qubits)**
- Circuits: U = Π R<sub>X</sub>(θ<sub>h</sub>) · Π R<sub>ZZ</sub>(−π/2), with each RZZ compiled to a single CZ and the trivial first layer removed. θ<sub>h</sub> ∈ {0, π/8, π/4, 3π/8}, t up to 20 (1881 CZ, two-qubit depth 38).
- No routing: every compiled circuit has exactly (t−1)·99 CZ gates, which shows the router added no SWAPs.
- Exact references without brute force, checked against each other to about 1e-14:
  1. Free fermions: Jordan-Wigner Majorana covariance evolution, plus an extra "ghost" spin that turns ⟨Z<sub>i</sub>⟩ into a Pfaffian. Exact at any depth, 0.3 s at N = 100.
  2. Light-cone windows: one (2t+2)-qubit statevector gives the values on all 100 sites.
  3. Clifford stabilisers: ⟨U Z<sub>c</sub> U<sup>†</sup>⟩ = +1.
- Noisy 100-qubit runs: at the Clifford points (θ<sub>h</sub> = 0, π/2) we simulate the whole noisy 100-qubit circuit exactly with a stabiliser simulator, including the CZ-folded circuits for ZNE. For the other angles we simulate noisy (2t+1)-qubit light-cone windows at 5 positions along the line. Compared with full-chain density-matrix simulation these agree to within 4×10⁻⁴, which is at most 4% of the (small) noise effect and far below the 0.05 threshold we use. The small difference comes from how single-qubit gates are compiled at the edge of the window.
- Mitigation and trust: TREX + ZNE, a trust horizon, and consistency checks that do not need the exact answer.

## Results

| Quantity | Our result | Reference | Difference |
|---|---|---|---|
| Largest Δt with error < 0.01, M<sub>z</sub> (Lie = Suzuki) | 0.1667 (12 steps) | exact `expm` | max err 0.0098 |
| Largest Δt with error < 0.01, M<sub>x</sub>: Lie / Suzuki | 0.0152 / 0.1818 | exact `expm` | 1848 vs 308 CX (Qiskit build) |
| Lie − Suzuki for M<sub>z</sub> and ⟨Z<sub>0</sub>Z<sub>5</sub>⟩ | ≤ 7e-16 | identical by construction | — |
| 12-spin circuit to t = 2: CZ / 2q depth / ESP | 176 / 32 / 0.50 | Qiskit default: 322 / 322 / 0.30 | 45% fewer CZ, 10× lower 2q depth |
| 12-spin M<sub>x</sub>, mean abs error vs noiseless circuit: raw → TREX + ZNE | 0.040 → 0.013 | noiseless Trotter circuit | about 3× lower |
| 100-qubit θ<sub>h</sub> = 0, t = 20 (1881 CZ): M<sub>z</sub> raw → TREX + ZNE | 0.78 → 0.96 | exact 1 | 0.035 |
| 100-qubit Clifford stabilisers, TREX + ZNE: weight 7 / weight 39 | 0.96 / −0.03 | exact +1 | 0.037 (passes) / 1.03 (fails, signal below the noise floor) |
| 100-qubit bulk ⟨Z⟩, θ<sub>h</sub> = π/8, π/4, 3π/8, t ≤ 7: max \|TREX+ZNE − exact\| | 0.043 | free-fermion exact | raw up to 0.13 |
| Trust horizon (\|ZNE − exact\| ≤ 0.05) on the 100-qubit emulator | θ<sub>h</sub> = 0: t = 20; non-Clifford angles: t = 7 (deepest we emulated); stabilisers: weight ≤ 7 | — | depends on the observable's weight |

![Trotter scan](figures/b3_dt_scan.png)
![Mitigation ladder](figures/i1_mitigation_ladder.png)
![100-qubit emulation](figures/a6_emulation.png)

## How we checked our results

- Beginner: compared with `scipy.linalg.expm` at every time step. The matrix engine was checked against the Qiskit `PauliEvolutionGate` circuits to 1e-14.
- Intermediate: we compared the exact dynamics, the noiseless Trotter circuit, our compact noise model and the full FakeFez noise model. The compact model agrees with the full model to within 0.002–0.02 on the raw M<sub>z</sub>.
- Advanced: the three exact methods agree with each other at N = 100 (about 1e-14) and with full statevector simulation at N = 12. Every noisy and mitigated value is compared with its exact value. Two more checks don't need the exact answer: whether the exponential and linear ZNE fits agree, and how the Clifford-point stabilisers decay. We also point out which agreements are weak evidence: when the exact answer is close to 0, noise pushes the result there anyway.

## Hardware runs

None. Creating the IBM Quantum account required a payment-card identity check that we could not complete before the deadline. Everything for a real-QPU run is ready and has been validated against the FakeFez target:

- `python hardware/run_hardware.py submit` picks the least-busy Heron device, finds the best 100-qubit line from that day's calibration, and submits three Open Plan jobs (no sessions):
  - A: `EstimatorV2` with TREX, gate twirling, XpXm dynamical decoupling and gate-folding ZNE (52 kicked Ising and 7 Clifford circuits).
  - B: the same options on the 12-spin TFIM circuits.
  - C: `SamplerV2` raw bitstrings plus readout calibration.
- We estimate about 4 of the 10 free QPU minutes (not measured). `submit --quick` runs a smaller Advanced subset (100-qubit circuits at t = 2–20 and 2 Clifford checks) in about 1–2 minutes.
- The script refuses to run on any non-Open (paid) instance.
- `fetch` saves the results to `results/hardware_*.npz`, and Sections 2.5 and 3.4–3.7 of the notebook then fill in automatically.

## Limitations

- No real-QPU data. Our noise model has depolarising gate errors and readout errors, but no coherent errors, crosstalk, leakage or drift. Real hardware would decay faster, so the ZNE trust horizon would be shorter. Twirling and dynamical decoupling, which target exactly those effects, are part of the QPU plan.
- The 100-qubit model is classically easy. Integrability is what lets us check the 100-qubit results exactly, and it is also why this particular model is easy for a classical computer. With a longitudinal field added, only the light-cone (t ≲ 11), Clifford and consistency checks would remain.
- ZNE by gate folding makes circuits 3–5× deeper. For the deepest circuits the λ = 5 signal is close to the noise floor. The residual at t = 20 (0.035, which is 3–5σ) is a systematic bias of the exponential fit, and the gap between the exponential and linear fits shows it. PEA or PEC would be the next thing to try.
- Emulator statistics: the Clifford-point runs use 400 shots per circuit. The light-cone windows average 200 noise trajectories and have no measurement shot noise.

## How to run

- Python version: 3.12 (tested on Linux; Colab also works)
- Get the code: `git clone https://github.com/wireshark10/IBM-Qiskit-Hackathon && cd IBM-Qiskit-Hackathon`
- Install: `pip install -r requirements.txt`
- Run: open `track3_dynamics_under_noise.ipynb` and run all cells (about 5 minutes; the cached sweeps load from `results/`).
  - To regenerate the caches: `python scripts/intermediate_sim.py` (about 20 min) and `python scripts/emulate_advanced.py z0|stab|windows` (about 25 min each).
  - To rebuild the slides: `python scripts/make_slides.py`.
  - For the QPU: `python hardware/run_hardware.py dry-run | submit | status | fetch`.

## Credits

- Track tutorial: adapted from the BasQ Qiskit Fall Fest 2026 tutorials by Benjamin Tirado (text CC BY 4.0, code Apache 2.0). Challenge prompt by Angel Rodriguez, Benjamin Tirado and Miguel Angel Jimenez, as adapted by Quantica, IIIT-Delhi.
- Papers:
  - Kim *et al.*, Nature 618, 500 (2023)
  - Childs *et al.*, PRX 11, 011020 (2021)
  - van den Berg *et al.*, PRA 105, 032620 (2022) (TREX)
  - Temme, Bravyi and Gambetta, PRL 119, 180509 (2017) (ZNE)
  - Bertini, Kos and Prosen, PRL 123, 210601 (2019)
  - Wimmer, ACM TOMS 38, 30 (2012) (Pfaffians)
- AI tools used: Claude (Anthropic), as a coding and writing assistant for drafting helper functions and notebook text and for debugging. All results were checked numerically in the notebook, and the team reviewed all of the analysis.
