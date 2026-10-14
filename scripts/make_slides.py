"""Build slides.pdf (8 slides, 16:9) from figures/ and results/ via headless Chrome.

    python scripts/make_slides.py
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
TEAM = json.loads((ROOT / "team.json").read_text()) if (ROOT / "team.json").exists() else {
    "team": "[Team name]", "members": ["[Member 1]", "[Member 2]"]}

jobs = json.loads((R / "hardware_jobs.json").read_text()) if (R / "hardware_jobs.json").exists() else None
horizon = json.loads((R / "trust_horizon.json").read_text())["horizon"] if (R / "trust_horizon.json").exists() else None
summary = json.loads((R / "summary.json").read_text()) if (R / "summary.json").exists() else {}
backend = jobs["backend"] if jobs else "IBM Heron QPU"


def fig(name, h=None):
    p = F / f"{name}.png"
    if not p.exists():
        return f'<div class="ph">[{name}.png: pending]</div>'
    style = f' style="max-height:{h}px"' if h else ""
    return f'<img src="{p.as_uri()}"{style}>'


def s(key, default="—"):
    return summary.get(key, default)


CSS = """
@page { size: 13.333in 7.5in; margin: 0 }
* { box-sizing: border-box }
body { margin: 0; font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; color: #161616; }
.slide { width: 13.333in; height: 7.5in; padding: 0.42in 0.55in 0.35in; page-break-after: always; position: relative; overflow: hidden; }
.slide:last-child { page-break-after: auto }
h1 { font-size: 30px; margin: 0 0 4px; letter-spacing: -0.3px }
h2 { font-size: 15px; font-weight: 500; color: #525252; margin: 0 0 14px }
.tag { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #fff; padding: 3px 9px; border-radius: 3px; margin-bottom: 8px }
.g { background: #24a148 } .o { background: #ff832b } .r { background: #da1e28 } .b { background: #0f62fe }
.cols { display: flex; gap: 22px; align-items: flex-start }
.cols > div { flex: 1 }
img { max-width: 100%; display: block }
ul { margin: 6px 0 0 18px; padding: 0; font-size: 14.5px; line-height: 1.42 }
li { margin-bottom: 5px }
table { border-collapse: collapse; font-size: 12.5px; width: 100% }
th, td { border-bottom: 1px solid #e0e0e0; padding: 4px 6px; text-align: left }
th { background: #f4f4f4 }
.tiles { display: flex; gap: 14px; margin: 18px 0 }
.tile { flex: 1; border-left: 4px solid #0f62fe; background: #f4f7ff; padding: 10px 14px }
.tile b { display: block; font-size: 26px; color: #0f62fe }
.tile span { font-size: 12.5px; color: #393939 }
.foot { position: absolute; bottom: 0.18in; left: 0.55in; right: 0.55in; font-size: 10.5px; color: #8d8d8d; display: flex; justify-content: space-between }
.key { background: #fff8e1; border-left: 4px solid #f1c21b; padding: 8px 12px; font-size: 14px; margin-top: 10px }
.ph { border: 2px dashed #c6c6c6; color: #8d8d8d; padding: 40px; text-align: center }
.small { font-size: 12.5px; color: #525252 }
"""

slides = []

slides.append(f"""
<div class="tag b">QISKIT FALL FEST 2026 · IIIT-DELHI · TRACK 3 QUANTUM SIMULATION · INTERMEDIATE + ADVANCED ANALYSIS</div>
<h1 style="font-size:40px;margin-top:16px">Dynamics Under Noise</h1>
<h2 style="font-size:20px">From an 8-spin Trotter study to a <b>certified 100-qubit kicked-Ising experiment</b> on a calibrated IBM Heron noise model</h2>
<div class="tiles">
 <div class="tile"><b>100 qubits</b><span>error-aware line on heavy-hex, <b>0 SWAPs</b></span></div>
 <div class="tile"><b>{s('max_cz', '1881')} CZ</b><span>deepest circuit, 2q depth {s('max_depth2q', '38')}, 20 Floquet steps</span></div>
 <div class="tile"><b>3 exact checks</b><span>free fermions + Pfaffians · light cone · Clifford, all at N = 100</span></div>
 <div class="tile"><b>{s('trust_headline', 'trust horizon')}</b><span>{s('trust_sub', 'deepest depth where mitigated = exact')}</span></div>
</div>
<ul>
 <li><b>Beginner:</b> for Z observables from |0…0⟩, Lie-Trotter <i>is</i> Suzuki-Trotter, exactly. Mx needs 2nd order: 12× fewer CX for 0.01 accuracy (154 vs 1834).</li>
 <li><b>Intermediate:</b> 2nd-order accuracy at 1st-order CZ cost. Our own TREX + ZNE on FakeFez noise. The noise-optimal Trotter step.</li>
 <li><b>Advanced:</b> 100-qubit kicked Ising (0 SWAPs) on a noise model from the chosen line's calibration, with TREX + ZNE, certified against exact answers at <b>every</b> depth. QPU jobs prepared and validated; no QPU access before the deadline.</li>
</ul>
<div style="margin-top:10px">{fig('a6_emulation', 255)}</div>
<div class="foot"><span>{TEAM['team']} · {', '.join(TEAM['members'])}</span><span>notebook: track3_dynamics_under_noise.ipynb</span></div>
""")

slides.append(f"""
<div class="tag g">BEGINNER · 8 SPINS · J = h = 1 · t ≤ 2</div>
<h1>Why the "right" Trotter step depends on what you measure</h1>
<div class="cols"><div style="flex:1.35">{fig('b3_dt_scan')}
<table style="margin-top:8px"><tr><th>formula</th><th>obs</th><th>largest Δt (err&lt;0.01)</th><th>CX Qiskit</th><th>CX merged</th></tr>
{s('thresholds_rows', '')}</table></div>
<div><ul>
<li>|0…0⟩ is an eigenstate of every Z<sub>i</sub>Z<sub>i+1</sub>, so the <b>first ZZ layer is a phase</b>.</li>
<li>A <b>final ZZ layer commutes</b> with every Z-basis observable.</li>
<li>Strip both boundary pieces: Lie and Suzuki apply the <b>same gates</b> for M<sub>z</sub>, ⟨Z<sub>0</sub>Z<sub>r</sub>⟩ (difference 1e-16). Lie is secretly 2nd order (slope 2.00).</li>
<li>For M<sub>x</sub> the final ZZ(Δt/2) is the correction: Lie is 1st order (slope 1.02), Δt = 0.015 vs 0.18.</li>
<li>Merging adjacent half-steps: 2nd order at <b>1st-order CX cost</b>.</li>
</ul><div class="key">Cheapest correct choice: Lie for Z-only; merged Suzuki (Δt = 2/11, 154 CX) for M<sub>x</sub>; Δt = 2/12 (168 CX) for both.</div></div></div>
<div class="foot"><span>exact reference: scipy.linalg.expm · circuits: PauliEvolutionGate + LieTrotter / SuzukiTrotter</span><span>§1</span></div>
""")

slides.append(f"""
<div class="tag o">INTERMEDIATE · CUT DEPTH BEFORE YOU MITIGATE</div>
<h1>Hardware-aware compilation: every CZ removed is error you never mitigate</h1>
<div class="cols"><div>
<table><tr><th>12-spin TFIM to t = 2 on FakeFez</th><th>CZ</th><th>2q depth</th><th>SWAPs</th><th>ESP</th></tr>
{s('compile_rows', '')}</table>
<ul style="margin-top:14px">
<li><b>Brickwork ordering</b>: even bonds, then odd. 2q depth per step is constant in N.</li>
<li><b>Merged half-steps</b> and a <b>dropped trivial first layer</b>.</li>
<li><b>Error-aware line</b>: beam search over the coupling map minimising Σ CZ + readout + SX infidelity from live calibration. Finds a 100-qubit path on 156-qubit Heron.</li>
<li>Kicked Ising at θ<sub>J</sub> = −π/2: each RZZ is <b>one CZ</b>. 100 qubits, 20 steps: {s('max_cz','1881')} CZ, 0 SWAPs.</li>
</ul></div>
<div>{fig('i2_tradeoff')}
<div class="key">Noise-limited plateau: for Δt ≈ 0.25–0.4 the raw error is flat (≈0.07), and finer steps make it worse (0.105 at Δt = 0.167). After ZNE all Δt ≤ 0.4 agree within shot noise, so the Trotter-only Δt ≈ 0.15 buys nothing and costs 50% more CZ.</div></div></div>
<div class="foot"><span>ESP = Π(1−ε_gate)·Π(1−ε_readout) from FakeFez calibration</span><span>§2.1, §2.4</span></div>
""")

slides.append(f"""
<div class="tag o">INTERMEDIATE · 12 SPINS · FakeFez NOISE · OUR OWN MITIGATION CODE</div>
<h1>Recovering the physics that noise hides</h1>
<div class="cols"><div style="flex:1.5">{fig('i1_mitigation_ladder')}{fig('i3_hardware_tfim', 230)}</div>
<div><ul>
<li><b>TREX</b>: random X before readout, undone classically, then rescaled by a twirled |0…0⟩ calibration.</li>
<li><b>ZNE</b>: every CZ folded to 3 or 5 CZs. Linear / exponential / Richardson to λ → 0.</li>
<li><b>Twirling</b> is a no-op on Pauli noise (we check it). It matters on hardware, where errors are coherent.</li>
<li>Compact noise model checked against the full FakeFez model.</li>
</ul>
<table style="margin-top:10px"><tr><th>mean |error| vs exact</th><th>M<sub>z</sub></th><th>M<sub>x</sub></th></tr>{s('ladder_rows','')}</table>
</div></div>
<div class="foot"><span>same 8 circuits re-run on the QPU (Job B) for the noisy-sim vs hardware comparison</span><span>§2.2–2.5</span></div>
""")

slides.append(f"""
<div class="tag r">ADVANCED · HOW DO WE KNOW THE ANSWER AT 100 QUBITS?</div>
<h1>Three independent exact certificates. No brute force.</h1>
<table style="font-size:14px"><tr><th>method</th><th>what it gives</th><th>cost</th><th>valid</th></tr>
<tr><td><b>Free fermions + Pfaffians</b><br><span class="small">JW in X basis, a ghost spin turns Z<sub>i</sub> into an even Majorana string; Wick → Pfaffian</span></td><td>every ⟨Z<sub>i</sub>⟩, any ⟨Z<sub>i</sub>Z<sub>j</sub>⟩</td><td>O(N³) · 0.3 s at N = 100, t = 20</td><td>integrable 1D kicked TFIM, <b>any depth</b></td></tr>
<tr><td><b>Light-cone window</b><br><span class="small">⟨Z<sub>k</sub>(t)⟩ depends only on [k−t, k+t]; one (2t+2)-qubit edge window + reflection gives all 100 sites</span></td><td>every ⟨Z<sub>i</sub>⟩, ⟨Z<sub>i</sub>Z<sub>i+1</sub>⟩</td><td>2<sup>2t+2</sup>, independent of N</td><td>any local circuit, t ≲ 11</td></tr>
<tr><td><b>Clifford point</b> θ<sub>h</sub> = π/2<br><span class="small">S = U Z<sub>c</sub> U<sup>†</sup>, weight 2t−1 Pauli</span></td><td>⟨S⟩ = +1 exactly</td><td>stabiliser tableau</td><td>any N, any depth</td></tr></table>
<div class="cols" style="margin-top:14px"><div><ul>
<li>Cross-validated: free-fermion vs light-cone at N = 100 agree to <b>~1e-14</b>, and both match statevector at N = 12.</li>
<li>Kicked Ising (Kim <i>et al.</i>, Nature 2023) on a 1D chain: U<sub>F</sub> = Π R<sub>X</sub>(θ<sub>h</sub>) · Π R<sub>ZZ</sub>(−π/2).</li>
<li>52 circuits: θ<sub>h</sub> ∈ {{0, π/8, π/4, 3π/8}} × t ∈ {{1…8, 10, 12, 14, 16, 20}}, plus 7 Clifford circuits.</li>
</ul></div><div><div class="key">Honest framing: integrability is <i>why</i> we can certify 100 qubits at depth 20. We use it as a calibrated benchmark of the device + mitigation stack, not as a claim of quantum advantage.</div></div></div>
<div class="foot"><span>ising_lab.free_fermion_profile · kicked_ising_exact_profile · clifford_stabilizer</span><span>§3.1</span></div>
""")

slides.append(f"""
<div class="tag r">ADVANCED · 100 QUBITS · CALIBRATED IBM HERON NOISE MODEL</div>
<h1>The 100-qubit experiment, noisy, mitigated, and checked against exact answers</h1>
<div class="cols"><div style="flex:2.1">{fig('a6_emulation', 520)}</div>
<div><ul>
<li><b>Clifford points</b> (θ<sub>h</sub> = 0, π/2): a stabiliser simulation of the <b>entire</b> noisy 100-qubit circuit is exact, so we run it at full size up to 20 steps / 1881 CZ, including folded ZNE circuits.</li>
<li><b>Other angles</b>: noise outside the light cone cannot matter, so we simulate exact (2t+1)-qubit windows at 5 positions on the line.</li>
<li>Noise: per-gate CZ/SX errors + asymmetric readout of the <i>chosen</i> 100 qubits (ibm_fez snapshot).</li>
</ul>
<div class="key">{s('emu_takeaway', 'TREX + ZNE restores the exact values; see notebook §3.3.')}</div></div></div>
<div class="foot"><span>no QPU access before the deadline (account verification); jobs are ready in hardware/run_hardware.py</span><span>§3.3</span></div>
""")

slides.append(f"""
<div class="tag r">ADVANCED · TRUST HORIZON · QPU-READY</div>
<h1>How far can we trust it, and how would we know without an exact answer?</h1>
<div class="cols"><div>{fig('a1_mz_100q', 430)}</div>
<div><ul>
<li><b>Trust horizon</b> (|ZNE − exact| ≤ 0.05): {s('horizon_text', 'see notebook')}.</li>
<li><b>Which agreements count</b>: noise pulls ⟨Z⟩ towards 0, so θ<sub>h</sub> = 3π/8 (exact ≈ 0) "agrees" for free. High-signal observables (θ<sub>h</sub> = 0, π/8, Clifford stabilisers) are the real test.</li>
<li><b>Without an exact answer</b>: exponential vs linear ZNE spread, Clifford-point decay and light-cone windows (t ≲ 11).</li>
<li><b>QPU-ready</b>: 3 jobs (EstimatorV2 with TREX + twirling + XpXm DD + ZNE; SamplerV2 raw + readout calibration), PUBs validated, ≈ 4 Open Plan minutes. Every CZ count equals (t−1)·99, proving 0 SWAPs.</li>
</ul><div class="key"><b>The horizon depends on observable weight.</b> Local ⟨Z⟩ at 1881 CZ is recovered to 0.96 (exact 1), but a Pauli spread over 15+ qubits decays below the shot-noise floor at λ = 5 and ZNE cannot bring it back: weight-7 stabilisers pass, weight-11 and beyond fail.</div></div></div>
<div class="foot"><span>exact M<sub>z</sub>(t) for the 4 angles on the left; free-fermion Pfaffians, 0.3 s each</span><span>§3.2–3.7</span></div>
""")

slides.append(f"""
<div class="tag b">LIMITATIONS · NEXT STEPS · REPRODUCIBILITY</div>
<h1>Where it works, where it breaks, what we would do next</h1>
<div class="cols"><div><ul>
<li><b>Classically easy by design</b>: integrability lets us certify. A longitudinal field h<sub>z</sub> breaks it, leaving only light cone + Clifford + consistency checks.</li>
<li><b>Gate-folding ZNE</b> multiplies depth by 3–5. At t ≥ 16 (1500+ CZ) λ = 5 nears the noise floor and the fits disagree.</li>
<li><b>No real-QPU data</b>: the noise model has no coherent errors, crosstalk or drift. Real hardware would decay faster. Twirling and DD target exactly those effects.</li>
<li><b>Next</b>: PEA-ZNE with learned layer noise (only 2 distinct CZ layers), fractional RZZ for the TFIM (halves depth), a non-integrable chain.</li>
</ul></div>
<div><table><tr><th>prepared QPU job</th><th>primitive</th><th>content</th></tr>
<tr><td>A</td><td>EstimatorV2: TREX, twirl, DD, ZNE</td><td>52 kicked-Ising + 7 Clifford circuits, 100 qubits</td></tr>
<tr><td>B</td><td>EstimatorV2: same options</td><td>8 TFIM circuits, 12 qubits</td></tr>
<tr><td>C</td><td>SamplerV2 (raw)</td><td>26 circuits + readout calibration</td></tr></table>
<ul style="margin-top:14px">
<li>Everything in one notebook that runs top to bottom (~5 min). QPU and long sweeps are cached in <code>results/</code>.</li>
<li><code>hardware/run_hardware.py</code>: dry-run / submit / fetch with the exact same circuits.</li>
<li>Tutorial credit: BasQ Qiskit Fall Fest 2026 (B. Tirado). AI assistance: Claude, for coding and writing, with every number verified in-notebook.</li>
</ul></div></div>
<div class="foot"><span>{TEAM['team']}</span><span>Qiskit Fall Fest 2026 · IIIT-Delhi</span></div>
""")

html = "<!doctype html><html><head><meta charset='utf-8'><title>Track 3 slides</title><style>" + CSS + "</style></head><body>"
html += "".join(f'<section class="slide">{x}</section>' for x in slides) + "</body></html>"
(ROOT / "scratch").mkdir(exist_ok=True)
out_html = ROOT / "scratch" / "slides.html"
out_html.write_text(html, encoding="utf-8")
pdf = ROOT / "slides.pdf"
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", out_html.as_uri()], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", pdf, "with", len(slides), "slides")
