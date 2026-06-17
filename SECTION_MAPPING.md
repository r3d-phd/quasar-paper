# Merged SiliQun Paper — Section Mapping Reference

**Merged title:** *SiliQun: A Software-Defined Quantum Firmware Framework for Silicon Spin Qubits with Deep Reinforcement Learning Validation*

**Target journals:**
1. *npj Quantum Information* (IF 6.6) — primary
2. *Quantum Science & Technology* (IF 5.6) — secondary
3. *IEEE Transactions on Quantum Engineering* (IF 4.0) — fallback

**Source papers:**
- **[P1]** `QUASAR_Paper1_Full_Manuscript.md` — SiliQun simulator / QUASAR v9.51 DRL result paper (~3,942 words)
- **[P6]** `paper6-siliqun-framework/*.tex` — SiliQun Framework / PGIRS methodology paper (~745 lines LaTeX, partially written)

---

## Section-by-Section Mapping Table

| Merged Section | Merged Subsection | Primary Source | Secondary Source | Action Required |
|---|---|---|---|---|
| Abstract | — | [P6] abstract (framework framing) | [P1] abstract (F≥0.99 result) | **Rewrite** — merge both into 250 words |
| §1 Introduction | Firmware gap + SDQF concept | [P6 §1, ¶1–3] | — | **Adapt** — use P6 intro as base |
| §1 Introduction | Fault-tolerance motivation | [P1 §1, ¶1–4] | — | **Integrate** — insert P1 motivation ¶ |
| §1 Introduction | Dual contribution statement | [NEW] | — | **Write** — framework + result framing |
| §1 Introduction | Contributions list | [P6 §1 contributions] | [P1 §1 contributions] | **Merge** — 6-item unified list |
| §2 Related Work | §2.1 RL Environments | [P6 §2.1] | [P1 §2 ¶1–4 gate synthesis] | **Augment** — insert P1 gate-synthesis ¶ into P6 §2.1 |
| §2 Related Work | §2.2 Pulse-Level Control | [P6 §2.2] | — | **Use as-is** |
| §2 Related Work | §2.3 Noise Frameworks | [P6 §2.3] | — | **Use as-is** |
| §2 Related Work | §2.4 Oxford DPhil (Orbell 2024) | [P6 §2.4] | — | **Use as-is** |
| §2 Related Work | §2.5 Fault-Tolerant RL | [P6 §2.5] | — | **Use as-is** |
| §2 Related Work | §2.6 Summary | [P6 §2.6] | — | **Use as-is** |
| §3 SiliQun Framework | §3.1 SDQF Architecture | [P6 §1 intro material] | [NEW formal definition] | **Write** — expand firmware gap into full subsection |
| §3 SiliQun Framework | §3.2 Lindblad Physics Engine | [P1 §3.2 SiMOS env] | [P6 physics paragraphs] | **Write** — merge P1 SiMOS + P6 Lindblad formalism |
| §3 SiliQun Framework | §3.3 Gymnasium Interface + MDP | [P1 §3.1 problem formulation] | [P6 §3 interface — to write] | **Write** — MDP (S,A,R,γ,T) + env design |
| §3 SiliQun Framework | §3.4 PGIRS Methodology | [P6 §3 PGIRS — to write] | — | **Write** — 5-phase engineering discipline |
| §3 SiliQun Framework | §3.5 Analytical Validation | [P6 abstract validation results] | [NEW figures] | **Write** — Rabi, T1 decay, Bell fidelity |
| §4 QUASAR v9.51 | §4.1 Agent Architecture | [P1 §3.3] | — | **Adapt** — reframe as "control plane agent" |
| §4 QUASAR v9.51 | §4.2 BRFD Reward Discovery | [P1 §3.3 BRFD] | — | **Adapt** — use P1 content |
| §4 QUASAR v9.51 | §4.3 DEHB Governance | [P1 §3.3 DEHB] | — | **Adapt** — use P1 content |
| §4 QUASAR v9.51 | §4.4 Curriculum + Warm-Start | [P1 §3.4] | — | **Adapt** — use P1 content |
| §5 Experiments | §5.1 Setup | [P1 §4.1] | [NEW reproducibility] | **Adapt** — add SiliQun version, seed, GitHub |
| §5 Experiments | §5.2 Primary Result (F≥0.99) | [P1 §4.2–4.6] | — | **Use as-is** — core result unchanged |
| §5 Experiments | §5.3 Cross-Algorithm Benchmark | [P6 abstract cross-algo] | [NEW full table] | **Write** — 7 DRL families × fidelity table |
| §5 Experiments | §5.4 Cross-Backend Validation | [P6 §5.3 — already written] | — | **Use as-is** (`\input{sec53_cross_backend_validation}`) |
| §5 Experiments | §5.5 Noise Robustness | [P1 §4.7 E5] | — | **Adapt** — use P1 E5 content |
| §6 Discussion | §6.1 F≥0.99 Significance | [P1 §5.1] | — | **Use as-is** |
| §6 Discussion | §6.2 Positioning vs Related Work | [P1 §5.2] | [P6 §2.4 Orbell] | **Merge** — combine P1 comparison + Oxford discussion |
| §6 Discussion | §6.3 PGIRS Generalisability | [NEW] | — | **Write** — beyond SiMOS applicability |
| §6 Discussion | §6.4 Limitations + Future Work | [P1 §5.3] | [NEW sim-to-hardware gap] | **Augment** — add simulation-to-hardware gap |
| §7 Conclusion | — | [P1 §5.4] | [P6 §1 last ¶] + [NEW ANDROMEDA] | **Rewrite** — unified conclusion |
| References | — | [P1 refs] + [P6 §2 bibitems] | [NEW QuantumExecutor] | **Merge + deduplicate** (~50–60 refs) |

---

## Content Reuse vs. New Writing Estimate

| Category | Word Estimate | Status |
|---|---|---|
| Sections usable as-is (P6 §2 related work, §5.4 cross-backend) | ~2,500 words | ✅ Written |
| Sections adapted from P1 (§4 agent, §5.1–5.2, §5.5, §6.1, §6.4) | ~3,000 words | ✅ Exists in P1, needs reframing |
| Sections requiring new writing (§3 framework, §3.5 validation, §6.3 generalisability) | ~2,500 words | ❌ Not yet written |
| Abstract + Introduction (merged rewrite) | ~1,000 words | ❌ Not yet written |
| **Total estimated manuscript** | **~9,000 words** | Target: 8,000–10,000 words (npj QI) |

---

## Writing Priority Order

The most efficient writing sequence for the merged manuscript is:

1. **§3.2 Lindblad Physics Engine** — highest scientific density, draws from both P1 and P6, anchors the framework claim
2. **§3.1 SDQF Architecture** — conceptual framing that everything else references
3. **§3.3 Gymnasium Interface + MDP** — technical specification, largely mechanical
4. **§3.4 PGIRS Methodology** — methodology section, P6's unique contribution
5. **§3.5 Analytical Validation** — three benchmarks + figures
6. **§5.3 Cross-Algorithm Benchmark** — table + brief analysis
7. **§6.3 PGIRS Generalisability** — discussion, can be written last
8. **Abstract + Introduction** — write last once all sections are stable
9. **References** — merge and deduplicate P1 + P6 reference lists

---

## Key Narrative Decisions for the Merged Paper

**The unifying thesis sentence** (to appear in the introduction):
> "SiliQun instantiates the data plane of the SDQF architecture — a physics-grounded Gymnasium environment calibrated to SiMOS hardware — and we validate it by demonstrating the first fault-tolerance-threshold gate synthesis result (F ≥ 0.99) on a four-qubit GHZ state under realistic silicon spin qubit noise."

**The framing shift from P1 to merged:**
- P1 framing: "We present QUASAR v9.51, which achieves F≥0.99."
- Merged framing: "We present SiliQun, a framework that makes such results reproducible and extensible; QUASAR v9.51 is the first agent we validate on it."

**The framing shift from P6 to merged:**
- P6 framing: "We present SiliQun as an open-source framework."
- Merged framing: "We present SiliQun and demonstrate its scientific utility through a comprehensive validation campaign including a fault-tolerance-threshold DRL result."

---

## Journal-Specific Formatting Notes

| Journal | Page limit | Column format | Ref style | Open access fee |
|---|---|---|---|---|
| npj Quantum Information | No hard limit (~8–12 pages) | Single column | Nature-style numbered | ~$3,200 (waivable) |
| Quantum Science & Technology | 20 pages | Single column | IoP numbered | ~$2,800 |
| IEEE TQE | 10 pages (regular) | Double column | IEEE | ~$2,000 |

**Recommendation:** Submit to npj QI first (highest IF, single-column, no hard page limit). If rejected, reformat for QST. IEEE TQE as reliable fallback.
