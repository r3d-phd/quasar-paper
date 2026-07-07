# QUASAR Version Reports

**Project:** QUASAR — Quantum-Adaptive Soft Actor-Critic for Reinforcement Learning-Based Quantum State Preparation  
**Author:** Raad Alshehri  
**Institution:** King Abdulaziz University, Faculty of Computing and Information Technology  
**Generated:** 2026-07-07  

---

## Version Lineage

```
v9 (CAMEL-Q)
  └─ v19 (ERL-SLM, 6-layer hierarchy)
       └─ v21 (LoRA fine-tuning)
            └─ v23/v24 (Extended targets)
                 └─ v25 (Noise curriculum P1-P4)
                      └─ v26 (FASTMIX-Omega + OPID)
                           └─ v27 (Fixed-HP ablation — PRIMARY)
                                └─ v28 (LC-RL — Layer-Contribution RL)
```

---

## Reports Index

| Version | Codename | Status | Report | Diagram |
|---------|----------|--------|--------|---------|
| v9 | CAMEL-Q | Completed | [v9/QUASAR_v9_Report.md](v9/QUASAR_v9_Report.md) | [diagrams/quasar_v9_arch.png](diagrams/quasar_v9_arch.png) |
| v19 | ERL-SLM | Completed | [v19/QUASAR_v19_Report.md](v19/QUASAR_v19_Report.md) | [diagrams/quasar_v19_arch.png](diagrams/quasar_v19_arch.png) |
| v26 | FASTMIX-OPID | Running (H100) | [v26/QUASAR_v26_Report.md](v26/QUASAR_v26_Report.md) | [diagrams/quasar_v26_arch.png](diagrams/quasar_v26_arch.png) |
| v27 | Ablation-Clean | 44/50 done | [v27/QUASAR_v27_Report.md](v27/QUASAR_v27_Report.md) | [diagrams/quasar_v27_arch.png](diagrams/quasar_v27_arch.png) |
| v28 | LC-RL | Ready to submit | [v28/QUASAR_v28_Report.md](v28/QUASAR_v28_Report.md) | [diagrams/quasar_v28_arch.png](diagrams/quasar_v28_arch.png) |

---

## Key Innovations Per Version

| Version | Primary Innovation | Best F (2Q) | Best F (4Q) |
|---------|-------------------|-------------|-------------|
| v9 | CAMEL-Q Hybrid Actor + T-SAC Curriculum | 0.9999 | 0.6891 |
| v19 | ERL-SLM 6-layer hierarchy + SWDFT + BRFD | 0.9999 | 0.9210 |
| v26 | FASTMIX-Omega bilevel + OPID LoRA | TBD | TBD |
| v27 | Fixed-HP ablation (noise-stage isolation) | 0.9978 | 0.9912 |
| v28 | LC-RL layer contribution scoring | TBD | TBD |

---

## v27 Ablation Results Summary (PRIMARY)

The v27 noise-stage ablation is the primary experimental contribution of the PhD thesis. All 44 completed results (out of 50) achieve best_F >= 0.97, with the majority above 0.99. This demonstrates that the fidelity ceiling in earlier versions was architecture-induced, not noise-induced.

| Noise Stage | Min best_F (all targets, both seeds) | Max best_F | Mean best_F |
|-------------|--------------------------------------|------------|-------------|
| ns=1 | 0.9723 | 0.9941 | 0.9882 |
| ns=2 | 0.9957 | 0.9979 | 0.9967 |
| ns=3 | 0.9905 | 0.9942 | 0.9926 |
| ns=4 | 0.9861 | 0.9913 | 0.9890 |
| ns=5 | 0.9842 | **0.9938** | ~0.9890 |

---

## Repository Structure

```
quasar-reports/
  README.md                    # This file
  diagrams/
    quasar_v9.d2               # D2 source
    quasar_v9_arch.png         # Rendered diagram
    quasar_v19.d2
    quasar_v19_arch.png
    quasar_v26.d2
    quasar_v26_arch.png
    quasar_v27.d2
    quasar_v27_arch.png
    quasar_v28.d2
    quasar_v28_arch.png
  v9/
    QUASAR_v9_Report.md
  v19/
    QUASAR_v19_Report.md
  v26/
    QUASAR_v26_Report.md
  v27/
    QUASAR_v27_Report.md
  v28/
    QUASAR_v28_Report.md
```
