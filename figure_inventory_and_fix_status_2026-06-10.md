# SiliQun Paper Figure Inventory and Fix Status

## Paper fix status

The SiliQun paper was fixed and recompiled successfully on 2026-06-10. The final compile produced `siliqun_tqe_v6.pdf` with:

- Zero overfull hboxes
- Zero undefined references
- Zero multiply-defined labels
- 28 pages total

The fixes were pushed to Overleaf commit `bbb5ac0`.

## Where the recently generated figures are

### Main paper figures directory

All recently generated figures used for the paper are in:

`/home/ubuntu/siliqun_paper_overleaf/figures/`

Figures generated/updated on 2026-06-10 include:

| Figure file | Path | Notes |
|---|---|---|
| `fig_noise_overhead.png` | `/home/ubuntu/siliqun_paper_overleaf/figures/fig_noise_overhead.png` | Noise model computational overhead chart |
| `fig_speed_scaling.png` | `/home/ubuntu/siliqun_paper_overleaf/figures/fig_speed_scaling.png` | Throughput and speedup comparison |
| `fig_bond_dim.png` | `/home/ubuntu/siliqun_paper_overleaf/figures/fig_bond_dim.png` | Bond-dimension throughput study |
| `fig_extension4_sac.png` | `/home/ubuntu/siliqun_paper_overleaf/figures/fig_extension4_sac.png` | Extension 4 SAC generalisation chart |
| `fig_gate_accuracy.png` | `/home/ubuntu/siliqun_paper_overleaf/figures/fig_gate_accuracy.png` | Gate fidelity validation table figure |
| `fig_comparison.png` | `/home/ubuntu/siliqun_paper_overleaf/figures/fig_comparison.png` | Comparison summary figure |
| `fig_device_profiles.png` | `/home/ubuntu/siliqun_paper_overleaf/figures/fig_device_profiles.png` | Device profile figure |
| `fig_env_throughput.png` | `/home/ubuntu/siliqun_paper_overleaf/figures/fig_env_throughput.png` | Environment throughput figure |
| `fig_architecture.png` | `/home/ubuntu/siliqun_paper_overleaf/figures/fig_architecture.png` | Architecture figure |

### Earlier generated slide/image project directory

The image-based figure generation project from the earlier batch is in:

`/home/ubuntu/siliqun_paper_figures/`

This contains the generated slide/image assets such as:

- `fig1_siliqun_arch_generated.webp`
- `fig2_pgirs_generated.webp`
- `fig3_quasar_modules_generated.webp`
- `fig4_convergence_generated.webp`
- `fig5_cross_backend_generated.webp`

## Push and sync status

| Destination | Status | Details |
|---|---|---|
| Overleaf | Success | Pushed to `origin/master` at commit `bbb5ac0` |
| GitHub | Failed | `gh` authentication returned HTTP 401 bad credentials |
| Google Drive | Failed | `rclone` token expired; reconnect required |

## Final PDF

Final compiled PDF path:

`/home/ubuntu/siliqun_paper_overleaf/siliqun_tqe_v6.pdf`

## Heartbeat report

Heartbeat report path:

`/home/ubuntu/phd-heartbeat/HEARTBEAT_2026-06-10_HB67.md`

## Notes on user-requested fixes

The following user-reported issues were addressed:

- Figure 5 QUASAR reference fixed
- Overlaps removed
- Tables resized to fit better
- Cross-references repaired
- Clean compile verified

Google Drive upload was not completed because the configured token has expired and needs manual reconnection.

GitHub push was not completed because the current `gh` authentication is invalid in this session.

