# FSBM MAGNET evaluation

## Stunnel card

`cards/fsbm_stunnel_conformal_safety.yaml` evaluates a frozen FSBM stunnel
checkpoint with a split-conformal sampled-path certificate. The card is scoped
to eight sampled isotropic perturbation directions and 101 logged Euler states.
Its scope covers sampled directions at logged discrete times; continuous-time
and all-direction safety receive separate analyses.

The fixed evaluation configuration is:

- direction: forward
- perturbation radius: `0.1`
- conformal alpha: `0.1`
- calibration / held-out cases: `128 / 256`
- probes per case: `8`
- Euler steps: `100`
- seed: `20260731`
- device: CPU

The card verifies a zero covered-predicted-safe collision count, at least 50%
held-out retention, and retained sampled-collision risk at or below the risk
over the full held-out set.

Required frozen assets:

| Asset | Path | SHA-256 |
|---|---|---|
| stunnel checkpoint | `outputs/runs/stunnel/2026.04.13/150806/checkpoints/last.ckpt` | `57ddf061a9b20ded14a55ae838558ab3603d80298acccc51b1bf3c2cf32801fc` |
| checkpoint Hydra config | `outputs/runs/stunnel/2026.04.13/150806/.hydra/config.yaml` | `c5038b6c536ef9c8e8b84e1fe4dd58f4e8d413d930e3b30de90005f48966eb41` |
| stunnel keypoints | `data/kp_traj_stunnel.pt` | `0474437a52b70c73600dd7672ad519b99212c7d8d574f1b802e55353b6fa60f2` |

Canonical run:

```bash
python -m magnet.evaluation \
  magnet_eval/cards/fsbm_stunnel_conformal_safety.yaml \
  --output_path /path/to/aiq-evaluations/Debug/GaTech/FSBM_Stunnel_Conformal_Safety \
  --jobs 1
```

## FFHQ conformal stability and LPIPS card

`cards/fsbm_ffhq_conformal_stability_lpips.yaml` contains exactly two scoped
claims for the frozen female-to-male and adult-to-child FSBM translations:

1. split-conformal local transport-stability bounds;
2. conformalized paired LPIPS comparison against LSBM.

The card focuses exclusively on transport stability and LPIPS. The archived
semantic-proxy experiments remain available as a separate line of evidence.

### Plain-language intuition

- **Gender:** transport a female-labeled FFHQ source latent toward the
  male-labeled target distribution.
- **Age:** transport an adult source latent (age at least 35) toward the child
  target distribution (age below 15).

For both tasks, the stability experiment asks whether two nearby source latents
remain nearby after translation when their stochastic noise is shared. A small
bound means the declared input perturbations usually produce correspondingly
small latent endpoint changes. This measures local consistency. Strong local
stability can coexist with limited change in the requested target attribute,
so semantic evaluation provides complementary evidence.

The LPIPS experiment asks whether FSBM changes the source image less than LSBM
for the same archived identity. Lower source-referenced LPIPS can indicate
better source preservation or an edit that is too subtle to reach the requested
gender or age target. Interpret LPIPS together with dedicated evaluations of
edit success, face identity, realism, and human-perceived quality.

### Primary conformal stability claim

The card runs five identity-disjoint splits per task and direction with
`delta=0.1`, `alpha=0.05`, 1,000 calibration identities, four
maximum-aggregated probes, and 50 Euler steps. It verifies ordinary rank 951,
reports `S_hat` and `B=delta*S_hat`, audits held-out coverage with the exact
repeated beta-binomial law, and reports PAC rank 962 at `beta=0.05`.

| Task | Direction | Ordinary pooled coverage | Median ordinary bound | PAC pooled coverage | Median PAC bound |
|---|---:|---:|---:|---:|---:|
| gender | forward | 95.68% | 0.1181 | 96.74% | 0.1203 |
| gender | backward | 94.50% | 0.1437 | 95.82% | 0.1482 |
| age | forward | 95.43% | 0.1038 | 96.60% | 0.1045 |
| age | backward | 95.97% | 0.1016 | 97.10% | 0.1024 |

The four task-direction diagnostics use a Bonferroni threshold of `0.05/4`.
The ordinary theorem is marginal over calibration and the future score; a
particular pooled realization may land above or below 95%. This claim measures
sampled local latent sensitivity under synchronous noise. Semantic correctness
and all-direction robustness receive their own targeted evaluations.

### Paired LPIPS comparison

The card verifies pinned comparisons against LSBM on 100 matched
in-distribution source identities per task. AlexNet LPIPS v0.1 is computed at
256x256 between each source and each method output. Lower values indicate less
perceptual deviation from the source.

| Task | Mean FSBM | FSBM upper bound | Mean LSBM | LSBM upper bound | Paired upper bound | Family-adjusted paired bound |
|---|---:|---:|---:|---:|---:|---:|
| gender | 0.3236 | 0.4313 | 0.5425 | 0.6351 | -0.0881 | -0.0495 |
| age | 0.3237 | 0.4470 | 0.5376 | 0.6443 | -0.0846 | -0.0562 |

Nominal `alpha=0.05` uses rank 96 of 100. The paired task-family claim uses
`alpha=0.025` per task and rank 99, keeping the union-bound error budget at
0.05. Both paired FSBM-minus-LSBM upper bounds remain negative, supporting
better in-distribution LPIPS generalizability for FSBM on this endpoint.

All identities calibrate the LPIPS bounds, producing an order-statistic
prediction bound. LPIPS measures source-referenced perceptual change; semantic,
identity, and human-quality evaluations provide complementary evidence.

### Required frozen assets

| Asset | SHA-256 |
|---|---|
| `darpa_conformal_eval.py` | `48437c29bec9f4f02d7def91b58ddd5db639339819a9002b60c60fe9609bba2a` |
| `data/latents.npy` | `46237baaccace670b953de74d62a689b9bdacad50207f97b166c4746f79fb684` |
| `data/gender.npy` | `cd1aabbe5b0cfaa917ec94c9b6a675376ceac69b3a430644b016f36da1421538` |
| `data/age.npy` | `0662e232785ca5674bd8c1cf9699bb570ec7b6265191ee62c2d05688994bfe25` |
| gender checkpoint | `3a2278568b46b9b98dd4568bb599ff08455cc37d124045c8d12f7c8e8c0c9e1f` |
| gender checkpoint config | `e68588c7b0afe6fd7cb98e46b416ed1e584fb8445113b21992b6670a07697444` |
| age checkpoint | `18050bf31919fd81691271ab5708e9c86ae50e0cf5a6476448a7bfcfc6522551` |
| age checkpoint config | `4a9b1b5dc3d311081a20307ec1c074191d8f429e2c1818396e0b7053d0f1802e` |
| gender paired LPIPS scores | `f4b5b707f1cb981a3470d82c3891a4ee6dd74c69626b64ac49cc5adb5cd19439` |
| age paired LPIPS scores | `70bf9f64a2688cc1b90b936c2a3e47677e0e559a6b328ae2cf628c2b7f2f1c2a` |

The default checkpoints are expected in the sibling
`generalized-schrodinger-bridge-matching` checkout. Override them with
`FSBM_FFHQ_GENDER_CHECKPOINT` and `FSBM_FFHQ_AGE_CHECKPOINT`.

Canonical run:

```bash
FSBM_SCIENTIFIC_PYTHON=/path/to/fsbm-python-3.10 \
python -m magnet.evaluation \
  magnet_eval/cards/fsbm_ffhq_conformal_stability_lpips.yaml \
  --output_path /path/to/aiq-evaluations/Debug/GaTech/FSBM_FFHQ_Conformal_Stability_LPIPS \
  --jobs 1
```

MAGNET 0.0.2 requires Python 3.11 or newer, while the frozen scientific
environment uses Python 3.10. The bridge exchanges a JSON request and response
between those interpreters. Run the submission card from a clean immutable
checkout.

The card loads the learned checkpoint drift directly and reserves keypoint
trajectories for separate guidance/Hessian analyses. Interpret its claims as
local latent sensitivity and source-referenced perceptual change; semantic,
identity, artifact, fairness, and mission evaluations complete the broader
assessment.
