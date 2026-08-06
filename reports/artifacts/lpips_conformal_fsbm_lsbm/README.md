# Paired FSBM–LSBM source-LPIPS audit

This is the corrected FSBM-versus-LSBM analysis.  FSBM outputs come from the
keypoint-enabled generation runs; LSBM outputs and source images come from the
matched full runs.  For every comparison, all 100 source PNGs were verified to
match the corresponding FSBM run byte-for-byte by sample ID.

LPIPS uses AlexNet v0.1, RGB inputs area-resized from 1024x1024 to 256x256,
and the source image as the reference.  With `n=100` and `alpha=0.05`, the
finite-sample conformal rank is `ceil(101 * 0.95) = 96`.

| Direction | q_FSBM | q_LSBM | q_delta | FSBM lower |
|---|---:|---:|---:|---:|
| gender p0 to p1 | 0.431324 | 0.635138 | -0.088115 | 100/100 |
| gender p1 to p0 | 0.411526 | 0.627935 | -0.101543 | 100/100 |
| age p0 to p1 | 0.447029 | 0.644325 | -0.084556 | 100/100 |
| age p1 to p0 | 0.441252 | 0.617675 | -0.085235 | 100/100 |

Here `delta = LPIPS(FSBM) - LPIPS(LSBM)`.  Every upper paired threshold is
negative, so under exchangeability the conformal procedure certifies lower
FSBM source-LPIPS than LSBM at 95% marginal coverage for each direction.

This metric quantifies source-image perceptual change.  It is not a certificate
of requested-attribute success, identity preservation, or overall translation
quality.  All available identities are used for calibration, so no separate
held-out coverage audit is included.
