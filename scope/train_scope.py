from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scope.training import train
else:
    from .training import train


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone conditional family/severity SCOPE trainer."
    )

    # Data
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--annotations-json", type=str, required=True)
    p.add_argument("--val-images-dir", type=str, default="")
    p.add_argument("--val-annotations-json", type=str, default="")

    p.add_argument("--output-dir", type=str, required=True)

    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--object-crop-prob", type=float, default=0.60)
    p.add_argument("--crop-margin", type=float, default=4.0)

    # Conditioning
    p.add_argument(
        "--codec-families",
        type=str,
        default="diff_jpeg,bmshj,bpg_like",
        help="Comma-separated codec families.",
    )
    p.add_argument(
        "--severity-min",
        type=float,
        default=0.0,
        help="Raw severity value mapped to normalized 0.",
    )
    p.add_argument(
        "--severity-max",
        type=float,
        default=1.0,
        help="Raw severity value mapped to normalized 1.",
    )
    p.add_argument(
        "--severity-levels",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help=(
            "Comma-separated NORMALIZED severity levels in [0, 1] to sample. "
            "0 = mild degradation / high bitrate, 1 = severe degradation / low bitrate. "
            "Use empty string for continuous uniform sampling."
        ),
    )
    p.add_argument(
        "--val-severity-levels",
        type=str,
        default="",
        help="Optional validation raw severity levels.",
    )
    p.add_argument(
        "--severity-probs",
        type=str,
        default="",
        help=(
            "Optional comma-separated sampling probabilities aligned with --severity-levels. "
            "Example for 0,0.2,0.4,0.6,0.8,1.0: 0.03,0.07,0.15,0.20,0.25,0.30"
        ),
    )
    p.add_argument(
        "--codec-family-probs",
        type=str,
        default="",
        help=(
            "Optional family sampling probabilities. Either aligned with --codec-families "
            "or named, e.g. diff_jpeg:0.34,bmshj:0.33,bpg_like:0.33"
        ),
    )
    p.add_argument(
        "--severity-levels-by-family",
        type=str,
        default="",
        help=(
            "Optional family-specific severity grids. Format: "
            "diff_jpeg:0,0.5,1;bmshj:0,0.142857,1;bpg_like:0,0.1,1. "
            "If set, it overrides the global --severity-levels during training."
        ),
    )
    p.add_argument(
        "--severity-probs-by-family",
        type=str,
        default="",
        help=(
            "Optional family-specific sampling probabilities aligned with "
            "--severity-levels-by-family. Missing family = uniform for that family."
        ),
    )
    p.add_argument(
        "--val-severity-levels-by-family",
        type=str,
        default="",
        help=(
            "Optional family-specific validation severity grids, same format as "
            "--severity-levels-by-family."
        ),
    )
    p.add_argument(
        "--invert-severity",
        action="store_true",
        help="Use if raw value is quality where larger means less severe.",
    )

    # Real codec-family proxies.
    p.add_argument(
        "--use-real-diffjpeg",
        action="store_true",
        help=(
            "Use necla-ml/Diff-JPEG with STE for the diff_jpeg family. "
            "If disabled, use the internal JPEG-like proxy."
        ),
    )
    p.add_argument(
        "--diffjpeg-quality-min",
        type=int,
        default=10,
        help="JPEG quality at severity=1. Diff-JPEG valid range should be <= 99.",
    )
    p.add_argument(
        "--diffjpeg-quality-max",
        type=int,
        default=99,
        help="JPEG quality at severity=0. necla-ml/Diff-JPEG tops at 99, not 100.",
    )
    p.add_argument(
        "--use-real-bmshj",
        action="store_true",
        help=(
            "Use real CompressAI BMSHJ forward for the bmshj family and "
            "likelihood bpp as rate term."
        ),
    )
    p.add_argument(
        "--bmshj-model",
        type=str,
        default="bmshj2018-hyperprior",
        choices=["bmshj2018-hyperprior", "bmshj2018-factorized"],
    )
    p.add_argument("--bmshj-quality-min", type=int, default=1)
    p.add_argument("--bmshj-quality-max", type=int, default=8)
    p.add_argument(
        "--bmshj-eval-mode",
        action="store_true",
        help=(
            "Use BMSHJ codec in eval mode. Default is train/noisy-quant mode, "
            "which usually gives better gradients for SCOPE training."
        ),
    )
    p.add_argument(
        "--bmshj-bpp-scale",
        type=float,
        default=0.03,
        help="Scale for CompressAI likelihood bpp before adding to rate loss.",
    )
    p.add_argument(
        "--rate-tv-scale",
        type=float,
        default=1.0,
        help="Scale for TV/high-frequency rate proxy. BMSHJ bpp is added separately.",
    )

    # Real/proxy codec parameter ranges used by the handcrafted proxies.
    # Manual eval severity can still be used; these ranges make training know
    # what severity=0 and severity=1 mean for each real codec family.
    p.add_argument("--jp2-rate-min", type=float, default=1.0)
    p.add_argument("--jp2-rate-max", type=float, default=192.0)
    p.add_argument("--bpg-qp-min", type=float, default=0.0)
    p.add_argument("--bpg-qp-max", type=float, default=51.0)
    p.add_argument(
        "--bpg-proxy-mode",
        type=str,
        default="medium",
        choices=["old", "medium", "strong"],
        help=(
            "BPG-like differentiable proxy variant: "
            "old=smooth/block proxy, medium=stable YCbCr/intra/transform proxy, "
            "strong=more aggressive experimental proxy."
        ),
    )

    # Model
    p.add_argument("--base-ch", type=int, default=48)
    p.add_argument("--family-emb-dim", type=int, default=8)
    p.add_argument("--cond-dim", type=int, default=64)
    p.add_argument("--max-delta-obj", type=float, default=0.09)
    p.add_argument("--max-delta-bg", type=float, default=0.03)
    p.add_argument("--attention-bias", type=float, default=-4.0)
    p.add_argument(
        "--bg-mode",
        type=str,
        default="blur_replace",
        choices=["residual", "blur_replace", "lowres_replace", "mean_replace"],
        help=(
            "Background composition mode. "
            "residual = old x + delta_bg behavior. "
            "blur_replace/lowres_replace/mean_replace = background branch replaces "
            "background with a cheaper low-frequency/flat version plus residual."
        ),
    )
    p.add_argument(
        "--bg-blur-kernel",
        type=int,
        default=51,
        help="Kernel used by blur_replace and bg_flat_mode=blur.",
    )
    p.add_argument(
        "--bg-lowres-scale",
        type=int,
        default=16,
        help="Downsampling factor used by lowres_replace and bg_flat_mode=lowres.",
    )

    # Training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--samples-per-epoch", type=int, default=20000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint used to initialize/fine-tune the SCOPE model weights.",
    )

    # Loss weights
    p.add_argument(
        "--attn-blur",
        type=int,
        default=7,
        help=(
            "Blur/dilation size for attention target built from annotation boxes. "
            "Use a small value to keep attention tight around GT boxes."
        ),
    )
    p.add_argument(
        "--det-mask-blur",
        type=int,
        default=31,
        help=(
            "Blur/dilation size for detector feature mask built from annotation boxes. "
            "Usually larger than attn-blur because YOLO features need object context."
        ),
    )
    p.add_argument("--attn-area-mult", type=float, default=1.0)

    p.add_argument(
        "--lambda-bg-flat",
        type=float,
        default=0.75,
        help=(
            "Weight for pushing background regions toward a cheap low-frequency/flat target "
            "rather than matching the clean background."
        ),
    )
    p.add_argument(
        "--bg-flat-mode",
        type=str,
        default="blur",
        choices=["blur", "lowres", "mean", "zero"],
        help="Cheap target used when lambda-bg-flat > 0.",
    )
    p.add_argument("--lambda-bg-smooth", type=float, default=1.00)
    p.add_argument("--lambda-rate", type=float, default=2.00)
    p.add_argument("--lambda-attn", type=float, default=1.00)
    p.add_argument("--lambda-attn-sparse", type=float, default=0.75)
    p.add_argument("--lambda-attn-area", type=float, default=2.00)
    p.add_argument("--lambda-delta", type=float, default=0.02)
    p.add_argument("--lambda-delta-tv", type=float, default=0.05)

    # Severity-dependent dynamic loss weights.
    # At high severity, task/object/FPN loss is upweighted and rate/background
    # simplification can be downweighted. This is intended to improve q1/q2 AP.
    p.add_argument("--task-severity-gain", type=float, default=0.0)
    p.add_argument("--rate-severity-drop", type=float, default=0.0)
    p.add_argument("--bg-severity-drop", type=float, default=0.0)
    p.add_argument("--min-rate-mult", type=float, default=0.25)
    p.add_argument("--min-bg-mult", type=float, default=0.25)

    # Frozen detector feature consistency.
    p.add_argument(
        "--detector-backend",
        type=str,
        default="yolo",
        choices=["none", "yolo", "detectron2"],
        help="Detector feature-distillation backend.",
    )
    p.add_argument("--detector-weights", type=str, default="")
    p.add_argument("--lambda-det", type=float, default=2.0)
    p.add_argument(
        "--detector-hook-layers",
        type=str,
        default="",
        help=(
            "Comma-separated Ultralytics YOLO layer indices for feature hooks. "
            "Example: 10,15,20. Empty = automatic mid/high-level layers."
        ),
    )
    p.add_argument(
        "--detector-feature-loss",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "l1", "mse"],
        help="Loss used for object-masked YOLO hook feature consistency.",
    )
    p.add_argument(
        "--print-detector-layers",
        action="store_true",
        help="Print Ultralytics YOLO layer indices before training.",
    )

    # Detectron2 Faster R-CNN / FPN feature distillation.
    p.add_argument(
        "--detectron2-config",
        type=str,
        default="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        help="Detectron2 model-zoo config or local config path.",
    )
    p.add_argument(
        "--detectron2-weights",
        type=str,
        default="",
        help="Optional local Detectron2 weights. Empty = model-zoo checkpoint.",
    )
    p.add_argument(
        "--detectron2-feature-names",
        type=str,
        default="p2,p3,p4,p5,p6",
        help="Comma-separated FPN feature names to distill.",
    )
    p.add_argument(
        "--detectron2-input-format",
        type=str,
        default="BGR",
        choices=["BGR", "RGB"],
        help="Input color order expected by the Detectron2 model.",
    )
    p.add_argument(
        "--d2-fpn-mask-weight",
        type=float,
        default=0.7,
        help="Weight for object/context-masked FPN feature loss.",
    )
    p.add_argument(
        "--d2-fpn-full-weight",
        type=float,
        default=0.3,
        help="Weight for full-image FPN feature loss, useful to preserve context.",
    )

    p.add_argument(
        "--assert-objstrong",
        action="store_true",
        help=(
            "Abort unless the critical object-strong/background-replacement "
            "settings are exactly the intended ones. Useful to catch stale scripts."
        ),
    )

    # Validation
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-steps", type=int, default=50)

    # Logging
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print training progress every N optimizer steps inside each epoch.",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
