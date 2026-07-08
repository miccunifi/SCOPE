#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import importlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from coco_eval import evaluate_coco_map
from dataset_coco import CocoDataset
from transmission import compressai_transmission, uniform_jpeg_transmission
from utils import ensure_dir, jpeg_encode_decode, read_image_bgr, save_json, write_image
from yolo_utils import YoloPredictor, remove_internal_xyxy


# ---------------------------------------------------------------------
# Make the training file importable when this script is run from repo root.
# ---------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
for _p in [
    _THIS_DIR,
    _THIS_DIR / "standalone_conditional_preeditor",
    Path.cwd(),
    Path.cwd() / "standalone_conditional_preeditor",
]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------
# Import the old FiLM / blur_replace model class.
# This is the correct class for checkpoints trained with:
#   base_ch=48, family_emb_dim=8, cond_dim=64, bg_mode=blur_replace.
# Do NOT import contextblockbg / strictlowpass first for this checkpoint.
# ---------------------------------------------------------------------
_IMPORT_ERRORS = []
ConditionalAttentionPreEditor = None

for _module_name in [
    "train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8_nojp2_unbiased_mediumbpg_sevbasis",
    "train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8_nojp2_unbiased_mediumbpg",
    "train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8",
    "train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8_fix_coco_d2weights",
    "train_preedit_attention_conditional_yolo_hooks_objectmask_boxattention_bgreplace_objstrong_checked",
    "train_preedit_attention_conditional_yolo_hooks_objectmask_boxattention_bgreplace",
]:
    try:
        _m = importlib.import_module(_module_name)
        ConditionalAttentionPreEditor = _m.ConditionalAttentionPreEditor
        print(f"Using ConditionalAttentionPreEditor from: {_module_name}")
        break
    except Exception as _e:
        _IMPORT_ERRORS.append((_module_name, repr(_e)))

if ConditionalAttentionPreEditor is None:
    msg = "Could not import the old FiLM / blur_replace ConditionalAttentionPreEditor.\n"
    msg += "Make sure one of these files exists inside standalone_conditional_preeditor/:\n"
    msg += "  - train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8_nojp2_unbiased_mediumbpg_sevbasis.py\n"
    msg += "  - train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8_nojp2_unbiased_mediumbpg.py\n"
    msg += "  - train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8.py\n"
    msg += "Import errors:\n"
    for _name, _err in _IMPORT_ERRORS:
        msg += f"  {_name}: {_err}\n"
    raise RuntimeError(msg)


YOLO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def _remap_predictions_for_eval(preds: List[Dict], category_map: str) -> List[Dict]:
    category_map = str(category_map or "offset").lower()
    if category_map in {"offset", "none"}:
        return preds

    if category_map != "coco91":
        raise ValueError(f"Unknown category_map={category_map!r}. Use 'offset' or 'coco91'.")

    out = []
    for p in preds:
        q = dict(p)
        cls_id = int(q["category_id"])
        if cls_id < 0 or cls_id >= len(YOLO80_TO_COCO91):
            raise ValueError(f"Invalid YOLO COCO80 class id {cls_id}; expected 0..79.")
        q["category_id"] = YOLO80_TO_COCO91[cls_id]
        out.append(q)

    return out


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def normalize_codec_family_name(name: str) -> str:
    aliases = {
        "diff_jpeg": "diff_jpeg",
        "jpeg": "diff_jpeg",
        "block_dct": "diff_jpeg",

        "jp2_like": "jp2_like",
        "jp2": "jp2_like",
        "jpeg2000": "jp2_like",
        "wavelet": "jp2_like",

        "bpg_like": "bpg_like",
        "bpg": "bpg_like",
        "intra_smooth": "bpg_like",

        "bmshj": "bmshj",
        "learned": "bmshj",
        "compressai": "bmshj",
        "cheng": "bmshj",
    }

    name = str(name)
    if name not in aliases:
        raise ValueError(
            f"Unknown codec_family={name!r}. "
            "Use diff_jpeg, jp2_like, bpg_like, bmshj, or cheng."
        )

    return aliases[name]


def infer_codec_family_from_experiment(exp_cfg: Dict) -> str:
    explicit = exp_cfg.get("codec_family")
    if explicit is not None:
        return normalize_codec_family_name(str(explicit))

    mode = str(exp_cfg.get("mode", ""))
    codec = str(exp_cfg.get("compressai_model", "")).lower()

    if mode in {"uniform", "jpeg", "attention_preedit_jpeg"}:
        return "diff_jpeg"
    if mode in {"jpeg2000", "jp2", "attention_preedit_jpeg2000", "attention_preedit_jp2"}:
        return "jp2_like"
    if mode in {"bpg", "attention_preedit_bpg"}:
        return "bpg_like"

    if any(k in codec for k in ["jpeg2000", "jp2", "j2k"]):
        return "jp2_like"
    if any(k in codec for k in ["bpg", "heic", "heif", "avif", "jxl"]):
        return "bpg_like"

    return "bmshj"


def infer_severity_from_experiment(exp_cfg: Dict) -> float:
    mode = str(exp_cfg.get("mode", ""))
    codec = str(exp_cfg.get("compressai_model", "")).lower()

    if mode in {"uniform", "jpeg", "attention_preedit_jpeg"}:
        q = float(exp_cfg.get("jpeg_quality", 10))
        return float(np.clip(1.0 - (q - 10.0) / max(60.0 - 10.0, 1e-6), 0.0, 1.0))

    if mode in {"jpeg2000", "jp2", "attention_preedit_jpeg2000", "attention_preedit_jp2"}:
        r = float(exp_cfg.get("jp2_rate", 32))
        r_min = float(exp_cfg.get("jp2_rate_min", 1))
        r_max = float(exp_cfg.get("jp2_rate_max", 192))
        return float(np.clip((r - r_min) / max(r_max - r_min, 1e-6), 0.0, 1.0))

    if mode in {"bpg", "attention_preedit_bpg"}:
        q = float(exp_cfg.get("bpg_qp", exp_cfg.get("bpg_quality", 40)))
        q_min = float(exp_cfg.get("bpg_qp_min", 28))
        q_max = float(exp_cfg.get("bpg_qp_max", 48))
        return float(np.clip((q - q_min) / max(q_max - q_min, 1e-6), 0.0, 1.0))

    if "bmshj" in codec:
        q = float(exp_cfg.get("compressai_quality", 1))
        return float(np.clip(1.0 - (q - 1.0) / 7.0, 0.0, 1.0))

    if "cheng" in codec or "mbt" in codec:
        q = float(exp_cfg.get("compressai_quality", 1))
        return float(np.clip(1.0 - (q - 1.0) / 5.0, 0.0, 1.0))

    return 0.5


def get_experiment_severity(exp_cfg: Dict) -> float:
    # 1) Preferred: explicit conditioning severity from YAML.
    cond_cfg = exp_cfg.get("condition", {}) or {}
    if "severity" in cond_cfg:
        return float(np.clip(float(cond_cfg["severity"]), 0.0, 1.0))

    # 2) Also allow severity directly inside the experiment.
    if "severity" in exp_cfg:
        return float(np.clip(float(exp_cfg["severity"]), 0.0, 1.0))

    # 3) Fallback: infer from codec settings.
    return infer_severity_from_experiment(exp_cfg)


_PREEDIT_CACHE: Dict[Tuple[str, str], Tuple[torch.nn.Module, Dict[str, int]]] = {}


def _ckpt_arg(ckpt: Dict, key: str, default):
    """Read value from checkpoint top-level, then checkpoint args, else default."""
    args = ckpt.get("args", {}) or {}

    # Some training scripts may save argparse.Namespace instead of dict.
    if not isinstance(args, dict):
        args = vars(args)

    if key in ckpt:
        return ckpt[key]
    if key in args:
        return args[key]

    return default


def load_conditional_preedit_model(ckpt_path: str, device: torch.device, exp_cfg: Dict):
    key = (str(Path(ckpt_path).resolve()), str(device))
    if key in _PREEDIT_CACHE:
        return _PREEDIT_CACHE[key]

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(
            f"Checkpoint {ckpt_path} must be the conditional training checkpoint with key 'model'."
        )

    state = _strip_module_prefix(ckpt["model"])

    codec_family_to_id = ckpt.get("codec_family_to_id")
    if codec_family_to_id is None:
        raise ValueError(f"Checkpoint {ckpt_path} has no codec_family_to_id.")

    codec_family_to_id = {str(k): int(v) for k, v in codec_family_to_id.items()}

    bg_mode = str(exp_cfg.get("bg_mode", _ckpt_arg(ckpt, "bg_mode", "blur_replace")))
    bg_blur_kernel = int(exp_cfg.get("bg_blur_kernel", _ckpt_arg(ckpt, "bg_blur_kernel", 51)))
    bg_lowres_scale = int(exp_cfg.get("bg_lowres_scale", _ckpt_arg(ckpt, "bg_lowres_scale", 16)))

    if bg_mode != "blur_replace":
        print(
            f"WARNING: this eval file is for the old FiLM blur_replace checkpoint, "
            f"but bg_mode={bg_mode!r}. Check your YAML."
        )

    model = ConditionalAttentionPreEditor(
        num_codec_families=len(codec_family_to_id),
        base_ch=int(exp_cfg.get("base_ch", _ckpt_arg(ckpt, "base_ch", 48))),
        family_emb_dim=int(exp_cfg.get("family_emb_dim", _ckpt_arg(ckpt, "family_emb_dim", 8))),
        cond_dim=int(exp_cfg.get("cond_dim", _ckpt_arg(ckpt, "cond_dim", 64))),
        max_delta_obj=float(exp_cfg.get("max_delta_obj", _ckpt_arg(ckpt, "max_delta_obj", 0.09))),
        max_delta_bg=float(exp_cfg.get("max_delta_bg", _ckpt_arg(ckpt, "max_delta_bg", 0.03))),
        attention_bias=float(exp_cfg.get("attention_bias", _ckpt_arg(ckpt, "attention_bias", -4.0))),
        bg_mode=bg_mode,
        bg_blur_kernel=bg_blur_kernel,
        bg_lowres_scale=bg_lowres_scale,
    )

    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    print(f"Loaded conditional pre-editor: {ckpt_path}")
    print(f"  codec_family_to_id: {codec_family_to_id}")
    print(f"  model class: {ConditionalAttentionPreEditor.__module__}.{ConditionalAttentionPreEditor.__name__}")
    print(f"  base_ch: {int(exp_cfg.get('base_ch', _ckpt_arg(ckpt, 'base_ch', 48)))}")
    print(f"  family_emb_dim: {int(exp_cfg.get('family_emb_dim', _ckpt_arg(ckpt, 'family_emb_dim', 8)))}")
    print(f"  cond_dim: {int(exp_cfg.get('cond_dim', _ckpt_arg(ckpt, 'cond_dim', 64)))}")
    print(f"  max_delta_obj: {float(exp_cfg.get('max_delta_obj', _ckpt_arg(ckpt, 'max_delta_obj', 0.09)))}")
    print(f"  max_delta_bg: {float(exp_cfg.get('max_delta_bg', _ckpt_arg(ckpt, 'max_delta_bg', 0.03)))}")
    print(f"  attention_bias: {float(exp_cfg.get('attention_bias', _ckpt_arg(ckpt, 'attention_bias', -4.0)))}")
    print(f"  bg_mode: {bg_mode}")
    print(f"  bg_blur_kernel: {bg_blur_kernel}")
    print(f"  bg_lowres_scale: {bg_lowres_scale}")

    _PREEDIT_CACHE[key] = (model, codec_family_to_id)
    return model, codec_family_to_id


def bgr_to_model_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )


def tensor_to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    rgb = np.round(x * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def apply_preedit_bgr_conditional(
    img_bgr: np.ndarray,
    exp_cfg: Dict,
    device,
    return_debug: bool = False,
):
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, codec_family_to_id = load_conditional_preedit_model(
        str(exp_cfg["preedit_ckpt"]),
        torch_device,
        exp_cfg,
    )

    family_name = infer_codec_family_from_experiment(exp_cfg)
    if family_name not in codec_family_to_id:
        raise ValueError(f"Family {family_name!r} not in checkpoint mapping {codec_family_to_id}")

    severity = get_experiment_severity(exp_cfg)

    x = bgr_to_model_tensor(img_bgr, torch_device)
    family_id = torch.tensor(
        [codec_family_to_id[family_name]],
        device=torch_device,
        dtype=torch.long,
    )
    severity_t = torch.tensor([severity], device=torch_device, dtype=x.dtype)

    edited, delta_obj, delta_bg, attention, effective_delta = model(
        x,
        severity=severity_t,
        codec_family=family_id,
    )

    edited_bgr = tensor_to_bgr_uint8(edited)

    if not return_debug:
        return edited_bgr, None

    return edited_bgr, {
        "edited_bgr": edited_bgr,
        "attention": attention.detach().cpu(),
        "effective_delta": effective_delta.detach().cpu(),
        "delta_obj": delta_obj.detach().cpu(),
        "delta_bg": delta_bg.detach().cpu(),
        "codec_family": family_name,
        "severity": severity,
    }


def predict_server(
    server_predictor: YoloPredictor,
    image_path: Path,
    image_id: int,
    category_id_offset: int,
    server_predict_cfg: Dict,
) -> List[Dict]:
    category_map = str(server_predict_cfg.get("category_map", "offset")).lower()
    pred_category_id_offset = 0 if category_map == "coco91" else category_id_offset

    preds = server_predictor.predict_full(
        image_path=image_path,
        image_id=image_id,
        category_id_offset=pred_category_id_offset,
        conf=float(server_predict_cfg.get("conf", 0.001)),
        iou=float(server_predict_cfg.get("iou", 0.7)),
        max_det=int(server_predict_cfg.get("max_det", 300)),
    )

    preds = _remap_predictions_for_eval(preds, category_map=category_map)
    return remove_internal_xyxy(preds)


def _as_arg_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return x.split()
    return [str(v) for v in x]


def real_bpg_transmission(
    img_bgr,
    bpg_qp: int,
    encoder_bin: str = "bpgenc",
    decoder_bin: str = "bpgdec",
    encoder_args=None,
):
    enc = shutil.which(str(encoder_bin)) or str(encoder_bin)
    dec = shutil.which(str(decoder_bin)) or str(decoder_bin)

    if shutil.which(str(encoder_bin)) is None and not Path(enc).exists():
        raise FileNotFoundError(f"Could not find BPG encoder {encoder_bin!r}.")
    if shutil.which(str(decoder_bin)) is None and not Path(dec).exists():
        raise FileNotFoundError(f"Could not find BPG decoder {decoder_bin!r}.")

    extra_args = _as_arg_list(encoder_args)

    with tempfile.TemporaryDirectory(prefix="bpg_eval_") as td:
        td = Path(td)
        in_png = td / "input.png"
        out_bpg = td / "compressed.bpg"
        out_png = td / "decoded.png"

        if not cv2.imwrite(str(in_png), img_bgr):
            raise RuntimeError(f"Failed to write temporary image: {in_png}")

        enc_cmd = [enc, "-q", str(int(bpg_qp)), *extra_args, "-o", str(out_bpg), str(in_png)]
        enc_res = subprocess.run(
            enc_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if enc_res.returncode != 0:
            raise RuntimeError(
                f"bpgenc failed:\ncommand: {' '.join(enc_cmd)}\nstderr: {enc_res.stderr}"
            )

        estimated_bytes = int(out_bpg.stat().st_size)

        dec_cmd = [dec, "-o", str(out_png), str(out_bpg)]
        dec_res = subprocess.run(
            dec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if dec_res.returncode != 0:
            raise RuntimeError(
                f"bpgdec failed:\ncommand: {' '.join(dec_cmd)}\nstderr: {dec_res.stderr}"
            )

        reconstructed = cv2.imread(str(out_png), cv2.IMREAD_COLOR)
        if reconstructed is None:
            raise RuntimeError(f"Failed to read decoded BPG image: {out_png}")

    return {
        "reconstructed_bgr": reconstructed,
        "estimated_bytes": estimated_bytes,
        "debug": None,
    }


def real_jpeg2000_transmission(
    img_bgr,
    jp2_rate: Optional[float] = None,
    encoder_bin: str = "opj_compress",
    decoder_bin: str = "opj_decompress",
    encoder_args=None,
    backend: str = "auto",
    cv2_compression_x1000: Optional[int] = None,
):
    backend = str(backend or "auto").lower()
    if backend not in {"auto", "openjpeg", "cv2"}:
        raise ValueError(f"Invalid jp2_backend={backend}")

    enc = shutil.which(str(encoder_bin)) or str(encoder_bin)
    dec = shutil.which(str(decoder_bin)) or str(decoder_bin)

    have_openjpeg = (
        (shutil.which(str(encoder_bin)) is not None or Path(enc).exists())
        and (shutil.which(str(decoder_bin)) is not None or Path(dec).exists())
    )

    if backend in {"auto", "openjpeg"} and have_openjpeg:
        extra_args = _as_arg_list(encoder_args)

        with tempfile.TemporaryDirectory(prefix="jp2_eval_") as td:
            td = Path(td)
            in_png = td / "input.png"
            out_jp2 = td / "compressed.jp2"
            out_png = td / "decoded.png"

            if not cv2.imwrite(str(in_png), img_bgr):
                raise RuntimeError(f"Failed to write temporary image: {in_png}")

            cmd = [enc, "-i", str(in_png), "-o", str(out_jp2)]
            if jp2_rate is not None:
                cmd += ["-r", str(float(jp2_rate))]
            cmd += extra_args

            enc_res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if enc_res.returncode != 0:
                raise RuntimeError(
                    f"opj_compress failed:\ncommand: {' '.join(cmd)}\nstderr: {enc_res.stderr}"
                )

            estimated_bytes = int(out_jp2.stat().st_size)

            dec_cmd = [dec, "-i", str(out_jp2), "-o", str(out_png)]
            dec_res = subprocess.run(
                dec_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if dec_res.returncode != 0:
                raise RuntimeError(
                    f"opj_decompress failed:\ncommand: {' '.join(dec_cmd)}\nstderr: {dec_res.stderr}"
                )

            reconstructed = cv2.imread(str(out_png), cv2.IMREAD_COLOR)
            if reconstructed is None:
                raise RuntimeError(f"Failed to read decoded JPEG2000 image: {out_png}")

        return {
            "reconstructed_bgr": reconstructed,
            "estimated_bytes": estimated_bytes,
            "debug": None,
        }

    if backend == "openjpeg":
        raise FileNotFoundError(
            f"Could not find OpenJPEG binaries {encoder_bin!r}/{decoder_bin!r}."
        )

    if cv2_compression_x1000 is None:
        cv2_compression_x1000 = (
            1000
            if jp2_rate is None
            else int(max(50, min(1000, round(1000.0 / max(float(jp2_rate), 1.0)))))
        )

    with tempfile.TemporaryDirectory(prefix="jp2_eval_cv2_") as td:
        td = Path(td)
        out_jp2 = td / "compressed.jp2"

        params = []
        if hasattr(cv2, "IMWRITE_JPEG2000_COMPRESSION_X1000"):
            params = [
                int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000),
                int(cv2_compression_x1000),
            ]

        ok = cv2.imwrite(str(out_jp2), img_bgr, params)
        if not ok or not out_jp2.exists():
            raise RuntimeError(
                "OpenCV could not write JPEG2000. "
                "Install OpenJPEG tools and use jp2_backend: openjpeg."
            )

        estimated_bytes = int(out_jp2.stat().st_size)
        reconstructed = cv2.imread(str(out_jp2), cv2.IMREAD_COLOR)

        if reconstructed is None:
            raise RuntimeError("OpenCV wrote a JP2 file but could not decode it.")

    return {
        "reconstructed_bgr": reconstructed,
        "estimated_bytes": estimated_bytes,
        "debug": None,
    }


def attention_jpeg_transmission(
    img_bgr,
    exp_cfg: Dict,
    jpeg_quality: int,
    device=None,
    return_debug: bool = False,
):
    edited, debug = apply_preedit_bgr_conditional(img_bgr, exp_cfg, device, return_debug)
    reconstructed, nbytes = jpeg_encode_decode(edited, jpeg_quality)

    return {
        "reconstructed_bgr": reconstructed,
        "estimated_bytes": int(nbytes),
        "debug": debug,
    }


def attention_bpg_transmission(
    img_bgr,
    exp_cfg: Dict,
    bpg_qp: int,
    device=None,
    return_debug: bool = False,
):
    edited, debug = apply_preedit_bgr_conditional(img_bgr, exp_cfg, device, return_debug)

    result = real_bpg_transmission(
        img_bgr=edited,
        bpg_qp=int(bpg_qp),
        encoder_bin=str(exp_cfg.get("bpg_encoder_bin", "bpgenc")),
        decoder_bin=str(exp_cfg.get("bpg_decoder_bin", "bpgdec")),
        encoder_args=exp_cfg.get("bpg_encoder_args", []),
    )
    result["debug"] = debug
    return result


def attention_jpeg2000_transmission(
    img_bgr,
    exp_cfg: Dict,
    jp2_rate: Optional[float] = None,
    device=None,
    return_debug: bool = False,
):
    edited, debug = apply_preedit_bgr_conditional(img_bgr, exp_cfg, device, return_debug)

    result = real_jpeg2000_transmission(
        img_bgr=edited,
        jp2_rate=jp2_rate,
        encoder_bin=str(exp_cfg.get("jp2_encoder_bin", "opj_compress")),
        decoder_bin=str(exp_cfg.get("jp2_decoder_bin", "opj_decompress")),
        encoder_args=exp_cfg.get("jp2_encoder_args", []),
        backend=str(exp_cfg.get("jp2_backend", "auto")),
        cv2_compression_x1000=exp_cfg.get("jp2_compression_x1000", None),
    )
    result["debug"] = debug
    return result


def attention_compressai_transmission(
    img_bgr,
    exp_cfg: Dict,
    compressai_model: str,
    compressai_quality: int,
    device=None,
    return_debug: bool = False,
):
    edited, debug = apply_preedit_bgr_conditional(img_bgr, exp_cfg, device, return_debug)

    result = compressai_transmission(
        img_bgr=edited,
        model_name=compressai_model,
        quality=compressai_quality,
        device=device,
    )

    return {
        "reconstructed_bgr": result.reconstructed_bgr,
        "estimated_bytes": int(result.estimated_bytes),
        "debug": debug,
    }


def _chosen_quality(exp_cfg: Dict):
    for k in [
        "jpeg_quality",
        "compressai_quality",
        "bpg_qp",
        "bpg_quality",
        "jp2_rate",
        "jp2_compression_x1000",
    ]:
        if k in exp_cfg:
            return exp_cfg[k]
    return ""


def run_single_experiment(
    exp_cfg: Dict,
    dataset: CocoDataset,
    output_dir: Path,
    server_predictor: YoloPredictor,
    category_id_offset: int,
    server_predict_cfg: Dict,
    device,
) -> Dict:
    name = exp_cfg["name"]
    mode = exp_cfg["mode"]

    exp_dir = ensure_dir(output_dir / name)
    recon_dir = ensure_dir(exp_dir / "reconstructed")
    debug_dir = ensure_dir(exp_dir / "attention_debug")

    save_debug = bool(exp_cfg.get("save_debug", False))
    max_debug_images = int(exp_cfg.get("max_debug_images", 25))
    saved_debug_count = 0

    predictions = []
    bandwidth_rows = []

    for record in tqdm(dataset.images, desc=name):
        img = read_image_bgr(record.path)
        debug = None

        if mode in {"uniform", "jpeg"}:
            result_obj = uniform_jpeg_transmission(
                img_bgr=img,
                quality=int(exp_cfg["jpeg_quality"]),
            )
            reconstructed = result_obj.reconstructed_bgr
            estimated_bytes = int(result_obj.estimated_bytes)

        elif mode in {"jpeg2000", "jp2"}:
            result = real_jpeg2000_transmission(
                img_bgr=img,
                jp2_rate=exp_cfg.get("jp2_rate", None),
                encoder_bin=str(exp_cfg.get("jp2_encoder_bin", "opj_compress")),
                decoder_bin=str(exp_cfg.get("jp2_decoder_bin", "opj_decompress")),
                encoder_args=exp_cfg.get("jp2_encoder_args", []),
                backend=str(exp_cfg.get("jp2_backend", "auto")),
                cv2_compression_x1000=exp_cfg.get("jp2_compression_x1000", None),
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])

        elif mode == "bpg":
            result = real_bpg_transmission(
                img_bgr=img,
                bpg_qp=int(exp_cfg.get("bpg_qp", exp_cfg.get("bpg_quality", 40))),
                encoder_bin=str(exp_cfg.get("bpg_encoder_bin", "bpgenc")),
                decoder_bin=str(exp_cfg.get("bpg_decoder_bin", "bpgdec")),
                encoder_args=exp_cfg.get("bpg_encoder_args", []),
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])

        elif mode == "compressai":
            result_obj = compressai_transmission(
                img_bgr=img,
                model_name=str(exp_cfg["compressai_model"]),
                quality=int(exp_cfg["compressai_quality"]),
                device=device,
            )
            reconstructed = result_obj.reconstructed_bgr
            estimated_bytes = int(result_obj.estimated_bytes)

        elif mode == "attention_preedit_jpeg":
            result = attention_jpeg_transmission(
                img,
                exp_cfg,
                int(exp_cfg["jpeg_quality"]),
                device=device,
                return_debug=save_debug and saved_debug_count < max_debug_images,
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])
            debug = result["debug"]

        elif mode == "attention_preedit_bpg":
            result = attention_bpg_transmission(
                img,
                exp_cfg,
                int(exp_cfg.get("bpg_qp", exp_cfg.get("bpg_quality", 40))),
                device=device,
                return_debug=save_debug and saved_debug_count < max_debug_images,
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])
            debug = result["debug"]

        elif mode in {"attention_preedit_jpeg2000", "attention_preedit_jp2"}:
            result = attention_jpeg2000_transmission(
                img,
                exp_cfg,
                exp_cfg.get("jp2_rate", None),
                device=device,
                return_debug=save_debug and saved_debug_count < max_debug_images,
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])
            debug = result["debug"]

        elif mode == "attention_preedit_compressai":
            result = attention_compressai_transmission(
                img,
                exp_cfg,
                str(exp_cfg["compressai_model"]),
                int(exp_cfg["compressai_quality"]),
                device=device,
                return_debug=save_debug and saved_debug_count < max_debug_images,
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])
            debug = result["debug"]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        recon_path = recon_dir / (Path(record.file_name).stem + ".png")
        write_image(recon_path, reconstructed)

        if save_debug and debug is not None and saved_debug_count < max_debug_images:
            stem = Path(record.file_name).stem

            write_image(debug_dir / f"{stem}_edited.png", debug["edited_bgr"])

            attn = debug["attention"][0, 0].numpy()
            cv2.imwrite(
                str(debug_dir / f"{stem}_attention.png"),
                np.round(np.clip(attn, 0, 1) * 255).astype(np.uint8),
            )

            delta = debug["effective_delta"][0].abs().mean(dim=0).numpy()
            delta = delta / max(float(delta.max()), 1e-8)
            cv2.imwrite(
                str(debug_dir / f"{stem}_delta.png"),
                np.round(delta * 255).astype(np.uint8),
            )

            with open(debug_dir / f"{stem}_meta.json", "w") as f:
                json.dump(
                    {
                        "codec_family": debug.get("codec_family"),
                        "severity": debug.get("severity"),
                    },
                    f,
                    indent=2,
                )

            saved_debug_count += 1

        preds = predict_server(
            server_predictor,
            recon_path,
            record.image_id,
            category_id_offset,
            server_predict_cfg,
        )
        predictions.extend(preds)

        h, w = img.shape[:2]
        num_pixels = max(1, int(h) * int(w))
        estimated_bpp = (float(estimated_bytes) * 8.0) / float(num_pixels)

        bandwidth_rows.append(
            {
                "image_id": record.image_id,
                "file_name": record.file_name,
                "width": int(w),
                "height": int(h),
                "num_pixels": num_pixels,
                "estimated_bytes": estimated_bytes,
                "estimated_bpp": estimated_bpp,
                "num_rois": 0,
                "chosen_quality": _chosen_quality(exp_cfg),
                "codec_family": infer_codec_family_from_experiment(exp_cfg),
                "severity": get_experiment_severity(exp_cfg),
            }
        )

    pred_json = exp_dir / "predictions_coco.json"
    save_json(predictions, pred_json)

    bandwidth_csv = exp_dir / "bandwidth.csv"
    with open(bandwidth_csv, "w", newline="") as f:
        fieldnames = [
            "image_id",
            "file_name",
            "width",
            "height",
            "num_pixels",
            "estimated_bytes",
            "estimated_bpp",
            "num_rois",
            "chosen_quality",
            "codec_family",
            "severity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bandwidth_rows)

    metrics = evaluate_coco_map(dataset.annotations_json, pred_json)

    avg_bytes = sum(r["estimated_bytes"] for r in bandwidth_rows) / max(1, len(bandwidth_rows))
    avg_bpp_per_image = sum(r["estimated_bpp"] for r in bandwidth_rows) / max(1, len(bandwidth_rows))
    total_bytes = sum(r["estimated_bytes"] for r in bandwidth_rows)
    total_pixels = sum(r["num_pixels"] for r in bandwidth_rows)
    dataset_bpp = (float(total_bytes) * 8.0) / float(max(1, total_pixels))

    summary = {
        "experiment": name,
        "mode": mode,
        "codec_family": infer_codec_family_from_experiment(exp_cfg),
        "severity": get_experiment_severity(exp_cfg),
        "avg_bytes_per_image": avg_bytes,
        "avg_bpp_per_image": avg_bpp_per_image,
        "dataset_bpp": dataset_bpp,
        "avg_rois_per_image": 0.0,
        **metrics,
    }

    save_json(summary, exp_dir / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = CocoDataset(
        images_dir=cfg["dataset"]["images_dir"],
        annotations_json=cfg["dataset"]["annotations_json"],
    )

    output_dir = ensure_dir(cfg["output_dir"])
    device = cfg.get("device", None)

    server_predictor = YoloPredictor(cfg["models"]["server_model"], device=device)

    summaries = []

    root_passthrough_keys = [
        "bpg_encoder_bin",
        "bpg_decoder_bin",
        "bpg_encoder_args",
        "jp2_encoder_bin",
        "jp2_decoder_bin",
        "jp2_encoder_args",

        "preedit_ckpt",
        "architecture",
        "condition_mode",
        "base_ch",
        "family_emb_dim",
        "cond_dim",
        "max_delta_obj",
        "max_delta_bg",
        "attention_bias",
        "bg_mode",
        "bg_blur_kernel",
        "bg_lowres_scale",
    ]

    for exp_cfg in cfg["experiments"]:
        exp_cfg = dict(exp_cfg)

        for k in root_passthrough_keys:
            if k in cfg and k not in exp_cfg:
                exp_cfg[k] = cfg[k]

        summaries.append(
            run_single_experiment(
                exp_cfg,
                dataset,
                output_dir,
                server_predictor,
                int(cfg.get("category_id_offset", 1)),
                cfg.get("server_predict", {}),
                device,
            )
        )

    summary_csv = output_dir / "all_results.csv"
    fieldnames = [
        "experiment",
        "mode",
        "codec_family",
        "severity",
        "avg_bytes_per_image",
        "avg_bpp_per_image",
        "dataset_bpp",
        "avg_rois_per_image",
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summaries:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nSaved summary to: {summary_csv}")


if __name__ == "__main__":
    main()