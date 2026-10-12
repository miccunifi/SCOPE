from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from scope.inference import load_scope_model
from .conditioning import (
    bgr_to_model_tensor,
    get_experiment_severity,
    infer_codec_family_from_experiment,
    tensor_to_bgr_uint8,
)
from .transmission import compressai_transmission, uniform_jpeg_transmission
from .utils import jpeg_encode_decode


@torch.no_grad()
def apply_scope_bgr_conditional(
    img_bgr: np.ndarray,
    exp_cfg: Dict,
    device,
    return_debug: bool = False,
):
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, codec_family_to_id = load_scope_model(
        str(exp_cfg["scope_ckpt"]),
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


def scope_jpeg_transmission(
    img_bgr,
    exp_cfg: Dict,
    jpeg_quality: int,
    device=None,
    return_debug: bool = False,
):
    edited, debug = apply_scope_bgr_conditional(img_bgr, exp_cfg, device, return_debug)
    reconstructed, nbytes = jpeg_encode_decode(edited, jpeg_quality)

    return {
        "reconstructed_bgr": reconstructed,
        "estimated_bytes": int(nbytes),
        "debug": debug,
    }


def scope_bpg_transmission(
    img_bgr,
    exp_cfg: Dict,
    bpg_qp: int,
    device=None,
    return_debug: bool = False,
):
    edited, debug = apply_scope_bgr_conditional(img_bgr, exp_cfg, device, return_debug)

    result = real_bpg_transmission(
        img_bgr=edited,
        bpg_qp=int(bpg_qp),
        encoder_bin=str(exp_cfg.get("bpg_encoder_bin", "bpgenc")),
        decoder_bin=str(exp_cfg.get("bpg_decoder_bin", "bpgdec")),
        encoder_args=exp_cfg.get("bpg_encoder_args", []),
    )
    result["debug"] = debug
    return result


def scope_jpeg2000_transmission(
    img_bgr,
    exp_cfg: Dict,
    jp2_rate: Optional[float] = None,
    device=None,
    return_debug: bool = False,
):
    edited, debug = apply_scope_bgr_conditional(img_bgr, exp_cfg, device, return_debug)

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


def scope_compressai_transmission(
    img_bgr,
    exp_cfg: Dict,
    compressai_model: str,
    compressai_quality: int,
    device=None,
    return_debug: bool = False,
):
    edited, debug = apply_scope_bgr_conditional(img_bgr, exp_cfg, device, return_debug)

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
