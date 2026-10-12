from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


def parse_csv_strings(s: str) -> List[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError(f"Empty CSV string: {s!r}")
    return out


def parse_csv_floats(s: str) -> List[float]:
    out = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError(f"Empty CSV float string: {s!r}")
    return out


def parse_sampling_probs(
    s: str,
    names: Optional[Sequence[str]] = None,
    expected_len: Optional[int] = None,
) -> Optional[List[float]]:
    """Parse comma-separated probabilities.

    Accepted formats:
      - "0.4,0.2,0.2,0.2"
      - "diff_jpeg:0.34,bmshj:0.33,bpg_like:0.33"

    If names is provided, named probabilities are returned in that order.
    Probabilities are normalized to sum to 1.
    """
    s = str(s or "").strip()
    if not s:
        return None

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None

    if any(":" in p for p in parts):
        if names is None:
            raise ValueError("Named probabilities require names.")
        mapping = {}
        for p in parts:
            if ":" not in p:
                raise ValueError(f"Mixed named/unnamed probability entry: {p!r}")
            k, v = p.split(":", 1)
            mapping[k.strip()] = float(v.strip())
        missing = [n for n in names if n not in mapping]
        if missing:
            raise ValueError(f"Missing probabilities for: {missing}")
        vals = [float(mapping[n]) for n in names]
    else:
        vals = [float(p) for p in parts]

    if expected_len is not None and len(vals) != int(expected_len):
        raise ValueError(f"Expected {expected_len} probabilities, got {len(vals)}: {vals}")

    total = float(sum(vals))
    if total <= 0:
        raise ValueError(f"Probability sum must be > 0, got {vals}")

    vals = [float(v) / total for v in vals]
    return vals


def parse_family_float_lists(s: str) -> Dict[str, List[float]]:
    """Parse family-specific float lists.

    Format:
        "diff_jpeg:0,0.25,0.5;bmshj:0,0.142857,1;bpg_like:0,0.1,1"

    This is used for family-specific severity grids. It lets JPEG sample
    severity values corresponding to JPEG qualities, BMSHJ sample its discrete
    quality indices, and BPG sample QP-derived severities.
    """
    s = str(s or "").strip()
    if not s:
        return {}

    out: Dict[str, List[float]] = {}
    for entry in [e.strip() for e in s.split(";") if e.strip()]:
        if ":" not in entry:
            raise ValueError(
                f"Invalid family float-list entry {entry!r}. Expected family:v1,v2,..."
            )
        name, values = entry.split(":", 1)
        vals = [float(v.strip()) for v in values.split(",") if v.strip()]
        if not vals:
            raise ValueError(f"No values specified for family {name!r}")
        out[name.strip()] = vals
    return out


def parse_family_prob_lists(s: str, levels_by_family: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Parse family-specific probability lists and normalize each family.

    Format mirrors parse_family_float_lists:
        "diff_jpeg:1,1,2;bmshj:1,2,2;bpg_like:1,1,2"

    Length for each family must match the corresponding severity level list.
    Missing family means uniform sampling for that family.
    """
    raw = parse_family_float_lists(s)
    out: Dict[str, List[float]] = {}
    for family, probs in raw.items():
        if family not in levels_by_family:
            raise ValueError(
                f"Severity probabilities specified for unknown family {family!r}. "
                f"Known families: {sorted(levels_by_family)}"
            )
        if len(probs) != len(levels_by_family[family]):
            raise ValueError(
                f"severity_probs_by_family[{family}] has length {len(probs)} but "
                f"severity_levels_by_family[{family}] has length {len(levels_by_family[family])}"
            )
        total = float(sum(probs))
        if total <= 0:
            raise ValueError(f"Probability sum for {family} must be > 0, got {probs}")
        out[family] = [float(p) / total for p in probs]
    return out


def build_codec_family_mapping(codec_families: Sequence[str]) -> Dict[str, int]:
    if len(set(codec_families)) != len(codec_families):
        raise ValueError(f"Duplicate codec family names: {codec_families}")
    return {name: idx for idx, name in enumerate(codec_families)}


def normalize_severity_tensor(
    raw: torch.Tensor,
    severity_min: float,
    severity_max: float,
    invert: bool = False,
) -> torch.Tensor:
    if severity_max <= severity_min:
        raise ValueError(
            f"severity_max must be > severity_min, got {severity_min=} {severity_max=}"
        )
    s = (raw.float() - float(severity_min)) / float(severity_max - severity_min)
    s = s.clamp(0.0, 1.0)
    if invert:
        s = 1.0 - s
    return s


def sample_conditions(
    batch_size: int,
    device: torch.device,
    codec_family_to_id: Dict[str, int],
    severity_levels: Optional[List[float]],
    severity_min: float,
    severity_max: float,
    invert_severity: bool,
    family_probs: Optional[List[float]] = None,
    severity_probs: Optional[List[float]] = None,
    severity_levels_by_family: Optional[Dict[str, List[float]]] = None,
    severity_probs_by_family: Optional[Dict[str, List[float]]] = None,
    id_to_codec_family: Optional[Dict[int, str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        family_ids: [B], long
        severity_raw: [B], float
        severity_norm: [B], float in [0, 1]

    The old sampler used one global severity grid for all codec families.
    This updated sampler can use family-specific grids/probabilities.

    Why this matters:
        codec parameter -> bitrate is not linear.
        A severity grid derived from actual JPEG qualities, BMSHJ quality
        indices, and BPG QP values gives better coverage than uniform 0..1.
    """
    n_families = len(codec_family_to_id)
    severity_levels_by_family = severity_levels_by_family or {}
    severity_probs_by_family = severity_probs_by_family or {}

    if family_probs is not None:
        p = torch.tensor(family_probs, dtype=torch.float32, device=device)
        p = p / p.sum().clamp_min(1e-8)
        family_ids = torch.multinomial(p, num_samples=batch_size, replacement=True).long()
    else:
        family_ids = torch.randint(
            low=0,
            high=n_families,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    if severity_levels_by_family:
        if id_to_codec_family is None:
            id_to_codec_family = {v: k for k, v in codec_family_to_id.items()}

        severity_values: List[torch.Tensor] = []
        for i in range(batch_size):
            family_id = int(family_ids[i].detach().cpu().item())
            family_name = id_to_codec_family[family_id]
            levels_list = severity_levels_by_family.get(family_name, severity_levels)

            if levels_list is None:
                # Fallback continuous uniform if no grid is defined for this family.
                val = torch.empty((), dtype=torch.float32, device=device)
                val.uniform_(float(severity_min), float(severity_max))
                severity_values.append(val)
                continue

            levels = torch.tensor(levels_list, dtype=torch.float32, device=device)
            probs_list = severity_probs_by_family.get(family_name, None)
            if probs_list is not None:
                p = torch.tensor(probs_list, dtype=torch.float32, device=device)
                if p.numel() != levels.numel():
                    raise ValueError(
                        f"severity_probs_by_family[{family_name}] length {p.numel()} "
                        f"!= levels length {levels.numel()}"
                    )
                p = p / p.sum().clamp_min(1e-8)
                idx = torch.multinomial(p, num_samples=1, replacement=True)[0]
            else:
                idx = torch.randint(0, levels.numel(), size=(), device=device)
            severity_values.append(levels[idx])

        severity_raw = torch.stack(severity_values, dim=0).to(device=device, dtype=torch.float32)

    elif severity_levels is not None:
        levels = torch.tensor(severity_levels, dtype=torch.float32, device=device)
        if severity_probs is not None:
            p = torch.tensor(severity_probs, dtype=torch.float32, device=device)
            if p.numel() != levels.numel():
                raise ValueError(
                    f"severity_probs length {p.numel()} != severity_levels length {levels.numel()}"
                )
            p = p / p.sum().clamp_min(1e-8)
            idx = torch.multinomial(p, num_samples=batch_size, replacement=True)
        else:
            idx = torch.randint(0, levels.numel(), size=(batch_size,), device=device)
        severity_raw = levels[idx]
    else:
        severity_raw = torch.empty(batch_size, dtype=torch.float32, device=device)
        severity_raw.uniform_(float(severity_min), float(severity_max))

    severity_norm = normalize_severity_tensor(
        severity_raw,
        severity_min=severity_min,
        severity_max=severity_max,
        invert=invert_severity,
    )

    return family_ids, severity_raw, severity_norm
