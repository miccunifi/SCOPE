from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .common import EXCLUDED_CODECS, EXCLUDED_FAMILIES, canonical_method_key, norm

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None


def is_ready(row: Dict[str, Any], metric: str, x_key: str) -> bool:
    if row.get("codec") in EXCLUDED_CODECS:
        return False
    if row.get("family") in EXCLUDED_FAMILIES:
        return False
    return row.get(metric, "") != "" and row.get(x_key, "") != ""


def detector_order(rows: List[Dict[str, Any]]) -> List[str]:
    labels = sorted({str(row["source_label"]) for row in rows})
    preferred = ["fastrcnn", "rtdetr", "yolo"]
    ordered: List[str] = []

    for preferred_key in preferred:
        for label in labels:
            if preferred_key in norm(label) and label not in ordered:
                ordered.append(label)

    for label in labels:
        if label not in ordered:
            ordered.append(label)

    return ordered


def method_order(
    rows: List[Dict[str, Any]],
    selected_method_keys: Optional[Set[str]] = None,
) -> List[str]:
    pairs: Dict[str, int] = {}

    for row in rows:
        if row.get("curve_type") != "scope":
            continue

        key = canonical_method_key(
            row.get("ablation_key", row.get("ablation_label", ""))
        )

        if selected_method_keys is not None and key not in selected_method_keys:
            continue

        idx = int(row.get("ablation_index", 999))
        pairs[key] = min(idx, pairs.get(key, idx))

    preferred_rank = {
        "full": 0,
        "no_task": 1,
        "no_bg": 2,
    }

    return [
        key
        for key, _idx in sorted(
            pairs.items(),
            key=lambda item: (
                preferred_rank.get(item[0], 999),
                item[1],
                item[0],
            ),
        )
    ]


def sort_curve_points(
    group: List[Dict[str, Any]],
    x_key: str,
) -> List[Dict[str, Any]]:
    return sorted(
        group,
        key=lambda row: (
            float(row[x_key]),
            float(row.get("level", 0.0) or 0.0),
            int(row.get("ablation_index", 999)),
            str(row.get("base_name", "")),
        ),
    )


def baseline_key(row: Dict[str, Any]) -> Any:
    level = row.get("level", "")

    try:
        level_float = float(level)
        if abs(level_float) > 1e-12:
            return ("level", round(level_float, 8))
    except Exception:
        pass

    base_name = str(row.get("base_name", ""))
    if base_name:
        return ("base", base_name)

    return ("experiment", str(row.get("experiment_folder", "")))


def deduplicate_baseline_points(
    group: List[Dict[str, Any]],
    x_key: str,
) -> List[Dict[str, Any]]:
    candidates = sorted(
        group,
        key=lambda row: (
            baseline_key(row),
            int(row.get("ablation_index", 999)),
            float(row[x_key]) if row.get(x_key, "") != "" else math.inf,
        ),
    )

    out: List[Dict[str, Any]] = []
    seen = set()

    for row in candidates:
        key = baseline_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    return sort_curve_points(out, x_key=x_key)


def get_curve_rows(
    rows: List[Dict[str, Any]],
    codec: str,
    detector_label: str,
    method_key: str,
    metric: str,
    x_key: str,
) -> List[Dict[str, Any]]:
    if method_key == "baseline":
        group = [
            row for row in rows
            if is_ready(row, metric, x_key)
            and row.get("codec") == codec
            and row.get("source_label") == detector_label
            and row.get("curve_type") == "baseline"
        ]
        return deduplicate_baseline_points(group, x_key=x_key)

    group = [
        row for row in rows
        if is_ready(row, metric, x_key)
        and row.get("codec") == codec
        and row.get("source_label") == detector_label
        and row.get("curve_type") == "scope"
        and canonical_method_key(row.get("ablation_key", "")) == method_key
    ]
    return sort_curve_points(group, x_key=x_key)


def collapse_duplicate_x(
    x: Sequence[float],
    y: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse identical rates by averaging the metric values.
    This is only for numerical stability when duplicate rate entries exist.
    """
    grouped: Dict[float, List[float]] = {}

    for xi, yi in zip(x, y):
        key = round(float(xi), 12)
        grouped.setdefault(key, []).append(float(yi))

    xs = np.array(sorted(grouped.keys()), dtype=np.float64)
    ys = np.array(
        [float(np.mean(grouped[key])) for key in sorted(grouped.keys())],
        dtype=np.float64,
    )

    finite = np.isfinite(xs) & np.isfinite(ys) & (xs > 0)
    return xs[finite], ys[finite]


def curve_arrays(
    curve_rows: Sequence[Dict[str, Any]],
    metric: str,
    x_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x = [float(row[x_key]) for row in curve_rows]
    y = [float(row[metric]) for row in curve_rows]
    return collapse_duplicate_x(x, y)


def build_interpolator(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
) -> Callable[[np.ndarray], np.ndarray]:
    if len(x) < 2:
        raise ValueError("At least two distinct x points are required.")

    if np.any(np.diff(x) <= 0):
        raise ValueError("Interpolation x values must be strictly increasing.")

    if kind == "pchip":
        if PchipInterpolator is None:
            raise RuntimeError(
                "SciPy is required for --interp pchip. "
                "Install scipy or use --interp linear."
            )
        interpolator = PchipInterpolator(x, y, extrapolate=False)

        def evaluate(query: np.ndarray) -> np.ndarray:
            return np.asarray(interpolator(query), dtype=np.float64)

        return evaluate

    if kind == "linear":
        def evaluate(query: np.ndarray) -> np.ndarray:
            return np.interp(query, x, y)

        return evaluate

    raise ValueError(f"Unknown interpolation kind: {kind}")


def make_rate_grid(
    xmin: float,
    xmax: float,
    n: int,
    spacing: str,
) -> np.ndarray:
    if not (np.isfinite(xmin) and np.isfinite(xmax)):
        raise ValueError("Grid limits must be finite.")
    if xmin <= 0:
        raise ValueError("Rate values must be positive.")
    if xmax <= xmin:
        raise ValueError("The common rate interval is empty.")
    if n < 2:
        raise ValueError("At least two grid points are required.")

    if spacing == "log":
        return np.geomspace(xmin, xmax, n)
    if spacing == "linear":
        return np.linspace(xmin, xmax, n)

    raise ValueError(f"Unknown spacing: {spacing}")


def trapezoid_average(
    values: np.ndarray,
    coordinate: np.ndarray,
) -> float:
    if len(values) < 2:
        return float("nan")

    width = float(coordinate[-1] - coordinate[0])
    if width <= 0:
        return float("nan")

    if hasattr(np, "trapezoid"):
        integral = float(np.trapezoid(values, coordinate))
    else:
        integral = float(np.trapz(values, coordinate))

    return integral / width


def average_metric_delta_equal_rate(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    interpolation: str,
    dense_points: int,
) -> Dict[str, Any]:
    xmin = max(float(np.min(reference_x)), float(np.min(test_x)))
    xmax = min(float(np.max(reference_x)), float(np.max(test_x)))

    result: Dict[str, Any] = {
        "status": "ok",
        "common_rate_min": xmin,
        "common_rate_max": xmax,
        "mean_metric_delta": "",
        "mean_metric_delta_x100": "",
        "min_metric_delta": "",
        "max_metric_delta": "",
    }

    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        result["status"] = "no_common_rate_range"
        return result

    ref_interp = build_interpolator(reference_x, reference_y, interpolation)
    test_interp = build_interpolator(test_x, test_y, interpolation)

    rate_grid = make_rate_grid(xmin, xmax, dense_points, spacing="log")
    ref_values = ref_interp(rate_grid)
    test_values = test_interp(rate_grid)
    delta = test_values - ref_values

    finite = np.isfinite(delta)
    if np.count_nonzero(finite) < 2:
        result["status"] = "interpolation_failed"
        return result

    log_rate = np.log(rate_grid[finite])
    delta = delta[finite]

    mean_delta = trapezoid_average(delta, log_rate)

    result.update(
        {
            "mean_metric_delta": mean_delta,
            "mean_metric_delta_x100": mean_delta * 100.0,
            "min_metric_delta": float(np.min(delta)),
            "max_metric_delta": float(np.max(delta)),
        }
    )
    return result


def pareto_monotone_curve(
    rate: np.ndarray,
    quality: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the non-dominated rate-quality envelope.

    A point is kept only if it improves quality over every lower-rate point.
    This makes quality strictly increasing and permits rate interpolation as
    a function of quality for BDBR.
    """
    rate, quality = collapse_duplicate_x(rate, quality)

    order = np.argsort(rate)
    rate = rate[order]
    quality = quality[order]

    kept_rate: List[float] = []
    kept_quality: List[float] = []
    best_quality = -math.inf

    for ri, qi in zip(rate, quality):
        if qi > best_quality + eps:
            kept_rate.append(float(ri))
            kept_quality.append(float(qi))
            best_quality = float(qi)

    return (
        np.asarray(kept_rate, dtype=np.float64),
        np.asarray(kept_quality, dtype=np.float64),
    )


def compute_bd_rate(
    reference_rate: np.ndarray,
    reference_quality: np.ndarray,
    test_rate: np.ndarray,
    test_quality: np.ndarray,
    interpolation: str,
    dense_points: int,
    min_points: int,
) -> Dict[str, Any]:
    """
    BDBR using the shared task-accuracy interval.

    Sign:
      negative -> test method uses less rate than the reference at equal quality
      positive -> test method uses more rate than the reference at equal quality
    """
    ref_rate, ref_quality = pareto_monotone_curve(
        reference_rate, reference_quality
    )
    test_rate, test_quality = pareto_monotone_curve(
        test_rate, test_quality
    )

    result: Dict[str, Any] = {
        "bdbr_percent": "",
        "common_quality_min": "",
        "common_quality_max": "",
        "reference_pareto_points": len(ref_rate),
        "test_pareto_points": len(test_rate),
        "bd_status": "ok",
    }

    if len(ref_rate) < min_points or len(test_rate) < min_points:
        result["bd_status"] = "too_few_pareto_points"
        return result

    qmin = max(float(np.min(ref_quality)), float(np.min(test_quality)))
    qmax = min(float(np.max(ref_quality)), float(np.max(test_quality)))

    result["common_quality_min"] = qmin
    result["common_quality_max"] = qmax

    if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
        result["bd_status"] = "no_common_quality_range"
        return result

    # Interpolate log(rate) as a function of quality.
    ref_interp = build_interpolator(
        ref_quality, np.log(ref_rate), interpolation
    )
    test_interp = build_interpolator(
        test_quality, np.log(test_rate), interpolation
    )

    quality_grid = np.linspace(qmin, qmax, dense_points)
    ref_log_rate = ref_interp(quality_grid)
    test_log_rate = test_interp(quality_grid)
    log_rate_difference = test_log_rate - ref_log_rate

    finite = np.isfinite(log_rate_difference)
    if np.count_nonzero(finite) < 2:
        result["bd_status"] = "interpolation_failed"
        return result

    average_log_difference = trapezoid_average(
        log_rate_difference[finite],
        quality_grid[finite],
    )

    result["bdbr_percent"] = (
        math.exp(average_log_difference) - 1.0
    ) * 100.0
    return result


def build_curves_for_group(
    rows: List[Dict[str, Any]],
    codec: str,
    detector_label: str,
    method_keys: Sequence[str],
    metric: str,
    x_key: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for method_key in ["baseline", *method_keys]:
        group = get_curve_rows(
            rows=rows,
            codec=codec,
            detector_label=detector_label,
            method_key=method_key,
            metric=metric,
            x_key=x_key,
        )

        if not group:
            continue

        x, y = curve_arrays(group, metric=metric, x_key=x_key)
        if len(x) >= 2:
            curves[method_key] = (x, y)

    return curves
