from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .common import methods_for_codec, pretty_detector, pretty_method
from .curves import (
    average_metric_delta_equal_rate, build_curves_for_group, build_interpolator,
    compute_bd_rate, make_rate_grid,
)


def build_interpolated_point_table(
    rows: List[Dict[str, Any]],
    codecs: Sequence[str],
    detectors: Sequence[str],
    table_methods: Sequence[str],
    metric: str,
    x_key: str,
    interpolation: str,
    table_points: int,
    table_spacing: str,
    allow_missing_methods: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    table_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []

    for codec in codecs:
        if codec == "cheng" and not any(row.get("codec") == codec for row in rows):
            continue
        codec_methods = methods_for_codec(codec, table_methods)
        if not codec_methods:
            continue
        for detector_label in detectors:
            curves = build_curves_for_group(
                rows=rows,
                codec=codec,
                detector_label=detector_label,
                method_keys=codec_methods,
                metric=metric,
                x_key=x_key,
            )

            required = {"baseline", *codec_methods}
            available = set(curves)
            missing = sorted(required - available)

            status: Dict[str, Any] = {
                "codec": codec,
                "detector_label": detector_label,
                "detector": pretty_detector(detector_label),
                "available_curves": ",".join(sorted(available)),
                "missing_curves": ",".join(missing),
                "common_rate_min": "",
                "common_rate_max": "",
                "status": "ok",
            }

            if "baseline" not in curves:
                status["status"] = "missing_baseline"
                status_rows.append(status)
                continue

            if missing and not allow_missing_methods:
                status["status"] = "missing_required_curves"
                status_rows.append(status)
                continue

            selected_curve_names = [
                name for name in ["baseline", *codec_methods]
                if name in curves
            ]

            xmin = max(
                float(np.min(curves[name][0]))
                for name in selected_curve_names
            )
            xmax = min(
                float(np.max(curves[name][0]))
                for name in selected_curve_names
            )

            status["common_rate_min"] = xmin
            status["common_rate_max"] = xmax

            if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
                status["status"] = "no_common_rate_range"
                status_rows.append(status)
                continue

            query_rates = make_rate_grid(
                xmin=xmin,
                xmax=xmax,
                n=table_points,
                spacing=table_spacing,
            )

            interpolated: Dict[str, np.ndarray] = {}
            failed = False

            for name in selected_curve_names:
                x, y = curves[name]
                try:
                    interpolator = build_interpolator(
                        x, y, interpolation
                    )
                    values = interpolator(query_rates)
                except Exception as exc:
                    print(
                        f"[WARN] interpolation failed for "
                        f"{codec}/{detector_label}/{name}: {exc}"
                    )
                    failed = True
                    break

                if not np.all(np.isfinite(values)):
                    failed = True
                    break

                interpolated[name] = values

            if failed:
                status["status"] = "interpolation_failed"
                status_rows.append(status)
                continue

            for point_index, rate in enumerate(query_rates):
                row: Dict[str, Any] = {
                    "codec": codec,
                    "detector_label": detector_label,
                    "detector": pretty_detector(detector_label),
                    "point_index": point_index,
                    x_key: float(rate),
                    "common_rate_min": xmin,
                    "common_rate_max": xmax,
                }

                for name in ["baseline", *codec_methods]:
                    row[name] = (
                        float(interpolated[name][point_index])
                        if name in interpolated
                        else ""
                    )

                baseline_value = row.get("baseline", "")
                full_value = row.get("full", "")

                for method_key in codec_methods:
                    method_value = row.get(method_key, "")

                    row[f"delta_{method_key}_vs_baseline"] = (
                        float(method_value) - float(baseline_value)
                        if method_value != "" and baseline_value != ""
                        else ""
                    )

                    row[f"delta_{method_key}_vs_baseline_x100"] = (
                        100.0 * (
                            float(method_value) - float(baseline_value)
                        )
                        if method_value != "" and baseline_value != ""
                        else ""
                    )

                    if (
                        method_key != "full"
                        and full_value != ""
                        and method_value != ""
                    ):
                        row[f"delta_full_vs_{method_key}"] = (
                            float(full_value) - float(method_value)
                        )
                        row[f"delta_full_vs_{method_key}_x100"] = (
                            100.0 * (
                                float(full_value) - float(method_value)
                            )
                        )

                table_rows.append(row)

            status_rows.append(status)

    return table_rows, status_rows


def build_pairwise_metrics(
    rows: List[Dict[str, Any]],
    codecs: Sequence[str],
    detectors: Sequence[str],
    method_keys: Sequence[str],
    metric: str,
    x_key: str,
    interpolation: str,
    dense_points: int,
    min_bd_points: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    for codec in codecs:
        if codec == "cheng" and not any(row.get("codec") == codec for row in rows):
            continue
        codec_methods = methods_for_codec(codec, method_keys)
        if not codec_methods:
            continue

        pair_specs: List[Tuple[str, str]] = [
            (method_key, "baseline") for method_key in codec_methods
        ]
        if "full" in codec_methods:
            pair_specs.extend(
                (method_key, "full")
                for method_key in codec_methods
                if method_key != "full"
            )

        for detector_label in detectors:
            curves = build_curves_for_group(
                rows=rows,
                codec=codec,
                detector_label=detector_label,
                method_keys=codec_methods,
                metric=metric,
                x_key=x_key,
            )

            for test_method, reference_method in pair_specs:
                row: Dict[str, Any] = {
                    "codec": codec,
                    "detector_label": detector_label,
                    "detector": pretty_detector(detector_label),
                    "test_method": test_method,
                    "test_method_display": pretty_method(test_method),
                    "reference_method": reference_method,
                    "reference_method_display": pretty_method(reference_method),
                    "interpolation": interpolation,
                    "metric": metric,
                    "rate_key": x_key,
                    "reference_points": "",
                    "test_points": "",
                    "common_rate_min": "",
                    "common_rate_max": "",
                    "mean_metric_delta_equal_rate": "",
                    "mean_metric_delta_equal_rate_x100": "",
                    "min_metric_delta_equal_rate": "",
                    "max_metric_delta_equal_rate": "",
                    "equal_rate_status": "",
                    "common_quality_min": "",
                    "common_quality_max": "",
                    "reference_pareto_points": "",
                    "test_pareto_points": "",
                    "bdbr_percent": "",
                    "bd_status": "",
                }

                if reference_method not in curves:
                    row["equal_rate_status"] = "missing_reference_curve"
                    row["bd_status"] = "missing_reference_curve"
                    output.append(row)
                    continue

                if test_method not in curves:
                    row["equal_rate_status"] = "missing_test_curve"
                    row["bd_status"] = "missing_test_curve"
                    output.append(row)
                    continue

                reference_x, reference_y = curves[reference_method]
                test_x, test_y = curves[test_method]

                row["reference_points"] = len(reference_x)
                row["test_points"] = len(test_x)

                equal_rate = average_metric_delta_equal_rate(
                    reference_x=reference_x,
                    reference_y=reference_y,
                    test_x=test_x,
                    test_y=test_y,
                    interpolation=interpolation,
                    dense_points=dense_points,
                )

                row["common_rate_min"] = equal_rate["common_rate_min"]
                row["common_rate_max"] = equal_rate["common_rate_max"]
                row["mean_metric_delta_equal_rate"] = equal_rate[
                    "mean_metric_delta"
                ]
                row["mean_metric_delta_equal_rate_x100"] = equal_rate[
                    "mean_metric_delta_x100"
                ]
                row["min_metric_delta_equal_rate"] = equal_rate[
                    "min_metric_delta"
                ]
                row["max_metric_delta_equal_rate"] = equal_rate[
                    "max_metric_delta"
                ]
                row["equal_rate_status"] = equal_rate["status"]

                bd_rate = compute_bd_rate(
                    reference_rate=reference_x,
                    reference_quality=reference_y,
                    test_rate=test_x,
                    test_quality=test_y,
                    interpolation=interpolation,
                    dense_points=dense_points,
                    min_points=min_bd_points,
                )

                row["common_quality_min"] = bd_rate["common_quality_min"]
                row["common_quality_max"] = bd_rate["common_quality_max"]
                row["reference_pareto_points"] = bd_rate[
                    "reference_pareto_points"
                ]
                row["test_pareto_points"] = bd_rate[
                    "test_pareto_points"
                ]
                row["bdbr_percent"] = bd_rate["bdbr_percent"]
                row["bd_status"] = bd_rate["bd_status"]

                output.append(row)

    return output
