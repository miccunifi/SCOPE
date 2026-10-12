from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .common import X_KEY_LABELS, methods_for_codec, pretty_method
from .results import count_values


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def format_number(value: Any, digits: int = 4) -> str:
    if value in ["", None]:
        return "--"
    try:
        number = float(value)
    except Exception:
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def write_latex_interpolated_tables(
    table_rows: List[Dict[str, Any]],
    codecs: Sequence[str],
    methods: Sequence[str],
    x_key: str,
    out_dir: Path,
    rate_digits: int,
    metric_digits: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    codec_pretty = {
        "jpeg": "JPEG",
        "bpg": "BPG",
        "bmshj": "BMSHJ",
        "cheng": "Cheng",
    }

    for codec in codecs:
        codec_methods = methods_for_codec(codec, methods)
        if not codec_methods:
            continue
        codec_rows = [
            row for row in table_rows
            if row.get("codec") == codec
        ]
        if not codec_rows:
            continue

        headers = ["Detector", X_KEY_LABELS.get(x_key, x_key), "Baseline"]
        headers.extend(pretty_method(method) for method in codec_methods)

        column_spec = "ll" + "r" * (1 + len(codec_methods))

        lines: List[str] = []
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(
            rf"\caption{{Interpolated equal-rate comparison for "
            rf"{codec_pretty.get(codec, codec)}. All methods are evaluated "
            rf"at the same interpolated rate values inside the common "
            rf"rate interval.}}"
        )
        lines.append(
            rf"\label{{tab:interpolated_{latex_escape(codec)}}}"
        )
        lines.append(r"\small")
        lines.append(r"\setlength{\tabcolsep}{6pt}")
        lines.append(rf"\begin{{tabular}}{{{column_spec}}}")
        lines.append(r"\toprule")
        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\midrule")

        detector_labels = []
        for row in codec_rows:
            detector = str(row["detector"])
            if detector not in detector_labels:
                detector_labels.append(detector)

        for detector_index, detector in enumerate(detector_labels):
            detector_rows = [
                row for row in codec_rows
                if row.get("detector") == detector
            ]
            detector_rows = sorted(
                detector_rows,
                key=lambda row: int(row["point_index"]),
            )

            for row_index, row in enumerate(detector_rows):
                detector_cell = (
                    latex_escape(detector) if row_index == 0 else ""
                )
                values = [
                    detector_cell,
                    format_number(row.get(x_key), rate_digits),
                    format_number(row.get("baseline"), metric_digits),
                ]
                values.extend(
                    format_number(row.get(method), metric_digits)
                    for method in methods
                )
                lines.append(" & ".join(values) + r" \\")

            if detector_index != len(detector_labels) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")
        lines.append("")

        path = out_dir / f"interpolated_{codec}.tex"
        path.write_text("\n".join(lines))


def build_bdbr_summary_rows(
    pairwise_metrics: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Keep only valid BDBR comparisons in a compact publication-oriented table.
    Negative BDBR means bitrate saving at the same downstream-task accuracy.
    """
    rows: List[Dict[str, Any]] = []

    for item in pairwise_metrics:
        rows.append(
            {
                "codec": item.get("codec", ""),
                "detector": item.get("detector", ""),
                "test_method": item.get("test_method_display", ""),
                "reference_method": item.get("reference_method_display", ""),
                "bdbr_percent": item.get("bdbr_percent", ""),
                "common_accuracy_min": item.get("common_quality_min", ""),
                "common_accuracy_max": item.get("common_quality_max", ""),
                "reference_pareto_points": item.get("reference_pareto_points", ""),
                "test_pareto_points": item.get("test_pareto_points", ""),
                "status": item.get("bd_status", ""),
            }
        )

    return rows


def write_latex_bdbr_table(
    rows: Sequence[Dict[str, Any]],
    path: Path,
    digits: int = 2,
) -> None:
    """
    Write one compact LaTeX table containing all BDBR comparisons.

    Interpretation:
      BDBR < 0: bitrate saving at equal downstream-task accuracy.
      BDBR > 0: bitrate increase at equal downstream-task accuracy.
    """
    valid_rows = [
        row for row in rows
        if row.get("status") == "ok"
        and row.get("bdbr_percent", "") not in ["", None]
    ]

    codec_pretty = {
        "jpeg": "JPEG",
        "bpg": "BPG",
        "bmshj": "BMSHJ",
        "cheng": "Cheng",
    }

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Bjøntegaard Delta Bitrate (BDBR) computed using "
        r"downstream-task accuracy as the quality axis. Negative values "
        r"indicate bitrate savings at the same accuracy.}"
    )
    lines.append(r"\label{tab:ablation_bdbr}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\begin{tabular}{lllr}")
    lines.append(r"\toprule")
    lines.append(r"Codec & Detector & Comparison & BDBR (\%) \\")
    lines.append(r"\midrule")

    for index, row in enumerate(valid_rows):
        codec = codec_pretty.get(
            str(row.get("codec", "")),
            str(row.get("codec", "")),
        )
        detector = str(row.get("detector", ""))
        comparison = (
            f"{row.get('test_method', '')} vs. "
            f"{row.get('reference_method', '')}"
        )
        value = format_number(row.get("bdbr_percent"), digits)

        lines.append(
            f"{latex_escape(codec)} & "
            f"{latex_escape(detector)} & "
            f"{latex_escape(comparison)} & "
            f"{value} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_diagnostics(
    rows: List[Dict[str, Any]],
    table_status: List[Dict[str, Any]],
    path: Path,
) -> None:
    lines: List[str] = []

    lines.append(f"rows: {len(rows)}")
    lines.append(f"ablation_label: {count_values(rows, 'ablation_label')}")
    lines.append(f"ablation_key: {count_values(rows, 'ablation_key')}")
    lines.append(f"codecs: {count_values(rows, 'codec')}")
    lines.append(f"families: {count_values(rows, 'family')}")
    lines.append(f"detectors: {count_values(rows, 'source_label')}")
    lines.append(f"curve_type: {count_values(rows, 'curve_type')}")
    lines.append("")
    lines.append("Interpolation-table status:")

    for status in table_status:
        lines.append(
            "  "
            f"{status.get('codec')}/{status.get('detector')}: "
            f"{status.get('status')} | "
            f"available={status.get('available_curves')} | "
            f"missing={status.get('missing_curves')} | "
            f"range=[{status.get('common_rate_min')}, "
            f"{status.get('common_rate_max')}]"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
