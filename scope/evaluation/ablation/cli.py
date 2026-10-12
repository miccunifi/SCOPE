from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .common import CODEC_ORDER, METRIC_KEYS, canonical_method_key, parse_labeled_path
from .curves import PchipInterpolator, detector_order, method_order
from .comparisons import build_interpolated_point_table, build_pairwise_metrics
from .results import (
    collect_rows, count_values, load_coco_image_size_map,
    write_collected_results, write_dict_rows,
)
from .tables import (
    build_bdbr_summary_rows, write_diagnostics, write_latex_bdbr_table,
    write_latex_interpolated_tables,
)


def parse_method_list(value: str) -> List[str]:
    methods: List[str] = []
    for item in value.split(","):
        key = canonical_method_key(item.strip())
        if key and key not in methods:
            methods.append(key)
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect ablation curves, interpolate every method at common "
            "rate points, write CSV/LaTeX tables, and compute equal-rate "
            "metric differences and BD-rate."
        )
    )

    parser.add_argument(
        "--detector-dir",
        "--eval-dir",
        "--eval-dirs",
        nargs="+",
        required=True,
        help=(
            "Old format: fastrcnn=path rtdetr=path yolo=path. "
            "Unified format: full:fastrcnn=path "
            "no_task:fastrcnn=path no_bg:fastrcnn=path."
        ),
    )
    parser.add_argument("--annotations-json", default=None)
    parser.add_argument(
        "--out-dir",
        default="tables_ablation_interpolated",
    )
    parser.add_argument(
        "--metric",
        default="AP",
        choices=METRIC_KEYS,
    )
    parser.add_argument(
        "--x-key",
        default="dataset_bpp",
        choices=[
            "dataset_bpp",
            "avg_bpp_per_image",
            "avg_kb_per_image",
        ],
    )
    parser.add_argument(
        "--scope-prefix", "--preedit-prefix",
        dest="scope_prefix",
        default="scope_,preedit_",
    )
    parser.add_argument(
        "--table-methods",
        default="full,no_task,no_bg",
        help=(
            "Comma-separated pre-edit methods to include in the common-rate "
            "table. Default: full,no_task,no_bg"
        ),
    )
    parser.add_argument(
        "--allow-missing-methods",
        action="store_true",
        help=(
            "Build a table from the available methods even when one requested "
            "method is missing. By default, the group is skipped."
        ),
    )
    parser.add_argument(
        "--interp",
        choices=["pchip", "linear"],
        default="pchip",
        help=(
            "Interpolation used for equal-rate tables and BD-rate. "
            "PCHIP avoids cubic-polynomial overshoot."
        ),
    )
    parser.add_argument(
        "--table-points",
        type=int,
        default=5,
        help=(
            "Number of equal-rate rows per codec/detector in the publication "
            "table. Default: 5."
        ),
    )
    parser.add_argument(
        "--table-spacing",
        choices=["log", "linear"],
        default="log",
        help=(
            "Spacing of the common rate points. Log spacing is recommended "
            "for bitrate curves."
        ),
    )
    parser.add_argument(
        "--dense-points",
        type=int,
        default=1001,
        help=(
            "Dense integration grid for average equal-rate differences and "
            "BD-rate. Default: 1001."
        ),
    )
    parser.add_argument(
        "--min-bd-points",
        type=int,
        default=3,
        help=(
            "Minimum number of non-dominated points required on each curve "
            "for BDBR. Default: 3."
        ),
    )
    parser.add_argument("--rate-digits", type=int, default=4)
    parser.add_argument("--metric-digits", type=int, default=4)

    args = parser.parse_args()

    if args.table_points < 2:
        raise SystemExit("--table-points must be at least 2.")
    if args.dense_points < 20:
        raise SystemExit("--dense-points should be at least 20.")
    if args.min_bd_points < 2:
        raise SystemExit("--min-bd-points must be at least 2.")
    if args.interp == "pchip" and PchipInterpolator is None:
        raise SystemExit(
            "SciPy is required for --interp pchip. "
            "Install scipy or run with --interp linear."
        )

    labeled_dirs = [parse_labeled_path(value) for value in args.detector_dir]
    out_dir = Path(args.out_dir)

    image_size_map = load_coco_image_size_map(args.annotations_json)

    rows = collect_rows(
        labeled_dirs=labeled_dirs,
        image_size_map=image_size_map,
        scope_prefix=args.scope_prefix,
    )

    if not rows:
        raise SystemExit("No summary.json files found.")

    table_methods = parse_method_list(args.table_methods)
    available_methods = method_order(rows)

    missing_globally = [
        method for method in table_methods
        if method not in available_methods
    ]
    if missing_globally:
        print(
            f"[WARN] requested table methods not found globally: "
            f"{missing_globally}"
        )

    detectors = detector_order(rows)

    write_collected_results(
        rows,
        out_dir / "collected_results.csv",
    )

    interpolated_rows, table_status = build_interpolated_point_table(
        rows=rows,
        codecs=CODEC_ORDER,
        detectors=detectors,
        table_methods=table_methods,
        metric=args.metric,
        x_key=args.x_key,
        interpolation=args.interp,
        table_points=args.table_points,
        table_spacing=args.table_spacing,
        allow_missing_methods=args.allow_missing_methods,
    )

    interpolated_fields = [
        "codec",
        "detector_label",
        "detector",
        "point_index",
        args.x_key,
        "common_rate_min",
        "common_rate_max",
        "baseline",
        *table_methods,
    ]

    for method in table_methods:
        interpolated_fields.extend(
            [
                f"delta_{method}_vs_baseline",
                f"delta_{method}_vs_baseline_x100",
            ]
        )
        if method != "full" and "full" in table_methods:
            interpolated_fields.extend(
                [
                    f"delta_full_vs_{method}",
                    f"delta_full_vs_{method}_x100",
                ]
            )

    write_dict_rows(
        interpolated_rows,
        out_dir / "interpolated_equal_rate_points.csv",
        interpolated_fields,
    )

    status_fields = [
        "codec",
        "detector_label",
        "detector",
        "available_curves",
        "missing_curves",
        "common_rate_min",
        "common_rate_max",
        "status",
    ]
    write_dict_rows(
        table_status,
        out_dir / "interpolation_status.csv",
        status_fields,
    )

    pairwise_metrics = build_pairwise_metrics(
        rows=rows,
        codecs=CODEC_ORDER,
        detectors=detectors,
        method_keys=table_methods,
        metric=args.metric,
        x_key=args.x_key,
        interpolation=args.interp,
        dense_points=args.dense_points,
        min_bd_points=args.min_bd_points,
    )

    pairwise_fields = [
        "codec",
        "detector_label",
        "detector",
        "test_method",
        "test_method_display",
        "reference_method",
        "reference_method_display",
        "interpolation",
        "metric",
        "rate_key",
        "reference_points",
        "test_points",
        "common_rate_min",
        "common_rate_max",
        "mean_metric_delta_equal_rate",
        "mean_metric_delta_equal_rate_x100",
        "min_metric_delta_equal_rate",
        "max_metric_delta_equal_rate",
        "equal_rate_status",
        "common_quality_min",
        "common_quality_max",
        "reference_pareto_points",
        "test_pareto_points",
        "bdbr_percent",
        "bd_status",
    ]
    write_dict_rows(
        pairwise_metrics,
        out_dir / "pairwise_equal_rate_and_bdbr.csv",
        pairwise_fields,
    )

    bdbr_summary_rows = build_bdbr_summary_rows(pairwise_metrics)
    bdbr_summary_fields = [
        "codec",
        "detector",
        "test_method",
        "reference_method",
        "bdbr_percent",
        "common_accuracy_min",
        "common_accuracy_max",
        "reference_pareto_points",
        "test_pareto_points",
        "status",
    ]
    write_dict_rows(
        bdbr_summary_rows,
        out_dir / "bdbr_summary.csv",
        bdbr_summary_fields,
    )
    write_latex_bdbr_table(
        rows=bdbr_summary_rows,
        path=out_dir / "latex" / "bdbr_summary.tex",
        digits=2,
    )

    write_latex_interpolated_tables(
        table_rows=interpolated_rows,
        codecs=CODEC_ORDER,
        methods=table_methods,
        x_key=args.x_key,
        out_dir=out_dir / "latex",
        rate_digits=args.rate_digits,
        metric_digits=args.metric_digits,
    )

    write_diagnostics(
        rows=rows,
        table_status=table_status,
        path=out_dir / "diagnostics.txt",
    )

    print(f"Collected rows: {len(rows)}")
    print("Methods:", count_values(rows, "ablation_key"))
    print("Codecs:", count_values(rows, "codec"))
    print("Detectors:", count_values(rows, "source_label"))
    print("")
    print("Wrote:")
    print(f"  {out_dir / 'collected_results.csv'}")
    print(f"  {out_dir / 'interpolated_equal_rate_points.csv'}")
    print(f"  {out_dir / 'interpolation_status.csv'}")
    print(f"  {out_dir / 'pairwise_equal_rate_and_bdbr.csv'}")
    print(f"  {out_dir / 'bdbr_summary.csv'}")
    print(f"  {out_dir / 'latex' / 'bdbr_summary.tex'}")
    print(f"  {out_dir / 'latex'}")
    print(f"  {out_dir / 'diagnostics.txt'}")
    print("")
    print("BDBR sign convention:")
    print("  negative = test method requires less rate than the reference")
    print("  positive = test method requires more rate than the reference")
