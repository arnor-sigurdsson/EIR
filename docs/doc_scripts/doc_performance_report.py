import argparse
import csv
import datetime
import json
import os
from typing import Any


def find_validation_logs(root_dir: str) -> list[str]:
    results = []
    for root, _, files in os.walk(root_dir):
        if "validation_average_history.log" in files:
            results.append(os.path.join(root, "validation_average_history.log"))
    return results


def extract_performance_metrics(file_path: str) -> dict[str, Any]:
    metrics = {
        "max_perf_average": None,
        "final_perf_average": None,
        "min_loss_average": None,
        "final_loss_average": None,
        "total_iterations": 0,
    }

    try:
        with open(file_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if not rows:
                print(f"Warning: No data in {file_path}")
                return metrics

            perf_values = [
                float(row["perf-average"]) for row in rows if "perf-average" in row
            ]
            loss_values = [
                float(row["loss-average"]) for row in rows if "loss-average" in row
            ]

            if perf_values:
                metrics["max_perf_average"] = max(perf_values)
                metrics["final_perf_average"] = perf_values[-1]

            if loss_values:
                metrics["min_loss_average"] = min(loss_values)
                metrics["final_loss_average"] = loss_values[-1]

            metrics["total_iterations"] = (
                int(rows[-1]["iteration"]) if rows and "iteration" in rows[-1] else 0
            )

    except Exception as e:
        print(f"Warning: Error processing {file_path}: {e}")

    return metrics


def get_experiment_id(file_path: str, root_dir: str) -> str:
    abs_file = os.path.abspath(file_path)
    abs_root = os.path.abspath(root_dir)

    exp_dir = os.path.dirname(abs_file)

    rel_path = os.path.relpath(exp_dir, abs_root)

    return rel_path


def collect_performance_data(root_dir: str) -> dict[str, dict[str, Any]]:
    log_files = find_validation_logs(root_dir)
    print(f"Found {len(log_files)} validation log files.")

    results = {}

    for log_file in log_files:
        exp_id = get_experiment_id(log_file, root_dir)
        metrics = extract_performance_metrics(log_file)

        if metrics["max_perf_average"] is not None:
            results[exp_id] = {
                **metrics,
                "log_file": os.path.relpath(log_file, os.path.abspath(root_dir)),
            }
        else:
            print(f"Warning: No valid performance metrics found in {log_file}")

    return results


def generate_report(data: dict[str, Any], output_format: str = "json") -> str:
    if output_format == "json":
        return json.dumps(data, indent=4, sort_keys=True)
    elif output_format == "csv":
        headers = [
            "experiment_id",
            "max_perf_average",
            "final_perf_average",
            "min_loss_average",
            "final_loss_average",
            "total_iterations",
            "log_file",
        ]
        lines = [",".join(headers)]

        for exp_id in sorted(data["experiments"].keys()):
            info = data["experiments"][exp_id]
            line_values = [
                exp_id,
                str(info["max_perf_average"]),
                str(info["final_perf_average"]),
                str(info["min_loss_average"]),
                str(info["final_loss_average"]),
                str(info["total_iterations"]),
                info["log_file"],
            ]
            lines.append(",".join(line_values))

        return "\n".join(lines)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance report for EIR tutorial runs"
    )
    parser.add_argument(
        "--root-dir",
        default="eir_tutorials/tutorial_runs",
        help="Root directory containing the tutorial runs",
    )
    parser.add_argument(
        "--output", default="performance_report.json", help="Output file path"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (json or csv)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        print(f"Error: Root directory '{args.root_dir}' does not exist.")
        return

    print(f"Collecting performance data from {args.root_dir}...")

    experiment_data = collect_performance_data(root_dir=args.root_dir)

    report = {
        "metadata": {
            "num_experiments": len(experiment_data),
            "date_generated": datetime.datetime.now().isoformat(),
        },
        "experiments": experiment_data,
    }

    report_content = generate_report(data=report, output_format=args.format)

    with open(args.output, "w") as f:
        f.write(report_content)

    print(f"Report generated at {args.output}")
    print(f"Found {len(experiment_data)} experiments with valid metrics.")


if __name__ == "__main__":
    main()
