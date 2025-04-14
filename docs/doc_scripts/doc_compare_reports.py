import argparse
import datetime
import json
from typing import Any

import numpy as np
import scipy.stats as stats


class TermColors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def load_report(file_path: str) -> dict[str, Any]:
    with open(file_path) as f:
        return json.load(f)


def identify_experiment_sets(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    baseline_exps = set(baseline["experiments"].keys())
    current_exps = set(current["experiments"].keys())

    common_experiments = baseline_exps.intersection(current_exps)
    new_experiments = current_exps.difference(baseline_exps)
    removed_experiments = baseline_exps.difference(current_exps)

    return common_experiments, new_experiments, removed_experiments


def collect_max_perf_metrics(
    baseline: dict[str, Any], current: dict[str, Any], common_experiments: set[str]
) -> tuple[list[float], list[float]]:
    baseline_metrics = []
    current_metrics = []

    for exp_id in common_experiments:
        if exp_id.startswith("i_"):
            continue

        baseline_exp = baseline["experiments"][exp_id]
        current_exp = current["experiments"][exp_id]

        baseline_value = baseline_exp.get("max_perf_average")
        current_value = current_exp.get("max_perf_average")

        if baseline_value is not None and current_value is not None:
            baseline_metrics.append(baseline_value)
            current_metrics.append(current_value)

    return baseline_metrics, current_metrics


def perform_wilcoxon_test(
    baseline_metrics: list[float],
    current_metrics: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    if len(baseline_metrics) < 5:
        return {
            "significant": False,
            "p_value": None,
            "mean_difference": None,
            "sample_size": len(baseline_metrics),
            "enough_samples": False,
        }

    diff_array = np.array(current_metrics) - np.array(baseline_metrics)
    mean_diff = float(np.mean(diff_array))

    try:
        w_stat, p_value = stats.wilcoxon(current_metrics, baseline_metrics)
        p_value = float(p_value)
    except ValueError:
        _w_stat, p_value = 0, 1.0

    return {
        "significant": p_value < alpha,
        "p_value": p_value,
        "mean_difference": mean_diff,
        "sample_size": len(baseline_metrics),
        "enough_samples": True,
    }


def compare_metrics(
    baseline_exp: dict[str, Any], current_exp: dict[str, Any]
) -> dict[str, float]:
    metrics = [
        "max_perf_average",
        "final_perf_average",
        "min_loss_average",
        "final_loss_average",
    ]

    changes = {}

    for metric in metrics:
        baseline_value = baseline_exp.get(metric)
        current_value = current_exp.get(metric)

        if baseline_value is not None and current_value is not None:
            abs_change = current_value - baseline_value
            rel_change = (
                (abs_change / baseline_value) * 100
                if baseline_value != 0
                else float("inf")
            )

            changes[f"{metric}_abs_change"] = abs_change
            changes[f"{metric}_rel_change"] = rel_change

    baseline_iters = baseline_exp.get("total_iterations", 0)
    current_iters = current_exp.get("total_iterations", 0)
    changes["iterations_change"] = current_iters - baseline_iters

    return changes


def classify_change(change: dict[str, float], thresholds: dict[str, float]) -> str:
    perf_change = change.get("max_perf_average_abs_change", 0)
    loss_change = change.get("min_loss_average_abs_change", 0)

    if perf_change >= thresholds["significant_improvement"]:
        return "significant_improvement"
    elif perf_change >= thresholds["improvement"]:
        return "improvement"
    elif perf_change >= thresholds["minor_improvement"]:
        return "minor_improvement"
    elif perf_change <= -thresholds["significant_regression"]:
        return "significant_regression"
    elif perf_change <= -thresholds["regression"]:
        return "regression"
    elif perf_change <= -thresholds["minor_regression"]:
        return "minor_regression"
    else:
        if loss_change <= -thresholds["improvement"]:
            return "improvement"
        elif loss_change >= thresholds["regression"]:
            return "regression"
        else:
            return "unchanged"


def generate_experiment_comparison(
    baseline: dict[str, Any],
    current: dict[str, Any],
    common_experiments: set[str],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    comparison_results = []

    for exp_id in common_experiments:
        if exp_id.startswith("i_"):
            continue

        baseline_exp = baseline["experiments"][exp_id]
        current_exp = current["experiments"][exp_id]

        changes = compare_metrics(baseline_exp, current_exp)

        change_type = classify_change(changes, thresholds)

        comparison = {
            "experiment_id": exp_id,
            "change_type": change_type,
            "baseline": {
                "max_perf_average": baseline_exp.get("max_perf_average"),
                "min_loss_average": baseline_exp.get("min_loss_average"),
                "total_iterations": baseline_exp.get("total_iterations"),
            },
            "current": {
                "max_perf_average": current_exp.get("max_perf_average"),
                "min_loss_average": current_exp.get("min_loss_average"),
                "total_iterations": current_exp.get("total_iterations"),
            },
            "changes": changes,
        }

        comparison_results.append(comparison)

    def sort_key(item):
        change_type = item["change_type"]
        perf_change = item["changes"].get("max_perf_average_abs_change", 0)

        if "improvement" in change_type:
            return (-3, -abs(perf_change))
        elif "regression" in change_type:
            return (-1, abs(perf_change))
        else:
            return (-2, 0)

    return sorted(comparison_results, key=sort_key)


def generate_summary(
    comparison_results: list[dict[str, Any]],
    new_experiments: set[str],
    removed_experiments: set[str],
    statistical_test: dict[str, Any],
) -> dict[str, Any]:
    change_counts = {
        "significant_improvement": 0,
        "improvement": 0,
        "minor_improvement": 0,
        "unchanged": 0,
        "minor_regression": 0,
        "regression": 0,
        "significant_regression": 0,
    }

    total_perf_change = 0
    total_loss_change = 0
    count = 0

    top_improvements = []
    top_regressions = []

    for result in comparison_results:
        change_type = result["change_type"]
        change_counts[change_type] += 1

        if "max_perf_average_abs_change" in result["changes"]:
            total_perf_change += result["changes"]["max_perf_average_abs_change"]
            count += 1

        if "min_loss_average_abs_change" in result["changes"]:
            total_loss_change += result["changes"]["min_loss_average_abs_change"]

        if "improvement" in change_type:
            top_improvements.append(result)

        if "regression" in change_type:
            top_regressions.append(result)

    top_improvements.sort(
        key=lambda x: x["changes"].get("max_perf_average_abs_change", 0), reverse=True
    )
    top_regressions.sort(
        key=lambda x: x["changes"].get("max_perf_average_abs_change", 0)
    )

    top_improvements = top_improvements[:5]
    top_regressions = top_regressions[:5]

    avg_perf_change = total_perf_change / count if count > 0 else 0
    avg_loss_change = total_loss_change / count if count > 0 else 0

    summary = {
        "total_experiments_compared": len(comparison_results),
        "new_experiments": len(new_experiments),
        "removed_experiments": len(removed_experiments),
        "change_counts": change_counts,
        "average_changes": {
            "perf_average": avg_perf_change,
            "loss_average": avg_loss_change,
        },
        "top_improvements": [r["experiment_id"] for r in top_improvements],
        "top_regressions": [r["experiment_id"] for r in top_regressions],
        "statistical_test": statistical_test,
    }

    return summary


def print_colored_summary(
    summary: dict[str, Any],
    comparison_results: list[dict[str, Any]],
) -> None:
    c = TermColors

    print(f"\n{c.BOLD}{c.UNDERLINE}EIR Performance Comparison Summary{c.END}")
    print(f"\nCompared {summary['total_experiments_compared']} experiments")
    print(f"New experiments: {summary['new_experiments']}")
    print(f"Removed experiments: {summary['removed_experiments']}")

    print(f"\n{c.BOLD}Change Distribution:{c.END}")
    counts = summary["change_counts"]
    print(
        f"  {c.GREEN}Significant improvements: "
        f"{counts['significant_improvement']}{c.END}"
    )
    print(f"  {c.GREEN}Improvements: {counts['improvement']}{c.END}")
    print(f"  {c.GREEN}Minor improvements: {counts['minor_improvement']}{c.END}")
    print(f"  Unchanged: {counts['unchanged']}")
    print(f"  {c.RED}Minor regressions: {counts['minor_regression']}{c.END}")
    print(f"  {c.RED}Regressions: {counts['regression']}{c.END}")
    print(
        f"  {c.RED}Significant regressions: {counts['significant_regression']}{c.END}"
    )

    avg = summary["average_changes"]
    perf_color = (
        c.GREEN if avg["perf_average"] > 0 else c.RED if avg["perf_average"] < 0 else ""
    )
    loss_color = (
        c.GREEN if avg["loss_average"] < 0 else c.RED if avg["loss_average"] > 0 else ""
    )

    print(f"\n{c.BOLD}Average Changes:{c.END}")
    print(f"  Performance: {perf_color}{avg['perf_average']:.6f}{c.END}")
    print(f"  Loss: {loss_color}{avg['loss_average']:.6f}{c.END}")

    test = summary.get("statistical_test")
    if test:
        print(f"\n{c.BOLD}Statistical Analysis (max_perf_average):{c.END}")

        if not test.get("enough_samples", False):
            print(
                f"  Not enough samples for statistical testing "
                f"(n={test['sample_size']})"
            )
        else:
            if test["significant"]:
                sig_color = c.GREEN if test["mean_difference"] > 0 else c.RED
                sig_mark = "✓"
            else:
                sig_color = ""
                sig_mark = "✗"

            print(
                f"  Wilcoxon test: {sig_color}Significant: {sig_mark} "
                f"(p={test['p_value']:.4f}){c.END}"
            )
            print(f"  Mean difference: {test['mean_difference']:.6f}")
            print(f"  Sample size: {test['sample_size']}")

    if summary["top_improvements"]:
        print(f"\n{c.BOLD}{c.GREEN}Top Improvements:{c.END}")
        for exp_id in summary["top_improvements"]:
            result = next(
                (r for r in comparison_results if r["experiment_id"] == exp_id), None
            )
            if result:
                change = result["changes"].get("max_perf_average_abs_change", 0)
                rel_change = result["changes"].get("max_perf_average_rel_change", 0)
                print(f"  {exp_id}: +{change:.6f} ({rel_change:.2f}%)")

    if summary["top_regressions"]:
        print(f"\n{c.BOLD}{c.RED}Top Regressions:{c.END}")
        for exp_id in summary["top_regressions"]:
            result = next(
                (r for r in comparison_results if r["experiment_id"] == exp_id), None
            )
            if result:
                change = result["changes"].get("max_perf_average_abs_change", 0)
                rel_change = result["changes"].get("max_perf_average_rel_change", 0)
                print(f"  {exp_id}: {change:.6f} ({rel_change:.2f}%)")

    print("\n")


def create_comparison_report(
    baseline_report: dict[str, Any],
    current_report: dict[str, Any],
    thresholds: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    common_exps, new_exps, removed_exps = identify_experiment_sets(
        baseline=baseline_report, current=current_report
    )

    baseline_metrics, current_metrics = collect_max_perf_metrics(
        baseline=baseline_report, current=current_report, common_experiments=common_exps
    )

    statistical_test = perform_wilcoxon_test(
        baseline_metrics, current_metrics, alpha=alpha
    )

    comparison_results = generate_experiment_comparison(
        baseline=baseline_report,
        current=current_report,
        common_experiments=common_exps,
        thresholds=thresholds,
    )

    summary = generate_summary(
        comparison_results=comparison_results,
        new_experiments=new_exps,
        removed_experiments=removed_exps,
        statistical_test=statistical_test,
    )

    report = {
        "metadata": {
            "baseline_report": {
                "date_generated": baseline_report["metadata"].get("date_generated"),
                "num_experiments": baseline_report["metadata"].get("num_experiments"),
                "root_directory": baseline_report["metadata"].get("root_directory"),
            },
            "current_report": {
                "date_generated": current_report["metadata"].get("date_generated"),
                "num_experiments": current_report["metadata"].get("num_experiments"),
                "root_directory": current_report["metadata"].get("root_directory"),
            },
            "comparison_date": datetime.datetime.now().isoformat(),
            "thresholds": thresholds,
            "statistical_significance_alpha": alpha,
        },
        "summary": summary,
        "statistical_test": statistical_test,
        "new_experiments": list(new_exps),
        "removed_experiments": list(removed_exps),
        "comparison_results": comparison_results,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Compare two EIR performance reports")
    parser.add_argument("baseline", help="Path to the baseline report file")
    parser.add_argument("current", help="Path to the current report file")
    parser.add_argument(
        "--output", default="comparison_report.json", help="Output file path"
    )
    parser.add_argument(
        "--significant-improvement",
        type=float,
        default=0.05,
        help="Threshold for significant improvement (default: 0.05)",
    )
    parser.add_argument(
        "--improvement",
        type=float,
        default=0.01,
        help="Threshold for improvement (default: 0.01)",
    )
    parser.add_argument(
        "--minor-improvement",
        type=float,
        default=0.001,
        help="Threshold for minor improvement (default: 0.001)",
    )
    parser.add_argument(
        "--significant-regression",
        type=float,
        default=0.05,
        help="Threshold for significant regression (default: 0.05)",
    )
    parser.add_argument(
        "--regression",
        type=float,
        default=0.01,
        help="Threshold for regression (default: 0.01)",
    )
    parser.add_argument(
        "--minor-regression",
        type=float,
        default=0.001,
        help="Threshold for minor regression (default: 0.001)",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )

    args = parser.parse_args()

    if args.no_color:
        TermColors.GREEN = ""
        TermColors.RED = ""
        TermColors.YELLOW = ""
        TermColors.BLUE = ""
        TermColors.BOLD = ""
        TermColors.UNDERLINE = ""
        TermColors.END = ""

    try:
        baseline_report = load_report(args.baseline)
        current_report = load_report(args.current)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return

    thresholds = {
        "significant_improvement": args.significant_improvement,
        "improvement": args.improvement,
        "minor_improvement": args.minor_improvement,
        "significant_regression": args.significant_regression,
        "regression": args.regression,
        "minor_regression": args.minor_regression,
    }

    report = create_comparison_report(
        baseline_report=baseline_report,
        current_report=current_report,
        thresholds=thresholds,
        alpha=args.alpha,
    )

    with open(args.output, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Comparison report generated at {args.output}")

    print_colored_summary(report["summary"], report["comparison_results"])


if __name__ == "__main__":
    main()
