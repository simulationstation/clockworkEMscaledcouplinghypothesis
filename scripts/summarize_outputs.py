#!/usr/bin/env python3
"""Generate machine-readable summary from analysis outputs."""

import json
from pathlib import Path

def load_json_safe(path):
    """Load JSON file, return empty dict if not found."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def main():
    outputs_dir = Path("outputs")
    summary = {"datasets": {}}

    # Baseline results
    baseline_dirs = {
        "higgs": outputs_dir / "higgs_atlas",
        "elastic": outputs_dir / "elastic_totem",
        "diffractive": outputs_dir / "diffractive_dis",
    }

    for name, dir_path in baseline_dirs.items():
        results_file = dir_path / "results.json"
        result = load_json_safe(results_file)

        dataset_summary = {
            "baseline": {
                "lambda_obs": result.get("lambda_stat"),
                "p_boot": result.get("p_value"),
                "chi2_rank1": result.get("chi2_rank1"),
                "chi2_rank2": result.get("chi2_rank2"),
                "ndof_rank1": result.get("ndof_rank1"),
                "ndof_rank2": result.get("ndof_rank2"),
                "n_bootstrap": result.get("n_bootstrap"),
                "significant": result.get("p_value", 1.0) < 0.05,
                "cross_checks": result.get("cross_checks", []),
            },
            "np": None,  # To be filled if NP results exist
        }

        # Check for NP results
        np_results_file = dir_path.parent / name / "np" / "np_results.json"
        if np_results_file.exists():
            np_result = load_json_safe(np_results_file)
            dataset_summary["np"] = {
                "lambda": np_result.get("lambda_stat"),
                "p_local": np_result.get("p_local"),
                "p_global": np_result.get("p_global"),
                "verdict": np_result.get("np_verdict"),
                "gini": np_result.get("localization", {}).get("gini") if np_result.get("localization") else None,
                "stability": np_result.get("stability_grade"),
            }

        summary["datasets"][name] = dataset_summary

    # Write summary
    with open(outputs_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {outputs_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
