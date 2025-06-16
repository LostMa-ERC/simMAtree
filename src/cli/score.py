from pathlib import Path

from src.utils.evaluation import evaluate_inference


def score(param_dict: dict, results_dir: str):
    filtered_param_dict = {
        key: value
        for key, value in param_dict.items()
        if value is not None and value != 0
    }

    # Check that some parameters are remaining
    if not filtered_param_dict:
        print("No valid parameters found after filtering.")
        return

    dir = Path(results_dir)
    dir.mkdir(exist_ok=True)

    # Get parameters names
    param_names = list(filtered_param_dict.keys())

    # Run evaluation
    summary, param_summary = evaluate_inference(
        true_params=filtered_param_dict,
        results_dir=dir,
        param_names=param_names,
    )

    # Print summary
    print("\n=== Summary of Evaluation Metrics ===")
    for metric, value in summary.items():
        print(f"{metric}: {value}")

    print("\n=== Parameter-Specific Metrics ===")
    for param, metrics in param_summary.items():
        print(f"\n{param}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
