from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization import generate_visual_reports_from_files


def run_global(repo_root: Path) -> None:
    runs_dir = repo_root / "results" / "runs"
    files = generate_visual_reports_from_files(
        confusion_path=runs_dir / "local_confusion_matrices.json",
        quantitative_summary_path=runs_dir / "local_quantitative_summary.json",
        output_dir=runs_dir / "plots",
        title_prefix="All Strategies",
    )
    print("Global plots:")
    for file_path in files:
        print(f"- {file_path}")


def run_per_strategy_model(repo_root: Path) -> None:
    base = repo_root / "results" / "runs" / "by_strategy"
    if not base.exists():
        print("No by_strategy folder found.")
        return

    print("Per strategy/model plots:")
    for strategy_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        for model_dir in sorted(path for path in strategy_dir.iterdir() if path.is_dir()):
            confusion_path = model_dir / "confusion_matrices.json"
            quantitative_path = model_dir / "quantitative_summary.json"
            if not confusion_path.exists() or not quantitative_path.exists():
                continue

            files = generate_visual_reports_from_files(
                confusion_path=confusion_path,
                quantitative_summary_path=quantitative_path,
                output_dir=model_dir / "plots",
                title_prefix=f"{strategy_dir.name} | {model_dir.name}",
            )
            if files:
                print(f"- {strategy_dir.name}/{model_dir.name}")
                for file_path in files:
                    print(f"  - {file_path}")


def main() -> None:
    run_global(REPO_ROOT)
    run_per_strategy_model(REPO_ROOT)


if __name__ == "__main__":
    main()
