"""Run output discipline: everything a figure or table needs lives on disk.

Every experimental cell writes `outputs/<run_id>/` containing:
  - config.json        every parameter + git SHA + NEExT version + timings
  - metrics.csv        per-(representation, split) rows
  - bag_predictions.csv  per-bag: size, labels, mean test score per representation
  - bag_table.csv      the raw bag table (sizes + labels) for size/saturation plots

`aggregate()` folds every run's metrics + config into one session-level
results.csv. Figures must regenerate from these CSVs alone — nothing that a
plot needs may exist only in stdout.
"""

import json
import subprocess
from pathlib import Path

import pandas as pd


def git_sha(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def neext_version() -> str:
    try:
        import NEExT

        return getattr(NEExT, "__version__", "unknown")
    except Exception:
        return "unknown"


def write_run(
    outputs_dir: Path,
    run_id: str,
    config: dict,
    metrics_rows: list,
    bag_predictions: pd.DataFrame,
    bag_table: pd.DataFrame,
) -> Path:
    run_dir = Path(outputs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    bag_predictions.to_csv(run_dir / "bag_predictions.csv", index=False)
    bag_table.to_csv(run_dir / "bag_table.csv", index=False)
    return run_dir


def run_complete(outputs_dir: Path, run_id: str) -> bool:
    """A run is resumable-skippable when its metrics.csv exists."""
    return (Path(outputs_dir) / run_id / "metrics.csv").exists()


def aggregate(outputs_dir: Path, config_keys: list) -> pd.DataFrame:
    """Fold all runs into one frame: config columns + per-split metric rows."""
    frames = []
    for run_dir in sorted(Path(outputs_dir).iterdir()):
        cfg_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"
        if not (cfg_path.exists() and metrics_path.exists()):
            continue
        config = json.loads(cfg_path.read_text())
        metrics = pd.read_csv(metrics_path)
        for key in config_keys:
            metrics[key] = config.get(key)
        metrics["run_id"] = run_dir.name
        frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(Path(outputs_dir) / ".." / "results.csv", index=False)
    return df
