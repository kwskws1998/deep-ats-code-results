"""Reference RWRL combined-challenge baselines from the official benchmark paper."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RWRLBaselineRow:
    algorithm: str
    difficulty: str
    task: str
    mean_return: float
    std_return: float


RWRL_REFERENCE_BASELINES: tuple[RWRLBaselineRow, ...] = (
    RWRLBaselineRow("D4PG", "none", "cartpole:swingup", 859.63, 5.68),
    RWRLBaselineRow("D4PG", "none", "walker:walk", 983.24, 9.70),
    RWRLBaselineRow("D4PG", "none", "quadruped:walk", 998.71, 0.32),
    RWRLBaselineRow("D4PG", "none", "humanoid:walk", 934.00, 27.34),
    RWRLBaselineRow("D4PG", "easy", "cartpole:swingup", 482.32, 84.56),
    RWRLBaselineRow("D4PG", "easy", "walker:walk", 514.44, 70.21),
    RWRLBaselineRow("D4PG", "easy", "quadruped:walk", 787.73, 86.95),
    RWRLBaselineRow("D4PG", "easy", "humanoid:walk", 102.92, 22.47),
    RWRLBaselineRow("D4PG", "medium", "cartpole:swingup", 175.47, 51.57),
    RWRLBaselineRow("D4PG", "medium", "walker:walk", 75.49, 16.94),
    RWRLBaselineRow("D4PG", "medium", "quadruped:walk", 268.01, 135.84),
    RWRLBaselineRow("D4PG", "medium", "humanoid:walk", 1.28, 0.99),
    RWRLBaselineRow("D4PG", "hard", "cartpole:swingup", 108.20, 57.97),
    RWRLBaselineRow("D4PG", "hard", "walker:walk", 59.85, 17.70),
    RWRLBaselineRow("D4PG", "hard", "quadruped:walk", 280.75, 123.21),
    RWRLBaselineRow("D4PG", "hard", "humanoid:walk", 1.27, 0.79),
    RWRLBaselineRow("DMPO", "none", "cartpole:swingup", 859.06, 18.07),
    RWRLBaselineRow("DMPO", "none", "walker:walk", 977.71, 14.50),
    RWRLBaselineRow("DMPO", "none", "quadruped:walk", 998.35, 3.71),
    RWRLBaselineRow("DMPO", "none", "humanoid:walk", 788.49, 33.88),
    RWRLBaselineRow("DMPO", "easy", "cartpole:swingup", 464.05, 89.11),
    RWRLBaselineRow("DMPO", "easy", "walker:walk", 474.44, 74.55),
    RWRLBaselineRow("DMPO", "easy", "quadruped:walk", 567.53, 210.54),
    RWRLBaselineRow("DMPO", "easy", "humanoid:walk", 1.33, 1.14),
    RWRLBaselineRow("DMPO", "medium", "cartpole:swingup", 155.63, 35.81),
    RWRLBaselineRow("DMPO", "medium", "walker:walk", 64.63, 17.03),
    RWRLBaselineRow("DMPO", "medium", "quadruped:walk", 180.30, 92.41),
    RWRLBaselineRow("DMPO", "medium", "humanoid:walk", 1.27, 0.90),
    RWRLBaselineRow("DMPO", "hard", "cartpole:swingup", 138.06, 55.82),
    RWRLBaselineRow("DMPO", "hard", "walker:walk", 63.05, 18.71),
    RWRLBaselineRow("DMPO", "hard", "quadruped:walk", 144.69, 92.85),
    RWRLBaselineRow("DMPO", "hard", "humanoid:walk", 1.40, 0.82),
)


def best_reference_rows() -> tuple[RWRLBaselineRow, ...]:
    best: dict[tuple[str, str], RWRLBaselineRow] = {}
    for row in RWRL_REFERENCE_BASELINES:
        key = (row.difficulty, row.task)
        if key not in best or row.mean_return > best[key].mean_return:
            best[key] = row
    return tuple(best[key] for key in sorted(best))


def write_rwrl_reference_csv(
    path: Path, rows: tuple[RWRLBaselineRow, ...] = RWRL_REFERENCE_BASELINES
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("algorithm", "difficulty", "task", "mean_return", "std_return"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
