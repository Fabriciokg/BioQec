"""Executa um piloto pequeno de mudança abrupta e salva resultados em CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from experiments.windowed_protocol import run_windowed_benchmark
from noise.nonstationary import default_regimes, make_abrupt_change


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("results/pilot.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regimes = default_regimes()
    trajectory = make_abrupt_change(
        regimes["low"],
        regimes["high"],
        T=args.cycles,
        seed=args.seed,
    )
    rows = run_windowed_benchmark(
        trajectory,
        calibration_regime=regimes["low"],
        distance=args.distance,
        window_size=args.window_size,
        shots_per_window=args.shots,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    print(f"Resultados salvos em {args.output}")


if __name__ == "__main__":
    main()
