"""How many FetchPush evaluation episodes are already solved before the robot moves?

FetchPush counts an episode as a success if, at the final step, the cube is within 5 cm of the
target. The cube and the target are placed at random, so sometimes the cube STARTS within 5 cm.
A policy that leaves the cube alone then "succeeds" for free. This script counts those starts for
the evaluation seeds 100000-100049: no policy, no rendering, CPU only.

    .venv/bin/python scripts/fetch_start_check.py                 # the count
    .venv/bin/python scripts/fetch_start_check.py --with-random   # + which random-policy wins they explain

--with-random reruns the random policy (no rendering needed) on the same seeds and compares its
successes with the already-solved starts. That is a rerun on this machine: MuJoCo physics can
differ slightly between computers, so it may not match the Kaggle eval_sanity numbers exactly.
"""

import argparse

import numpy as np

from vla.envs import fetch
from vla.evaluate import SEED_START

N_EPISODES = 50


def start_distances(seeds) -> list[dict]:
    """For each seed: the cube-to-target distance (m) right after reset, the env's own success
    threshold, and whether the episode would already count as solved."""
    rows = []
    for seed in seeds:
        env = fetch.make_env(int(seed), render=False)
        raw = env.reset()["raw"]
        threshold = float(env.env.unwrapped.distance_threshold)  # 0.05 m in FetchPush
        env.close()
        distance = float(np.linalg.norm(raw["achieved_goal"] - raw["desired_goal"]))
        rows.append({"seed": int(seed), "distance": distance, "threshold": threshold,
                     "solved_at_start": distance < threshold})
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--with-random", action="store_true", help="also rerun the random policy and compare")
    args = parser.parse_args(argv)

    seeds = range(args.seed_start, args.seed_start + args.episodes)
    rows = start_distances(seeds)
    solved = [r for r in rows if r["solved_at_start"]]
    d = np.array([r["distance"] for r in rows])
    print(f"FetchPush, seeds {seeds[0]}-{seeds[-1]} ({len(rows)} episodes), success threshold "
          f"{100 * rows[0]['threshold']:.0f} cm")
    print(f"Already solved at the start (cube within {100 * rows[0]['threshold']:.0f} cm of the target): "
          f"{len(solved)}/{len(rows)} = {len(solved) / len(rows):.0%}")
    for r in solved:
        print(f"  seed {r['seed']}: {100 * r['distance']:.1f} cm")
    print(f"Start distance: min {100 * d.min():.1f} cm, median {100 * np.median(d):.1f} cm, "
          f"max {100 * d.max():.1f} cm")
    edges = [0, 5, 10, 15, 20, 25, 30, 100]
    counts, _ = np.histogram(100 * d, bins=edges)
    for lo, hi, n in zip(edges, edges[1:], counts):
        label = f"{lo:>2d}-{hi:<3d}cm" if hi < 100 else f"{lo:>2d}+ cm  "
        print(f"  {label} {n:3d} {'#' * int(n)}")

    if args.with_random:
        from vla.evaluate import RandomPolicy, evaluate_policy

        print("\nRandom policy on the same seeds (rerun here, no rendering):")
        result = evaluate_policy(RandomPolicy(), ("fetch",), len(rows), seeds[0], 8, log=lambda *_: None)
        wins = {e["seed"] for e in result["fetch"]["episodes"] if e["success"]}
        at_start = {r["seed"] for r in solved}
        print(f"  random successes:                    {len(wins)}/{len(rows)} = {len(wins) / len(rows):.0%}")
        print(f"  ... on starts already solved:        {len(wins & at_start)}  {sorted(wins & at_start)}")
        print(f"  ... on starts NOT already solved:    {len(wins - at_start)}  {sorted(wins - at_start)}")
        lost = sorted(at_start - wins)
        print(f"  already-solved starts it LOST:       {len(lost)}  {lost}"
              + ("  (the random arm knocked the cube away)" if lost else ""))


if __name__ == "__main__":
    main()
