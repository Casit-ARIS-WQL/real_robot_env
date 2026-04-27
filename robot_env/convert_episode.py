"""
convert_episode.py
------------------
Converts episode.json from a columnar (key-per-field) layout to a row-based
layout with the following structure:

{
  "episode_id": "ep_000000",
  "task": "<task>",
  "milestones": [],
  "frames": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "reward": 0.0,
      "is_terminal": false,
      "observation": {
        "state": [...],
        "state_joint": [...],
        "images": {
          "right":  {"path": "...", "timestamp": 0.0},
          "top":    {"path": "...", "timestamp": 0.0},
          "left":   {"path": "...", "timestamp": 0.0},
          "top_2":  {"path": "...", "timestamp": 0.0}
        }
      },
      "action": {
        "cartesian": [...],
        "joint": [...]
      }
    },
    ...
  ]
}

Usage
-----
    python convert_episode.py [--input episode.json] [--output episode_formatted.json]
                              [--task pick_and_place] [--reward-terminal 1.0]
"""

import argparse
import json
import sys


def build_episode_id(episode_index_list: list) -> str:
    """Derive a string episode id from the first element of episode_index."""
    idx = episode_index_list[0] if episode_index_list else 0
    return f"ep_{int(idx):06d}"


def convert(data: dict, task: str, reward_terminal: float) -> dict:
    """Transform columnar episode data into a row-based episode dict."""

    n_frames = len(data["frame_index"])

    episode_id = build_episode_id(data["episode_index"])

    frames = []
    for i in range(n_frames):
        is_terminal = bool(data["next.done"][i])

        observation = {
            "state": data["observation.state"][i],
            "state_joint": data["observation.state_joint"][i],
            "images": {
                "right": data["observation.images.right"][i],
                "top": data["observation.images.top"][i],
                "left": data["observation.images.left"][i],
                "top_2": data["observation.images.top_2"][i],
            },
        }

        action = {
            "cartesian": data["action"][i],
            "joint": data["action_joint"][i],
        }

        frame = {
            "frame_index": data["frame_index"][i],
            "timestamp": data["timestamp"][i],
            "reward": reward_terminal if is_terminal else 0.0,
            "is_terminal": is_terminal,
            "observation": observation,
            "action": action,
        }
        frames.append(frame)

    return {
        "episode_id": episode_id,
        "task": task,
        "milestones": [],
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reformat episode.json into a row-based episode structure."
    )
    parser.add_argument(
        "--input",
        default="episode.json",
        help="Path to the source episode JSON file (default: episode.json)",
    )
    parser.add_argument(
        "--output",
        default="episode_formatted.json",
        help="Path to write the reformatted JSON file (default: episode_formatted.json)",
    )
    parser.add_argument(
        "--task",
        default="pick_and_place",
        help='Task label to embed in the output (default: "pick_and_place")',
    )
    parser.add_argument(
        "--reward-terminal",
        type=float,
        default=1.0,
        help="Reward value assigned to the terminal frame (default: 1.0)",
    )
    args = parser.parse_args()

    print(f"Reading {args.input} …")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {
        "episode_index",
        "frame_index",
        "timestamp",
        "next.done",
        "observation.state",
        "observation.state_joint",
        "observation.images.right",
        "observation.images.top",
        "observation.images.left",
        "observation.images.top_2",
        "action",
        "action_joint",
    }
    missing = required_keys - set(data.keys())
    if missing:
        print(f"ERROR: Missing keys in input file: {missing}", file=sys.stderr)
        sys.exit(1)

    episode = convert(data, task=args.task, reward_terminal=args.reward_terminal)

    print(f"Writing {args.output} … ({len(episode['frames'])} frames)")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
