"""Instruction sentences: 10 paraphrases per task, the first 8 for training, the last 2 held out.

With one task per robot, language works much like a task ID. Training on 8 paraphrases and
testing on 2 unseen ones shows the policy is not tied to one exact sentence.

All 20 sentences have a fixed global index (pusht 0-9, fetch 10-19); MiniLM embeds them once
and the feature cache stores them in that order (language.npy).
"""

ROBOTS = ("pusht", "fetch")
N_TRAIN = 8

PARAPHRASES = {
    "pusht": [
        "push the T onto the target",
        "push the T-shaped block onto the target",
        "slide the T into the outline",
        "move the T block so it covers the T-shaped target",
        "push the gray T until it lines up with the green outline",
        "align the T-shaped piece with the target shape",
        "nudge the T block onto its matching outline",
        "get the T to sit exactly on the target",
        # held out
        "shove the T-shaped object into the marked T area",
        "place the T over the outline by pushing it",
    ],
    "fetch": [
        "push the cube to the red target",
        "slide the block onto the red dot",
        "move the cube to the target with the gripper",
        "push the box until it reaches the red marker",
        "use the arm to push the cube onto the goal",
        "nudge the block over to the red spot",
        "get the cube onto the red target on the table",
        "push the small block to the goal position",
        # held out
        "shove the cube toward the red mark until it is on it",
        "bring the block to the red goal by pushing it",
    ],
}


def paraphrases(robot: str, split: str = "train") -> list[str]:
    """split: "train" (8), "heldout" (2) or "all" (10)."""
    sentences = PARAPHRASES[robot]
    if split == "train":
        return sentences[:N_TRAIN]
    if split == "heldout":
        return sentences[N_TRAIN:]
    if split == "all":
        return list(sentences)
    raise ValueError(f"unknown split {split!r}; use 'train', 'heldout' or 'all'")


def all_instructions() -> list[str]:
    """All 20 sentences in global-index order (the row order of language.npy)."""
    return [s for robot in ROBOTS for s in PARAPHRASES[robot]]


def instruction_id(robot: str, k: int) -> int:
    """Global index of paraphrase k (0-9) of this robot's task."""
    if not 0 <= k < len(PARAPHRASES[robot]):
        raise IndexError(k)
    return ROBOTS.index(robot) * len(PARAPHRASES["pusht"]) + k
