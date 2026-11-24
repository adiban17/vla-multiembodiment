"""Scripted closed-loop FetchPush expert (it "cheats" by reading the cube position from the sim).

Every step it re-reads the cube and goal, so it corrects itself if the cube slides or turns:

    d = unit vector from cube to goal (on the table plane)
    1. LIFT     hand is low and not lined up -> go straight up first (moving sideways while
                low would sweep the cube off the table)
    2. TRAVEL   fly at a safe height to a point `behind` metres behind the cube (opposite d)
    3. LOWER    once above that point, go down to cube height
    4. PUSH     hand is low and lined up behind the cube -> move along d, re-centring on the
                cube-goal line every step, slowing down as the cube nears the goal
    5. HOLD     cube within `goal_tol` of the goal -> stay still

action = clip(gain * (target - hand), -1, 1) for dx, dy, dz, plus small Gaussian noise so the demos
vary; gripper command fixed at 0 (FetchPush's gripper is locked anyway).
"""

import numpy as np


class FetchPushExpert:
    def __init__(
        self,
        rng=None,
        noise_std=0.1,  # action noise (1.0 = 5 cm), so demos are not all identical
        gain=10.0,  # K in action = K * (target - hand)
        behind=0.06,  # where to put the hand before pushing: 6 cm behind the cube centre
        contact=0.045,  # hand-to-cube-centre distance while pushing
        safe_height=0.06,  # travel this far above the cube
        goal_tol=0.02,  # stop pushing when the cube is this close to the goal
        align_tol=0.02,  # max sideways offset from the push line that still counts as lined up
        push_speed=0.8,  # cap on the push action, so the cube does not fly past the goal
        advance=0.08,  # while pushing, aim this far ahead of the contact point (sets push speed)
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.noise_std = noise_std
        self.gain = gain
        self.behind = behind
        self.contact = contact
        self.safe_height = safe_height
        self.goal_tol = goal_tol
        self.align_tol = align_tol
        self.push_speed = push_speed
        self.advance = advance
        self.phase = None  # last phase name, handy for debugging

    def act(self, raw_obs) -> np.ndarray:
        o = raw_obs["observation"]
        hand, cube, goal = o[0:3], o[3:6], raw_obs["desired_goal"]

        to_goal = goal[:2] - cube[:2]
        dist = float(np.linalg.norm(to_goal))
        if dist < self.goal_tol:
            self.phase = "hold"
            return np.zeros(4, dtype=np.float32)
        d = to_goal / dist

        rel = hand[:2] - cube[:2]
        along = float(rel @ d)  # negative = hand is behind the cube
        lateral = float(np.linalg.norm(rel - along * d))
        low = hand[2] < cube[2] + 0.02
        lined_up = lateral < self.align_tol and -(self.behind + 0.03) < along < -0.02

        behind_xy = cube[:2] - d * self.behind
        above_behind = np.linalg.norm(hand[:2] - behind_xy) <= 0.015
        cap = 1.0
        if low and lined_up:
            self.phase = "push"
            advance = min(dist, self.advance)
            target_xy = cube[:2] - d * self.contact + d * advance
            target = np.array([*target_xy, cube[2]])
            cap = self.push_speed
        elif above_behind:
            self.phase = "lower"
            target = np.array([*behind_xy, cube[2]])
        elif hand[2] < cube[2] + self.safe_height - 0.02:
            # Never move sideways while low: the hand would sweep the cube off the table.
            self.phase = "lift"
            target = np.array([hand[0], hand[1], cube[2] + self.safe_height])
        else:
            self.phase = "travel"
            target = np.array([*behind_xy, cube[2] + self.safe_height])

        move = np.clip(self.gain * (target - hand), -cap, cap)
        move = move + self.rng.normal(0.0, self.noise_std, size=3)
        return np.append(np.clip(move, -1.0, 1.0), 0.0).astype(np.float32)
