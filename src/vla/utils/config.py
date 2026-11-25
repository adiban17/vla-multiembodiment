"""Load a run's config: configs/base.yaml + one file per group + command-line overrides.

    load_config(["cond=token", "size=tiny", "seed=1", "train.steps=5000"])

- `cond=X`, `adapt=X`, `baseline=X`, `size=X` pick configs/<group>/X.yaml and merge it over
  base.yaml, in that order (so size=tiny shrinks a baseline too).
- Everything else is a dotted override (OmegaConf "dotlist"), applied last.
- A misspelled key (train.step=5) is an error instead of a silently ignored new setting.
- `robot` must fit the model: the VLA trains on both robots, a baseline on exactly one.
"""

from omegaconf import DictConfig, OmegaConf

from vla.utils.runtime import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs"
GROUPS = ("cond", "adapt", "baseline", "size")
ROBOT_CHOICES = ("both", "pusht", "fetch")


def group_choices(group: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / group).glob("*.yaml"))


def load_config(overrides=()) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_DIR / "base.yaml")
    choices = {group: cfg[group] for group in GROUPS}
    dotlist = []
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key=value")
        if key in GROUPS:
            choices[key] = value
        else:
            dotlist.append(item)

    for group in GROUPS:
        path = CONFIG_DIR / group / f"{choices[group]}.yaml"
        if not path.is_file():
            raise ValueError(f"unknown {group} {choices[group]!r}; choose from {group_choices(group)}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
        cfg[group] = choices[group]

    OmegaConf.set_struct(cfg, True)  # from here on, unknown keys are errors
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    for group in GROUPS:  # a group can only be chosen by name, never changed by a dotted override
        cfg[group] = choices[group]
    check_robot(cfg)
    return cfg


def check_robot(cfg) -> None:
    """The VLA trains on both robots; a baseline is a specialist for one."""
    if cfg.robot not in ROBOT_CHOICES:
        raise ValueError(f"unknown robot {cfg.robot!r}; choose from {ROBOT_CHOICES}")
    if cfg.baseline != "none" and cfg.robot == "both":
        raise ValueError(f"baseline {cfg.baseline} trains on one robot: add robot=pusht or robot=fetch")
    if cfg.baseline == "none" and cfg.robot != "both":
        raise ValueError(f"the VLA policy trains on both robots (robot=both), got robot={cfg.robot}")


def to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)
