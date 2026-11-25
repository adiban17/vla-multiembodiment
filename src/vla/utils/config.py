"""Load a run's config: configs/base.yaml + one file per group + command-line overrides.

    load_config(["cond=token", "size=tiny", "seed=1", "train.steps=5000"])

- `cond=X`, `adapt=X`, `size=X` pick configs/<group>/X.yaml and merge it over base.yaml.
- Everything else is a dotted override (OmegaConf "dotlist"), applied last.
- A misspelled key (train.step=5) is an error instead of a silently ignored new setting.
"""

from omegaconf import DictConfig, OmegaConf

from vla.utils.runtime import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs"
GROUPS = ("cond", "adapt", "size")


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
    return cfg


def to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)
