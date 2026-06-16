
import importlib

PARADIGMS = {
    "autoregressive": "training.paradigms.autoregressive",
    "diffusion": "training.paradigms.diffusion",
    "lad": "training.paradigms.lad",
}


def get_paradigm(name: str):
    try:
        module_path = PARADIGMS[name]
    except KeyError:
        raise ValueError(f"Unknown paradigm {name!r}. Available: {sorted(PARADIGMS)}")
    return importlib.import_module(module_path)
