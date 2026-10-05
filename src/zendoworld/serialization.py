"""Read datasets and caches saved before the package reorganization."""
import json
import pickle
from pathlib import Path

_MODULES = json.loads(Path(__file__).with_name("_legacy_modules.json").read_text())


def remap_module(module):
    for old in sorted(_MODULES, key=len, reverse=True):
        if module == old or module.startswith(old + "."):
            return _MODULES[old] + module[len(old):]
    return module


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(remap_module(module), name)


def load_pickle(file, **kwargs):
    return LegacyUnpickler(file, **kwargs).load()


# PyTorch accepts a pickle-module interface when loading legacy flat datasets.
Unpickler = LegacyUnpickler
load = load_pickle
