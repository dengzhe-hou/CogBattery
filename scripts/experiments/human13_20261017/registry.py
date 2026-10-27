"""Explicit dispatch for the three task modules, without plugin discovery."""
from . import memory, tom_mc, wm_cc

MODULES = (wm_cc, memory, tom_mc)


def module_for(task):
    matches = [module for module in MODULES if task in module.TASKS]
    if len(matches) != 1:
        raise ValueError(f'Expected one implementation for {task}, found {len(matches)}')
    return matches[0]
