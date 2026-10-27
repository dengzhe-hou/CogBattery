"""Prepare the verified adult CVLT-II Standard source without model calls."""
import json
from pathlib import Path

from .common import write_json

SOURCE = Path(__file__).with_name('cvlt_standard_source.json')


def build_materials():
    """Return fresh literal source material and explicit text-adaptation notes."""
    return json.loads(SOURCE.read_text())


def write_materials(path):
    path = Path(path)
    write_json(path, build_materials())
    return path
