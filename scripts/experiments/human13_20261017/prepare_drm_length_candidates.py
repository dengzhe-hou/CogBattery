"""Build both sourced CITY alternatives for offline length inspection only.

Neither version is identified as Ball's original word pool. Every output stays
blocked; the formal prepared materials and SPEC are never written here.
"""
import json
from pathlib import Path
import sys

from . import memory
from .common import ROOT, RAW, write_json

OUTPUT = ROOT / "results/human13_20261017/compact_candidate_54/length_only_word_candidates"
WORKBOOK = RAW / "ball2022_memory/stimuli/Alakbarova et al. (2021) - Exp 2.xls.xlsx"
MANIFEST = ROOT / "results/human13_20261017/prepared_54_public_versions/manifest.json"


def prepare(output=OUTPUT):
    import openpyxl
    # Reuse the installed project SPSS reader; no new dependency/network access.
    sys.path.insert(0, str(ROOT / ".deps/public_human_structure_20261015"))
    output = Path(output)
    manifest = json.loads(MANIFEST.read_text())
    workbook = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    results = {}
    for variant in ("conv", "div"):
        rows = [{"row": index, "sentence": row[1], "word": row[1].split()[-1]}
                for index, row in enumerate(workbook[variant].iter_rows(values_only=True), 1)
                if row[0] == "study" and row[2] == "CITY"]
        assert len(rows) == 8 and len({row["word"] for row in rows}) == 8
        source = {"workbook": str(WORKBOOK), "sheet": variant, "city_rows": rows,
                  "source_url": "https://osf.io/download/euzgd/",
                  "interpretation": "Upstream sentence-final word set; not confirmed as Ball's CITY word-version choice."}
        candidate_dir = output / variant
        candidate_path = candidate_dir / "city_words_length_only.json"
        write_json(candidate_path, {"source_reference": f"{WORKBOOK}, sheet {variant}, CITY study rows 10–17; length candidate only",
                                    "words": [row["word"] for row in rows],
                                    "purpose": "offline_length_only_not_experiment_material_selection",
                                    "source": source})
        config = dict(manifest["config"], materials_dir=str(candidate_dir), drm_city_words=str(candidate_path))
        bundle = memory.build_drm(config)
        bundle["episodes"] = [e for e in bundle["episodes"] if e["condition"].startswith("words_")]
        assert len(bundle["episodes"]) == 2 and sum(len(e["steps"]) for e in bundle["episodes"]) == 448
        bundle["status"] = "blocked"
        bundle["blockers"] = ["LENGTH CHECK ONLY: Ball CITY version remains unverified. This sourced upstream candidate is not an approved experiment material choice."]
        bundle["provenance"].update({"purpose": "offline_length_only_not_experiment_material_selection",
                                      "candidate": variant, "source_candidate": source,
                                      "scope": "Two word-version warning conditions only; both alternatives retained separately.",
                                      "planned_word_steps_per_model": 448,
                                      "models": manifest["spec"]["models"],
                                      "inference_authorized": False})
        path = candidate_dir / "drm_words_length_only.json"
        write_json(path, bundle)
        results[variant] = str(path)
    workbook.close()
    return results


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
