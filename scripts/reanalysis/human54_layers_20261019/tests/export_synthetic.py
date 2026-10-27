"""Emit clearly synthetic integration inputs without reading any model records."""
from pathlib import Path
from .test_model_scoring import BUNDLES,fixtures
from ..model_scoring import analyze_cell
from ..model_run import write_csv,write_json,structure_rows


def main():
    root=Path(__file__).resolve().parents[4]
    out=root/'results/reanalysis/human54_layers_20261019/synthetic/model_fixture'
    tables={key:[] for key in ('endpoints','profiles','blocks','diagnostics')}
    for task,bundle in BUNDLES.items():
        result=analyze_cell('SYNTHETIC_FIXTURE',task,bundle,fixtures(task))
        for key in tables: tables[key].extend(dict(r,synthetic=True,family_raw='fixture',family_merged='fixture',native_thinking=False,panel54=True,panel19=True,panel51=True) for r in result[key])
    write_csv(out/'layer1/endpoints.csv',tables['endpoints'])
    write_csv(out/'layer1/profiles.csv',tables['profiles'])
    write_csv(out/'layer1/nb_block_counts.csv',tables['blocks'])
    write_csv(out/'layer1/diagnostics.csv',tables['diagnostics'])
    write_csv(out/'layer2/structure_scores.csv',[dict(r,synthetic=True) for r in structure_rows(tables['endpoints'])])
    write_csv(out/'coverage/models.csv',[dict(model='SYNTHETIC_FIXTURE',family_raw='fixture',family_merged='fixture',native_thinking=False,panel54=True,panel19=True,panel51=True,synthetic=True)])
    write_json(out/'SYNTHETIC_MANIFEST.json',dict(synthetic=True,not_formal=True,source='prepared material gold answers, no model records',tasks=list(BUNDLES),rows={k:len(v) for k,v in tables.items()}))
    schema=dict(endpoints=sorted({(r['task'],r['endpoint']) for r in tables['endpoints']}),profiles={})
    for r in tables['profiles']: schema['profiles'].setdefault(r['task']+'|'+r['profile'],set()).add(r['cell'])
    schema['profiles']={k:sorted(v) for k,v in schema['profiles'].items()}
    write_json(root/'scripts/reanalysis/human54_layers_20261019/model_output_schema.json',schema)
    print(out)

if __name__=='__main__':main()
