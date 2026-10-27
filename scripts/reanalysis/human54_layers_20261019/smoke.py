"""Exercise complete analysis interfaces with synthetic model tables only."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .analyze import (OUT, endpoint_comparisons, behavior_directions,
                      profile_comparisons, run_structure, save_json)


def main():
    start=time.monotonic()
    source=OUT/'synthetic/model_fixture'
    out=OUT/'synthetic/integration'
    (out/'layer1').mkdir(parents=True,exist_ok=True)
    (out/'layer2').mkdir(exist_ok=True)
    if not (out/'human_side').exists():
        (out/'human_side').symlink_to(OUT/'human_side',target_is_directory=True)
    human=json.loads((OUT/'human_side/references.json').read_text())
    endpoints=pd.read_csv(source/'layer1/endpoints.csv')
    profiles=pd.read_csv(source/'layer1/profiles.csv',dtype={'cell':str})
    blocks=pd.read_csv(source/'layer1/nb_block_counts.csv')
    fixture_models=set(endpoints.model)
    if len(fixture_models)!=1 or not profiles.model.isin(fixture_models).all() or not blocks.model.isin(fixture_models).all():
        raise ValueError('Synthetic fixture model identifiers must agree across exported tables')
    panels={'synthetic':fixture_models}
    families={mapping:{model:'fixture' for model in fixture_models} for mapping in ('merged','raw')}
    n_pairs,n_panels=endpoint_comparisons(endpoints,human['endpoints'],families,panels,out/'layer1')
    panel_check=pd.read_csv(out/'layer1/panel_differences.csv')
    if not n_pairs or not n_panels or not panel_check.models.gt(0).any():
        raise ValueError('Synthetic smoke must exercise nonempty endpoint and panel comparisons')
    behavior_directions(endpoints,families,panels,out/'layer1')
    n_profiles=profile_comparisons(profiles,human['profiles'],blocks,out/'layer1')
    structure=pd.read_csv(source/'layer2/structure_scores.csv')
    # Independent continuous synthetic columns exercise defined covariance;
    # constant-task behavior has a separate deterministic test.
    rng=np.random.default_rng(20261018)
    expanded=[]
    names={f'synthetic_{i}' for i in range(14)}
    for i,model in enumerate(sorted(names)):
        frame=structure.copy();frame['model']=model
        for task in sorted(frame.task.unique()):
            frame.loc[frame.task.eq(task),'value']=rng.uniform(.1,.9)
        expanded.append(frame)
        families['merged'][model]=f'family_{i//2}'
        families['raw'][model]=f'raw_{i//2}'
    n_structures=run_structure(pd.concat(expanded,ignore_index=True),human['matrices'],families,
                               {'synthetic14':names},out/'layer2')
    manifest=dict(synthetic=True,formal_records_read=0,model_calls=0,
                  source='perfect-answer engineering fixtures plus continuous synthetic structure columns',
                  endpoint_comparisons=n_pairs,panel_comparisons=n_panels,profiles=n_profiles,
                  structure_runs=n_structures,wall_seconds=time.monotonic()-start)
    save_json(out/'SMOKE_COMPLETE.json',manifest)
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
