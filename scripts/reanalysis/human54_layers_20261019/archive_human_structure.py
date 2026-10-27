"""Archive all human structural statistics before the single formal model pass."""
import json
import numpy as np
import pandas as pd
from .analyze import OUT, SEED, structural_population, save_json


def main():
    human=json.loads((OUT/'human_side/references.json').read_text())
    seed=int(np.random.SeedSequence(SEED).generate_state(2)[0])
    for matrix in human['matrices']:
        rules=('valid_only','invalid_wrong') if matrix['rule']=='both' else (matrix['rule'],)
        for rule in rules:
            folder=OUT/'human_side/structures'/f"{matrix['name']}_{rule}"
            if (folder/'summary.json').exists():
                continue  # Already archived; never rerun complete statistics just to reformat.
            frame=pd.DataFrame(matrix['values'],columns=matrix['columns'],dtype=float)
            result,_=structural_population(frame,matrix['columns'],'human',False,folder,seed)
            save_json(folder/'summary.json',result)


if __name__=='__main__':main()
