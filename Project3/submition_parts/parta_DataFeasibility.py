import pandas as pd 
import numpy as np
import sys 
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.insert.path(0, str(project_root))

def split_with_purge(df,train_frac=0.6, val_frac = 0.2, purge = 5):
    n = len (df)
    i_tr = int(n* train_frac)
    i_va = int (n* (train_frac + val_frac))
    train = df.iloc [:i_tr -purge]
    val = df.iloc [i_tr : i_va -purge]
    test = df.iloc [i_va:]

    return train,val, test 

def episodes_count(y):
    y = np.asarray(y).astype(int)
    if len(y) == 0:
        return 0 
    edges = np.diff(np.contantenate([[0],y,[0]]))
    return int(np.susm(edges==1))

def feasibility_report(model_data, q_pct = 0.8):
    train, val, test = split_with_purge(model_data)
    q = train ["rv_past"].quantile(q_pct)
    print (f"Threshold q_{q_pct:.2f} (train only): {q:.6f}")
    print(f"Split sizes (after  purge): train = {len(train)}, val = {len(val)}, test = {len(test)}")
    print()
    header = f"{split}"