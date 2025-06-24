# app/supervision.py
from snorkel.labeling import labeling_function, PandasLFApplier, LabelModel
import pandas as pd
from app.config import LABEL_CARDINALITY

@labeling_function()
def lf_contains_positive(x):
    return 1 if "good" in x.text.lower() else 0

@labeling_function()
def lf_contains_negative(x):
    return 2 if "bad" in x.text.lower() else 0

def weak_supervision(df: pd.DataFrame) -> pd.DataFrame:
    lfs = [lf_contains_positive, lf_contains_negative]
    applier = PandasLFApplier(lfs)
    L = applier.apply(df)
    label_model = LabelModel(cardinality=LABEL_CARDINALITY, verbose=False)
    label_model.fit(L_train=L, n_epochs=500)
    df['weak_label'] = label_model.predict(L=L)
    return df
