from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(scale_numeric=True, with_derived=True):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    num = Pipeline(num_steps)

    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    num_cols = config.NUM_COLS_KEPT + (config.DERIVED_COLS if with_derived else [])

    return ColumnTransformer(
        transformers=[
            ("num", num, num_cols),
            ("cat", cat, config.CAT_COLS_KEPT),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
