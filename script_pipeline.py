import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


FEATURES = [
    "Amplitude",
    "Mean",
    "Beyond1Std",
    "FluxPercentileRatioMid35",
    "Freq1_harmonics_amplitude_0",
    "Freq2_harmonics_amplitude_0",
    "Skew",
    "PeriodLS",
    "SmallKurtosis",
    "Anderson_Darling_",
    "Stetson_K_",
    "MaxSlope",
]


data = {
    "raw": pd.read_csv("outputs/nb3/features__TESS_lightcurves_raw.csv"),
    "median_detrended": pd.read_csv(
        "outputs/nb3/features__TESS_lightcurves_median_after_detrended.csv"
    ),
    "outlier_cleaned": pd.read_csv(
        "outputs/nb3/features__TESS_lightcurves_outliercleaned.csv"
    ),
}


def replace_nan_values(X: np.ndarray) -> np.ndarray:
    """
    Replace nan values with 0.
    """
    return np.nan_to_num(X)


transformer_nan = FunctionTransformer(replace_nan_values)


# Read data
df = data["outlier_cleaned"]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True, how="all")

all_features = list(df.columns.values)
all_features.remove("type")
all_features.remove("object_id")
all_features.remove("AndersonDarling")
all_features.remove("StetsonK")

X = df[all_features].values
y = df["type"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  #9:1

# Define the procesing pipeline
pipe = make_pipeline(
    StandardScaler(), transformer_nan, PCA(n_components=10), KMeans(n_clusters=10)
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Print confusion matrix for test
df = pd.DataFrame({"type": y_test, "pred": y_pred, "ones": np.ones(y_test.shape)})
print(
    df.pivot_table(values="ones", index="type", columns="pred", aggfunc="sum").fillna(0)
)

# print()
# print()
# print()

# pipe.fit(X, y)
# y_pred = pipe.predict(X)
# df = pd.DataFrame({"type": y, "pred": y_pred, "ones": np.ones(y_pred.shape)})
# print(
#     df.pivot_table(values="ones", index="type", columns="pred", aggfunc="sum").fillna(0)
# )
