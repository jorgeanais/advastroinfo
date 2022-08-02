import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer

import hdbscan



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
# X = df[FEATURES].values
y = df["type"].values


# Normalize data
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)  # Problems with NAN values!!!
# Detect nan values np.isinf
bad_indices = np.where(np.isnan(X_scaled))

# Replace nan values with mean # np.nanmean(X_scaled, axis=0) # np.nanmean(X_scaled, axis=0)
X_scaled[bad_indices] = 0.0

# PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)


# Following instructions from Notebook 7 to perform cross-validation ###########

folds = 10
k_fold = StratifiedKFold(folds, shuffle=True, random_state=2)   # Now using Stratified KFold

predicted_targets = np.array([])
actual_targets = np.array([])

input_folds = np.empty(folds, dtype=object)
result_folds = np.empty(folds, dtype=object)

fold = 0
for train_ix, test_ix in k_fold.split(X, y):
    train_x, train_y, test_x, test_y = X[train_ix], y[train_ix], X[test_ix], y[test_ix]


    print("Fold: ", fold)

    # Feature Scaling 
    scaler = StandardScaler()  
    train_x = np.nan_to_num(scaler.fit_transform(train_x))
    test_x = np.nan_to_num(scaler.transform(test_x))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True, cluster_selection_method="eom")
    clusterer.fit(test_x)

    # Predict the labels of the test set samples
    # predicted_labels = clusterer.predict(test_x)
    predicted_labels, _ = hdbscan.approximate_predict(clusterer, test_x)

    predicted_targets = np.append(predicted_targets, predicted_labels)
    actual_targets = np.append(actual_targets, test_y)

    # instead, save predicted, actual in 2d matrixpredite
    input_folds[fold] = test_y
    result_folds[fold] = predicted_labels

    fold = fold + 1


# Print confusion matrix for test
actual_labels = np.unique(actual_targets)  # In case a class is not present in the actual targets

# Print confusion matrix for test
raw_results_df = pd.DataFrame({"True label": actual_targets, "Predicted label": predicted_targets, "ones": np.ones(actual_targets.shape)})
cm_df = raw_results_df.pivot_table(values="ones", index="True label", columns="Predicted label", aggfunc="sum").fillna(0)
cm_df.to_csv("outputs/semisupervised_results/cm_semisupervised_HDBSCAN.csv")
print(cm_df)

cm_df_p = cm_df.apply(lambda x: x/x.sum(), axis=1)
print(cm_df_p)

plt.figure(figsize=(8, 8))
sns.heatmap(cm_df_p, annot=True, fmt=".2f", cbar=False, cmap="Blues")
plt.tight_layout()
plt.savefig("outputs/semisupervised_results/CM_StratifiedKFold_HDBSCAN.pdf")
plt.show()

