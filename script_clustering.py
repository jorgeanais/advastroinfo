import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)
# kmeans = KMeans(n_clusters=10, random_state=0).fit(Xtrain)
# y_pred = kmeans.predict(Xtrain)


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


# Search for best K
sum_of_squared_distances = []
k_range = range(1, 20)

for num_clusters in k_range:
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X_pca)
    sum_of_squared_distances.append(kmeans.inertia_)

plt.plot(k_range, sum_of_squared_distances, "bx-")
plt.xlabel("Values of K")
plt.ylabel("Sum of squared distances")
plt.title("Elbow Method")
plt.show()

# From the results I selected k=7, but then I use k=10
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_pca)
y_pred = kmeans.predict(X_pca)

