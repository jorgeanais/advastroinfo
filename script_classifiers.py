"""
This script compute Random Forest classifier using also StratifiedKFold cross validation.
The result is a confusion matrix.
Results are very similar to the ones obtained in the previous script (script_nb7.py).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from sklearn import svm

SELECTED_FEATURES = [
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

# Read data ###################################################################
df = data["outlier_cleaned"]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True, how="all")

all_features = list(df.columns.values)
all_features.remove("type")
all_features.remove("object_id")
all_features.remove("AndersonDarling")
all_features.remove("StetsonK")

X = df[all_features].values ## all_features
y = df["type"].values

# Following Instructions from Notebook 7 ########################################

folds = 10
k_fold = StratifiedKFold(folds, shuffle=True, random_state=1)   # Now using Stratified KFold

predicted_targets = np.array([])
actual_targets = np.array([])

input_folds = np.empty(folds, dtype=object)
result_folds = np.empty(folds, dtype=object)

fold = 0
for train_ix, test_ix in k_fold.split(X, y):
    train_x, train_y, test_x, test_y = X[train_ix], y[train_ix], X[test_ix], y[test_ix]

    # print('X size: ', X.size)
    # print('test_x size: ', test_x.size)

    # Fit the classifier
    # classifier = svm.SVC().fit(train_x, train_y)

    print("Fold: ", fold)
    # print(np.unique(test_y, return_counts=True))

    # Feature Scaling
    scaler = StandardScaler()
    train_x = np.nan_to_num(scaler.fit_transform(train_x))
    test_x = np.nan_to_num(scaler.transform(test_x))

    classifier = RandomForestClassifier(
        n_estimators=10, criterion="entropy", random_state=42
    )
    # classifier = svm.SVC()

    classifier.fit(train_x, train_y)

    # Predict the labels of the test set samples
    predicted_labels = classifier.predict(test_x)

    predicted_targets = np.append(predicted_targets, predicted_labels)
    actual_targets = np.append(actual_targets, test_y)

    # instead, save predicted, actual in 2d matrixpredite
    input_folds[fold] = test_y
    result_folds[fold] = predicted_labels

    fold = fold + 1


# print(np.unique(actual_targets, return_counts=True))

# Print confusion matrix for test
#df = pd.DataFrame({"actual": actual_targets, "pred": predicted_targets, "ones": np.ones(actual_targets.shape)})
#print(
#    df.pivot_table(values="ones", index="actual", columns="pred", aggfunc="sum").fillna(0)
#)

actual_labels = np.unique(actual_targets)  # In case a class is not present in the actual targets
cnf_matrix = confusion_matrix(actual_targets, predicted_targets, labels=actual_labels)


# print(cnf_matrix)


# here we calculate the scores for all three classes to report the classifier performance

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)  # Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
# Specificity or true negative rate
TNR = TN / (TN + FP)
# Precision or positive predictive value
PPV = TP / (TP + FP)
# Negative predictive value
NPV = TN / (TN + FN)
# Fall out or false positive rate
FPR = FP / (FP + TN)
# False negative rate
FNR = FN / (TP + FN)
# False discovery rate
FDR = FP / (TP + FP)
# Overall accuracy for each class
ACC = (TP + TN) / (TP + FP + FN + TN)

print("TPR -------------------")
print("\n".join(f"  {x:<8} {y:.2f}" for x, y in zip(actual_labels, TPR)))

print("FPR -------------------")
print("\n".join(f"  {x:<8} {y:.2f}" for x, y in zip(actual_labels, FPR)))

print("ACC -------------------")
print("\n".join(f"  {x:<8} {y:.2f}" for x, y in zip(actual_labels, ACC)))

# Save results
metrics_df = pd.DataFrame(
    {
        "Specificity": TNR,
        "Precision": PPV,
        "Negative predictive value": NPV,
        "False positive rate": FPR,
        "False discovery rate" : FDR,
        "Accuracy": ACC

    }, index=actual_labels
)

print(metrics_df)

# reuse plotting code from notebook 5


import seaborn as sns


def to_density(cf):
    """
    This function will take in a confusion matrix cf and return the relative 'density' of every element in each row.
    ---------
    cf: Confusion matrix to be passed in
    """
    density = []
    n, k = cf.shape
    for i in range(n):
        density_row = []
        for j in range(k):
            total_stars = sum(cf[i])
            density_row.append(cf[i][j] / total_stars)
        density.append(density_row)
    return np.array(density)


def make_confusion_matrix(
    cf_,
    xlabel,
    ylabel,
    group_names=None,
    categories_x="auto",
    categories_y="auto",
    count=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="Blues",
    title=None,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf_:            Confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    cf = to_density(cf_) 

    # Generate the labels for the matrix elements:
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.2f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_labels, group_counts)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # Set figure paramaters:
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # Make the heatmap:
    plt.figure(figsize=figsize)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        yticklabels=categories_y,
        xticklabels=categories_x,
    )

    if xyplotlabels:
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

    if title:
        plt.title(title)


# Make a confusion matrix plot for 10-fold cross-validation:

make_confusion_matrix(
    cnf_matrix.T,
    xlabel="True label",
    ylabel="Predicted label",
    categories_x=actual_labels,
    categories_y=actual_labels,
    count=True,
    figsize=(6, 6),
)  # categories_x=all_features,categories_y=all_features
plt.show()

