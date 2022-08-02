# advastroinfo

Python code and jupyter notebooks used as part of the course Lessons in Advanced Astroinformatics.

Folders `_data` and `outputs` are not included in this repository due to its large size. Some sample results are available in `selected_results`.

## Contents

### Notebooks
1. `notebook_1a.ipynb`: Getting Started
2. `notebook_1b.ipynb`: Plotting TESS light curves
3. `notebook_2.ipynb`: Light curve features (I): Fourier transformation, Lomb-Scargle Periodogram, and the feature extraction from the `feets` package
4. `notebook_3.ipynb`: Light curve features (II)
5. `notebook_4.ipynb`: Recap
6. `notebook_5.ipynb`: Machine Learning: Intro to Scikit-Learn (example: iris dataset, Scikit-Lear estimator object, Supervised learning: k-nearest neighbors, model validation, confusion matrix, )
7. `notebook_6.ipynb`: Machine Learning: Intro to Scikit-Learn (Binary classification, ROC, Completeness vs Efficiency, Multiclass classifiers, feature scaling, Random Forest, k-Fold verification)
8. `notebook_7.ipynb`: Supervised Classification, Data Processing Pipelines. Data processing pipelines.
9. `notebook_8.ipynb`: Optimizing Source Code: Monitoring code execution time, Avoid slow program/structures, compiled code Cython, Reusing data structures (serializing with pickle), memoization, parallelization.  Imbalance datasets: stratified k-fold cross-validation.
10. `notebook_9.ipynb`: Python plotting best practices.



### Scripts
0. The code used to plot the ligthcurves is available at `notebook_1b.ipynb` and results are available at `selected_results/nb_1b/`.
1. `script_extract_features.py`: Extract features from all lc data, using feet + custom versions of AndersonDarling stat and Stetson K-index. For each version of the lc it produces a table with the object name, variability type and features values. Results are available at `selected_results/nb_3/`.
2. `script_corner_plots.py`: Using the tables generated before, it produces corner plots for each variability type individually. Results are available at `selected_results/nb_4/`.
3. `script_semisupervised.py`: Semi-supervised learning. Results are in `selected_results/semisupervised/`.
4. `script_supervised.py`: Application of random forest classification and feature importance. Results are in `selected_results/supervised/`.


### Auxiliary scripts
1. `script_clustering.py` and `script_pipeline.py`: Script used to fine tune K-mean algorithm.
2. `script_nb7.py`: Script used to test RandomForestClassifier using K-fold. Confusion matrix plot is available at  `selected_results/nb_7/`.
3. `script_nb8.py`: Same as `script_nb7.py` but using StratifiedKFold. Confusion matrix plot is available at  `selected_results/nb_8/`.



## Amount of objects per stage and variability type

| Main Group | Type    | Description                                                        | Raw  | Median Detrended | Outlier clean |
|------------|---------|--------------------------------------------------------------------|------|------------------|---------------|
| Eclipsing  | E       | Eclipsing binary systems                                           |    8 |                8 |             8 |
| Eclipsing  | EA      | beta Persei-type (Algol) eclipsing systems                         |  447 |              447 |           447 |
| Eclipsing  | EB      | beta Lyrae-type eclipsing systems                                  |  121 |              121 |           121 |
| Eclipsing  | EC      | Contact binaries                                                   |  124 |              124 |           124 |
| Eclipsing  | ED      | Detached eclipsing binaries                                        |   54 |               54 |            54 |
| Eclipsing  | EW      | W Ursae Majoris-type eclipsing variables                           | 1092 |             1092 |          1092 |
| Pulsating  | CEP     | Cepheids                                                           |   22 |               22 |            22 |
| Pulsating  | DCEP    | Classical Cepheids                                                 |   77 |               77 |            77 |
| Pulsating  | DCEP-FU | Fundamental mode classical Cepheids                                |   10 |               10 |            10 |
| Pulsating  | DCEPS   | delta Cep variables having light amplitudes                        |    6 |                6 |             6 |
| Pulsating  | DSCT    | Variables of the delta Scuti type                                  |   71 |               71 |            71 |
| Pulsating  | HADS    | High Amplitude delta Scuti stars.                                  |   31 |               31 |            31 |
| Pulsating  | L       | Slow irregular variables                                           |  236 |              236 |           236 |
| Pulsating  | RR      | Variables of the RR Lyrae type                                     |    6 |                6 |             6 |
| Pulsating  | RRAB    | RR Lyrae variables with asymmetric light curves                    |  443 |              443 |           443 |
| Pulsating  | RRAB_BL | RR Lyrae stars showing the Blazhko effect                          |   55 |               55 |            55 |
| Pulsating  | RRC     | RR Lyrae variables with nearly symmetric light curves              |  203 |              203 |           203 |
| Pulsating  | RRD     | Double-mode RR Lyrae stars                                         |   23 |               23 |            23 |
| Pulsating  | SR      | Semi-regular variables                                             |  992 |              992 |           992 |
| Rotating   | ACV     | alpha2 Canum Venaticorum variables.                                |   21 |               21 |            21 |
| Rotating   | ROT     | Classical T Tauri stars showing periodic variability due to spots. |  790 |              790 |           790 |
| Rotating   | RS      | RS Canum Venaticorum-type binary systems.                          |   18 |               18 |            18 |

