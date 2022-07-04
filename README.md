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
6. `notebook_5.ipynb`: Machine Learning: Intro to Scikit-Learn


### Scripts
1. `script_extract_features.py`: Extract features from all lc data, using feet + custom versions of AndersonDarling stat and Stetson K-index. For each version of the lc it produces a table with the object name, variability type and features values. Results are available at `selected_results/nb_3/`.
2. `script_corner_plots.py`: Using the tables generated before, it produces corner plots for each variability type individually.

## Amount of objects per stage and variability type

| Type    | Raw  | Median Detrended | Outlier clean |
|---------|------|------------------|---------------|
| ACV     | 21   | 21               | 21            |
| CEP     | 22   | 22               | 22            |
| DCEP    | 77   | 77               | 77            |
| DCEP-FU | 10   | 10               | 10            |
| DCEPS   | 6    | 6                | 6             |
| DSCT    | 71   | 71               | 71            |
| E       | 8    | 8                | 8             |
| EA      | 447  | 447              | 447           |
| EB      | 121  | 121              | 121           |
| EC      | 124  | 124              | 124           |
| ED      | 54   | 54               | 54            |
| EW      | 1092 | 1092             | 1092          |
| HADS    | 31   | 31               | 31            |
| L       | 236  | 236              | 236           |
| ROT     | 790  | 790              | 790           |
| RR      | 6    | 6                | 6             |
| RRAB    | 443  | 443              | 443           |
| RRAB_BL | 55   | 55               | 55            |
| RRC     | 203  | 203              | 203           |
| RRD     | 23   | 23               | 23            |
| RS      | 18   | 18               | 18            |
| SR      | 992  | 992              | 992           |