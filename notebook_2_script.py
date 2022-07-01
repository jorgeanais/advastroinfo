"""
Perform fourier transformation, lomb-Scargle Periodogram,
and the feature extraction from the feets package on all
stars from the TESS sample.
"""
import glob
from tkinter import SEPARATOR
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# from astroML.time_series import lomb_scargle, lomb_scargle_BIC, lomb_scargle_bootstrap
import feets


def fourier_transorm(x: np.ndarray, y: np.ndarray, k: int = 50) -> np.ndarray:
    """
    Compute the Fourier transform of a time series. Truncated at k frequencies/modes.
    """
    y_fft = np.fft.fft(y)
    y_fft[k + 1 : -k] = 0  # truncate to only include the first k modes
    return np.fft.ifft(y_fft).real


def extract_features(
    time: np.ndarray, magnitude: np.ndarray, error: np.ndarray
) -> dict[str, float]:
    """
    Extract features from the feets package.
    """
    FEATURE_LIST = [
        "CAR_mean",
        "CAR_sigma",
        "CAR_tau",
        "Eta_e",
        "LinearTrend",
        "MaxSlope",
        "PeriodLS",
        "Period_fit",
        "Psi_CS",
        "Psi_eta",
        "SlottedA_length",
        "StetsonK",
        "StetsonK_AC",
        "StructureFunction_index_21",
        "StructureFunction_index_31",
        "StructureFunction_index_32",
    ]

    fs = feets.FeatureSpace(only=FEATURE_LIST)
    features, values = fs.extract(time, magnitude, error)
    return dict(zip(features, values))


def process_directory(input_dir: str) -> None:
    """
    Process all light curves in a given folder.
    """

    COLNAMES = {
        "_TESS_lightcurves_raw": ["JD", "mag", "err"],
        "_TESS_lightcurves_median_after_detrended": [
            "JD",
            "mag_clean",
            "mag_after_cbv",
            "err",
        ],
        "_TESS_lightcurves_outliercleaned": ["JD", "mag", "err"],
    }

    SEPARATOR = {
        "_TESS_lightcurves_raw": " ",
        "_TESS_lightcurves_median_after_detrended": " ",
        "_TESS_lightcurves_outliercleaned": ",",
    }

    FEET_COLUMNS = {
        "_TESS_lightcurves_raw": ["JD", "mag", "err"],
        "_TESS_lightcurves_median_after_detrended": ["JD", "mag_after_cbv", "err"],
        "_TESS_lightcurves_outliercleaned": ["JD", "mag", "err"],
    }

    NANVALUES = ["*********", "********", "9.999999", "NaN"]

    # Recursively search for all light curves in the input directory
    lc_files = glob.glob(f"{input_dir}/**/*.lc", recursive=True)

    list_of_features = []
    # Load data and save results
    for i, f in enumerate(lc_files):
        stage, vtype, fname = f.split("/")[-3:]
        df = pd.read_csv(
            f,
            names=COLNAMES[stage],
            sep=SEPARATOR[stage],
            dtype=np.float64,
            na_values=NANVALUES,
        ).dropna(how="all")

        time, mag, error = [df[col].values for col in FEET_COLUMNS[stage]]
        extracted_features = extract_features(time, mag, error)
        extracted_features["object_id"] = fname.replace(".lc", "")
        extracted_features["type"] = vtype

        list_of_features.append(extracted_features)

        # Stop at X iterations. For testing purposes only.
        if i == -1:
            break

    pd.DataFrame(list_of_features).to_csv("outputs/nb2/features.csv")


def main() -> None:
    process_directory("_data/_TESS_lightcurves_outliercleaned")


if __name__ == "__main__":
    main()
