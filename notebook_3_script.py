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
from scipy import stats

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
    EX_FEATURE_LIST = ["AndersonDarling", "StetsonK", "StetsonK_AC"]

    fs = feets.FeatureSpace(data=['time', 'magnitude', 'error'], exclude=EX_FEATURE_LIST)
    features, values = fs.extract(time, magnitude, error)
    return dict(zip(features, values))


def Anderson_Darling_feature(array: np.ndarray) -> float:
    r = stats.anderson(array)
    return r.statistic


def Stetson_K_index(mag: np.ndarray, error: np.ndarray) -> float:
    """Robust measure of the kurtosis of the magnitude histogram. Stetson (1996)"""
    
    n = len(mag)
    N = n  # (?)
    bias = np.sqrt(n / (n - 1.))
    mean_mag = np.mean(mag)
    
    delta = bias * (mag - mean_mag) / error
    return np.sum(np.abs(delta)) / np.sqrt(np.sum(np.power(delta, 2))) / np.sqrt(N)
    


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

    # Recursively search for all light curves in the input directory
    lc_files = glob.glob(f"{input_dir}/**/*.lc", recursive=True)

    list_of_features = []
    # Load data and save results
    for i, f in enumerate(lc_files):
        stage, vtype, fname = f.split("/")[-3:]
        df = pd.read_csv(f, names=COLNAMES[stage], sep=SEPARATOR[stage]).dropna(
            how="all"
        )

        time, mag, error = [df[col].values for col in FEET_COLUMNS[stage]]
        extracted_features = extract_features(time, mag, error)
        extracted_features["object_id"] = fname.replace(".lc", "")
        extracted_features["type"] = vtype

        list_of_features.append(extracted_features)

        # Stop at X iterations. For testing purposes only.
        if i == -1:
            break

    pd.DataFrame(list_of_features).to_csv("outputs/nb3/features_.csv")


def main() -> None:
    process_directory("_data/_TESS_lightcurves_outliercleaned")


if __name__ == "__main__":
    main()
