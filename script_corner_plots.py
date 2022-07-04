import numpy as np
import pandas as pd
import corner
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


FEATURES = [
        "Amplitude",
        "Mean",
        #"Beyond1Std",
        "FluxPercentileRatioMid35",
        "Freq1_harmonics_amplitude_0",
        # "Freq2_harmonics_amplitude_0",
        # "Skew",
        "PeriodLS",
        # "SmallKurtosis",
        "Anderson_Darling_",
        "Stetson_K_",
        # "MaxSlope",
    ]


def main():

    # Load data
    data = {
        "raw": pd.read_csv(
            "outputs/nb3/features__TESS_lightcurves_raw.csv"
        ),
        "median_detrended": pd.read_csv(
            "outputs/nb3/features__TESS_lightcurves_median_after_detrended.csv"  # Problem with type
        ),
        "outlier_cleaned": pd.read_csv(
            "outputs/nb3/features__TESS_lightcurves_outliercleaned.csv"
        ),
    }

    # List all variability types using clean lc as reference (assuming all are the same)
    variability_types = list(data["raw"]["type"].unique())

    for vtype in variability_types:
        cornerplot_by_variability_type(vtype, data, FEATURES)


def cornerplot_by_variability_type(
    vtype: str, data: dict[str, pd.DataFrame], cols: list[str] | None = None
) -> None:
    """
    Corner plot for a given variability type (`vtype`) across all input dataframes.
    Each input dataframe represent a stage: raw, median_detrended, outlier_cleaned.
    `cols` indicate the column names to be plotted.
    """

    print(vtype)
    # Select only data with matching variability type
    data_selection = {
        lc_stage: df.query(f"type=='{vtype}'").copy() for lc_stage, df in data.items()
    }

    # Plot all common columns (features) if `cols` is None
    if cols is None:
        for df in data_selection.values():
            aux_cols = set(df.columns.values)
            cols = cols.intersection(aux_cols) if cols is not None else aux_cols
        cols = list(cols)

    # Check number of sources
    for df in data_selection.values():
        if len(df) <= 20:
            print(f"{vtype} has {len(df)} sources. Not enough to plot.")
            return None
        

    # Assign a different color for each lc stage
    n = len(data_selection)
    print(n)
    cmap = plt.cm.get_cmap('gist_rainbow', n)
    colors = [cmap(i) for i in range(n)]

    # Generate the corner plot
    fig = None
    for (stage, df), color in zip(data_selection.items(), colors):
        print(stage, color)
        # quantiles=[0.16, 0.5, 0.84]
        fig = corner.corner(
            df[cols].values,
            labels=cols,
            quantiles=[],
            show_titles=True,
            title_kwargs={"fontsize": 8},
            color=color,
            fig=fig,
        )
    
    # https://stackoverflow.com/questions/56590009/how-to-format-the-corner-plot-in-python
    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=list(data_selection.keys())[i])
            for i in range(n)
        ],
        fontsize=8, frameon=False,
        bbox_to_anchor=(1, n+1),
        loc="upper right"
    )
    plt.suptitle(f"{vtype}", fontsize=14)
    plt.savefig(f"outputs/nb4/cornerplot_{vtype}.png")
    plt.close()


if __name__ == "__main__":
    main()
