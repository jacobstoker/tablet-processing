import pandas as pd
import numpy as np

"""
Very work in progress initial draft (including this vague description)
Average a dataframe by one of its columns with a bin width
"""


def create_average_dataframe(input_df: pd.DataFrame, bin_width: int):
    df = input_df.copy()
    pressure_averaged_dfs = []

    for pressure, pressure_df in df.groupby("Compressional Pressure"):
        # Averaging identical Standard Force values
        identical_force_grouped = (
            pressure_df.groupby("Standard Force")
            .agg(
                {
                    "Standard Travel": "mean",
                    "Tool Separation": "mean",
                    "Absolute cross": "mean",
                }
            )
            .reset_index()
        )

        identical_force_grouped["Force Bin"] = pd.cut(
            identical_force_grouped["Standard Force"],
            bins=np.arange(
                0,
                identical_force_grouped["Standard Force"].max() + bin_width,
                bin_width,
            ),
        )

        bin_grouped = (
            identical_force_grouped.groupby("Force Bin")
            .agg(
                {
                    "Standard Travel": "mean",
                    "Tool Separation": "mean",
                    "Absolute cross": "mean",
                    "Standard Force": "mean",
                }
            )
            .reset_index()
        )

        bin_grouped["Compressional Pressure"] = pressure
        bin_grouped.drop(columns="Force Bin", inplace=True)

        pressure_averaged_dfs.append(bin_grouped)

    result_df = pd.concat(pressure_averaged_dfs, ignore_index=True)

    return result_df
