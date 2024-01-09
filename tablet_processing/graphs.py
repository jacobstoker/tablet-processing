import logging
import numpy as np
import pandas as pd
from config import RADIUS_SQUARED, RADIUS_SQUARED_CM, TOOL_HEIGHT
import plotly.express as px
from scipy.optimize import curve_fit


def reduce_df_to_closest_point(
    max_value: int, increment: int, input_df: pd.DataFrame, column_to_reduce: str
) -> pd.DataFrame:
    """Return a dataframe containing the closest points in "column_to_reduce" to the series (0->max_value, increment)"""
    target_pressures = np.arange(0, max_value, increment)
    subset_data = pd.DataFrame()
    for experiment in input_df["experiment"].unique():
        experiment_data = input_df[input_df["experiment"] == experiment]
        for target_pressure in target_pressures:
            closest_point = experiment_data.iloc[
                (experiment_data[column_to_reduce] - target_pressure)
                .abs()
                .argsort()[:1]
            ]
            subset_data = pd.concat([subset_data, closest_point])
    return subset_data


def experiments_scatterplot(
    experiments_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    line_fit: str,
):
    """Return a figure for a plotly scatter graph for the experiments dataframe for the specified columns"""
    fig = px.scatter(
        experiments_df,
        x=x_column,
        y=y_column,
        color="experiment",
    )

    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, title=title)

    if line_fit:
        for experiment, experiment_df in experiments_df.groupby("experiment"):
            x_data = experiment_df[x_column]
            y_data = experiment_df[y_column]

            if line_fit == "linear":
                m, b = np.polyfit(x_data, y_data, 1)
                fig.add_traces(px.line(x=x_data, y=m * x_data + b).data[0])

            elif line_fit == "quadratic":
                a, b, c = np.polyfit(x_data, y_data, 2)
                fig.add_traces(
                    px.line(x=x_data, y=a * x_data**2 + b * x_data + c).data[0]
                )

            elif line_fit == "exponential":
                exp_params, _ = curve_fit(
                    lambda t, a, b: a * np.exp(b * t), x_data, y_data
                )
                fig.add_traces(
                    px.line(
                        x=x_data,
                        y=exp_params[0] * np.exp(exp_params[1] * x_data),
                    ).data[0]
                )

            elif line_fit == "power":
                pow_params, _ = curve_fit(lambda t, a, b: a * t**b, x_data, y_data)
                fig.add_traces(
                    px.line(x=x_data, y=pow_params[0] * x_data ** pow_params[1]).data[0]
                )

    return fig


def setup_graph_columns(experiment_stage: str, experiment_df: pd.DataFrame):
    """Given a dictionary of experiment dataframes and a graph type, calculate values for any missing columns"""
    add_common_columns(experiment_df)
    if experiment_stage == "Making":
        add_common_making_columns(experiment_df)
        add_heckel_columns(experiment_df)
        add_compressibility_columns(experiment_df)
    else:
        add_common_breaking_columns(experiment_df)
        add_compactability_columns(experiment_df)
        add_tabletability_columns(experiment_df)


def create_graph_csvs(experiment_dataframes: dict):
    """Create the graph CSVs for compactability and tabletability"""
    compactability_dataframes = calculate_compactability(experiment_dataframes)
    save_df_dict_as_csv(compactability_dataframes, "compactability")
    tabletability_dataframes = calculate_tabletability(experiment_dataframes)
    save_df_dict_as_csv(tabletability_dataframes, "tabletability")


def save_df_dict_as_csv(experiment_dataframes: dict, name: str):
    for experiment, df in experiment_dataframes.items():
        path_to_csv = experiment / "CSV" / f"{name}.csv"
        df.to_csv(str(path_to_csv), index=False)


def add_common_columns(experiment_df: pd.DataFrame):
    """Calculate any missing columns in the dataframes that are needed for all graphs"""
    if "true_density_binary_mixture" not in experiment_df.columns:
        experiment_df["true_density_binary_mixture"] = 1 / (
            (experiment_df["M_a"] / experiment_df["true_density_a"])
            + (experiment_df["M_b"] / experiment_df["true_density_b"])
        )
    if "pressure" not in experiment_df.columns:
        experiment_df["pressure"] = (
            experiment_df["Standard Force"] / (np.pi * RADIUS_SQUARED)
        ) / 1e6  # convert to MPa


def add_common_making_columns(experiment_df: pd.DataFrame):
    """Calculate any missing columns in the dataframes that are needed for Making graphs"""
    if "height_of_powder" not in experiment_df.columns:
        experiment_df["height_of_powder"] = (
            experiment_df["Tool Separation"] - TOOL_HEIGHT
        )
    if "bulk_density" not in experiment_df.columns:
        experiment_df["bulk_density"] = experiment_df["Mass"] / (
            np.pi * RADIUS_SQUARED_CM * experiment_df["height_of_powder"]
        )
    if "relative_density" not in experiment_df.columns:
        experiment_df["relative_density"] = (
            experiment_df["bulk_density"] / experiment_df["true_density_binary_mixture"]
        )


def add_heckel_columns(experiment_df: pd.DataFrame):
    """Calculate any missing columns in the dataframes that are needed for a Heckel graph"""
    if "ln(1/1-D)" not in experiment_df.columns:
        experiment_df["ln(1/1-D)"] = np.log(1 / (1 - experiment_df["relative_density"]))


def add_compressibility_columns(experiment_df: pd.DataFrame):
    """Calculate any missing columns in the dataframes that are needed for a Compressibility graph"""
    if "porosity" not in experiment_df.columns:
        experiment_df["porosity"] = 1 - experiment_df["relative_density"]


def add_common_breaking_columns(experiment_df: pd.DataFrame):
    """Calculate any missing columns in the dataframes that are needed for Breaking graphs"""
    if "tensile_strength" not in experiment_df.columns:
        max_force = experiment_df["Standard Force"].max()
        experiment_df["tensile_strength"] = (2 * max_force) / (
            np.pi
            * (experiment_df["diameter"] / 1000)
            * (experiment_df["av height"] / 1000)
        )  # /1000 to convert meters -> milimeters
        experiment_df["tensile_strength"] = (
            experiment_df["tensile_strength"] / 1e6
        )  # convert to MPa


def add_compactability_columns(experiment_df: pd.DataFrame):
    """Calculate any missing columns in the dataframes that are needed for a Compactability graph"""
    if "out_of_die_radius" not in experiment_df.columns:
        experiment_df["out_of_die_radius"] = (
            experiment_df["diameter"] / 20
        )  # divide by 2 to get radius, divide by 10 to get cm

    if "out_of_die_height" not in experiment_df.columns:
        experiment_df["out_of_die_height"] = experiment_df["av height"] / 10  # mm -> cm

    if "bulk_density" not in experiment_df.columns:
        experiment_df["bulk_density"] = experiment_df["Mass"] / (
            np.pi
            * pow(experiment_df["out_of_die_radius"], 2)
            * experiment_df["out_of_die_height"]
        )

    if "relative_density" not in experiment_df.columns:
        experiment_df["relative_density"] = (
            experiment_df["bulk_density"] / experiment_df["true_density_binary_mixture"]
        )

    if "porosity" not in experiment_df.columns:
        experiment_df["porosity"] = 1 - experiment_df["relative_density"]


def add_tabletability_columns(experiment_df: pd.DataFrame):
    """Calculate any missing columns in the dataframes that are needed for a Tabletability graph"""
    # TODO: only the first volume occupancy row is needed at the moment - will more be needed in future graphs?
    if "volume_occupancy" not in experiment_df.columns:
        experiment_df["volume_occupancy"] = (
            experiment_df["M_b"] * experiment_df["true_density_binary_mixture"]
        ) / experiment_df["true_density_b"]


def calculate_compactability(experiment_dataframes: dict) -> dict:
    """Return a dictionary of new dataframes with calculated Compactability values from the input experimental data"""
    compactability_dataframes = {}
    for experiment, experiment_df in experiment_dataframes.items():
        compactability_df = (
            experiment_df.groupby("Compressional Pressure")
            .agg({"porosity": "mean", "tensile_strength": "mean"})
            .reset_index()
        )
        compactability_dataframes[experiment] = compactability_df

    return compactability_dataframes


def calculate_tabletability(experiment_dataframes: dict) -> dict:
    """Return a dictionary of new dataframes with calculated Tabletability values from the input experimental data"""
    tabletability_dataframes = {}
    for experiment, experiment_df in experiment_dataframes.items():
        average_tensile_strength = experiment_df["tensile_strength"].mean()
        volume_occupancy = float(
            experiment_df.iloc[0, experiment_df.columns.get_loc("volume_occupancy")]
        )
        tabletability_dataframes[experiment] = pd.DataFrame(
            {
                "volume_occupancy": [volume_occupancy],
                "average_tensile_strength": [average_tensile_strength],
            }
        )

    return tabletability_dataframes


x_y_column_names = {
    "Heckel": ("pressure", "ln(1/1-D)"),
    "Compressibility": ("pressure", "porosity"),
    "Compactability": ("porosity", "tensile_strength"),
    "Tabletability": ("volume_occupancy", "average_tensile_strength"),
}


axis_and_title = {
    "Heckel": {
        "x_label": "Pressure (MPa)",
        "y_label": "ln(1/1-D)",
        "title": "Comparison of powders Heckel curves",
    },
    "Compressibility": {
        "x_label": "Pressure (MPa)",
        "y_label": "Porosity (1-RD)",
        "title": "Compressibility Profiles (0.1 V/V)",
    },
    "Compactability": {
        "x_label": "Out-of-die Porosity",
        "y_label": "Tensile Strength MPa",
        "title": "Compactibility Comparison",
    },
    "Tabletability": {
        "x_label": "V/V of Additive",
        "y_label": "Tensile Strength (MPa)",
        "title": "Tensile Strength of Tablets at MPa",
    },
}
