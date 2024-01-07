from pathlib import Path
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import RADIUS_SQUARED, RADIUS_SQUARED_CM, TOOL_HEIGHT, graph_to_stage
import plotly.express as px

logging.basicConfig(level=logging.INFO)


def get_list_of_subdirectories(base_directory: Path) -> list:
    """Return a list containing the full paths to all subdirectories in a given directory"""
    return [
        subdirectory
        for subdirectory in base_directory.iterdir()
        if subdirectory.is_dir()
    ]


def get_experiment_dataframe(
    experiment_path: Path,
    experiment_stage: str,
    force_redo: bool,
) -> pd.DataFrame:
    """Return a dataframe from the input experiment path, either reading or creating a new one where necessary"""

    experiment_data_path = (
        experiment_path / f"{experiment_path.name}_{experiment_stage.lower()}_data.csv"
    )

    if experiment_data_path.exists() and not force_redo:
        experiment_data_df = pd.read_csv(str(experiment_data_path))
        logging.debug(f"NOTE: CSV {experiment_data_path} exists - not regenerating")
    else:
        experiment_data_df = create_experiment_dataframe(
            experiment_path=experiment_path,
            experiment_stage=experiment_stage,
        )
        experiment_data_df.to_csv(str(experiment_data_path), index=False)

    return experiment_data_df


def create_experiment_dataframe(
    experiment_path: Path, experiment_stage: str
) -> pd.DataFrame:
    """Return a combined dataframe from the experiment_details and each corresponding tablet"""
    experiment_data_path = experiment_path / experiment_stage

    experiment_details_path = experiment_path / f"{experiment_path.name}_details.csv"
    experiment_details = pd.read_csv(str(experiment_details_path))
    tablet_numbers = experiment_details["Tablet number"].tolist()

    tablet_dfs = []
    for tablet_number in tablet_numbers:
        tablet_file_path = experiment_data_path / f"Tablet {tablet_number}.txt"
        if tablet_file_path.exists():
            experiment_data = pd.read_csv(str(tablet_file_path))
            experiment_data["Tablet number"] = tablet_number
            tablet_dfs.append(experiment_data)
        else:
            logging.debug(
                f"NOTE: Skipping tablet {tablet_number} because {tablet_file_path} does not exist"
            )

    combined_df = pd.concat(tablet_dfs, ignore_index=True)
    merged_df = pd.merge(experiment_details, combined_df, on="Tablet number")

    return merged_df


def get_all_experiment_dataframes(experiment_stage: str, force_redo: bool) -> dict:
    """Return a dictionary of all experiment names, and their corresponding DataFrame"""
    path_to_data = Path("data")
    experiments = get_list_of_subdirectories(path_to_data)
    experiment_dataframes = {}
    for experiment in experiments:
        experiment_data_df = get_experiment_dataframe(
            experiment_path=experiment,
            experiment_stage=experiment_stage,
            force_redo=force_redo,
        )
        experiment_dataframes[experiment.name] = experiment_data_df

    return experiment_dataframes


def get_experiment_dataframes(
    experiment_stage: str, force_redo: bool, selected_experiments: list
):
    """Return a dictionary of all experiment names, and their corresponding DataFrame"""
    path_to_data = Path("data")
    experiments = get_list_of_subdirectories(path_to_data)
    if selected_experiments:
        experiments = [
            experiment
            for experiment in experiments
            if any(
                selected_experiment in str(experiment)
                for selected_experiment in selected_experiments
            )
        ]

    experiment_dataframes = {}
    for experiment in experiments:
        experiment_data_df = get_experiment_dataframe(
            experiment_path=experiment,
            experiment_stage=experiment_stage,
            force_redo=force_redo,
        )
        experiment_dataframes[experiment.name] = experiment_data_df

    return experiment_dataframes


def setup_graph(x_label: str, y_label: str, title: str):
    """Apply labels, titles etc to the matplotlib graph"""
    sns.set(style="darkgrid")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def plot_experiments(experiment_dataframes: dict, x_column: str, y_column: str):
    """Plot a matplotlib scatter graph for a given dictionary of dataframes and the specified columns within them"""
    for experiment, experiment_df in experiment_dataframes.items():
        missing_columns = [
            column
            for column in [x_column, y_column]
            if column not in experiment_df.columns
        ]
        if missing_columns:
            logging.error(
                f"ERROR: {missing_columns} are not in the dataframe for {experiment}"
            )

        plt.scatter(experiment_df[x_column], experiment_df[y_column], label=experiment)
    plt.legend()
    plt.show()
    return


def plot_experiments_plotly(experiment_dataframes: dict, x_column: str, y_column: str):
    """Plot a matplotlib scatter graph for a given dictionary of dataframes and the specified columns within them"""
    fig = px.scatter()
    for experiment, experiment_df in experiment_dataframes.items():
        missing_columns = [
            column
            for column in [x_column, y_column]
            if column not in experiment_df.columns
        ]
        if missing_columns:
            logging.error(
                f"ERROR: {missing_columns} are not in the dataframe for {experiment}"
            )
        fig.add_trace(px.scatter(experiment_df, x=x_column, y=y_column).data[0])

    return fig


def setup_and_plot_graph(graph_type: str, experiment_dataframes: dict):
    """
    Given a dictionary of experiment dataframes and a graph type, calculate values for any missing columns
    and plot the appropriate graph
    """
    add_common_columns(experiment_dataframes)
    if graph_type == "Heckel":
        setup_graph(
            x_label="Pressure (MPa)",
            y_label="ln(1/1-D)",
            title="Comparison of powders Heckel curves",
        )
        add_common_making_columns(experiment_dataframes)
        add_heckel_columns(experiment_dataframes)
        plot_experiments(
            experiment_dataframes=experiment_dataframes,
            x_column="pressure",
            y_column="ln(1/1-D)",
        )
    elif graph_type == "Compressibility":
        setup_graph(
            x_label="Pressure (MPa)",
            y_label="Porosity (1-RD)",
            title="Compressibility Profiles (0.1 V/V)",
        )
        add_common_making_columns(experiment_dataframes)
        add_compressibility_columns(experiment_dataframes)
        # for experiment, df in experiment_dataframes.items():
        #     print(f"\n\n{experiment}")
        #     print(df)
        plot_experiments(
            experiment_dataframes=experiment_dataframes,
            x_column="pressure",
            y_column="porosity",
        )
    elif graph_type == "Compactability":
        setup_graph(
            x_label="Out-of-die Porosity",
            y_label="Tensile Strength MPa",
            title="Compactibility Comparison",
        )
        add_common_breaking_columns(experiment_dataframes)
        add_compactability_columns(experiment_dataframes)
        compactability_dataframes = calculate_compactability(experiment_dataframes)
        plot_experiments(
            experiment_dataframes=compactability_dataframes,
            x_column="porosity",
            y_column="tensile_strength",
        )
    elif graph_type == "Tabletability":
        setup_graph(
            x_label="V/V of Additive",
            y_label="Tensile Strength (MPa)",
            title="Tensile Strength of Tablets at MPa",
        )
        add_common_breaking_columns(experiment_dataframes)
        add_tabletability_columns(experiment_dataframes)
        tabletability_dataframes = calculate_tabletability(experiment_dataframes)
        plot_experiments(
            experiment_dataframes=tabletability_dataframes,
            x_column="volume_occupancy",
            y_column="average_tensile_strength",
        )


def setup_graph_columns(graph_type: str, experiment_dataframes: dict):
    """
    Given a dictionary of experiment dataframes and a graph type, calculate values for any missing columns
    and plot the appropriate graph
    """
    add_common_columns(experiment_dataframes)
    if graph_type == "Heckel":
        add_common_making_columns(experiment_dataframes)
        add_heckel_columns(experiment_dataframes)
        return ("pressure", "ln(1/1-D)", experiment_dataframes)
    elif graph_type == "Compressibility":
        add_common_making_columns(experiment_dataframes)
        add_compressibility_columns(experiment_dataframes)
        return ("pressure", "porosity", experiment_dataframes)
    elif graph_type == "Compactability":
        add_common_breaking_columns(experiment_dataframes)
        add_compactability_columns(experiment_dataframes)
        compactability_dataframes = calculate_compactability(experiment_dataframes)
        return ("porosity", "tensile_strength", compactability_dataframes)
    elif graph_type == "Tabletability":
        add_common_breaking_columns(experiment_dataframes)
        add_tabletability_columns(experiment_dataframes)
        tabletability_dataframes = calculate_tabletability(experiment_dataframes)
        return (
            "volume_occupancy",
            "average_tensile_strength",
            tabletability_dataframes,
        )
    else:
        logging.error(f"ERROR: graph type {graph_type} is unsupported")
        exit()


def add_common_columns(experiment_dataframes: dict):
    """Calculate any missing columns in the dataframes that are needed for all graphs"""
    for experiment, experiment_df in experiment_dataframes.items():
        if "true_density_binary_mixture" not in experiment_df.columns:
            experiment_df["true_density_binary_mixture"] = 1 / (
                (experiment_df["M_a"] / experiment_df["true_density_a"])
                + (experiment_df["M_b"] / experiment_df["true_density_b"])
            )
        if "pressure" not in experiment_df.columns:
            experiment_df["pressure"] = (
                experiment_df["Standard Force"] / (np.pi * RADIUS_SQUARED)
            ) / 1e6  # convert to MPa


def add_common_making_columns(experiment_dataframes: dict):
    """Calculate any missing columns in the dataframes that are needed for Making graphs"""
    for experiment, experiment_df in experiment_dataframes.items():
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
                experiment_df["bulk_density"]
                / experiment_df["true_density_binary_mixture"]
            )


def add_heckel_columns(experiment_dataframes: dict):
    """Calculate any missing columns in the dataframes that are needed for a Heckel graph"""
    for experiment, experiment_df in experiment_dataframes.items():
        if "ln(1/1-D)" not in experiment_df.columns:
            experiment_df["ln(1/1-D)"] = np.log(
                1 / (1 - experiment_df["relative_density"])
            )


def add_compressibility_columns(experiment_dataframes: dict):
    """Calculate any missing columns in the dataframes that are needed for a Compressibility graph"""
    for experiment, experiment_df in experiment_dataframes.items():
        if "porosity" not in experiment_df.columns:
            experiment_df["porosity"] = 1 - experiment_df["relative_density"]


def add_common_breaking_columns(experiment_dataframes: dict):
    """Calculate any missing columns in the dataframes that are needed for Breaking graphs"""
    for experiment, experiment_df in experiment_dataframes.items():
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


def add_compactability_columns(experiment_dataframes: dict):
    """Calculate any missing columns in the dataframes that are needed for a Compactability graph"""
    for experiment, experiment_df in experiment_dataframes.items():
        if "out_of_die_radius" not in experiment_df.columns:
            experiment_df["out_of_die_radius"] = (
                experiment_df["diameter"] / 20
            )  # divide by 2 to get radius, divide by 10 to get cm

        if "out_of_die_height" not in experiment_df.columns:
            experiment_df["out_of_die_height"] = (
                experiment_df["av height"] / 10
            )  # mm -> cm

        if "bulk_density" not in experiment_df.columns:
            experiment_df["bulk_density"] = experiment_df["Mass"] / (
                np.pi
                * pow(experiment_df["out_of_die_radius"], 2)
                * experiment_df["out_of_die_height"]
            )

        if "relative_density" not in experiment_df.columns:
            experiment_df["relative_density"] = (
                experiment_df["bulk_density"]
                / experiment_df["true_density_binary_mixture"]
            )

        if "porosity" not in experiment_df.columns:
            experiment_df["porosity"] = 1 - experiment_df["relative_density"]


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


def add_tabletability_columns(experiment_dataframes: dict):
    """Calculate any missing columns in the dataframes that are needed for a Tabletability graph"""
    # TODO: only the first volume occupancy row is needed at the moment - will more be needed in future graphs?
    for experiment, experiment_df in experiment_dataframes.items():
        if "volume_occupancy" not in experiment_df.columns:
            experiment_df["volume_occupancy"] = (
                experiment_df["M_b"] * experiment_df["true_density_binary_mixture"]
            ) / experiment_df["true_density_b"]


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


if __name__ == "__main__":
    graph = "Heckel"
    stage = graph_to_stage[graph]

    # experiment_dataframes = {}
    # experiment_dataframes["Making"] = get_all_experiment_dataframes(
    #     experiment_stage="Making", force_redo=False
    # )
    # experiment_dataframes["Breaking"] = get_all_experiment_dataframes(
    #     experiment_stage="Breaking", force_redo=False
    # )
    # experiment_dataframes = get_all_experiment_dataframes(
    #     experiment_stage=stage, force_redo=True
    # )
    # setup_and_plot_graph(graph_type=graph, experiment_dataframes=experiment_dataframes)
    get_experiment_dataframes("Breaking", False, ["002ETHMCC", "002MGSTMCC"])
