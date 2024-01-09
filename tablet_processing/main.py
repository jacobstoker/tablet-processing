from pathlib import Path
import logging
import pandas as pd
from graphs import setup_graph_columns, create_graph_csvs
from config import graph_info


def get_list_of_subdirectories(base_directory: Path) -> list:
    """Return a list containing the full paths to all subdirectories in a given directory"""
    return [
        subdirectory
        for subdirectory in base_directory.iterdir()
        if subdirectory.is_dir()
    ]


def create_all_experiment_csvs(experiment_stage: str, force_redo: bool):
    """Create a CSV for each experiment in the 'data' directory"""
    path_to_data = Path("data")
    experiments = get_list_of_subdirectories(path_to_data)

    if experiment_stage == "Breaking":
        experiment_dataframes = {}
        for experiment in experiments:
            experiment_df = create_experiment_csv(
                experiment_path=experiment,
                experiment_stage=experiment_stage,
                force_redo=force_redo,
            )
            experiment_dataframes[experiment] = experiment_df
        create_graph_csvs(experiment_dataframes=experiment_dataframes)
    else:
        for experiment in experiments:
            create_experiment_csv(
                experiment_path=experiment,
                experiment_stage=experiment_stage,
                force_redo=force_redo,
            )


def create_experiment_csv(
    experiment_path: Path, experiment_stage: str, force_redo: bool
):
    """Generate a combined dataframe from the experiment_details and data from each corresponding tablet"""

    experiment_csv_path = (
        experiment_path
        / "CSV"
        / f"{experiment_path.name}_{experiment_stage.lower()}_data.csv"
    )

    if experiment_csv_path.exists() and not force_redo:
        logging.debug(f"NOTE: CSV {experiment_csv_path} exists - not regenerating")
        return

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

    setup_graph_columns(experiment_stage=experiment_stage, experiment_df=merged_df)

    experiment_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(str(experiment_csv_path), index=False)
    return merged_df


def get_experiment_dataframes(graph_type: str, selected_experiments: list):
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

    stage = graph_info[graph_type].stage
    experiment_dataframes = []
    for experiment in experiments:
        experiment_data_root = experiment / "CSV"
        if graph_type == "Compactability":
            experiment_data_path = experiment_data_root / "compactability.csv"
        elif graph_type == "Tabletability":
            experiment_data_path = experiment_data_root / "tabletability.csv"
        else:
            experiment_data_path = (
                experiment_data_root / f"{experiment.name}_{stage.lower()}_data.csv"
            )
        experiment_data_df = pd.read_csv(str(experiment_data_path))
        experiment_data_df["experiment"] = experiment.name
        experiment_dataframes.append(experiment_data_df)

    combined_df = pd.concat(experiment_dataframes, ignore_index=True)
    return combined_df


# create_all_experiment_csvs("Making", True)
# create_all_experiment_csvs("Breaking", True)

# graph = "Compactability"

# dfs = get_experiment_dataframes(graph, [])
# for experiment, df in dfs.items():
#     print(f"\n\n{experiment}")
#     print(df.head())
#     print(f"Plotting {x_y_column_names[graph]}")
