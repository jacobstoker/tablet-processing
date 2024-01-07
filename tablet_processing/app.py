from dash import Dash, html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
from main import get_experiment_dataframes, setup_graph_columns, plot_experiments_plotly
from config import graph_to_stage

app = Dash(__name__)

df = pd.read_csv("data/002ETHMCC/002ETHMCC_breaking_data.csv")

app.layout = html.Div(
    [
        html.Div(children="Tablet Processing"),
        html.Hr(),
        dcc.RadioItems(
            options=["Heckel", "Compactability", "Compressibility", "Tabletability"],
            value="Heckel",
            id="graph-type-radio",
        ),
        html.Details(
            [
                html.Div(
                    [
                        dcc.Checklist(
                            id="experiments-multi-dropdown",
                            options=[
                                {"label": "002ETHMCC", "value": "002ETHMCC"},
                                {"label": "002LYCOMCC", "value": "002LYCOMCC"},
                                {"label": "002MGSTMCC", "value": "002MGSTMCC"},
                                {"label": "002PARAMCC", "value": "002PARAMCC"},
                            ],
                            value=["002ETHMCC"],
                        )
                    ],
                    className="updates-list",
                ),
                html.Summary(
                    html.Code("Experiments"),
                    className="updates-header",
                ),
            ],
            id="experiment-selection",
        ),
        html.Details(
            [
                html.Div(
                    [
                        dcc.Input(
                            id="graph-title", type="text", placeholder="Graph Title"
                        ),
                        dcc.Input(
                            id="x-axis-title", type="text", placeholder="X-Axis Title"
                        ),
                        dcc.Input(
                            id="y-axis-title", type="text", placeholder="Y-Axis Title"
                        ),
                    ],
                    className="graph-titles",
                ),
                html.Summary(
                    html.Code("Graph Titles"),
                    className="updates-graph-titles",
                ),
            ],
            id="graph-title-selection",
        ),
        # dcc.Input(id="graph-title", type="text", placeholder="Graph Title"),
        # dcc.Input(id="x-axis-title", type="text", placeholder="X-Axis Title"),
        # dcc.Input(id="y-axis-title", type="text", placeholder="Y-Axis Title"),
        html.Button("Update Graph", id="update-graph-button"),
        dcc.Graph(figure={}, id="main-graph"),
        html.Div(id="title-output"),
        html.Div(id="x-axis-output"),
        html.Div(id="y-axis-output"),
        dcc.Textarea(
            id="active-checkboxes-output",
            value="",
            readOnly=True,
            style={"width": "100%"},
        ),
    ]
)


# @callback(
#     Output("active-checkboxes-output", "value"),
#     [Input("checkboxes", "value")],
#     [State("checkboxes", "options")],
# )
# def update_active_checkboxes(checkbox_values, checkbox_options):
#     if checkbox_values:
#         active_checkboxes = [
#             option["label"]
#             for option in checkbox_options
#             if option["value"] in checkbox_values
#         ]
#         return "\n".join(active_checkboxes)
#     return ""


@callback(
    Output("main-graph", "figure"),
    [
        Input("update-graph-button", "n_clicks"),
        State("graph-type-radio", "value"),
        State("experiments-multi-dropdown", "value"),
        State("graph-title", "value"),
        State("x-axis-title", "value"),
        State("y-axis-title", "value"),
    ],
)
def create_graph(button_clicks, graph_type, experiments, title, x_axis, y_axis):
    if button_clicks is None:
        raise PreventUpdate

    stage = graph_to_stage[graph_type]
    experiment_dataframes = get_experiment_dataframes(
        experiment_stage=stage, force_redo=False, selected_experiments=experiments
    )
    x_col, y_col, experiment_dfs = setup_graph_columns(
        graph_type=graph_type, experiment_dataframes=experiment_dataframes
    )

    fig = plot_experiments_plotly(
        experiment_dataframes=experiment_dfs, x_column=x_col, y_column=y_col
    )
    return fig


# @callback(
#     Output("active-checkboxes-output", "value"),
#     [Input("graph-type-radio", "value"), Input("experiments-multi-dropdown", "value")],
# )
# def update_graph(graph_type, selected_experiments):
#     return f"Plotting a {graph_type} graph for {selected_experiments}"


# @callback(Output("title-output", "children"), Input("graph-title", "value"))
# def update_title(updated_title):
#     return updated_title


# @callback(Output("x-axis-output", "children"), Input("x-axis-title", "value"))
# def update_x_axis(updated_title):
#     return updated_title


# @callback(Output("y-axis-output", "children"), Input("y-axis-title", "value"))
# def update_y_axis(updated_title):
#     return updated_title


# @callback(
#     Output(component_id="controls-and-graph", component_property="figure"),
#     Input(component_id="graph-type-radio", component_property="value"),
# )
# def update_graph(graph_type):
#     x_col, y_col, df = setup_graph_columns(
#         graph_type=graph_type, experiment_dataframes=experiment_dataframes
#     )
#     print(df.head())
#     fig = px.scatter(df, x=x_col, y=y_col)
#     return fig


if __name__ == "__main__":
    # experiment_dataframes = {}
    # experiment_dataframes["Making"] = get_all_experiment_dataframes(
    #     experiment_stage="Making", force_redo=False
    # )
    # experiment_dataframes["Breaking"] = get_all_experiment_dataframes(
    #     experiment_stage="Breaking", force_redo=False
    # )
    app.run_server(debug=True)

"""
Experiment Buttons
- Call get_experiment_dataframe on the experiment_dataframes


Graph Radio
- Call setup_graph_columns



"""
