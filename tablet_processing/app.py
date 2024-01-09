from dash import Dash, html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate
from config import graph_info
from main import get_experiment_dataframes, create_all_experiment_csvs
from graphs import (
    experiments_scatterplot,
    reduce_df_to_closest_point,
)

app = Dash(__name__)

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
                        html.Label("Title"),
                        dcc.Input(
                            id="graph-title",
                            type="text",
                            placeholder="Graph Title",
                            style={"width": "100%"},
                        ),
                    ],
                    className="graph-titles",
                ),
                html.Div(
                    [
                        html.Label("X-Axis"),
                        dcc.Input(
                            id="x-axis-title",
                            type="text",
                            placeholder="X-Axis Title",
                            style={"width": "100%"},
                        ),
                    ],
                    className="graph-titles",
                ),
                html.Div(
                    [
                        html.Label("Y-Axis"),
                        dcc.Input(
                            id="y-axis-title",
                            type="text",
                            placeholder="Y-Axis Title",
                            style={"width": "100%"},
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
        html.Button("Update Graph", id="update-graph-button"),
        dcc.Graph(figure={}, id="main-graph"),
    ]
)


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

    experiments_df = get_experiment_dataframes(
        graph_type=graph_type, selected_experiments=experiments
    )

    x_col = graph_info[graph_type].x_column
    y_col = graph_info[graph_type].y_column

    if graph_type == "Compressibility":
        experiments_df = reduce_df_to_closest_point(
            max_value=250,
            increment=25,
            input_df=experiments_df,
            column_to_reduce="pressure",
        )

    fig = experiments_scatterplot(
        experiments_df=experiments_df,
        x_column=x_col,
        y_column=y_col,
        x_label=x_axis,
        y_label=y_axis,
        title=title,
        line_fit="linear",
    )

    return fig


@callback(Output("x-axis-title", "value"), Input("graph-type-radio", "value"))
def update_x_axis_placeholder(graph_type):
    return graph_info[graph_type].x_label


@callback(Output("y-axis-title", "value"), Input("graph-type-radio", "value"))
def update_y_axis_placeholder(graph_type):
    return graph_info[graph_type].y_label


@callback(Output("graph-title", "value"), Input("graph-type-radio", "value"))
def update_graph_title_placeholder(graph_type):
    return graph_info[graph_type].title


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
    create_all_experiment_csvs("Making", True)
    create_all_experiment_csvs("Breaking", True)
    app.run_server(debug=True)
