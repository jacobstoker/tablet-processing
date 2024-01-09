RADIUS = 0.01 / 2  # m
RADIUS_SQUARED = pow(RADIUS, 2)  # m^2
RADIUS_SQUARED_CM = 1 / 4
TOOL_HEIGHT = 70.3

graph_to_stage = {
    "Heckel": "Making",
    "Compressibility": "Making",
    "Compactability": "Breaking",
    "Tabletability": "Breaking",
}


class GraphInfo:
    def __init__(self, stage, x_column, y_column, x_label, y_label, title):
        self.stage = stage
        self.x_column = x_column
        self.y_column = y_column
        self.x_label = x_label
        self.y_label = y_label
        self.title = title


graph_info = {
    "Heckel": GraphInfo(
        stage="Making",
        x_column="pressure",
        y_column="ln(1/1-D)",
        x_label="Pressure (MPa)",
        y_label="ln(1/1-D)",
        title="Comparison of powders Heckel curves",
    ),
    "Compressibility": GraphInfo(
        stage="Making",
        x_column="pressure",
        y_column="porosity",
        x_label="Pressure (MPa)",
        y_label="Porosity (1-RD)",
        title="Compressibility Profiles (0.1 V/V)",
    ),
    "Compactability": GraphInfo(
        stage="Breaking",
        x_column="porosity",
        y_column="tensile_strength",
        x_label="Out-of-die Porosity",
        y_label="Tensile Strength MPa",
        title="Compactibility Comparison",
    ),
    "Tabletability": GraphInfo(
        stage="Breaking",
        x_column="volume_occupancy",
        y_column="average_tensile_strength",
        x_label="V/V of Additive",
        y_label="Tensile Strength (MPa)",
        title="Tensile Strength of Tablets at MPa",
    ),
}
