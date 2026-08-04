import plotly.graph_objects as go
import plotly.offline

def plot_tokens(sentences_data, title, dims=[0, 1, 2]):
    data = [
        go.Scatter3d(
            x = sentences_data["words"][:, dims[0]],
            y = sentences_data["words"][:, dims[1]],
            z = sentences_data["words"][:, dims[2]],
            mode = "markers+text",
            marker=dict(
                size = 6,
                color = sentences_data["color"]
            ),
            text = sentences_data["labels"],
            hoverinfo="text"
        )for sentences_data in sentences_data
    ]
    layout = go.Layout(
    scene = dict(
        xaxis_title="Hardness",
        yaxis_title="Brightness",
        zaxis_title="Redness"
    ),
    title = "title")

    fig = go.Figure(data = data, layout = layout)
    plotly.offline.iplot(fig)