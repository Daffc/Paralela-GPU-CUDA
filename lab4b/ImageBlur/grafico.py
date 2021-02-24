import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Bar(
        x=["0","1","3","4","5","6","7","8","9","Sunflower","Lena"], 
        y=[305.920, 1131.008, 400.896, 4483.840, 5455.104, 9920.768, 10198.016, 4596.992, 1606.912, 253.184, 1130.752], 
        name="ImageBlur (no shared mem)"),
        secondary_y=False,
)

fig.add_trace(
    go.Bar(
        x=["0","1","3","4","5","6","7","8","9","Sunflower","Lena"], 
        y=[167.936, 508.160, 206.080, 1843.968, 2034.944, 3770.112, 3805.952, 1870.080, 718.080, 160.000, 731.904], 
        name="ImageBlurSHM + Conversions"),
        secondary_y=False,
)

fig.add_trace(
    go.Bar(
        x=["0","1","3","4","5","6","7","8","9","Sunflower","Lena"], 
        y=[118.016, 428.032, 160.000, 1614.848, 1792.000, 3343.104, 3337.984, 1617.920, 617.984, 89.088, 424.960], 
        name="ImageBlurSHM (kernel only)"),
        secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=["0","1","3","4","5","6","7","8","9","Sunflower","Lena"], 
    y=[1.822, 2.226, 1.945, 2.432, 2.681, 2.631, 2.679, 2.458, 2.238, 1.582, 1.545], 
    name="Speedup (with conversions)"),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(x=["0","1","3","4","5","6","7","8","9","Sunflower","Lena"], 
    y=[2.592, 2.642, 2.506, 2.777, 3.044, 2.968, 3.055, 2.841, 2.600, 2.842, 2.661], 
    name="Speedup (kernel only)"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Comparação entre kernels (GTX970)",
)

# Set x-axis title
fig.update_xaxes(
    title_text="Imagens",
    type='category'
)

# Set y-axes titles
fig.update_yaxes(
    title_text="Tempo em Microsegundos <b>μs</b>", 
    tick0=0, 
    dtick=1000,
    rangemode="tozero",
    secondary_y=False,
    )
fig.update_yaxes(
    title_text="<b>Speedup</b>", 
    tick0=0, 
    dtick=0.25,
    rangemode="tozero",
    secondary_y=True
)

fig.show()