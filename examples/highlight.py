import pandas as pd
import matplotlib.pyplot as plt
import paxplot

# Import data
path_to_data = paxplot.datasets.tradeoff()
df = pd.read_csv(path_to_data)
cols = df.columns

# Split data
df_highlight = df[df['A'] < 20]
df_grey = df[df['A'] >= 20]

# Create figure
paxfig = paxplot.pax_parallel(n_axes=len(cols))
paxfig.plot(df_highlight.to_numpy())

# Add colorbar for highlighted
color_col = 0
paxfig.add_colorbar(
    ax_idx=color_col,
    cmap='viridis',
    colorbar_kwargs={'label': cols[color_col]}
)

# Add grey data
paxfig.plot(df_grey.to_numpy(), line_kwargs={'alpha': 0.5, 'color': 'grey'})

# Add labels
paxfig.set_labels(cols)

plt.show(block=False)
