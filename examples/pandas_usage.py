import pandas as pd
import matplotlib.pyplot as plt
import paxplot

# Import data
path_to_data = paxplot.datasets.tradeoff()
df = pd.read_csv(path_to_data)
cols = df.columns

# Create figure
paxfig = paxplot.pax_parallel(n_axes=len(cols))
paxfig.plot(df.to_numpy())

# Add labels
paxfig.set_labels(cols)

# Add colorbar
color_col = 0
paxfig.add_colorbar(
    ax_idx=color_col,
    cmap='viridis',
    colorbar_kwargs={'label': cols[color_col]}
)

plt.show()
