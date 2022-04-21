import matplotlib.pyplot as plt
import pandas as pd
import paxplot

# Data
df = pd.DataFrame(
    {
        'J': ['A', 'A', 'A', 'B', 'B'],
        'K': [0, 1, 2, 3, 4]
    }
)
cols = df.columns

# Create figure
paxfig = paxplot.pax_parallel(n_axes=len(cols))
paxfig.plot(df.to_numpy())

plt.show()
