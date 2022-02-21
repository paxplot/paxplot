import paxplot
import matplotlib.pyplot as plt

data = [
    [0.0, 0.0, 2.0, 0.5],
    [1.0, 1.0, 1.0, 1.0],
    [3.0, 2.0, 0.0, 1.0],
]

# Basic plot
paxfig = paxplot.pax_parallel(n_axes=4)
paxfig.plot(data)
plt.show()

# Adding labels
paxfig = paxplot.pax_parallel(n_axes=4)
paxfig.plot(data)
paxfig.set_label(
    ax_idx=0,
    label='A'
)
paxfig.set_label(
    ax_idx=1,
    label='B'
)
paxfig.set_label(
    ax_idx=2,
    label='C'
)
paxfig.set_label(
    ax_idx=3,
    label='D'
)
plt.show()

# Exporting
paxfig.savefig('my_plot.png')