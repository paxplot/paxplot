import paxplot
import matplotlib.pyplot as plt

data = [
    [0.0, 0.0, 2.0, 0.5],
    [1.0, 1.0, 1.0, 1.0],
    [3.0, 2.0, 0.0, 1.0],
]

# Change uniform tick spacing
paxfig = paxplot.pax_parallel(n_axes=4)
paxfig.plot(data)
paxfig.set_even_ticks(
    ax_idx=0,
    n_ticks=15,
)
paxfig.set_even_ticks(
    ax_idx=1,
    n_ticks=16,
    precision=3
)
plt.show()

# Custom ticks
paxfig = paxplot.pax_parallel(n_axes=4)
paxfig.plot(data)
paxfig.set_ticks(
    ax_idx=0,
    ticks=[0.0, 1.0, 2.0, 3.0],
    labels=['$my_{heart}$', 'code to', '=', '1612']
)
paxfig.set_ticks(
    ax_idx=2,
    ticks=[0.0, 1.0, 1.5, 2.0],
)
plt.show()

# Limits
paxfig = paxplot.pax_parallel(n_axes=4)
paxfig.plot(data)
paxfig.set_lim(ax_idx=0, bottom=-1.0, top=3.0)
paxfig.set_lim(ax_idx=2, bottom=1.0, top=3.0)
plt.show()

# Invert
paxfig = paxplot.pax_parallel(n_axes=4)
paxfig.plot(data)
paxfig.invert_axis(ax_idx=0)
paxfig.invert_axis(ax_idx=1)
plt.show()

# Add legend
paxfig = paxplot.pax_parallel(n_axes=4)
paxfig.plot(data)
paxfig.add_legend(labels=['A', 'B', 'C'])
plt.show()

# Add colorbar
paxfig = paxplot.pax_parallel(n_axes=4)
paxfig.plot(data)
paxfig.set_label(
    ax_idx=0,
    label='Column A'
)
paxfig.add_colorbar(
    ax_idx=0,
    cmap='viridis',
    colorbar_kwargs={'label': 'Column A'}
)
plt.show()

# Accessing matplotlib objects
paxfig = paxplot.pax_parallel(n_axes=4)
paxfig.plot(data)
paxfig.axes[0].annotate('My Label', (0.3, 0.55))
paxfig.axes[0].arrow(
    x=0.5,
    y=0.52, 
    dx=0.0, 
    dy=-0.05, 
    head_width=0.03, 
    head_length=0.02
) 
plt.show()
