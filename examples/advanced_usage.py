import paxplot
import matplotlib.pyplot as plt

data = [
    [0.0, 0.0, 2.0],
    [1.0, 1.0, 1.0],
    [3.0, 2.0, 0.0],
]

# Change uniform tick spacing
paxfig = paxplot.pax_parallel(n_axes=3)
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
paxfig = paxplot.pax_parallel(n_axes=3)
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
paxfig = paxplot.pax_parallel(n_axes=3)
paxfig.plot(data)
paxfig.set_lim(ax_idx=0, bottom=-1.0, top=3.0)
paxfig.set_lim(ax_idx=2, bottom=1.0, top=3.0)
plt.show()

# Invert
paxfig = paxplot.pax_parallel(n_axes=3)
paxfig.plot(data)
paxfig.invert_axis(ax_idx=0)
paxfig.invert_axis(ax_idx=1)
plt.show()
