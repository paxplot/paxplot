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
plt.show(block=False)

# Adding labels
paxfig = paxplot.pax_parallel(n_axes=4)
paxfig.plot(data)
paxfig.set_labels(['A', 'B', 'C', 'D'])
plt.show(block=False)

# Exporting
paxfig.savefig('my_plot.png')
