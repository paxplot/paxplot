import paxplot
import matplotlib.pyplot as plt

data = [
    [0.0, 0.0, 2.0],
    [1.0, 1.0, 1.0],
    [3.0, 2.0, 0.0],
]

paxfig = paxplot.pax_parallel(n_axes=3)
paxfig.plot(data)
plt.show()