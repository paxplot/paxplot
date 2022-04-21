import matplotlib.pyplot as plt
import paxplot

paxfig = paxplot.pax_parallel(n_axes=3)
paxfig.plot(
    [
        [0, 0, 0],
        [1, 1, 1]
    ]
)
paxfig.plot([[2, 2, 2]])

plt.show(block=False)
