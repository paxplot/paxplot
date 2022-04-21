import matplotlib.pyplot as plt
from numpy import block
import paxplot

paxfig = paxplot.pax_parallel(n_axes=3)
paxfig.plot(
    [
        [0, 0, 0],
        [1, 1, 1]
    ],
    line_kwargs={'color': 'green'}
)

plt.show(block=False)
