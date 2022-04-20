Examples
========

Usage with Pandas
-----------------

The `Pandas <https://pandas.pydata.org/>`_ library offers many tools for importing and processing data. Here is an example of using Paxplot to plot data in a Pandas `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_. We will use the :code:`tradeoff` dataset included in Paxplot, but the workflow is fairly generic.

.. code-block:: python

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


.. image:: _static/pandas.svg

Highlight Solutions
-------------------

test