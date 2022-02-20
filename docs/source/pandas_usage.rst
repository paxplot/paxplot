Usage with Pandas
=================

The `Pandas <https://pandas.pydata.org/>`_ library offers many tools for importing and processing data. Here is an example of using paxplot to plot data in a pandas `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_. We will use the :code:`tradeoff` dataset included with paxplot, but the example is fairly generic to any dataframe.

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
    for i, col in enumerate(cols):
        paxfig.set_label(i, col)

    # Add colorbar
    color_col = 0
    paxfig.add_colorbar(
        ax_idx=color_col,
        cmap='viridis',
        colorbar_kwargs={'label': cols[color_col]}
    )

    plt.show()


.. image:: images/pandas.svg