Advanced Usage
==============

We run through the advanced functionality of paxplot using the following synthetic dataset. Note that paxplot requires its input be a list of lists or a similar matrix-like format.

.. code-block:: python

   data = [
      [0.0, 0.0, 2.0],
      [1.0, 1.0, 1.0],
      [3.0, 2.0, 0.0],
   ]

Change Number of Ticks
----------------------
By default, paxplot chooses evenly-spaced ticks between the upper and lower limits of the plotted data. You can change that option!

.. code-block:: python

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

.. image:: _static/even_ticks.svg

Custom Ticks
------------
Paxplot also gives you the flexibility to set whatever ticks you want, and they can say whatever you want!

.. code-block:: python

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

.. image:: _static/custom_ticks.svg

Change Axis Limits
------------------
By default, paxplot chooses the axis limits as bounds of the plotted data. You can also change that!

.. code-block:: python

    paxfig = paxplot.pax_parallel(n_axes=3)
    paxfig.plot(data)
    paxfig.set_lim(ax_idx=0, bottom=-1.0, top=3.0)
    paxfig.set_lim(ax_idx=2, bottom=1.0, top=3.0)
    plt.show()

.. image:: _static/limits.svg

Axis Inversion
--------------
Sometimes it's helpful to invert (flip) an axis.

.. code-block:: python

    paxfig = paxplot.pax_parallel(n_axes=3)
    paxfig.plot(data)
    paxfig.invert_axis(ax_idx=0)
    paxfig.invert_axis(ax_idx=1)
    plt.show()

.. image:: _static/invert.svg

Adding a Legend
---------------
It can be nice to plot a legend to identify each line. This works well if you have a few observations.

.. code-block:: python

    paxfig = paxplot.pax_parallel(n_axes=3)
    paxfig.plot(data)
    paxfig.add_legend(labels=['Line A', 'Line B', 'Line C'])
    plt.show()

.. image:: _static/legend.svg

Adding a Legend
---------------
If you have many observations, it is helpful to use a colorbar to identify each line. You should also reference the pandas integration example for another example of using a colorbar.

.. code-block:: python

    paxfig = paxplot.pax_parallel(n_axes=3)
    paxfig.plot(data)
    paxfig.set_label(
        ax_idx=2,
        label='Column C'
    )
    paxfig.add_colorbar(
        ax_idx=2,
        cmap='viridis',
        colorbar_kwargs={'label': 'Column C'}
    )
    plt.show()

.. image:: _static/colorbar.svg

Accessing Matplotlib Objects
----------------------------
Paxplot is an extension of matplotlib's `subplots <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_ wrapper. Paxplot gives you the ability to acess the individual matplotlib axes as well as all the associated functionality using :code:`paxfig.axes`. To demonstrate this, imagine you want to annotate your paxplot with a label and arrow. That functionality has not been explicitly added to paxplot, however it does exit for matplotlib `axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.arrow.html>`_. Paxplot still allows us to axcess that functionality!

.. code-block:: python

    paxfig = paxplot.pax_parallel(n_axes=3)
    paxfig.plot(data)
    paxfig.axes[0].annotate('My Label', (0.3, 0.6))
    paxfig.axes[0].arrow(0.42, 0.56, 0.0, -0.09, head_width=0.03)
    plt.show()

.. warning::
    
    Access matplotib axes with caution. Some axes functions can break your paxfig object.

.. image:: _static/arrow.svg