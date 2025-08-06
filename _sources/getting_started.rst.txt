Getting Started
===============

Installation
------------

To use Paxplot, first install it using PyPI:

.. code-block:: console

   $ pip install paxplot

Basic Usage
-----------
We run through the basic functionality of Paxplot using the following synthetic dataset. Note that Paxplot requires its input be in a list of lists or a similar matrix-like format. For usage with Pandas, click `here <examples.html#pandas>`__.

.. code-block:: python

   data = [
      [0.0, 0.0, 2.0, 0.5],
      [1.0, 1.0, 1.0, 1.0],
      [3.0, 2.0, 0.0, 1.0],
   ]

Creating a Simple Plot
^^^^^^^^^^^^^^^^^^^^^^
First, we will create a simple plot of our :code:`data`.

.. code-block:: python

   import paxplot
   import matplotlib.pyplot as plt

   paxfig = paxplot.pax_parallel(n_axes=4)
   paxfig.plot(data)
   plt.show()

.. image:: _static/basic.svg

Adding Labels
^^^^^^^^^^^^^
Let's say these columns in :code:`data` correspond to the labels A, B, and C. We can add those labels!

.. code-block:: python

   paxfig = paxplot.pax_parallel(n_axes=4)
   paxfig.plot(data)
   paxfig.set_labels(['A', 'B', 'C', 'D'])
   plt.show()

.. image:: _static/labels.svg

Saving Your Plot
^^^^^^^^^^^^^^^^
You can easily export your plot in many standard vector and raster formats.

.. code-block:: python

   paxfig.savefig('my_plot.png')

Paxplot has lots of additional functionality. Continue onto the next section to see additional examples.
