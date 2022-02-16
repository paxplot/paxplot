Getting Started
===============

Installation
------------

To use paxplot, first install it using pip:

.. code-block:: console

   $ pip install paxplot

Basic Usage
-----------
We run through the basic functionality of paxplot using the following synthetic dataset. Note that paxplot requires its input be a list of lists or a similar matrix-like format. For usage with pandas see advanced usage.

.. code-block:: python

   data = [
      [0.0, 0.0, 2.0],
      [1.0, 1.0, 1.0],
      [3.0, 2.0, 0.0],
   ]

Creating a Simple Plot
^^^^^^^^^^^^^^^^^^^^^^
First, we will create a simple plot of our :code:`data`

.. code-block:: python

   import paxplot
   import matplotlib.pyplot as plt

   paxfig = core.pax_parallel(n_axes=3)
   paxfig.plot(data)
   plt.show()

.. image:: images/basic.svg