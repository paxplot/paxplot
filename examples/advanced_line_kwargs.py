"""
Demonstrating the use of linekwargs to represent categorical dimensions,
using the common use case of multiobjective model evaluation.
"""
import matplotlib.pyplot as plt
import pandas as pd
import paxplot
from paxplot.datasets import modelevaluation

df = pd.read_csv(modelevaluation())

# map the values from categorical column "data" to linetypes:
period2linestyle = {"cal": "-",
                    "val": ":"}

# map the values from categorical column "model" to hues:
models = df["model"].unique()
n_models = models.shape[0]
qualcmap = plt.get_cmap("tab10")
model2hue = {m: qualcmap.colors[i] for i, m in enumerate(models)}

# prepare data to plot:
quant_cols = df.drop(["model", "data"], axis=1).columns.values
df = df.set_index(["model", "data"])
n_axes = len(quant_cols)

# make figure and plot the metrics for each model and data combination:
paxfig = paxplot.pax_parallel(n_axes=n_axes)
for m in models:
    for d in ["cal", "val"]:
        plotdata = df[quant_cols].loc[m, d]
        paxfig.plot(plotdata.to_numpy(),
                    line_kwargs={"linestyle": period2linestyle[d],
                                 "color": model2hue[m]})

# adjust axis limits to meaningful values and add axis labels:
metric2axlim = {"nse": [.7, 1], "lognse": [.7, 1],
                "kge12": [.7, 1], "aeq5": [15, 0],
                "spearman_r": [.8, 1], "rmse": [.25, 0],
                "logrmse": [.15, 0]}
for i, metric in enumerate(quant_cols):
    paxfig.set_lim(i, *metric2axlim[metric])
    paxfig.set_label(i, metric)

plt.show()
