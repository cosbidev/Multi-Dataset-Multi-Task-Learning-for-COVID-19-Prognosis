import pandas as pd
import os
import collections
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Table results
scens = ["CV", "LOCO", "EV"]
cv = 10
table_stats = pd.read_csv("./docs/result_table", index_col="Model")

colors = sns.color_palette("hls", len(table_stats))

# Plot
plt.figure(figsize=(8, 5))
sns.boxplot(x="Scen", y="Acc", hue="Model", data=table_stats, palette=colors, fliersize=0)
plt.axvline(x=0.5, color="k", linewidth=1)
plt.axvline(x=1.5, color="k", linewidth=1)
plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.ylabel("Acc(%)")
plt.xlabel("")
plt.tight_layout()
plt.savefig("./docs/Articolo/MEDIA/Img/results", dpi=500)
plt.show()
