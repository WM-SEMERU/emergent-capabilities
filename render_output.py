import matplotlib.pyplot as plt
from statistics import median
import numpy as np

class OutputRenderer:
    def __init__(self, baseline=0.0, metric="(Unspecified metric)"):
        self.x_values = [0.35, 2.00, 6.00, 16.00]
        self.box_color = "Pink"
        self.baseline = baseline
        self.metric = metric

    
    def set_lim(self):
        # hardcoded based on our problem's specifications
        plt.xlim(-1, 17)
        plt.ylim(0, 1)

    
    def draw_box(self, ax):
        bplot = ax.boxplot(
            self.y_values,
            positions=self.x_values,
            widths=1,
            manage_ticks=False,
            patch_artist=True,
            zorder=5,
            medianprops=dict(color="black"),
        )
        
        for patch in bplot["boxes"]:
            patch.set_facecolor(self.box_color)
    
    def draw_line(self, ax):
        medians = [median(vals) for vals in self.y_values]
        line = ax.plot(
            self.x_values,
            medians,
            marker="o",
            color="b",
            linestyle="-",
            markerfacecolor="none",
            markeredgewidth=2,
            markersize=8,
            # alpha=0.5,
            zorder=8,
        )

    def draw_random_annotation(self):
        plt.axhline(
            y=self.baseline,
            color="r",
            # alpha=0.5,
            zorder=2,
        )
        offset = (-0.1, 0.02)
        plt.text(
            plt.xlim()[0] + offset[0],
            self.baseline + offset[1],
            "baseline\nmetric",
            color="r",
            horizontalalignment="right",
        )
    
    def meta_info(self):
        plt.xlabel("Parameters (in billions)")
        plt.ylabel(self.metric)
        plt.title("Model Performance")
    
    def draw_bands(self, ax):
        q1 = [np.percentile(val, 25) for val in self.y_values]
        q3 = [np.percentile(val, 75) for val in self.y_values]
        iqr = [b - a for a, b in zip(q1, q3)]
        # print(q1, q3)
        ax.fill_between(
            self.x_values,
            list(map(lambda x, y: x - 1.5 * y, q1, iqr)),
            list(map(lambda x, y: x + 1.5 * y, q3, iqr)),
            alpha=0.1,
            zorder=5,
            color="b",
        )
        ax.fill_between(
            self.x_values,
            q1,
            q3,
            alpha=0.2,
            zorder=5,
            color="b",
        )
    
    def render(self, ys):
        self.y_values = [
            y if isinstance(y, list)
            else [y]
            for y in ys
        ]

        fig = plt.figure(figsize=(10, 6))
    
        ax1 = fig.add_subplot(111)
        plt.grid(True)
        self.set_lim()
        self.meta_info()
    
        self.draw_random_annotation()
        
        self.draw_bands(ax1)
        self.draw_line(ax1)
        self.draw_box(ax1)
        
        plt.show()