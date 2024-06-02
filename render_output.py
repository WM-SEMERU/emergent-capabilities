import matplotlib.pyplot as plt
from statistics import median
import numpy as np

class OutputRenderer:
    def __init__(self, baseline=0.0, metric="(Unspecified metric)"):
        self.x_values = [0.35, 2.70, 6.10, 16.10]
        self.box_color = "Pink"
        self.baseline = baseline
        self.metric = metric

    
    def set_lim(self, y_max=None):
        if y_max is None:
            y_max = 1
        # mostly hardcoded based on our problem's specifications
        plt.xlim(-1, 17)
        plt.ylim(0, y_max)

    
    def draw_box(self, ax, ys):
        bplot = ax.boxplot(
            ys,
            positions=self.x_values,
            widths=1,
            manage_ticks=False,
            patch_artist=True,
            zorder=5,
            medianprops=dict(color="black"),
        )
        
        for patch in bplot["boxes"]:
            patch.set_facecolor(self.box_color)
    
    def draw_line(self, ax, ys, label=None, color="b"):
        medians = [median(vals) for vals in ys]
        line = ax.plot(
            self.x_values,
            medians,
            marker="o",
            color=color,
            linestyle="-",
            markerfacecolor="none",
            markeredgewidth=2,
            markersize=8,
            # alpha=0.5,
            zorder=8,
            label=label,
        )

    def draw_random_annotation(self, y_max=None):
        plt.axhline(
            y=self.baseline,
            color="orange",
            # alpha=0.5,
            zorder=2,
        )
        if y_max is None:
            y_max = 1
        offset = (-0.1, 0.02 * y_max)
        plt.text(
            plt.xlim()[0] + offset[0],
            self.baseline + offset[1],
            "baseline\nmetric",
            color="orange",
            horizontalalignment="right",
        )
    
    def meta_info(self):
        plt.xlabel("Parameters (in billions)")
        plt.ylabel(self.metric)
        plt.title("Model Performance")
    
    def draw_bands(self, ax, ys, color="b"):
        q1 = [np.percentile(val, 25) for val in ys]
        q3 = [np.percentile(val, 75) for val in ys]
        iqr = [b - a for a, b in zip(q1, q3)]
        # print(q1, q3)
        ax.fill_between(
            self.x_values,
            list(map(lambda x, y: x - 1.5 * y, q1, iqr)),
            list(map(lambda x, y: x + 1.5 * y, q3, iqr)),
            alpha=0.1,
            zorder=5,
            color=color,
        )
        ax.fill_between(
            self.x_values,
            q1,
            q3,
            alpha=0.2,
            zorder=5,
            color=color,
        )
    
    def render(self, ys, y_max=None, save=None):
        y_lines = ys
        if not isinstance(y_lines, dict):
            y_lines = { "unnamed": y_lines }

        for key, ys in y_lines.items():
            ys = [
                y if isinstance(y, list)
                else [y]
                for y in ys
            ]
            y_lines[key] = ys

        fig = plt.figure(figsize=(10, 6))
    
        ax1 = fig.add_subplot(111)
        plt.grid(True)
        self.set_lim(y_max=y_max)
        self.meta_info()
    
        self.draw_random_annotation(y_max=y_max)

        colors = ["b", "g", "r"]
        for idx, (key, ys) in enumerate(y_lines.items()):
            color = colors[idx % len(colors)]
            self.draw_bands(ax1, ys, color=color)
            self.draw_line(ax1, ys, label=key, color=color)
            self.draw_box(ax1, ys)

        plt.legend()

        if save is not None:
            # save must come before show
            plt.savefig(save, bbox_inches="tight")
            print("Saved figure to", save)
        
        plt.show()