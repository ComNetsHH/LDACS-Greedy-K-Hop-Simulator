import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import seaborn as sns
import sys
import os
from matplotlib.patches import Patch

# Set matplotlib to use LaTeX for rendering text
plt.rcParams.update({
    'font.family': 'lmodern',
    # "font.serif": 'Times',
    'font.size': 30,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.latex.preamble': r'\usepackage{lmodern}'
})

# Plot bars
def plot_error_bar(mean_rates, margin_errors, strategies, x_data, xlabel, ylabel, path=None, filename=None, title=None, set_ylim=1.02, width=0.1, figsize=(16, 8), style_combinations={}, enable_legend=True, capsize=5, legend_info=[], bar_spacing=0.0):
    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=figsize)

    # Positioning for bars
    x_pos = np.arange(len(x_data))
    width = width  # Adjust width for separation
    num_strategies = len(strategies)

    # Color palette
    colors = [plt.get_cmap('tab10', num_strategies)(i) for i in range(num_strategies)]

    # Plot bars for each strategy
    for id, x_value in enumerate(x_data):
        # Calculate the starting position for the group of bars at this x_data point
        group_start = x_pos[id] - (num_strategies - 1) * (width + bar_spacing) / 2
        for idx, strategy in enumerate(strategies):
            mean = mean_rates[strategy][id]
            moe = margin_errors[strategy][id]
            # bar_position = x_pos[id] + idx * width - (num_strategies - 1) * width / 2
            # Calculate the position of each bar within the group, incorporating bar_spacing
            bar_position = group_start + idx * (width + bar_spacing)
            if style_combinations:
                # Apply color and hatch based on the strategy
                color, hatch = style_combinations[strategy]
                ax.bar(bar_position, mean,
                    color=color, edgecolor='black', hatch=hatch,
                    yerr=moe, capsize=capsize, width=width,
                    label=strategy if id == 0 else None)  # Label only for the first set
            else:
                ax.bar(bar_position, mean,
                    color=colors[idx], edgecolor='black',
                    yerr=moe, capsize=capsize, width=width,
                    label=strategy if id == 0 else None)  # Label only for the first degree

    # Set labels, title, and ticks
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_data)
    ax.tick_params(axis='both')
    if set_ylim:
        ax.set_ylim(0, set_ylim)
    if title:
        ax.set_title(title)

    # Grid styling
    ax.xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    if set_ylim:
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

    # Legend
    if enable_legend and legend_info:
        legend_patches = [Patch(facecolor=color, label=label, hatch=hatch, edgecolor='black') for label, color, hatch in legend_info]
        ax.legend(handles=legend_patches, loc='best')
    elif enable_legend:
        ax.legend(loc='best')

    # Save and show the figure
    # plt.tight_layout()
    # Save plots if required
    if path is not None and filename is not None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, filename + '.pdf'), format='pdf', bbox_inches="tight")
        plt.savefig(os.path.join(path, filename + '.png'), format='png', bbox_inches="tight")
    # Show the plot
    # plt.show()