"""Script that collects all plotting functions."""

#%% Import libraries
import os
import logging
import math
import random
import string
from copy import deepcopy
from datetime import timedelta

import config as cnf
import helper as helper
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from matplotlib.ticker import FixedFormatter, FixedLocator
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
from matplotlib.cm import ScalarMappable, get_cmap
from collections import defaultdict, Counter
import heapq
from matplotlib.patches import Rectangle
import string
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import ternary

#%% Figure 3: Energy demand overview

COLOR_DICT = {'Electricity': "#E3B51E", 'District heating':  "#E32E1E", 'Ambient energy':  "#EDE0B4",
       'Solar thermal':  "#E37D1E", 'Biomass': "#095C0D", 'Bio-CH4':  "#4EB88D", 'Gas':  "#A11EE3", 'Waste':  "#8D6805",
       'Bio-diesel':  "#421EE3", 'Fossil fuels':  "#121212", 'Syn-kerosene':  "#1ECFE3", 'Kerosene':  "#3465A1",
       'Hydrogen':  "#9B4D57", 'Buildings':  "#E31E1E", 'Transport':  "#359D2C", 'Industry': "#3C1EE3"}


def energy_balance_year(df, color_dict, save_path="figure3.png", dpi=300):

    # Split data by Type
    df_sector = df[df["Type"] == "Sector"]
    df_carrier = df[df["Type"] == "Carrier"]

    # -----------------------------------------
    # Compute max stacked value for y-limit
    # -----------------------------------------
    max_sector = df_sector.groupby("Scenario")["Value"].sum().max()
    max_carrier = df_carrier.groupby("Scenario")["Value"].sum().max()
    ymax = 1.3 * max(max_sector, max_carrier)

    # -----------------------------------------
    # Create figure
    # -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    ax1, ax2 = axes

    ax1.set_facecolor("whitesmoke")
    ax2.set_facecolor("whitesmoke")

    bar_width = 0.5           # thinner bars
    ticklabel_fontsize = 16   # bigger tick labels

    # Apply tick label fontsize
    ax1.tick_params(axis="both", labelsize=ticklabel_fontsize)
    ax2.tick_params(axis="both", labelsize=ticklabel_fontsize)

    # -----------------------------------------
    # 1) SECTOR stacked bars
    # -----------------------------------------
    scenarios_sector = df_sector["Scenario"].unique()
    x_sector = np.arange(len(scenarios_sector))  # numeric positions for bars

    for i, scenario in enumerate(scenarios_sector):
        subset = df_sector[df_sector["Scenario"] == scenario]
        bottom = 0
        for _, row in subset.iterrows():
            ax1.bar(
                x_sector[i],
                row["Value"],
                bottom=bottom,
                color=color_dict.get(row["Carrier"], "gray"),
                label=row["Carrier"],
                width=bar_width,
                edgecolor="black",
                linewidth=1
            )
            bottom += row["Value"]

    ax1.set_title("By sector", fontsize=ticklabel_fontsize)
    ax1.set_ylabel("Energy demand [TWh/a]", fontsize=ticklabel_fontsize+3, labelpad=15)
    ax1.set_ylim(0, ymax)
    ax1.set_xticks(x_sector)
    ax1.set_xticklabels(scenarios_sector, fontsize=ticklabel_fontsize, rotation=0, ha='center')
    ax1.set_xticklabels([s.replace(" ", "\n") for s in scenarios_sector],
                        fontsize=ticklabel_fontsize, rotation=0, ha='center')

    # -----------------------------------------
    # 2) CARRIER stacked bars
    # -----------------------------------------
    scenarios_carrier = df_carrier["Scenario"].unique()
    x_carrier = np.arange(len(scenarios_carrier))

    for i, scenario in enumerate(scenarios_carrier):
        subset = df_carrier[df_carrier["Scenario"] == scenario]
        bottom = 0
        for _, row in subset.iterrows():
            ax2.bar(
                x_carrier[i],
                row["Value"],
                bottom=bottom,
                color=color_dict.get(row["Carrier"], "gray"),
                label=row["Carrier"],
                width=bar_width,
                edgecolor="black",
                linewidth=1
            )
            bottom += row["Value"]

    ax2.set_title("By carrier", fontsize=ticklabel_fontsize)
    ax2.set_ylim(0, ymax)
    ax2.set_xticks(x_carrier)
    ax2.set_xticklabels([s.replace(" ", "\n") for s in scenarios_carrier],
                        fontsize=ticklabel_fontsize, rotation=0, ha='center')
    # -----------------------------------------
    # Remove axes and ticks
    # -----------------------------------------
    for ax in [ax1, ax2]:
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.set_xticks(range(len(ax.get_xticks())))  # preserve x positions
        ax.tick_params(axis="x", labelsize=ticklabel_fontsize)
        ax.tick_params(axis="y", labelsize=ticklabel_fontsize)

        # Add horizontal grid
        ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

    # -----------------------------------------
    # Add reference horizontal lines and arrows
    # -----------------------------------------
    # Compute reference totals
    ref_sector_total = df_sector[df_sector["Scenario"] == "REF 2022"]["Value"].sum()
    ref_carrier_total = df_carrier[df_carrier["Scenario"] == "REF 2022"]["Value"].sum()

    # Sector subplot
    for i, scenario in enumerate(scenarios_sector):
        if scenario == "REF 2022":
            continue
        total_val = df_sector[df_sector["Scenario"] == scenario]["Value"].sum()
        ax1.hlines(
            y=ref_sector_total,
            xmin=x_sector[i] - 0.2,
            xmax=x_sector[i] + 0.2,
            color="black",
            linewidth=2
        )
        ax1.annotate(
            '',
            xy=(x_sector[i], total_val),
            xytext=(x_sector[i], ref_sector_total),
            arrowprops=dict(arrowstyle='->', linestyle='--', color='black', linewidth=2, mutation_scale=20)
        )

        pct_change = (total_val - ref_sector_total) / ref_sector_total * 100
        ax1.annotate(
            f"{pct_change:.0f}%",
            xy=(x_sector[i], ref_sector_total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
            # fontweight="bold"
        )

    # Carrier subplot
    for i, scenario in enumerate(scenarios_carrier):
        if scenario == "REF 2022":
            continue
        total_val = df_carrier[df_carrier["Scenario"] == scenario]["Value"].sum()
        ax2.hlines(
            y=ref_carrier_total,
            xmin=x_carrier[i] - 0.2,
            xmax=x_carrier[i] + 0.2,
            color="black",
            linewidth=2
        )
        ax2.annotate(
            '',
            xy=(x_carrier[i], total_val),
            xytext=(x_carrier[i], ref_carrier_total),
            arrowprops=dict(arrowstyle='->', linestyle='--', color='black', linewidth=2, mutation_scale=20)
        )

        pct_change = (total_val - ref_carrier_total) / ref_carrier_total * 100
        ax2.annotate(
            f"{pct_change:.0f}%",
            xy=(x_carrier[i], ref_carrier_total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
            # fontweight="bold"
        )

    # -----------------------------------------
    # Legends (each 1 column)
    # -----------------------------------------
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))

    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))

    fontsize_legend = 14

    # Put both legends on the right side, left-aligned
    legend1 = fig.legend(
        by_label1.values(),
        by_label1.keys(),
        title="Sectors:",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.98),
        ncol=1,
        frameon=False,
        fontsize=fontsize_legend,
        title_fontsize=fontsize_legend
    )

    legend2 = fig.legend(
        by_label2.values(),
        by_label2.keys(),
        title="Carriers:",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.75),
        ncol=1,
        frameon=False,
        fontsize=fontsize_legend,
        title_fontsize=fontsize_legend
    )

    letters = ["a)", "b)"]

    for ax, letter in zip([ax1, ax2], letters):
        ax.annotate(
            letter,
            xy=(0.05, 0.96),               # relative position: (0=left, 1=top)
            xycoords='axes fraction', # use axes fraction coordinates
            fontsize=16,
            # fontweight='bold',
            ha='left',
            va='top'
        )

    # Left-align legend text
    legend1._legend_box.align = "left"
    legend2._legend_box.align = "left"
    legend1.get_title().set_ha("left")
    legend2.get_title().set_ha("left")

    plt.subplots_adjust(wspace=30)

    plt.tight_layout()

    # Save
    fig.savefig(os.path.join(cnf.FIGURE_FILE_PATH, save_path),
                dpi=dpi, bbox_inches="tight")

    plt.show()


def energy_supply(df, color_dict, save_path="figure4.png", dpi=300):

    # Split data by Type
    df_split = df.copy()
    df_split[["scenario", "scenario_link"]] = df_split["scenario"].str.rsplit("-", n=1, expand=True)

    filtered_scenario_link = ["limited", "unlimited"]
    df_split = df_split[df_split["scenario_link"].isin(filtered_scenario_link)]

    carriers = df_split["carrier"].unique()
    scenarios = df_split["scenario"].unique()
    scenarios_link = df_split["scenario_link"].unique()

    df_elec = df_split[df_split["carrier"] == "electricity"]
    df_heat = df_split[df_split["carrier"] == "heat"]

    # -----------------------------------------
    # Compute max stacked value for y-limit
    # -----------------------------------------
    max = df_split.groupby(["scenario", "carrier","scenario_link"])["value"].sum().reset_index()
    max["value"] *= 1.3

    # -----------------------------------------
    # Create figure
    # -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=False)
    ax1, ax2 = axes

    ax1.set_facecolor("whitesmoke")
    ax2.set_facecolor("whitesmoke")

    bar_width = 0.5           # thinner bars
    ticklabel_fontsize = 16   # bigger tick labels

    # Apply tick label fontsize
    ax1.tick_params(axis="both", labelsize=ticklabel_fontsize)
    ax2.tick_params(axis="both", labelsize=ticklabel_fontsize)

    # -----------------------------------------
    # 1) ELEC stacked bars
    # -----------------------------------------
    scenarios_sector = df_elec["scenario"].unique()
    x_sector = np.arange(len(scenarios_sector))  # numeric positions for bars

    for i, scenario in enumerate(scenarios_sector):
        subset = df_elec[df_elec["Scenario"] == scenario]
        bottom = 0
        for _, row in subset.iterrows():
            ax1.bar(
                x_sector[i],
                row["value"],
                bottom=bottom,
                color=color_dict.get(row["Carrier"], "gray"),
                label=row["Carrier"],
                width=bar_width,
                edgecolor="black",
                linewidth=1
            )
            bottom += row["Value"]

    ax1.set_title("By sector", fontsize=ticklabel_fontsize)
    ax1.set_ylabel("Energy demand [TWh/a]", fontsize=ticklabel_fontsize+3, labelpad=15)
    ax1.set_ylim(0, max[max["carrier"]=="electricity"]["value"].max())
    ax1.set_xticks(x_sector)
    ax1.set_xticklabels(scenarios_sector, fontsize=ticklabel_fontsize, rotation=0, ha='center')
    ax1.set_xticklabels([s.replace(" ", "\n") for s in scenarios_sector],
                        fontsize=ticklabel_fontsize, rotation=0, ha='center')

    # -----------------------------------------
    # 2) CARRIER stacked bars
    # -----------------------------------------
    scenarios_carrier = df_carrier["Scenario"].unique()
    x_carrier = np.arange(len(scenarios_carrier))

    for i, scenario in enumerate(scenarios_carrier):
        subset = df_carrier[df_carrier["Scenario"] == scenario]
        bottom = 0
        for _, row in subset.iterrows():
            ax2.bar(
                x_carrier[i],
                row["Value"],
                bottom=bottom,
                color=color_dict.get(row["Carrier"], "gray"),
                label=row["Carrier"],
                width=bar_width,
                edgecolor="black",
                linewidth=1
            )
            bottom += row["Value"]

    ax2.set_title("By carrier", fontsize=ticklabel_fontsize)
    ax2.set_ylim(0, max[max["carrier"]=="heat"]["value"].max())
    ax2.set_xticks(x_carrier)
    ax2.set_xticklabels([s.replace(" ", "\n") for s in scenarios_carrier],
                        fontsize=ticklabel_fontsize, rotation=0, ha='center')
    # -----------------------------------------
    # Remove axes and ticks
    # -----------------------------------------
    for ax in [ax1, ax2]:
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.set_xticks(range(len(ax.get_xticks())))  # preserve x positions
        ax.tick_params(axis="x", labelsize=ticklabel_fontsize)
        ax.tick_params(axis="y", labelsize=ticklabel_fontsize)

        # Add horizontal grid
        ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

    # -----------------------------------------
    # Add reference horizontal lines and arrows
    # -----------------------------------------
    # Compute reference totals
    ref_sector_total = df_sector[df_sector["Scenario"] == "REF 2022"]["Value"].sum()
    ref_carrier_total = df_carrier[df_carrier["Scenario"] == "REF 2022"]["Value"].sum()

    # Sector subplot
    for i, scenario in enumerate(scenarios_sector):
        if scenario == "REF 2022":
            continue
        total_val = df_sector[df_sector["Scenario"] == scenario]["Value"].sum()
        ax1.hlines(
            y=ref_sector_total,
            xmin=x_sector[i] - 0.2,
            xmax=x_sector[i] + 0.2,
            color="black",
            linewidth=2
        )
        ax1.annotate(
            '',
            xy=(x_sector[i], total_val),
            xytext=(x_sector[i], ref_sector_total),
            arrowprops=dict(arrowstyle='->', linestyle='--', color='black', linewidth=2, mutation_scale=20)
        )

        pct_change = (total_val - ref_sector_total) / ref_sector_total * 100
        ax1.annotate(
            f"{pct_change:.0f}%",
            xy=(x_sector[i], ref_sector_total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
            # fontweight="bold"
        )

    # Carrier subplot
    for i, scenario in enumerate(scenarios_carrier):
        if scenario == "REF 2022":
            continue
        total_val = df_carrier[df_carrier["Scenario"] == scenario]["Value"].sum()
        ax2.hlines(
            y=ref_carrier_total,
            xmin=x_carrier[i] - 0.2,
            xmax=x_carrier[i] + 0.2,
            color="black",
            linewidth=2
        )
        ax2.annotate(
            '',
            xy=(x_carrier[i], total_val),
            xytext=(x_carrier[i], ref_carrier_total),
            arrowprops=dict(arrowstyle='->', linestyle='--', color='black', linewidth=2, mutation_scale=20)
        )

        pct_change = (total_val - ref_carrier_total) / ref_carrier_total * 100
        ax2.annotate(
            f"{pct_change:.0f}%",
            xy=(x_carrier[i], ref_carrier_total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
            # fontweight="bold"
        )

    # -----------------------------------------
    # Legends (each 1 column)
    # -----------------------------------------
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))

    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))

    fontsize_legend = 14

    # Put both legends on the right side, left-aligned
    legend1 = fig.legend(
        by_label1.values(),
        by_label1.keys(),
        title="Sectors:",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.98),
        ncol=1,
        frameon=False,
        fontsize=fontsize_legend,
        title_fontsize=fontsize_legend
    )

    legend2 = fig.legend(
        by_label2.values(),
        by_label2.keys(),
        title="Carriers:",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.75),
        ncol=1,
        frameon=False,
        fontsize=fontsize_legend,
        title_fontsize=fontsize_legend
    )

    letters = ["a)", "b)"]

    for ax, letter in zip([ax1, ax2], letters):
        ax.annotate(
            letter,
            xy=(0.05, 0.96),               # relative position: (0=left, 1=top)
            xycoords='axes fraction', # use axes fraction coordinates
            fontsize=16,
            # fontweight='bold',
            ha='left',
            va='top'
        )

    # Left-align legend text
    legend1._legend_box.align = "left"
    legend2._legend_box.align = "left"
    legend1.get_title().set_ha("left")
    legend2.get_title().set_ha("left")

    plt.subplots_adjust(wspace=30)

    plt.tight_layout()

    # Save
    fig.savefig(os.path.join(cnf.FIGURE_FILE_PATH, save_path),
                dpi=dpi, bbox_inches="tight")

    plt.show()



def dispatch_subplots(  # noqa: PLR0912, PLR0913, PLR0915
    loc, carrier, ind_vars, ind_var_idx, fix_vars_1, fix_vars_2, time_subset_dict, resample=None
):
    """Create subplots for hourly energy dispatch."""
    filtered_scenarios = [
        key
        for key in cnf.SCENARIO_NAMES.keys()
        if any(substring in key for substring in ind_vars)
        and all(substring in key for substring in [fix_vars_1, fix_vars_2])
    ]
    # Data
    df_, y_max = get_scen_loc_carrier(filtered_scenarios, carrier, loc, time_subset_dict)
    df_dispatch = deepcopy(df_)

    tech_rename = helper.format_tech_names(df_dispatch["techs"].unique())
    colors = {tech_rename[t]: helper.get_tech_color(t) for t in tech_rename.keys()}
    patterns = {tech_rename[t]: helper.get_tech_pattern(t) for t in tech_rename.keys()}

    df_dispatch["techs"] = df_dispatch["techs"].replace(tech_rename)

    # Set up the subplot grid
    n_rows = len(ind_vars)
    n_cols = len(time_subset_dict.keys())
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex="col", sharey=True
    )
    axes = np.atleast_2d(axes)  # Ensure axes are always 2D
    xy_pos_dict = {}

    # Iterate over scenarios (rows)
    for i, scenario in enumerate(filtered_scenarios):
        scenario_df = df_dispatch[df_dispatch["scenario"].str.contains(scenario)]

        # Iterate over time subsets (columns)
        for j, season in enumerate(time_subset_dict.keys()):
            ax = axes[i, j]

            subset_df = scenario_df[scenario_df["season"] == season]

            # Group and sum the data
            subset_df["timesteps"] = pd.to_datetime(subset_df["timesteps"])
            subset_df = subset_df.set_index("timesteps")

            if resample != "H":
                subset_df_res = (
                    subset_df.groupby(["scenario", "locs", "techs", "carriers", "unit"])
                    .resample(resample)
                    .mean()
                    .reset_index()
                )
                subset_df_res["timesteps"] = pd.to_datetime(subset_df_res["timesteps"])
                subset_df_res = subset_df_res.set_index("timesteps")
            else:
                subset_df_res = subset_df

            unit = cnf.ENERGY_POWER_UNIT_DICT[subset_df_res.unit.iloc[-1]]

            flow_in_data_ = subset_df_res.pivot_table(
                index=subset_df_res.index,  # Use the original index (timesteps)
                columns="techs",  # Make 'techs' as columns
                values="flow_in",  # Use 'flow_in' values
                aggfunc="sum",  # In case of duplicates
                fill_value=0,  # Replace NaN with 0 directly
            )

            flow_out_data_ = subset_df_res.pivot_table(
                index=subset_df_res.index,  # Use the original index (timesteps)
                columns="techs",  # Make 'techs' as columns
                values="flow_out",  # Use 'flow_in' values
                aggfunc="sum",  # In case of duplicates
                fill_value=0,  # Replace NaN with 0 directly
            )

            column_sums_out = flow_out_data_.sum(axis=0)
            flow_out_data_ = flow_out_data_.loc[:, column_sums_out != 0]

            column_sums_in = flow_in_data_.sum(axis=0)
            flow_in_data_ = flow_in_data_.loc[:, column_sums_in != 0]

            demand_only_techs = set(flow_in_data_.columns) - set(flow_out_data_.columns)

            if demand_only_techs:
                demand_sum = flow_in_data_[demand_only_techs].sum(axis=1)

                # Plot inverted demand sum as black line
                ax.plot(
                    demand_sum.index,
                    -demand_sum,  # flip sign to positive
                    color="black",
                    linewidth=1.5,
                    linestyle="-",
                    label="Demand",
                    zorder=5,
                )

            for df_name, df in [
                ("flow_out_data_", flow_out_data_),
                ("flow_in_data_", flow_in_data_),
            ]:
                cols_lower = {c.lower(): c for c in df.columns}  # map lowercase → original
                if "transmission" in cols_lower:
                    trans_col = cols_lower["transmission"]  # original casing
                    new_cols = [c for c in df.columns if c != trans_col] + [trans_col]
                    if df_name == "flow_out_data_":
                        flow_out_data_ = df[new_cols]
                    else:
                        flow_in_data_ = df[new_cols]

            hatches_flow_out = [
                patterns[tech] for tech in flow_out_data_.columns
            ]  # assuming patterns dictionary is set up for techs
            hatches_flow_in = [patterns[tech] for tech in flow_in_data_.columns]

            # Flow out stacked area plot with hatches
            flow_out_stacks = ax.stackplot(
                flow_out_data_.index,
                flow_out_data_.T,
                labels=flow_out_data_.columns,
                colors=[colors[tech] for tech in flow_out_data_.columns],
            )
            for stack, hatch in zip(flow_out_stacks, hatches_flow_out):
                stack.set_hatch(hatch)
                stack.set_edgecolor("black")
                stack.set_linewidth(0.5)

            # Flow in stacked area plot with hatches
            flow_in_stacks = ax.stackplot(
                flow_in_data_.index,
                flow_in_data_.T,
                labels=flow_in_data_.columns,
                colors=[colors[tech] for tech in flow_in_data_.columns],
            )
            for stack, hatch in zip(flow_in_stacks, hatches_flow_in):
                stack.set_hatch(hatch)
                stack.set_edgecolor("black")
                stack.set_linewidth(0.5)

            ax.set_xlim(flow_in_data_.index.min(), flow_in_data_.index.max())
            # Set grid lines to appear behind other plot elements
            ax.set_axisbelow(True)

            # Enable grid on y-axis with dashed light gray lines
            ax.yaxis.grid(True, linestyle="--", color="lightgray", linewidth=0.9)
            ax.xaxis.grid(True, linestyle="--", color="lightgray", linewidth=0.9)

            ax.axhline(y=0, color="black", linewidth=1)

            xy_pos_dict[f"{i}-{j}"] = {}
            xy_pos_dict[f"{i}-{j}"]["y0"] = ax.get_position().y0
            xy_pos_dict[f"{i}-{j}"]["y1"] = ax.get_position().y1
            xy_pos_dict[f"{i}-{j}"]["x0"] = ax.get_position().x0
            xy_pos_dict[f"{i}-{j}"]["x1"] = ax.get_position().x1

            if i == 0:
                ax.set_title(f"{season.capitalize()}", fontsize=20)
                x_pos_max = ax.get_position().x1

            if j == 0:
                scenario_tag = get_scenario_tag(scenario, ind_var_idx)
                parts = scenario_tag.split("\n")
                bold_lines = [r"$\mathbf{" + p.replace(" ", r"\ ") + "}$" for p in parts]
                formatted_label = "\n".join(bold_lines)

                ax.set_ylabel(
                    formatted_label + "\nDispatch [" + unit + "]",
                    fontsize=25,
                    labelpad=10,
                )

            # Formatting x-axis based on full_year or subset
            if season == "full_year":
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            else:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

            ax.tick_params(
                axis="x", colors="white", labelcolor="black", labelsize=15
            )  # Adjust x-tick label font size
            ax.tick_params(axis="y", labelsize=16)  # Adjust x-tick label font size

    # Flow in and Flow out stacked area plots
    supply_handles = [
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="square,pad=0.3",
            edgecolor="black",
            facecolor=colors[tech],
            hatch=patterns[tech],
            label=tech,
        )
        for tech in flow_out_data_.columns
    ]

    demand_handles = [
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="square,pad=0.3",
            edgecolor="black",
            facecolor=colors[tech],
            hatch=patterns[tech],
            label=tech,
        )
        for tech in flow_in_data_.columns
    ]

    # Create a legend handle for the black line
    black_line_handle = Line2D(
        [0], [0], color="black", linewidth=2, linestyle="-", label="Net load"
    )

    # Add to your demand legend
    demand_handles.append(black_line_handle)

    if n_rows >= 3:
        y_label_supply = 0.95
    else:
        y_label_supply = 1.02

    y_label_delta = 0.03 * 3 / n_rows
    x_label_ = 0.98

    # Legend configuration for supply
    legend_supply = fig.legend(
        handles=supply_handles,
        title="Supply:",
        loc="upper left",
        bbox_to_anchor=(x_label_, y_label_supply),
        fontsize=17,
        title_fontsize=17,
        edgecolor="none",
    )

    legend_demand = fig.legend(
        handles=demand_handles,
        title="Demand:",
        loc="upper left",
        bbox_to_anchor=(
            x_label_,
            y_label_supply - y_label_delta * (1 + len(flow_out_data_.columns)),
        ),
        fontsize=17,
        title_fontsize=17,
        edgecolor="none",
    )

    legend_supply._legend_box.align = "left"
    legend_demand._legend_box.align = "left"
    ax.add_artist(legend_supply)
    ax.add_artist(legend_demand)

    fig.text(
        x=xy_pos_dict["0-0"]["x1"]
        + (xy_pos_dict["0-1"]["x0"] - xy_pos_dict["0-0"]["x1"]) / 2
        + 0.035,  # Position relative to the figure's width (slightly to the left)
        y=1.05,  # Center vertically across the figure
        s=f"{helper.get_country_name(loc, False).upper()}",
        fontsize=30,
        ha="center",
        va="center",
        rotation="horizontal",
        # transform=fig.transFigure,  # Ensure the position is relative to the entire figure
    )

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)  # Remove the top spine
        ax.spines["right"].set_visible(False)  # Remove the right spine
        ax.spines["bottom"].set_visible(False)  # Remove the bottom spine (optional)
        ax.spines["left"].set_visible(False)

    plt.subplots_adjust(hspace=0.2)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    fig.savefig(
        f"{cnf.FIGURE_FILE_PATH}/dispatch/{ind_var_idx}_{loc}_{fix_vars_1}_{fix_vars_2}_{carrier}.png",
        bbox_inches="tight",
        dpi=600,
    )

def get_scenario_tag(scenario, ind_var_idx):
    try:
        scenario_tag = cnf.SCENARIO_LABEL_MAPPING[scenario.split("-")[ind_var_idx]]
    except KeyError:
        scenario_tag = scenario.split("-")[ind_var_idx].replace("_", "-").upper()
    return scenario_tag


def get_scen_loc_carrier(filtered_scenarios, carrier, loc, time_subset_dict):
    """."""
    tech_dict = {
        "transmission": ["CHE_AUT", "CHE_DEU", "CHE_FRA", "CHE_ITA"],
        "chp": [
            "chp_biofuel_extraction",
            "chp_biogas",
            "chp_hydrogen",
            "chp_methane_extraction",
            "chp_wte_back_pressure",
            "chp_biofuel_extraction_ccs",
            "chp_biogas_ccs",
            "chp_wte_back_pressure_ccs",
        ],
        "ccgt": ["ccgt", "ccgt_ccs"],
        "wind": ["wind_onshore", "wind_offshore"],
        "hydro": ["hydro_reservoir", "hydro_run_of_river"],
        "pv": ["roof_mounted_pv", "open_field_pv"],
        "demand": [
            "demand_elec",
            "demand_rail",
            "electric_hob",
            "electric_heater",
            "hydrogen_to_liquids",
        ],
        "ev_charging": [
            "bus_charging",
            "heavy_duty_charging",
            "light_duty_charging",
            "motorcycle_charging",
        ],
        "domestic_storage": ["heat_storage_small"],
        "district_storage": ["heat_storage_big"],
        "heat_pump": ["hp"],
        "unidirectional_charging": ["v1g_charging"],
        "bidirectional_charging": ["v2g_charging"],
    }
    techs_to_drop = []

    dispatch_dict = {}
    y_max = {}

    y_max[loc] = {}
    max_list = []
    dispatch_dict[loc] = {}
    df_loc = []

    for scenario in filtered_scenarios:
        print(f"Loading dispatch results: {loc} - {carrier} - {scenario}")
        df_tot = pd.read_hdf(
            f"{cnf.DISPATCH_PATH}/{scenario}/{scenario}-{carrier}-{loc}.h5", key="df"
        )
        df_red = helper.dispatch_df_reduction(df_tot, tech_dict, techs_to_drop)
        df_red[["flow_in", "flow_out"]] *= 1000  # Convertion to GW
        df_red["unit"] = "gwh"
        max_list.append(
            df_red.groupby(["timesteps"])["flow_in", "flow_out"]
            .sum()
            .reset_index()["flow_out"]
            .max()
        )

        for time_key, time_value in time_subset_dict.items():
            if isinstance(time_value, str):
                time_mask = _get_time_range(time_value)
            else:
                time_mask = time_value
            df_timemasked = df_red[
                (pd.to_datetime(df_red["timesteps"]) >= pd.to_datetime(time_mask[0]))
                & (pd.to_datetime(df_red["timesteps"]) <= pd.to_datetime(time_mask[1]))
            ]
            df_timemasked["season"] = time_key
            df_loc.append(df_timemasked)

    dispatch_dict[loc] = pd.concat(df_loc, ignore_index=True)
    y_max[loc] = max(max_list)

    df_dispatch = dispatch_dict[loc]

    threshold = 0.05  # Threshold 50 MW
    invalid_techs = []
    for tech in df_dispatch.techs.unique():
        max_out = df_dispatch[df_dispatch["techs"] == tech]["flow_out"].max()
        max_in = df_dispatch[df_dispatch["techs"] == tech]["flow_in"].min()
        if (max_out < threshold) & (max_in > -threshold):
            invalid_techs.append(tech)

    df_dispatch = df_dispatch[~df_dispatch["techs"].isin(invalid_techs)]

    return df_dispatch, y_max[loc]


def _get_time_range(t_sub, model_year=2018):
    """_summary_.

    Args:
        t_sub (string): _description_
        model_year (int, optional): Year of the model. Defaults to 2018.

    Returns:
        time_range: time mask
    """
    if t_sub == "full_year":
        time_range = [f"{model_year}-01-01", f"{model_year}-12-31"]
    else:
        month = helper.MONTH_DICT[t_sub]
        if month == "01":
            start = f"{model_year}-{month}-25"
            end = f"{model_year}-{month}-29"
        elif month == "02":
            start = f"{model_year}-{month}-15"
            end = f"{model_year}-{month}-19"
        elif month == "03":
            start = f"{model_year}-{month}-15"
            end = f"{model_year}-{month}-19"
        elif month == "06":
            start = f"{model_year}-{month}-14"
            end = f"{model_year}-{month}-18"
        elif month == "07":
            start = f"{model_year}-{month}-12"
            end = f"{model_year}-{month}-16"
        elif month == "08":
            start = f"{model_year}-{month}-16"
            end = f"{model_year}-{month}-20"
        time_range = [start, end]
    return time_range


def dispatch_subplots_country(  # noqa: PLR0912, PLR0913, PLR0915
    loc_list, carrier, fix_vars_1, fix_vars_2, fix_vars_3, time_subset_dict, resample=None
):
    """Create subplots for hourly energy dispatch."""
    filtered_scenarios = [
        key
        for key in cnf.SCENARIO_NAMES.keys()
        if all(substring in key for substring in [fix_vars_1, fix_vars_2, fix_vars_3])
    ]

    # Data
    df_dispatch = dict()
    for loc in loc_list:
        df_, y_max_country = get_scen_loc_carrier(
            filtered_scenarios, carrier, loc, time_subset_dict
        )
        df_dispatch[loc] = df_
        y_max = max(y_max_country, y_max_country)

    df_dispatch = pd.concat(df_dispatch.values(), ignore_index=True)

    tech_rename = helper.format_tech_names(df_dispatch["techs"].unique())
    colors = {tech_rename[t]: helper.get_tech_color(t) for t in tech_rename.keys()}
    patterns = {tech_rename[t]: helper.get_tech_pattern(t) for t in tech_rename.keys()}

    df_dispatch["techs"] = df_dispatch["techs"].replace(tech_rename)
    supply_sequences = []  # list of lists (one per subplot)
    demand_sequences = []  # list of lists
    # Set up the subplot grid
    n_rows = len(loc_list)
    n_cols = len(time_subset_dict.keys())
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 3.5 * n_rows),
        sharex="col",
        sharey=False,
    )
    axes = np.atleast_2d(axes)  # Ensure axes are always 2D
    xy_pos_dict = {}

    # Iterate over scenarios (rows)
    for i, loc in enumerate(loc_list):
        scenario_df = df_dispatch[df_dispatch["locs"].str.contains(loc)]

        # Iterate over time subsets (columns)
        for j, season in enumerate(time_subset_dict.keys()):
            ax = axes[i, j]

            subset_df = scenario_df[scenario_df["season"] == season]

            # Group and sum the data
            subset_df["timesteps"] = pd.to_datetime(subset_df["timesteps"])
            subset_df = subset_df.set_index("timesteps")

            if resample != "H":
                subset_df_res = (
                    subset_df.groupby(["scenario", "locs", "techs", "carriers", "unit"])
                    .resample(resample)
                    .mean()
                    .reset_index()
                )
                subset_df_res["timesteps"] = pd.to_datetime(subset_df_res["timesteps"])
                subset_df_res = subset_df_res.set_index("timesteps")
            else:
                subset_df_res = subset_df

            unit = cnf.ENERGY_POWER_UNIT_DICT[subset_df_res.unit.iloc[-1]]

            flow_in_data_ = subset_df_res.pivot_table(
                index=subset_df_res.index,  # Use the original index (timesteps)
                columns="techs",  # Make 'techs' as columns
                values="flow_in",  # Use 'flow_in' values
                aggfunc="sum",  # In case of duplicates
                fill_value=0,  # Replace NaN with 0 directly
            )

            flow_out_data_ = subset_df_res.pivot_table(
                index=subset_df_res.index,  # Use the original index (timesteps)
                columns="techs",  # Make 'techs' as columns
                values="flow_out",  # Use 'flow_in' values
                aggfunc="sum",  # In case of duplicates
                fill_value=0,  # Replace NaN with 0 directly
            )

            column_sums_out = flow_out_data_.sum(axis=0)
            flow_out_data_ = flow_out_data_.loc[:, column_sums_out != 0]

            column_sums_in = flow_in_data_.sum(axis=0)
            flow_in_data_ = flow_in_data_.loc[:, column_sums_in != 0]

            demand_only_techs = set(flow_in_data_.columns) - set(flow_out_data_.columns)

            if demand_only_techs:
                demand_sum = flow_in_data_[demand_only_techs].sum(axis=1)

                # Plot inverted demand sum as black line
                ax.plot(
                    demand_sum.index,
                    -demand_sum,  # flip sign to positive
                    color="black",
                    linewidth=1.5,
                    linestyle="-",
                    label="Demand",
                    zorder=5,
                )

            # if "transmission" in flow_out_data_.columns:
            #     flow_out_data_ = flow_out_data_[
            #         [col for col in flow_out_data_.columns if col != "transmission"]
            #         + ["transmission"]
            #     ]
            # if "transmission" in flow_in_data_.columns:
            #     flow_in_data_ = flow_in_data_[
            #         [col for col in flow_in_data_.columns if col != "transmission"]
            #         + ["transmission"]
            #     ]

            for df_name, df in [
                ("flow_out_data_", flow_out_data_),
                ("flow_in_data_", flow_in_data_),
            ]:
                cols_lower = {c.lower(): c for c in df.columns}  # map lowercase → original
                if "transmission" in cols_lower:
                    trans_col = cols_lower["transmission"]  # original casing
                    new_cols = [c for c in df.columns if c != trans_col] + [trans_col]
                    if df_name == "flow_out_data_":
                        flow_out_data_ = df[new_cols]
                    else:
                        flow_in_data_ = df[new_cols]

            hatches_flow_out = [
                patterns[tech] for tech in flow_out_data_.columns
            ]  # assuming patterns dictionary is set up for techs
            hatches_flow_in = [patterns[tech] for tech in flow_in_data_.columns]

            # Flow out stacked area plot with hatches
            flow_out_stacks = ax.stackplot(
                flow_out_data_.index,
                flow_out_data_.T,
                labels=flow_out_data_.columns,
                colors=[colors[tech] for tech in flow_out_data_.columns],
            )
            for stack, hatch in zip(flow_out_stacks, hatches_flow_out):
                stack.set_hatch(hatch)
                stack.set_edgecolor("white")
                stack.set_linewidth(0.5)

            # Flow in stacked area plot with hatches
            flow_in_stacks = ax.stackplot(
                flow_in_data_.index,
                flow_in_data_.T,
                labels=flow_in_data_.columns,
                colors=[colors[tech] for tech in flow_in_data_.columns],
            )
            for stack, hatch in zip(flow_in_stacks, hatches_flow_in):
                stack.set_hatch(hatch)
                stack.set_edgecolor("white")
                stack.set_linewidth(0.5)

            # NEW: record the order used in this subplot (for merging later)
            supply_sequences.append(list(flow_out_data_.columns))
            demand_sequences.append(list(flow_in_data_.columns))

            ax.set_xlim(flow_in_data_.index.min(), flow_in_data_.index.max())
            # Set grid lines to appear behind other plot elements
            ax.set_axisbelow(True)

            # Enable grid on y-axis with dashed light gray lines
            ax.yaxis.grid(True, linestyle="--", color="lightgray", linewidth=0.9)
            ax.xaxis.grid(True, linestyle="--", color="lightgray", linewidth=0.9)

            ax.axhline(y=0, color="black", linewidth=1)

            xy_pos_dict[f"{i}-{j}"] = {}
            xy_pos_dict[f"{i}-{j}"]["y0"] = ax.get_position().y0
            xy_pos_dict[f"{i}-{j}"]["y1"] = ax.get_position().y1
            xy_pos_dict[f"{i}-{j}"]["x0"] = ax.get_position().x0
            xy_pos_dict[f"{i}-{j}"]["x1"] = ax.get_position().x1

            if i == 0:
                ax.set_title(f"{season.capitalize()}", fontsize=20)
                x_pos_max = ax.get_position().x1

            if j == 0:
                scenario_tag = helper.get_country_name(loc)
                ax.set_ylabel(
                    r"$\mathbf{{" + scenario_tag + "}}$\nEnergy flows\n[" + unit + "]",
                    fontsize=25,
                    labelpad=10,
                )  # Demand(-) / Supply(+)
            # else:
            #     ax.tick_params(axis="y", which="both", color="white")
            ax.tick_params(axis="y", which="both", color="gray")

            # Formatting x-axis based on full_year or subset
            if season == "full_year":
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            else:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

            ax.tick_params(
                axis="x", colors="white", labelcolor="black", labelsize=15, pad=10
            )  # Adjust x-tick label font size
            ax.tick_params(axis="y", labelsize=16, pad=3)  # Adjust x-tick label font size

    fig.align_ylabels(axes[:, 0])

    # NEW: merge the orders across all subplots
    MIN_OCCURRENCES_FOR_LEGEND = 1  # set to 2 to drop items that appear only once
    supply_order = merge_legend_orders(
        supply_sequences, min_occurrences=MIN_OCCURRENCES_FOR_LEGEND
    )
    demand_order = merge_legend_orders(
        demand_sequences, min_occurrences=MIN_OCCURRENCES_FOR_LEGEND
    )

    # Build legends in the same order
    supply_handles = [
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="square,pad=0.3",
            edgecolor="white",
            facecolor=colors[tech],
            hatch=patterns[tech],
            label=tech,
        )
        for tech in supply_order
    ]

    demand_handles = [
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="square,pad=0.3",
            edgecolor="white",
            facecolor=colors[tech],
            hatch=patterns[tech],
            label=tech,
        )
        for tech in demand_order
    ]

    # Create a legend handle for the black line
    black_line_handle = Line2D(
        [0], [0], color="black", linewidth=2, linestyle="-", label="Net load"
    )

    # Add to your demand legend
    demand_handles.append(black_line_handle)

    if n_rows >= 3:
        y_label_supply = 0.94
        fontsize_legend = 19
        y_label_delta = 0.06 * 2 / n_rows
    else:
        y_label_supply = 0.98
        fontsize_legend = 14
        y_label_delta = 0.045 * 2 / n_rows

    # y_label_delta = 0.04 * 3 / n_rows
    x_label_ = 0.98

    # Legend configuration for supply
    legend_supply = fig.legend(
        handles=supply_handles,
        title="Supply:",
        loc="upper left",
        bbox_to_anchor=(x_label_, y_label_supply),
        fontsize=fontsize_legend,
        title_fontsize=fontsize_legend,
        edgecolor="none",
    )

    legend_demand = fig.legend(
        handles=demand_handles,
        title="Demand:",
        loc="upper left",
        bbox_to_anchor=(
            x_label_,
            y_label_supply - y_label_delta * (1 + len(supply_order)),
        ),
        fontsize=fontsize_legend,
        title_fontsize=fontsize_legend,
        edgecolor="none",
    )

    legend_supply._legend_box.align = "left"
    legend_demand._legend_box.align = "left"
    ax.add_artist(legend_supply)
    ax.add_artist(legend_demand)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)  # Remove the top spine
        ax.spines["right"].set_visible(False)  # Remove the right spine
        ax.spines["bottom"].set_visible(False)  # Remove the bottom spine (optional)
        ax.spines["left"].set_visible(False)

    plt.subplots_adjust(hspace=0.35)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    fig.savefig(
        f"{cnf.FIGURE_FILE_PATH}/dispatch/paper_{fix_vars_1}_{fix_vars_2}_{fix_vars_3}_{carrier}.png",
        bbox_inches="tight",
        dpi=600,
    )


def merge_legend_orders(sequences, min_occurrences=1):
    """
    Merge multiple ordered sequences into a single global order that
    preserves pairwise ordering seen locally (stable topo-sort).
    - sequences: list[list[str]]
    - min_occurrences: keep items that appear in >= this many sequences
    Returns a list[str] with the merged order.
    """
    # Count in how many sequences each item appears (unique per sequence)
    counts = Counter()
    first_pos = {}  # label -> (seq_idx, pos_in_seq) for tie-breaking
    for s_idx, seq in enumerate(sequences):
        seen = set()
        for p_idx, label in enumerate(seq):
            if label not in first_pos:
                first_pos[label] = (s_idx, p_idx)
            if label not in seen:
                counts[label] += 1
                seen.add(label)

    # Filter items by min_occurrences
    keep = {lab for lab, c in counts.items() if c >= min_occurrences}
    if not keep:
        return []

    # Build precedence graph (u -> v if u appears before v in any sequence)
    graph = defaultdict(set)
    indeg = {lab: 0 for lab in keep}
    for seq in sequences:
        # remove items not kept, keep the order of the rest
        filtered = [x for x in seq if x in keep]
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                u, v = filtered[i], filtered[j]
                if v not in graph[u]:
                    graph[u].add(v)
                    indeg[v] = indeg.get(v, 0) + 1
                    indeg.setdefault(u, indeg.get(u, 0))

    # Kahn’s algorithm with a heap; ties broken by earliest first_pos
    heap = []
    for n, d in indeg.items():
        if d == 0:
            heapq.heappush(heap, (first_pos.get(n, (10**9, 10**9)), n))

    order = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in sorted(graph.get(u, []), key=lambda x: first_pos.get(x, (10**9, 10**9))):
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (first_pos.get(v, (10**9, 10**9)), v))

    # If cycles/leftovers remain, append by first_pos order to stay deterministic
    if len(order) < len(keep):
        leftover = [n for n in keep if n not in order]
        leftover.sort(key=lambda x: first_pos.get(x, (10**9, 10**9)))
        order.extend(leftover)
    return order



def dispatch_check(  # noqa: PLR0912, PLR0913, PLR0915
    loc, carrier, fix_vars_1, fix_vars_2, fix_vars_3, time_subset_dict, resample=None
):
    """Create two subplots (Winter & Summer) for hourly energy dispatch for a single country."""
    loc_list = [loc]  # ensure list-like behavior

    filtered_scenarios = [
        key
        for key in cnf.SCENARIO_NAMES.keys()
        if {fix_vars_1, fix_vars_2, fix_vars_3}.issubset(set(key.split("-")))
    ]
    width_per_day = 1  # inches per day (tweak as you like)

    total_days_dict = {}
    for season, (start_str, end_str) in time_subset_dict.items():
        start = pd.to_datetime(f"{start_str}")  # dummy year for date math
        end = pd.to_datetime(f"{end_str}")
        total_days_dict[season] = (end - start).days + 1

    max_days = max(total_days_dict.values())  # choose the longest period for sizing
    n_day_ticks = math.ceil(max_days / 12)

    fig_width = max_days * width_per_day

    # Data
    df_dispatch = dict()
    for loc in loc_list:
        df_, y_max_country = get_scen_loc_carrier(
            filtered_scenarios, carrier, loc, time_subset_dict
        )
        df_dispatch[loc] = df_
        y_max = max(y_max_country, y_max_country)

    df_dispatch = pd.concat(df_dispatch.values(), ignore_index=True)

    tech_rename = helper.format_tech_names(df_dispatch["techs"].unique())
    colors = {tech_rename[t]: helper.get_tech_color(t) for t in tech_rename.keys()}
    patterns = {tech_rename[t]: helper.get_tech_pattern(t) for t in tech_rename.keys()}

    df_dispatch["techs"] = df_dispatch["techs"].replace(tech_rename)
    supply_sequences = []
    demand_sequences = []

    # --- CHANGED HERE: fix to 2 rows (Winter & Summer), 1 column ---
    n_rows = 2
    n_cols = 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, 3.8 * n_rows),
        sharex=False,
        sharey=True,
    )
    axes = np.atleast_2d(axes)  # always 2D
    axes = axes.reshape(2, 1)  # ensure shape (2,1)
    xy_pos_dict = {}

    # --- Iterate over 2 seasons explicitly ---
    seasons = list(time_subset_dict.keys())[:2]  # expect "winter", "summer"

    for i, season in enumerate(seasons):
        ax = axes[i, 0]
        scenario_df = df_dispatch[df_dispatch["locs"].str.contains(loc)]
        subset_df = scenario_df[scenario_df["season"] == season]

        subset_df["timesteps"] = pd.to_datetime(subset_df["timesteps"])
        subset_df = subset_df.set_index("timesteps")

        if resample != "H":
            subset_df_res = (
                subset_df.groupby(["scenario", "locs", "techs", "carriers", "unit"])
                .resample(resample)
                .mean()
                .reset_index()
            )
            subset_df_res["timesteps"] = pd.to_datetime(subset_df_res["timesteps"])
            subset_df_res = subset_df_res.set_index("timesteps")
        else:
            subset_df_res = subset_df

        unit = cnf.ENERGY_POWER_UNIT_DICT[subset_df_res.unit.iloc[-1]]

        flow_in_data_ = subset_df_res.pivot_table(
            index=subset_df_res.index,
            columns="techs",
            values="flow_in",
            aggfunc="sum",
            fill_value=0,
        )

        flow_out_data_ = subset_df_res.pivot_table(
            index=subset_df_res.index,
            columns="techs",
            values="flow_out",
            aggfunc="sum",
            fill_value=0,
        )

        column_sums_out = flow_out_data_.sum(axis=0)
        flow_out_data_ = flow_out_data_.loc[:, column_sums_out != 0]

        column_sums_in = flow_in_data_.sum(axis=0)
        flow_in_data_ = flow_in_data_.loc[:, column_sums_in != 0]

        demand_only_techs = set(flow_in_data_.columns) - set(flow_out_data_.columns)

        if demand_only_techs:
            demand_sum = flow_in_data_[demand_only_techs].sum(axis=1)
            ax.plot(
                demand_sum.index,
                -demand_sum,
                color="black",
                linewidth=1.5,
                linestyle="-",
                label="Demand",
                zorder=5,
            )

        for df_name, df in [
            ("flow_out_data_", flow_out_data_),
            ("flow_in_data_", flow_in_data_),
        ]:
            cols_lower = {c.lower(): c for c in df.columns}
            if "transmission" in cols_lower:
                trans_col = cols_lower["transmission"]
                new_cols = [c for c in df.columns if c != trans_col] + [trans_col]
                if df_name == "flow_out_data_":
                    flow_out_data_ = df[new_cols]
                else:
                    flow_in_data_ = df[new_cols]

        hatches_flow_out = [patterns[tech] for tech in flow_out_data_.columns]
        hatches_flow_in = [patterns[tech] for tech in flow_in_data_.columns]

        flow_out_stacks = ax.stackplot(
            flow_out_data_.index,
            flow_out_data_.T,
            labels=flow_out_data_.columns,
            colors=[colors[tech] for tech in flow_out_data_.columns],
        )
        for stack, hatch in zip(flow_out_stacks, hatches_flow_out):
            stack.set_hatch(hatch)
            stack.set_edgecolor("white")
            stack.set_linewidth(0.5)

        flow_in_stacks = ax.stackplot(
            flow_in_data_.index,
            flow_in_data_.T,
            labels=flow_in_data_.columns,
            colors=[colors[tech] for tech in flow_in_data_.columns],
        )
        for stack, hatch in zip(flow_in_stacks, hatches_flow_in):
            stack.set_hatch(hatch)
            stack.set_edgecolor("white")
            stack.set_linewidth(0.5)

        supply_sequences.append(list(flow_out_data_.columns))
        demand_sequences.append(list(flow_in_data_.columns))

        ax.set_xlim(flow_in_data_.index.min(), flow_in_data_.index.max())
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", color="lightgray", linewidth=0.9)
        ax.xaxis.grid(True, linestyle="--", color="lightgray", linewidth=0.9)
        ax.axhline(y=0, color="black", linewidth=1)

        xy_pos_dict[f"{i}-0"] = {}
        xy_pos_dict[f"{i}-0"]["y0"] = ax.get_position().y0
        xy_pos_dict[f"{i}-0"]["y1"] = ax.get_position().y1
        xy_pos_dict[f"{i}-0"]["x0"] = ax.get_position().x0
        xy_pos_dict[f"{i}-0"]["x1"] = ax.get_position().x1

        if i == 0:
            scenario = translate_scenario_key(scenario_df["scenario"].iloc[0]).replace("-"," ")
            ax.set_title(f"{scenario}\n{season.capitalize()}", fontsize=20)
        if i == 1:
            ax.set_title(f"{season.capitalize()}", fontsize=20)

        ax.set_ylabel(
            f"Energy flows\n[{unit}]",
            fontsize=20,
            labelpad=10,
        )
        ax.tick_params(axis="y", which="both", color="gray")

        if season == "full_year":
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        else:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=24 * n_day_ticks))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

        ax.tick_params(axis="x", colors="white", labelcolor="black", labelsize=13, pad=10)
        ax.tick_params(axis="y", labelsize=16, pad=3)

    fig.align_ylabels(axes[:, 0])

    # --- rest identical ---
    MIN_OCCURRENCES_FOR_LEGEND = 1
    supply_order = merge_legend_orders(
        supply_sequences, min_occurrences=MIN_OCCURRENCES_FOR_LEGEND
    )
    demand_order = merge_legend_orders(
        demand_sequences, min_occurrences=MIN_OCCURRENCES_FOR_LEGEND
    )

    supply_handles = [
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="square,pad=0.3",
            edgecolor="white",
            facecolor=colors[tech],
            hatch=patterns[tech],
            label=tech,
        )
        for tech in supply_order
    ]

    demand_handles = [
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="square,pad=0.3",
            edgecolor="white",
            facecolor=colors[tech],
            hatch=patterns[tech],
            label=tech,
        )
        for tech in demand_order
    ]

    black_line_handle = Line2D(
        [0], [0], color="black", linewidth=2, linestyle="-", label="Net load"
    )
    demand_handles.append(black_line_handle)

    y_label_supply = 0.90
    fontsize_legend = 14
    y_label_delta = 0.045 * 2 / n_rows

    x_label_ = 1.00

    legend_supply = fig.legend(
        handles=supply_handles,
        title="Supply:",
        loc="upper left",
        bbox_to_anchor=(x_label_, y_label_supply),
        fontsize=fontsize_legend,
        title_fontsize=fontsize_legend,
        edgecolor="none",
    )

    legend_demand = fig.legend(
        handles=demand_handles,
        title="Demand:",
        loc="upper left",
        bbox_to_anchor=(
            x_label_,
            y_label_supply - y_label_delta * (1 + len(supply_order)),
        ),
        fontsize=fontsize_legend,
        title_fontsize=fontsize_legend,
        edgecolor="none",
    )

    legend_supply._legend_box.align = "left"
    legend_demand._legend_box.align = "left"
    ax.add_artist(legend_supply)
    ax.add_artist(legend_demand)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.subplots_adjust(hspace=0.65)
    plt.tight_layout()

    fig.savefig(
        f"{cnf.FIGURE_FILE_PATH}/dispatch/check_{fix_vars_1}_{fix_vars_2}_{fix_vars_3}_{carrier}_{loc}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

def translate_scenario_key(key: str) -> str:
    tokens = key.split("-")
    translated = [
        cnf.SCENARIO_LABEL_MAPPING.get(tok, tok)
        for tok in tokens
    ]
    return "-".join(translated[:-1])  # dump last translated token


def monthly_dispatch(scenario_list, carrier, loc):
    """"Plot the monthly production."""

    month_prod = helper.flow_in_out_sum_1M(
        scenario_list, carrier, loc,
        tech_dict=None, force_rerun=False
    )

    # Summarize by month
    red_month_prod = (
        month_prod.groupby(["scenario", "techs", "locs", "carriers", "unit", 'timesteps'])[
            ["flow_in", "flow_out"]
        ]
        .sum()
        .reset_index()
    )

    threshold = 0.198  # TWh
    red_month_prod = red_month_prod[
        (red_month_prod['flow_in'].abs() + red_month_prod['flow_out'].abs()) >= threshold
    ].copy()

    # Convert timesteps to datetime
    red_month_prod["timesteps"] = pd.to_datetime(red_month_prod["timesteps"])

    # Rename techs + assign color/pattern
    tech_rename = helper.format_tech_names(month_prod["techs"].unique())
    colors = {tech_rename[t]: helper.get_tech_color(t) for t in tech_rename.keys()}
    patterns = {tech_rename[t]: helper.get_tech_pattern(t) for t in tech_rename.keys()}
    red_month_prod["techs"] = red_month_prod["techs"].replace(tech_rename)

    # Max y (for axis scaling)
    max_y = (
        month_prod.groupby(["scenario", "timesteps"])[["flow_in", "flow_out"]]
        .sum()
        .reset_index()["flow_out"]
        .max()
    )

    scenarios = red_month_prod["scenario"].unique()

    # ---- Plot each scenario ----
    for scen in scenarios:

        df_s = red_month_prod[red_month_prod["scenario"] == scen]
        df_s = df_s.copy()
        df_s["timesteps"] = pd.to_datetime(df_s["timesteps"], errors="coerce")
        df_s["timesteps"] = df_s["timesteps"].dt.to_period("M").dt.start_time
        df_s = df_s.sort_values("timesteps")

        fig = plt.figure(figsize=(8, 4))

        # Pivot using the datetime timesteps as index
        pos = (
            df_s.pivot(index="timesteps", columns="techs", values="flow_out")
                .fillna(0)
                .sort_index()
        )

        neg = (
            df_s.pivot(index="timesteps", columns="techs", values="flow_in")
                .fillna(0)
                .sort_index()
        )

        supply_handles, supply_labels = [], []
        demand_handles, demand_labels = [], []

        # ---- Positive (Supply) ----
        bottom = pd.Series(0, index=pos.index)
        for tech in pos.columns:
            values = pos[tech]
            if (values != 0).any():
                h = plt.bar(
                    pos.index,
                    values,
                    bottom=bottom,
                    color=colors.get(tech, None),
                    hatch=patterns.get(tech, None),
                    width=20,        # reasonable width for monthly bars
                    align="center"
                )
                supply_handles.append(h[0])
                supply_labels.append(tech)
            bottom += values

        # ---- Negative (Demand) ----
        bottom = pd.Series(0, index=neg.index)
        for tech in neg.columns:
            values = neg[tech]
            if (values != 0).any():
                h = plt.bar(
                    neg.index,
                    values,
                    bottom=bottom,
                    color=colors.get(tech, None),
                    hatch=patterns.get(tech, None),
                    width=20,
                    align="center"
                )
                demand_handles.append(h[0])
                demand_labels.append(tech)
            bottom += values

        ax = plt.gca()

        ax.set_axisbelow(True)     # <-- critical
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.8)

        # Zero line
        plt.axhline(0, color="black", linewidth=1)

        # Axis formatting
        plt.title(f"{format_scenario_label(scen)}")
        # plt.xlabel("Month")
        plt.ylabel("Electricity supply (+) and demand (-)\n[TWh]")
        plt.ylim(-max_y * 1.1, max_y * 1.1)

        start, end = df_s['timesteps'].min(), df_s['timesteps'].max()
        delta = pd.Timedelta(days=5)  # shrink 5 days from each end
        # plt.xlim(start + delta, end - delta)

        # Format x-axis ticks as month abbreviations
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

        legend_supply = fig.legend(
            handles=supply_handles,
            labels=supply_labels,
            title="Supply (+):",
            loc="upper left",
            bbox_to_anchor=(1.01, 0.94),   # right of axes
            borderaxespad=0.0,
            frameon=False,
        )

        legend_demand = fig.legend(
            handles=demand_handles,
            labels=demand_labels,
            title="Demand (-):",
            loc="upper left",
            bbox_to_anchor=(1.01, 0.51),  # slightly below supply
            borderaxespad=0.0,
            frameon=False,
        )

        legend_supply._legend_box.align = "left"
        legend_demand._legend_box.align = "left"
        ax.add_artist(legend_supply)
        ax.add_artist(legend_demand)

        plt.tight_layout()
        fig.savefig(
            f"{cnf.FIGURE_FILE_PATH}/dispatch/monthly_dispatch_{scen}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.show()


def dual_values_by_scenario(df):
    """
    Plot dual_value in subplots grouped by scenario prefix (before '-').
    Each full scenario is a different color.
    dual_value is sorted from highest to lowest within each scenario.
    """

    df = df.copy()

    # Extract scenario prefix (e.g. '2030')
    df["scenario_prefix"] = df["scenario"].str.split("-").str[0]

    prefixes = df["scenario_prefix"].unique()
    n = len(prefixes)

    fig, axes = plt.subplots(n, 1, figsize=(9, 3.5 * n), sharex=False)

    if n == 1:
        axes = [axes]

    for ax, prefix in zip(axes, prefixes):
        df_prefix = df[df["scenario_prefix"] == prefix]

        scenarios = df_prefix["scenario"].unique()

        for scenario in scenarios:
            subset = df_prefix[df_prefix["scenario"] == scenario]

            # Sort dual_value from high to low
            subset_sorted = subset.sort_values("dual_value", ascending=False)

            ax.plot(
                subset_sorted["dual_value"].values,
                label=scenario,
                linewidth=1.5
            )

        ax.set_title(f"Scenario prefix: {prefix}")
        ax.set_ylabel("dual_value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 250)

    axes[-1].set_xlabel("Sorted index (high → low)")
    plt.tight_layout()

    fig.savefig(
        f"{cnf.FIGURE_FILE_PATH}/dual_value.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.show()


def stacked_system_cost_by_group(
    df,
    groups_tech_dict,
    groups_colour_dict,
    value_col="total_system_cost",
    scenario_col="scenario",
    tech_col="techs",
    xlabel="Total system cost (billion € year$^{-1}$)",
    figsize=(9, 6),
):
    # Reverse mapping: tech -> group
    tech_to_group = {
        tech: group
        for group, techs in groups_tech_dict.items()
        for tech in techs
    }

    df_plot = df.copy()
    df_plot["tech_group"] = (
        df_plot[tech_col]
        .map(tech_to_group)
        .fillna(df_plot[tech_col])
    )

    df_grouped = (
        df_plot
        .groupby([scenario_col, "tech_group"], as_index=False)
        .agg({value_col: "sum"})
    )

    pivot = df_grouped.pivot(
        index=scenario_col,
        columns="tech_group",
        values=value_col
    ).fillna(0)

    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(len(pivot))
    pos_left = np.zeros(len(pivot))
    neg_left = np.zeros(len(pivot))

    for group in pivot.columns:
        values = pivot[group].values
        color = groups_colour_dict.get(group, "grey")

        pos_vals = np.where(values > 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)

        ax.barh(y, pos_vals, left=pos_left, color=color, label=group)
        ax.barh(y, neg_vals, left=neg_left, color=color)

        pos_left += pos_vals
        neg_left += neg_vals

    # --- NEW: total system cost markers ---
    total_cost = pivot.sum(axis=1).values

    ax.scatter(
        total_cost,
        y,
        marker="D",
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Total system cost",
    )

    # Formatting
    formatted_labels = [
        format_scenario_label(s)
        for s in pivot.index
    ]

    ax.set_yticks(y)
    ax.set_yticklabels(formatted_labels)
    ax.set_xlabel(xlabel)
    ax.axvline(0, color="black", linewidth=0.8)
    min_x, max_x = ax.get_xlim()
    ax.set_xlim(min(min_x,-2), max_x)

    handles, labels = ax.get_legend_handles_labels()

    # Move "Total system cost" to the end
    total_label = "Total system cost"
    if total_label in labels:
        idx = labels.index(total_label)
        handles.append(handles.pop(idx))
        labels.append(labels.pop(idx))

    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        ncol=1,
        frameon=False,
        title="Cost group:",
    )

    legend._legend_box.align = "left"
    legend.get_title().set_ha("left")

    plt.tight_layout()
    plt.savefig(
        f"{cnf.FIGURE_FILE_PATH}/stacked_system_cost_by_group.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

def format_scenario_label(
    scenario,
    scenario_label_mapping=cnf.SCENARIO_LABEL_MAPPING,
    sep_in="-",
    sep_out=" | ",
):
    """Format a scenario string into a more readable label."""
    parts = scenario.split(sep_in)

    # Always keep the first part (year)
    labels = [parts[0]]

    # Map remaining parts if possible
    labels += [
        scenario_label_mapping[p]
        for p in parts[1:]
        if p in scenario_label_mapping
    ]

    return sep_out.join(labels)


def stacked_net_import_two_panels(
    df_fuels,
    df_electricity,
    fuel_carriers,
    carrier_base_colour,
    scenario_col="scenario",
    carrier_col="carriers",
    value_col="net_import",
    flow_col="flow",
    elec_value_col="value",
    xlabel="Import (+) / Export (-) (TWh)",
    figsize=(14, 5),
):
    """Plot the fuel and electrcity import/export."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # ==========================================================
    # Left panel: fuels
    # ==========================================================
    df_f = df_fuels[df_fuels[carrier_col].isin(fuel_carriers)]

    pivot_f = df_f.pivot_table(
        index=scenario_col,
        columns=carrier_col,
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )

    y = np.arange(len(pivot_f))
    pos_left = np.zeros(len(pivot_f))
    neg_left = np.zeros(len(pivot_f))

    fuel_handles = []

    for carrier in pivot_f.columns:
        values = pivot_f[carrier].values

        base = carrier.split("_")[0]
        color = carrier_base_colour.get(base, "grey")
        hatch = "//" if "_fossil" in carrier else None

        pos_vals = np.where(values > 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)

        axes[0].barh(
            y, pos_vals, left=pos_left,
            color=color, hatch=hatch, edgecolor="white",
            label=carrier
        )
        axes[0].barh(
            y, neg_vals, left=neg_left,
            color=color, hatch=hatch, edgecolor="white"
        )

        pos_left += pos_vals
        neg_left += neg_vals

        fuel_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, hatch=hatch, edgecolor="white", label = format_carrier_label(carrier)
)
        )

    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel(xlabel)
    axes[0].set_title("Fuels")
    min_x, max_x = axes[0].get_xlim()
    axes[0].set_xlim(min_x*1.15, max_x*1.15)

    # Formatting
    formatted_labels = [
        format_scenario_label(s)
        for s in pivot_f.index
    ]

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(formatted_labels)

    leg_0 = axes[0].legend(
        handles=fuel_handles,
        title="Fuel carrier:",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    leg_0._legend_box.align = "left"
    leg_0.get_title().set_ha("left")

    # ==========================================================
    # Right panel: electricity
    # ==========================================================
    pivot_e = df_electricity.pivot_table(
        index=scenario_col,
        columns=flow_col,
        values=elec_value_col,
        aggfunc="sum",
        fill_value=0,
    )

    pos_left = np.zeros(len(pivot_e))
    neg_left = np.zeros(len(pivot_e))

    elec_handles = []

    for flow, color in zip(["import", "export"], ["#ffea08", "#ab9d00"]):
        if flow not in pivot_e:
            continue

        values = pivot_e[flow].values

        pos_vals = values if flow == "import" else np.zeros_like(values)
        neg_vals = values if flow == "export" else np.zeros_like(values)

        axes[1].barh(y, pos_vals, left=pos_left, color=color, edgecolor="white", label=flow)
        axes[1].barh(y, neg_vals, left=neg_left, color=color, edgecolor="white")

        pos_left += pos_vals
        neg_left += neg_vals

        elec_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white", label=flow.capitalize())
        )

    # --- Net electricity import (diamond) ---
    net_elec = pivot_e.sum(axis=1).values

    axes[1].scatter(
        net_elec,
        y,
        marker="D",
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )

    elec_handles.append(
        Line2D([0], [0], marker="D", color="black",
               markerfacecolor="white", linestyle="None",
               label="Net electricity import")
    )

    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel(xlabel)
    axes[1].set_title("Electricity")
    min_x, max_x = axes[1].get_xlim()
    axes[1].set_xlim(min_x*1.15, max_x*1.15)

    leg_1 = axes[1].legend(
        handles=elec_handles,
        title="Electricity:",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    leg_1._legend_box.align = "left"
    leg_1.get_title().set_ha("left")

    plt.tight_layout()
    plt.savefig(
        f"{cnf.FIGURE_FILE_PATH}/net_import_fuel_elec.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def format_carrier_label(carrier):
    parts = carrier.split("_")

    first = "CO₂" if parts[0] == "co2" else parts[0].capitalize()

    if len(parts) > 1:
        rest = f"({' '.join(parts[1:])})"
        return f"{first} {rest}"
    else:
        return first


def stacked_net_import_by_year(
    df_fuels,
    df_electricity,
    fuel_carriers,
    carrier_base_colour,
    scenario_col="scenario",
    carrier_col="carriers",
    value_col="net_import",
    flow_col="flow",
    elec_value_col="value",
    xlabel="Import (+) / Export (-) (TWh)",
    figsize_per_row=(14, 3.5),
):
    """Plot fuel and electricity imports/exports, split by scenario prefix (e.g., 2030, 2050)."""

    # Extract scenario prefix (first part before '-')
    df_fuels["scenario_prefix"] = df_fuels[scenario_col].str.split("-").str[0]
    df_electricity["scenario_prefix"] = df_electricity[scenario_col].str.split("-").str[0]

    unique_prefixes = sorted(pd.concat([df_fuels["scenario_prefix"], df_electricity["scenario_prefix"]]).unique())
    nrows = len(unique_prefixes)

    fig, axes = plt.subplots(nrows, 2, figsize=(figsize_per_row[0], figsize_per_row[1] * nrows), sharey=False, sharex=True)
    if nrows == 1:
        axes = np.array([axes])  # Ensure axes is 2D array even if 1 row

    for i, prefix in enumerate(unique_prefixes):
        # Filter scenarios for this prefix
        df_f_group = df_fuels[df_fuels["scenario_prefix"] == prefix]
        df_e_group = df_electricity[df_electricity["scenario_prefix"] == prefix]

        # -------------------- Fuels panel --------------------
        pivot_f = df_f_group[df_f_group[carrier_col].isin(fuel_carriers)].pivot_table(
            index=scenario_col, columns=carrier_col, values=value_col, aggfunc="sum", fill_value=0
        )

        y = np.arange(len(pivot_f))
        pos_left = np.zeros(len(pivot_f))
        neg_left = np.zeros(len(pivot_f))
        fuel_handles = []

        for carrier in pivot_f.columns:
            values = pivot_f[carrier].values
            base = carrier.split("_")[0]
            color = carrier_base_colour.get(base, "grey")
            hatch = "//" if "_fossil" in carrier else None

            pos_vals = np.where(values > 0, values, 0)
            neg_vals = np.where(values < 0, values, 0)

            axes[i, 0].barh(y, pos_vals, left=pos_left, color=color, hatch=hatch, edgecolor="white")
            axes[i, 0].barh(y, neg_vals, left=neg_left, color=color, hatch=hatch, edgecolor="white")

            pos_left += pos_vals
            neg_left += neg_vals

            fuel_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, hatch=hatch, edgecolor="white",
                              label=format_carrier_label(carrier))
            )

        axes[i, 0].axvline(0, color="black", linewidth=0.8)
        axes[i, 0].set_xlabel(xlabel)
        axes[i, 0].set_title(f"Fuels ({prefix})")
        axes[i, 0].set_yticks(y)
        axes[i, 0].set_yticklabels([format_scenario_label(s) for s in pivot_f.index])

        leg_0 = axes[i, 0].legend(handles=fuel_handles, title="Fuel carrier:", loc="center left",
                                   bbox_to_anchor=(1.02, 0.5), frameon=False)
        leg_0._legend_box.align = "left"
        leg_0.get_title().set_ha("left")

        # -------------------- Electricity panel --------------------
        pivot_e = df_e_group.pivot_table(
            index=scenario_col, columns=flow_col, values=elec_value_col, aggfunc="sum", fill_value=0
        )

        pos_left = np.zeros(len(pivot_e))
        neg_left = np.zeros(len(pivot_e))
        elec_handles = []

        for flow, color in zip(["import", "export"], ["#ffea08", "#ab9d00"]):
            if flow not in pivot_e:
                continue
            values = pivot_e[flow].values
            pos_vals = values if flow == "import" else np.zeros_like(values)
            neg_vals = values if flow == "export" else np.zeros_like(values)

            axes[i, 1].barh(y, pos_vals, left=pos_left, color=color, edgecolor="white")
            axes[i, 1].barh(y, neg_vals, left=neg_left, color=color, edgecolor="white")

            pos_left += pos_vals
            neg_left += neg_vals

            elec_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white", label=flow.capitalize())
            )

        # Net electricity import
        net_elec = pivot_e.sum(axis=1).values
        axes[i, 1].scatter(net_elec, y, marker="D", facecolors="white", edgecolors="black", linewidths=1.0, zorder=5)
        elec_handles.append(Line2D([0], [0], marker="D", color="black", markerfacecolor="white",
                                   linestyle="None", label="Net electricity import"))

        axes[i, 1].axvline(0, color="black", linewidth=0.8)
        axes[i, 1].set_xlabel(xlabel)
        axes[i, 1].set_title(f"Electricity ({prefix})")
        axes[i, 1].set_yticks(y)
        axes[i, 1].set_yticklabels([format_scenario_label(s) for s in pivot_e.index])

        leg_1 = axes[i, 1].legend(handles=elec_handles, title="Electricity:", loc="center left",
                                   bbox_to_anchor=(1.02, 0.5), frameon=False)
        leg_1._legend_box.align = "left"
        leg_1.get_title().set_ha("left")

    plt.tight_layout()
    plt.savefig(
        f"{cnf.FIGURE_FILE_PATH}/net_import_fuel_elec_by_year.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
