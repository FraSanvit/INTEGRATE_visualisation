"""Visualization functions for the INTEGRATE paper."""

# %% Import libraries
import itertools
import logging
import math
import os
from copy import deepcopy
from math import ceil
from statistics import mean

import config as cnf
import plot as plot
import helper as helper
import numpy as np
import pandas as pd
import calliope_postprocessing as calliope

import string

# %% Figure 3: Energy demand overview

df = pd.read_csv(os.path.join(cnf.INPUT_FILE_PATH, "figure3_energy_demand.csv"))

plot.energy_balance_year(df, plot.COLOR_DICT, save_path="figure3_energy.png", dpi=300)

# %% Figure 4

df = pd.read_csv(os.path.join(cnf.INPUT_FILE_PATH, "figure_4_energy_supply.csv"))

df_split = df.copy()
df_split[["scenario", "scenario_link"]] = df_split["scenario"].str.rsplit("-", n=1, expand=True)

# %% Postprocessing calliope dispatch results + plotting

carrier_list = ["electricity"]
loc_list = ["AUT"]  # cnf.LOC_LIST

force_rerun = False
scenario_list = cnf.SCENARIO_NAMES.keys()

calliope.postprocess_dispatch(carrier_list, loc_list, scenario_list, force_rerun)

for scen in ["int", "ref"]:
    for year in ["2030", "2050"]:
        for link in ["island", "noisland", "superisland"]:
            plot.dispatch_check(  # noqa: PLR0912, PLR0913, PLR0915
                "AUT",
                "electricity",
                link,
                scen,
                year,
                {"winter": ["2018-01-10", "2018-01-20"], "summer": ["2018-08-10", "2018-08-20"]},
                resample="H",
            )

# %% Monthly production from RES

scenario_list = list(cnf.SCENARIO_NAMES.keys())
carrier = "electricity"
loc = "AUT"

plot.monthly_dispatch(scenario_list, carrier, loc)

# %% Load duration curve
scenario_list = [
    "2030-int-island-1h",
    "2030-int-noisland-1h",
    "2030-ref-island-1h",
    "2030-ref-noisland-1h",
    "2050-int-island-1h",
    "2050-int-noisland-1h",
    "2050-ref-island-1h",
    "2050-ref-noisland-1h",
]

carrier = "electricity"
loc = "AUT"

dual_elec = helper.merge_scenario_duals_carrier_loc(carrier, scenario_list, loc)
plot.dual_values_by_scenario(dual_elec)

# %% AUT energy system costs
scenario_list = list(cnf.SCENARIO_NAMES.keys())

carriers = ["hydrogen", "diesel", "kerosene", "methanol", "methane", "electricity", "co2"]
country_list = ["AUT"]

trade_path = os.path.join(cnf.RESULTS_PATH, "costs", "trade_revenues.csv")

if os.path.exists(trade_path):
    trade = pd.read_csv(trade_path)
else:
    trade = calliope.trade_processing(scenario_list, carriers, country_list)


df_cost = helper.cost_per_loctech(scenario_list)

trade = trade.rename(columns={"carriers":"techs","trade_cost": "total_system_cost"})
trade["techs"] = trade["techs"].astype(str) + "_trade"

# System benefits (keeping 'locs')
df_cost_aut = (
    df_cost[df_cost["locs"]==country_list[0]]
    .groupby(["scenario", "locs", "techs", "unit"])["total_system_cost"]
    .sum()
    .reset_index()
)

df_cost_aut_tot = pd.concat([df_cost_aut, trade], ignore_index=True)

# Remove the co2 trade from 2030 scenarios (as there is no cost)
mask = (
    df_cost_aut_tot["scenario"].str.contains("2030")
    & (df_cost_aut_tot["techs"] == "co2_trade")
)

df_cost_aut_tot = df_cost_aut_tot[~mask]

plot.stacked_system_cost_by_group(
    df=df_cost_aut_tot[~df_cost_aut_tot["techs"].str.contains("storage_co2")],
    groups_tech_dict=cnf.GROUPS_TECH_DICT_COST,
    groups_colour_dict=cnf.GROUPS_TECH_DICT_COST_COLOUR
)

# Import and export totals

output_path_net_import = os.path.join(cnf.RESULTS_PATH, "import-export", "net_import_sum.csv")
output_path_elec_flow = os.path.join(cnf.RESULTS_PATH, "import-export", "import_export_elec_df.csv")


if os.path.exists(output_path_net_import) and os.path.exists(output_path_elec_flow):
    net_import_sum = pd.read_csv(output_path_net_import)
    elec_sum = pd.read_csv(output_path_elec_flow)
else:
    calliope.net_import_processing(scenario_list, carriers, country_list)
    net_import_sum = pd.read_csv(output_path_net_import)
    elec_sum = pd.read_csv(output_path_elec_flow)

fossil_import_list = ["oil", "diesel", "kerosene", "methanol", "methane", "coal"]

flow_out_sum = helper.merge_scenario_output(scenario_list, "flow_out_sum")
flow_fossil_aut = flow_out_sum[
    (flow_out_sum["techs"].str.contains("supply"))
    & (flow_out_sum["locs"]==country_list[0])
    & (flow_out_sum["carriers"].isin(fossil_import_list))
].rename(columns={"flow_out_sum":"net_import"})

flow_fossil_aut["carriers"] = flow_fossil_aut["carriers"].astype(str) + "_fossil"

tot_import = pd.concat([
        net_import_sum,
        flow_fossil_aut[["scenario", "locs", "carriers", "net_import", "unit"]]], ignore_index=True)
tot_import["unit"] = tot_import["unit"].replace({"twh":"TWh"})

# Add supply

fuel_carriers = [
    c for c in tot_import["carriers"].unique()
    if c not in {"co2", "electricity"}
]  # replace with your fuel carriers

electricity_carriers = ["electricity"]                 # replace with your electricity carriers

carrier_colors = {
    "hydrogen": "#1f77b4",
    "diesel": "#ff7f0e",
    "kerosene": "#2ca02c",
    "methanol": "#d62728",
    "electricity": "#9467bd",
    "methane": "#1f68b4",
}

plot.stacked_net_import_two_panels(
    tot_import,
    elec_sum,
    fuel_carriers,
    cnf.CARRIER_COLOUR,
)

plot.stacked_net_import_by_year(
    tot_import,
    elec_sum,
    fuel_carriers,
    cnf.CARRIER_COLOUR,
)
