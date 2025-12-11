"""Visualization functions for the INTEGRATE paper."""

#%% Import libraries
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

plot.energy_balance_year(
    df,
    plot.COLOR_DICT,
    save_path="figure3_energy.png",
    dpi=300
)

# %% Figure 4

df = pd.read_csv(os.path.join(cnf.INPUT_FILE_PATH, "figure_4_energy_supply.csv"))

df_split = df.copy()
df_split[["scenario", "scenario_link"]] = df_split["scenario"].str.rsplit("-", n=1, expand=True)

#%% Postprocessing calliope dispatch results + plotting

carrier_list = ["electricity"]
loc_list = ["AUT"] # cnf.LOC_LIST

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

#%% Monthly production from RES

scenario_list = list(cnf.SCENARIO_NAMES.keys())
carrier = "elctricity"
loc = "AUT"

plot.monthly_dispatch(scenario_list, carrier, loc)


