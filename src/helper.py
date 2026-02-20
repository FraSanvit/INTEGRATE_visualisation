"""Helper functions."""

import itertools
import logging
import os
from copy import deepcopy

import config as cnf
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from copy import deepcopy
import pycountry

loc_column_name = "locs"
carrier_column_name = "carriers"

MONTH_DICT = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


def get_results_df(scenario: str, file_name: str) -> pd.DataFrame:
    """Fetch results for a given scenario."""
    file_path = os.path.join(cnf.DATA_PATH, scenario, file_name)
    try:
        df_import = pd.read_csv(file_path)
        if "scenario" not in df_import.columns:
            df_import = duals_harmonization(df_import, scenario)
        return df_import
    except FileNotFoundError:
        print(f"{file_name} not found in {file_path}")
        return None


def duals_harmonization(df_input, scenario):
    """Old duals have missing columns, so this function aims at harmonising the duals."""
    duals = df_input
    dict_uom = {
        "carriers_transport": "EUR2015/km",
        "carriers_co2": "EUR2015/ton",
        "carriers_energy": "EUR2015/MWh",
    }

    dict_multiplier = {"carriers_transport": 10, "carriers_co2": 1000000, "carriers_energy": 10000}

    duals["scenario"] = scenario
    carriers = duals["carrier"].unique()

    carriers_co2 = duals["carrier"][
        duals["carrier"].str.contains("co2", case=False, na=False)
    ].unique()
    carriers_transport = duals["carrier"][
        duals["carrier"].str.contains("transport", case=False, na=False)
        & ~duals["carrier"].str.contains("co2", case=False, na=False)
    ].unique()
    carriers_energy = list(set(carriers) - set(carriers_co2) - set(carriers_transport))

    dict_carriers = {
        "carriers_co2": carriers_co2,
        "carriers_transport": carriers_transport,
        "carriers_energy": carriers_energy,
    }

    for carrier_list_name, carrier_list in dict_carriers.items():
        duals.loc[duals.carrier.isin(carrier_list), "unit"] = dict_uom[carrier_list_name]
        duals["dual-value"] = np.where(
            duals["unit"] == dict_uom[carrier_list_name],
            duals["dual-value"] * dict_multiplier[carrier_list_name],
            duals["dual-value"],
        )

    duals = duals[["scenario", "region", "carrier", "timestep", "unit", "dual-value"]]

    return duals


def parse_location_carrier(df: pd.DataFrame, location, carrier):
    """Parse data based on location and carrier."""
    parsed_df = df.loc[(df["locs"] == location) & (df["carriers"] == carrier)]
    return parsed_df.reset_index(drop=True)


def parse_transmission_location_carrier(df: pd.DataFrame, location, carrier):
    """Parse data based on location and carrier."""
    parsed_df = df.loc[(df["importing_region"] == location) & (df["carriers"] == carrier)]
    return parsed_df.reset_index(drop=True)


def parsing_dispatch(scenario, loc, carrier):
    """Parse the input-output files and the transmission data."""
    # Load the data
    df_in = get_results_df(scenario, "flow_in.csv")
    df_out = get_results_df(scenario, "flow_out.csv")
    df_transmission = get_results_df(scenario, "net_import.csv")

    # # Converting carriers into commodities
    # df_in["carriers"] = df_in["carriers"].replace(cnf.CARRIER_DICT)
    # df_out["carriers"] = df_out["carriers"].replace(cnf.CARRIER_DICT)
    # df_transmission["carriers"] = df_transmission["carriers"].replace(cnf.CARRIER_DICT)

    # Filter the data according to the loc and carrier and postprocessing
    df_in = parse_location_carrier(df_in, loc, carrier)
    df_in["flow_in"] *= -1
    df_out = parse_location_carrier(df_out, loc, carrier)
    df_transmission = parse_transmission_location_carrier(df_transmission, loc, carrier)
    # df_transmission["carriers"] = df_transmission["carriers"].replace("transport", "co2")

    # Converting carriers into commodities
    # df_in["carriers"] = df_in["carriers"].replace(cnf.CARRIER_DICT)
    # df_out["carriers"] = df_out["carriers"].replace(cnf.CARRIER_DICT)
    # df_transmission["carriers"] = df_transmission["carriers"].replace(cnf.CARRIER_DICT)

    df_in_out = pd.concat([df_in, df_out], ignore_index=True).fillna(0)

    return df_in_out, df_transmission


def merge_scenario_output(scenario_list, file_name):
    """Merge capacity results from different scenarios."""
    cap_dict = {}
    scenario_names = [f for f in scenario_list if os.path.isdir(os.path.join(cnf.DATA_PATH, f))]

    for scen in scenario_names:
        if scen in scenario_list:
            cap_dict[scen] = get_results_df(scen, f"{file_name}.csv")

    if cap_dict:  # Check if duals_dict is not empty
        data = pd.concat(cap_dict.values(), ignore_index=True)
    else:
        data = pd.DataFrame()  # Return an empty DataFrame if no valid scenarios are found

    return data


def merge_dispatch(df_in_out, df_transmission):
    """Merge together the input-output file and transmission data and harmonise them."""
    _df_in_out = deepcopy(df_in_out)
    # Process storage techs
    for loc in _df_in_out["locs"].unique():
        for store_tech in cnf.STORAGE_TECHS:
            _net = _df_in_out[(_df_in_out["techs"] == store_tech) & (_df_in_out["locs"] == loc)]
            _net = _net.groupby(
                ["scenario", "techs", "locs", "carriers", "unit", "timesteps"]
            ).sum()
            _net = _net.reset_index()
            _sum = _net["flow_in"] + _net["flow_out"]
            _net["flow_in"] = _sum
            _net["flow_out"] = _sum
            _net["flow_in"] = _net["flow_in"].apply(lambda x: x if x < 0 else 0)
            _net["flow_out"] = _net["flow_out"].apply(lambda x: x if x > 0 else 0)

            _df_in_out = _df_in_out.drop(_df_in_out[_df_in_out["techs"] == store_tech].index)
            _df_in_out = pd.concat([_df_in_out, _net], ignore_index=True).fillna(0)

    # Process import export techs
    if any("distribution" in tech for tech in _df_in_out["techs"].unique()):
        _df_in_out["techs"] = (
            _df_in_out["techs"].str.replace("_import", "").str.replace("_export", "")
        )
        grouped_df = (
            _df_in_out.groupby(["scenario", "techs", "locs", "carriers", "unit", "timesteps"])
            .sum()
            .reset_index()
        )
        grouped_df["balance"] = grouped_df["flow_in"] + grouped_df["flow_out"]
        grouped_df["flow_out"] = grouped_df["balance"].clip(lower=0)
        grouped_df["flow_in"] = grouped_df["balance"].clip(upper=0)
        grouped_df = grouped_df.drop(columns=["balance"])
        grouped_df.loc[
            (grouped_df["flow_in"] != 0) & grouped_df["techs"].str.contains("distribution"), "techs"
        ] += "_export"
        grouped_df.loc[
            (grouped_df["flow_out"] != 0) & grouped_df["techs"].str.contains("distribution"),
            "techs",
        ] += "_import"
        _df_in_out = grouped_df

    # Process transmission data
    _df_transmission = df_transmission
    _df_transmission["techs"] = (
        _df_transmission["importing_region"] + "_" + _df_transmission["exporting_region"]
    )
    _df_transmission = _df_transmission.rename(columns={"importing_region": "locs"})
    _df_transmission["flow_out"] = _df_transmission["0"].apply(lambda x: x if x > 0 else 0)
    _df_transmission["flow_in"] = _df_transmission["0"].apply(lambda x: x if x < 0 else 0)
    _df_transmission = _df_transmission.drop(columns=["exporting_region", "0"])

    # Combine all data
    df_tot = pd.concat([_df_in_out, _df_transmission], ignore_index=True)

    return df_tot


def get_country_name(alpha3_code, newline=True):
    """Return the full country name from an alpha-3 country code."""
    try:
        if alpha3_code == "MKD":
            country = "North Macedonia"
        else:
            if len(alpha3_code) == 3:
                country = pycountry.countries.get(alpha_3=alpha3_code)
            elif len(alpha3_code) == 2:
                country = pycountry.countries.get(alpha_2=alpha3_code)
            else:
                return alpha3_code
            country = country.name if country else alpha3_code
        if newline:
            # Add a newline at the last space in the country name
            if " " in country:
                country = country.rsplit(" ", 1)  # Split from the right at the last space
                country = country[0] + "\n" + country[1]  # Rejoin with \n
        return country
    except KeyError:
        return alpha3_code  # Return the code if not found


def dispatch_df_reduction(df_tot, tech_dict, techs_to_drop):
    """Reduce the size of the dispatch dataframe by aggregating techs."""
    df_reduced = deepcopy(df_tot)
    tech_mapping = {item: key for key, values in tech_dict.items() for item in values}
    df_reduced["techs"] = df_reduced["techs"].replace(tech_mapping)

    # Collapsing all the transmission techs
    df_reduced["techs"] = df_reduced["techs"].apply(lambda x: "transmission" if x.isupper() else x)
    df_reduced = df_reduced[~df_reduced["techs"].isin(techs_to_drop)]

    df_agg = (
        df_reduced.groupby(
            ["scenario", "techs", "locs", "carriers", "unit", "timesteps", "flow_in"]
        )["flow_out"]
        .sum()
        .reset_index()
    )

    return df_agg


def format_tech_names(techs):
    """Format the name of the technologies to show in the plot legend."""
    if isinstance(techs, str):
        techs = [techs]  # Convert single item into a list

    formatted_techs = {}
    remove_list = ["back pressure", "extraction", "underground", "distribution"]
    uppercase_list = ["ev", "chp", "pv", "ccgt", "dac"]

    for tech in techs:
        # If all letters in `tech` are uppercase, set to "Transmission"
        if tech.isupper():
            formatted_techs[tech] = "Transmission"
            continue

        # Remove underscores and specified items
        formatted_tech = tech.replace("_", " ")
        for item in remove_list:
            formatted_tech = formatted_tech.replace(item, "")

        # Map via TECH_DICT if available
        if formatted_tech in cnf.TECH_DICT:
            formatted_tech = cnf.TECH_DICT[formatted_tech]

        # Clean words
        formatted_tech = formatted_tech.replace("industry", "")
        formatted_tech = formatted_tech.replace("transport", "")
        formatted_tech = formatted_tech.replace("dh", "district heating")
        formatted_tech = formatted_tech.replace("  ", " ")
        formatted_tech = formatted_tech.strip()

        # Split into words and format
        words = formatted_tech.split()
        new_words = []
        for i, w in enumerate(words):
            if w.lower() in uppercase_list:
                new_words.append(w.upper())
            elif i == 0:
                new_words.append(w.capitalize())  # only first word capitalized
            else:
                new_words.append(w.lower())  # rest stay lowercase
        formatted_tech = " ".join(new_words)

        formatted_techs[tech] = formatted_tech

    return formatted_techs


def get_tech_pattern(tech):
    """Helper for technology markers."""
    for key in cnf.PATTERN_DICT:
        if key in tech:
            return cnf.PATTERN_DICT[key]
    return ""


def get_tech_color(tech: str, tech_dict=cnf.TECH_COLORS):
    """Helper for technology colors."""

    def string_to_number(tech: str):
        """Associate a string to one unique number (unidirectional)."""
        # Each link matches a unique number
        string = tech.lower()
        number = sum(
            (ord(letter) - ord("a") + 1) * (i + 1)
            for i, letter in enumerate(string)
            if letter.isalpha()
        )
        return number

    if "|" in tech:
        tech = tech.split("|")[0]
    grey_scale = [f"rgb({i}, {i}, {i})" for i in range(256)][:256]

    if all(not char.islower() for char in tech):
        return grey_scale[round(string_to_number(tech) / 2.5)]

    # Known technology → color from config
    elif tech in tech_dict:
        return tech_dict[tech]

    # Unknown technology → fallback gray
    else:
        return "gray"


def df_tech_grouping(df, tech_groups, target_column):
    """The function groups together the df according to the tech_groups."""
    df_ext = deepcopy(df)

    def assign_tech_group_(tech, tech_groups):
        for group, tech_list in tech_groups.items():
            if tech in tech_list:
                return group
        return None  # Tech not found in any group

    # Apply the custom function to create a new column 'tech_group'
    df_ext["tech_group"] = df_ext["techs"].map(lambda tech: assign_tech_group_(tech, tech_groups))

    columns_to_keep = [
        item for item in list(df_ext.columns) if item not in ["techs", target_column]
    ]
    # Group by 'tech_group' and sum the 'value' column
    grouped_df = df_ext.groupby(columns_to_keep)[target_column].sum().reset_index()

    # Rename the 'tech_group' column to 'techs' for consistency
    grouped_df = grouped_df.rename(columns={"tech_group": "techs"})

    return grouped_df


def flow_in_out_sum_1M(scenario_list, carrier, loc, tech_dict=None, force_rerun=True):
    """Calculate the monthly sum of flow_in and flow_out for given scenarios."""
    file_path = os.path.join(cnf.RESULTS_PATH, "flow_in_out", "flow_in_out_1M.csv")

    if os.path.exists(file_path) and not force_rerun:
        in_out_month = pd.read_csv(file_path)

    else:
        if tech_dict is None:
            tech_group = cnf.GROUPS_TECH_DICT_MINIMAL  # GROUPS_TECH_DICT_FLOW

        flow_dict = {}

        for scenario in scenario_list:
            print(f"Loading dispatch results: {scenario}")
            df_tot = pd.read_hdf(
                f"{cnf.DISPATCH_PATH}/{scenario}/{scenario}-{carrier}-{loc}.h5", key="df"
            )
            df_red = dispatch_df_reduction(df_tot, tech_dict=tech_group, techs_to_drop=[])
            df_red["timesteps"] = pd.to_datetime(df_red["timesteps"])

            df_monthly = (
                df_red.groupby(["scenario", "techs", "locs", "carriers", "unit"])
                .resample("M", on="timesteps")[["flow_in", "flow_out"]]
                .sum()
                .reset_index()
            )

            flow_dict[scenario] = df_monthly

        in_out_month = pd.concat(flow_dict.values(), ignore_index=True)
        in_out_month.to_csv(file_path, index=False)
    return in_out_month


def flow_in_out_year(scenario_list, loc, tech_dict=None, force_rerun=False):
    """Calculate the yearly sum of flow_in and flow_out for given scenarios."""
    file_path = os.path.join(cnf.RESULTS_PATH, "flow_in_out", "flow_in_out_year.csv")

    if os.path.exists(file_path) and not force_rerun:
        in_out_year = pd.read_csv(file_path)
    else:

        flow_in_sum = merge_scenario_output(scenario_list, "flow_in_sum")
        flow_out_sum = merge_scenario_output(scenario_list, "flow_out_sum")
        flow_tot = pd.concat([flow_in_sum, flow_out_sum], ignore_index=True).fillna(0)
        flow_tot = flow_tot[
            flow_tot["carriers"] != "electricity"
        ]  # to avoid double counting with the transmission

        in_out_month_el = flow_in_out_sum_1M(
            scenario_list, "electricity", loc, tech_dict, force_rerun
        )

        in_out_year_el = (
            in_out_month_el.groupby(["scenario", "techs", "locs", "carriers", "unit"])[
                ["flow_in", "flow_out"]
            ]
            .sum()
            .reset_index()
        )

        in_out_year_el = in_out_year_el.rename(
            columns={"flow_in": "flow_in_sum", "flow_out": "flow_out_sum"}
        )

        in_out_year = pd.concatenate([flow_tot, in_out_year_el], ignore_index=True)

        in_out_year.to_csv(file_path, index=False)

    return in_out_year


def merge_scenario_duals_carrier_loc(carrier, scenario_list, loc):
    """Merge results from different scenarios."""
    duals_dict = {}

    for scen in scenario_list:
        if scen in cnf.SCENARIO_NAMES.keys():
            if carrier != "co2":
                duals_ = get_results_df(scen, "duals.csv")
            else:
                duals_ = get_results_df(scen, "duals.csv")
            duals_dict[scen] = duals_[(duals_["carriers"] == carrier) & (duals_["locs"] == loc)]

    if duals_dict:  # Check if duals_dict is not empty
        data = pd.concat(duals_dict.values(), ignore_index=True)
    else:
        data = pd.DataFrame()  # Return an empty DataFrame if no valid scenarios are found

    return data


def merge_scenario_duals_carrier(carrier, scenario_list):
    """Merge results from different scenarios."""
    duals_dict = {}

    for scen in scenario_list:
        if scen in cnf.SCENARIO_NAMES.keys():
            if carrier != "co2":
                duals_ = get_results_df(scen, "duals.csv")
            else:
                duals_ = get_results_df(scen, "duals.csv")
            duals_dict[scen] = duals_[(duals_["carriers"] == carrier)]

    if duals_dict:  # Check if duals_dict is not empty
        data = pd.concat(duals_dict.values(), ignore_index=True)
    else:
        data = pd.DataFrame()  # Return an empty DataFrame if no valid scenarios are found

    return data


def cost_per_loctech(scenario_list):
    cost_dict = {}
    scenario_names = [f for f in scenario_list if os.path.isdir(os.path.join(cnf.DATA_PATH, f))]

    for scen in scenario_names:
        transmission_cost = get_results_df(scen, "total_transmission_costs.csv")
        transmission_cost = transmission_cost.rename(
            columns={"0": "total_system_cost", "exporting_region": "locs"}
        ).drop("importing_region", axis=1)
        transmission_cost["total_system_cost"] = transmission_cost["total_system_cost"].div(1)

        system_cost = get_results_df(scen, "total_system_cost.csv")

        scen_cost = pd.concat([system_cost, transmission_cost], ignore_index=True)
        cost_dict[scen] = scen_cost

    data = pd.concat(cost_dict.values(), ignore_index=True)

    return data


def yearly_data(loc, carrier, resample, minimum: float = 0.001):
    """Create the dataframe for the yearly plot."""
    df_scens = {}
    for scenario in cnf.SCENARIO_NAMES.keys():
        df_in_out, df_transmission = parsing_dispatch(scenario, loc, carrier)
        df_scens[scenario] = merge_dispatch(df_in_out, df_transmission)

    data = pd.concat(df_scens.values(), ignore_index=True)

    # Converting index into datetime and resampling
    data["timesteps"] = pd.to_datetime(data["timesteps"])
    data = data.set_index("timesteps")
    if resample:
        data = (
            data.groupby(["scenario", "locs", "techs", "carriers", "unit"])
            .resample(resample)["flow_in", "flow_out"]
            .sum()
        )
    data = data.reset_index()
    data = pd.melt(
        data,
        id_vars=data.columns.difference(["flow_in", "flow_out"]),
        value_vars=["flow_in", "flow_out"],
        var_name="flow",
        value_name="value",
    )

    data = data[abs(data["value"]) >= minimum]
    return data


def flow_in_out_sum(scenario_list, force_rerun=False):
    """Merging together demand and supply."""
    file_path = os.path.join(cnf.RESULTS_PATH, "flow_in_out", "flow_in_out.csv")

    if os.path.exists(file_path) and not force_rerun:
        # Load the existing file
        merged_df = pd.read_csv(file_path)
        file_scenario_list = merged_df["scenario"].unique()

        # Identify missing scenarios
        missing_scenarios = set(scenario_list) - set(file_scenario_list)

        if not missing_scenarios:
            print("✅ All scenarios already exist in file. Returning cached results.")
            return merged_df
        else:
            print(f"⚠️ Missing scenarios detected: {missing_scenarios}. Recomputing only those...")
    else:
        # If no file or force_rerun=True → start fresh
        merged_df = pd.DataFrame()
        missing_scenarios = scenario_list

    # --- Compute only missing scenarios ---
    flow_dict = {}
    scenario_names = [f for f in missing_scenarios if os.path.isdir(os.path.join(cnf.DATA_PATH, f))]

    for scen in scenario_names:
        print(f"Merging flows of scenario: {scen}")
        # FLOW in and out
        df_in = (
            get_results_df(scen, "flow_in.csv")
            # .groupby(["scenario", "techs", "locs", "carriers", "unit"])["flow_in"]
            # .sum()
            # .reset_index()
        )
        df_in["flow_in"] *= -1

        df_out = (
            get_results_df(scen, "flow_out.csv")
            # .groupby(["scenario", "techs", "locs", "carriers", "unit"])["flow_out"]
            # .sum()
            # .reset_index()
        )

        df_transmission = get_results_df(scen, "net_import.csv")
        df_transmission = df_transmission.rename(columns={"importing_region": "locs"}).drop(
            "exporting_region", axis=1
        )
        df_transmission["flow_in"] = df_transmission["0"].apply(lambda x: x if x < 0 else 0)
        df_transmission["flow_out"] = df_transmission["0"].apply(lambda x: x if x > 0 else 0)
        df_transmission = df_transmission.drop(columns=["0"])
        df_transmission["techs"] = "transmission"

        df_in_out = pd.concat([df_in, df_out, df_transmission], ignore_index=True).fillna(0)
        df_in_out["net"] = df_in_out["flow_in"] + df_in_out["flow_out"]
        df_in_out["flow_in"] = df_in_out["net"].apply(lambda x: x if x < 0 else 0)
        df_in_out["flow_out"] = df_in_out["net"].apply(lambda x: x if x > 0 else 0)
        df_in_out = df_in_out.drop(columns=["net"])

        df_in_out = (
            df_in_out.groupby(["scenario", "techs", "locs", "carriers", "unit"])[
                ["flow_in", "flow_out"]
            ]
            .sum()
            .reset_index()
        )

        flow_dict[scen] = df_in_out

    # Concatenate old + new
    if not merged_df.empty:
        merged_df = pd.concat(
            [merged_df, pd.concat(flow_dict.values(), ignore_index=True)], ignore_index=True
        )
    else:
        merged_df = pd.concat(flow_dict.values(), ignore_index=True)

    # Save updated results
    merged_df.to_csv(file_path, index=False)

    # Filtering only requested scenarios
    merged_df_filtered = merged_df[merged_df["scenario"].isin(scenario_list)].copy()

    return merged_df_filtered
