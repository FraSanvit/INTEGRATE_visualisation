"""Run the postprocessing."""

#%% Import
import logging
import os

import config as cnf
import helper as helper
import pandas as pd
import tables   # PyTables errors come from here


logging.basicConfig(level=logging.INFO)

#%% Postprocess calliope dispatch results

def postprocess_dispatch(carrier_list, loc_list, scenario_list, force_rerun=False):
    """Postprocess Calliope dispatch results for given carriers, locations, and scenarios."""
    def safe_read_hdf(path, key="df"):
        if not os.path.exists(path):
            return None

        try:
            return pd.read_hdf(path, key=key)

        except (OSError, IOError, tables.HDF5ExtError, tables.NoSuchNodeError) as e:
            print(f"[WARNING] Failed to read {path}: {e}")
            print(f"[WARNING] Deleting corrupted file: {path}")
            try:
                os.remove(path)
            except Exception as rm_err:
                print(f"[ERROR] Could not delete file {path}: {rm_err}")
            return None

    for scenario in scenario_list:
        for carrier in carrier_list:
            for loc in loc_list:

                file_path = f"{cnf.DISPATCH_PATH}/{scenario}/{scenario}-{carrier}-{loc}.h5"

                # -------------------------------
                # Step 1: If file exists, test integrity
                # -------------------------------
                if os.path.exists(file_path):
                    df_test = safe_read_hdf(file_path, key="df")

                    if df_test is not None:
                        # File is readable
                        if not force_rerun:
                            print(f"File exists and is OK, skipping: {file_path}")
                            continue
                        else:
                            print(f"Force rerun enabled, rewriting file: {file_path}")
                            # delete to regenerate
                            os.remove(file_path)

                    else:
                        # df_test is None -> safe_read_hdf already deleted corrupted file
                        print(f"Corrupted file removed, regenerating: {file_path}")

                # -------------------------------
                # Step 2: Regenerate file
                # -------------------------------
                print(f"Post-processing dispatch results: {loc} - {carrier} - {scenario}")
                df_in_out, df_transmission = helper.parsing_dispatch(scenario, loc, carrier)
                df_tot = helper.merge_dispatch(df_in_out, df_transmission)

                os.makedirs(f"{cnf.DISPATCH_PATH}/{scenario}", exist_ok=True)

                df_tot.to_hdf(
                    file_path,
                    key="df",
                    mode="w",
                    format="table",
                    complib="zlib",
                    complevel=6,
                )

# %% Compute trades

def trade_processing(scenario_list, carriers, country_list):
    """Compute trade costs/revenues for given scenarios, carriers, and countries."""
    emission_supply = helper.merge_scenario_output(scenario_list, "total_system_emissions")
    trade_dict: dict[str, dict[str, dict[str, float]]] = {}

    for carrier in carriers:  # <-- added loop over carriers
        trade_dict[carrier] = {}

        for scenario in scenario_list:
            trade_dict[carrier][scenario] = {}

            filtered_scenario_list = [s for s in scenario_list if s == scenario]
            if not filtered_scenario_list:
                continue

            print(f"Loading duals: [{carrier}]")
            duals_ = helper.merge_scenario_duals_carrier(carrier, filtered_scenario_list)

            for loc in country_list:
                print(f"{carrier} | {scenario} | {loc}")

                # Read dispatch data
                df_dispatch = pd.read_hdf(
                    f"{cnf.RESULTS_PATH}/dispatch/{scenario}/{scenario}-{carrier}-{loc}.h5",
                    key="df",
                )

                if carrier == "electricity":
                    df_trans = df_dispatch[df_dispatch["techs"].str.match(r"^[A-Z]")].copy()
                    df_trans[["import", "export"]] = df_trans["techs"].str.split(
                        "_", n=1, expand=True
                    )
                    df_trans["net_import"] = df_trans["flow_in"] + df_trans["flow_out"]

                    # Prepare duals for merging
                    duals_rename = duals_.rename(
                        columns={"locs": "merge_locs", "dual_value": "dual_value"}
                    )
                    df_trans["merge_locs"] = df_trans.apply(
                        lambda x: x["export"] if x["net_import"] < 0 else x["import"], axis=1
                    )

                    # Merge on scenario, timestep, and the chosen locs
                    df_merged = df_trans.merge(
                        duals_rename,
                        left_on=["scenario", "timesteps", "merge_locs"],
                        right_on=["scenario", "timesteps", "merge_locs"],
                        how="left",
                    )
                    # Compute trade value
                    df_merged = df_merged.drop(columns="merge_locs")
                    df_merged["trade"] = df_merged["dual_value"] * df_merged["net_import"]
                    df_merged["trade"] *= 1 / 1000  # billion EUR
                    trade_carrier_loc_scen = df_merged["trade"].sum()

                elif carrier == "co2":
                    captured_and_stored = (
                        df_dispatch[df_dispatch["techs"] == "demand_co2"]
                        .groupby(["scenario", "techs"])[["flow_in", "flow_out"]]
                        .sum()
                        .reset_index()["flow_in"]
                        .iloc[0]
                    )/(10) # from kt to Mt

                    co2_value = (
                        duals_.groupby(["scenario", "locs", "carriers", "unit"])["dual_value"]
                        .mean()
                        .reset_index()
                    )
                    co2_loc_scen = co2_value[
                        (co2_value["locs"] == loc) & (co2_value["scenario"] == scenario)
                    ]
                    emission_loc_scen = (
                        emission_supply[
                            (emission_supply["locs"] == loc)
                            & (emission_supply["scenario"] == scenario)
                        ]
                        .groupby(["scenario", "locs", "unit"])["total_system_emissions"]
                        .sum()
                        .reset_index()
                    )

                    if emission_loc_scen.empty or co2_loc_scen.empty:
                        trade_carrier_loc_scen = 0
                    else:
                        trade_carrier_loc_scen = (
                            (emission_loc_scen["total_system_emissions"].iloc[0] + captured_and_stored)
                            * co2_loc_scen["dual_value"].iloc[0]
                            / 1000
                        )

                else:
                    df_distr = df_dispatch[df_dispatch["techs"].str.contains("distribution")].copy()
                    df_distr["net_import"] = df_distr["flow_in"] + df_distr["flow_out"]

                    # Merge on scenario, timestep, and the chosen locs
                    df_merged = df_distr.merge(
                        duals_,
                        left_on=["scenario", "carriers", "timesteps", "locs"],
                        right_on=["scenario", "carriers", "timesteps", "locs"],
                        how="left",
                    )
                    # Compute trade value
                    df_merged["trade"] = df_merged["dual_value"] * df_merged["net_import"]
                    df_merged["trade"] *= 1 / 1000  # billion EUR
                    trade_carrier_loc_scen = df_merged["trade"].sum()

                trade_dict[carrier][scenario][loc] = trade_carrier_loc_scen

    # Flatten results into a DataFrame
    all_frames = []
    for carrier, scen_dict in trade_dict.items():
        for scenario, loc_dict in scen_dict.items():
            for loc, num in loc_dict.items():
                df_copy = pd.DataFrame(
                    [
                        {
                            "scenario": scenario,
                            "locs": loc,
                            "carriers": carrier,
                            "trade_cost": num,
                            "unit": "billion_2015eur",
                        }
                    ]
                )
                all_frames.append(df_copy)

    trade = pd.concat(all_frames, ignore_index=True)

    # Save results
    output_path = os.path.join(cnf.RESULTS_PATH, "costs", "trade_revenues.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    trade.to_csv(output_path, index=False)

    return trade


def net_import_processing(scenario_list, carriers, country_list):
    """Compute import export sums for given scenarios, carriers, and countries."""
    emission_supply = helper.merge_scenario_output(scenario_list, "total_system_emissions")
    imp_exp: dict[str, dict[str, dict[str, float]]] = {}
    elec_net_import: dict[str, dict[str, dict[str, float]]] = {}

    # initialise once
    for scenario in scenario_list:
        elec_net_import[scenario] = {}

    for carrier in carriers:  # <-- added loop over carriers
        imp_exp[carrier] = {}

        for scenario in scenario_list:
            imp_exp[carrier][scenario] = {}

            filtered_scenario_list = [s for s in scenario_list if s == scenario]
            if not filtered_scenario_list:
                continue

            for loc in country_list:
                print(f"{carrier} | {scenario} | {loc}")
                
                # initialise per loc only once
                if loc not in elec_net_import[scenario]:
                    elec_net_import[scenario][loc] = {}

                # Read dispatch data
                df_dispatch = pd.read_hdf(
                    f"{cnf.RESULTS_PATH}/dispatch/{scenario}/{scenario}-{carrier}-{loc}.h5",
                    key="df",
                )

                if carrier == "electricity":
                    df_trans = df_dispatch[df_dispatch["techs"].str.match(r"^[A-Z]")].copy()
                    df_trans[["import", "export"]] = df_trans["techs"].str.split(
                        "_", n=1, expand=True
                    )
                    df_trans["net_import"] = df_trans["flow_in"] + df_trans["flow_out"]

                    # Prepare duals for merging
                    df_trans["merge_locs"] = df_trans.apply(
                        lambda x: x["export"] if x["net_import"] < 0 else x["import"], axis=1
                    )

                    # import_ = df_trans["flow_out"].sum()
                    # export_ = df_trans["flow_in"].sum()
                    elec_net_import[scenario][loc]["import"] = df_trans["flow_out"].sum()
                    elec_net_import[scenario][loc]["export"] = df_trans["flow_in"].sum()

                    carrier_loc_scen = df_trans["net_import"].sum()

                elif carrier == "co2":
                    captured_and_stored = (
                        df_dispatch[df_dispatch["techs"] == "demand_co2"]
                        .groupby(["scenario", "techs"])[["flow_in", "flow_out"]]
                        .sum()
                        .reset_index()["flow_in"]
                        .iloc[0]
                    )/(10) # from kt to Mt

                    emission_loc_scen = (
                        emission_supply[
                            (emission_supply["locs"] == loc)
                            & (emission_supply["scenario"] == scenario)
                        ]
                        .groupby(["scenario", "locs", "unit"])["total_system_emissions"]
                        .sum()
                        .reset_index()
                    )

                    if emission_loc_scen.empty:
                        carrier_loc_scen = 0
                    else:
                        carrier_loc_scen = (emission_loc_scen["total_system_emissions"].iloc[0] + captured_and_stored
                        )

                else:
                    df_distr = df_dispatch[df_dispatch["techs"].str.contains("distribution")].copy()
                    df_distr["net_import"] = df_distr["flow_in"] + df_distr["flow_out"]

                    carrier_loc_scen = df_distr["net_import"].sum()

                imp_exp[carrier][scenario][loc] = carrier_loc_scen

    # Flatten results into a DataFrame
    all_frames = []
    for carrier, scen_dict in imp_exp.items():
        for scenario, loc_dict in scen_dict.items():
            for loc, num in loc_dict.items():
                df_copy = pd.DataFrame(
                    [
                        {
                            "scenario": scenario,
                            "locs": loc,
                            "carriers": carrier,
                            "net_import": num,
                            "unit": "TWh",
                        }
                    ]
                )
                all_frames.append(df_copy)

    imp_exp_df = pd.concat(all_frames, ignore_index=True)
    # correct unit for CO2 flows
    imp_exp_df.loc[imp_exp_df["carriers"] == "co2", "unit"] = "MtCO2"

    # Save results
    output_path = os.path.join(cnf.RESULTS_PATH, "import-export", "net_import_sum.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    imp_exp_df.to_csv(output_path, index=False)

    # Flatten results into a DataFrame
    all_frames = []
    for scenario, scen_dict in elec_net_import.items():
        for loc, loc_dict in scen_dict.items():
            for type, value in loc_dict.items():
                df_copy = pd.DataFrame(
                    [
                        {
                            "scenario": scenario,
                            "locs": loc,
                            "flow": type,
                            "value": value,
                            "unit": "TWh",
                        }
                    ]
                )
                all_frames.append(df_copy)

    elec_df = pd.concat(all_frames, ignore_index=True)

    # Save results
    output_path = os.path.join(cnf.RESULTS_PATH, "import-export", "import_export_elec_df.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    elec_df.to_csv(output_path, index=False)
