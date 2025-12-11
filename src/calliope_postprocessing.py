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

# %%
