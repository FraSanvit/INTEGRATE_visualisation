"""Holds generic configuration.

Used to standardize the look of plots throughout the report.
"""

# DATA_PATH = "D:\\transfer\V2G-final\\friendly-results\\2050"
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
FIGURE_FILE_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "figures"))
DISPATCH_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "results", "dispatch"))
MODEL_OUTPUT = "C:\\Users\\sanvi\\GitHub\\INTEGRATE\\results"
DATA_PATH = "D:\\transfer\\INTEGRATE\\friendly-results"



AUX_DATA_PATH = "C:\\Users\\sanvi\GitHub\\INTEGRATE_visualisation\\data"
FIG_PATH = "C:\\Users\\sanvi\\GitHub\\INTEGRATE_visualisation\\visualisation\\figures"
RESULTS_PATH = "C:\\Users\\sanvi\\GitHub\\INTEGRATE_visualisation\\results"
CF_INPUT_PATH = "D:\\transfer\\V2G\\2050\\model\\national"

# DATA_PATH = "C:\\Users\\sanvi\\GitHub\\V2G_visualisation\\data"

TECH_COLORS = {
    # Solar
    "open_field_pv": "#FFD800",
    "roof_mounted_pv": "#FFD800",
    "alpine_pv_subsidized": "#B8D800",
    "alpine_pv_unsubsidized": "#91AB00",
    "pv": "#FFD800",
    "alpine_pv": "#fd1f1f",
    # Wind
    "wind_onshore": "#1300FF",
    "wind_offshore": "#0A0088",
    "wind": "#1300FF",
    # Hydro
    "hydro_reservoir": "#34D8FF",
    "hydro_run_of_river": "#9CECFF",
    "hydro": "#9CECFF",
    "hydropower": "#34D8FF",
    # Base demands
    "demand_heat": "#FFA351",
    "demand_elec": "#FFC100",
    "demand_industry_co2": "#FFA351",
    "demand_industry_hydrogen": "#FFA351",
    "demand_industry_methane": "#FFA351",
    "demand_industry_methanol": "#FFA351",
    "demand_industry_kerosene": "#FFA351",
    "demand_industry_diesel": "#FFA351",
    "demand_heavy_duty_transport": "#73CC7B",
    "demand_light_duty_transport": "#B9FDBF",
    "demand_industry_district_heating": "#FFA351",
    "demand_industry_biomethane": "#FFA351",
    "demand_industry_biofuel": "#FFA351",
    "demand_industry_waste": "#FFA351",
    "demand_bus_transport": "#73CC23",
    "demand_motorcycle_transport": "#73CC23",
    "demand_passenger_car_transport": "#FFA351",
    "demand_co2": "#a4a4a4",
    "demand": "#FFA351",
    # Hydrogen
    "chp_hydrogen": "#25C29C",
    "electrolysis": "#DA94DA",
    # Imports/exports
    "hydrogen_distribution_import": "#C2C2C2",
    "syn_diesel_distribution_import": "#C2C2C2",
    "syn_methane_distribution_import": "#C2C2C2",
    "syn_methanol_distribution_import": "#C2C2C2",
    "syn_kerosene_distribution_import": "#C2C2C2",
    "biomethane_distribution_import": "#C2C2C2",
    "co2_export": "#20D6FF",
    "hydrogen_distribution_export": "#5A5A5A",
    "syn_diesel_distribution_export": "#C2C2C2",
    "syn_methane_distribution_export": "#5A5A5A",
    "syn_kerosene_distribution_export": "#5A5A5A",
    "syn_methanol_distribution_export": "#5A5A5A",
    "biomethane_distribution_export": "#086800",
    # Transport
    "heavy_transport_ev": "#88C95B",
    "heavy_transport_ice": "#8B5741",
    "heavy_transport_fcev": "#199378",
    "light_transport_ev": "#B0FF79",
    "light_duty_transport_ev": "#B0FF79",
    "light_duty_ev_battery": "#B0FF79",
    "light_transport_ice": "#FF9365",
    "light_transport_fcev": "#7AD9C4",
    "light_duty_transport_fcev": "#7AD9C4",
    "demand_rail": "#FFC100",
    "passenger_car_transport_ev": "#B0FF79",
    "passenger_car_transport_fcev": "#2200FF",
    "passenger_car_transport_ice": "#A92300",
    "ldv_transport_ev": "#B0FF30",
    "ldv_transport_fcev": "#3296CF",
    "ldv_transport_ice": "#A92300",
    "light_duty_transport_ice": "#A92300",
    "motorcycle_transport_ev": "#86CF32",
    "motorcycle_transport_ice": "#A92300",
    "passenger_car_ev_battery": "#0B5A2F",
    "bus_transport_ev": "#88C95B",
    "bus_ev_battery": "#88C95B",
    "bus_transport_fcev": "#2200FF",
    "bus_transport_ice": "#8B5741",
    "heavy_duty_transport_ev": "#88C95B",
    "heavy_duty_ev_battery": "#88C95B",
    "heavy_duty_transport_fcev": "#2200FF",
    "heavy_duty_transport_ice": "#8B5741",
    "motorcycle_ev_battery": "#88C95B",
    "transport": "#75e75f",
    # Storage
    "heat_storage_big": "#00728B",
    "heat_storage_small": "#20D6FF",
    "hydrogen_storage": "#FF80D1",
    "hydrogen_underground_storage": "#FF56A0",
    "storage_co2": "#734012",
    "methane_storage": "#734012",
    "battery": "#FBFF6F",
    "pumped_hydro": "#00A8D0",
    "district_storage": "#00728B",
    "domestic_storage": "#20D6FF",
    # Flexibility
    "flexibility_electricity": "#CDD27F",
    "flexibility_heat": "#D2A17F",
    # "Old" firm capacity
    "ccgt": "#717171",
    "ccgt_ccs": "#717171",
    "coal_power_plant": "#3C3C3C",
    "coal_power_plant_ccs": "#3C3C3C",
    # District heating gas
    "chp_methane_extraction": "#E868FC",
    "chp_methane_extraction_ccs": "#E868FC",
    # District heating waste
    "chp_wte_back_pressure": "#D3E240",
    "chp_wte_back_pressure_ccs": "#D3E240",
    # Distric heating bio
    "chp_biofuel_extraction": "#57EA5E",
    "chp_biofuel_extraction_ccs": "#57EA5E",
    "chp_biogas": "#D957EA",
    "chp_biogas_ccs": "#D957EA",
    "biofuel_thermal": "#015002",
    # District heating HP
    "geothermal_dh": "#855900F0",
    # Dispatchable
    "chp": "#2a8106",
    "chp_ccs": "#57EA5E",
    # Individual heating electric
    "electric_heater": "#8B0400",
    "hp": "#E24040",
    "heat_pump": "#E24040",
    "rooftop_solar_thermal": "#E8E812",
    # Individual heating fossil
    "methane_boiler": "#561B6F",
    "biofuel_boiler": "#008B06",
    "oil_boiler": "#633636",
    # DAC
    "dac": "#816FFF",
    # Bio-based fuels
    "biofuel_to_liquids": "#e4007c",
    "biogas_upgrading_ccs": "#0AC727",
    "biofuel_to_diesel": "#D957EA",
    "biofuel_to_methane": "#643D75",
    "biogas_upgrading": "#60F777",
    "biofuel_to_methanol": "#D957EA",
    # H2-based fuels
    "hydrogen_to_methane": "#CD7CF0",
    "hydrogen_to_methanol": "#9D51BD",
    "hydrogen_to_liquids": "#5E1F78",
    # Fossil fuels
    "methanol_supply": "#000000",
    "kerosene_supply": "#000000",
    "methane_supply": "#000000",
    "diesel_supply": "#000000",
    # Cooking
    "electric_hob": "#A54949",
    "gas_hob": "#561B6F",
    # Curtailment
    "curtailment": "#CDCDCD",
    # Supply
    "biofuel_supply": "#27BD50",
    "waste_supply": "#D1C814",
    "biogas_supply": "#27BD50",
    # Charging techs
    "uncontrolled_charging": "#FF923B",
    "v1g_charging": "#41FF3B",
    "unidirectional_charging":"#41FF3B",
    "v2g_charging": "#F03BFF",
    "bidirectional_charging":"#F03BFF",
    "hydrogen_station": "#0fe6b7",
    "bus_charging": "#0E4D25",
    "heavy_duty_charging": "#0E4D25",
    "light_duty_charging": "#197E36",
    "motorcycle_charging": "#197E36",
    "ev_charging": "#197E36",
    # Nuclear
    "nuclear": "#FF5733",
    # Transmission
    "dc_transmission": "#4F2626",
    "transmission": "#d5d5d5",
}

LOC_LIST = [
    "AUT",
    "BEL",
    "BGR",
    "HRV",
    "CYP",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "IRL",
    "ITA",
    "LVA",
    "LTU",
    "LUX",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "ESP",
    "SWE",
    "GBR",
    "ALB",
    "BIH",
    "MKD",
    "MNE",
    "NOR",
    "SRB",
    "CHE",
    "ISL",
]

SCENARIO_COLORS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
]


SCENARIO_NAMES = {
    "2030-int-island-1h": "2030-int-island",
    "2030-int-noisland-1h": "2030-int-noisland",
    "2030-int-superisland-1h": "2030-int-superisland",
    "2030-ref-island-1h": "2030-ref-island",
    "2030-ref-noisland-1h": "2030-ref-noisland",
    "2030-ref-superisland-1h": "2030-ref-superisland",
    "2050-int-island-1h": "2050-int-island",
    "2050-int-noisland-1h": "2050-int-noisland",
    "2050-int-superisland-1h": "2050-int-superisland",
    "2050-ref-island-1h": "2050-ref-island",
    "2050-ref-noisland-1h": "2050-ref-noisland",
    "2050-ref-superisland-1h": "2050-ref-superisland",
}

SCENARIO_NAMES_FULL = {
    "2030-int-island-1h": "2030-int-island",
    "2030-int-noisland-1h": "2030-int-noisland",
    "2030-int-superisland-1h": "2030-int-superisland",
    "2030-ref-island-1h": "2030-ref-island",
    "2030-ref-noisland-1h": "2030-ref-noisland",
    "2030-ref-superisland-1h": "2030-ref-superisland",
    "2050-int-island-1h": "2050-int-island",
    "2050-int-noisland-1h": "2050-int-noisland",
    "2050-int-superisland-1h": "2050-int-superisland",
    "2050-ref-island-1h": "2050-ref-island",
    "2050-ref-noisland-1h": "2050-ref-noisland",
    "2050-ref-superisland-1h": "2050-ref-superisland",
}

STORAGE_TECHS = [
    "heat_storage_small",
    "heat_storage_big",
    "battery",
    "hydrogen_storage",
    "methane_storage",
    "pumped_hydro",
    "passenger_car_ev_battery",
    "bus_ev_battery",
    "v2g_charging",
    "bus_ev_battery",
    "heavy_duty_ev_battery",
    "light_duty_ev_battery",
    "motorcycle_ev_battery",
]

CARRIER_DICT = {"biomethane": "methane"}

UNIT_CNF = {
    "tw": {"name": "GW", "factor": 1e3},
    "100kt_per_hour": {"name": "ton/h", "factor": 1e5},
}

UNIT_CONVERSION_ENERGY_DICT = {
    "tw": {"factor": 1, "unit": "TW"},
    "twh": {"factor": 1, "unit": "TWh"},
    "100kt": {"factor": 10, "unit": "Mt"},
    "100kt_per_hour": {"factor": 10, "unit": "Mt_per_hour"},
    "billion_km": {"factor": 1, "unit": "billion_km"},
    "billion_km_per_hour": {"factor": 1, "unit": "billion_km_per_hour"},
    "gw": {"factor": 1, "unit": "GW"},
}

UNIT_CONVERSION_CAP_DICT = {
    "tw": {"factor": 1000, "unit": "GW"},
    "twh": {"factor": 1000, "unit": "GWh"},
    "100kt": {"factor": 10, "unit": "Mt"},
    "100kt_per_hour": {"factor": 10, "unit": "Mt_per_hour"},
    "billion_km": {"factor": 1, "unit": "billion_km"},
    "billion_km_per_hour": {"factor": 1, "unit": "billion_km_per_hour"},
}

UNIT_DICT = {
    "transport": "Bkm",
    "electricity": "TWh",
    "methane": "TWh",
    "methanol": "TWh",
    "diesel": "TWh",
    "heat": "TWh",
    "hydrogen": "TWh",
    "kerosene": "TWh",
    "co2": "kton",
    "biofuel": "TWh",
    "waste": "TWh",
    "passenger_car_hydrogen": "TWh",
    "passenger_car_electricity": "TWh",
    "biogas": "TWh",
    "chp_biofuel_extraction_heat": "TWh",
    "methane_heat": "TWh",
    "wte_back_pressure_heat": "TWh",
    "hp_heat": "TWh",
    "syn_methane": "TWh",
    "chp_methane_extraction_heat": "TWh",
    "heavy_transport": "100 Mvkm",
    "cooking": "TWh",
    "syn_kerosene": "TWh",
    "syn_methanol": "TWh",
    "hydrogen_heat": "TWh",
    "motorcycle_transport": "100 Mvkm",
    "chp_biogas_heat": "TWh",
    "ldv_transport": "100 Mvkm",
    "coal": "TWh",
    "biofuel_heat": "TWh",
    "electric_heater_heat": "TWh",
    "syn_diesel": "TWh",
    "passenger_car_transport": "100 Mvkm",
    "biomethane": "TWh",
}

PATTERN_DICT = {
    "ccs": "/",
    "export": "x",
    "import": "x",
    "charging": "//",
    "station": ".",
}

CAP_TRESHOLD = {
    "electricity": 0.1,
    "heat": 0.1,
    "co2": 0.1,
    "hydrogen": 0.1,
    "methane": 0.1,
    "methanol": 0.1,
    "kerosene": 0.1,
    "diesel": 0.1,
}

TECH_DICT = {"hp": "heat pump"}

MIN_POS = 1e-4
MIN_NEG = -1e-4

ENERGY_POWER_UNIT_DICT = {
    "twh": "TW",
    "billion_km": "billion_km_per_hour",
    "100kt": "100kt_per_hour",
    "gwh": "GW",
}
# Technology specific

CHARGING_TECH_LIST = ["uncontrolled_charging", "v1g_charging", "v2g_charging"]

GROUPS_TECH_DICT = {
    "Dispatchable": ["ccgt", "chp"],
    "Hydro": ["hydro_reservoir", "hydro_run_of_river"],
    "Pumped hydro": ["pumped_hydro"],
    "Solar": ["pv"],
    "Wind": ["wind"],
    "Battery": ["battery"],
}

charging_tech_list = [
    "uncontrolled_charging",
    "v1g_charging",
    "v2g_charging",
    "hydrogen_station",
    "bus_charging",
    "heavy_duty_charging",
    "light_duty_charging",
    "motorcycle_charging",
]


GROUPS_TECH_DICT_COMPLETE = {
    "ccgt": ["ccgt", "ccgt_ccs"],
    "hydro": ["hydro_reservoir", "hydro_run_of_river"],
    "solar": ["open_field_pv", "roof_mounted_pv"],
    "wind": [
        "wind_offshore",
        "wind_onshore",
    ],
    "battery": ["battery"],
    "fuel import": [
        "syn_kerosene_distribution_import",
        "syn_methane_distribution_import",
        "syn_methanol_distribution_import",
        "syn_diesel_distribution_import",
        "hydrogen_distribution_import",
    ],
    "fuel export": [
        "syn_diesel_distribution_export",
        "syn_kerosene_distribution_export",
        "syn_methane_distribution_export",
        "syn_methanol_distribution_export",
        "hydrogen_distribution_export",
    ],
    "heat technology": ["biofuel_boiler", "methane_boiler", "electric_heater", "hp"],
    "biofuel supply": ["biofuel_supply", "biogas_supply"],
    "waste supply": ["waste_supply"],
    "biogas upgrading": [
        "biogas_upgrading",
        "biogas_upgrading_ccs",
    ],
    "other EV charging": [
        "bus_charging",
        "heavy_duty_charging",
        "light_duty_charging",
        "motorcycle_charging",
    ],
    "other EV battery": [
        "bus_ev_battery",
        "heavy_duty_ev_battery",
        "light_duty_ev_battery",
        "motorcycle_ev_battery",
    ],
    "chp": [
        "chp_biofuel_extraction",
        "chp_biogas",
        "chp_hydrogen",
        "chp_methane_extraction",
        "chp_wte_back_pressure",
    ],
    "chp_ccs": [
        "chp_biofuel_extraction_ccs",
        "chp_biogas_ccs",
        "chp_wte_back_pressure_ccs",
    ],
    "dac": ["dac"],
    "CO2 storage": ["demand_co2", "storage_co2"],
    "electrolysis": ["electrolysis"],
    "heat strorage": ["heat_storage_big", "heat_storage_small"],
    "fuel storage": ["hydrogen_storage", "methane_storage"],
    "pumped hydro": ["pumped_hydro"],
    "passenger EV battery": ["passenger_car_ev_battery"],
    "charging": ["v1g_charging"],
    "V2G": ["v2g_charging"],
    "nuclear": ["nuclear"],
    "hydrogen stations": ["hydrogen_station"],
    "fuel production": [
        "biofuel_to_diesel",
        "biofuel_to_liquids",
        "biofuel_to_methane",
        "biofuel_to_methanol",
        "hydrogen_to_liquids",
        "hydrogen_to_methane",
        "hydrogen_to_methanol",
    ],
    "transmission": ["ac_transmission", "dc_transmission"],
}


GROUPS_TECH_DICT_COST = {
    "hydro": ["hydro_reservoir", "hydro_run_of_river"],
    "solar": ["open_field_pv", "roof_mounted_pv"],
    "wind": [
        "wind_offshore",
        "wind_onshore",
    ],
    "battery": ["battery"],
    "fuel import/export": [
        "syn_kerosene_distribution_import",
        "syn_methane_distribution_import",
        "syn_methanol_distribution_import",
        "syn_diesel_distribution_import",
        "hydrogen_distribution_import",
        "syn_diesel_distribution_export",
        "syn_kerosene_distribution_export",
        "syn_methane_distribution_export",
        "syn_methanol_distribution_export",
        "hydrogen_distribution_export",
    ],
    "heat technology": ["biofuel_boiler", "methane_boiler", "electric_heater", "hp"],
    "biofuel and waste supply": ["biofuel_supply", "biogas_supply", "waste_supply"],
    "other EV charging": [
        "bus_charging",
        "heavy_duty_charging",
        "light_duty_charging",
        "motorcycle_charging",
    ],
    "other EV battery": [
        "bus_ev_battery",
        "heavy_duty_ev_battery",
        "light_duty_ev_battery",
        "motorcycle_ev_battery",
    ],
    "chp/ccgt": [
        "chp_biofuel_extraction",
        "chp_biogas",
        "chp_hydrogen",
        "chp_methane_extraction",
        "chp_wte_back_pressure",
        "chp_biofuel_extraction_ccs",
        "chp_biogas_ccs",
        "chp_wte_back_pressure_ccs",
        "ccgt",
        "ccgt_ccs",
    ],
    "dac": ["dac"],
    "electrolysis": ["electrolysis"],
    "heat storage": ["heat_storage_big", "heat_storage_small"],
    "CO2 and fuel storage": ["demand_co2", "storage_co2", "hydrogen_storage", "methane_storage"],
    "pumped hydro": ["pumped_hydro"],
    "passenger EV battery": ["passenger_car_ev_battery"],
    "charging": ["v1g_charging"],
    "V2G": ["v2g_charging"],
    "nuclear": ["nuclear"],
    "hydrogen stations": ["hydrogen_station"],
    "fuel production": [
        "biofuel_to_diesel",
        "biofuel_to_liquids",
        "biofuel_to_methane",
        "biofuel_to_methanol",
        "hydrogen_to_liquids",
        "hydrogen_to_methane",
        "hydrogen_to_methanol",
        "biogas_upgrading",
        "biogas_upgrading_ccs",
    ],
    "transmission": ["ac_transmission", "dc_transmission"],
}


GROUPS_TECH_DICT_COST_COLOUR = {
    "hydro": "#34D8FF",
    "solar": "#FFD800",
    "wind": "#1300FF",
    "battery": "#FBFF6F",
    "fuel import/export": "##933824",
    "heat technology": "#ff3f16",
    "biofuel and waste supply": "#82d463",
    "other EV charging": "#90ff8c",
    "other EV battery": "#67e936",
    "chp/ccgt": "#CECECE",
    "dac": "#816FFF",
    "electrolysis": "#DA94DA",
    "heat storage": "#7bd5d8",
    "CO2 and fuel storage": "#d87b7b",
    "pumped hydro": "#00A8D0",
    "passenger EV battery": "#5ce423",
    "charging": "#41ff3b",
    "V2G": "#f03bf0",
    "nuclear": "#ff0d0d",
    "hydrogen stations": "#0fe6b7",
    "fuel production": "#643d75",
    "transmission": "#595959",
}

GROUPS_TECH_DICT_MINIMAL = {
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
    "heat_pump": ["hp", "geothermal_dh"],
    "unidirectional_charging": ["v1g_charging"],
    "bidirectional_charging": ["v2g_charging"],
}


GROUPS_TECH_DICT_FLOW = {
    "hydro": ["hydro_reservoir", "hydro_run_of_river"],
    "solar": ["open_field_pv", "roof_mounted_pv"],
    "wind": [
        "wind_offshore",
        "wind_onshore",
    ],
    "ccgt": [
        "ccgt",
        "ccgt_ccs",
    ],
    "chp_biofuel": [
        "chp_biofuel_extraction",
        "chp_biofuel_extraction_ccs",
    ],
    "chp_biogas": [
        "chp_biogas",
        "chp_biogas_ccs",
    ],
    "chp_hydrogen/methane": [
        "chp_hydrogen",
        "chp_methane_extraction",
    ],
    "chp_wte": [
        "chp_wte_back_pressure",
        "chp_wte_back_pressure_ccs",
    ],
    "dac": ["dac"],
    "electric_heater": ["electric_heater"],
    "electrolysis": ["electrolysis"],
    "heat_pump": ["hp"],
    "battery": ["battery"],
    "other_ev_charging": [
        "bus_charging",
        "heavy_duty_charging",
        "light_duty_charging",
        "motorcycle_charging",
    ],
    "other_ev_battery": [
        "bus_ev_battery",
        "heavy_duty_ev_battery",
        "light_duty_ev_battery",
        "motorcycle_ev_battery",
    ],
    "geological_storage": ["demand_co2", "storage_co2"],
    "unidirectional_charging": ["v1g_charging"],
    "bidirectional_charging": ["v2g_charging"],
    "biofuel_to_fuel": [
        "biofuel_to_diesel",
        "biofuel_to_liquids",
        "biofuel_to_methane",
        "biofuel_to_methanol",
    ],
    "hydrogen_to_fuel": [
        "hydrogen_to_liquids",
        "hydrogen_to_methane",
        "hydrogen_to_methanol",
    ],
    "biogas_upgrading": [
        "biogas_upgrading",
        "biogas_upgrading_ccs",
    ],
    "transmission": ["ac_transmission", "dc_transmission"],
    "biofuel_boiler": ["biofuel_boiler"],
    "methane_boiler": ["methane_boiler"],
    "domestic_heat_storage": ["heat_storage_small"],
    "district_heat_storage": ["heat_storage_big"],
    "demand_industry": [
        "demand_industry_kerosene",
        "demand_industry_methane",
        "demand_industry_methanol",
        "demand_industry_diesel",
        "demand_industry_hydrogen",
    ],
    "fuel_import": [
        "syn_diesel_distribution_import",
        "syn_methane_distribution_import",
        "syn_methanol_distribution_import",
        "syn_kerosene_distribution_import",
        "hydrogen_distribution_import",
    ],
    "fuel_export": [
        "syn_diesel_distribution_export",
        "syn_methane_distribution_export",
        "syn_methanol_distribution_export",
        "syn_kerosene_distribution_export",
        "hydrogen_distribution_export",
    ],
    "passenger_fcev": ["passenger_car_transport_fcev"],
    "passenger_ev": ["passenger_car_transport_ev"],
}

GROUPS_TECH_DICT_COLOUR = {
    "hydro": "#34D8FF",
    "solar": "#FFD800",
    "wind": "#1300FF",
    "battery": "#FBFF6F",
    "ccgt": "#CECECE",
    "chp_biofuel": "#4aaa47",
    "chp_biogas": "#f396ee",
    "chp_hydrogen/methane": "#47aa92",
    "chp_wte": "#b9bd65",
    "other_ev_charging": "#90ff8c",
    "geological_storage": "#d87b7b",
    "unidirectional_charging": "#41ff3b",
    "bidirectional_charging": "#f03bf0",
    "biofuel_to_fuel": "#1f6021",
    "hydrogen_to_fuel": "#2a8b9a",
    "biogas_upgrading": "#d0427a",
    "transmission": "#595959",
    "domestic_heat_storage": "#7bd5d8",
    "district_heat_storage": "#4799aa",
    "hydrogen": "#7bd5d0",
    "electricity": "#FFC300",
    "diesel": "#972020",
    "heat_pump": "#E24040",
    "demand_industry": "#FFA351",
    "fuel_import": "#C2C2C2",
    "fuel_export": "#C2C2C2",
    "passenger_ev":"#49a664",
    "passenger_fcev":"#45a8dd",
}


CONNECTION_DICT = {
    "fixed_charge": "uncoordinated-charging-profile",
    "uncontrolled": "uncoordinated-charging-profile",
    "v1g": "v1g-charging-profile",
    "v2g": "v2g-charging-profile",
}

SCENARIO_LABEL_MAPPING = {
    "int": "LED",
    "ref": "TECH",
    "island": "Limited",
    "noisland": "Unlimited",
    "superisland": "Autarky",
}
