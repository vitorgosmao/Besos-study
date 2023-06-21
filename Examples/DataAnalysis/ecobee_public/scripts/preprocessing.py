import os
import pandas as pd
from datetime import datetime

ROOT_DIR = "./"  # os.path.abspath(os.path.dirname("../ecobee_public/"))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
METADATA_FILE = os.path.join(RAW_DATA_DIR, "metadata.csv")
CITY_DATA_DIR = os.path.join(RAW_DATA_DIR, "Toronto")
TIME_CONSTANT_DIR = os.path.join(DATA_DIR, "model_results")
NEW_DATA_DIR = TIME_CONSTANT_DIR


def create_new_data_file(model_type, filename):
    filepath = os.path.join(NEW_DATA_DIR, model_type)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    return os.path.join(filepath, filename)


def choose_metadata(
    no_heat_pump, style=None,
):
    """"""
    metadata_df = pd.read_csv(METADATA_FILE, index_col="Identifier")
    selected_df = metadata_df[
        (metadata_df["ProvinceState"] == "ON") | (metadata_df["ProvinceState"] == "NY")
    ]
    if style:
        selected_df = selected_df[selected_df["Style"] == style]

    if no_heat_pump:
        selected_df = selected_df[selected_df["Has a Heat Pump"] == False]

    # selected_df = metadata_df[(metadata_df["City"]=="Toronto") & (metadata_df["Style"] == "detached")].copy()

    return selected_df


def load_building_data(filename):
    path = os.path.join(CITY_DATA_DIR, filename)

    def FtoC(temperature):
        return (temperature - 32.0) * 5.0 / 9.0

    def normalize_runtime(runtime):
        return runtime / 300.0

    usecols = [
        "DateTime",
        "T_stp_heat",
        "T_stp_cool",
        "Thermostat_Temperature",
        "T_out",
        "T_ctrl",
        "auxHeat1",
        "auxHeat2",
        "auxHeat3",
        "compHeat1",
        "compHeat2",  # "compHeat3",
        "compCool1",
        "compCool2",  # "compCool3",
        "Remote_Sensor_1_Temperature",
        "Remote_Sensor_2_Temperature",
        "Remote_Sensor_3_Temperature",
    ]

    temperature_attributes = [
        "T_stp_heat",
        "T_stp_cool",
        "Thermostat_Temperature",
        "T_out",
        "T_ctrl",
        "Remote_Sensor_1_Temperature",
        "Remote_Sensor_2_Temperature",
        "Remote_Sensor_3_Temperature",
    ]
    runtime_attributes = [
        "auxHeat1",
        "auxHeat2",
        "auxHeat3",
        "compHeat1",
        "compHeat2",  # "compHeat3",
        "compCool1",
        "compCool2",
    ]  # "compCool3"]

    df = pd.read_csv(
        path,
        index_col="DateTime",
        # usecols = usecols,
        parse_dates=["DateTime"],
    )

    df[temperature_attributes] = df[temperature_attributes].apply(FtoC)
    df[runtime_attributes] = df[runtime_attributes].apply(normalize_runtime)

    return df


def limit_time_ranges(months, hours, building_df):

    start_month, end_month = months
    start_hour, end_hour = hours

    # choose only preselected months and hours from the time-series
    #     print(building_df.index.month)
    limited_building_df = building_df[
        (building_df.index.month >= start_month)
        | (building_df.index.month <= end_month)
    ]

    limited_building_df = limited_building_df[
        (limited_building_df.index.hour >= start_hour)
        | (limited_building_df.index.hour <= end_hour)
    ]
    if len(limited_building_df) == 0:
        raise ValueError("No data")

    return limited_building_df
