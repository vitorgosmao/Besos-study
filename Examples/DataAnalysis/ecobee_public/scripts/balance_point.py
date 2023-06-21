import pandas as pd
import numpy as np
from scipy import stats

import scripts.preprocessing as pp


def resample_data(df):
    resampled = df.resample("D").mean()
    df_xy = resampled[["T_out", "auxHeat1"]]
    df_xy_nonzero = df_xy.loc[df_xy["auxHeat1"] != 0]
    df_final = df_xy_nonzero.dropna()

    return df_final


def reject_outliers_balance_point(data):
    u = np.mean(data["T_out"])
    s = np.std(data["T_out"])
    data_filtered = data[(data["T_out"] > (u - 1 * s)) & (data["T_out"] < (u + 1 * s))]
    return data_filtered


def find_slope(df, reject_outliers=True):
    df_final = resample_data(df)

    if reject_outliers:
        df_final = reject_outliers_balance_point(df_final)

    x = df_final["T_out"]
    y = df_final["auxHeat1"]
    line = stats.linregress(x, y)

    return line


def analyze_building_balance_point(
    months, hours, filename, building_df, reject_outliers=True
):

    limited_building_df = pp.limit_time_ranges(months, hours, building_df)

    line = find_slope(limited_building_df, reject_outliers)
    result = line_to_dict(line, filename)
    return result


def line_to_dict(line, filename):
    """Take relevant parts of scipy stats line object"""
    slope = line.slope
    intercept = line.intercept
    r_value = line.rvalue
    p_value = line.pvalue
    stderr = line.stderr

    return {
        "filename": filename,
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "stderr": stderr,
    }
