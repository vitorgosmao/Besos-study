import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import scripts.preprocessing as pp

MODEL_TYPE = "decay_curves"


def result_to_dict(start_time, end_time, T_in_mean, T_out_mean, result):
    tau = result[0][0]
    T0 = result[0][1]
    perr = np.sqrt(np.diag(result[1]))
    tau_variance = perr[0]
    T0_variance = perr[1]

    # TODO
    # pass average temperature inside/outside
    result = {
        "start_time": start_time,
        "end_time": end_time,
        "T_in_mean": T_in_mean,
        "T_out_mean": T_out_mean,
        "tau": tau,
        "T0": T0,
        "tau_variance": tau_variance,
        "T0_variance": T0_variance,
    }
    return result


def decay_curve(t, tau, theta0):
    return theta0 * np.exp(-t / tau)


def fit_decay_curve(building_df, initial_guess=[100.0, 10.0]):
    xdata = np.arange(0, (len(building_df) * 5) - 1, 5.0)
    # TODO
    # think about statinary outside temperatures
    ydata = (
        np.array(building_df["Thermostat_Temperature"]) - building_df["T_out"].mean()
    )

    result = curve_fit(decay_curve, xdata, ydata, p0=initial_guess)  # was 10
    return result


def select_intervals(months, hours, decay_curve_params, building_df):

    time_intervals = []

    # make a copy
    building_df_copy = building_df.copy()

    # select based on params
    (
        setpoint_derivative_threshold,
        T_out_stationarity,
        T_in_derivative_threshold,
        T_in_out_diff,
        proportion_heating,
        max_duration,
        min_duration,
    ) = decay_curve_params

    # find difference between time steps using earlier time step
    building_df_copy["d_T_stp_heat"] = building_df_copy["T_stp_heat"].diff()
    building_df_copy["d_T_in"] = building_df_copy["Thermostat_Temperature"].diff()
    building_df_copy["T_in_out_diff"] = (
        building_df_copy["Thermostat_Temperature"] - building_df_copy["T_out"]
    )

    # label whether it is a setpoint drop
    building_df_copy["is_setpoint_drop"] = (
        building_df_copy["d_T_stp_heat"] <= setpoint_derivative_threshold
    )

    # label whether it is a point that we're interested in
    building_df_copy["is_interesting"] = (
        (building_df_copy["d_T_in"] < T_in_derivative_threshold)
        & (building_df_copy["auxHeat1"] <= proportion_heating)
        & (building_df_copy["T_in_out_diff"] >= T_in_out_diff)
    )
    # TODO
    # decide how to deal with stationarity of outside temperature

    # drop data by limiting time ranges
    limited_building_df = pp.limit_time_ranges(months, hours, building_df_copy)

    # iterate throguh setpoint drops
    setpoint_drops = limited_building_df[
        limited_building_df["is_setpoint_drop"] == True
    ].index  # .values

    for index in setpoint_drops:
        start_time = index + pd.Timedelta(5, "m")
        end_time = limited_building_df["is_interesting"].loc[
            start_time : start_time + pd.Timedelta(max_duration, "h")
        ].idxmin() - pd.Timedelta(
            5, "m"
        )  # drop last minutes ("bad" poinyt)

        subtracted = end_time - start_time
        if end_time - start_time > pd.Timedelta(max_duration, "h"):
            # this is discontinuous data
            pass
        else:
            # keep them if they're long enough
            if end_time - start_time > pd.Timedelta(min_duration, "m"):
                time_intervals.append((start_time, end_time))

    # TODO:
    # check for NaNs and other bad data
    return time_intervals


def analyze_building_decay_curves(
    months, hours, duration, filename, building_df, decay_curve_params, initial_guess
):
    time_intervals = select_intervals(months, hours, decay_curve_params, building_df)
    results = []

    for time_interval in time_intervals:
        try:
            result = fit_decay_curve(
                building_df[time_interval[0] : time_interval[1]], initial_guess
            )
            result_dict = result_to_dict(
                time_interval[0],
                time_interval[1],
                building_df.loc[
                    time_interval[0] : time_interval[1], "Thermostat_Temperature"
                ].mean(),
                building_df.loc[time_interval[0] : time_interval[1], "T_out"].mean(),
                result,
            )
            results.append(result_dict)

        except Exception as e:
            pass

    results_df = pd.DataFrame(results)
    filepath = pp.create_new_data_file(MODEL_TYPE, filename)
    results_df.to_csv(filepath, index=False)
