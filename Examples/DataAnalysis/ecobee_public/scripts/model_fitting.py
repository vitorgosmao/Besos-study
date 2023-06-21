import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import traceback

import scripts.preprocessing as pp


MODEL_TYPE = "model_fitting"
# 5 minute time step
dt = 5.0


def solve_euler(variables_for_single_fit, dt, p):
    """ Solve for internal temperature using Euler's method of numerical differentiation """
    t, T_ext, d_heat, T_data, T0, _ = variables_for_single_fit
    T = []
    T.append(T0)
    for i in range(1, len(t)):
        if len(p) == 3:
            # solve for Euler's method using an additional error term that represents internal gains
            T_i = (
                T[i - 1]
                + p[0] * ((T_ext[i - 1] - T[i - 1]) + p[1] * d_heat[i - 1] + p[2]) * dt
            )
        if len(p) == 2:
            T_i = (
                T[i - 1]
                + p[0] * ((T_ext[i - 1] - T[i - 1]) + p[1] * d_heat[i - 1]) * dt
            )
        T.append(T_i)
    return T


def ode_solution_helper(variables_for_fit, dt, p):
    """ Solve for many different intervals """
    Ts = []
    for v in variables_for_fit:
        T = solve_euler(v, dt, p)
        Ts.append(T)
    return Ts


def data_frame_to_lists(df):
    """
    Converts a dataframe into lists and initial values that can be used for training

    param df: A dataframe that has been indexed according to preselected intervals
    """
    t = np.array(df.index.values, dtype="M8[m]").astype(float)
    t = t - t[0]
    T_ext = df["T_out"].values
    d_heat = df["auxHeat1"].values
    T_data = df["Thermostat_Temperature"].values
    T0 = T_data[0]
    indices = df.index.values
    return t, T_ext, d_heat, T_data, T0, indices


def initialize_variables_for_fit(df, intervals):
    """
    Create arrays from the dataframe.
    These are needed to do the integration and to plot the results
    """
    variables_for_fit = []
    for start, end in intervals:
        df_new = df[start:end]
        vars = data_frame_to_lists(df_new)

        variables_for_fit.append(vars)

    return variables_for_fit


def fit_model(df, initial_guess_model_fitting, intervals):
    """
    Returns scipy OptimizeResult
    """

    def ode_solution(p):
        """Euler's method for finding internal temperature from"""
        return ode_solution_helper(variables_for_fit, dt, p)

    def residuals(p):
        """calculate difference between integrated ODE and actual inside temperature"""
        soln = ode_solution(p)
        T_data_final = []
        solution = []
        for v, s in zip(variables_for_fit, soln):
            (t, T_ext, d_heat, T_data, T0, _) = v
            T_data_final = T_data_final + list(T_data)
            solution = solution + list(s)
        solution = np.array(solution)
        T_data_final = np.array(T_data_final)
        f = solution - T_data_final
        return np.array(f)

    # convert series to numpy arrays
    variables_for_fit = initialize_variables_for_fit(df, intervals)

    # pre-process data to replace NaNs in T_out with previous value
    for v in variables_for_fit:
        (t, T_ext, d_heat, T_data, T0, _) = v
        for i in range(1, len(t)):
            if np.isnan(T_ext[i]):
                T_ext[i] = T_ext[i - 1]
            if np.isnan(T_data[i]):
                T_data[i] = T_data[i - 1]
            if np.isnan(d_heat[i]):
                d_heat[i] = d_heat[i]

    # set different bounds depending on the amount of initial parameters
    if len(initial_guess_model_fitting) == 3:
        bounds = ([1e-6, 1e-6, 1e-6], [10, 500, 500])
    elif len(initial_guess_model_fitting) == 2:
        bounds = ([1e-6, 1e-6], [10, 500])

    # minimize residuals using scipy.optimize least_squares for nonlinear regression
    solution = least_squares(
        fun=residuals,
        # method='lm',  DIDN'T WORK
        x0=initial_guess_model_fitting,
        bounds=bounds,
    )
    return solution


def solution_to_dict(solution, intervals):
    """
    Takes scipy OptimizeResult and converts it into a dictionary that will be saved to a csv
    """
    parameters = solution.x
    cost = solution.cost
    status = solution.status
    active_mask = solution.active_mask

    result = {
        "intervals": intervals,
        "interval_amount": len(intervals),
        "tau": parameters[0],
        "RK": parameters[1],
        "cost": cost,
        "status": status,
        "active_mask_0": active_mask[0],
        "active_masks_1": active_mask[1],
    }
    return result


def subtract_time(time_1, time_2):
    time = time_1 - time_2
    if time < 0:
        time = 24 + time
    return time


def select_intervals(months, hours, building_df, model_fitting_params):
    """
    Select the time intervals that will be used to train the data.

    Each interval is subject to certain constraints, according to model_fitting_params. These constraints are discussed
    further in the paper.

    TODO - comment and clean up this function
    """
    (
        interval_amnt,
        duration,
        heating_runtime_upper_bound,
        heating_runtime_lower_bound,
        indoor_variance_threshold,
        _,
    ) = model_fitting_params

    intervals = []
    start_hour, end_hour = hours
    limited_building_df = pp.limit_time_ranges(months, hours, building_df)

    # select duration for data
    data_start_date = limited_building_df.index[0]
    data_end_date = limited_building_df.index[-1]
    num_days = data_end_date - data_start_date

    hour_range = (start_hour, subtract_time(end_hour, duration))

    # TODO
    #     if abs(hour_range[0] - hour_range[1]) > duration:
    #         raise ValueError("You have specified a time duration that is longer than the specified hours")

    for i in range(interval_amnt):
        random_day = num_days * np.random.random()

        # NOTE THIS WON'T WORK IF USING HOURS THAT DO NOT CROSS OVERNIGHT
        # TODO - fix and test
        hours = list(range(hour_range[0], 24))
        hours = hours + list(range(0, hour_range[1]))

        random_hour = np.random.choice(hours)

        start_date = data_start_date + random_day
        start_date.replace(hour=random_hour)

        end_date = start_date + pd.Timedelta(duration, "h")

        building_times = building_df[start_date:end_date]
        useful_attributes = ["Thermostat_Temperature", "T_out", "auxHeat1"]
        if (building_times["auxHeat1"].mean() > heating_runtime_upper_bound) or (
            building_times["auxHeat1"].mean() < heating_runtime_lower_bound
        ):
            continue

        if (building_times[useful_attributes].isna().values.any()) or (
            len(building_times) == 0
        ):
            # TODO retry
            # TODO also check that building_times has the correct number of values
            # print("skipping because of null value")
            continue
        if building_times["Thermostat_Temperature"].var() < indoor_variance_threshold:
            # print("continuing because not enough indoor variance")
            continue

        intervals.append((start_date.ceil("5min"), end_date.ceil("5min")))
        # intervals.append((start_date, end_date))
    return intervals


def test_model(solution, months, hours, model_fitting_params, building_df, result_dict):
    """
    Test the model for overfitting. Use the RC and RK values, test how they work over several intervals and record
    the cost.

    """
    intervals = select_intervals(months, hours, building_df, model_fitting_params)
    p = solution.x
    dt = 5.0
    i = 0
    for start, end in intervals:
        # solve using the predict values
        df = building_df[start:end]
        variables_for_fit = data_frame_to_lists(df)
        _, _, _, T_data, _, _ = variables_for_fit
        T = solve_euler(variables_for_fit, dt, p)

        # find the cost and record the result
        cost = T - T_data
        cost_name = "cost_" + str(i)
        result_dict[cost_name] = sum(cost ** 2)
        i += 1

    return result_dict


def analyze_building_model_fitting(
    months, hours, filename, building_df, model_fitting_params, cost_threshold
):
    """
    Analyze the buildings using the model fitting method

    This method uses Euler's method for numerical differentiation to create a dataset that a model is trained over.
    The dataset consists of multiple time periods. At the beginning of each time period the y value (Indoor temperature)
    is reset to prevent the predicted differential value from deviating too far from the real value for temperature.

    For each building, the multiple training periods are used. The resulting RC and RK values are tested on different
    portions of the building data to check for overfitting.

    During the analysis phase, buildings with very high costs can be discarded to improve results.
            
    """
    results = []
    (_, _, _, _, _, initial_guess_model_fitting) = model_fitting_params

    # Train the model 10 different times
    # TODO this should be a parameter passed in from main.py
    for _ in range(10):
        try:
            intervals = select_intervals(
                months, hours, building_df, model_fitting_params
            )
            if len(intervals) != 0:
                # train the model
                solution = fit_model(
                    building_df, initial_guess_model_fitting, intervals
                )

                # save the solution
                result_dict = solution_to_dict(solution, intervals)
                result_dict = test_model(
                    solution,
                    months,
                    hours,
                    model_fitting_params,
                    building_df,
                    result_dict,
                )
                if result_dict.get("cost") > cost_threshold:
                    # skip because cost is high
                    continue
                results.append(result_dict)
        except Exception as e:
            print(e)
            # print(traceback.format_exc())

    # convert the solution to a CSV
    results_df = pd.DataFrame(results)
    filepath = pp.create_new_data_file(MODEL_TYPE, filename)
    results_df.to_csv(filepath, index=False)
