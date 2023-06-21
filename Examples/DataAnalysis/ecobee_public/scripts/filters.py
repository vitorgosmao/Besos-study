# general conditions that apply to all models
months = (11, 2)
hours = (20, 5)

# conditions for metadata
metadata_style = "detached"

# conditions for balance point
reject_outliers = True

# conditions for model fitting
interval_amnt = 30
duration = 3
heating_runtime_upper_bound = 0.8
heating_runtime_lower_bound = 0.05
indoor_variance_threshold = 0.2
cost_threshold = 100
initial_guess_model_fitting = [0.01, 100]
# initial_guess_model_fitting = [0.01, 100, 0.01]

# conditions for decay curve fitting
setpoint_derivative_threshold = -2.0
T_out_stationarity = 1.0
T_in_derivative_threshold = 0.00
T_in_out_diff = 5.0
proportion_heating = 0.1
max_duration = 6  # in hours
min_duration = 10.0  # in minutes
initial_guess = [100.0, 10.0]

# additional conditions
only_auxHeat1 = True
no_smart_recovery = False
no_heat_pump = True

model_fitting_params = (
    interval_amnt,
    duration,
    heating_runtime_upper_bound,
    heating_runtime_lower_bound,
    indoor_variance_threshold,
    initial_guess_model_fitting,
)

decay_curve_params = (
    setpoint_derivative_threshold,
    T_out_stationarity,
    T_in_derivative_threshold,
    T_in_out_diff,
    proportion_heating,
    max_duration,
    min_duration,
)

params_dict = {
    "months": months,
    "hours": hours,
    "metadata_style": metadata_style,
    "reject_outliers": reject_outliers,
    "interval_amnt": interval_amnt,
    "duration": duration,
    "heating_runtime_upper_bound": heating_runtime_upper_bound,
    "heating_runtime_lower_bound": heating_runtime_lower_bound,
    "setpoint_derivative_threshold": setpoint_derivative_threshold,
    "T_out_stationarity": T_out_stationarity,
    "T_in_derivative_threshold": T_in_derivative_threshold,
    "T_in_out_diff": T_in_out_diff,
    "proportion_heating": proportion_heating,
    "max_duration": max_duration,
    "min_duration": min_duration,
    "initial_guess": initial_guess,
    "only_auxHeat1": only_auxHeat1,
    "no_smart_recovery": no_smart_recovery,
    "indoor_variance_threshold": indoor_variance_threshold,
    "initial_guess_model_fitting": initial_guess_model_fitting,
    "no_heat_pump": no_heat_pump,
}
