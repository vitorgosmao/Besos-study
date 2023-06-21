import sys
import os
import calendar
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from ast import literal_eval

import scripts.preprocessing as pp

# declare variables
PLOT_DIR = os.path.join(pp.DATA_DIR, "plots")


def get_balance_points(filename, balance_point_df):
    bp = balance_point_df[balance_point_df["filename"] == filename]
    return bp


def get_euler(euler_path, filename):
    ef = os.path.join(euler_path, filename)
    ef_df = pd.read_csv(ef)
    ef_df["tau"] = 1 / ef_df["tau"]
    return ef_df


def get_decay(decay_path, filename):
    dc = os.path.join(decay_path, filename)
    dc_df = pd.read_csv(dc)
    dc_df["tau"] = dc_df["tau"]
    return dc_df


def get_filenames(directories):
    """
    Get the filenames that exist for the specified directories.
    
    param directories: a list of strings which are the paths to the directories
    """
    filenames = set()
    for directory in directories:
        for d in os.listdir(directory):
            filenames.add(d)
    return list(filenames)


def create_datetime_labels(df_copy):
    # prepare date names for columns
    df_copy["start_time"] = pd.to_datetime(df_copy["start_time"])
    df_copy["start_month"] = df_copy["start_time"].dt.month
    df_copy["start_day"] = df_copy["start_time"].dt.day
    df_copy["start_hour"] = df_copy["start_time"].dt.hour
    df_copy["start_year"] = df_copy["start_time"].dt.year

    df_copy["start_month"] = df_copy["start_month"].apply(
        lambda x: calendar.month_abbr[x]
    )
    df_copy["start"] = (
        df_copy["start_year"].astype(str)
        + ", "
        + df_copy["start_month"].astype(str)
        + ", "
        + df_copy["start_day"].astype(str)
        + ", "
        + df_copy["start_hour"].astype(str)
    )
    return df_copy


def plot_summary_bars(df, y, error, ax, title):
    df_copy = df.copy()
    df_copy = create_datetime_labels(df_copy)

    # create plot
    df_copy.set_index("start", drop=True, inplace=True)
    df_copy[y].plot.bar(yerr=df_copy[error], ax=ax, title=title)


def save_figure(plot_path, filename):
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    plot_filename = os.path.join(plot_path, filename[0:-4])
    plt.savefig(plot_filename)


def plot_results(house_euler, house_decay, house_balance_point, plot_path, filename):
    f, axes = plt.subplots(1, 3, figsize=(10, 4))

    # plot model
    plot_summary_bars(house_euler, "tau", "cost", axes[0], "Eulers Method: tau")
    plot_summary_bars(house_euler, "RK", "cost", axes[1], "Eulers Method: RK")

    # plot decay curves
    plot_summary_bars(house_decay, "tau", "tau_variance", axes[2], "Decay Curves: tau")

    txt = (
        "Balance point slope: "
        + str(house_balance_point.slope.values[0])
        + "   RK mean: "
        + str(house_euler["RK"].mean())
    )
    f.suptitle("Results for: " + filename)

    f.text(0.5, 0.001, txt, ha="center")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(plot_path, filename)


def check_skip(df):
    """ If True is returned, skip this house 
    
    Skip values if the start and end times are 0, aka no results were returned
    """
    skip = len(df) == 0 or (df["start_time"] == 0).any()
    # TODO remove values from the decay cures if there was infinite error
    return skip


def drop_indices_at_bounds(house_euler):
    """
    drop values that hit the following bounds, because it means the algorithm did not converge:
    [10-6, 500] = RK bounds = euler's method
    [10-6, 10^6] = RC bounds = euler's method
    None for decay curves
    """
    drop_indexes = list(house_euler[np.isclose(500, house_euler["RK"])].index)
    if drop_indexes:
        house_euler.drop(drop_indexes, inplace=True)
    drop_indexes = list(house_euler[np.isclose(1000000, house_euler["tau"])].index)
    if drop_indexes:
        house_euler.drop(drop_indexes, inplace=True)

    drop_indexes = list(house_euler[house_euler["tau"] >= 1000000].index)
    if drop_indexes:
        house_euler.drop(drop_indexes, inplace=True)

    return house_euler


def drop_bad_indices(house_euler):
    # drop values where algorithm couldn't converge
    house_euler = drop_indices_at_bounds(house_euler)
    # house_euler['cost_mean_test'] = house_euler.filter(regex=("cost_.*")).mean(axis=1)
    # house_euler = house_euler[house_euler['cost_mean_test'] < 35]
    return house_euler


def plot_all_results(
    filenames, decay_path, euler_path, metadata_df, balance_point_df, plot_path
):
    print("Plotting results for each house...")
    for filename in filenames:
        try:
            house_decay = get_decay(decay_path, filename)
            house_euler = get_euler(euler_path, filename)
            house_balance_point = get_balance_points(filename, balance_point_df)
            if check_skip(house_decay) or check_skip(house_euler):
                pass

            else:
                house_euler = drop_bad_indices(house_euler)
                plot_results(
                    house_euler, house_decay, house_balance_point, plot_path, filename
                )
        except pd.errors.EmptyDataError as e:
            print("File was empty, skip it")
        except pd.errors.ParserError as e:
            print("Received a parse error for file", filename)
        except FileNotFoundError as e:
            # print(e)
            pass
        except Exception as e:
            print(e)
            traceback.print_exc()


def create_summary_stats(
    filenames, euler_path, decay_path, balance_point_df, drop_outliers
):
    summary_dfs = []
    for filename in filenames:
        row = {}
        row["filename"] = filename
        try:
            # Summarize house euler
            house_euler = get_euler(euler_path, filename)
            if drop_outliers:
                house_euler = drop_bad_indices(house_euler)

            # divide by 12 to convert to hours
            house_euler["tau"] = house_euler["tau"].divide(12)

            row["euler_RK_mean"] = house_euler["RK"].mean()
            row["euler_RK_median"] = house_euler["RK"].median()
            row["euler_RK_std"] = house_euler["RK"].std()
            row["euler_RC_mean"] = house_euler["tau"].mean()
            row["euler_RC_median"] = house_euler["tau"].median()
            row["euler_RC_std"] = house_euler["tau"].std()
            row["euler_cost"] = house_euler["cost"].mean()

            costs = house_euler.filter(regex=("cost_.*")).mean(axis=1)
            if len(costs) > 0:
                min_cost = costs.idxmin()
                row["euler_RC_min_cost"] = house_euler.loc[min_cost]["tau"]
                row["euler_RK_min_cost"] = house_euler.loc[min_cost]["RK"]
                row["euler_cost_mean"] = (
                    house_euler.filter(regex=("cost_.*")).mean(axis=1).mean()
                )

        except Exception as e:
            row["euler_RK_mean"] = np.nan
            row["euler_RK_median"] = np.nan
            row["euler_RK_std"] = np.nan
            row["euler_RC_mean"] = np.nan
            row["euler_RC_median"] = np.nan
            row["euler_RC_std"] = np.nan
            row["euler_cost"] = np.nan
            row["euler_RC_min_cost"] = np.nan
            row["euler_test_cost"] = np.nan
            row["euler_RK_min_cost"] = np.nan

        try:
            # Summarize house decay
            house_decay = get_decay(decay_path, filename)

            # divide by 12 to conver to hours
            house_decay["tau"] = house_decay["tau"].divide(12)

            row["decay_amnt"] = len(house_decay)
            row["decay_RC_mean"] = house_decay["tau"].mean()
            row["decay_RC_median"] = house_decay["tau"].median()
            row["decay_RC_std"] = house_decay["tau"].std()
            row["decay_RC_cost"] = house_decay["tau_variance"].mean()
        except:
            row["decay_RC_mean"] = np.nan
            row["decay_RC_median"] = np.nan
            row["decay_RC_std"] = np.nan

        try:
            house_balance_point = get_balance_points(filename, balance_point_df)
            row["balance_RK"] = house_balance_point["slope"]
            row["balance_stderr"] = house_balance_point["stderr"]
            row["balance_rvalue"] = house_balance_point["r_value"]
            row["balance_p_value"] = house_balance_point["p_value"]
        except:
            row["balance_RK"] = np.nan()
            row["balance_stderr"] = np.nan()
            row["balance_rvalue"] = np.nan()
            row["balance_p_value"] = np.nan()

        row_df = pd.DataFrame(row)
        summary_dfs.append(row_df)

    summary_df = pd.concat(summary_dfs)
    return summary_df


def plot_correlations(x_val, y_val):
    plt.figure()
    ax = plt.axes()

    ax.scatter(x_val, y_val)
    x = np.linspace(0, 170, 1000)
    ax.plot(x, x, color="red")
    ax.set_title("Eulers Method vs Balance Point Plot")
    ax.set_xlabel("RK From Blance Point Plot")
    ax.set_ylabel("RK From Eulers Method")

    save_figure(plot_path, "bp_rks_vs_euler_rks")


def get_paths(data_dir):
    decay_curve_path = os.path.join(pp.TIME_CONSTANT_DIR, "decay_curves")
    model_fitting_path = os.path.join(
        pp.TIME_CONSTANT_DIR, "model_fitting"
    )  # model fitting = euler's method
    balance_point_file = os.path.join(pp.TIME_CONSTANT_DIR, "balance_point.csv")
    metadata_file = os.path.join(pp.RAW_DATA_DIR, "metadata.csv")
    plot_path = PLOT_DIR  # os.path.join(PLOT_PATH, data_dir)
    return (
        decay_curve_path,
        model_fitting_path,
        balance_point_file,
        metadata_file,
        plot_path,
    )


def get_balance_point_df(balance_point_file):
    balance_point_df = pd.read_csv(balance_point_file)
    # change slope to match euler's method
    balance_point_df["slope"] = -1 / balance_point_df["slope"]

    # TODO - may only want to look at balance point plots that are well correlated. Add check for this.
    balance_point_df = balance_point_df[balance_point_df["r_value"] < -0.5]
    return balance_point_df


def run_analysis(data_dir):
    # set filepaths
    (
        decay_curve_path,
        model_fitting_path,
        balance_point_file,
        metadata_file,
        plot_path,
    ) = get_paths(data_dir)

    # set dataframes
    metadata_df = pd.read_csv(metadata_file)
    balance_point_df = get_balance_point_df(balance_point_file)

    # get filenames
    filenames = get_filenames([decay_curve_path, model_fitting_path])

    # TODO - need to make a validation set to avoid over fitting.

    summary_df = create_summary_stats(
        filenames,
        model_fitting_path,
        decay_curve_path,
        balance_point_df,
        drop_outliers=True,
    )

    # plot_correlations(summary_df['balance_RK'], summary_df['euler_RK_mean'])
    plot_path = PLOT_DIR
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    # print(summary_df_no_outliers['euler_cost_mean'])
    summary_df.to_csv(os.path.join(plot_path, "summary_no_outliers.csv"))

    print("Done. Results saved to", plot_path)


if __name__ == "__main__":
    """ Run the analysis for specified results 
    
    sys.argv[1]: The name of the data to analyze,  ex.  
    """
    data_dir = sys.argv[1]
    run_analysis(data_dir)
