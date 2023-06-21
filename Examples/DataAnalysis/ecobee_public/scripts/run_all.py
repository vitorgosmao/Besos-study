import os
import pandas as pd
import traceback


import scripts.preprocessing as pp
import scripts.balance_point as bp
import scripts.decay_curves as dc
import scripts.model_fitting as mf
from scripts.filters import *


def run_all_methods():
    balance_point_results = []
    selected_df = pp.choose_metadata(no_heat_pump, metadata_style)

    if not os.path.isdir(pp.NEW_DATA_DIR):
        os.mkdir(pp.NEW_DATA_DIR)

    i = 0
    for row in selected_df.iterrows():
        filename = row[1].filename
        try:
            building_df = pp.load_building_data(filename)
        except (FileNotFoundError) as error:
            continue

        # Skip buildings with particular conditions
        if only_auxHeat1:
            building_df["auxHeat2"] = building_df.auxHeat2.fillna(0.0)
            if not (building_df["auxHeat2"] == 0.0).all():
                continue

        if no_smart_recovery:
            try:
                if (building_df["Event"] == "Smart Recovery").any():
                    print("skipping from smart recovery")
                    continue
            except:
                continue

        # Analyze balance points
        try:
            result = bp.analyze_building_balance_point(
                months, hours, filename, building_df, reject_outliers
            )
            balance_point_results.append(result)
        except Exception as e:
            print(e)
            # print(traceback.format_exc())
            pass

        # analyze model fitting
        try:
            mf.analyze_building_model_fitting(
                months,
                hours,
                filename,
                building_df,
                model_fitting_params,
                cost_threshold,
            )
        except Exception as e:
            print(e)
            # traceback.print_exc();
            pass

        # analyze decay curves
        try:
            dc.analyze_building_decay_curves(
                months,
                hours,
                duration,
                filename,
                building_df,
                decay_curve_params,
                initial_guess,
            )
        except Exception as e:
            print(e)
            # print(traceback.format_exc())
            pass

    # write results from balance point to CSV
    balance_point_results_df = pd.DataFrame(balance_point_results)
    balance_point_results_df.to_csv(
        os.path.join(pp.NEW_DATA_DIR, "balance_point.csv"), index=False
    )

    path = os.path.join(pp.NEW_DATA_DIR, "params.csv")
    params_df = pd.DataFrame(params_dict)
    params_df.to_csv(path, index=False)
    print("Done. See results in", pp.NEW_DATA_DIR)
    return pp.NEW_DATA_DIR
