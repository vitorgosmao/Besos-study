# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## EvaluatorEH
#
# This notebook covers how to use the PyEHub Evaluator (`EvaluatorEH`).

from besos.evaluator import EvaluatorEH
from besos import eppy_funcs as ef
from besos import pyehub_funcs as pf
from besos.parameters import FieldSelector, Parameter
from besos.problem import EPProblem, Problem, EHProblem
import numpy as np
import pandas as pd
import sampling

# This evaluator needs an energy hub model, and a problem with parameters that can modify it, and objectives that correspond to outputs from the solution of the model.
# Parameters are provided as a list of key list mapping lists for the different variables inside the model.
# Outputs are provided as a list of the keys from the solution of the model.

# +
hub = pf.get_hub()

parameters = [["LOADS", "Elec"], ["LOADS", "Heat"]]
objectives = ["total_cost", "total_carbon"]
problem = EHProblem(parameters, objectives)
evaluatorEH = EvaluatorEH(problem, hub)
# -

# Input values for overwritting the specified parameters can be given in the form of single values, a dictionary time series, a dataframe of single values, or a dataframe of time series.

default_timeseries = [
    {
        0: 1.0,
        1: 4.0,
        2: 4.0,
        3: 4.0,
        4: 4.0,
        5: 4.0,
        6: 4.0,
        7: 4.0,
        8: 4.0,
        9: 4.0,
        10: 4.0,
    },
    {
        0: 20.0,
        1: 20.0,
        2: 20.0,
        3: 20.0,
        4: 20.0,
        5: 20.0,
        6: 20.0,
        7: 12.0,
        8: 12.0,
        9: 12.0,
        10: 12.0,
    },
]
modified_heat = [
    {
        0: 1.0,
        1: 4.0,
        2: 4.0,
        3: 4.0,
        4: 4.0,
        5: 4.0,
        6: 4.0,
        7: 4.0,
        8: 4.0,
        9: 4.0,
        10: 4.0,
    },
    {
        0: 18.0,
        1: 18.0,
        2: 18.0,
        3: 18.0,
        4: 18.0,
        5: 18.0,
        6: 18.0,
        7: 16.0,
        8: 16.0,
        9: 16.0,
        10: 16.0,
    },
]
modified_elec = [
    {
        0: 4.0,
        1: 8.0,
        2: 6.0,
        3: 5.0,
        4: 7.0,
        5: 7.0,
        6: 7.0,
        7: 7.0,
        8: 7.0,
        9: 7.0,
        10: 7.0,
    },
    {
        0: 20.0,
        1: 20.0,
        2: 20.0,
        3: 20.0,
        4: 20.0,
        5: 20.0,
        6: 20.0,
        7: 12.0,
        8: 12.0,
        9: 12.0,
        10: 12.0,
    },
]
modified_both = [
    {
        0: 4.0,
        1: 8.0,
        2: 6.0,
        3: 5.0,
        4: 7.0,
        5: 7.0,
        6: 7.0,
        7: 7.0,
        8: 7.0,
        9: 7.0,
        10: 7.0,
    },
    {
        0: 18.0,
        1: 18.0,
        2: 18.0,
        3: 18.0,
        4: 18.0,
        5: 18.0,
        6: 18.0,
        7: 16.0,
        8: 16.0,
        9: 16.0,
        10: 16.0,
    },
]
timeseries_df = pd.DataFrame(
    np.array([default_timeseries, modified_heat, modified_elec, modified_both]),
    columns=["p1", "p2"],
)

# Normally the evaluator can be called directly with the input values but if using a dataframe as input df_apply must be used.

result = evaluatorEH.df_apply(timeseries_df)
result
