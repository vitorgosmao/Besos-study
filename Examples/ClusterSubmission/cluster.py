import pickle
from besos import eppy_funcs as ef
import besos.sampling as sampling
from besos.problem import EPProblem
from besos.evaluator import EvaluatorEP
from besos.parameters import (
    wwr,
    RangeParameter,
    FieldSelector,
    FilterSelector,
    GenericSelector,
    Parameter,
    expand_plist,
)

import time
from parameter_sets import (
    parameter_set,
)  # parameter_sets is another .py file containing parameter sets

parameters = parameter_set(4)
problem = EPProblem(parameters, ["Electricity:Facility"])
building = ef.get_building()

samples = sampling.dist_sampler(
    sampling.lhs, problem, num_samples=5, criterion="maximin"
)

now = time.time()
evaluator = EvaluatorEP(problem, building, multi=False)
outputs = evaluator.df_apply(samples, keep_input=True)
passedtime = round(time.time() - now, 2)

timestr = (
    "Time to evaluate "
    + str(len(samples))
    + " samples: "
    + str(passedtime)
    + " seconds."
)

with open("time.cluster", "wb") as timecluster:
    pickle.dump(timestr, timecluster)
with open("op.out", "wb") as op:
    pickle.dump(outputs, op)

timestr
