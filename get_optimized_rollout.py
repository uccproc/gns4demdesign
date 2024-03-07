import numpy as np
from scipy.optimize import minimize
from learnedsimulator_calling_2 import learnedsimulator_infer
import numpy as np
import torch

initial_design_paramDict = {'friction':0.55, 'restitution':0.73}
initial_design_params = list(initial_design_paramDict.values())

def f_M(design_params):
    simulator2infer = learnedsimulator_infer(design_params)
    return simulator2infer

result = f_M(initial_design_params).infer2()

print(result)


