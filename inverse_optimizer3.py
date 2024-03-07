import numpy as np
from scipy.optimize import minimize
from learnedsimulator_calling_2 import learnedsimulator_infer
import numpy as np
import torch
import pandas as pd
from scipy.spatial.distance import cdist

initial_design_paramDict = {'friction':0.31, "restitution":0.31}
initial_design_params = list(initial_design_paramDict.values())
moments = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
target_runouts = np.array([0.3, 1.2, 2.7, 4.2, 5.2, 6.0, 6.0]) # This is for quartz
# target_runouts = np.array([0.3, 1.0, 2.4, 3.6, 4.7, 5.2, 5.2]) # This is for copper

def f_M(design_params):
    simulator2infer = learnedsimulator_infer(design_params)
    return simulator2infer

def find_leading_cluster2D(df, threshold_distance=0.001, cluster_size=8):
    # Sort the dataframe by 'X' to start with the leftmost (leading) particles
    df_sorted = df.sort_values(by='X')

    # Iterate over the sorted dataframe
    for index, particle in df_sorted.iterrows():
        # Compute distances from current particle to all others
        distances = cdist(df_sorted[['X', 'Y']], [particle[['X', 'Y']]])

        # Find particles within the threshold distance
        nearby_particles = df_sorted[(distances <= threshold_distance).flatten()]

        # Check if the nearby particles form a cluster of the desired size
        if len(nearby_particles) >= cluster_size:
            # Return the 'X' position of the first particle in the cluster
            return particle['X']

    # If no cluster meets the criteria, return None
    return None
def f_R(output_dict, target_runouts, moments):
    predicted_runouts = []
    predicted_rollout = output_dict['predicted_rollout']
    y_max = predicted_rollout[0, :, 1].max()
    x_min = predicted_rollout[0, :, 0].min()
    x_max = predicted_rollout[0, :, 0].max()
    # initial_data = pd.DataFrame(predicted_rollout[0, :, :], columns = ['X', 'Y'])
    # initial_leading_position = find_leading_cluster2(initial_data)
    initial_leading_position = 0.49
    initial_width = x_max - x_min
    initial_height = y_max
    t_star = np.sqrt(initial_height/9.81)
    # t_star = 0.078
    # print("t_star is : ", t_star)
    print("initial width is: ", initial_width)
    moment_indexlist = []
    for i in moments:
        moment_index_i = int(round(i*t_star/0.005, 0)) - 6
        moment_indexlist.append(moment_index_i)
    print(moment_indexlist)
    for i in moment_indexlist:
        data = predicted_rollout[i, :, :]
        df = pd.DataFrame(data, columns = ['X', 'Y'])
        leading_position = find_leading_cluster2D(df)
        if leading_position is not None:
            predicted_runouts.append({
                'ith moment': i,
                'predicted_runouts': (0.5 - leading_position) / (initial_width) - 1
            })
        else:
            print(f"No leading cluster found in {file}")
        # Loop through the histogram values

    leading_df = pd.DataFrame(predicted_runouts)
    print(f"The predicted runouts are:  {leading_df['predicted_runouts'].values}")
    print(f"The target runouts are:  {target_runouts}")

    loss = np.mean((target_runouts - leading_df['predicted_runouts']) ** 2)
    print(f"The current loss are:  {loss}")
    return loss  # Negative because we're using a minimizer

def objective_function(design_params):
    result = f_M(design_params).infer2()
    if result is not None:
        output_dict, _ = result
    else:
        print("f_M(design_params).infer() returned None")
        # Handle this case appropriately
    reward = f_R(output_dict, target_runouts, moments)
    return reward

# def callback(intermediate_result):
#     print(f'Current friction value is: {intermediate_result.x}')
def callback(xk):
    print(f'Current parameters: {xk}')

# bounds = [(0, 1), (0, 1), (0, 1)]
bounds = [(0.3, 0.9), (0.3, 0.9)]
# result = minimize(objective_function, initial_design_params, method='Powell', bounds=bounds, options={'xtol': 1e-2, 'maxiter': 20}, callback=callback)
result = minimize(objective_function, initial_design_params, method='L-BFGS-B', bounds=bounds, options={'xtol': 1e-2, 'maxiter': 20}, callback=callback)
# result = minimize(objective_function, initial_design_params, method='Nelder-Mead', bounds=bounds, options={'xtol': 1e-2, 'maxiter': 50}, callback=callback)

# Optimized design parameters
optimized_design_params = result.x

# Final value of the objective function (should be minimized)
final_loss = result.fun

print("Optimized Design Parameters:", optimized_design_params)
print("Final Objective Value:", final_loss)
