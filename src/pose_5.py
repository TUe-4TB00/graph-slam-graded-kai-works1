import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()
    #print("\nFinal Result:\n{}".format(result))

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals
    best_pose = None
    best_landmark = None
    best_selection_score = np.inf
    best_full_score = None

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            temp_graph = gtsam.NonlinearFactorGraph(graph)
            temp_estimate = gtsam.Values(initial_estimate)

            temp_graph, temp_estimate = add_pose(temp_graph, temp_estimate, pose_5)

            result1 = optimize(temp_graph, temp_estimate)

            temp_graph = add_landmark_measurement(temp_graph, result1, pose_5, landmark)

            result2 = optimize(temp_graph, temp_estimate)

            #TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
            marginals = gtsam.Marginals(temp_graph, result2)

            selection_score = marginals.marginalCovariance(L(landmark)).sum() # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()

            full_score = (marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum())

            if selection_score < best_selection_score:
                best_selection_score = selection_score
                best_pose = pose_name                  # chosen pose option
                best_landmark = landmark               # chosen landmark (1 or 2)
                best_full_score = full_score

    return best_pose, best_landmark, best_full_score

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None
    best_landmark = None
    best_error = np.inf

    ground_truth = {1: np.array([0.0, 0.0]), 2: np.array([2.0, 0.0]), 3: np.array([4.0, 0.0])}

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            temp_graph = gtsam.NonlinearFactorGraph(graph)
            temp_estimate = gtsam.Values(initial_estimate)

            temp_graph, temp_estimate = add_pose(temp_graph, temp_estimate, pose_5)

            result1 = optimize(temp_graph, temp_estimate)

            temp_graph = add_landmark_measurement(temp_graph, result1, pose_5, landmark)

            result2 = optimize(temp_graph, temp_estimate)

            poses = gtsam.utilities.allPose2s(result2)

            #TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
            list_of_errors = []

            for i in [1, 2, 3]:
                pose = poses.atPose2(X(i))

                estimated_position = np.array([pose.x(), pose.y()])

                error = np.linalg.norm(estimated_position - ground_truth[i])

                list_of_errors.append(error)

            #TODO: compute the sum of the errors and return it along with the best pose and landmark
            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < best_error:
                best_error = sum_of_errors
                best_pose = pose_name      # chosen pose option
                best_landmark = landmark   # chosen landmark (1 or 2)

    return best_pose, best_landmark, best_error