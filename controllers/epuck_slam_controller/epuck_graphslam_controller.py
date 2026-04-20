from controller import Robot, Keyboard
import numpy as np
import math
import matplotlib.pyplot as plt

from graphslam_engine import Pose2D, GraphBuilder, SLAMVisualizer, GraphPruner, SparseGraphOptimizer

WHEEL_RADIUS = 0.0205  
WHEEL_BASE = 0.0585     
MAX_SPEED = 4

def extract_multiple_landmarks(valid_ranges, valid_angles, current_pose):
    """
    Groups adjacent Lidar hits into distinct clusters (objects).
    Returns a list of global (X, Y) coordinates for the center of each object.
    """
    if len(valid_ranges) == 0:
        return []

    # Constants for clustering
    LM_MIN_RANGE = 0.05
    LM_MAX_RANGE = 2.5
    LM_CLUSTER_GAP = 0.15 # Max distance between points to be considered the same object
    LM_MIN_CLUSTER_PTS = 2
    
    # Calculate local coordinates for all valid hits
    xs = valid_ranges * np.cos(valid_angles)
    ys = valid_ranges * np.sin(valid_angles)
    
    clusters = []
    current_cluster = []

    # Group points into clusters based on distance gaps
    for i in range(len(valid_ranges)):
        r = valid_ranges[i]
        if not (LM_MIN_RANGE <= r <= LM_MAX_RANGE):
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
            continue
            
        if not current_cluster:
            current_cluster.append(i)
        else:
            # Check the distance between this point and the last point in the cluster
            gap = math.hypot(xs[i] - xs[current_cluster[-1]], ys[i] - ys[current_cluster[-1]])
            if gap <= LM_CLUSTER_GAP:
                current_cluster.append(i)
            else:
                clusters.append(current_cluster)
                current_cluster = [i]
                
    if current_cluster:
        clusters.append(current_cluster)

    # Calculate the global center of each valid cluster
    landmarks = []
    for cl in clusters:
        if len(cl) >= LM_MIN_CLUSTER_PTS:
            # Get the center in local coordinates
            cx = float(np.mean(xs[cl]))
            cy = float(np.mean(ys[cl]))
            
            # Convert local to global coordinates
            global_x = current_pose.x + cx * math.cos(current_pose.theta) - cy * math.sin(current_pose.theta)
            global_y = current_pose.y + cx * math.sin(current_pose.theta) + cy * math.cos(current_pose.theta)
            
            landmarks.append((global_x, global_y))
            
    return landmarks

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # --- Keyboard Setup ---
    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    # --- Motor Setup ---
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # --- Encoder Setup ---
    left_sensor = robot.getDevice('left wheel sensor')
    right_sensor = robot.getDevice('right wheel sensor')
    left_sensor.enable(timestep)
    right_sensor.enable(timestep)

    # --- Lidar Setup ---
    lidar = robot.getDevice('lidar') 
    lidar.enable(timestep)
    lidar.enablePointCloud()
    fov = lidar.getFov()
    horizontal_res = lidar.getHorizontalResolution()
    angles = np.linspace(fov / 2.0, -fov / 2.0, horizontal_res)

    # --- Engine Setup ---
    graph_builder = GraphBuilder()
    visualizer = SLAMVisualizer(graph_builder.pose_graph)

    current_pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    initial_uncertainty = np.eye(3) * 0.01
    graph_builder.add_odometry_measurement(current_pose, initial_uncertainty)

    prev_left_enc = 0.0
    prev_right_enc = 0.0
    odom_uncertainty = np.diag([0.05, 0.05, 0.02]) 
    step_counter = 0

    print("Clean baseline loaded. Click inside the Webots 3D view and use W/A/S/D to drive.")

    while robot.step(timestep) != -1:
        step_counter += 1

        # 1. Keyboard Control
        key = keyboard.getKey()
        left_speed, right_speed = 0.0, 0.0
        if key == ord('W'):
            left_speed, right_speed = MAX_SPEED, MAX_SPEED
        elif key == ord('S'):
            left_speed, right_speed = -MAX_SPEED, -MAX_SPEED
        elif key == ord('A'):
            left_speed, right_speed = -MAX_SPEED / 2.0, MAX_SPEED / 2.0
        elif key == ord('D'):
            left_speed, right_speed = MAX_SPEED / 2.0, -MAX_SPEED / 2.0

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        # 2. Odometry Update
        left_enc = left_sensor.getValue()
        right_enc = right_sensor.getValue()
        d_left = (left_enc - prev_left_enc) * WHEEL_RADIUS
        d_right = (right_enc - prev_right_enc) * WHEEL_RADIUS
        prev_left_enc, prev_right_enc = left_enc, right_enc


        d_center = (d_right + d_left) / 2.0
        d_theta = (d_right - d_left) / WHEEL_BASE

        current_pose.x += d_center * math.cos(current_pose.theta + d_theta / 2.0)
        current_pose.y += d_center * math.sin(current_pose.theta + d_theta / 2.0)
        current_pose.theta += d_theta

        # Update map every 20 steps
        if step_counter % 20 == 0:
            new_measurement = Pose2D(current_pose.x, current_pose.y, current_pose.theta)
            graph_builder.add_odometry_measurement(new_measurement, odom_uncertainty)
            
            # 3. Lidar Processing
            ranges = np.array(lidar.getRangeImage())
            valid_indices = np.isfinite(ranges)
            valid_ranges = ranges[valid_indices]
            valid_angles = angles[valid_indices]

            scan_x = current_pose.x + valid_ranges * np.cos(current_pose.theta + valid_angles)
            scan_y = current_pose.y + valid_ranges * np.sin(current_pose.theta + valid_angles)

            obs_uncertainty = np.diag([0.1, 0.1]) 
            
            landmark_list = extract_multiple_landmarks(valid_ranges, valid_angles, current_pose)
            
            # Loop through every object we found in this single scan
            for lm_coords in landmark_list:
                matched_id = graph_builder.process_landmark_observation(
                    global_x=lm_coords[0],
                    global_y=lm_coords[1],
                    uncertainty=obs_uncertainty
                )
                
            # --- Prune and Optimize the Graph Periodically ---
            if step_counter % 60 == 0:
                # 1. Clean the graph
                pruner = GraphPruner(graph_builder.pose_graph)
                pruner.prune_unobserved_poses()
                
                """
                # 2. Relax the rubber bands (Optimize)
                optimizer = SparseGraphOptimizer(graph_builder.pose_graph)
                optimizer.optimize(max_iterations=5, tolerance=1e-4) 

                # --- NEW: 3. Sync the Controller to the Optimized Map ---
                # Fetch the newly corrected pose from the engine
                corrected_pose = graph_builder.pose_graph.nodes[graph_builder.current_pose_id]
                
                # Overwrite the controller's drifting dead-reckoning
                current_pose.x = corrected_pose.x
                current_pose.y = corrected_pose.y
                current_pose.theta = corrected_pose.theta
                print("Controller synchronized with optimized map!")
                """

            # 4. Visualization
            visualizer.visualize_graph()
            visualizer.ax.scatter(scan_x, scan_y, c='r', s=2, label='Current Scan') 
            plt.pause(0.01)

if __name__ == "__main__":
    main()