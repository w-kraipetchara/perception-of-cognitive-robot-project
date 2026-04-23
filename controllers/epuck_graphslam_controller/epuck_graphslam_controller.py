from controller import Robot, Keyboard
import numpy as np
import math
import matplotlib.pyplot as plt

from graphslam_engine import Pose2D, GraphBuilder, SLAMVisualizer, GraphPruner, SparseGraphOptimizer
from occupancy_grid import OccupancyGrid, FrontierExplorer
from global_planner import GlobalPlanner

# ---------------------------------------------------------------------------
# Robot physical constants
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.0205
WHEEL_BASE   = 0.0585
MAX_SPEED    = 4.0

# ---------------------------------------------------------------------------
# Navigation tuning constants
# ---------------------------------------------------------------------------
WAYPOINT_REACH_DIST  = 0.08   # metres — how close counts as "reached"
REPLAN_INTERVAL      = 40     # steps between full replans
HEADING_KP           = 4.0    # proportional gain for heading correction
BASE_SPEED           = 2.0    # straight-line cruise speed (rad/s)
MAX_TURN_CORRECTION  = 2.5    # max speed delta applied by heading controller


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------

def extract_multiple_landmarks(valid_ranges, valid_angles, current_pose):
    """
    Groups adjacent lidar hits into clusters (objects).
    Returns a list of global (x, y) coordinates for each cluster centre.
    """
    if len(valid_ranges) == 0:
        return []

    LM_MIN_RANGE     = 0.05
    LM_MAX_RANGE     = 2.5
    LM_CLUSTER_GAP   = 0.15
    LM_MIN_CLUSTER_PTS = 2

    xs = valid_ranges * np.cos(valid_angles)
    ys = valid_ranges * np.sin(valid_angles)

    clusters = []
    current_cluster = []

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
            gap = math.hypot(xs[i] - xs[current_cluster[-1]],
                             ys[i] - ys[current_cluster[-1]])
            if gap <= LM_CLUSTER_GAP:
                current_cluster.append(i)
            else:
                clusters.append(current_cluster)
                current_cluster = [i]

    if current_cluster:
        clusters.append(current_cluster)

    landmarks = []
    for cl in clusters:
        if len(cl) >= LM_MIN_CLUSTER_PTS:
            cx = float(np.mean(xs[cl]))
            cy = float(np.mean(ys[cl]))
            gx = current_pose.x + cx * math.cos(current_pose.theta) - cy * math.sin(current_pose.theta)
            gy = current_pose.y + cx * math.sin(current_pose.theta) + cy * math.cos(current_pose.theta)
            landmarks.append((gx, gy))

    return landmarks


# ---------------------------------------------------------------------------
# Waypoint follower (local planner — pure pursuit heading controller)
# ---------------------------------------------------------------------------

def compute_wheel_speeds(current_pose, waypoints, wp_index):
    """
    Given the current pose and the active waypoint list, compute
    (left_speed, right_speed, updated_wp_index).

    Strategy:
      1. Check if the current waypoint has been reached → advance index.
      2. Compute heading error to the current waypoint.
      3. Convert heading error to a differential speed correction.

    Returns (left_speed, right_speed, wp_index).
    Returns (0, 0, wp_index) if waypoints are exhausted.
    """
    if not waypoints or wp_index >= len(waypoints):
        return 0.0, 0.0, wp_index

    # Advance past any already-reached waypoints
    while wp_index < len(waypoints):
        tx, ty = waypoints[wp_index]
        dist = math.hypot(tx - current_pose.x, ty - current_pose.y)
        if dist < WAYPOINT_REACH_DIST:
            wp_index += 1
        else:
            break

    if wp_index >= len(waypoints):
        return 0.0, 0.0, wp_index

    tx, ty = waypoints[wp_index]

    # Desired heading angle
    desired_heading = math.atan2(ty - current_pose.y, tx - current_pose.x)

    # Heading error, normalised to [-pi, pi]
    heading_error = desired_heading - current_pose.theta
    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

    # Proportional correction
    correction = HEADING_KP * heading_error
    correction = max(-MAX_TURN_CORRECTION, min(MAX_TURN_CORRECTION, correction))

    # If pointing far away from target, turn on the spot instead of drifting
    if abs(heading_error) > math.pi / 2:
        left_speed  = -correction
        right_speed =  correction
    else:
        left_speed  = BASE_SPEED - correction
        right_speed = BASE_SPEED + correction

    # Clamp to motor limits
    left_speed  = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

    return left_speed, right_speed, wp_index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    robot    = Robot()
    timestep = int(robot.getBasicTimeStep())

    # --- Keyboard ---
    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    # --- Motors ---
    left_motor  = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # --- Encoders ---
    left_sensor  = robot.getDevice('left wheel sensor')
    right_sensor = robot.getDevice('right wheel sensor')
    left_sensor.enable(timestep)
    right_sensor.enable(timestep)

    # --- Lidar ---
    lidar = robot.getDevice('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()
    fov            = lidar.getFov()
    horizontal_res = lidar.getHorizontalResolution()
    angles         = np.linspace(fov / 2.0, -fov / 2.0, horizontal_res)

    # --- SLAM engine ---
    graph_builder = GraphBuilder()
    visualizer    = SLAMVisualizer(graph_builder.pose_graph)

    current_pose       = Pose2D(x=0.0, y=0.0, theta=0.0)
    initial_uncertainty = np.eye(3) * 0.01
    graph_builder.add_odometry_measurement(current_pose, initial_uncertainty)

    prev_left_enc  = 0.0
    prev_right_enc = 0.0
    odom_uncertainty = np.diag([0.05, 0.05, 0.02])

    # --- Occupancy grid + navigation stack ---
    occ_grid = OccupancyGrid(width_m=10.0, height_m=10.0, resolution=0.05,
                             origin_x=-5.0, origin_y=-5.0)
    explorer  = FrontierExplorer(occ_grid)
    planner   = GlobalPlanner(occ_grid, inflation_radius=0.10)

    waypoints = []   # active path: list of (wx, wy)
    wp_index  = 0    # index of the waypoint we are currently heading to

    step_counter  = 0
    manual_override = False   # True while a keyboard key is held

    print("Autonomous mode active. Use W/A/S/D to take manual control at any time.")

    while robot.step(timestep) != -1:
        step_counter += 1

        # ----------------------------------------------------------------
        # 1. Keyboard — manual override (takes priority over autonomous nav)
        # ----------------------------------------------------------------
        key = keyboard.getKey()
        manual_left, manual_right = 0.0, 0.0

        if key == ord('W'):
            manual_left, manual_right = MAX_SPEED, MAX_SPEED
            manual_override = True
        elif key == ord('S'):
            manual_left, manual_right = -MAX_SPEED, -MAX_SPEED
            manual_override = True
        elif key == ord('A'):
            manual_left, manual_right = -MAX_SPEED / 2.0, MAX_SPEED / 2.0
            manual_override = True
        elif key == ord('D'):
            manual_left, manual_right = MAX_SPEED / 2.0, -MAX_SPEED / 2.0
            manual_override = True
        else:
            manual_override = False

        # ----------------------------------------------------------------
        # 2. Odometry update
        # ----------------------------------------------------------------
        left_enc  = left_sensor.getValue()
        right_enc = right_sensor.getValue()
        d_left  = (left_enc  - prev_left_enc)  * WHEEL_RADIUS
        d_right = (right_enc - prev_right_enc) * WHEEL_RADIUS
        prev_left_enc, prev_right_enc = left_enc, right_enc

        d_center = (d_right + d_left) / 2.0
        d_theta  = (d_right - d_left) / WHEEL_BASE

        current_pose.x     += d_center * math.cos(current_pose.theta + d_theta / 2.0)
        current_pose.y     += d_center * math.sin(current_pose.theta + d_theta / 2.0)
        current_pose.theta += d_theta

        # ----------------------------------------------------------------
        # 3. Periodic map update + replanning (every 20 steps)
        # ----------------------------------------------------------------
        if step_counter % 20 == 0:

            # Add pose to SLAM graph
            new_measurement = Pose2D(current_pose.x, current_pose.y, current_pose.theta)
            graph_builder.add_odometry_measurement(new_measurement, odom_uncertainty)

            # Lidar processing
            ranges        = np.array(lidar.getRangeImage())
            valid_indices = np.isfinite(ranges)
            valid_ranges  = ranges[valid_indices]
            valid_angles  = angles[valid_indices]

            # Update occupancy grid
            occ_grid.update(current_pose.x, current_pose.y,
                            valid_ranges, valid_angles, current_pose.theta)

            # Landmark extraction → SLAM graph
            obs_uncertainty = np.diag([0.1, 0.1])
            landmark_list   = extract_multiple_landmarks(valid_ranges, valid_angles, current_pose)
            for lm_coords in landmark_list:
                graph_builder.process_landmark_observation(
                    global_x=lm_coords[0],
                    global_y=lm_coords[1],
                    uncertainty=obs_uncertainty
                )

            # Replan path periodically or when waypoints are exhausted
            if step_counter % REPLAN_INTERVAL == 0 or wp_index >= len(waypoints):
                frontier_goal = explorer.get_best_goal(current_pose.x, current_pose.y)

                if frontier_goal:
                    print(f"Frontier goal → ({frontier_goal[0]:.2f}, {frontier_goal[1]:.2f})")

                new_path = planner.plan(current_pose.x, current_pose.y,
                                        frontier_goal=frontier_goal)
                if new_path:
                    waypoints = new_path
                    wp_index  = 0
                    print(f"New path: {len(waypoints)} waypoints")
                else:
                    print("No path found — holding position.")

            # Graph pruning
            if step_counter % 60 == 0:
                pruner = GraphPruner(graph_builder.pose_graph)
                pruner.prune_unobserved_poses()

            # Visualisation
            scan_x = current_pose.x + valid_ranges * np.cos(current_pose.theta + valid_angles)
            scan_y = current_pose.y + valid_ranges * np.sin(current_pose.theta + valid_angles)
            visualizer.visualize_graph()
            visualizer.ax.scatter(scan_x, scan_y, c='r', s=2, label='Current Scan')

            # Draw active waypoints on the map
            if waypoints and wp_index < len(waypoints):
                future = waypoints[wp_index:]
                wx_list = [p[0] for p in future]
                wy_list = [p[1] for p in future]
                visualizer.ax.plot(wx_list, wy_list, 'c--', linewidth=1.0, label='Planned path')
                visualizer.ax.plot(*waypoints[wp_index], 'go', markersize=6, label='Next waypoint')

            plt.pause(0.01)

        # ----------------------------------------------------------------
        # 4. Motor commands — manual overrides autonomous
        # ----------------------------------------------------------------
        if manual_override:
            left_motor.setVelocity(manual_left)
            right_motor.setVelocity(manual_right)
            # Reset path so robot replans from scratch when released
            waypoints = []
            wp_index  = 0
        else:
            # Autonomous waypoint following
            auto_left, auto_right, wp_index = compute_wheel_speeds(
                current_pose, waypoints, wp_index
            )
            left_motor.setVelocity(auto_left)
            right_motor.setVelocity(auto_right)


if __name__ == "__main__":
    main()