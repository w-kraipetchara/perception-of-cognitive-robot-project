"""
epuck_slam_controller.py
========================
E-Puck SLAM controller with:
  - WASD manual control
  - World-frame landmark matching (fixes scattered landmarks)
  - Loop closure (snaps path when robot revisits known area)
  - Pre-allocated GraphSLAM matrix (no expand corruption)
  - Correct xi sign convention
"""

from controller import Robot, Keyboard
import numpy as np
import math
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════

MAX_POSES       = 500    # max pose slots
NUM_LANDMARKS   = 20     # more slots — world-frame matching needs room
ODOM_SIGMA      = 0.05   # odometry noise
SENSOR_SIGMA    = 0.1    # lidar noise
UPDATE_DIST     = 0.05   # meters — new pose every 5cm

LIDAR_MAX       = 2.5
LIDAR_MIN       = 0.1
CLUSTER_DIST    = 0.2
MIN_CLUSTER_PTS = 3

# Loop closure config
LOOP_CLOSE_DIST  = 0.15  # meters — if SLAM pose is within this of a
                          # previous pose, trigger loop closure
LOOP_CLOSE_SIGMA = 0.01  # very tight constraint for loop closure

# Landmark matching config
LM_MATCH_DIST   = 0.3    # meters — world-frame distance to match
                          # a new observation to an existing landmark


# ══════════════════════════════════════════════════════════════════
# GraphSLAM Engine
# ══════════════════════════════════════════════════════════════════

class GraphSLAM:
    def __init__(self, max_poses, num_landmarks):
        self.max_poses     = max_poses
        self.num_landmarks = num_landmarks
        self.num_poses     = 1

        self.dim   = (max_poses + num_landmarks) * 2
        self.Omega = np.zeros((self.dim, self.dim))
        self.xi    = np.zeros((self.dim, 1))

        # Anchor pose 0 at origin
        for i in range(2):
            self.Omega[i, i] += 1e6
            self.xi[i, 0]    += 0.0

    def pose_idx(self, t):
        return t

    def landmark_idx(self, lm_id):
        return self.max_poses + lm_id

    def add_pose(self):
        if self.num_poses < self.max_poses:
            self.num_poses += 1

    def add_constraint(self, idx1, idx2, vec, sigma):
        weight = 1.0 / (sigma + 1e-9)
        for k in range(2):
            r1 = idx1 * 2 + k
            r2 = idx2 * 2 + k
            self.Omega[r1, r1] += weight
            self.Omega[r2, r2] += weight
            self.Omega[r1, r2] -= weight
            self.Omega[r2, r1] -= weight
            self.xi[r1, 0]     += vec[k] * weight
            self.xi[r2, 0]     -= vec[k] * weight

    def solve(self):
        pose_rows = list(range(self.num_poses * 2))
        lm_start  = self.max_poses * 2
        lm_rows   = list(range(lm_start, lm_start + self.num_landmarks * 2))
        idx       = pose_rows + lm_rows

        O_sub = self.Omega[np.ix_(idx, idx)]
        x_sub = self.xi[idx]

        try:
            mu = np.linalg.solve(O_sub, x_sub)
        except np.linalg.LinAlgError:
            mu = np.linalg.lstsq(O_sub, x_sub, rcond=None)[0]

        return mu.reshape(-1, 2)


# ══════════════════════════════════════════════════════════════════
# Landmark Registry — World-Frame Matching
# ══════════════════════════════════════════════════════════════════

class LandmarkRegistry:
    """
    Maintains a persistent map of landmarks in world frame.
    When a new observation arrives, checks if it matches an
    existing landmark (within LM_MATCH_DIST). If yes — reuses
    that landmark's ID. If no — registers a new one.

    This is what fixes the scattered landmark problem.
    Previously, angle-based IDs meant the same wall corner
    got a different ID every time the robot faced a different way.
    """

    def __init__(self, max_landmarks):
        self.max_landmarks = max_landmarks
        self.landmarks     = {}   # {lm_id: [wx, wy]}
        self.next_id       = 0

    def match_or_register(self, wx, wy):
        """
        Returns (lm_id, is_new).
        Matches to existing landmark if within LM_MATCH_DIST,
        otherwise creates a new one.
        """
        best_id   = None
        best_dist = float('inf')

        for lm_id, (lx, ly) in self.landmarks.items():
            d = math.sqrt((wx - lx)**2 + (wy - ly)**2)
            if d < best_dist:
                best_dist = d
                best_id   = lm_id

        if best_dist < LM_MATCH_DIST:
            # Update position with running average for stability
            old = self.landmarks[best_id]
            self.landmarks[best_id] = [
                (old[0] + wx) / 2.0,
                (old[1] + wy) / 2.0
            ]
            return best_id, False

        # New landmark
        if self.next_id >= self.max_landmarks:
            # Registry full — match to closest even if far
            return best_id if best_id is not None else 0, False

        lm_id = self.next_id
        self.landmarks[lm_id] = [wx, wy]
        self.next_id += 1
        return lm_id, True


# ══════════════════════════════════════════════════════════════════
# Loop Closure Detector
# ══════════════════════════════════════════════════════════════════

class LoopClosureDetector:
    """
    Checks if the current robot pose is close to a previously
    visited pose. If yes, adds a tight constraint between them
    in the SLAM graph — this is loop closure.

    Prevents drift from accumulating over long runs.
    Only checks poses that are old enough (MIN_POSE_GAP) to
    avoid false closures on consecutive steps.
    """

    def __init__(self, min_pose_gap=20):
        self.pose_history = []    # list of [x, y] at each SLAM update
        self.min_pose_gap = min_pose_gap
        self.closures     = 0

    def add_pose(self, x, y):
        self.pose_history.append([x, y])

    def check(self, slam, curr_pose_idx, x, y):
        """
        Compare current position against all old poses.
        Returns True if loop closure was added.
        """
        n = len(self.pose_history)
        if n < self.min_pose_gap:
            return False

        closed = False
        # Only check poses old enough to be a real loop
        for i, (px, py) in enumerate(self.pose_history[:-self.min_pose_gap]):
            dist = math.sqrt((x - px)**2 + (y - py)**2)
            if dist < LOOP_CLOSE_DIST:
                # Found a loop — add tight constraint
                slam.add_constraint(
                    slam.pose_idx(i),
                    slam.pose_idx(curr_pose_idx),
                    [0.0, 0.0],    # these poses should be at same location
                    LOOP_CLOSE_SIGMA
                )
                print(f"[LOOP] Closure: pose {curr_pose_idx} → pose {i} "
                      f"(dist {dist:.3f}m)")
                self.closures += 1
                closed = True
                break   # one closure per step is enough

        return closed


# ══════════════════════════════════════════════════════════════════
# Lidar Landmark Extraction
# ══════════════════════════════════════════════════════════════════

def extract_landmarks(lidar, robot_x, robot_y, robot_theta):
    pc = lidar.getPointCloud()
    if not pc:
        return []

    pts = []
    for p in pc:
        dist = math.sqrt(p.x**2 + p.y**2)
        if LIDAR_MIN < dist < LIDAR_MAX:
            pts.append([p.x, p.y])

    if len(pts) < MIN_CLUSTER_PTS:
        return []

    pts      = np.array(pts)
    assigned = [False] * len(pts)
    clusters = []

    for i in range(len(pts)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(pts)):
            if not assigned[j]:
                if np.linalg.norm(pts[i] - pts[j]) < CLUSTER_DIST:
                    cluster.append(j)
                    assigned[j] = True
        if len(cluster) >= MIN_CLUSTER_PTS:
            clusters.append(cluster)

    if not clusters:
        return []

    landmarks = []
    for cluster in clusters:
        rx, ry = np.mean(pts[cluster], axis=0)

        # Transform to world frame
        wx = robot_x + rx * math.cos(robot_theta) - ry * math.sin(robot_theta)
        wy = robot_y + rx * math.sin(robot_theta) + ry * math.cos(robot_theta)

        landmarks.append({
            'dx': rx, 'dy': ry,   # robot frame — for constraint vector
            'wx': wx, 'wy': wy    # world frame — for matching
        })

    return landmarks


# ══════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════

robot    = Robot()
timestep = int(robot.getBasicTimeStep())
keyboard = Keyboard()
keyboard.enable(timestep)

left_motor  = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

left_ps  = robot.getDevice('left wheel sensor')
right_ps = robot.getDevice('right wheel sensor')
left_ps.enable(timestep)
right_ps.enable(timestep)

lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()

# SLAM + helpers
slam     = GraphSLAM(max_poses=MAX_POSES, num_landmarks=NUM_LANDMARKS)
registry = LandmarkRegistry(max_landmarks=NUM_LANDMARKS)
looper   = LoopClosureDetector(min_pose_gap=20)

# Robot state
x, y, theta      = 0.0, 0.0, 0.0
last_l, last_r   = 0.0, 0.0
accumulated_dist = 0.0
dr_path          = [[0.0, 0.0]]

# Live map
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('#1a1a2e')

print("=" * 50)
print("  E-Puck SLAM  —  WASD + Loop Closure")
print("  W=forward  S=back  A=left  D=right")
print("  Drive around the box and come back")
print("  to the start to trigger loop closure!")
print("=" * 50)


# ══════════════════════════════════════════════════════════════════
# Main Loop
# ══════════════════════════════════════════════════════════════════

while robot.step(timestep) != -1:

    # 1. Odometry
    cl = left_ps.getValue()
    cr = right_ps.getValue()
    dl = (cl - last_l) * 0.0205
    dr = (cr - last_r) * 0.0205
    last_l, last_r = cl, cr

    dist   = (dl + dr) / 2.0
    dtheta = (dr - dl) / 0.052
    dx     = dist * math.cos(theta)
    dy     = dist * math.sin(theta)

    x     += dx
    y     += dy
    theta += dtheta
    dr_path.append([x, y])
    accumulated_dist += abs(dist)

    # 2. SLAM update every UPDATE_DIST meters
    if accumulated_dist > UPDATE_DIST and slam.num_poses < MAX_POSES:
        slam.add_pose()
        curr = slam.num_poses - 1

        # Odometry constraint
        slam.add_constraint(
            slam.pose_idx(curr - 1),
            slam.pose_idx(curr),
            [dx, dy],
            ODOM_SIGMA
        )

        # Record pose for loop closure
        looper.add_pose(x, y)

        # Loop closure check
        looper.check(slam, curr, x, y)

        # Landmark constraints with world-frame matching
        for lm in extract_landmarks(lidar, x, y, theta):
            lm_id, is_new = registry.match_or_register(lm['wx'], lm['wy'])

            if is_new:
                print(f"[MAP] New landmark LM{lm_id} at "
                      f"({lm['wx']:.3f}, {lm['wy']:.3f})")

            slam.add_constraint(
                slam.pose_idx(curr),
                slam.landmark_idx(lm_id),
                [lm['dx'], lm['dy']],
                SENSOR_SIGMA
            )

        accumulated_dist = 0.0

        # 3. Solve and draw
        mu        = slam.solve()
        slam_path = mu[:slam.num_poses]
        est_lms   = mu[slam.num_poses:]

        ax.clear()
        ax.set_facecolor('#16213e')
        ax.set_title(
            f'SLAM Map  |  Pose {curr}  |  '
            f'LM: {registry.next_id}  |  '
            f'Loops: {looper.closures}',
            color='white', fontsize=11
        )
        ax.set_xlabel('X (m)', color='grey')
        ax.set_ylabel('Y (m)', color='grey')
        ax.tick_params(colors='grey')
        ax.grid(True, color='#2a2a4a', linewidth=0.5)
        ax.set_aspect('equal')

        # Dead reckoning
        dr = np.array(dr_path)
        ax.plot(dr[:,0], dr[:,1], color='#ff4fc8', linewidth=1,
                linestyle='--', alpha=0.5, label='Dead Reckoning')

        # SLAM path
        ax.plot(slam_path[:,0], slam_path[:,1],
                color='#448aff', linewidth=2, label='SLAM Path')
        ax.scatter(*slam_path[0],  color='#69f0ae', s=150,
                   zorder=5, marker='*', label='Start')
        ax.scatter(*slam_path[-1], color='#ff6e40', s=150,
                   zorder=5, marker='D', label='Current')

        # Estimated landmarks (only registered ones)
        if registry.next_id > 0:
            lm_positions = np.array(list(registry.landmarks.values()))
            ax.scatter(lm_positions[:,0], lm_positions[:,1],
                       c='#ff1744', marker='x', s=120,
                       linewidths=2, zorder=6, label='Landmarks')
            for lm_id, (lx, ly) in registry.landmarks.items():
                ax.annotate(f' LM{lm_id}', (lx, ly),
                            color='#ff6e6e', fontsize=8)

        # Draw loop closure markers
        if looper.closures > 0:
            ax.scatter(*slam_path[0], color='#ffeb3b', s=200,
                       zorder=7, marker='o', alpha=0.5,
                       label='Loop Closure')

        ax.legend(loc='best', fontsize=8,
                  facecolor='#1a1a2e', labelcolor='white',
                  framealpha=0.8)
        plt.draw()
        plt.pause(0.001)

    # 4. WASD
    key    = keyboard.getKey()
    ls = rs = 0.0
    if   key == ord('W'): ls = rs =  5.0
    elif key == ord('S'): ls = rs = -5.0
    elif key == ord('A'): ls, rs = -2.5,  2.5
    elif key == ord('D'): ls, rs =  2.5, -2.5

    left_motor.setVelocity(ls)
    right_motor.setVelocity(rs)