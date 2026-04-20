# ── stdlib ────────────────────────────────────────────────────────────────────
import math
import time
import queue
import threading
import numpy as np

# matplotlib MUST be imported and configured before any other matplotlib call.
# "Agg" is a non-interactive, non-GUI backend that produces no windows by
# itself.  We then replace it on the main thread with the real backend after
# choosing one that matches the platform.
import matplotlib
# Defer backend selection to _choose_backend() called from main() on the
# main thread — never from a background thread.
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── WeBots ────────────────────────────────────────────────────────────────────
from controller import Robot, Keyboard, Lidar, Motor

# ── GraphSLAM engine ──────────────────────────────────────────────────────────
from graphslam_engine import (
    Pose2D,
    Edge,
    Measurement,
    LaserScan,
    PoseGraph,
    GraphBuilder,
    RobustKernel,
    SparseGraphOptimizer,
    LoopClosureDetector,
    SLAMVisualizer,
)

# ════════════════════════════════════════════════════════════════════════════
#  TUNEABLE CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

WHEEL_RADIUS  = 0.0205   # m  (e-puck standard)
AXLE_LENGTH   = 0.0585    # m
MAX_SPEED     = 4     # rad/s

LINEAR_SPEED  = 0.15     # m/s   (W / S)
ANGULAR_SPEED = 1.2      # rad/s (A / D)

LIDAR_DEVICE_NAME = "LDS-01"   # change to match your world
LIDAR_FREQUENCY   = 5.0        # Hz

ODOM_SIGMA = np.array([0.02, 0.02, 0.01])   # σ_x, σ_y, σ_θ

# Landmark extraction
LM_MIN_RANGE       = 0.05   # m
LM_MAX_RANGE       = 2.5    # m
LM_CLUSTER_GAP     = 0.15   # m
LM_MIN_CLUSTER_PTS = 2
LM_MAX_CLUSTER_PTS = 25

# Landmark association & confirmation
LANDMARK_ASSOC_DIST = 0.30   # m
LANDMARK_MIN_OBS    = 3
LM_OBS_SIGMA        = np.array([0.05, 0.05])

OPTIMISE_EVERY_N_STEPS = 20

MAP_REFRESH_HZ  = 4
MAP_REFRESH_SEC = 1.0 / MAP_REFRESH_HZ

LOOP_CLOSURE_DIST = 1.5
LOOP_CLOSURE_SIM  = 0.80


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _choose_backend() -> None:
    """
    Pick and activate a working interactive matplotlib backend.
    Must be called on the MAIN thread before any plt.* that opens a window.
    """
    for backend in ("MacOSX", "TkAgg", "Qt5Agg", "WXAgg", "Agg"):
        try:
            matplotlib.use(backend, force=True)
            import importlib
            importlib.reload(plt)          # re-bind pyplot to the new backend
            if backend != "Agg":
                print(f"[VIS] matplotlib backend: {backend}")
            return
        except Exception:
            continue


def _unicycle_to_wheels(linear_v: float, angular_v: float):
    left  = (linear_v - angular_v * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    right = (linear_v + angular_v * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    m = max(abs(left), abs(right))
    if m > MAX_SPEED:
        left  *= MAX_SPEED / m
        right *= MAX_SPEED / m
    return left, right


def _build_descriptor(ranges: np.ndarray, n_bins: int = 36) -> np.ndarray:
    n      = len(ranges)
    bins   = np.zeros(n_bins, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int32)
    for i, r in enumerate(ranges):
        if math.isfinite(r) and r > 0.0:
            idx = int(i / n * n_bins) % n_bins
            bins[idx]   += r
            counts[idx] += 1
    mask = counts > 0
    bins[mask] /= counts[mask]
    norm = np.linalg.norm(bins)
    return bins / norm if norm > 1e-9 else bins


def _descriptor_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def _project_scan(pose: Pose2D, scan: LaserScan) -> np.ndarray:
    ang = scan.angles + pose.theta
    return np.column_stack([pose.x + scan.ranges * np.cos(ang),
                            pose.y + scan.ranges * np.sin(ang)])


# ════════════════════════════════════════════════════════════════════════════
#  ODOMETRY INTEGRATOR
# ════════════════════════════════════════════════════════════════════════════

class OdometryIntegrator:
    def __init__(self):
        self.current_pose = Pose2D(0.0, 0.0, 0.0)
        self.last_delta   = Pose2D(0.0, 0.0, 0.0)
        self._prev_l = self._prev_r = None

    def update(self, lp: float, rp: float) -> None:
        if self._prev_l is None:
            self._prev_l, self._prev_r = lp, rp
            return
        dl = (lp - self._prev_l) * WHEEL_RADIUS
        dr = (rp - self._prev_r) * WHEEL_RADIUS
        self._prev_l, self._prev_r = lp, rp
        ds     = (dl + dr) / 2.0
        dtheta = (dr - dl) / AXLE_LENGTH
        dx     = ds * math.cos(self.current_pose.theta + dtheta / 2.0)
        dy     = ds * math.sin(self.current_pose.theta + dtheta / 2.0)
        self.last_delta   = Pose2D(dx, dy, dtheta)
        self.current_pose = Pose2D(
            self.current_pose.x     + dx,
            self.current_pose.y     + dy,
            self.current_pose.theta + dtheta,
        )


# ════════════════════════════════════════════════════════════════════════════
#  LANDMARK EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════

class LandmarkExtractor:
    """
    Detect point-like geometric features from a LaserScan by clustering
    adjacent scan points.  Returns (range_m, bearing_rad) in sensor frame.
    """

    def extract(self, scan: LaserScan) -> list[tuple[float, float]]:
        ranges, angles = scan.ranges, scan.angles
        n = len(ranges)
        if n == 0:
            return []

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        clusters: list[list[int]] = []
        cur: list[int] = []

        for i in range(n):
            r = ranges[i]
            if not (LM_MIN_RANGE <= r <= LM_MAX_RANGE):
                if cur:
                    clusters.append(cur); cur = []
                continue
            if not cur:
                cur.append(i)
            else:
                gap = math.hypot(xs[i] - xs[cur[-1]], ys[i] - ys[cur[-1]])
                if gap <= LM_CLUSTER_GAP:
                    cur.append(i)
                else:
                    clusters.append(cur); cur = [i]
        if cur:
            clusters.append(cur)

        obs = []
        for cl in clusters:
            if not (LM_MIN_CLUSTER_PTS <= len(cl) <= LM_MAX_CLUSTER_PTS):
                continue
            cx = float(np.mean(xs[cl]))
            cy = float(np.mean(ys[cl]))
            obs.append((math.hypot(cx, cy), math.atan2(cy, cx)))
        return obs


# ════════════════════════════════════════════════════════════════════════════
#  LANDMARK  (internal record)
# ════════════════════════════════════════════════════════════════════════════

class _Landmark:
    __slots__ = ("lm_id", "x", "y", "obs_count", "confirmed", "node_id")

    def __init__(self, lm_id: int, x: float, y: float, node_id: int):
        self.lm_id     = lm_id
        self.x         = x
        self.y         = y
        self.obs_count = 1
        self.confirmed = False
        self.node_id   = node_id

    def update_position(self, x: float, y: float) -> None:
        alpha  = 1.0 / self.obs_count
        self.x = (1.0 - alpha) * self.x + alpha * x
        self.y = (1.0 - alpha) * self.y + alpha * y


# ════════════════════════════════════════════════════════════════════════════
#  LANDMARK MAP
# ════════════════════════════════════════════════════════════════════════════

class LandmarkMap:
    """
    Data association, PoseGraph node management, and observation-edge
    creation via GraphBuilder.add_landmark_observation().
    """

    def __init__(self, pose_graph: PoseGraph, graph_builder: GraphBuilder):
        self._pg      = pose_graph
        self._gb      = graph_builder
        self._lms: list[_Landmark] = []
        self._next_id = 0

    @property
    def landmarks(self) -> list[_Landmark]:
        return self._lms

    def process_observations(self, robot_pose: Pose2D,
                              observations: list[tuple[float, float]]) -> None:
        obs_cov = np.diag(LM_OBS_SIGMA ** 2)

        for r_obs, b_obs in observations:
            # Sensor → world frame
            wa = robot_pose.theta + b_obs
            wx = robot_pose.x + r_obs * math.cos(wa)
            wy = robot_pose.y + r_obs * math.sin(wa)

            # Data association
            lm = self._associate(wx, wy)

            if lm is None:
                lm_node_id = self._pg.add_node(
                    Pose2D(wx, wy, 0.0),
                    np.diag([LM_OBS_SIGMA[0]**2, LM_OBS_SIGMA[1]**2, 1e-9]),
                )
                lm = _Landmark(self._next_id, wx, wy, node_id=lm_node_id)
                self._next_id += 1
                self._lms.append(lm)
            else:
                lm.obs_count += 1
                lm.update_position(wx, wy)
                self._pg.nodes[lm.node_id] = Pose2D(lm.x, lm.y, 0.0)
                if lm.obs_count >= LANDMARK_MIN_OBS:
                    lm.confirmed = True

            # Measurement in robot frame
            dx = wx - robot_pose.x
            dy = wy - robot_pose.y
            c  =  math.cos(robot_pose.theta)
            s  =  math.sin(robot_pose.theta)
            meas = Measurement(x=c*dx + s*dy, y=-s*dx + c*dy)

            # Observation edge via GraphBuilder
            try:
                self._gb.add_landmark_observation(
                    landmark_id = lm.node_id,
                    measurement = meas,
                    uncertainty = obs_cov,
                )
                if self._pg.edges:
                    self._pg.edges[-1].type = "observation"
            except TypeError:
                info = np.linalg.inv(obs_cov)
                self._pg.add_edge(
                    from_id     = self._gb.current_pose_id,
                    to_id       = lm.node_id,
                    measurement = meas,
                    information = info,
                )
                self._pg.edges[-1].type = "observation"

    def _associate(self, wx: float, wy: float) -> "_Landmark | None":
        best, best_d = None, LANDMARK_ASSOC_DIST
        for lm in self._lms:
            d = math.hypot(lm.x - wx, lm.y - wy)
            if d < best_d:
                best_d, best = d, lm
        return best


# ════════════════════════════════════════════════════════════════════════════
#  MAP SNAPSHOT  (SLAM thread → main thread, thread-safe)
# ════════════════════════════════════════════════════════════════════════════

class _MapSnapshot:
    __slots__ = ("nodes", "edges", "covariances", "scan_cloud",
                 "robot_x", "robot_y", "robot_theta",
                 "n_nodes", "n_edges", "step",
                 "lm_confirmed", "lm_candidate")

    def __init__(self, pose_graph: PoseGraph, scan_points: list,
                 robot_pose: Pose2D, step: int,
                 landmark_list: list):
        self.nodes       = list(pose_graph.nodes)
        self.edges       = list(pose_graph.edges)
        self.covariances = dict(pose_graph.covariances)
        self.n_nodes     = len(self.nodes)
        self.n_edges     = len(self.edges)
        self.step        = step
        self.robot_x     = robot_pose.x
        self.robot_y     = robot_pose.y
        self.robot_theta = robot_pose.theta
        valid            = [p for p in scan_points if p is not None]
        self.scan_cloud  = np.vstack(valid) if valid else None
        self.lm_confirmed = [(lm.lm_id, lm.x, lm.y, lm.node_id)
                              for lm in landmark_list if lm.confirmed]
        self.lm_candidate = [(lm.x, lm.y)
                              for lm in landmark_list if not lm.confirmed]


# ════════════════════════════════════════════════════════════════════════════
#  DRAW HELPER  (called only from the main thread)
# ════════════════════════════════════════════════════════════════════════════

def _draw_snapshot(snap: _MapSnapshot,
                   viz: SLAMVisualizer,
                   proxy_pg: PoseGraph) -> None:
    """
    Apply a snapshot to the proxy PoseGraph, then redraw via SLAMVisualizer
    plus the additional landmark and LIDAR overlays.
    Must be called from the main thread only.
    """
    # Sync proxy
    proxy_pg.nodes      = snap.nodes
    proxy_pg.edges      = snap.edges
    proxy_pg.covariances = snap.covariances

    # 1. Engine base layers: path, all edges, uncertainty ellipses
    viz.visualize_graph()
    ax = viz.ax

    # 2. LIDAR point-cloud
    if snap.scan_cloud is not None:
        ax.scatter(snap.scan_cloud[:, 0], snap.scan_cloud[:, 1],
                   s=1, c="cyan", alpha=0.20, zorder=2, label="LIDAR hits")

    # 3. Observation edges (pose → landmark) — redrawn explicitly for z-order
    for edge in snap.edges:
        if edge.type != "observation":
            continue
        if edge.from_id >= len(snap.nodes) or edge.to_id >= len(snap.nodes):
            continue
        pf = snap.nodes[edge.from_id]
        pt = snap.nodes[edge.to_id]
        ax.plot([pf.x, pt.x], [pf.y, pt.y],
                color="gold", lw=0.6, alpha=0.5, zorder=3)

    # 4. Candidate landmarks
    if snap.lm_candidate:
        ax.scatter([p[0] for p in snap.lm_candidate],
                   [p[1] for p in snap.lm_candidate],
                   marker="x", s=30, c="silver", zorder=4,
                   label="Landmark (candidate)")

    # 5. Confirmed landmarks
    if snap.lm_confirmed:
        ax.scatter([p[1] for p in snap.lm_confirmed],
                   [p[2] for p in snap.lm_confirmed],
                   marker="*", s=120, c="limegreen",
                   edgecolors="darkgreen", linewidths=0.5,
                   zorder=5, label="Landmark (confirmed)")
        for lm_id, lx, ly, _ in snap.lm_confirmed:
            ax.annotate(f"L{lm_id}", xy=(lx, ly),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=6, color="darkgreen", zorder=6)

    # 6. Robot pose arrow
    al = 0.08
    ax.annotate("",
                xy     = (snap.robot_x + al * math.cos(snap.robot_theta),
                           snap.robot_y + al * math.sin(snap.robot_theta)),
                xytext = (snap.robot_x, snap.robot_y),
                arrowprops=dict(arrowstyle="->", color="orange", lw=2.0),
                zorder=7)
    ax.plot(snap.robot_x, snap.robot_y, "o",
            color="orange", markersize=6, zorder=7, label="Robot")

    # 7. Title / legend
    ax.set_title(
        f"GraphSLAM  —  {snap.n_nodes} nodes | {snap.n_edges} edges | "
        f"LM: {len(snap.lm_confirmed)} conf + {len(snap.lm_candidate)} cand | "
        f"step {snap.step}",
        fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=7, markerscale=2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    viz.fig.canvas.draw_idle()
    viz.fig.canvas.flush_events()


# ════════════════════════════════════════════════════════════════════════════
#  SLAM THREAD  (runs WeBots robot.step loop on a background thread)
# ════════════════════════════════════════════════════════════════════════════

class _SLAMThread(threading.Thread):
    """
    Runs the entire WeBots control + SLAM pipeline on a background thread,
    keeping the main thread free for matplotlib GUI work.
    Snapshots are pushed into self.snap_queue for the main thread to draw.
    """

    def __init__(self):
        super().__init__(name="SLAMThread", daemon=True)
        self.snap_queue = queue.Queue(maxsize=2)
        self._done_evt  = threading.Event()

    def is_done(self) -> bool:
        return self._done_evt.is_set()

    def _push(self, ctrl: "EPuckGraphSLAMController") -> None:
        snap = _MapSnapshot(
            pose_graph    = ctrl.pose_graph,
            scan_points   = ctrl._scan_points,
            robot_pose    = ctrl.odom.current_pose,
            step          = ctrl._step_counter,
            landmark_list = ctrl.lm_map.landmarks,
        )
        # Drop stale snapshot rather than block
        try:
            self.snap_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.snap_queue.put_nowait(snap)
        except queue.Full:
            pass

    def run(self) -> None:
        ctrl = EPuckGraphSLAMController(self)
        ctrl.run()
        self._done_evt.set()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN CONTROLLER  (created and run inside _SLAMThread)
# ════════════════════════════════════════════════════════════════════════════

class EPuckGraphSLAMController:
    """WeBots e-puck controller: WASD tele-op + Landmark GraphSLAM."""

    def __init__(self, slam_thread: _SLAMThread):
        self._thread = slam_thread

        # ── WeBots ──────────────────────────────────────────────────────────
        self.robot  = Robot()
        self.ts_ms  = int(self.robot.getBasicTimeStep())

        # ── Motors ──────────────────────────────────────────────────────────
        self.left_motor  = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # ── Encoders ────────────────────────────────────────────────────────
        self.left_enc  = self.robot.getDevice("left wheel sensor")
        self.right_enc = self.robot.getDevice("right wheel sensor")
        self.left_enc.enable(self.ts_ms)
        self.right_enc.enable(self.ts_ms)

        # ── Keyboard ────────────────────────────────────────────────────────
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.ts_ms)

        # ── LIDAR ───────────────────────────────────────────────────────────
        self.lidar = self.robot.getDevice(LIDAR_DEVICE_NAME)
        if self.lidar is None:
            for name in ("lidar", "Lidar", "LiDAR", "rplidar", "sick"):
                self.lidar = self.robot.getDevice(name)
                if self.lidar:
                    break
        if self.lidar is None:
            print("[WARN] No LIDAR found – landmark SLAM disabled.")
        else:
            lidar_ms = max(self.ts_ms, int(1000.0 / LIDAR_FREQUENCY))
            self.lidar.enable(lidar_ms)
            self.lidar.enablePointCloud()
            print(f"[INFO] LIDAR '{self.lidar.getName()}' enabled ({lidar_ms} ms).")

        # ── SLAM graph ──────────────────────────────────────────────────────
        self.pose_graph    = PoseGraph()
        self.graph_builder = GraphBuilder()
        self.graph_builder.pose_graph = self.pose_graph
        self.odom   = OdometryIntegrator()
        self.kernel = RobustKernel()
        self.lcd    = LoopClosureDetector(
            pose_graph           = self.pose_graph,
            distance_threshold   = LOOP_CLOSURE_DIST,
            similarity_threshold = LOOP_CLOSURE_SIM,
        )
        self.lm_extractor = LandmarkExtractor()
        self.lm_map       = LandmarkMap(self.pose_graph, self.graph_builder)

        self._scan_counter = 0
        self._step_counter = 0
        self._descriptors: list       = []
        self._scan_points: list       = []
        self._running                 = True

        # Seed origin node
        origin_cov = np.diag([1e-9, 1e-9, 1e-9])
        fid = self.pose_graph.add_node(Pose2D(0.0, 0.0, 0.0), origin_cov)
        self.graph_builder.current_pose_id = fid
        self._descriptors.append(np.zeros(36))
        self._scan_points.append(None)

        self._thread._push(self)   # push initial empty snapshot
        print("[INFO] Landmark GraphSLAM ready.  Map window is live.")
        print("[INFO] Controls:  W=fwd  S=back  A=left  D=right  Q=quit")

    # ── Keyboard ─────────────────────────────────────────────────────────────

    def _read_keyboard(self) -> tuple[float, float]:
        lin = ang = 0.0
        key = self.keyboard.getKey()
        while key != -1:
            ch = chr(key & 0xFF).upper()
            if   ch == "W": lin  += LINEAR_SPEED
            elif ch == "S": lin  -= LINEAR_SPEED
            elif ch == "A": ang  += ANGULAR_SPEED
            elif ch == "D": ang  -= ANGULAR_SPEED
            elif ch == "Q": self._running = False
            key = self.keyboard.getKey()
        return lin, ang

    # ── Motors ───────────────────────────────────────────────────────────────

    def _set_velocity(self, lin: float, ang: float) -> None:
        lw, rw = _unicycle_to_wheels(lin, ang)
        self.left_motor.setVelocity(lw)
        self.right_motor.setVelocity(rw)

    # ── LIDAR ────────────────────────────────────────────────────────────────

    def _capture_scan(self):
        if self.lidar is None:
            return None, None
        raw = self.lidar.getRangeImage()
        if raw is None or len(raw) == 0:
            return None, None
        n         = len(raw)
        fov       = self.lidar.getFov()
        max_r     = self.lidar.getMaxRange()
        angles    = np.linspace(fov / 2.0, -fov / 2.0, n)
        ranges    = np.array(raw, dtype=np.float32)
        ranges    = np.where(np.isfinite(ranges), ranges, max_r)
        scan      = LaserScan(ranges=ranges, angles=angles, id=self._scan_counter)
        self._scan_counter += 1
        return scan, _build_descriptor(ranges)

    # ── SLAM update ───────────────────────────────────────────────────────────

    def _slam_update(self) -> None:
        # 1. Dead-reckoning
        self.odom.update(self.left_enc.getValue(), self.right_enc.getValue())
        d = self.odom.last_delta
        if abs(d.x) < 1e-4 and abs(d.y) < 1e-4 and abs(d.theta) < 1e-4:
            return

        # 2. New pose node + odometry edge
        odom_cov = np.diag(ODOM_SIGMA ** 2)
        info     = np.linalg.inv(odom_cov)
        new_pose = Pose2D(self.odom.current_pose.x,
                          self.odom.current_pose.y,
                          self.odom.current_pose.theta)
        new_id = self.pose_graph.add_node(new_pose, odom_cov)
        self.pose_graph.add_edge(
            from_id=self.graph_builder.current_pose_id,
            to_id=new_id, measurement=d, information=info)
        self.pose_graph.edges[-1].type = "odometry"
        self.graph_builder.current_pose_id = new_id

        # 3. LIDAR scan
        scan, descriptor = self._capture_scan()
        self._descriptors.append(descriptor if descriptor is not None
                                 else np.zeros(36))
        self._scan_points.append(
            _project_scan(new_pose, scan) if scan is not None else None)

        # 4. Landmark extraction & observation edges
        if scan is not None:
            obs = self.lm_extractor.extract(scan)
            if obs:
                self.lm_map.process_observations(new_pose, obs)

        # 5. Loop-closure detection
        if scan is not None and descriptor is not None \
                and len(self.pose_graph.nodes) > 55:
            for cid, cp in self._find_candidates(new_pose):
                if _descriptor_similarity(descriptor,
                                          self._descriptors[cid]) >= LOOP_CLOSURE_SIM:
                    rel = self._relative_pose(cp, new_pose)
                    self.pose_graph.add_edge(
                        from_id=cid, to_id=new_id,
                        measurement=rel, information=np.diag([50., 50., 20.]))
                    self.pose_graph.edges[-1].type = "loop_closure"
                    print(f"[SLAM] Loop closure: {cid} ↔ {new_id}")

        # 6. Periodic Huber re-weighting
        self._step_counter += 1
        if self._step_counter % OPTIMISE_EVERY_N_STEPS == 0:
            self._optimise_graph()

        # 7. Push snapshot to main thread
        self._thread._push(self)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_candidates(self, q: Pose2D, k: int = 10):
        n = len(self.pose_graph.nodes)
        hits = []
        for i in range(max(0, n - 51)):
            p = self.pose_graph.nodes[i]
            if math.hypot(p.x - q.x, p.y - q.y) <= LOOP_CLOSURE_DIST:
                hits.append((i, p))
        hits.sort(key=lambda t: math.hypot(t[1].x - q.x, t[1].y - q.y))
        return hits[:k]

    @staticmethod
    def _relative_pose(o: Pose2D, t: Pose2D) -> Pose2D:
        c, s = math.cos(o.theta), math.sin(o.theta)
        dx, dy = t.x - o.x, t.y - o.y
        return Pose2D(c*dx + s*dy, -s*dx + c*dy, t.theta - o.theta)

    def _optimise_graph(self) -> None:
        lc = [e for e in self.pose_graph.edges if e.type == "loop_closure"]
        if not lc:
            return
        t0 = time.perf_counter()
        for edge in lc:
            err = np.array([edge.measurement.x, edge.measurement.y,
                            edge.measurement.theta])
            _, w = self.kernel.huber_loss(err, delta=0.5)
            edge.information = edge.information * w
        print(f"[SLAM] Huber re-weight: {len(lc)} LC edges "
              f"({(time.perf_counter()-t0)*1000:.1f} ms)")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        print("[INFO] SLAM running.  Click the WeBots 3D view to capture keys.")
        while self.robot.step(self.ts_ms) != -1 and self._running:
            lin, ang = self._read_keyboard()
            self._set_velocity(lin, ang)
            self._slam_update()
        self._set_velocity(0.0, 0.0)
        self._thread._push(self)   # final snapshot
        self._print_summary()

    def _print_summary(self) -> None:
        n   = len(self.pose_graph.nodes)
        ne  = len(self.pose_graph.edges)
        oe  = sum(1 for e in self.pose_graph.edges if e.type == "odometry")
        le  = sum(1 for e in self.pose_graph.edges if e.type == "loop_closure")
        ob  = sum(1 for e in self.pose_graph.edges if e.type == "observation")
        p   = self.pose_graph.nodes[-1] if n else Pose2D(0, 0, 0)
        lms = self.lm_map.landmarks
        print("\n── GraphSLAM Session Summary ─────────────────────────────────")
        print(f"  Pose nodes  : {n}")
        print(f"  Edges       : {ne}  "
              f"(odometry={oe}, loop_closure={le}, observation={ob})")
        print(f"  Landmarks   : {sum(l.confirmed for l in lms)} confirmed + "
              f"{sum(not l.confirmed for l in lms)} candidate = {len(lms)} total")
        print(f"  Final pose  : x={p.x:.3f} m  y={p.y:.3f} m  "
              f"θ={math.degrees(p.theta):.1f}°")
        print("──────────────────────────────────────────────────────────────\n")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT  ─  main thread owns ALL matplotlib work
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── 1. Pick backend on the main thread ───────────────────────────────────
    _choose_backend()

    # ── 2. Create SLAMVisualizer (figure + axes) on the main thread ──────────
    proxy_pg = PoseGraph()
    viz      = SLAMVisualizer(proxy_pg)
    viz.fig.canvas.manager.set_window_title(
        "GraphSLAM Live Map — E-Puck  (Landmark SLAM)")
    plt.ion()
    viz.fig.show()

    # ── 3. Start SLAM on a background thread ─────────────────────────────────
    slam_thread = _SLAMThread()
    slam_thread.start()

    # ── 4. Main-thread draw loop ──────────────────────────────────────────────
    while not slam_thread.is_done():
        t0   = time.perf_counter()
        snap = None
        # Drain queue — keep only the newest snapshot
        while True:
            try:
                snap = slam_thread.snap_queue.get_nowait()
            except queue.Empty:
                break
        if snap is not None:
            try:
                _draw_snapshot(snap, viz, proxy_pg)
            except Exception as exc:
                print(f"[VIS] draw error: {exc}")
        # Keep GUI responsive even with no new data
        try:
            viz.fig.canvas.flush_events()
        except Exception:
            pass
        elapsed = time.perf_counter() - t0
        rem     = MAP_REFRESH_SEC - elapsed
        if rem > 0:
            plt.pause(rem)   # yields control to the OS event loop

    # ── 5. Final draw, then block until user closes the window ───────────────
    # Drain any last snapshot
    snap = None
    while True:
        try:
            snap = slam_thread.snap_queue.get_nowait()
        except queue.Empty:
            break
    if snap is not None:
        try:
            _draw_snapshot(snap, viz, proxy_pg)
        except Exception as exc:
            print(f"[VIS] final draw error: {exc}")

    plt.ioff()
    print("[VIS] Simulation ended.  Close the map window to exit.")
    plt.show(block=True)
    slam_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()