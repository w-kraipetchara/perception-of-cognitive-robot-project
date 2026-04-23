from dataclasses import dataclass
import numpy as np
import math
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import spatial
from functools import lru_cache

@dataclass
class Pose2D:
    """Represents a 2D robot pose (x, y, θ)"""
    x: float
    y: float
    theta: float = 0.0
    
    def to_matrix(self) -> np.ndarray:
        """Convert to homogeneous transformation matrix"""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0,  0,    1]
        ])
    
    def inverse(self) -> 'Pose2D':
        """Compute inverse transformation"""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        x = -c * self.x - s * self.y
        y =  s * self.x - c * self.y
        return Pose2D(x, y, -self.theta)

@dataclass
class Edge:
    """Represents a constraint between two nodes in the graph"""
    from_id: int
    to_id: int
    measurement: Pose2D
    information: np.ndarray
    type: str = "odometry"  # Could be "odometry", "loop_closure", or "observation"
    
@dataclass
class Control:
    """SAMPLE DATA!! Represents control input from the robot for motion prediction. Note that this is a simple representation and NEEDS to include more complex control inputs."""
    velocity: float
    angular_velocity: float
    dt: float

@dataclass
class Measurement:
    """SAMPLE DATA!! Represents a measurement (e.g., landmark observation)"""
    x: float
    y: float

@dataclass
class LaserScan:
    """SAMPLE DATA!! Represents a laser scan with range measurements"""
    ranges: np.ndarray
    angles: np.ndarray
    id: int  # Unique identifier for the scan

@dataclass
class LoopClosure:
    """Represents a detected loop closure constraint"""
    pose_id: int
    transform: Pose2D


    
class GraphSLAM:                 # Main class for GraphSLAM implementation
    def __init__(self):
        self.poses = []          # Robot pose nodes
        self.landmarks = []      # Landmark nodes
        self.odom_edges = []     # Odometry constraints
        self.obs_edges = []      # Observation constraints
        self.loop_edges = []     # Loop closure constraints
    
    def predict_motion(prev_pose: Pose2D, control: Control) -> Pose2D:
        """
        Predict next pose based on control input and motion model
        """
        dx = control.velocity * math.cos(prev_pose.theta) * control.dt
        dy = control.velocity * math.sin(prev_pose.theta) * control.dt
        dtheta = control.angular_velocity * control.dt
        
        return Pose2D(
            x=prev_pose.x + dx,
            y=prev_pose.y + dy,
            theta=prev_pose.theta + dtheta
        )

    def compute_error(pred: Measurement, actual: Measurement, information: np.ndarray) -> float:
        """
        Compute weighted error between predicted and actual measurements
        """
        error = pred - actual
        return error.T @ information @ error
    
class GraphOptimizer:            # Class responsible for optimizing the graph
    def optimize(self, 
                 max_iterations: int = 100, 
                 convergence_thresh: float = 1e-6) -> None:
        """
        Optimize the pose graph using Gauss-Newton method
        """
        for iteration in range(max_iterations):
            # Compute total error
            prev_error = self.compute_total_error()
            
            # Build linear system
            H = np.zeros((self.state_dim, self.state_dim)) #
            b = np.zeros(self.state_dim)
            
            # Add constraints
            self.add_pose_constraints(H, b)
            self.add_landmark_constraints(H, b)
            self.add_loop_closure_constraints(H, b)
            
            # Solve system
            dx = np.linalg.solve(H, -b)
            
            # Update poses and landmarks
            self.update_state(dx)
            
            # Check convergence
            current_error = self.compute_total_error()
            if abs(prev_error - current_error) < convergence_thresh:
                break

    def update_state(self, dx: np.ndarray) -> None:
        """
        Applies the calculated correction (Delta mu) to all nodes in the graph.
        Matches the formula: mu_hat = mu_0 + Delta_mu
        """
        # In a standard pose graph, each pose has 3 degrees of freedom (x, y, theta)
        # We iterate through the nodes and apply the corresponding slice of dx
        for i, node in enumerate(self.pose_graph.nodes):
            idx = i * 3
            
            # Apply translations
            node.x += dx[idx]
            node.y += dx[idx+1]
            
            # Apply rotation and normalize to [-pi, pi]
            if hasattr(node, 'theta'):  # Check if it's a Pose2D (Landmarks might only have x,y)
                node.theta += dx[idx+2]
                node.theta = (node.theta + math.pi) % (2 * math.pi) - math.pi

    def compute_total_error(self) -> float:
        """
        Calculates the sum of all errors across every edge in the graph.
        Used to check if the optimizer has converged.
        """
        total_error = 0.0
        
        # We need an instance of the sparse optimizer to reuse our error calculation logic
        # (Assuming you are using SparseGraphOptimizer's math)
        math_helper = SparseGraphOptimizer(self.pose_graph)
        
        for edge in self.pose_graph.edges:
            # Get the raw error [dx, dy, dtheta]
            err_vector = math_helper.compute_edge_error(edge)
            
            # Multiply by the information matrix (weight): e^T * Omega * e
            weighted_error = err_vector.T @ edge.information @ err_vector
            total_error += float(weighted_error)
            
        return total_error
    
class PoseGraph:
    def __init__(self):
        self.nodes: list[Pose2D] = []
        self.edges: list[Edge] = []
        self.covariances: dict[int, np.ndarray] = {}
        
    def add_node(self, pose: Pose2D, 
                 covariance: np.ndarray = None) -> int:
        """Add a new node to the graph"""
        node_id = len(self.nodes)
        self.nodes.append(pose)
        if covariance is not None:
            self.covariances[node_id] = covariance
        return node_id
    
    def add_edge(self, from_id: int, to_id: int,
                 measurement: Pose2D,
                 information: np.ndarray,
                 edge_type: str = "odometry") -> None:
        """Add a new edge (constraint) to the graph"""
        # Pass the edge_type to the Edge dataclass as 'type'
        edge = Edge(from_id, to_id, measurement, information, type=edge_type) 
        self.edges.append(edge)
    
class GraphBuilder:
    def __init__(self):
        self.pose_graph = PoseGraph()
        self.current_pose_id = None
        # --- UPDATED: Track the pose AND the number of times it has been seen ---
        self.known_landmarks: dict[int, dict] = {} # Format will be {node_id: {'pose': Pose2D, 'count': int}}
        
    def add_odometry_measurement(self, 
                               odom_measurement: Pose2D,
                               uncertainty: np.ndarray) -> None:
        """Add new pose and odometry edge to graph"""
        # Create new pose node
        new_pose = odom_measurement
        new_pose_id = self.pose_graph.add_node(new_pose)
        
        # Add odometry edge if not first pose
        if self.current_pose_id is not None:
            # --- NEW: Calculate the RELATIVE step between the two poses ---
            prev_pose = self.pose_graph.nodes[self.current_pose_id]
            dx = new_pose.x - prev_pose.x
            dy = new_pose.y - prev_pose.y
            
            # Rotate into the previous pose's local frame
            c = math.cos(prev_pose.theta)
            s = math.sin(prev_pose.theta)
            
            rel_x = c * dx + s * dy
            rel_y = -s * dx + c * dy
            
            # Find the difference in angle and normalize it
            rel_theta = (new_pose.theta - prev_pose.theta + math.pi) % (2 * math.pi) - math.pi
            
            # This is the actual "rubber band" constraint!
            relative_meas = Pose2D(rel_x, rel_y, rel_theta)
            
            information = np.linalg.inv(uncertainty) #Rt inverse
            self.pose_graph.add_edge(
                from_id=self.current_pose_id,
                to_id=new_pose_id,
                measurement=relative_meas,   # <--- Pass the relative step!
                information=information,
                edge_type="odometry"
            )
        
        self.current_pose_id = new_pose_id

    def process_landmark_observation(self, 
                                     global_x: float, 
                                     global_y: float, 
                                     uncertainty: np.ndarray,
                                     chi_square_threshold: float = 5.99) -> int:
        """
        Uses a probabilistic correspondence function to match and merge landmarks.
        """
        best_match_id = None
        min_dist = float('inf')
        
        # Inverse of the measurement covariance matrix (Q_t^-1)
        info_matrix = np.linalg.inv(uncertainty) 
        
        # 1. Landmark Correspondence Function (Probabilistic Match)
        for lm_id, lm_data in self.known_landmarks.items():
            lm_pose = lm_data['pose']
            
            # Calculate the difference vector between the new reading and known landmark
            dx = global_x - lm_pose.x
            dy = global_y - lm_pose.y
            diff = np.array([dx, dy])
            
            # Calculate Mahalanobis distance squared (statistical distance)
            # This accounts for the convolution of the normal distributions
            mahalanobis_sq = diff.T @ info_matrix @ diff
            
            # 5.99 is the 95% confidence interval for 2 degrees of freedom (X and Y)
            if mahalanobis_sq < min_dist and mahalanobis_sq < chi_square_threshold:
                min_dist = mahalanobis_sq
                best_match_id = lm_id
                
        # 2. Merge Redundant Landmarks or Create New
        if best_match_id is not None:
            # Merge: Update the existing landmark's position (running average)
            lm_data = self.known_landmarks[best_match_id]
            lm_data['count'] += 1
            alpha = 1.0 / lm_data['count']
            
            lm_pose = lm_data['pose']
            lm_pose.x = (1.0 - alpha) * lm_pose.x + alpha * global_x
            lm_pose.y = (1.0 - alpha) * lm_pose.y + alpha * global_y
            
        else:
            # Register a brand new landmark
            lm_pose = Pose2D(x=global_x, y=global_y, theta=0.0)
            best_match_id = self.pose_graph.add_node(lm_pose)
            self.known_landmarks[best_match_id] = {'pose': lm_pose, 'count': 1}
            print(f"Engine: Registered brand new landmark at Node {best_match_id}")
            
        # 3. Add the observation edge connecting the robot to this landmark
        robot_pose = self.pose_graph.nodes[self.current_pose_id]
        dx = global_x - robot_pose.x
        dy = global_y - robot_pose.y
        c, s = math.cos(robot_pose.theta), math.sin(robot_pose.theta)
        
        local_meas = Measurement(
            x = c * dx + s * dy,
            y = -s * dx + c * dy
        )
        
        self.pose_graph.add_edge(
            from_id=self.current_pose_id,
            to_id=best_match_id,
            measurement=local_meas,
            information=info_matrix,
            edge_type="observation"
        )
        
        return best_match_id
    
    def add_landmark_observation(self,
                               landmark_id: int,
                               measurement: Measurement,
                               uncertainty: np.ndarray) -> None:
        """Add landmark observation edge"""
        information = np.linalg.inv(uncertainty)
        self.pose_graph.add_edge(
            self.current_pose_id,
            landmark_id,
            measurement,
            information,
            edge_type="observation"
        )

class LoopClosureDetector:
    def __init__(self, pose_graph: PoseGraph,
                 distance_threshold: float = 2.0,
                 similarity_threshold: float = 0.75):
        self.pose_graph = pose_graph
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.descriptor_database = []
        
    def detect_loop_closures(self, 
                           current_pose: Pose2D,
                           current_scan: LaserScan,
                           current_descriptor: np.ndarray) -> list[LoopClosure]:
        """Detect potential loop closures"""
        loop_closures = []
        
        # Find nearby poses
        nearby_poses = self.find_nearby_poses(current_pose)
         
        for pose_id, pose in nearby_poses:
            # Skip recent poses
            if abs(pose_id - len(self.pose_graph.nodes) + 1) < 50:
                continue
                
            # Check scan similarity
            if self.check_scan_similarity(current_scan, 
                                       self.descriptor_database[pose_id],
                                       current_descriptor):
                # Compute relative transform
                transform = self.compute_relative_transform(
                    current_pose, pose)
                    
                # Verify geometric consistency
                if self.verify_geometric_consistency(transform):
                    loop_closures.append(
                        LoopClosure(pose_id, transform))
        
        return loop_closures
    
    def check_scan_similarity(self, 
                            current_scan: LaserScan,
                            stored_scan: LaserScan,
                            current_descriptor: np.ndarray) -> bool:
        """Check if two laser scans are similar"""
        score = self.compute_descriptor_similarity(
            current_descriptor,
            self.descriptor_database[stored_scan.id]
        )
        return score > self.similarity_threshold
    
class SparseGraphOptimizer:
    def __init__(self, pose_graph: PoseGraph):
        self.pose_graph = pose_graph

    def update_state(self, dx: np.ndarray) -> None:
        """
        Applies the calculated correction (Delta mu) to all nodes in the graph.
        Matches the formula: mu_hat = mu_0 + Delta_mu
        """
        for i, node in enumerate(self.pose_graph.nodes):
            idx = i * 3
            
            # Apply translations
            node.x += float(dx[idx])
            node.y += float(dx[idx+1])
            
            # Apply rotation and normalize to [-pi, pi]
            if hasattr(node, 'theta'):  
                node.theta += float(dx[idx+2])
                node.theta = (node.theta + math.pi) % (2 * math.pi) - math.pi

    def compute_total_error(self) -> float:
        """
        Calculates the sum of all errors across every edge in the graph.
        Used to check if the optimizer has converged.
        """
        total_error = 0.0
        
        for edge in self.pose_graph.edges:
            # Get the raw error [dx, dy, dtheta] natively
            err_vector = self.compute_edge_error(edge)
            
            # Multiply by the information matrix (weight): e^T * Omega * e
            weighted_error = err_vector.T @ edge.information @ err_vector
            total_error += float(weighted_error)
            
        return total_error
        
    def optimize(self, max_iterations: int = 10, tolerance: float = 1e-4) -> None:
        """
        Executes the iterative refinement loop to minimize global map error.
        """
        print("--- Starting Graph Optimization ---")
        
        # Step 1: Calculate the starting global error
        current_error = self.compute_total_error()
        print(f"Initial Error: {current_error:.4f}")
        
        for iteration in range(max_iterations):
            # Step 2: Assemble the giant Information Matrix (H) and Vector (b)
            H, b = self.build_sparse_system()
            
            # Step 3: Solve the linear system for the correction vector (Delta mu)
            # We use SciPy's sparse solver because it is lightning fast for SLAM matrices
            dx = splinalg.spsolve(H, -b)
            
            # Step 4: Update the estimates recursively (mu_hat = mu_0 + Delta_mu)
            self.update_state(dx)
            
            # Step 5: Check for convergence
            new_error = self.compute_total_error()
            error_diff = current_error - new_error
            
            print(f"Iteration {iteration + 1}: Error = {new_error:.4f} (Diff: {error_diff:.6f})")
            
            # If the error stops dropping significantly, the map is perfectly relaxed!
            if error_diff < tolerance and error_diff >= 0:
                print("Optimizer has successfully converged!")
                break
                
            # If the error increased, the math overshot. A more advanced solver 
            # (like Levenberg-Marquardt) would handle this, but for Gauss-Newton, 
            # we just track it.
            if error_diff < 0:
                print("Warning: Error increased. Graph might be poorly constrained.")
                
            current_error = new_error

    def build_sparse_system(self) -> tuple[sparse.csr_matrix, np.ndarray]:
        """
        Assembles the global Information Matrix (H) and Vector (b) 
        by linearizing the non-linear graph using Jacobians.
        """
        # Each pose has 3 degrees of freedom (x, y, theta)
        n_poses = len(self.pose_graph.nodes)
        system_size = n_poses * 3
        
        # Initialize the massive H matrix (using a sparse matrix to save memory) 
        # and the b vector with zeros.
        H = sparse.lil_matrix((system_size, system_size))
        b = np.zeros(system_size)
        
        # Iterate through every edge to add its mathematical tension to the system
        for edge in self.pose_graph.edges:
            id_i = edge.from_id
            id_j = edge.to_id
            idx_i = id_i * 3
            idx_j = id_j * 3
            
            # 1. Get the raw error and the Jacobians (Linearization)
            e = self.compute_edge_error(edge)
            J_i, J_j = self.compute_edge_jacobians(edge)
            
            # Omega is the weight of this specific edge
            omega = edge.information
            
            # 2. Add the mathematical tension to the H Matrix
            H[idx_i:idx_i+3, idx_i:idx_i+3] += J_i.T @ omega @ J_i
            H[idx_j:idx_j+3, idx_j:idx_j+3] += J_j.T @ omega @ J_j
            H[idx_i:idx_i+3, idx_j:idx_j+3] += J_i.T @ omega @ J_j
            H[idx_j:idx_j+3, idx_i:idx_i+3] += J_j.T @ omega @ J_i
            
            # 3. Add the tension to the b Vector
            b[idx_i:idx_i+3] += J_i.T @ omega @ e
            b[idx_j:idx_j+3] += J_j.T @ omega @ e
            
        # 4. Lock the very first node in place! (Anchor point)
        # If we don't do this, the entire map will float away during optimization.
        H[0:3, 0:3] += np.eye(3) * 10000.0 
            
        # --- NEW: 5. Lock the rotation (theta) of all landmarks! ---
        # Landmarks are 2D points, not poses. We lock their 3rd degree of freedom 
        # so the matrix solver doesn't encounter a row of zeroes.
        for edge in self.pose_graph.edges:
            if edge.type == "observation":
                lm_idx = edge.to_id * 3
                # Inject a massive weight onto the diagonal for this node's theta
                H[lm_idx+2, lm_idx+2] += 10000.0
            
        return H.tocsr(), b
    
    def compute_edge_error(self, edge: Edge) -> np.ndarray:
        """Compute the difference between predicted and actual measurement."""
        pose_i = self.pose_graph.nodes[edge.from_id]

        if edge.type == "observation":
            # Pose to Landmark error calculation
            landmark_j = self.pose_graph.nodes[edge.to_id] 
            dx = landmark_j.x - pose_i.x
            dy = landmark_j.y - pose_i.y
            
            c, s = math.cos(pose_i.theta), math.sin(pose_i.theta)
            pred_x = c * dx + s * dy
            pred_y = -s * dx + c * dy
            
            return np.array([
                pred_x - edge.measurement.x,
                pred_y - edge.measurement.y
            ])
        else:
            # Pose to Pose error calculation (Odometry / Loop Closure)
            pose_j = self.pose_graph.nodes[edge.to_id]
            dx = pose_j.x - pose_i.x
            dy = pose_j.y - pose_i.y
            
            c, s = math.cos(pose_i.theta), math.sin(pose_i.theta)
            pred_x = c * dx + s * dy
            pred_y = -s * dx + c * dy
            pred_theta = pose_j.theta - pose_i.theta
            
            # Normalize angular error to be within [-pi, pi]
            err_theta = (pred_theta - edge.measurement.theta + math.pi) % (2 * math.pi) - math.pi
            
            return np.array([
                pred_x - edge.measurement.x,
                pred_y - edge.measurement.y,
                err_theta
            ])

    def compute_edge_jacobians(self, edge: Edge) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Jacobians (derivatives) of the error function."""
        pose_i = self.pose_graph.nodes[edge.from_id]
        c, s = math.cos(pose_i.theta), math.sin(pose_i.theta)
        
        if edge.type == "observation":
            landmark_j = self.pose_graph.nodes[edge.to_id]
            dx = landmark_j.x - pose_i.x
            dy = landmark_j.y - pose_i.y
            
            # Jacobian with respect to the observing pose
            J_i = np.array([
                [-c, -s, -s * dx + c * dy],
                [ s, -c, -c * dx - s * dy]
            ])
            # Jacobian with respect to the landmark position
            J_j = np.array([
                [ c,  s, 0],
                [-s,  c, 0]
            ])
            return J_i, J_j
        else:
            pose_j = self.pose_graph.nodes[edge.to_id]
            dx = pose_j.x - pose_i.x
            dy = pose_j.y - pose_i.y
            
            # Jacobian with respect to the starting pose
            J_i = np.array([
                [-c, -s, -s * dx + c * dy],
                [ s, -c, -c * dx - s * dy],
                [ 0,  0, -1]
            ])
            # Jacobian with respect to the destination pose
            J_j = np.array([
                [ c,  s, 0],
                [-s,  c, 0],
                [ 0,  0, 1]
            ])
            return J_i, J_j
    
class RobustKernel:
    def huber_loss(self, error: np.ndarray, delta: float) -> tuple[float, float]:
        """
        Implement Huber loss function for robust optimization
        Returns (weighted_error, weight)
        """
        error_norm = np.linalg.norm(error)
        
        if error_norm <= delta:
            # Quadratic region
            return error_norm**2, 1.0
        else:
            # Linear region
            return 2 * delta * error_norm - delta**2, delta / error_norm

class RobustGraphOptimizer(GraphOptimizer):
    def __init__(self, pose_graph: PoseGraph, kernel: RobustKernel):
        super().__init__(pose_graph)
        self.kernel = kernel
        
    def compute_weighted_error(self, edge: Edge) -> np.ndarray:
        """Compute error with robust weighting"""
        error = self.compute_edge_error(edge)
        _, weight = self.kernel.huber_loss(error, delta=1.0)
        return error * weight
    
class GraphPruner:
    def __init__(self, pose_graph: PoseGraph):
        self.pose_graph = pose_graph

    def prune_unobserved_poses(self) -> None:
        """
        Removes poses that contain no landmark information by merging 
        their adjacent odometry edges to minimize graph size.
        """
        # 1. Find all poses that actually saw a landmark
        observed_poses = set()
        for edge in self.pose_graph.edges:
            if edge.type == "observation":
                observed_poses.add(edge.from_id)
                
        # Always keep the very first and very last pose
        observed_poses.add(0)
        observed_poses.add(len(self.pose_graph.nodes) - 1)

        new_edges = []
        i = 0
        
        # 2. Iterate through edges to rebuild the odometry chain
        while i < len(self.pose_graph.edges):
            edge = self.pose_graph.edges[i]
            
            # Keep all non-odometry edges as they are
            if edge.type != "odometry":
                new_edges.append(edge)
                i += 1
                continue

            from_id = edge.from_id
            to_id = edge.to_id
            current_measurement = edge.measurement
            # Invert information matrix to get covariance (uncertainty)
            current_covariance = np.linalg.inv(edge.information)

            # 3. If the destination node saw nothing, chain it to the next one!
            while to_id not in observed_poses and to_id < len(self.pose_graph.nodes) - 1:
                # Find the next odometry edge
                next_edge = next((e for e in self.pose_graph.edges 
                                  if e.type == "odometry" and e.from_id == to_id), None)
                
                if next_edge is None:
                    break
                    
                # Merge the measurements (simple addition for X, Y, Theta)
                current_measurement = Pose2D(
                    x = current_measurement.x + next_edge.measurement.x,
                    y = current_measurement.y + next_edge.measurement.y,
                    theta = current_measurement.theta + next_edge.measurement.theta
                )
                
                # Combine uncertainties (Cov(A+B) = Cov(A) + Cov(B))
                next_covariance = np.linalg.inv(next_edge.information)
                current_covariance = current_covariance + next_covariance
                
                # Move our pointer forward
                to_id = next_edge.to_id
                
            # Create the newly combined edge
            combined_edge = Edge(
                from_id=from_id,
                to_id=to_id,
                measurement=current_measurement,
                information=np.linalg.inv(current_covariance),
                type="odometry"
            )
            new_edges.append(combined_edge)
            
            # Fast-forward our main loop index past the edges we just bypassed
            while i < len(self.pose_graph.edges) and self.pose_graph.edges[i].from_id < to_id:
                i += 1

        # 4. Replace the old bloated edge list with our pruned list
        self.pose_graph.edges = new_edges
        print(f"Graph pruned! Reduced to {len(self.pose_graph.edges)} essential edges.")

# ==============================================================================    
    
class DynamicSLAM(GraphSLAM):
    def __init__(self):
        super().__init__()
        self.dynamic_object_detector = DynamicObjectDetector()   # Placeholder for dynamic object detection module NOT from Computer Vision. A separate module that analyzes laser scans to identify dynamic objects based on motion patterns and scan inconsistencies.
        self.static_map = OccupancyGrid()                        
        
    def process_scan(self, scan: LaserScan) -> None:
        """Process laser scan considering dynamic objects"""
        # Detect and remove dynamic objects
        static_scan = self.dynamic_object_detector.filter_dynamic_objects(scan)
        
        # Update static map
        self.static_map.update(static_scan)
        
        # Add to pose graph with increased uncertainty for dynamic regions
        uncertainty = self.compute_adaptive_uncertainty(scan)
        self.add_scan_to_graph(static_scan, uncertainty)

class MultiRobotSLAM:
    def __init__(self, robot_count: int):
        self.robots = [GraphSLAM() for _ in range(robot_count)]
        self.relative_poses = {}
        self.shared_landmarks = {}
        
    def merge_maps(self) -> None:
        """Merge individual robot maps"""
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                # Find common landmarks
                common = self.find_common_landmarks(
                    self.robots[i], self.robots[j])
                
                if len(common) >= 3:
                    # Compute relative transformation
                    transform = self.compute_relative_transform(
                        self.robots[i], self.robots[j], common)
                    
                    # Merge pose graphs
                    self.merge_pose_graphs(i, j, transform)

class SLAMVisualizer:
    def __init__(self, pose_graph: PoseGraph):
        self.pose_graph = pose_graph
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
    def visualize_graph(self) -> None:
        """Visualize current state of pose graph"""
        self.ax.clear()
        
        # Plot poses
        poses = np.array([[pose.x, pose.y] for pose in self.pose_graph.nodes])
        self.ax.plot(poses[:, 0], poses[:, 1], 'b.-', label='Robot Path')
        
        # Plot edges
        for edge in self.pose_graph.edges:
            if edge.type == "odometry":
                color = 'g'
            elif edge.type == "loop_closure":
                color = 'r'
            else:
                color = 'y'
                
            self.plot_edge(edge, color)
        
        # Plot uncertainty ellipses
        self.plot_uncertainty_ellipses()
        
        plt.legend()
        plt.axis('equal')
        plt.draw()

    def plot_edge(self, edge: Edge, color: str) -> None:
        """
        Draws a line connecting two nodes based on an edge constraint.
        """
        # Safely fetch the starting and ending nodes using their IDs
        if edge.from_id < len(self.pose_graph.nodes) and edge.to_id < len(self.pose_graph.nodes):
            pose_from = self.pose_graph.nodes[edge.from_id]
            pose_to = self.pose_graph.nodes[edge.to_id]
            
            # Plot a dashed line between the two X/Y coordinate pairs
            self.ax.plot(
                [pose_from.x, pose_to.x], 
                [pose_from.y, pose_to.y], 
                color=color, 
                linestyle='--',   # Dashed lines look cleaner for constraints
                linewidth=1.0,
                alpha=0.7         # Slight transparency so it doesn't hide the robot path
            )
        
    def plot_uncertainty_ellipses(self) -> None:
        """Plot uncertainty ellipses for poses"""
        for node_id, covariance in self.pose_graph.covariances.items():
            pose = self.pose_graph.nodes[node_id]
            eigenvals, eigenvecs = np.linalg.eig(covariance[:2, :2])
            angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
            
            ellip = patches.Ellipse(
                (pose.x, pose.y),
                2 * np.sqrt(eigenvals[0]),
                2 * np.sqrt(eigenvals[1]),
                angle=np.degrees(angle),
                fill=False,
                color='gray',
                alpha=0.3
            )
            self.ax.add_patch(ellip)

class OptimizedGraphSLAM:
    def __init__(self):
        self.pose_graph = PoseGraph()
        self.node_cache = lru_cache(1000)
        self.kdtree = None
        
    def update_spatial_index(self) -> None:
        """Update KD-tree for spatial queries"""
        poses = np.array([[pose.x, pose.y] for pose in self.pose_graph.nodes])
        self.kdtree = spatial.cKDTree(poses)
        
    def find_nearby_poses(self, 
                         query_pose: Pose2D, 
                         radius: float) -> list[int]:
        """Find poses within radius using KD-tree"""
        if self.kdtree is None:
            self.update_spatial_index()
            
        query_point = np.array([query_pose.x, query_pose.y])
        indices = self.kdtree.query_ball_point(query_point, radius)
        return indices
