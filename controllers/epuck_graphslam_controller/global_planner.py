import heapq
import math
from occupancy_grid import OccupancyGrid, FREE, OCCUPIED, UNKNOWN

class GlobalPlanner:
    """
    A* path planner operating on an OccupancyGrid.

    Priority logic (matches the assignment spec):
        1. If a target cell is known → plan path to target.
        2. Otherwise            → plan path to best frontier goal.

    Usage:
        planner = GlobalPlanner(occupancy_grid)
        waypoints = planner.plan(start_world_xy, goal_world_xy)
        # waypoints is a list of (wx, wy) tuples from start to goal,
        # or None if no path exists.
    """

    # 8-connected movement (allows diagonal steps)
    # (dcol, drow, cost)
    _MOVES = [
        (-1,  0, 1.0),  # left
        ( 1,  0, 1.0),  # right
        ( 0, -1, 1.0),  # down
        ( 0,  1, 1.0),  # up
        (-1, -1, 1.414),  # diagonal
        (-1,  1, 1.414),
        ( 1, -1, 1.414),
        ( 1,  1, 1.414),
    ]

    def __init__(self, occupancy_grid: OccupancyGrid, inflation_radius=0.10):
        """
        Args:
            occupancy_grid:   Shared OccupancyGrid instance (updated by lidar).
            inflation_radius: Obstacles are inflated by this many metres so the
                              robot body stays clear of walls.
        """
        self.grid             = occupancy_grid
        self.inflation_cells  = max(1, int(inflation_radius / occupancy_grid.resolution))
        self._inflated        = None   # computed on demand, cached until grid changes
        self._grid_version    = -1     # simple dirty flag

        # Priority mode
        self.target_cell = None   # set externally when target is spotted in SLAM map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_target(self, world_x, world_y):
        """Call this when the robot spots its target object in the map."""
        cell = self.grid.world_to_grid(world_x, world_y)
        if cell:
            self.target_cell = cell

    def clear_target(self):
        self.target_cell = None

    def plan(self, robot_x, robot_y, frontier_goal=None):
        """
        Plan a path.  Returns a list of world (x, y) waypoints (start → goal),
        or None if no path is found.

        Priority:
            1. Known target cell (set via set_target())
            2. Frontier goal passed in from FrontierExplorer
        """
        start_cell = self.grid.world_to_grid(robot_x, robot_y)
        if start_cell is None:
            return None

        # Decide goal
        if self.target_cell is not None:
            goal_cell = self.target_cell
        elif frontier_goal is not None:
            goal_cell = self.grid.world_to_grid(frontier_goal[0], frontier_goal[1])
        else:
            return None  # Nothing to plan towards

        if goal_cell is None:
            return None

        # Snap goal to nearest free cell if it landed on an obstacle
        goal_cell = self._nearest_free(goal_cell, max_search=20)
        if goal_cell is None:
            return None

        # Run A*
        cell_path = self._astar(start_cell, goal_cell)
        if cell_path is None:
            return None

        # Smooth and convert to world coordinates
        smoothed  = self._smooth_path(cell_path)
        waypoints = [self.grid.grid_to_world(c, r) for (c, r) in smoothed]
        return waypoints

    # ------------------------------------------------------------------
    # A* core
    # ------------------------------------------------------------------

    def _astar(self, start, goal):
        """
        A* on the inflated occupancy grid.
        Returns list of (col, row) from start to goal, or None.
        """
        inflated = self._get_inflated_grid()

        sc, sr = start
        gc, gr = goal

        # Each heap entry: (f_score, g_score, col, row)
        open_heap = []
        heapq.heappush(open_heap, (0.0, 0.0, sc, sr))

        came_from = {}            # (col, row) → (col, row)
        g_score   = {start: 0.0}

        visited = set()

        while open_heap:
            f, g, c, r = heapq.heappop(open_heap)

            if (c, r) in visited:
                continue
            visited.add((c, r))

            if (c, r) == (gc, gr):
                return self._reconstruct_path(came_from, (gc, gr))

            for dc, dr, move_cost in self._MOVES:
                nc, nr = c + dc, r + dr

                if not self.grid.in_bounds(nc, nr):
                    continue
                if inflated[nr, nc] == OCCUPIED:
                    continue
                # Treat unknown cells as passable but costly
                extra = 2.0 if inflated[nr, nc] == UNKNOWN else 0.0

                tentative_g = g + move_cost + extra

                if tentative_g < g_score.get((nc, nr), float('inf')):
                    g_score[(nc, nr)] = tentative_g
                    came_from[(nc, nr)] = (c, r)
                    h = self._heuristic(nc, nr, gc, gr)
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, nc, nr))

        return None  # No path found

    # ------------------------------------------------------------------
    # Heuristic — Manhattan distance scaled to grid cost
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic(c, r, gc, gr):
        """
        Octile distance: better than Manhattan for 8-connected grids.
        Avoids over-estimation while being tighter than pure Manhattan.
        """
        dx = abs(c - gc)
        dy = abs(r - gr)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    # ------------------------------------------------------------------
    # Path reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Path smoothing — removes redundant intermediate waypoints
    # ------------------------------------------------------------------

    def _smooth_path(self, path, step=4):
        """
        Simple waypoint decimation: keep every Nth cell plus the goal.
        Reduces the number of waypoints the local planner has to follow.
        """
        if len(path) <= 2:
            return path
        kept = path[::step]
        if kept[-1] != path[-1]:
            kept.append(path[-1])
        return kept

    # ------------------------------------------------------------------
    # Obstacle inflation
    # ------------------------------------------------------------------

    def _get_inflated_grid(self):
        """
        Returns a version of the occupancy grid where every OCCUPIED cell
        has been expanded by inflation_cells in all directions.
        Cached — recomputed only when the underlying grid changes.
        """
        import numpy as np
        from scipy.ndimage import binary_dilation

        # Simple version check using total hit count as a proxy for "changed"
        current_version = int(self.grid.grid.sum())
        if self._inflated is not None and current_version == self._grid_version:
            return self._inflated

        occupied_mask = (self.grid.grid == OCCUPIED)

        # Build a circular structuring element for inflation
        r = self.inflation_cells
        size = 2 * r + 1
        y, x = np.ogrid[-r:r+1, -r:r+1]
        struct = (x**2 + y**2 <= r**2)

        inflated_mask = binary_dilation(occupied_mask, structure=struct)

        self._inflated = self.grid.grid.copy()
        self._inflated[inflated_mask] = OCCUPIED

        self._grid_version = current_version
        return self._inflated

    # ------------------------------------------------------------------
    # Snap goal to nearest free cell
    # ------------------------------------------------------------------

    def _nearest_free(self, cell, max_search=20):
        """
        If the goal cell is occupied or unknown, spiral outward to find
        the nearest FREE cell within max_search steps.
        """
        c, r = cell
        if self.grid.is_free(c, r):
            return cell

        for radius in range(1, max_search + 1):
            for dc in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    if abs(dc) != radius and abs(dr) != radius:
                        continue  # only check the ring edge
                    nc, nr = c + dc, r + dr
                    if self.grid.in_bounds(nc, nr) and self.grid.is_free(nc, nr):
                        return (nc, nr)
        return None  # No free cell found nearby