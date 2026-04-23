import numpy as np
import math
from collections import deque

# Cell states
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2

class OccupancyGrid:
    """
    A 2D grid map built incrementally from lidar scans.

    Each cell stores one of three states:
        UNKNOWN  (0) — never seen
        FREE     (1) — lidar beam passed through here
        OCCUPIED (2) — lidar beam ended here (hit something)

    World coordinates (metres) are converted to grid indices via:
        col = int((world_x - origin_x) / resolution)
        row = int((world_y - origin_y) / resolution)
    """

    def __init__(self, width_m=10.0, height_m=10.0, resolution=0.05, origin_x=-5.0, origin_y=-5.0):
        """
        Args:
            width_m:     Physical width  of the map in metres.
            height_m:    Physical height of the map in metres.
            resolution:  Size of one cell in metres (e.g. 0.05 = 5 cm).
            origin_x:    World X of the grid's bottom-left corner.
            origin_y:    World Y of the grid's bottom-left corner.
        """
        self.resolution = resolution
        self.origin_x   = origin_x
        self.origin_y   = origin_y

        self.cols = int(width_m  / resolution)
        self.rows = int(height_m / resolution)

        # Main grid — starts fully unknown
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Hit / miss counters for probabilistic updates (optional but cleaner)
        self._hits   = np.zeros((self.rows, self.cols), dtype=np.int32)
        self._misses = np.zeros((self.rows, self.cols), dtype=np.int32)

        # Thresholds: how many hits/misses before we commit to a state
        self.HIT_THRESHOLD   = 2
        self.MISS_THRESHOLD  = 3

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_grid(self, wx, wy):
        """Convert world (x, y) metres → (col, row) indices. Returns None if outside grid."""
        col = int((wx - self.origin_x) / self.resolution)
        row = int((wy - self.origin_y) / self.resolution)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return col, row
        return None

    def grid_to_world(self, col, row):
        """Convert (col, row) → world (x, y) centre of that cell."""
        wx = self.origin_x + (col + 0.5) * self.resolution
        wy = self.origin_y + (row + 0.5) * self.resolution
        return wx, wy

    def in_bounds(self, col, row):
        return 0 <= col < self.cols and 0 <= row < self.rows

    # ------------------------------------------------------------------
    # Core update — called every scan
    # ------------------------------------------------------------------

    def update(self, robot_x, robot_y, valid_ranges, valid_angles, robot_theta):
        """
        Integrate one lidar scan into the map.

        For every valid beam:
          1. Raytrace from robot to the hit point — mark all cells along
             the ray as FREE (the beam passed through them cleanly).
          2. Mark the endpoint cell as OCCUPIED (the beam hit something).

        Args:
            robot_x, robot_y: Robot world position in metres.
            valid_ranges:     1-D numpy array of valid range readings (m).
            valid_angles:     1-D numpy array of corresponding angles (rad),
                              in the robot's local frame.
            robot_theta:      Robot heading in radians.
        """
        robot_cell = self.world_to_grid(robot_x, robot_y)
        if robot_cell is None:
            return  # Robot outside map — skip

        for i in range(len(valid_ranges)):
            r     = valid_ranges[i]
            angle = robot_theta + valid_angles[i]

            # World position of the beam endpoint
            hit_x = robot_x + r * math.cos(angle)
            hit_y = robot_y + r * math.sin(angle)

            hit_cell = self.world_to_grid(hit_x, hit_y)

            # Raytrace: mark all cells between robot and hit as FREE
            free_cells = self._bresenham(robot_cell[0], robot_cell[1],
                                         hit_cell[0] if hit_cell else
                                         self.world_to_grid(
                                             robot_x + (r - self.resolution) * math.cos(angle),
                                             robot_y + (r - self.resolution) * math.sin(angle)
                                         )[0] if self.world_to_grid(
                                             robot_x + (r - self.resolution) * math.cos(angle),
                                             robot_y + (r - self.resolution) * math.sin(angle)
                                         ) else robot_cell[0],
                                         hit_cell[1] if hit_cell else robot_cell[1])

            for (c, rw) in free_cells[:-1]:  # exclude endpoint
                if self.in_bounds(c, rw):
                    self._misses[rw, c] += 1
                    if self._misses[rw, c] >= self.MISS_THRESHOLD:
                        self.grid[rw, c] = FREE

            # Mark endpoint as OCCUPIED
            if hit_cell:
                c, rw = hit_cell
                self._hits[rw, c] += 1
                if self._hits[rw, c] >= self.HIT_THRESHOLD:
                    self.grid[rw, c] = OCCUPIED

    # ------------------------------------------------------------------
    # Bresenham's line algorithm
    # ------------------------------------------------------------------

    def _bresenham(self, c0, r0, c1, r1):
        """
        Returns a list of (col, row) cells on the line from (c0,r0) to (c1,r1).
        Classic integer rasterisation — fast, no floating point.
        """
        cells = []
        dc = abs(c1 - c0)
        dr = abs(r1 - r0)
        sc = 1 if c0 < c1 else -1
        sr = 1 if r0 < r1 else -1
        err = dc - dr

        c, r = c0, r0
        while True:
            cells.append((c, r))
            if c == c1 and r == r1:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c   += sc
            if e2 < dc:
                err += dc
                r   += sr
        return cells

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self, col, row):
        if self.in_bounds(col, row):
            return int(self.grid[row, col])
        return UNKNOWN

    def is_free(self, col, row):
        return self.get_state(col, row) == FREE

    def is_occupied(self, col, row):
        return self.get_state(col, row) == OCCUPIED

    def is_unknown(self, col, row):
        return self.get_state(col, row) == UNKNOWN


# ======================================================================
# Frontier Explorer
# ======================================================================

class FrontierExplorer:
    """
    Identifies frontier regions (free cells adjacent to unknown cells),
    scores each region by total unknown neighbours, and returns the
    best exploration target for the global path planner.

    Usage:
        explorer = FrontierExplorer(occupancy_grid)
        goal_world_xy = explorer.get_best_goal(robot_x, robot_y)
    """

    # 4-connected neighbours (up/down/left/right)
    _DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 8-connected neighbours (includes diagonals) — used for region grouping
    _DIRS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

    # ------------------------------------------------------------------
    # Step 1: find all frontier cells
    # ------------------------------------------------------------------

    def _find_frontier_cells(self):
        """
        A frontier cell is a FREE cell that has at least one UNKNOWN
        neighbour in the 4-connected sense.

        Returns a set of (col, row) tuples.
        """
        frontier = set()
        g = self.grid

        for row in range(g.rows):
            for col in range(g.cols):
                if g.grid[row, col] != FREE:
                    continue
                # Check 4-connected neighbours for any UNKNOWN
                for dc, dr in self._DIRS4:
                    nc, nr = col + dc, row + dr
                    if g.in_bounds(nc, nr) and g.grid[nr, nc] == UNKNOWN:
                        frontier.add((col, row))
                        break
        return frontier

    # ------------------------------------------------------------------
    # Step 2: group frontier cells into contiguous regions
    # ------------------------------------------------------------------

    def _group_into_regions(self, frontier_cells):
        """
        BFS over the frontier cell set using 8-connectivity.
        Returns a list of regions, where each region is a list of (col, row).
        """
        remaining = set(frontier_cells)
        regions   = []

        while remaining:
            seed    = next(iter(remaining))
            region  = []
            queue   = deque([seed])
            visited = {seed}

            while queue:
                cell = queue.popleft()
                region.append(cell)
                remaining.discard(cell)

                col, row = cell
                for dc, dr in self._DIRS8:
                    nc, nr = col + dc, row + dr
                    nb = (nc, nr)
                    if nb in remaining and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

            regions.append(region)
        return regions

    # ------------------------------------------------------------------
    # Step 3: score each region
    # ------------------------------------------------------------------

    def _score_region(self, region):
        """
        Score = total number of UNKNOWN 4-connected neighbours across
        all cells in the region.

        Higher score → more unexplored area behind this frontier.
        """
        g     = self.grid
        score = 0
        for (col, row) in region:
            for dc, dr in self._DIRS4:
                nc, nr = col + dc, row + dr
                if g.in_bounds(nc, nr) and g.grid[nr, nc] == UNKNOWN:
                    score += 1
        return score

    # ------------------------------------------------------------------
    # Step 4: pick the best goal cell
    # ------------------------------------------------------------------

    def get_best_goal(self, robot_x, robot_y, min_region_size=3):
        """
        Main entry point.  Returns the world (x, y) of the best
        exploration target, or None if no frontiers remain (map complete).

        Strategy:
            1. Find all frontier cells.
            2. Group into regions.
            3. Score each region (sum of unknown neighbours).
            4. Pick the highest-scoring region.
            5. Within that region, pick the cell nearest the robot.

        Args:
            robot_x, robot_y:  Current robot world position.
            min_region_size:   Ignore tiny frontier fragments smaller than this.
        """
        frontier_cells = self._find_frontier_cells()

        if not frontier_cells:
            return None  # Exploration complete — no frontiers left

        regions = self._group_into_regions(frontier_cells)

        # Filter out tiny noise regions
        regions = [r for r in regions if len(r) >= min_region_size]

        if not regions:
            return None

        # Score every region
        scored = [(self._score_region(r), r) for r in regions]

        # Best region = highest score
        best_score, best_region = max(scored, key=lambda x: x[0])

        # Within the best region, pick the cell nearest the robot
        robot_cell = self.grid.world_to_grid(robot_x, robot_y)
        if robot_cell is None:
            robot_cell = (self.grid.cols // 2, self.grid.rows // 2)

        rc, rr = robot_cell
        nearest_cell = min(best_region,
                           key=lambda c: (c[0] - rc) ** 2 + (c[1] - rr) ** 2)

        goal_wx, goal_wy = self.grid.grid_to_world(*nearest_cell)
        return goal_wx, goal_wy

    # ------------------------------------------------------------------
    # Debug helper: returns all scored regions (useful for visualisation)
    # ------------------------------------------------------------------

    def get_all_scored_regions(self, min_region_size=3):
        """
        Returns list of (score, region_cells) sorted best-first.
        Useful for drawing frontier regions on the SLAM visualiser.
        """
        frontier_cells = self._find_frontier_cells()
        if not frontier_cells:
            return []

        regions = self._group_into_regions(frontier_cells)
        regions = [r for r in regions if len(r) >= min_region_size]

        scored = [(self._score_region(r), r) for r in regions]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored