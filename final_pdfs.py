from __future__ import annotations

import math 
import os
import sys
from dataclasses import dataclass

# Set SDL before importing pygame so the simulation can run without a
# visible window: HEADLESS=1 python3 pygame_simulator/final_pdfs.py
HEADLESS = (
    os.environ.get(
        "HEADLESS",
        "0",
    )
    == "1"
)

if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame


# Import the original simulator as a library.  Its own Robot class and SPH
# update functions remain the single implementation of robot behavior.
os.environ["SPH_DFS_LIBRARY_MODE"] = "1"
import single_junction_sph_dfs_environment as environment


# =========================================================
# Map geometry
# =========================================================

# Keep the complete map (geometry, LiDAR ranges, and motion distances) at
# 80% of its previous size while preserving its current screen center.
MAP_SIZE_RATIO = 0.72
BASE_MAP_SCALE = 0.4
MAP_SCALE = BASE_MAP_SCALE * MAP_SIZE_RATIO
REFERENCE_MAP_SCALE = 0.5
SIMULATION_LENGTH_SCALE = (
    MAP_SCALE
    / REFERENCE_MAP_SCALE
)
REFERENCE_ORIGIN = pygame.Vector2(78, 39)
REFERENCE_MAP_CENTER = pygame.Vector2(
    0.5 * (78 + 524),
    0.5 * (39 + 500),
)
MAP_ORIGIN = (
    pygame.Vector2(205, 55)
    + (REFERENCE_MAP_CENTER - REFERENCE_ORIGIN)
    * (BASE_MAP_SCALE - MAP_SCALE)
)

REFERENCE_OUTER_BOUNDARY = (
    (208, 39),
    (524, 39),
    (524, 346),
    (302, 346),
    (302, 500),
    (208, 500),
    (208, 346),
    (78, 346),
    (78, 253),
    (208, 253),
)

REFERENCE_OBSTACLE = pygame.Rect(302, 133, 133, 120)


def scale_point(point: tuple[int, int]) -> tuple[int, int]:
    position = (
        MAP_ORIGIN
        + (pygame.Vector2(point) - REFERENCE_ORIGIN) * MAP_SCALE
    )

    return round(position.x), round(position.y)


OUTER_BOUNDARY = tuple(
    scale_point(point)
    for point in REFERENCE_OUTER_BOUNDARY
)

OBSTACLE_RECT = pygame.Rect(
    scale_point(REFERENCE_OBSTACLE.topleft),
    (
        round(REFERENCE_OBSTACLE.width * MAP_SCALE),
        round(REFERENCE_OBSTACLE.height * MAP_SCALE),
    ),
)

BASE_POSITION = pygame.Vector2(
    scale_point((255, 500))
)


BASE_CORRIDOR_TOP_LEFT = scale_point((208, 346))
BASE_CORRIDOR_BOTTOM_RIGHT = scale_point((302, 500))

BASE_CORRIDOR_RECT = pygame.Rect(
    BASE_CORRIDOR_TOP_LEFT,
    (
        BASE_CORRIDOR_BOTTOM_RIGHT[0]
        - BASE_CORRIDOR_TOP_LEFT[0],

        BASE_CORRIDOR_BOTTOM_RIGHT[1]
        - BASE_CORRIDOR_TOP_LEFT[1],
    ),
)


# The Anchor has the same 360-degree ray sensor used by the LiDAR reference
# simulation.  Only the Anchor owns this long-range sensor.
LIDAR_RAYS = 360
LIDAR_MAX_RANGE = 150.0 * SIMULATION_LENGTH_SCALE
W_TAU_NOISE_FRACTION = 0.05
W_TAU_ALPHA = 0.30
ADAPTIVE_W_MARGIN_RATIO = 0.05
W_TAU_MERGE_GAP_DEG = 3.0
W_TAU_MIN_OPENING_WIDTH_DEG = 5.0
W_TAU_BOUNDARY_SEARCH_DEG = 6.0
W_TAU_GRADIENT_MAD_SCALE = 4.0
W_TAU_MIN_GRADIENT_THRESHOLD = 0.05
ANCHOR_YAW_DEG = -90.0
ANCHOR_FORWARD_SPEED = 24.0 * SIMULATION_LENGTH_SCALE

# Junction center로 들어가는 최대 전후 속도
ANCHOR_CENTERING_SPEED = 48.0 * SIMULATION_LENGTH_SCALE

ANCHOR_BRANCH_ENTRY_MARGIN = 5.0 * SIMULATION_LENGTH_SCALE

# Base corridor 추종용
ANCHOR_LATERAL_GAIN = 0.8
ANCHOR_MAX_LATERAL_SPEED = 6.0 * SIMULATION_LENGTH_SCALE

# Junction center 이동에서만 사용할 lateral limit
ANCHOR_CENTER_MAX_LATERAL_SPEED = 10.0 * SIMULATION_LENGTH_SCALE

# Junction center로 접근할 때 local position error gain
ANCHOR_LOCAL_CENTER_GAIN = 7.0

ANCHOR_LOCAL_CENTER_TOLERANCE = 1.00 * SIMULATION_LENGTH_SCALE
ANCHOR_CENTER_STABLE_SCANS = 3


ANCHOR_ENTRANCE_STATIONARY_SCANS = 5
LATERAL_BASELINE_SAMPLES = 10
TAU = LIDAR_MAX_RANGE * W_TAU_NOISE_FRACTION
LATERAL_RANGE_JUMP_THRESHOLD = 2.0 * TAU
STATIONARY_MIN_PERSISTENT_OBSERVATIONS = 24
STATIONARY_PERSISTENCE_RATIO = 0.60
STATIONARY_ASSOCIATION_TOLERANCE_DEG = max(
    2.0 * W_TAU_MERGE_GAP_DEG,
    W_TAU_MIN_OPENING_WIDTH_DEG,
)
ANCHOR_TARGET_SAMPLE_TOLERANCE = 2.0 * SIMULATION_LENGTH_SCALE
FRONT_OPENING_MAX_ABS_ANGLE = 45.0

KNOWN_CORRIDOR_WIDTH = (302.0 - 208.0) * MAP_SCALE
KNOWN_JUNCTION_DEPTH = (346.0 - 253.0) * MAP_SCALE

FINAL_SIDE_MERGE_STABLE_SCANS = 8
FINAL_BASE_RETURN_STABLE_SCANS = 8

KNOWN_BASE_CORRIDOR_LENGTH = (
    (500.0 - 346.0)
    * MAP_SCALE
)

ROOT_CENTER_TO_BASE_DISTANCE = (
    0.5 * KNOWN_JUNCTION_DEPTH
    + KNOWN_BASE_CORRIDOR_LENGTH
)

# =========================================================
# Anchor branch exploration
# =========================================================

ANCHOR_EXPLORE_HALF_FOV_DEG = 90.0

# 단순히 옆 벽까지 보이는 정도는 free path로 인정하지 않는다.
# corridor width의 60% 이상 열린 방향만 진행 가능한 gap으로 본다.
ANCHOR_TRAVERSABLE_RANGE = (
    0.60 * KNOWN_CORRIDOR_WIDTH
)

ANCHOR_MIN_FREE_GAP_WIDTH_DEG = 12.0

ANCHOR_MAX_TURN_RATE_DEG = 140.0

ANCHOR_MIN_EXPLORE_SPEED = 3.0 * SIMULATION_LENGTH_SCALE

DEAD_END_CONFIRM_FRAMES = 3

# =========================================================
# Anchor - swarm front following during branch exploration
# =========================================================

# Anchor가 가장 앞선 NORMAL보다
# 몇 GRID_SPACING 앞에 있도록 할 것인지.
ANCHOR_FRONT_TARGET_GAP_ROWS = 2.0

# 거리 오차를 Anchor 속도 보정으로 바꾸는 gain.
ANCHOR_FRONT_GAP_GAIN = 1.5

# 현재 Branch corridor로 볼 lateral 범위.
ANCHOR_FRONT_LATERAL_MARGIN_ROWS = 1.0

# 너무 멀리 있는 robot은 현재 front 후보에서 제외.
ANCHOR_FRONT_OBSERVATION_HOPS = 2.0

# =========================================================
# Marker detection
# =========================================================

MARKER_DETECTION_RANGE = (
    LIDAR_MAX_RANGE
)

MARKER_HALF_ANGLE_DEG = 20.0

MARKER_LINE_OF_SIGHT_MARGIN = 2.0 * SIMULATION_LENGTH_SCALE

# Branch 탐색 시작 직후 Root Junction의 sibling Marker를
# 잘못 보는 것을 막기 위한 local-odometry activation distance.
MARKER_ACTIVATION_DISTANCE = (
    1.0 * KNOWN_CORRIDOR_WIDTH
)

# =========================================================
# Backtracking Shepherd
# =========================================================

# Shepherd chain에서 이웃 로봇 사이의 목표 간격.
# world 좌표 target이 아니라 robot-to-robot relative spacing.
BACKTRACK_SHEPHERD_SPACING_RATIO = 1.0

# =========================================================
# Segmented Backtracking:
# straight PUSH -> corner RELEASE -> turn -> re-recruit
# =========================================================

# 현재 return 방향 정면이 막히고,
# free gap이 이 각도 이상 옆으로 꺾이면 corner로 본다.
BACKTRACK_CORNER_MIN_TURN_DEG = 30.0

# 코너를 돈 뒤 새 corridor와 거의 정렬되었다고 보는 각도
BACKTRACK_CORNER_EXIT_ANGLE_DEG = 12.0

# 몇 frame 연속 새 corridor가 안정적으로 보여야
# 새 Shepherd 모집을 시작할지
BACKTRACK_CORNER_EXIT_STABLE_SCANS = 3

# 기존보다 강하게 밀기
BACKTRACK_PUSH_SPEED_RATIO = 1.00
# Shepherd cohort가 wall 때문에 return 방향으로 진행할 수 없을 때
# 전체 topology를 유지한 채 lateral 방향으로 이동하는 속도.
BACKTRACK_WALL_DETACH_SPEED_RATIO = 0.35

# Shepherd motor drive가 NORMAL과 실제 접촉했을 때
# 전달되는 물리적 contact acceleration의 최대 비율.
#
# NORMAL에게 Junction 방향 goal force를 주는 것이 아니라
# robot-to-robot collision response이다.
BACKTRACK_CONTACT_ACCEL_RATIO = 2.5
# =========================================================
# Dead-end wall Shepherd
# =========================================================

# Dead-end wall에서 첫 번째 robot layer로 인정할 깊이.
# wall과 거의 붙어 있는 한 층만 사용.
BACKTRACK_DEAD_END_ONE_HOP_DEPTH = (
    environment.ROBOT_RADIUS
    + 0.5 * environment.GRID_SPACING
)

# WL~WR coverage에서 허용할 아주 작은 수치 오차.
BACKTRACK_DEAD_END_GAP_TOLERANCE = (
    0.25 * environment.ROBOT_RADIUS
)

# =========================================================
# Dead-end wall based Shepherd recruitment
# =========================================================

# wall-contact seed에서 몇 robot-hop까지 Shepherd로 포함할지.
#
# 1:
#   wall 바로 앞 layer + 그 뒤 1-hop
#
# 2:
#   wall 바로 앞 layer + 1-hop + 2-hop
#
# 우선 1로 시작.
BACKTRACK_DEAD_END_WALL_HOPS = 1

# LiDAR wall과 robot center 사이의 여유가
# 이 정도 이내이면 wall-contact seed로 본다.
BACKTRACK_WALL_SEED_GAP = (
    environment.ROBOT_RADIUS
    + 1.0 * environment.GRID_SPACING
)

# robot-hop 판정용 물리적 이웃 거리.
# COMM_RANGE 전체를 쓰지 않는다.
BACKTRACK_WALL_HOP_RADIUS = max(
    2.2 * environment.ROBOT_RADIUS,
    1.6 * environment.GRID_SPACING,
)

# =========================================================
# Pressure Push -> Flow Backtracking transition
# =========================================================

# NORMAL robot이 Junction 방향으로 실제 이동한다고
# 인정할 최소 Anchor-local reverse speed.
#
# Anchor-local:
# +x = 탐색 진행방향
# -x = Junction 복귀방향
BACKTRACK_REVERSE_SPEED_THRESHOLD = 0.5 * SIMULATION_LENGTH_SCALE

# 평가 대상 NORMAL 중 이 비율 이상이
# 실제 Junction 방향으로 움직여야 reverse flow 인정.
BACKTRACK_REVERSE_RATIO_THRESHOLD = 0.60

# 너무 적은 robot만 보고 flow라고 판단하지 않도록
# 최소 평가 robot 수.
BACKTRACK_REVERSE_MIN_ROBOTS = 8

# reverse flow가 연속 몇 frame 유지되어야
# FLOW_BACKTRACK으로 넘어가는지.
BACKTRACK_REVERSE_STABLE_SCANS = 3

SWARM_RETURN_STABLE_SCANS = 3

# Shepherd wall 뒤쪽 몇 communication range까지
# reverse-flow 평가 대상으로 볼 것인지.
BACKTRACK_FLOW_EVAL_HOPS = 3.0

# Initial Shepherd formation
ENTRANCE_CAPTURE_MARGIN = 2.0 * SIMULATION_LENGTH_SCALE

# Branch entrance보다 Junction 쪽에서
# 얼마나 일찍 Initial Shepherd로 인정할지
ENTRANCE_DEPTH_CAPTURE_MARGIN = (
    5.0 * SIMULATION_LENGTH_SCALE
)

INITIAL_SHEPHERD_HOPS = 3

# Branch 안으로 가장 멀리 팽창한 front edge를
# 한 개의 seed layer로 묶을 depth band.
# staggered arrangement를 고려해 약 1.5 row 사용.
INITIAL_SHEPHERD_FRONT_BAND_ROWS = 1.5

INITIAL_SHEPHERD_HOP_DEPTH_TOLERANCE_RATIO = 0.5

FRONT_EDGE_SEARCH_DEG = 60.0
FRONT_MOUTH_WIDTH_MIN_RATIO = 0.65
FRONT_MOUTH_WIDTH_MAX_RATIO = 1.35
FRONT_MOUTH_LATERAL_TOLERANCE = KNOWN_CORRIDOR_WIDTH * 0.35


# This is selected from the initial deployment's row/column topology, before
# the simulation begins.  It is not inferred from runtime world positions.
INITIAL_ANCHOR_ID: int | None = None

_ROLE_DEBUG_LAST: dict[str, str] = {}


def role_debug(key: str, message: str) -> None:
    """Print role diagnostics only when the state represented by a key changes."""
    if _ROLE_DEBUG_LAST.get(key) != message:
        _ROLE_DEBUG_LAST[key] = message
        print(message)


def compute_adaptive_worst_wall_range(
    left_range: float | None,
    right_range: float | None,
) -> float | None:
    if (
        left_range is None
        or right_range is None
        or not math.isfinite(left_range)
        or not math.isfinite(right_range)
        or left_range <= 0.0
        or right_range <= 0.0
    ):
        return None
    return math.hypot(left_range + right_range, max(left_range, right_range))


def select_adaptive_w_tau_threshold(adaptive_w: float | None) -> float | None:
    if adaptive_w is None:
        return None
    lower = adaptive_w * (1.0 + ADAPTIVE_W_MARGIN_RATIO)
    upper = LIDAR_MAX_RANGE * (1.0 - W_TAU_NOISE_FRACTION)
    if lower >= upper:
        return None
    return lower + W_TAU_ALPHA * (upper - lower)


def _rect_wall_segments(rectangle: pygame.Rect) -> tuple[tuple[pygame.Vector2, pygame.Vector2], ...]:
    """Return the four wall segments of a rectangular obstacle."""
    top_left = pygame.Vector2(rectangle.topleft)
    top_right = pygame.Vector2(rectangle.topright)
    bottom_right = pygame.Vector2(rectangle.bottomright)
    bottom_left = pygame.Vector2(rectangle.bottomleft)
    return (
        (top_left, top_right),
        (top_right, bottom_right),
        (bottom_right, bottom_left),
        (bottom_left, top_left),
    )


LIDAR_WALL_SEGMENTS = tuple(
    (
        pygame.Vector2(OUTER_BOUNDARY[index]),
        pygame.Vector2(OUTER_BOUNDARY[(index + 1) % len(OUTER_BOUNDARY)]),
    )
    for index in range(len(OUTER_BOUNDARY))
) + _rect_wall_segments(OBSTACLE_RECT)


@dataclass(frozen=True)
class LidarScan:
    """Anchor-local angle/range observation, independent of map coordinates."""

    angles_deg: tuple[float, ...]
    ranges: tuple[float, ...]
    max_range: float


@dataclass(frozen=True)
class LocalObservation:
    """The runtime boundary: relative swarm state plus one Anchor LiDAR scan."""

    timestamp: float
    robot_ids: tuple[int, ...]
    relative_positions: tuple[pygame.Vector2, ...]
    velocities: tuple[pygame.Vector2, ...]
    lidar_robot_id: int
    lidar_scan: LidarScan


def smooth_ranges(ranges: tuple[float, ...] | list[float], window_size: int = 5) -> tuple[float, ...]:
    """Circular moving-average smoothing from the LiDAR reference pipeline."""
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    half = window_size // 2
    count = len(ranges)
    return tuple(
        sum(ranges[(index + offset) % count] for offset in range(-half, half + 1))
        / window_size
        for index in range(count)
    )


def circular_range_gradient(scan: LidarScan) -> tuple[float, ...]:
    """Forward circular range gradient used to identify opening boundaries."""
    count = len(scan.angles_deg)
    return tuple(
        scan.ranges[(index + 1) % count] - scan.ranges[index]
        for index in range(count)
    )


def extract_lateral_wall_ranges(scan: LidarScan, window_size: int = 5) -> tuple[float, float]:
    """Return body-relative left (-90°) and right (+90°) LiDAR ranges."""
    smoothed = smooth_ranges(scan.ranges, window_size)

    def lateral_median(target_angle: float) -> float:
        nearby = [
            measured_range
            for angle, measured_range in zip(scan.angles_deg, smoothed)
            if abs(((angle - target_angle + 180.0) % 360.0) - 180.0) <= 2.0
        ]
        return sorted(nearby)[len(nearby) // 2]

    right_range = lateral_median(90.0)
    left_range = lateral_median(-90.0)

    return left_range, right_range


class AnchorLidar:
    """Local 360-degree LiDAR adapted from the attached LiDAR simulator."""

    def __init__(self) -> None:
        self.angles = tuple(
            -180.0 + 360.0 * index / LIDAR_RAYS
            for index in range(LIDAR_RAYS)
        )
        self.ranges = [LIDAR_MAX_RANGE] * LIDAR_RAYS
        self.open_support = [False] * LIDAR_RAYS
        self.opening_groups: list[dict] = []
        self.last_valid_left_wall_range: float | None = None
        self.last_valid_right_wall_range: float | None = None
        self.last_valid_adaptive_w: float | None = None
        self.active_adaptive_w: float | None = None
        self.selected_w_tau_threshold: float | None = None
        self.threshold_interval_valid = False
        self.threshold_locked = False
        self.locked_w_tau_threshold: float | None = None
        self.locked_adaptive_w: float | None = None
        self.junction_evidence = False
        self.lateral_range_history: list[tuple[float, float]] = []
        self.lateral_baseline_left: float | None = None
        self.lateral_baseline_right: float | None = None
        self.stationary_samples = 0
        self.stationary_tracks: list[dict] = []
        self.last_scan = LidarScan(
            self.angles,
            tuple(self.ranges),
            LIDAR_MAX_RANGE,
        )

    @staticmethod
    def _ray_hit(
        origin: pygame.Vector2,
        direction: pygame.Vector2,
        segment: tuple[pygame.Vector2, pygame.Vector2],
    ) -> float | None:
        start, end = segment
        edge = end - start
        denominator = direction.x * edge.y - direction.y * edge.x
        if abs(denominator) < 1.0e-10:
            return None

        offset = start - origin
        ray_t = (offset.x * edge.y - offset.y * edge.x) / denominator
        segment_t = (offset.x * direction.y - offset.y * direction.x) / denominator
        if ray_t >= 0.0 and -1.0e-9 <= segment_t <= 1.0 + 1.0e-9:
            return ray_t
        return None

    def scan(self, position: pygame.Vector2, yaw_degrees: float) -> LidarScan:
        """Measure wall ranges using only the Anchor's local pose and rays."""
        ranges: list[float] = []
        for angle in self.angles:
            radians = math.radians(yaw_degrees + angle)
            direction = pygame.Vector2(math.cos(radians), math.sin(radians))
            hits = (
                self._ray_hit(position, direction, wall)
                for wall in LIDAR_WALL_SEGMENTS
            )
            nearest = min(
                (hit for hit in hits if hit is not None),
                default=LIDAR_MAX_RANGE,
            )
            ranges.append(min(LIDAR_MAX_RANGE, nearest))
        self.ranges = ranges
        self.last_scan = LidarScan(
            self.angles,
            tuple(ranges),
            LIDAR_MAX_RANGE,
        )
        left_range, right_range = extract_lateral_wall_ranges(
            self.last_scan,
            window_size=5,
        )
        current_w = None
        if (
            left_range < LIDAR_MAX_RANGE - 1.0
            and right_range < LIDAR_MAX_RANGE - 1.0
        ):
            current_w = compute_adaptive_worst_wall_range(
                left_range,
                right_range,
            )
        if not self.threshold_locked and current_w is not None:
            self.last_valid_left_wall_range = left_range
            self.last_valid_right_wall_range = right_range
            self.last_valid_adaptive_w = current_w
        if self.threshold_locked:
            self.active_adaptive_w = self.locked_adaptive_w
            self.selected_w_tau_threshold = self.locked_w_tau_threshold
        else:
            self.active_adaptive_w = (
                current_w if current_w is not None else self.last_valid_adaptive_w
            )
            self.selected_w_tau_threshold = select_adaptive_w_tau_threshold(
                self.active_adaptive_w
            )
        self.threshold_interval_valid = self.selected_w_tau_threshold is not None
        self._classify_openings()
        self.junction_evidence = (
            self.threshold_interval_valid and len(self.opening_groups) >= 3
        )
        return self.last_scan

    def _classify_openings(self) -> None:
        """Infer opening sectors solely from the Anchor's local scan."""
        smoothed = smooth_ranges(self.last_scan.ranges, window_size=5)
        threshold = self.selected_w_tau_threshold
        if threshold is None:
            self.open_support = [False] * LIDAR_RAYS
            self.opening_groups = []
            return

        self.open_support = [value >= threshold for value in smoothed]
        gradients = [
            smoothed[(index + 1) % LIDAR_RAYS] - smoothed[index]
            for index in range(LIDAR_RAYS)
        ]
        self.opening_groups = []

        for start in range(LIDAR_RAYS):
            if not self.open_support[start] or self.open_support[(start - 1) % LIDAR_RAYS]:
                continue
            indices = [start]
            cursor = (start + 1) % LIDAR_RAYS
            while self.open_support[cursor] and cursor != start:
                indices.append(cursor)
                cursor = (cursor + 1) % LIDAR_RAYS
            if len(indices) < 5:
                continue

            coarse_start = (indices[0] - 1) % LIDAR_RAYS
            coarse_end = indices[-1] % LIDAR_RAYS
            start_candidates = [
                (coarse_start + delta) % LIDAR_RAYS
                for delta in range(-6, 7)
            ]
            end_candidates = [
                (coarse_end + delta) % LIDAR_RAYS
                for delta in range(-6, 7)
            ]
            start_index = max(start_candidates, key=lambda index: gradients[index])
            end_index = (min(end_candidates, key=lambda index: gradients[index]) + 1) % LIDAR_RAYS
            start_angle = self.angles[start_index]
            end_angle = self.angles[end_index]
            width = (end_angle - start_angle) % 360.0
            if width < 5.0 or width >= 359.0:
                continue
            center_angle = ((start_angle + width / 2.0 + 180.0) % 360.0) - 180.0
            start_rad = math.radians(start_angle)
            end_rad = math.radians(end_angle)
            mouth_a = pygame.Vector2(
                smoothed[start_index] * math.cos(start_rad),
                smoothed[start_index] * math.sin(start_rad),
            )
            mouth_b = pygame.Vector2(
                smoothed[end_index] * math.cos(end_rad),
                smoothed[end_index] * math.sin(end_rad),
            )
            mouth_vector = mouth_b - mouth_a
            if mouth_vector.length_squared() <= environment.EPSILON:
                continue

            mouth_tangent = mouth_vector.normalize()
            branch_axis = pygame.Vector2(
                -mouth_tangent.y,
                mouth_tangent.x,
            )
            center_rad = math.radians(center_angle)
            opening_direction = pygame.Vector2(
                math.cos(center_rad),
                math.sin(center_rad),
            )
            if branch_axis.dot(opening_direction) < 0.0:
                branch_axis *= -1.0

            self.opening_groups.append(
                {
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                    "center_angle": center_angle,
                    "mouth_a": mouth_a,
                    "mouth_b": mouth_b,
                    "mouth_midpoint": (mouth_a + mouth_b) * 0.5,
                    "mouth_tangent": mouth_tangent,
                    "branch_axis": branch_axis,
                }
            )


def world_offset_to_anchor_local(
    offset: pygame.Vector2,
    yaw_degrees: float,
) -> pygame.Vector2:
    """Rotate a world-frame offset into Anchor forward/right coordinates."""
    yaw = math.radians(yaw_degrees)
    forward = pygame.Vector2(math.cos(yaw), math.sin(yaw))
    right = pygame.Vector2(-math.sin(yaw), math.cos(yaw))
    return pygame.Vector2(offset.dot(forward), offset.dot(right))


def anchor_local_to_world(
    local_vector: pygame.Vector2,
    yaw_degrees: float,
) -> pygame.Vector2:
    """Rotate an Anchor forward/right vector back into world coordinates."""
    yaw = math.radians(yaw_degrees)
    forward = pygame.Vector2(math.cos(yaw), math.sin(yaw))
    right = pygame.Vector2(-math.sin(yaw), math.cos(yaw))
    return forward * local_vector.x + right * local_vector.y

def physical_is_walkable(
    position: pygame.Vector2,
    radius: float,
) -> bool:
    """
    실제 지도 polygon + obstacle만 이용한 충돌 판정.

    UP / LEFT / RIGHT / BOTTOM 같은
    사전 정의 region은 이동 가능 여부에 사용하지 않는다.
    """

    x = int(
        round(position.x)
    )

    y = int(
        round(position.y)
    )

    # `radius`는 SPH/robot model의 반경이다. 화면에서는 더 큰 원과
    # 두께가 있는 벽을 그리므로, 충돌은 그려진 robot 외곽이 벽의
    # 안쪽 면에 접촉하는 거리에서 멈추도록 더 큰 접촉 반경을 쓴다.
    contact_radius = max(
        radius,
        WALL_CONTACT_RADIUS,
    )

    pixel_radius = max(
        1,
        int(math.ceil(contact_radius)),
    )

    diagonal = int(
        round(
            pixel_radius
            / math.sqrt(2.0)
        )
    )

    test_points = [
        (x, y),

        (
            x + pixel_radius,
            y,
        ),

        (
            x - pixel_radius,
            y,
        ),

        (
            x,
            y + pixel_radius,
        ),

        (
            x,
            y - pixel_radius,
        ),

        (
            x + diagonal,
            y + diagonal,
        ),

        (
            x + diagonal,
            y - diagonal,
        ),

        (
            x - diagonal,
            y + diagonal,
        ),

        (
            x - diagonal,
            y - diagonal,
        ),
    ]

    mask = (
        environment.walkable_mask
    )

    mask_width, mask_height = (
        mask.get_size()
    )

    for px, py in test_points:

        if not (
            0 <= px < mask_width
            and
            0 <= py < mask_height
        ):
            return False

        if (
            mask.get_at(
                (
                    px,
                    py,
                )
            )
            == 0
        ):
            return False

    return True



def build_local_observation(
    anchor: environment.Robot,
    robots: list[environment.Robot],
    lidar: AnchorLidar,
    anchor_yaw_degrees: float,
) -> LocalObservation:
    """Expose only Anchor-relative swarm state and the latest local scan."""
    return LocalObservation(
        timestamp=environment.simulation_time,
        robot_ids=tuple(robot.robot_id for robot in robots),
        relative_positions=tuple(
            world_offset_to_anchor_local(
                robot.position - anchor.position,
                anchor_yaw_degrees,
            )
            for robot in robots
        ),
        velocities=tuple(
            world_offset_to_anchor_local(
                robot.observed_velocity,
                anchor_yaw_degrees,
            )
            for robot in robots
        ),
        lidar_robot_id=anchor.robot_id,
        lidar_scan=lidar.last_scan,
    )


class LocalObservationBuilder:
    """Explicit simulator-ground-truth to local-runtime observation boundary."""

    @staticmethod
    def build(
        anchor: environment.Robot,
        robots: list[environment.Robot],
        lidar: AnchorLidar,
        anchor_yaw_degrees: float,
    ) -> LocalObservation:
        return build_local_observation(
            anchor,
            robots,
            lidar,
            anchor_yaw_degrees,
        )


def lidar_detects_junction(lidar: AnchorLidar) -> bool:
    """Junction evidence from the adaptive W-tau detector only."""
    return bool(lidar.junction_evidence)


def circular_error(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def lidar_range_at_local_angle(scan: LidarScan, target_angle: float) -> float:
    index = min(
        range(len(scan.angles_deg)),
        key=lambda item: circular_error(scan.angles_deg[item], target_angle),
    )
    return float(scan.ranges[index])

def extract_dead_end_wall_segment(
    scan: LidarScan,
) -> tuple[
    pygame.Vector2,
    pygame.Vector2,
] | None:
    """
    Dead-end에서 Anchor-local 좌표로
    전방 wall의 왼쪽/오른쪽 끝점을 생성한다.

    +x = Anchor forward
    -y = left
    +y = right
    """

    front_range = (
        lidar_range_at_local_angle(
            scan,
            0.0,
        )
    )

    (
        left_range,
        right_range,
    ) = (
        extract_lateral_wall_ranges(
            scan
        )
    )

    if (
        not math.isfinite(front_range)
        or not math.isfinite(left_range)
        or not math.isfinite(right_range)
    ):
        return None

    if (
        front_range
        >= scan.max_range - 1.0
        or left_range
        >= scan.max_range - 1.0
        or right_range
        >= scan.max_range - 1.0
    ):
        return None

    # Anchor-local dead-end wall endpoints
    wall_left = pygame.Vector2(
        front_range,
        -left_range,
    )

    wall_right = pygame.Vector2(
        front_range,
        right_range,
    )

    return (
        wall_left,
        wall_right,
    )

def reset_junction_entrance_detector(
    lidar: AnchorLidar,
) -> None:
    """
    새로운 corridor에서 Junction entrance를 다시 찾기 위해
    이전 corridor의 lateral baseline을 제거한다.

    world/map position은 사용하지 않는다.
    """

    lidar.lateral_range_history.clear()

    lidar.lateral_baseline_left = (
        None
    )

    lidar.lateral_baseline_right = (
        None
    )


def update_junction_entrance_detector(
    lidar: AnchorLidar,
) -> tuple[bool, float, float, float, float]:
    left = lidar_range_at_local_angle(lidar.last_scan, -90.0)
    right = lidar_range_at_local_angle(lidar.last_scan, 90.0)
    history = lidar.lateral_range_history
    if len(history) < LATERAL_BASELINE_SAMPLES:
        history.append((left, right))
        return False, left, right, 0.0, 0.0
    window = history[-LATERAL_BASELINE_SAMPLES:]
    baseline_left = sorted(value[0] for value in window)[LATERAL_BASELINE_SAMPLES // 2]
    baseline_right = sorted(value[1] for value in window)[LATERAL_BASELINE_SAMPLES // 2]
    delta_left, delta_right = left - baseline_left, right - baseline_right
    reached = (
        delta_left > LATERAL_RANGE_JUMP_THRESHOLD
        and delta_right > LATERAL_RANGE_JUMP_THRESHOLD
    )
    if reached:
        lidar.lateral_baseline_left = baseline_left
        lidar.lateral_baseline_right = baseline_right
    else:
        history.append((left, right))
        if len(history) > 30:
            del history[:-30]
    return reached, left, right, delta_left, delta_right


def freeze_stationary_threshold(lidar: AnchorLidar) -> None:
    adaptive_w = compute_adaptive_worst_wall_range(
        lidar.lateral_baseline_left,
        lidar.lateral_baseline_right,
    )
    threshold = select_adaptive_w_tau_threshold(adaptive_w)
    if adaptive_w is None or threshold is None:
        raise RuntimeError("Missing lateral corridor baseline.")
    lidar.threshold_locked = True
    lidar.locked_adaptive_w = lidar.active_adaptive_w = adaptive_w
    lidar.locked_w_tau_threshold = lidar.selected_w_tau_threshold = threshold
    lidar.stationary_samples = 0
    lidar.stationary_tracks.clear()

def update_stationary_junction_confirmation(
    lidar: AnchorLidar,
) -> bool:

    lidar.stationary_samples += 1

    for opening in lidar.opening_groups:

        center = opening["center_angle"]

        match = next(
            (
                track
                for track in lidar.stationary_tracks
                if circular_error(
                    center,
                    track["center"],
                )
                <= STATIONARY_ASSOCIATION_TOLERANCE_DEG
            ),
            None,
        )

        if match is None:

            lidar.stationary_tracks.append(
                {
                    "center": center,
                    "observations": 1,
                }
            )

        else:

            match["observations"] += 1

    persistent_tracks = [
        track
        for track in lidar.stationary_tracks
        if (
            track["observations"]
            >= STATIONARY_MIN_PERSISTENT_OBSERVATIONS
            and
            track["observations"]
            / lidar.stationary_samples
            >= STATIONARY_PERSISTENCE_RATIO
        )
    ]

    if lidar.stationary_samples % 10 == 0:

        left_range = lidar_range_at_local_angle(
            lidar.last_scan,
            -90.0,
        )

        front_range = lidar_range_at_local_angle(
            lidar.last_scan,
            0.0,
        )

        right_range = lidar_range_at_local_angle(
            lidar.last_scan,
            90.0,
        )

        print(
            "[StationaryJunctionDebug] "
            f"samples={lidar.stationary_samples} "
            f"T={lidar.selected_w_tau_threshold:.2f} "
            f"ranges="
            f"L:{left_range:.2f},"
            f"F:{front_range:.2f},"
            f"R:{right_range:.2f} "
            f"opening_count={len(lidar.opening_groups)} "
            f"opening_angles="
            f"{[round(o['center_angle'], 1) for o in lidar.opening_groups]} "
            f"tracks="
            f"{[(round(t['center'], 1), t['observations']) for t in lidar.stationary_tracks]} "
            f"persistent={len(persistent_tracks)}"
        )

    return len(persistent_tracks) >= 3


def stop_anchor(anchor: environment.Robot) -> None:
    """Clear every dynamic state of the independently controlled Anchor."""
    anchor.velocity.update(0.0, 0.0)
    anchor.acceleration.update(0.0, 0.0)
    anchor.commanded_velocity.update(0.0, 0.0)
    anchor.observed_velocity.update(0.0, 0.0)


def anchor_command_name(
    motion_mode: str,
) -> str:
    """Return the command that the Anchor is currently executing."""
    return {
        "ROOT_SETUP": "FOLLOW_CORRIDOR",
        "BRANCH_ENTRY": "ENTER_BRANCH",
        "BRANCH_EXPLORE": "EXPLORE_BRANCH",
        "BACKTRACK_WAIT_SHEPHERD": "RECRUIT_SHEPHERD",
        "PRESSURE_PUSH": "PUSH_SHEPHERD",
        "FLOW_BACKTRACK": "RETURN_TO_JUNCTION",
        "BACKTRACK_WALL_DETACH": "DETACH_SHEPHERD_FROM_WALL",
        "BACKTRACK_CORNER_TURN": "TURN_RETURN_CORNER",
        "RETURN_JUNCTION_ENTRANCE": "HOLD_AT_JUNCTION_MOUTH",
        "RETURN_CENTER_ESTIMATE": "ESTIMATE_JUNCTION_CENTER",
        "RETURN_CENTER_TRANSIT": "MOVE_TO_JUNCTION_CENTER",
        "RETURN_JUNCTION_CENTERED": "HOLD_AT_JUNCTION_CENTER",
        "WAIT_SWARM_RETURN": "WAIT_FOR_SWARM_RETURN",
        "FINALIZE_BRANCH_RETURN": "RELEASE_BACKTRACK_SHEPHERD",
        "REFORM_VISITED_BRANCH_SHEPHERD": "REFORM_ENTRANCE_SHEPHERD",
        "SELECT_NEXT_BRANCH": "SELECT_NEXT_BRANCH",
        "ROOT_COMPLETE": "PREPARE_FINAL_RETURN",
        "FINAL_RETURN_PREP": "ASSIGN_FINAL_PUSH",
        "FINAL_WAIT_SIDE_MERGE": "WAIT_SIDE_SWARM_MERGE",
        "FINAL_BASE_PUSH": "FINAL_BASE_PUSH",
        "FINAL_ANCHOR_RETURN": "RETURN_ANCHOR_TO_BASE",
        "SYSTEM_COMPLETE": "SYSTEM_STOP",
    }.get(
        motion_mode,
        "UNKNOWN",
    )


def integrate_anchor_local_command(
    anchor: environment.Robot,
    local_velocity: pygame.Vector2,
    yaw_degrees: float,
    dt: float,
) -> pygame.Vector2:
    """Actuate a local command and return actual local-frame odometry."""
    previous_position = anchor.position.copy()
    world_velocity = anchor_local_to_world(local_velocity, yaw_degrees)
    next_position = anchor.position + world_velocity * dt
    anchor.previous_position.update(anchor.position)

    if physical_is_walkable(
    next_position,
    anchor.radius,
    ):
        anchor.position.update(next_position)
        anchor.velocity.update(world_velocity)
    else:
        anchor.velocity.update(0.0, 0.0)

    anchor.acceleration.update(0.0, 0.0)
    anchor.commanded_velocity.update(anchor.velocity)
    anchor.observed_velocity.update(anchor.velocity)
    return world_offset_to_anchor_local(
        anchor.position - previous_position,
        yaw_degrees,
    )


def follow_corridor_locally(
    anchor: environment.Robot,
    observation: LocalObservation,
    yaw_degrees: float,
    dt: float,
) -> pygame.Vector2:
    """Move forward while centring from only local left/right LiDAR returns."""
    left_range, right_range = extract_lateral_wall_ranges(observation.lidar_scan)
    lateral_error = right_range - left_range
    lateral_speed = max(
        -ANCHOR_MAX_LATERAL_SPEED,
        min(
            ANCHOR_MAX_LATERAL_SPEED,
            ANCHOR_LATERAL_GAIN * lateral_error,
        ),
    )
    return integrate_anchor_local_command(
        anchor,
        pygame.Vector2(ANCHOR_FORWARD_SPEED, lateral_speed),
        yaw_degrees,
        dt,
    )


def follow_backtrack_corridor_with_comm_guard(
    anchor: environment.Robot,
    robots: list[environment.Robot],
    lidar: AnchorLidar,
    shepherd_ids: set[int],
    yaw_degrees: float,
    dt: float,
) -> pygame.Vector2:
    """
    Backtracking 중 Anchor는 혼자 선행하지 않는다.

    1. Anchor-local 관측에서 NORMAL swarm의 선두 cohort를 찾음
    2. 선두의 실제 observed velocity에 맞춰 전진
    3. 선두와 목표 간격보다 가까워지면 Anchor 전진 억제
    4. LiDAR lateral centering은 계속 사용
    5. captured Shepherd와 direct COMM_RANGE도 유지

    global/world robot position 기반 navigation은 사용하지 않는다.
    """

    # =====================================================
    # 1. 이동 전 Anchor-local observation
    # =====================================================

    observation = (
        LocalObservationBuilder.build(
            anchor,
            robots,
            lidar,
            yaw_degrees,
        )
    )

    robots_by_id = {
        robot.robot_id: robot
        for robot in robots
    }

    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    local_velocity_by_id = {
        robot_id: local_velocity
        for robot_id, local_velocity
        in zip(
            observation.robot_ids,
            observation.velocities,
        )
    }

    # =====================================================
    # 2. 현재 return corridor 안의 NORMAL robot만 사용
    #
    # Backtracking frame:
    # +x = Junction 복귀 방향
    # -x = Dead-end 방향
    # =====================================================

    half_width = (
        0.5 * KNOWN_CORRIDOR_WIDTH
        + environment.GRID_SPACING
    )

    normal_candidates: list[
        tuple[
            int,
            pygame.Vector2,
        ]
    ] = []

    for (
        robot_id,
        local_position,
    ) in local_position_by_id.items():

        if (
            robot_id
            == observation.lidar_robot_id
        ):
            continue

        robot = robots_by_id[
            robot_id
        ]

        if (
            robot.role
            != "NORMAL"
        ):
            continue

        # 현재 corridor 폭 안의 swarm만.
        if (
            abs(local_position.y)
            > half_width
        ):
            continue

        # Anchor에서 너무 멀리 떨어진 다른 영역 robot 제외.
        if (
            abs(local_position.x)
            >
            BACKTRACK_FLOW_EVAL_HOPS
            * environment.COMM_RANGE
        ):
            continue

        normal_candidates.append(
            (
                robot_id,
                local_position,
            )
        )

    # NORMAL이 없다면 Anchor 혼자 가지 않는다.
    if not normal_candidates:

        stop_anchor(
            anchor
        )

        role_debug(
            "backtrack-anchor-front-hold",
            (
                "[BacktrackAnchorFrontHold] "
                "reason=NO_NORMAL_SWARM"
            ),
        )

        return pygame.Vector2()

    # =====================================================
    # 3. Swarm 선두 추출
    #
    # 단일 가장 앞 robot 하나를 쓰면 outlier에 민감하므로
    # 가장 앞쪽 NORMAL 몇 대를 cohort로 사용한다.
    # =====================================================

    normal_candidates.sort(
        key=lambda item:
        item[1].x,
        reverse=True,
    )

    front_count = min(
        len(normal_candidates),
        max(
            3,
            len(normal_candidates) // 5,
        ),
    )

    front_cohort = (
        normal_candidates[
            :front_count
        ]
    )

    front_x_values = sorted(
        local_position.x
        for _, local_position
        in front_cohort
    )

    # 한 대의 튀는 robot이 아니라
    # 선두 cohort의 median longitudinal position.
    front_x = (
        front_x_values[
            len(front_x_values) // 2
        ]
    )

    front_speed_values = sorted(
        max(
            0.0,
            local_velocity_by_id[
                robot_id
            ].x,
        )
        for robot_id, _
        in front_cohort
    )

    front_speed = (
        front_speed_values[
            len(front_speed_values) // 2
        ]
    )

    # =====================================================
    # 4. Anchor가 swarm 선두와 함께 가도록 longitudinal 제어
    #
    # front_x:
    #   > 0 : swarm 선두가 Anchor보다 앞
    #   < 0 : Anchor가 이미 swarm보다 앞
    #
    # Anchor는 선두보다 일정 거리 뒤를 유지.
    # =====================================================

    target_front_gap = max(
        2.0
        * environment.ROBOT_RADIUS,
        environment.GRID_SPACING,
    )

    front_gap_error = (
        front_x
        - target_front_gap
    )

    # Anchor가 이미 선두에 너무 가까이 있거나
    # 선두보다 앞서 있으면 앞으로 가지 않는다.
    if (
        front_gap_error
        <= 0.0
    ):

        forward_speed = 0.0

    else:

        # 군집 선두의 실제 속도를 기본으로 사용.
        #
        # Anchor가 뒤처진 경우에만 gap 오차만큼
        # 약간 catch-up 허용.
        catchup_speed = (
            1.5
            * front_gap_error
        )

        forward_speed = min(
            ANCHOR_FORWARD_SPEED,
            front_speed
            + catchup_speed,
        )

        forward_speed = max(
            0.0,
            forward_speed,
        )

    # =====================================================
    # 5. Lateral control은 기존 LiDAR corridor centering 사용
    # =====================================================

    (
        left_range,
        right_range,
    ) = extract_lateral_wall_ranges(
        observation.lidar_scan
    )

    lateral_error = (
        right_range
        - left_range
    )

    lateral_speed = max(
        -ANCHOR_MAX_LATERAL_SPEED,
        min(
            ANCHOR_MAX_LATERAL_SPEED,
            ANCHOR_LATERAL_GAIN
            * lateral_error,
        ),
    )

    # =====================================================
    # 6. Anchor 실제 이동
    # =====================================================

    before_position = (
        anchor.position.copy()
    )

    before_previous = (
        anchor.previous_position.copy()
    )

    local_delta = (
        integrate_anchor_local_command(
            anchor,
            pygame.Vector2(
                forward_speed,
                lateral_speed,
            ),
            yaw_degrees,
            dt,
        )
    )

    # =====================================================
    # 7. 이동 후 captured Shepherd와 communication 검사
    # =====================================================

    post_observation = (
        LocalObservationBuilder.build(
            anchor,
            robots,
            lidar,
            yaw_degrees,
        )
    )

    post_local_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(
            post_observation.robot_ids,
            post_observation.relative_positions,
        )
    }

    active_shepherd_ids = {
        robot_id
        for robot_id
        in shepherd_ids
        if (
            robot_id
            in post_local_by_id
            and robot_id
            in robots_by_id
            and robots_by_id[
                robot_id
            ].role
            == "SHEPHERD"
        )
    }

    communication_ok = (
        bool(
            active_shepherd_ids
        )
        and all(
            post_local_by_id[
                robot_id
            ].length()
            <= environment.COMM_RANGE

            for robot_id
            in active_shepherd_ids
        )
    )

    if not communication_ok:

        # Anchor 이동만 되돌린다.
        anchor.position.update(
            before_position
        )

        anchor.previous_position.update(
            before_previous
        )

        stop_anchor(
            anchor
        )

        role_debug(
            "backtrack-anchor-comm-hold",
            (
                "[BacktrackAnchorCommHold] "
                f"shepherds="
                f"{len(active_shepherd_ids)} "
                "reason=COMM_RANGE"
            ),
        )

        return pygame.Vector2()

    role_debug(
        "backtrack-anchor-front-follow",
        (
            "[BacktrackAnchorFrontFollow] "
            f"front_count="
            f"{front_count} "
            f"front_x="
            f"{front_x:.2f} "
            f"target_gap="
            f"{target_front_gap:.2f} "
            f"front_speed="
            f"{front_speed:.2f} "
            f"anchor_speed="
            f"{forward_speed:.2f}"
        ),
    )

    return local_delta

def anchor_can_command_group(
    observation: LocalObservation,
    group_ids: set[int],
) -> bool:

    local_by_id = {
        robot_id: local_position
        for (
            robot_id,
            local_position,
        )
        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    return bool(
        group_ids
        and
        all(
            robot_id in local_by_id
            and
            local_by_id[
                robot_id
            ].length()
            <= environment.COMM_RANGE

            for robot_id
            in group_ids
        )
    )


def find_branch_free_gap(
    scan: LidarScan,
) -> float | None:
    """
    Anchor 전방 반구의 LiDAR에서
    실제로 진행 가능한 연속 free gap을 찾는다.

    반환값:
        Anchor-local target angle [deg]

    gap이 없으면:
        None
    """

    smoothed = smooth_ranges(
        scan.ranges,
        window_size=5,
    )

    samples = [
        (
            angle,
            measured_range,
        )

        for angle, measured_range
        in zip(
            scan.angles_deg,
            smoothed,
        )

        if (
            -ANCHOR_EXPLORE_HALF_FOV_DEG
            <= angle
            <= ANCHOR_EXPLORE_HALF_FOV_DEG
        )
    ]

    gaps: list[
        list[
            tuple[
                float,
                float,
            ]
        ]
    ] = []

    current_gap: list[
        tuple[
            float,
            float,
        ]
    ] = []

    for (
        angle,
        measured_range,
    ) in samples:

        if (
            measured_range
            >= ANCHOR_TRAVERSABLE_RANGE
        ):

            current_gap.append(
                (
                    angle,
                    measured_range,
                )
            )

        else:

            if current_gap:

                gaps.append(
                    current_gap
                )

                current_gap = []

    if current_gap:

        gaps.append(
            current_gap
        )

    valid_gaps = [
        gap

        for gap
        in gaps

        if (
            gap[-1][0]
            - gap[0][0]
        )
        >= ANCHOR_MIN_FREE_GAP_WIDTH_DEG
    ]

    if not valid_gaps:

        return None

    # 가장 넓은 free gap을 우선한다.
    # 같은 폭이면 정면에 가까운 gap을 우선한다.
    def gap_score(
        gap,
    ):

        start_angle = (
            gap[0][0]
        )

        end_angle = (
            gap[-1][0]
        )

        width = (
            end_angle
            - start_angle
        )

        center_angle = (
            0.5
            * (
                start_angle
                + end_angle
            )
        )

        mean_range = (
            sum(
                measured_range

                for _,
                measured_range
                in gap
            )
            / len(gap)
        )

        return (
            width,
            mean_range,
            -abs(
                center_angle
            ),
        )

    best_gap = max(
        valid_gaps,
        key=gap_score,
    )

    return (
        0.5
        * (
            best_gap[0][0]
            + best_gap[-1][0]
        )
    )

def compute_branch_explore_anchor_speed_cap(
    observation: LocalObservation,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    corridor_width: float,
) -> tuple[
    float,
    int | None,
    float,
    float,
]:
    """
    Branch exploration 중 Anchor가
    swarm의 가장 앞선 NORMAL보다 일정 거리 앞에서
    함께 움직이도록 전진 속도 상한을 계산한다.

    사용 정보:
        Anchor-relative robot position
        Anchor-relative observed velocity

    global/map robot position은 사용하지 않는다.

    return:
        speed_cap
        front_robot_id
        front_gap
        front_robot_speed
    """

    half_width = (
        0.5 * corridor_width
        + ANCHOR_FRONT_LATERAL_MARGIN_ROWS
        * environment.GRID_SPACING
    )

    observation_depth = (
        ANCHOR_FRONT_OBSERVATION_HOPS
        * environment.COMM_RANGE
    )

    candidates: list[
        tuple[
            int,
            pygame.Vector2,
            pygame.Vector2,
        ]
    ] = []

    for (
        robot_id,
        local_position,
        local_velocity,
    ) in zip(
        observation.robot_ids,
        observation.relative_positions,
        observation.velocities,
    ):

        # Anchor 자신 제외
        if (
            robot_id
            == observation.lidar_robot_id
        ):
            continue

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        # swarm front는 NORMAL만 본다.
        if (
            robot.role
            != "NORMAL"
        ):
            continue

        # 현재 Branch corridor 폭 밖은 제외.
        if (
            abs(local_position.y)
            > half_width
        ):
            continue

        # 현재 Anchor와 너무 멀리 떨어진 다른 영역 제외.
        if (
            abs(local_position.x)
            > observation_depth
        ):
            continue

        candidates.append(
            (
                robot_id,
                local_position,
                local_velocity,
            )
        )

    # =====================================================
    # 주변에 NORMAL swarm이 없다면
    # Anchor 혼자 전진하지 않는다.
    # =====================================================

    if not candidates:

        return (
            0.0,
            None,
            float("inf"),
            0.0,
        )

    # =====================================================
    # +x가 Anchor 진행 방향.
    #
    # 따라서 local x가 가장 큰 NORMAL =
    # 현재 swarm의 가장 앞선 robot.
    # =====================================================

    (
        front_robot_id,
        front_position,
        front_velocity,
    ) = max(
        candidates,
        key=lambda item:
        item[1].x,
    )

    # Anchor 뒤에 있으면 양수 gap.
    #
    # 예:
    # front x = -4
    # → Anchor가 4만큼 앞에 있음.
    front_gap = (
        -front_position.x
    )

    target_gap = max(
        3.0 * environment.ROBOT_RADIUS,
        ANCHOR_FRONT_TARGET_GAP_ROWS
        * environment.GRID_SPACING,
    )

    # NORMAL 선두가 실제 앞으로 움직이는 속도.
    front_speed = max(
        0.0,
        front_velocity.x,
    )

    # =====================================================
    # Anchor longitudinal controller
    #
    # target:
    #
    #     front robot x = -target_gap
    #
    # Anchor가 너무 앞:
    #     front_gap > target_gap
    #     → 속도 감소
    #
    # 너무 가까움:
    #     front_gap < target_gap
    #     → 속도 증가
    # =====================================================

    gap_error = (
        target_gap
        - front_gap
    )

    speed_cap = (
        front_speed
        + ANCHOR_FRONT_GAP_GAIN
        * gap_error
    )

    speed_cap = max(
        0.0,
        min(
            ANCHOR_FORWARD_SPEED,
            speed_cap,
        ),
    )

    return (
        speed_cap,
        front_robot_id,
        front_gap,
        front_speed,
    )


def move_anchor_through_free_gap(
    anchor: environment.Robot,
    scan: LidarScan,
    yaw_degrees: float,
    target_local_angle: float,
    dt: float,
    max_forward_speed: float | None = None,
) -> tuple[float, pygame.Vector2]:
    """
    LiDAR free gap 방향으로 Anchor heading을
    조금씩 회전시키고 독립적으로 전진한다.

    반환값:
        갱신된 runtime yaw
    """

    max_turn_this_step = (
        ANCHOR_MAX_TURN_RATE_DEG
        * dt
    )

    applied_turn = max(
        -max_turn_this_step,
        min(
            max_turn_this_step,
            target_local_angle,
        ),
    )

    new_yaw = normalize_angle(
        yaw_degrees
        + applied_turn
    )

    front_range = (
        lidar_range_at_local_angle(
            scan,
            0.0,
        )
    )

    # 코너에서 정면이 막혀 있고
    # target gap이 옆에 있으면
    # 먼저 제자리에서 방향을 돌린다.
    if (
        front_range
        < ANCHOR_TRAVERSABLE_RANGE
        and
        abs(
            target_local_angle
        )
        > 30.0
    ):

        forward_speed = (
            0.0
        )

    else:

        clearance_ratio = max(
            0.0,
            min(
                1.0,
                front_range
                / ANCHOR_TRAVERSABLE_RANGE,
            ),
        )

        turn_ratio = max(
            0.20,
            1.0
            - abs(
                target_local_angle
            )
            / 90.0,
        )

        speed_ratio = min(
            clearance_ratio,
            turn_ratio,
        )

        forward_speed = (
            ANCHOR_MIN_EXPLORE_SPEED
            + (
                ANCHOR_FORWARD_SPEED
                - ANCHOR_MIN_EXPLORE_SPEED
            )
            * speed_ratio
        )

    # =====================================================
    # Branch exploration에서 swarm front가 따라오는 속도보다
    # Anchor가 혼자 빨리 가지 못하도록 전진 속도 제한.
    #
    # LiDAR가 정한 안전 속도를 증가시키지는 않고,
    # 필요할 때 감소시키기만 한다.
    # =====================================================

    if (
        max_forward_speed
        is not None
    ):

        forward_speed = min(
            forward_speed,
            max(
                0.0,
                max_forward_speed,
            ),
        )

    local_delta = (
        integrate_anchor_local_command(
            anchor,
            pygame.Vector2(
                forward_speed,
                0.0,
            ),
            new_yaw,
            dt,
        )
    )

    return (
        new_yaw,
        local_delta,
    )


def move_anchor_with_backtracking_shepherds(
    anchor: environment.Robot,
    scan: LidarScan,
    shepherd_ids: set[int],
    robots_by_id: dict[int, environment.Robot],
    yaw_degrees: float,
    target_local_angle: float,
    dt: float,
) -> tuple[
    float,
    pygame.Vector2,
    bool,
]:
    """
    Backtracking corner에서 Anchor의 translation + rotation을
    동일 Shepherd cohort가 그대로 복사한다.

    Shepherd ID 재선정 없음.
    상대 대형 그대로 유지.
    """

    active_ids = {
        robot_id
        for robot_id in shepherd_ids
        if (
            robots_by_id[robot_id].role
            == "SHEPHERD"
            and getattr(
                robots_by_id[robot_id],
                "shepherd_mode",
                None,
            )
            == "PUSH"
        )
    }

    if not active_ids:

        stop_anchor(anchor)

        return (
            yaw_degrees,
            pygame.Vector2(),
            False,
        )

    # =====================================================
    # 1. 이동 전 Anchor pose
    # =====================================================

    old_anchor_position = (
        anchor.position.copy()
    )

    old_anchor_previous = (
        anchor.previous_position.copy()
    )

    old_yaw = (
        yaw_degrees
    )

    # =====================================================
    # 2. Shepherd들의 현재 Anchor-local offset 저장
    #
    # 이것이 capture topology다.
    # =====================================================

    local_offsets = {}

    old_shepherd_positions = {}

    for robot_id in active_ids:

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        old_shepherd_positions[
            robot_id
        ] = robot.position.copy()

        local_offsets[
            robot_id
        ] = (
            world_offset_to_anchor_local(
                robot.position
                - old_anchor_position,
                old_yaw,
            )
        )

    # =====================================================
    # 3. Anchor가 기존 LiDAR controller로 코너링
    # =====================================================

    (
        new_yaw,
        anchor_delta,
    ) = (
        move_anchor_through_free_gap(
            anchor,
            scan,
            old_yaw,
            target_local_angle,
            dt,
        )
    )

    new_anchor_position = (
        anchor.position.copy()
    )

    # =====================================================
    # 4. 동일 Shepherd가 새로운 Anchor pose를 그대로 복사
    #
    # old local offset을 new yaw로 회전시킨다.
    # =====================================================

    target_positions = {}

    for robot_id in active_ids:

        target_position = (
            new_anchor_position
            + anchor_local_to_world(
                local_offsets[
                    robot_id
                ],
                new_yaw,
            )
        )

        # Shepherd가 벽을 통과해야 하는 motion이면
        # 이번 Anchor motion 자체를 취소.
        if not physical_is_walkable(
            target_position,
            robots_by_id[
                robot_id
            ].radius,
        ):

            anchor.position.update(
                old_anchor_position
            )

            anchor.previous_position.update(
                old_anchor_previous
            )

            stop_anchor(
                anchor
            )

            role_debug(
                "backtrack-corner-rigid-block",
                (
                    "[BacktrackCornerRigidBlock] "
                    f"robot_id={robot_id}"
                ),
            )

            return (
                old_yaw,
                pygame.Vector2(),
                False,
            )

        target_positions[
            robot_id
        ] = target_position

    # =====================================================
    # 5. 실제 Shepherd rigid motion 적용
    # =====================================================

    for robot_id in active_ids:

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        old_position = (
            old_shepherd_positions[
                robot_id
            ]
        )

        target_position = (
            target_positions[
                robot_id
            ]
        )

        robot.previous_position.update(
            old_position
        )

        robot.position.update(
            target_position
        )

        realized_velocity = (
            (
                target_position
                - old_position
            )
            / max(
                dt,
                environment.EPSILON,
            )
        )

        robot.velocity.update(
            realized_velocity
        )

        robot.observed_velocity.update(
            realized_velocity
        )

        robot.commanded_velocity.update(
            realized_velocity
        )

        robot.acceleration.update(
            0.0,
            0.0,
        )

    role_debug(
        "backtrack-anchor-copy",
        (
            "[BacktrackAnchorCopy] "
            f"count={len(active_ids)} "
            f"yaw={old_yaw:.2f}"
            f"->{new_yaw:.2f} "
            f"anchor_move="
            f"{anchor_delta.length():.4f}"
        ),
    )

    return (
        new_yaw,
        anchor_delta,
        True,
    )

def trim_backtracking_shepherds_to_corridor(
    observation: LocalObservation,
    active_shepherd_ids: set[int],
    robots_by_id: dict[int, environment.Robot],
) -> tuple[set[int], set[int]]:
    """
    Anchor가 새 corridor 방향을 바라본 뒤,
    LiDAR로 측정한 corridor 폭에 들어오지 못하는
    바깥쪽 Shepherd를 NORMAL로 해제한다.

    Returns
    -------
    kept_ids:
        계속 Shepherd로 유지할 IDs

    released_ids:
        NORMAL로 해제한 IDs
    """

    if not active_shepherd_ids:
        return set(), set()

    # -----------------------------------------------------
    # 1. 새 corridor의 좌/우 wall range
    # -----------------------------------------------------

    left_range = lidar_range_at_local_angle(
        observation.lidar_scan,
        +90.0,
    )

    right_range = lidar_range_at_local_angle(
        observation.lidar_scan,
        -90.0,
    )

    # Anchor가 완전히 중앙이 아닐 수 있으므로
    # 더 좁은 쪽을 기준으로 안전한 half-width를 만든다.
    safe_half_width = (
        min(
            left_range,
            right_range,
        )
        - environment.ROBOT_RADIUS
        - 0.5 * environment.GRID_SPACING
    )

    safe_half_width = max(
        environment.ROBOT_RADIUS,
        safe_half_width,
    )

    # -----------------------------------------------------
    # 2. Anchor-local robot positions
    # -----------------------------------------------------

    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    kept_ids: set[int] = set()
    released_ids: set[int] = set()

    # -----------------------------------------------------
    # 3. 새 corridor 폭 안에 들어오는 Shepherd만 유지
    # -----------------------------------------------------

    for robot_id in active_shepherd_ids:

        local_position = (
            local_position_by_id.get(
                robot_id
            )
        )

        if local_position is None:
            # 관측에서 사라진 경우에는 일단 유지
            kept_ids.add(robot_id)
            continue

        if (
            abs(local_position.y)
            <= safe_half_width
        ):
            kept_ids.add(robot_id)

        else:
            released_ids.add(robot_id)

    # -----------------------------------------------------
    # 4. 폭 밖 Shepherd → NORMAL
    # -----------------------------------------------------

    for robot_id in released_ids:

        robot = robots_by_id[
            robot_id
        ]

        release_robot_to_normal(
            robot
        )

    print(
        "[BacktrackCorridorTrim] "
        f"left={left_range:.2f} "
        f"right={right_range:.2f} "
        f"safe_half_width={safe_half_width:.2f} "
        f"before={len(active_shepherd_ids)} "
        f"kept={len(kept_ids)} "
        f"released={len(released_ids)} "
        f"released_ids={sorted(released_ids)}"
    )

    return (
        kept_ids,
        released_ids,
    )

def detect_backtrack_corner(
    scan: LidarScan,
) -> float | None:
    """
    현재 Anchor가 Junction 방향으로 돌아가는 도중
    정면 corridor가 끝나고 좌/우로 꺾이는 corner인지 판단.

    반환:
        corner 방향의 Anchor-local free-gap angle

    corner가 아니면:
        None
    """

    target_gap_angle = (
        find_branch_free_gap(
            scan
        )
    )

    if target_gap_angle is None:
        return None

    front_range = (
        lidar_range_at_local_angle(
            scan,
            0.0,
        )
    )

    corner_detected = (
        front_range
        < ANCHOR_TRAVERSABLE_RANGE
        and
        abs(target_gap_angle)
        >= BACKTRACK_CORNER_MIN_TURN_DEG
    )

    if not corner_detected:
        return None

    return target_gap_angle


def backtrack_corridor_reacquired(
    scan: LidarScan,
    target_gap_angle: float | None,
) -> bool:
    """
    코너를 돈 뒤 새로운 직선 corridor에
    Anchor가 다시 들어왔는지 LiDAR로 판단.
    """

    if target_gap_angle is None:
        return False

    (
        left_range,
        right_range,
    ) = extract_lateral_wall_ranges(
        scan
    )

    if (
        not math.isfinite(left_range)
        or
        not math.isfinite(right_range)
        or
        left_range >= scan.max_range - 1.0
        or
        right_range >= scan.max_range - 1.0
    ):
        return False

    measured_width = (
        left_range
        + right_range
    )

    corridor_width_valid = (
        0.75 * KNOWN_CORRIDOR_WIDTH
        <= measured_width
        <= 1.25 * KNOWN_CORRIDOR_WIDTH
    )

    heading_aligned = (
        abs(target_gap_angle)
        <= BACKTRACK_CORNER_EXIT_ANGLE_DEG
    )

    return (
        corridor_width_valid
        and
        heading_aligned
    )

def detect_visible_marker_ahead(
    observation: LocalObservation,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    lidar: AnchorLidar,
    ignore_marker_id: int | None,
) -> int | None:
    """
    Anchor 전방에서 실제로 보이는 Marker를 찾는다.

    현재 Branch를 시작할 때 통과하는
    자기 Branch의 Marker는 ignore한다.
    """

    nearest_marker_id: (
        int | None
    ) = None

    nearest_distance = (
        float("inf")
    )

    detection_range = min(
        MARKER_DETECTION_RANGE,
        observation.lidar_scan.max_range,
    )

    for (
        robot_id,
        local_position,
    ) in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):

        if (
            robot_id
            == ignore_marker_id
        ):
            continue

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        if (
            robot.role
            != "MARKER"
        ):
            continue

        # Anchor 뒤쪽 Marker는 무시
        if (
            local_position.x
            <= 0.0
        ):
            continue
        # =============================================
        # 현재 Anchor가 진행 중인 corridor의
        # 중앙 영역에 있는 Marker만 인정한다.
        #
        # 다른 Branch/Junction 옆쪽 Marker를
        # 잘못 감지하지 않기 위한 local geometric gate.
        # =============================================

        if (
            abs(local_position.y)
            >
            0.45
            * KNOWN_CORRIDOR_WIDTH
        ):
            continue

        distance = (
            local_position.length()
        )

        if (
            distance
            > detection_range
        ):
            continue

        angle = math.degrees(
            math.atan2(
                local_position.y,
                local_position.x,
            )
        )

        if (
            abs(angle)
            > MARKER_HALF_ANGLE_DEG
        ):
            continue

        # 그 bearing에서 Marker보다 먼저
        # wall이 있으면 보이지 않는 Marker.
        wall_range = (
            lidar_range_at_local_angle(
                lidar.last_scan,
                angle,
            )
        )

        if (
            distance
            >
            wall_range
            + MARKER_LINE_OF_SIGHT_MARGIN
        ):
            continue

        if (
            distance
            < nearest_distance
        ):

            nearest_distance = (
                distance
            )

            nearest_marker_id = (
                robot_id
            )

    return nearest_marker_id

def find_branch_id_by_marker(
    marker_id: int,
    branch_states: dict,
) -> str | None:

    for (
        branch_id,
        state,
    ) in branch_states.items():

        if (
            state.get(
                "marker_id"
            )
            == marker_id
        ):

            return branch_id

    return None

def mark_branch_visited(
    branch_id: str,
    branch_states: dict,
    reason: str,
) -> None:

    state = (
        branch_states[
            branch_id
        ]
    )

    state[
        "visit_state"
    ] = "VISITED"

    if (
        state.get(
            "marker_id"
        )
        is not None
    ):

        state[
            "marker_state"
        ] = "VISITED"

    print(
        "[BranchVisited] "
        f"branch={branch_id} "
        f"reason={reason}"
    )


def find_next_unvisited_branch(
    branch_order: list[str],
    branch_states: dict,
) -> str | None:

    """
    현재 Junction의 DFS branch order를 따라
    아직 방문하지 않은 Branch 하나를 반환한다.

    B0/B1/B2의 물리적 방향은 전혀 사용하지 않는다.
    """

    for branch_id in branch_order:

        if (
            branch_states[
                branch_id
            ][
                "visit_state"
            ]
            == "UNVISITED"
        ):

            return branch_id

    return None


def find_final_push_branch(
    branches: list[dict],
) -> dict:

    return min(
        branches,
        key=lambda branch:
        abs(
            normalize_angle(
                branch["center_angle"]
            )
        ),
    )


def extract_front_mouth_corners(
    lidar: AnchorLidar,
) -> tuple[
    pygame.Vector2,
    pygame.Vector2,
] | None:

    if len(lidar.opening_groups) < 3:
        print(
            "[FrontCornerFail] "
            f"reason=NOT_ENOUGH_OPENINGS "
            f"count={len(lidar.opening_groups)}"
        )
        return None

    scan = lidar.last_scan

    smoothed = smooth_ranges(
        scan.ranges,
        window_size=5,
    )

    # =====================================================
    # 1. LEFT / FRONT / RIGHT opening 구분
    # =====================================================

    front_opening = min(
        lidar.opening_groups,
        key=lambda opening: abs(
            normalize_angle(
                opening["center_angle"]
            )
        ),
    )

    left_opening = min(
        lidar.opening_groups,
        key=lambda opening: circular_error(
            opening["center_angle"],
            -90.0,
        ),
    )

    right_opening = min(
        lidar.opening_groups,
        key=lambda opening: circular_error(
            opening["center_angle"],
            90.0,
        ),
    )

    if (
        front_opening is left_opening
        or front_opening is right_opening
        or left_opening is right_opening
    ):
        print(
            "[FrontCornerFail] "
            "reason=DUPLICATED_OPENING"
        )
        return None

    left_angle = normalize_angle(
        left_opening["center_angle"]
    )

    front_angle = normalize_angle(
        front_opening["center_angle"]
    )

    right_angle = normalize_angle(
        right_opening["center_angle"]
    )

    # =====================================================
    # 2. 두 opening 사이에서 가장 가까운 wall point
    #    = 물리적인 Junction corner
    # =====================================================

    def find_corner_between(
        angle_a: float,
        angle_b: float,
    ) -> tuple[
        pygame.Vector2,
        float,
        float,
    ] | None:

        lower = min(angle_a, angle_b)
        upper = max(angle_a, angle_b)

        indices = [
            index
            for index, angle
            in enumerate(scan.angles_deg)
            if lower <= angle <= upper
        ]

        if not indices:
            return None

        corner_index = min(
            indices,
            key=lambda index: smoothed[index],
        )

        corner_angle = (
            scan.angles_deg[corner_index]
        )

        corner_range = (
            smoothed[corner_index]
        )

        radians = math.radians(
            corner_angle
        )

        corner = pygame.Vector2(
            corner_range * math.cos(radians),
            corner_range * math.sin(radians),
        )

        return (
            corner,
            corner_angle,
            corner_range,
        )

    left_result = find_corner_between(
        left_angle,
        front_angle,
    )

    right_result = find_corner_between(
        front_angle,
        right_angle,
    )

    if (
        left_result is None
        or right_result is None
    ):
        print(
            "[FrontCornerFail] "
            "reason=NO_CORNER"
        )
        return None

    (
        front_left,
        front_left_angle,
        front_left_range,
    ) = left_result

    (
        front_right,
        front_right_angle,
        front_right_range,
    ) = right_result

    if front_left.y > front_right.y:
        front_left, front_right = (
            front_right,
            front_left,
        )

        front_left_angle, front_right_angle = (
            front_right_angle,
            front_left_angle,
        )

        front_left_range, front_right_range = (
            front_right_range,
            front_left_range,
        )

    print(
        "[FrontMouthCorners] "
        f"opening_angles="
        f"L:{left_angle:.1f},"
        f"F:{front_angle:.1f},"
        f"R:{right_angle:.1f} "
        f"corner_angles="
        f"FL:{front_left_angle:.1f},"
        f"FR:{front_right_angle:.1f} "
        f"FL=({front_left.x:.2f},"
        f"{front_left.y:.2f}) "
        f"FR=({front_right.x:.2f},"
        f"{front_right.y:.2f}) "
        f"ranges="
        f"{front_left_range:.2f},"
        f"{front_right_range:.2f}"
    )

    return (
        front_left,
        front_right,
    )


def get_lateral_opening(
    lidar: AnchorLidar,
    target_angle: float,
) -> dict | None:
    """Select one lateral opening nearest the requested local bearing."""
    candidates = [
        opening
        for opening in lidar.opening_groups
        if abs(
            normalize_angle(opening["center_angle"] - target_angle)
        ) <= 60.0
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda opening: abs(
            normalize_angle(opening["center_angle"] - target_angle)
        ),
    )


def extract_base_entrance_corners(
    lidar: AnchorLidar,
) -> tuple[pygame.Vector2, pygame.Vector2] | None:
    """Return the Base-side corners of the stationary lateral openings."""
    left_opening = get_lateral_opening(lidar, -90.0)
    right_opening = get_lateral_opening(lidar, 90.0)
    if left_opening is None or right_opening is None:
        return None

    base_left = min(
        (left_opening["mouth_a"], left_opening["mouth_b"]),
        key=lambda point: point.x,
    ).copy()
    base_right = min(
        (right_opening["mouth_a"], right_opening["mouth_b"]),
        key=lambda point: point.x,
    ).copy()
    if base_left.y > base_right.y:
        base_left, base_right = base_right, base_left

    base_width = (base_right - base_left).length()
    base_midpoint = (base_left + base_right) * 0.5
    if not (
        KNOWN_CORRIDOR_WIDTH * 0.65
        <= base_width
        <= KNOWN_CORRIDOR_WIDTH * 1.35
    ):
        print("[RejectBaseCorners] " f"reason=width width={base_width:.2f}")
        return None
    if abs(base_midpoint.x) > 8.0:
        print("[RejectBaseCorners] " f"reason=depth mid_x={base_midpoint.x:.2f}")
        return None
    return base_left, base_right


def diagonal_segment_intersection(
    first_start: pygame.Vector2,
    first_end: pygame.Vector2,
    second_start: pygame.Vector2,
    second_end: pygame.Vector2,
) -> pygame.Vector2 | None:
    """Return an intersection only when it lies inside both diagonal segments."""
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    determinant = first_direction.cross(second_direction)
    if abs(determinant) <= environment.EPSILON:
        return None

    offset = second_start - first_start
    first_t = offset.cross(second_direction) / determinant
    second_t = offset.cross(first_direction) / determinant
    if not (0.0 <= first_t <= 1.0 and 0.0 <= second_t <= 1.0):
        return None
    return first_start + first_direction * first_t


def estimate_local_junction_center(
    base_left: pygame.Vector2,
    base_right: pygame.Vector2,
    front_left: pygame.Vector2,
    front_right: pygame.Vector2,
) -> pygame.Vector2 | None:

    center = diagonal_segment_intersection(
        base_left,
        front_right,
        base_right,
        front_left,
    )

    if center is None:

        print(
            "[JunctionCenterFail] "
            "reason=DIAGONALS_DO_NOT_INTERSECT "
            f"BL={base_left} "
            f"BR={base_right} "
            f"FL={front_left} "
            f"FR={front_right}"
        )

        return None

    print(
        "[JunctionDiagonalGeometry] "
        f"BL=({base_left.x:.2f},"
        f"{base_left.y:.2f}) "
        f"BR=({base_right.x:.2f},"
        f"{base_right.y:.2f}) "
        f"FL=({front_left.x:.2f},"
        f"{front_left.y:.2f}) "
        f"FR=({front_right.x:.2f},"
        f"{front_right.y:.2f}) "
        f"C=({center.x:.2f},"
        f"{center.y:.2f})"
    )

    return center


def move_anchor_to_locked_center(
    anchor: environment.Robot,
    locked_target_local: pygame.Vector2,
    traveled_local: pygame.Vector2,
    yaw_degrees: float,
    dt: float,
) -> tuple[bool, pygame.Vector2]:
    """Transit using a stationary-scan target and local odometry only."""
    remaining = locked_target_local - traveled_local
    print(
        "[AnchorCenterTransit] "
        f"target=({locked_target_local.x:.2f},{locked_target_local.y:.2f}) "
        f"travel=({traveled_local.x:.2f},{traveled_local.y:.2f}) "
        f"remaining=({remaining.x:.2f},{remaining.y:.2f})"
    )

    if remaining.length() <= ANCHOR_LOCAL_CENTER_TOLERANCE:
        stop_anchor(anchor)
        return True, pygame.Vector2()

    forward_speed = max(
        -ANCHOR_CENTERING_SPEED,
        min(
            ANCHOR_CENTERING_SPEED,
            ANCHOR_LOCAL_CENTER_GAIN * remaining.x,
        ),
    )
    lateral_speed = max(
        -ANCHOR_CENTER_MAX_LATERAL_SPEED,
        min(
            ANCHOR_CENTER_MAX_LATERAL_SPEED,
            ANCHOR_LOCAL_CENTER_GAIN * remaining.y,
        ),
    )
    local_delta = integrate_anchor_local_command(
        anchor,
        pygame.Vector2(forward_speed, lateral_speed),
        yaw_degrees,
        dt,
    )
    return False, local_delta


def normalize_angle(
    angle: float,
) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def angle_inside_sector(
    angle: float,
    start: float,
    end: float,
) -> bool:
    angle = angle % 360.0
    start = start % 360.0
    end = end % 360.0

    if start <= end:
        return start <= angle <= end

    return angle >= start or angle <= end

def ray_to_branch_mouth_range(
    direction: pygame.Vector2,
    mouth_a: pygame.Vector2,
    mouth_b: pygame.Vector2,
) -> float | None:
    """
    Anchor-local origin (0, 0)에서 direction 방향으로 ray를 쐈을 때
    Branch mouth segment [mouth_a, mouth_b]와 만나는 range를 계산한다.

    절대좌표 사용 없음.
    """

    mouth_vector = mouth_b - mouth_a

    denominator = direction.cross(
        mouth_vector
    )

    if abs(denominator) <= environment.EPSILON:
        return None

    ray_range = (
        mouth_a.cross(mouth_vector)
        / denominator
    )

    segment_ratio = (
        mouth_a.cross(direction)
        / denominator
    )

    if ray_range < 0.0:
        return None

    if not (
        0.0 <= segment_ratio <= 1.0
    ):
        return None

    return float(ray_range)


def robot_is_at_branch_entrance(
    local_position: pygame.Vector2,
    branch: dict,
) -> bool:
    """
    Initial Shepherd 후보 판정.

    Branch entrance 폭 안에 있고 entrance 직전 margin 이상 들어온
    robot은 Branch 안쪽 어느 깊이에 있든 Shepherd 후보로 유지한다.
    """

    entrance_midpoint = (
        branch["entrance_midpoint"]
    )

    axis = (
        branch["branch_axis"]
    ).normalize()

    tangent = (
        branch["entrance_tangent"]
    ).normalize()

    entrance_width = (
        branch["entrance_b"]
        - branch["entrance_a"]
    ).length()

    half_width = (
        0.5 * entrance_width
    )

    radius = (
        environment.ROBOT_RADIUS
    )

    offset = (
        local_position
        - entrance_midpoint
    )

    depth = offset.dot(
        axis
    )

    lateral = offset.dot(
        tangent
    )

    # Branch 통로 폭 안에 있는가
    inside_width = (
        abs(lateral)
        <= (
            half_width
            + radius
            + ENTRANCE_CAPTURE_MARGIN
        )
    )

    if not inside_width:
        return False

    return (
        depth
        >= -(
            radius
            + ENTRANCE_DEPTH_CAPTURE_MARGIN
        )
    )

def robot_has_direct_anchor_los(
    robot_id: int,
    observation: LocalObservation,
) -> bool:
    """
    해당 robot이 현재 Anchor를 직접 볼 수 있는지 검사한다.

    사용:
        - Anchor-relative robot position
        - Anchor LiDAR wall range

    global map position은 사용하지 않는다.
    """

    local_position = next(
        (
            position
            for rid, position
            in zip(
                observation.robot_ids,
                observation.relative_positions,
            )
            if rid == robot_id
        ),
        None,
    )

    if local_position is None:
        return False

    distance = (
        local_position.length()
    )

    if (
        distance
        <= environment.EPSILON
    ):
        return False

    # Anchor 기준 robot bearing
    bearing_deg = math.degrees(
        math.atan2(
            local_position.y,
            local_position.x,
        )
    )

    # 해당 방향에서 Anchor LiDAR가 본
    # 가장 가까운 wall까지 거리
    wall_range = (
        lidar_range_at_local_angle(
            observation.lidar_scan,
            bearing_deg,
        )
    )

    # robot보다 먼저 wall이 있으면
    # Anchor ↔ robot LOS가 막힌 것.
    visible = (
        distance
        <= wall_range
        + environment.ROBOT_RADIUS
    )

    return visible


def update_junction_broadcast_receivers(
    observation: LocalObservation,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
) -> set[int]:
    """
    Anchor의 JUNCTION_CONFIRMED broadcast를
    직접 받을 수 있는 NORMAL robot ID 반환.

    여기서는 direct communication range만 검사한다.
    실제 Shepherd가 될 때 LOS는 별도로 검사한다.
    """

    received_ids: set[int] = set()

    for (
        robot_id,
        local_position,
    ) in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):

        # Anchor 자신 제외
        if (
            robot_id
            == observation.lidar_robot_id
        ):
            continue

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        if (
            robot.role
            != "NORMAL"
        ):
            continue

        # Anchor와 직접 communication 가능한가?
        if (
            local_position.length()
            > environment.COMM_RANGE
        ):
            continue

        received_ids.add(
            robot_id
        )

    return received_ids


def branch_mouth_is_fully_covered(
    branch: dict,
    robot_local_positions: list[
        pygame.Vector2
    ],
) -> tuple[bool, float]:

    """
    Shepherd 후보 중 entrance 주변에 존재하는 로봇들이
    Branch 입구 전체 폭을 충분히 막았는지 검사한다.

    Shepherd membership은 Branch 안쪽까지 계속 유지하지만,
    sealing 판단에는 entrance 주변 로봇들만 사용한다.
    """

    entrance_a = (
        branch["entrance_a"]
    )

    entrance_b = (
        branch["entrance_b"]
    )

    entrance_midpoint = (
        branch["entrance_midpoint"]
    )

    tangent = (
        branch["entrance_tangent"]
    ).normalize()

    axis = (
        branch["branch_axis"]
    ).normalize()

    entrance_width = (
        entrance_b
        - entrance_a
    ).length()

    if (
        entrance_width
        <= environment.EPSILON
    ):
        return False, float("inf")

    radius = (
        environment.ROBOT_RADIUS
    )

    # 입구 주변 약 4열 깊이까지
    # physical sealing 판단에 사용
    seal_depth = (
        4.0
        * environment.GRID_SPACING
    )

    # Shepherd 한 대가 입구 폭을 덮는 것으로
    # 인정할 lateral half-span
    coverage_half_span = max(
        radius,
        0.5
        * environment.GRID_SPACING,
    )

    intervals: list[
        tuple[float, float]
    ] = []

    for position in robot_local_positions:

        offset = (
            position
            - entrance_midpoint
        )

        depth = offset.dot(
            axis
        )

        # entrance 직전 margin부터
        # entrance 안쪽 seal_depth까지만
        # 실제 입구 봉쇄 판정에 사용
        if (
            depth
            < -(
                radius
                + ENTRANCE_CAPTURE_MARGIN
            )
        ):
            continue

        if depth > seal_depth:
            continue

        lateral = (
            position
            - entrance_a
        ).dot(
            tangent
        )

        left = max(
            0.0,
            lateral
            - coverage_half_span,
        )

        right = min(
            entrance_width,
            lateral
            + coverage_half_span,
        )

        if right >= left:
            intervals.append(
                (
                    left,
                    right,
                )
            )

    if not intervals:
        return (
            False,
            entrance_width,
        )

    intervals.sort()

    covered_until = 0.0
    max_gap = 0.0

    for left, right in intervals:

        if left > covered_until:

            max_gap = max(
                max_gap,
                left - covered_until,
            )

        covered_until = max(
            covered_until,
            right,
        )

    if covered_until < entrance_width:

        max_gap = max(
            max_gap,
            entrance_width
            - covered_until,
        )

    gap_tolerance = (
        0.25
        * environment.GRID_SPACING
    )

    sealed = (
        max_gap
        <= gap_tolerance
    )

    return (
        sealed,
        max_gap,
    )


def collect_initial_shepherd_hop_region(
    observation: LocalObservation,
    branch: dict,
    candidate_ids: set[int],
) -> tuple[
    set[int],
    list[set[int]],
]:
    """
    각 Branch에서 가장 깊이 팽창한 NORMAL robot 1대를
    Hop 0 seed로 선택한다.

    그 seed로부터 동일한 environment.COMM_RANGE를 사용하여
    Hop 1 -> Hop 2 -> Hop 3까지 graph-neighbor를 확장한다.

    absolute/map position은 사용하지 않는다.
    """

    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    entrance_midpoint = (
        branch["entrance_midpoint"]
    )

    axis = (
        branch["branch_axis"]
    ).normalize()

    tangent = (
        branch["entrance_tangent"]
    ).normalize()


    # =================================================
    # 1. Branch-local depth / lateral
    # =================================================

    depth_by_id: dict[int, float] = {}
    lateral_by_id: dict[int, float] = {}

    for robot_id in candidate_ids:

        position = (
            local_position_by_id.get(
                robot_id
            )
        )

        if position is None:
            continue

        offset = (
            position
            - entrance_midpoint
        )

        depth_by_id[
            robot_id
        ] = offset.dot(
            axis
        )

        lateral_by_id[
            robot_id
        ] = offset.dot(
            tangent
        )


    if not depth_by_id:

        return (
            set(),
            [set()],
        )


    # =================================================
    # 2. Hop 0 =
    #    Branch 안으로 가장 멀리 팽창한 robot 1대
    #
    # depth가 같으면 Branch centerline에 가까운 robot 우선
    # =================================================

    seed_id = max(
        depth_by_id,

        key=lambda robot_id: (
            depth_by_id[
                robot_id
            ],
            -abs(
                lateral_by_id[
                    robot_id
                ]
            ),
        ),
    )


    layers: list[
        set[int]
    ] = [
        {
            seed_id
        }
    ]

    visited = {
        seed_id
    }


    # =================================================
    # 3. Hop1 -> Hop2 -> Hop3
    #
    # 순수 robot-to-robot communication graph
    # 동일한 COMM_RANGE 사용
    # =================================================

    for _hop in range(
        1,
        INITIAL_SHEPHERD_HOPS
        + 1,
    ):

        previous_layer = (
            layers[-1]
        )

        next_layer: set[int] = (
            set()
        )

        for source_id in previous_layer:

            source_position = (
                local_position_by_id[
                    source_id
                ]
            )

            for candidate_id in candidate_ids:

                if (
                    candidate_id
                    in visited
                ):
                    continue

                candidate_position = (
                    local_position_by_id.get(
                        candidate_id
                    )
                )

                if (
                    candidate_position
                    is None
                ):
                    continue

                if (
                    source_position.distance_to(
                        candidate_position
                    )
                    >
                    environment.COMM_RANGE
                ):
                    continue

                next_layer.add(
                    candidate_id
                )


        layers.append(
            next_layer
        )

        visited.update(
            next_layer
        )

        if not next_layer:
            break


    hop_region_ids = set().union(
        *layers
    )

    return (
        hop_region_ids,
        layers,
    )

def shepherd_hop_layer_is_fully_covered(
    observation: LocalObservation,
    branch: dict,
    layer_ids: set[int],
) -> tuple[bool, float]:

    if not layer_ids:
        return False, float("inf")

    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    entrance_midpoint = (
        branch["entrance_midpoint"]
    )

    tangent = (
        branch["entrance_tangent"]
    ).normalize()

    entrance_width = (
        branch["entrance_b"]
        - branch["entrance_a"]
    ).length()

    half_width = (
        0.5 * entrance_width
    )

    coverage_half_span = max(
    environment.ROBOT_RADIUS,
    WALL_CONTACT_RADIUS,
    )

    intervals = []

    for robot_id in layer_ids:

        position = (
            local_position_by_id.get(
                robot_id
            )
        )

        if position is None:
            continue

        lateral = (
            position
            - entrance_midpoint
        ).dot(
            tangent
        )

        # midpoint 기준 좌표를
        # [0, entrance_width] 좌표로 변환
        center = (
            lateral
            + half_width
        )

        left = max(
            0.0,
            center - coverage_half_span,
        )

        right = min(
            entrance_width,
            center + coverage_half_span,
        )

        if right >= left:
            intervals.append(
                (left, right)
            )

    if not intervals:
        return False, entrance_width

    intervals.sort()

    covered_until = 0.0
    max_gap = 0.0

    for left, right in intervals:

        if left > covered_until:
            max_gap = max(
                max_gap,
                left - covered_until,
            )

        covered_until = max(
            covered_until,
            right,
        )

    if covered_until < entrance_width:

        max_gap = max(
            max_gap,
            entrance_width
            - covered_until,
        )

    gap_tolerance = (
        2.0
        * environment.ROBOT_RADIUS
    )

    covered = (
        max_gap
        <= gap_tolerance
    )

    return covered, max_gap


def form_initial_junction_shepherd_boundaries(
    observation: LocalObservation,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    branches: list[dict],
    branch_states: dict,
    junction_message_received_ids: set[int],
) -> bool:

    """
    CASE 1:
    Base에서 SPH swarm이 Junction으로 팽창한 뒤,
    각 Branch entrance에 도달한 robot들을
    Initial Shepherd로 만든다.

    아직 entrance 전체가 막히지 않았으면
    Shepherd들도 SPH로 계속 움직인다.

    entrance 전체가 충분히 막힌 순간,
    그 시점에 entrance formation band 안에 있는
    Shepherd들을 즉시 freeze한다.
    """

    local_position_by_id = {
        robot_id: local_position

        for (
            robot_id,
            local_position,
        )

        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    for branch in branches:

        branch_id = (
            branch["id"]
        )

        state = (
            branch_states[
                branch_id
            ]
        )

        # 이미 완성된 wall은 다시 건드리지 않는다.
        if state[
            "initial_sealed"
        ]:
            continue

        # =================================================
        # 현재 이 순간 entrance band에 있는 robot들
        # =================================================

        current_ids: set[int] = (
            set()
        )

        for (
            robot_id,
            local_position,
        ) in zip(
            observation.robot_ids,
            observation.relative_positions,
        ):

            # Anchor 제외
            if (
                robot_id
                == observation.lidar_robot_id
            ):
                continue

            
            robot = (
                robots_by_id[
                    robot_id
                ]
            )

            # =================================================
            # NORMAL robot이 처음 Shepherd가 되려면
            #
            # 1. Anchor의 Junction broadcast를 받았어야 하고
            # 2. 현재 Anchor와 direct LOS가 있어야 한다.
            #
            # 이미 이 Branch의 Shepherd가 된 robot은
            # 다시 이 조건을 검사하지 않는다.
            # =================================================

            # Initial formation 중에는 역할을 바꾸지 않는다.
            # Branch 안으로 자연스럽게 팽창한 NORMAL만 후보.
            if (
                robot.role
                != "NORMAL"
            ):
                continue
             # =================================================
            # DEBUG: Initial Shepherd candidate
            # =================================================

            entrance_midpoint = branch["entrance_midpoint"]
            axis = branch["branch_axis"].normalize()
            tangent = branch["entrance_tangent"].normalize()

            offset = local_position - entrance_midpoint

            depth = offset.dot(axis)
            lateral = offset.dot(tangent)

            entrance_width = (
                branch["entrance_b"]
                - branch["entrance_a"]
            ).length()

            half_width = 0.5 * entrance_width

            if robot.role == "NORMAL":
                print(
                    "[InitialShepherdCandidateDebug] "
                    f"id={robot_id} "
                    f"branch={branch_id} "
                    f"depth={depth:.2f} "
                    f"lateral={lateral:.2f} "
                    f"limit="
                    f"{half_width + environment.ROBOT_RADIUS + ENTRANCE_CAPTURE_MARGIN:.2f} "
                    f"received="
                    f"{robot_id in junction_message_received_ids} "
                    f"los="
                    f"{robot_has_direct_anchor_los(robot_id, observation)}"
                )


            # =================================================
            # ⑤ 실제 Branch entrance에 들어온 robot인가?
            # =================================================
            
            if not robot_is_at_branch_entrance(
                local_position,
                branch,
            ):
                continue

            current_ids.add(
                robot_id
            )
            
                # =================================================
        # 1. 아직 모두 NORMAL인 Branch candidate
        # =================================================

        candidate_ids = set(
            current_ids
        )

        candidate_positions = [
            local_position_by_id[
                robot_id
            ]

            for robot_id
            in candidate_ids

            if robot_id
            in local_position_by_id
        ]


     


        # =================================================
        # 3. Branch가 충분히 찼으면
        #
        # 실제 가장 바깥 front edge를 Hop0으로 잡고
        # Hop1 -> Hop2 -> Hop3을 Junction 방향으로 수집
        # =================================================

        (
            hop_region_ids,
            hop_layers,
        ) = collect_initial_shepherd_hop_region(
            observation,
            branch,
            candidate_ids,
        )


        required_layer_count = (
            INITIAL_SHEPHERD_HOPS
            + 1
        )


        required_hop_ready = (
            len(hop_layers)
            >= required_layer_count

            and

            all(
                len(
                    hop_layers[
                        hop_index
                    ]
                )
                > 0

                for hop_index
                in range(
                    required_layer_count
                )
            )
        )


        hop_counts = [
            len(layer)
            for layer
            in hop_layers
        ]
        # =================================================
        # Hop0 ~ Hop3 전체 cohort가
        # Branch 폭을 실제로 막을 수 있는지 검사
        # =================================================

        (
            cohort_covered,
            max_gap,
        ) = shepherd_hop_layer_is_fully_covered(
            observation,
            branch,
            hop_region_ids,
        )

        # =================================================
        # 4. 완료 조건
        #
        # A. Branch mouth가 실제로 충분히 채워짐
        # B. front layer 자체가 full-width
        #    (collect 함수 내부에서 검사)
        # C. Hop0 -> Hop1 -> Hop2 -> Hop3 모두 존재
        # =================================================

        sealed = (
            
            cohort_covered
        )


        role_debug(
            f"initial-shepherd-{branch_id}",
            (
                "[InitialShepherdCoverage] "
                f"branch={branch_id} "
                f"candidate_count={len(candidate_ids)} "
                f"hop_counts={hop_counts} "
                f"cohort_count={len(hop_region_ids)} "
                f"cohort_covered={cohort_covered} "
                f"max_gap={max_gap:.3f} "
                f"sealed={sealed}"
            ),
        )


        if not sealed:
            continue

     
        # =================================================
        # 6. 완료 순간
        #    현재 Initial Shepherd 전체 즉시 freeze
        # =================================================

        frozen_ids = set(
            hop_region_ids
        )

        for robot_id in frozen_ids:

            robot = (
                robots_by_id[
                    robot_id
                ]
            )

            # 이 순간 처음으로
            # NORMAL -> SHEPHERD 역할 변경
            if (
                robot.role
                != "NORMAL"
            ):
                raise RuntimeError(
                    "Initial Shepherd cohort contains "
                    f"non-NORMAL robot: "
                    f"id={robot_id} "
                    f"role={robot.role}"
                )

            assign_initial_shepherd_role(
                robot,
                branch_id,
            )

            

        for robot_id in frozen_ids:

            robot = (
                robots_by_id[
                    robot_id
                ]
            )

            freeze_role_robot(
                robot,
                "SHEPHERD",
                branch_id,
            )

        state[
            "initial_shepherd_ids"
        ] = set(
            frozen_ids
        )

        state[
            "initial_sealed"
        ] = True
        # =================================================
        # 7. Branch state 확정
        # =================================================

        state[
            "initial_shepherd_ids"
        ] = set(
            frozen_ids
        )

        state[
            "initial_sealed"
        ] = True


        print(
            "[InitialShepherdBoundaryLocked] "
            f"branch={branch_id} "
            f"count={len(frozen_ids)} "
            f"hop_counts={hop_counts} "
            
            f"max_gap={max_gap:.3f} "
            f"ids={sorted(frozen_ids)}"
        )
    return all(
        branch_states[
            branch["id"]
        ]["initial_sealed"]

        for branch
        in branches
    )


def get_anchor_nearby_backtracking_normals(
    observation: LocalObservation,
    robots_by_id: dict[int, environment.Robot],
) -> list[int]:

    nearby_ids = []

    for robot_id, local_position in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):

        if robot_id == observation.lidar_robot_id:
            continue

        robot = robots_by_id[robot_id]

        if robot.role != "NORMAL":
            continue

        # Anchor 진행방향 뒤쪽만
        if local_position.x >= -environment.ROBOT_RADIUS:
            continue

        # Anchor와 직접 communication 가능한 범위
        if (
            local_position.length()
            > environment.COMM_RANGE
        ):
            continue

        nearby_ids.append(
            robot_id
        )

    return nearby_ids

def detect_complete_dead_end_wall_layer(
    observation: LocalObservation,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
) -> tuple[
    bool,
    set[int],
    float,
    float,
]:
    """
    LiDAR가 얻은 dead-end wall WL~WR를 기준으로,

    1. wall 바로 앞 1-hop depth 안에 있는 NORMAL을 찾고
    2. 그 robot body가 WL~WR 전체 폭을 실제로 덮는지 검사한다.

    Returns:
        ready
        wall_robot_ids
        wall_width
        max_gap
    """

    wall_segment = (
        extract_dead_end_wall_segment(
            observation.lidar_scan
        )
    )

    if wall_segment is None:

        return (
            False,
            set(),
            0.0,
            float("inf"),
        )

    (
        wall_left,
        wall_right,
    ) = wall_segment

    wall_vector = (
        wall_right
        - wall_left
    )

    wall_width = (
        wall_vector.length()
    )

    if (
        wall_width
        <= environment.EPSILON
    ):

        return (
            False,
            set(),
            wall_width,
            float("inf"),
        )

    # WL -> WR 방향 unit vector
    tangent = (
        wall_vector
        / wall_width
    )

    local_position_by_id = {
        robot_id: local_position

        for (
            robot_id,
            local_position,
        )

        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    wall_robot_ids: set[int] = set()

    intervals: list[
        tuple[
            float,
            float,
        ]
    ] = []

    # =====================================================
    # Wall 바로 앞 1-hop layer만 선택
    # =====================================================

    for (
        robot_id,
        position,
    ) in local_position_by_id.items():

        if (
            robot_id
            == observation.lidar_robot_id
        ):
            continue

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        if (
            robot.role
            != "NORMAL"
        ):
            continue

        # ---------------------------------------------
        # WL 기준 wall tangent 방향 위치
        # ---------------------------------------------

        relative = (
            position
            - wall_left
        )

        along_wall = (
            relative.dot(
                tangent
            )
        )

        # wall segment 좌우 바깥 robot 제외
        if (
            along_wall
            <
            -environment.ROBOT_RADIUS
            or
            along_wall
            >
            wall_width
            + environment.ROBOT_RADIUS
        ):
            continue

        # ---------------------------------------------
        # robot center와 wall line 사이 수직거리
        # ---------------------------------------------

        projected_point = (
            wall_left
            + tangent
            * min(
                wall_width,
                max(
                    0.0,
                    along_wall,
                ),
            )
        )

        distance_from_wall = (
            position
            - projected_point
        ).length()

        # wall 바로 앞 1-hop layer만.
        if (
            distance_from_wall
            >
            BACKTRACK_DEAD_END_ONE_HOP_DEPTH
        ):
            continue

        wall_robot_ids.add(
            robot_id
        )

        # ---------------------------------------------
        # 이 robot body가 wall 폭에서 덮는 interval
        # ---------------------------------------------

        left = max(
            0.0,
            along_wall
            - environment.ROBOT_RADIUS,
        )

        right = min(
            wall_width,
            along_wall
            + environment.ROBOT_RADIUS,
        )

        if (
            right
            >= left
        ):

            intervals.append(
                (
                    left,
                    right,
                )
            )

    # =====================================================
    # WL -> WR 전체 coverage 검사
    # =====================================================

    if not intervals:

        return (
            False,
            wall_robot_ids,
            wall_width,
            wall_width,
        )

    intervals.sort()

    covered_until = 0.0
    max_gap = 0.0

    for (
        left,
        right,
    ) in intervals:

        if (
            left
            > covered_until
        ):

            max_gap = max(
                max_gap,
                left
                - covered_until,
            )

        covered_until = max(
            covered_until,
            right,
        )

    # 마지막 robot -> WR 끝점까지 gap
    if (
        covered_until
        < wall_width
    ):

        max_gap = max(
            max_gap,
            wall_width
            - covered_until,
        )

    ready = (
        max_gap
        <=
        BACKTRACK_DEAD_END_GAP_TOLERANCE
    )

    return (
        ready,
        wall_robot_ids,
        wall_width,
        max_gap,
    )

def collect_dead_end_wall_shepherd_cohort(
    observation: LocalObservation,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    max_hops: int,
) -> tuple[
    set[int],
    list[set[int]],
]:
    """
    Dead-end LiDAR wall 바로 앞에 자연스럽게 쌓인 NORMAL을
    seed layer로 잡고, 그 layer로부터 max_hops만큼
    robot-to-robot physical adjacency를 확장한다.

    사용 정보:
        - Anchor-local robot relative position
        - Anchor-local LiDAR wall range
        - robot-to-robot relative distance

    global/map robot position은 사용하지 않는다.
    """

    local_position_by_id = {
        robot_id: local_position

        for (
            robot_id,
            local_position,
        )

        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    # =====================================================
    # 1. Dead-end wall 바로 앞 NORMAL = seed layer
    # =====================================================

    wall_seed_ids: set[int] = set()

    for (
        robot_id,
        local_position,
    ) in local_position_by_id.items():

        if (
            robot_id
            == observation.lidar_robot_id
        ):
            continue

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        if (
            robot.role
            != "NORMAL"
        ):
            continue

        # Dead-end를 바라보고 있는 Anchor 기준
        # 앞쪽 robot만 후보.
        if (
            local_position.x
            <= 0.0
        ):
            continue

        # 현재 corridor 폭 바깥 robot 제외.
        if (
            abs(local_position.y)
            >
            (
                0.5
                * KNOWN_CORRIDOR_WIDTH
                + environment.GRID_SPACING
            )
        ):
            continue

        robot_distance = (
            local_position.length()
        )

        if (
            robot_distance
            <= environment.EPSILON
        ):
            continue

        # Anchor에서 해당 robot이 보이는 local bearing.
        robot_bearing_deg = (
            math.degrees(
                math.atan2(
                    local_position.y,
                    local_position.x,
                )
            )
        )

        # 뒤쪽 robot은 제외.
        if (
            abs(robot_bearing_deg)
            >
            ANCHOR_EXPLORE_HALF_FOV_DEG
        ):
            continue

        # 같은 bearing 방향으로 LiDAR가 본 실제 wall 거리.
        wall_range = (
            lidar_range_at_local_angle(
                observation.lidar_scan,
                robot_bearing_deg,
            )
        )

        # robot center에서 wall까지 남은 local 거리.
        wall_gap = (
            wall_range
            - robot_distance
        )

        # LiDAR가 본 wall 바로 앞에 있는 NORMAL이면
        # wall-contact seed.
        if (
            -environment.ROBOT_RADIUS
            <= wall_gap
            <= BACKTRACK_WALL_SEED_GAP
        ):

            wall_seed_ids.add(
                robot_id
            )

    if not wall_seed_ids:

        return (
            set(),
            [],
        )

    # =====================================================
    # 2. seed layer에서 robot-hop 확장
    # =====================================================

    layers: list[
        set[int]
    ] = [
        set(
            wall_seed_ids
        )
    ]

    selected_ids = set(
        wall_seed_ids
    )

    frontier = set(
        wall_seed_ids
    )

    for _hop in range(
        1,
        max_hops + 1,
    ):

        next_frontier: set[int] = set()

        for source_id in frontier:

            source_position = (
                local_position_by_id[
                    source_id
                ]
            )

            for (
                candidate_id,
                candidate_position,
            ) in local_position_by_id.items():

                if (
                    candidate_id
                    in selected_ids
                ):
                    continue

                if (
                    candidate_id
                    == observation.lidar_robot_id
                ):
                    continue

                candidate_robot = (
                    robots_by_id[
                        candidate_id
                    ]
                )

                if (
                    candidate_robot.role
                    != "NORMAL"
                ):
                    continue

                # Anchor 앞쪽의 dead-end compressed crowd만.
                if (
                    candidate_position.x
                    <= 0.0
                ):
                    continue

                if (
                    abs(candidate_position.y)
                    >
                    (
                        0.5
                        * KNOWN_CORRIDOR_WIDTH
                        + environment.GRID_SPACING
                    )
                ):
                    continue

                relative_position = (
                    candidate_position
                    - source_position
                )

                if (
                    relative_position.length()
                    <= BACKTRACK_WALL_HOP_RADIUS
                ):

                    next_frontier.add(
                        candidate_id
                    )

        if not next_frontier:
            break

        layers.append(
            next_frontier
        )

        selected_ids.update(
            next_frontier
        )

        frontier = (
            next_frontier
        )

    return (
        selected_ids,
        layers,
    )


def find_backtracking_seed(
    observation: LocalObservation,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
) -> int | None:

    candidates: list[
        tuple[
            float,
            int,
        ]
    ] = []

    for (
        robot_id,
        local_position,
    ) in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):

        if (
            robot_id
            == observation.lidar_robot_id
        ):
            continue

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        if (
            robot.role
            != "NORMAL"
        ):
            continue

        # Anchor의 현재 진행방향 기준 뒤쪽만
        if (
            local_position.x
            >= -environment.ROBOT_RADIUS
        ):
            continue

        distance = (
            local_position.length()
        )

        # 바로 근처의 NORMAL만 seed 후보
        if (
            distance
            > environment.COMM_RANGE
        ):
            continue

        candidates.append(
            (
                distance,
                robot_id,
            )
        )

    if not candidates:

        return None

    candidates.sort()

    return (
        candidates[0][1]
    )

def collect_backtracking_seed_one_two_hop(
    observation: LocalObservation,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    seed_id: int,
) -> tuple[
    set[int],
    list[set[int]],
]:
    """
    Anchor 주변 NORMAL seed를 기준으로 1-hop / 2-hop Shepherd
    후보군을 모집한다. map상의 절대 위치는 사용하지 않는다.
    """

    local_position_by_id = {
        robot_id: local_position

        for (
            robot_id,
            local_position,
        )

        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    seed_position = (
        local_position_by_id.get(
            seed_id
        )
    )

    if seed_position is None:
        return set(), [set(), set(), set()]

    seed_robot = robots_by_id[seed_id]

    if seed_robot.role != "NORMAL":
        return set(), [set(), set(), set()]

    eligible: dict[
        int,
        pygame.Vector2,
    ] = {}

    for (
        robot_id,
        local_position,
    ) in local_position_by_id.items():

        if (
            robot_id
            == observation.lidar_robot_id
        ):
            continue

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        if (
            robot.role
            != "NORMAL"
        ):
            continue

        # Anchor 진행방향 기준 뒤쪽
        if (
            local_position.x
            >= -environment.ROBOT_RADIUS
        ):
            continue

        eligible[
            robot_id
        ] = local_position

    if seed_id not in eligible:
        return set(), [set(), set(), set()]

    # =============================================
    # Hop 0 = Anchor 주변에서 선택된 NORMAL seed
    # =============================================

    seed_layer = {seed_id}

    # =============================================
    # Hop 1 = seed와 직접 communication 가능한 NORMAL
    # =============================================

    one_hop: set[int] = set()

    for candidate_id, candidate_position in eligible.items():

        if candidate_id == seed_id:
            continue

        if (
            (candidate_position - seed_position).length()
            <= environment.COMM_RANGE
        ):
            one_hop.add(candidate_id)

    # =============================================
    # 2-hop
    #
    # 1-hop robot과 직접 communication 가능한
    # 추가 NORMAL robot
    # =============================================

    two_hop: set[int] = (
        set()
    )

    for source_id in one_hop:

        source_position = (
            eligible[
                source_id
            ]
        )

        for (
            candidate_id,
            candidate_position,
        ) in eligible.items():

            if (
                candidate_id == seed_id
                or candidate_id in one_hop
            ):
                continue

            relative_position = (
                candidate_position
                - source_position
            )

            if (
                relative_position.length()
                <= environment.COMM_RANGE
            ):

                two_hop.add(
                    candidate_id
                )

    cohort_ids = (
        seed_layer
        | one_hop
        | two_hop
    )

    return (
        cohort_ids,
        [
            seed_layer,
            one_hop,
            two_hop,
        ],
    )

def required_backtracking_crowd_count(
    corridor_width: float,
) -> int:
    """
    Anchor 주변에 충분한 backtracking crowd가 형성되었다고
    보는 최소 robot 수를 계산한다.

    absolute position은 사용하지 않는다.
    """

    target_spacing = max(
        2.0 * environment.ROBOT_RADIUS,
        environment.GRID_SPACING
        * BACKTRACK_SHEPHERD_SPACING_RATIO,
    )

    full_width_count = (
        math.ceil(
            corridor_width
            / target_spacing
        )
        + 1
    )

    # 통로 전체를 한 줄로 막는 수가 아니라,
    # Anchor 주변에 충분한 crowd가 형성되었다고 보는 최소 수.
    return max(
        2,
        full_width_count,
    )


def activate_backtracking_shepherds_in_place(
    robot_ids: list[int],
    robots_by_id: dict[int, environment.Robot],
    branch_id: str,
) -> None:
    """
    required가 처음 충족된 순간의 위치를 전혀 바꾸지 않고
    membership만 NORMAL -> SHEPHERD로 전환한다.
    """

    for robot_id in robot_ids:

        robot = robots_by_id[robot_id]

        if robot.role != "NORMAL":
            raise RuntimeError(
                f"Captured backtracking robot is not NORMAL: "
                f"id={robot_id} role={robot.role}"
            )

        before = robot.position.copy()

        robot.role = "SHEPHERD"
        robot.role_branch = branch_id
        robot.shepherd_mode = "PUSH"

        robot.role_frozen = False
        robot.role_frozen_position = None

        # stale NORMAL dynamics 제거.
        # 위치는 절대로 변경하지 않는다.
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
        robot.commanded_velocity.update(0.0, 0.0)
        robot.observed_velocity.update(0.0, 0.0)

        if robot.position.distance_to(before) > environment.EPSILON:
            raise RuntimeError(
                "Backtracking Shepherd role transition "
                "caused a position jump."
            )


def select_backtracking_formation_cohort(
    observation: LocalObservation,
    cohort_ids: set[int],
    seed_id: int,
    required_count: int,
) -> set[int]:
    """
    Seed 기준 0~2 hop cohort에서
    필요한 수의 Shepherd를 선택한다.

    Seed robot은 반드시 포함한다.
    """

    local_position_by_id = {
        robot_id: local_position

        for (
            robot_id,
            local_position,
        )
        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    if (
        seed_id
        not in cohort_ids
        or
        seed_id
        not in local_position_by_id
    ):

        return set()

    seed_position = (
        local_position_by_id[
            seed_id
        ]
    )

    other_candidates = [
        robot_id

        for robot_id
        in cohort_ids

        if (
            robot_id
            != seed_id
            and
            robot_id
            in local_position_by_id
        )
    ]

    # 절대 위치가 아니라 seed와 robot 사이의 상대거리만 사용한다.
    other_candidates.sort(
        key=lambda robot_id:
        (
            local_position_by_id[
                robot_id
            ]
            - seed_position
        ).length()
    )

    selected = {
        seed_id
    }

    remaining_count = max(
        0,
        required_count - 1,
    )

    selected.update(
        other_candidates[
            :remaining_count
        ]
    )

    return selected

def relay_backtracking_shepherd_command(
    observation: LocalObservation,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    seed_id: int,
    formation_ids: set[int],
    branch_id: str,
    command: str,
    left_wall_range: float | None = None,
    right_wall_range: float | None = None,
) -> set[int]:
    """Relay FORM/PUSH/RELEASE through the selected Shepherd formation."""

    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    if (
        seed_id not in formation_ids
        or seed_id not in local_position_by_id
    ):
        return set()

    received = {seed_id}
    frontier = {seed_id}

    while frontier:

        next_frontier: set[int] = set()

        for source_id in frontier:

            source_position = (
                local_position_by_id.get(
                    source_id
                )
            )

            if source_position is None:
                continue

            for candidate_id in formation_ids:

                if candidate_id in received:
                    continue

                candidate_position = (
                    local_position_by_id.get(
                        candidate_id
                    )
                )

                if candidate_position is None:
                    continue

                if (
                    candidate_position
                    - source_position
                ).length() <= environment.COMM_RANGE:

                    received.add(
                        candidate_id
                    )

                    next_frontier.add(
                        candidate_id
                    )

        frontier = next_frontier

    for robot_id in received:
        robot = robots_by_id[robot_id]

        if command == "FORM":
            robot.role = "SHEPHERD"
            robot.role_branch = branch_id
            robot.shepherd_mode = "FORM"
            robot.role_frozen = False
            robot.role_frozen_position = None

            if left_wall_range is not None:
                robot.shepherd_left_wall_range = float(left_wall_range)

            if right_wall_range is not None:
                robot.shepherd_right_wall_range = float(right_wall_range)

            robot.velocity.update(0.0, 0.0)
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
            robot.commanded_velocity.update(0.0, 0.0)
            robot.observed_velocity.update(0.0, 0.0)

        elif command == "PUSH":
            robot.shepherd_mode = "PUSH"
            robot.role_frozen = False
            robot.role_frozen_position = None

        elif command == "RELEASE":
            release_robot_to_normal(robot)

    print(
        "[ShepherdRelay] "
        f"branch={branch_id} "
        f"seed={seed_id} "
        f"command={command} "
        f"received={len(received)}/{len(formation_ids)} "
        f"ids={sorted(received)}"
    )

    return received


def start_backtracking_shepherd_formation(
    observation: LocalObservation,
    formation_ids: set[int],
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    seed_id: int,
    branch_id: str,
    event_valid: bool,
) -> bool:

    if not event_valid:

        print(
            "[BacktrackingFormationRejected] "
            "reason=NO_VALID_TERMINAL_EVENT"
        )

        return False

    received_ids = relay_backtracking_shepherd_command(
        observation,
        robots_by_id,
        seed_id,
        formation_ids,
        branch_id,
        command="FORM",
    )

    all_received = (
        received_ids
        == formation_ids
    )

    if not all_received:

        print(
            "[BacktrackingFormationRelayIncomplete] "
            f"branch={branch_id} "
            f"received={len(received_ids)} "
            f"required={len(formation_ids)}"
        )

        return False

    print(
        "[BacktrackingFormationBroadcast] "
        f"branch={branch_id} "
        f"seed={seed_id} "
        f"count={len(formation_ids)} "
        f"ids={sorted(formation_ids)}"
    )

    return True

def backtracking_forward_wall_blockers(
    shepherd_ids: list[int],
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    return_yaw_deg: float,
    dt: float,
) -> set[int]:
    """
    현재 captured Shepherd topology가
    다음 return-direction PUSH 한 step을 수행할 때
    wall에 걸리는 Shepherd ID를 반환한다.

    실제 robot 위치는 바꾸지 않는다.
    """

    active_ids = [
        robot_id
        for robot_id in shepherd_ids
        if (
            robots_by_id[robot_id].role
            == "SHEPHERD"
            and getattr(
                robots_by_id[robot_id],
                "shepherd_mode",
                None,
            )
            == "PUSH"
        )
    ]

    if not active_ids:
        return set()

    push_speed = (
        BACKTRACK_PUSH_SPEED_RATIO
        * environment.INITIAL_SAFE_MAX_SPEED
    )

    forward_delta = (
        anchor_local_to_world(
            pygame.Vector2(
                push_speed * dt,
                0.0,
            ),
            return_yaw_deg,
        )
    )

    return {
        robot_id
        for robot_id in active_ids
        if not physical_is_walkable(
            robots_by_id[
                robot_id
            ].position
            + forward_delta,
            robots_by_id[
                robot_id
            ].radius,
        )
    }

def detach_backtracking_shepherd_group_step(
    observation: LocalObservation,
    shepherd_ids: list[int],
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    return_yaw_deg: float,
    dt: float,
    locked_direction: str | None,
) -> tuple[
    bool,
    set[int],
    str,
]:
    """
    Backtracking Shepherd 전체 topology를 유지한 채
    wall contact에서 lateral 방향으로 분리한다.

    핵심:
    - detach episode 시작 시 LEFT / RIGHT를 한 번만 선택
    - 이후 blocker 구성이 바뀌어도 방향을 뒤집지 않음
    - forward PUSH가 가능해질 때까지 같은 방향으로 이동
    """

    active_ids = [
        robot_id
        for robot_id in shepherd_ids
        if (
            robots_by_id[robot_id].role
            == "SHEPHERD"
            and getattr(
                robots_by_id[robot_id],
                "shepherd_mode",
                None,
            )
            == "PUSH"
        )
    ]

    if not active_ids:

        return (
            True,
            set(),
            "NONE",
        )

    # =====================================================
    # 1. 현재 return 방향으로 바로 PUSH 가능한지 확인
    # =====================================================

    blocked_ids = (
        backtracking_forward_wall_blockers(
            active_ids,
            robots_by_id,
            return_yaw_deg,
            dt,
        )
    )

    if not blocked_ids:

        return (
            True,
            set(),
            locked_direction
            if locked_direction is not None
            else "NONE",
        )

    # =====================================================
    # 2. Anchor-local Shepherd 위치
    # =====================================================

    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    (
        left_range,
        right_range,
    ) = extract_lateral_wall_ranges(
        observation.lidar_scan
    )

    contact_radius = max(
        WALL_CONTACT_RADIUS,
        environment.ROBOT_RADIUS,
    )

    # =====================================================
    # 3. rigid Shepherd 전체가 LEFT / RIGHT로
    #    얼마나 이동할 여유가 있는지 계산
    #
    # Anchor-local:
    #   y < 0 : LEFT
    #   y > 0 : RIGHT
    # =====================================================

    left_rooms: list[float] = []
    right_rooms: list[float] = []

    for robot_id in active_ids:

        local_position = (
            local_position_by_id.get(
                robot_id
            )
        )

        if local_position is None:
            continue

        left_room = (
            local_position.y
            + left_range
            - contact_radius
        )

        right_room = (
            right_range
            - local_position.y
            - contact_radius
        )

        left_rooms.append(
            left_room
        )

        right_rooms.append(
            right_room
        )

    minimum_left_room = (
        min(left_rooms)
        if left_rooms
        else 0.0
    )

    minimum_right_room = (
        min(right_rooms)
        if right_rooms
        else 0.0
    )

    # =====================================================
    # 4. 방향 선택
    #
    # 이미 direction이 lock되어 있으면 절대 다시 선택하지 않음.
    # =====================================================

    if (
        locked_direction
        in (
            "LEFT",
            "RIGHT",
        )
    ):

        chosen_direction = (
            locked_direction
        )

    else:

        if (
            minimum_left_room
            >
            minimum_right_room
        ):

            chosen_direction = (
                "LEFT"
            )

        else:

            chosen_direction = (
                "RIGHT"
            )

    lateral_sign = (
        -1.0
        if chosen_direction == "LEFT"
        else +1.0
    )

    # =====================================================
    # 5. 이번 substep lateral displacement
    # =====================================================

    detach_speed = (
        BACKTRACK_WALL_DETACH_SPEED_RATIO
        * environment.INITIAL_SAFE_MAX_SPEED
    )

    local_detach_delta = (
        pygame.Vector2(
            0.0,
            lateral_sign
            * detach_speed
            * dt,
        )
    )

    world_detach_delta = (
        anchor_local_to_world(
            local_detach_delta,
            return_yaw_deg,
        )
    )

    # =====================================================
    # 6. topology 전체가 같은 delta로 움직일 수 있는지 검사
    # =====================================================

    group_can_move = all(
        physical_is_walkable(
            robots_by_id[
                robot_id
            ].position
            + world_detach_delta,
            robots_by_id[
                robot_id
            ].radius,
        )

        for robot_id
        in active_ids
    )

    if not group_can_move:

        role_debug(
            "backtrack-detach-no-room",
            (
                "[BacktrackDetachNoRoom] "
                f"direction={chosen_direction} "
                f"blocked_ids="
                f"{sorted(blocked_ids)} "
                f"left_room="
                f"{minimum_left_room:.3f} "
                f"right_room="
                f"{minimum_right_room:.3f}"
            ),
        )

        return (
            False,
            blocked_ids,
            chosen_direction,
        )

    # =====================================================
    # 7. Shepherd 전체에 SAME lateral delta 적용
    #
    # capture topology 유지.
    # =====================================================

    realized_velocity = (
        world_detach_delta
        / max(
            dt,
            environment.EPSILON,
        )
    )

    for robot_id in active_ids:

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        old_position = (
            robot.position.copy()
        )

        robot.previous_position.update(
            old_position
        )

        robot.position += (
            world_detach_delta
        )

        robot.velocity.update(
            realized_velocity
        )

        robot.observed_velocity.update(
            realized_velocity
        )

        robot.commanded_velocity.update(
            realized_velocity
        )

        robot.acceleration.update(
            0.0,
            0.0,
        )

    # =====================================================
    # 8. lateral 이동 후 다시 forward PUSH 가능 여부 검사
    # =====================================================

    remaining_blocked_ids = (
        backtracking_forward_wall_blockers(
            active_ids,
            robots_by_id,
            return_yaw_deg,
            dt,
        )
    )

    detach_ready = (
        len(
            remaining_blocked_ids
        )
        == 0
    )

    role_debug(
        "backtrack-wall-detach",
        (
            "[BacktrackWallDetach] "
            f"blocked_ids="
            f"{sorted(remaining_blocked_ids)} "
            f"direction="
            f"{chosen_direction} "
            f"count="
            f"{len(active_ids)} "
            f"delta="
            f"{world_detach_delta.length():.4f} "
            f"ready={detach_ready}"
        ),
    )

    return (
        detach_ready,
        remaining_blocked_ids,
        chosen_direction,
    )
def update_pressure_push(
    observation: LocalObservation,
    shepherd_ids: list[int],
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    return_yaw_deg: float,
    dt: float,
) -> dict[int, pygame.Vector2]:
    """
    Backtracking Shepherd physical pressure push.

    원칙:
    1. required 순간 capture한 Shepherd topology 유지
    2. 모든 Shepherd에 동일 rigid displacement 적용
    3. wall / 다른 physical robot 관통 금지
    4. NORMAL이 이동을 막으면 physical contact acceleration 전달
    5. NORMAL에게 별도 goal/backtracking force는 주지 않음
    """

    contact_accelerations: dict[
        int,
        pygame.Vector2,
    ] = {}

    if not shepherd_ids:
        return contact_accelerations

    # =====================================================
    # 현재 실제 PUSH 역할을 가진 Shepherd만 사용
    # =====================================================

    active_ids = [
        robot_id
        for robot_id in shepherd_ids
        if (
            robots_by_id[robot_id].role
            == "SHEPHERD"
            and getattr(
                robots_by_id[robot_id],
                "shepherd_mode",
                None,
            )
            == "PUSH"
        )
    ]

    if not active_ids:
        return contact_accelerations

    # =====================================================
    # Anchor가 명령한 Junction return direction
    # =====================================================

    push_speed = (
        BACKTRACK_PUSH_SPEED_RATIO
        * environment.INITIAL_SAFE_MAX_SPEED
    )

    desired_world_velocity = (
        anchor_local_to_world(
            pygame.Vector2(
                push_speed,
                0.0,
            ),
            return_yaw_deg,
        )
    )

    desired_delta = (
        desired_world_velocity
        * dt
    )

    if (
        desired_world_velocity.length_squared()
        <= environment.EPSILON
    ):
        return contact_accelerations

    return_direction = (
        desired_world_velocity.normalize()
    )

    active_set = set(
        active_ids
    )

    # =====================================================
    # NORMAL에게 physical contact load 누적하는 helper
    # =====================================================

    def add_normal_contact(
        shepherd: environment.Robot,
        other_id: int,
        other: environment.Robot,
    ) -> None:

        # 실제로 밀어야 하는 swarm robot만.
        if other.role != "NORMAL":
            return

        contact_direction = (
            other.position
            - shepherd.position
        )

        if (
            contact_direction.length_squared()
            <= environment.EPSILON
        ):
            contact_direction = (
                return_direction.copy()
            )

        else:
            contact_direction = (
                contact_direction.normalize()
            )

        # Shepherd의 return 방향 앞쪽에 있는
        # NORMAL에 대해서만 PUSH 전달.
        if (
            contact_direction.dot(
                return_direction
            )
            <= 0.0
        ):
            return

        contact_acceleration = (
            contact_direction
            * (
                BACKTRACK_CONTACT_ACCEL_RATIO
                * environment.MAX_ACCELERATION
            )
        )

        contact_accelerations[
            other_id
        ] = (
            contact_accelerations.get(
                other_id,
                pygame.Vector2(),
            )
            + contact_acceleration
        )

    # =====================================================
    # 1. Wall collision
    #
    # topology 전체가 같이 갈 수 있는 최대 displacement 계산
    # =====================================================

    def wall_safe(
        fraction: float,
    ) -> bool:

        delta = (
            desired_delta
            * fraction
        )

        return all(
            physical_is_walkable(
                robots_by_id[robot_id].position
                + delta,
                robots_by_id[robot_id].radius,
            )
            for robot_id
            in active_ids
        )

    safe_fraction = 1.0

    if not wall_safe(1.0):

        low = 0.0
        high = 1.0

        for _ in range(14):

            middle = (
                0.5
                * (low + high)
            )

            if wall_safe(middle):
                low = middle
            else:
                high = middle

        safe_fraction = low

    # =====================================================
    # 2. Shepherd ↔ physical robot collision
    #
    # NORMAL을 관통하는 대신,
    # 접촉하면 NORMAL에 physical contact acceleration을 전달.
    # =====================================================

    movement = desired_delta

    movement_sq = (
        movement.length_squared()
    )

    if (
        movement_sq
        > environment.EPSILON
    ):

        for shepherd_id in active_ids:

            shepherd = (
                robots_by_id[shepherd_id]
            )

            for (
                other_id,
                other,
            ) in robots_by_id.items():

                if (
                    other_id
                    in active_set
                ):
                    continue

                relative_start = (
                    shepherd.position
                    - other.position
                )

                minimum_distance = (
                    shepherd.radius
                    + other.radius
                )

                c = (
                    relative_start.length_squared()
                    - minimum_distance**2
                )

                # =================================================
                # A. 이미 접촉 / 아주 약하게 overlap 상태
                # =================================================

                if c <= 0.0:

                    moving_into_robot = (
                        relative_start.dot(
                            movement
                        )
                        < 0.0
                    )

                    if moving_into_robot:

                        # 관통 금지.
                        safe_fraction = 0.0

                        # 대신 실제 NORMAL이면
                        # Shepherd motor load를 물리적으로 전달.
                        add_normal_contact(
                            shepherd,
                            other_id,
                            other,
                        )

                    continue

                # =================================================
                # B. 현재는 떨어져 있지만
                # 이번 step의 movement에서 collision 예정
                # =================================================

                a = movement_sq

                b = (
                    2.0
                    * relative_start.dot(
                        movement
                    )
                )

                discriminant = (
                    b * b
                    - 4.0 * a * c
                )

                if discriminant < 0.0:
                    continue

                hit_fraction = (
                    -b
                    - math.sqrt(
                        discriminant
                    )
                ) / (
                    2.0 * a
                )

                if (
                    0.0
                    <= hit_fraction
                    <= safe_fraction
                ):

                    # collision 직전까지만 Shepherd 이동 가능.
                    safe_fraction = max(
                        0.0,
                        hit_fraction
                        - 1.0e-4,
                    )

                    # NORMAL이면 실제 physical contact load 생성.
                    add_normal_contact(
                        shepherd,
                        other_id,
                        other,
                    )

    # =====================================================
    # 3. Shepherd rigid translation
    #
    # 모든 Shepherd가 정확히 SAME delta를 받는다.
    # 따라서 capture topology 자체는 바뀌지 않는다.
    # =====================================================

    actual_delta = (
        desired_delta
        * safe_fraction
    )

    actual_velocity = (
        actual_delta
        / max(
            dt,
            environment.EPSILON,
        )
    )

    for robot_id in active_ids:

        robot = (
            robots_by_id[robot_id]
        )

        old_position = (
            robot.position.copy()
        )

        robot.previous_position.update(
            old_position
        )

        robot.position += (
            actual_delta
        )

        robot.velocity.update(
            actual_velocity
        )

        robot.observed_velocity.update(
            actual_velocity
        )

        # Motor command는 계속 Anchor가 지정한
        # Junction return direction을 요구한다.
        robot.commanded_velocity.update(
            desired_world_velocity
        )

        # Shepherd는 SPH acceleration으로 움직이는 것이 아니다.
        robot.acceleration.update(
            0.0,
            0.0,
        )

    role_debug(
        "physical-pressure-push",
        (
            "[PhysicalPressurePush] "
            f"count={len(active_ids)} "
            "requested="
            f"{desired_delta.length():.4f} "
            "actual="
            f"{actual_delta.length():.4f} "
            "safe_fraction="
            f"{safe_fraction:.4f} "
            "contact_normals="
            f"{len(contact_accelerations)} "
            "topology_locked=True"
        ),
    )

    return contact_accelerations


def update_final_base_push(
    final_push_ids: set[int],
    robots_by_id: dict[int, environment.Robot],
    reference_yaw_deg: float,
    dt: float,
) -> None:

    push_speed = (
        0.35
        * environment.INITIAL_SAFE_MAX_SPEED
    )

    local_command = pygame.Vector2(
        -push_speed,
        0.0,
    )

    world_velocity = (
        anchor_local_to_world(
            local_command,
            reference_yaw_deg,
        )
    )

    for robot_id in final_push_ids:

        robot = robots_by_id[
            robot_id
        ]

        if robot.role != "SHEPHERD":
            continue

        next_position = (
            robot.position
            + world_velocity
            * dt
        )

        if physical_is_walkable(
            next_position,
            robot.radius,
        ):

            robot.previous_position.update(
                robot.position
            )

            robot.position.update(
                next_position
            )

            robot.velocity.update(
                world_velocity
            )

        else:
            robot.velocity.update(
                0.0,
                0.0,
            )


def evaluate_normal_reverse_flow(
    observation: LocalObservation,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    chain_order: list[int],
    corridor_width: float,
) -> tuple[
    bool,
    int,
    int,
    float,
    float,
]:
    """
    Pressure Push에 의해 NORMAL swarm의 실제 reverse flow가
    형성되었는지 Anchor-local observed velocity로 판정한다.

    absolute/world position은 사용하지 않는다.

    Anchor-local during backtracking:

        +x = Junction 복귀 방향
        -x = Dead-end 방향

    return:
        ready
        eligible_count
        reverse_count
        reverse_ratio
        mean_reverse_speed
    """

    # =====================================================
    # 1. Anchor-local robot position / velocity table
    # =====================================================

    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    local_velocity_by_id = {
        robot_id: local_velocity
        for robot_id, local_velocity
        in zip(
            observation.robot_ids,
            observation.velocities,
        )
    }

    # =====================================================
    # 2. Backtracking Shepherd의 현재 longitudinal x
    # =====================================================

    shepherd_x_values = sorted(
        local_position_by_id[
            robot_id
        ].x

        for robot_id
        in chain_order

        if robot_id
        in local_position_by_id
    )

    if not shepherd_x_values:

        return (
            False,
            0,
            0,
            0.0,
            0.0,
        )

    shepherd_x = (
        shepherd_x_values[
            len(shepherd_x_values) // 2
        ]
    )

    # =====================================================
    # 3. Shepherd 앞쪽의 local swarm만 평가
    #
    # 너무 멀리 있는 다른 영역 robot까지 포함하지 않는다.
    # =====================================================

    evaluation_depth = (
        BACKTRACK_FLOW_EVAL_HOPS
        * environment.COMM_RANGE
    )

    half_width = (
        0.5
        * corridor_width
        + environment.GRID_SPACING
    )

    eligible_ids: list[int] = []

    reverse_speeds: list[float] = []

    # =====================================================
    # 4. NORMAL 실제 observed velocity 검사
    # =====================================================

    for (
        robot_id,
        local_position,
    ) in local_position_by_id.items():

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        # NORMAL만 평가
        if (
            robot.role
            != "NORMAL"
        ):
            continue

        # ---------------------------------------------
        # Shepherd보다 Dead-end 쪽은 평가 제외.

        if (
            local_position.x
            <
            shepherd_x
            - environment.GRID_SPACING
        ):
            continue

        # Shepherd 앞쪽이더라도 너무 멀리 있는 robot 제외.
        if (
            local_position.x
            >
            shepherd_x
            + evaluation_depth
        ):
            continue

        # 현재 corridor 폭 밖의 robot 제외
        if (
            abs(
                local_position.y
            )
            >
            half_width
        ):
            continue

        local_velocity = (
            local_velocity_by_id.get(
                robot_id
            )
        )

        if (
            local_velocity
            is None
        ):
            continue

        eligible_ids.append(
            robot_id
        )

        return_speed = (
            local_velocity.x
        )

        if (
            return_speed
            >=
            BACKTRACK_REVERSE_SPEED_THRESHOLD
        ):

            reverse_speeds.append(
                return_speed
            )

    # =====================================================
    # 5. Reverse-flow ratio 계산
    # =====================================================

    eligible_count = (
        len(
            eligible_ids
        )
    )

    reverse_count = (
        len(
            reverse_speeds
        )
    )

    if (
        eligible_count
        == 0
    ):

        return (
            False,
            0,
            0,
            0.0,
            0.0,
        )

    reverse_ratio = (
        reverse_count
        / eligible_count
    )

    if reverse_speeds:

        mean_reverse_speed = (
            sum(
                reverse_speeds
            )
            / len(
                reverse_speeds
            )
        )

    else:

        mean_reverse_speed = (
            0.0
        )

    # =====================================================
    # 6. 최종 reverse-flow 판정
    # =====================================================

    ready = (
        eligible_count
        >= BACKTRACK_REVERSE_MIN_ROBOTS
        and
        reverse_ratio
        >= BACKTRACK_REVERSE_RATIO_THRESHOLD
    )

    return (
        ready,
        eligible_count,
        reverse_count,
        reverse_ratio,
        mean_reverse_speed,
    )


def prepare_backtracking_shepherd_push(
    observation: LocalObservation,
    chain_order: list[int],
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    seed_id: int,
    active_branch_id: str,
) -> bool:

    formation_ids = set(
        chain_order
    )

    received_ids = relay_backtracking_shepherd_command(
        observation,
        robots_by_id,
        seed_id,
        formation_ids,
        active_branch_id,
        command="PUSH",
    )

    all_received = (
        received_ids
        == formation_ids
    )

    if not all_received:

        print(
            "[BacktrackingPushRelayIncomplete] "
            f"branch={active_branch_id} "
            f"received={len(received_ids)} "
            f"required={len(formation_ids)}"
        )

        return False

    print(
        "[BacktrackingShepherdReadyForPush] "
        f"branch={active_branch_id} "
        f"seed={seed_id} "
        f"count={len(chain_order)}"
    )

    return True


def release_robot_to_normal(
    robot: environment.Robot,
) -> None:

    robot.role = "NORMAL"
    robot.role_branch = None

    robot.role_frozen = False
    robot.role_frozen_position = None

    robot.shepherd_mode = None

    robot.velocity.update(0.0, 0.0)
    robot.acceleration.update(0.0, 0.0)
    robot.filtered_acceleration.update(0.0, 0.0)
    robot.commanded_velocity.update(0.0, 0.0)
    robot.observed_velocity.update(0.0, 0.0)


def release_shepherd_group(
    robot_ids: set[int],
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
) -> None:

    for robot_id in robot_ids:

        robot = robots_by_id[
            robot_id
        ]

        if (
            robot.role
            != "SHEPHERD"
        ):
            continue

        # =============================================
        # SHEPHERD → NORMAL
        # =============================================

        robot.role = (
            "NORMAL"
        )

        robot.role_branch = (
            None
        )

        robot.role_frozen = (
            False
        )

        robot.role_frozen_position = (
            None
        )

        # =============================================
        # 별도의 진행 속도나 방향은 주지 않는다.
        # 이후 운동은 다시 SPH가 결정한다.
        # =============================================

        robot.velocity.update(
            0.0,
            0.0,
        )

        robot.acceleration.update(
            0.0,
            0.0,
        )

        robot.filtered_acceleration.update(
            0.0,
            0.0,
        )

        robot.commanded_velocity.update(
            0.0,
            0.0,
        )

        robot.observed_velocity.update(
            0.0,
            0.0,
        )


def branch_swarm_return_complete(
    observation: LocalObservation,
    branch: dict,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
    backtrack_ids: set[int],
) -> tuple[
    bool,
    list[int],
    list[int],
]:
    """
    현재 탐색 Branch 내부에
    NORMAL 또는 Backtracking Shepherd가
    더 이상 남아 있지 않은지 검사한다.

    world/map 위치는 사용하지 않고
    Junction-centered Anchor-local branch geometry만 사용한다.
    """

    axis = (
        branch["branch_axis"]
    ).normalize()

    tangent = (
        branch["entrance_tangent"]
    ).normalize()

    entrance_midpoint = (
        branch["entrance_midpoint"]
    )

    entrance_width = (
        branch["entrance_b"]
        - branch["entrance_a"]
    ).length()

    half_width = (
        0.5 * entrance_width
    )

    depth_margin = (
        environment.ROBOT_RADIUS
        + ENTRANCE_CAPTURE_MARGIN
    )

    lateral_margin = (
        environment.ROBOT_RADIUS
        + ENTRANCE_CAPTURE_MARGIN
    )

    remaining_normals: list[int] = []
    remaining_backtrack: list[int] = []

    for (
        robot_id,
        local_position,
    ) in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):

        if (
            robot_id
            == observation.lidar_robot_id
        ):
            continue

        robot = (
            robots_by_id[
                robot_id
            ]
        )

        offset = (
            local_position
            - entrance_midpoint
        )

        depth = (
            offset.dot(
                axis
            )
        )

        lateral = (
            offset.dot(
                tangent
            )
        )

        # 이 Branch corridor 폭 안에 있는가
        inside_branch_width = (
            abs(lateral)
            <=
            half_width
            + lateral_margin
        )

        if (
            not inside_branch_width
        ):
            continue

        # entrance보다 Branch 안쪽에 남아 있는가
        still_inside_branch = (
            depth
            >
            depth_margin
        )

        if (
            not still_inside_branch
        ):
            continue

        if (
            robot.role
            == "NORMAL"
        ):
            remaining_normals.append(
                robot_id
            )

        if (
            robot_id
            in backtrack_ids
        ):
            remaining_backtrack.append(
                robot_id
            )

    complete = (
       
        len(remaining_backtrack)
        == 0
    )

    return (
        complete,
        remaining_normals,
        remaining_backtrack,
    )


def swarm_base_return_complete(
    observation: LocalObservation,
) -> bool:
    """
    Root center에 고정된 Anchor-local frame에서 모든 swarm robot이
    Base corridor의 최종 회수 영역에 들어왔는지 검사한다.

    local -x는 Junction → Base 방향이며, map world pose나
    BASE_POSITION은 사용하지 않는다.
    """

    base_return_depth = (
        ROOT_CENTER_TO_BASE_DISTANCE
        - 0.5 * KNOWN_BASE_CORRIDOR_LENGTH
    )

    base_lateral_limit = (
        0.5 * KNOWN_CORRIDOR_WIDTH
        + environment.ROBOT_RADIUS
    )

    for (
        robot_id,
        local_position,
    ) in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):

        if (
            robot_id
            == observation.lidar_robot_id
        ):
            continue

        if (
            local_position.x
            > -base_return_depth
            or
            abs(local_position.y)
            > base_lateral_limit
        ):
            return False

    return True

# def backtracking_wall_ready(
#     observation: LocalObservation,
#     wall_ids: set[int],
#     corridor_width: float,
# ) -> tuple[
#     bool,
#     float,
# ]:

#     local_position_by_id = {
#         robot_id:
#         local_position

#         for (
#             robot_id,
#             local_position,
#         )

#         in zip(
#             observation.robot_ids,
#             observation.relative_positions,
#         )
#     }

#     half_width = (
#         0.5
#         * corridor_width
#     )

#     coverage_half_span = max(
#         environment.ROBOT_RADIUS,
#         0.5
#         * environment.GRID_SPACING,
#     )

#     intervals: list[
#         tuple[
#             float,
#             float,
#         ]
#     ] = []

#     for robot_id in wall_ids:

#         position = (
#             local_position_by_id.get(
#                 robot_id
#             )
#         )

#         if position is None:
#             continue

#         left = max(
#             -half_width,
#             position.y
#             - coverage_half_span,
#         )

#         right = min(
#             half_width,
#             position.y
#             + coverage_half_span,
#         )

#         if right >= left:

#             intervals.append(
#                 (
#                     left,
#                     right,
#                 )
#             )

#     if not intervals:

#         return (
#             False,
#             corridor_width,
#         )

#     intervals.sort()

#     covered_until = (
#         -half_width
#     )

#     max_gap = (
#         0.0
#     )

#     for (
#         left,
#         right,
#     ) in intervals:

#         if (
#             left
#             > covered_until
#         ):

#             max_gap = max(
#                 max_gap,
#                 left
#                 - covered_until,
#             )

#         covered_until = max(
#             covered_until,
#             right,
#         )

#     if (
#         covered_until
#         < half_width
#     ):

#         max_gap = max(
#             max_gap,
#             half_width
#             - covered_until,
#         )

#     gap_tolerance = (
#         BACKTRACK_GAP_TOLERANCE_ROWS
#         * environment.GRID_SPACING
#     )

#     ready = (
#         max_gap
#         <= gap_tolerance
#     )

#     return (
#         ready,
#         max_gap,
#     )

# def activate_backtracking_shepherd_group(
#     wall_ids: set[int],
#     robots_by_id: dict[
#         int,
#         environment.Robot,
#     ],
#     active_branch_id: str,
# ) -> None:

#     for robot_id in wall_ids:

#         robot = (
#             robots_by_id[
#                 robot_id
#             ]
#         )

#         freeze_role_robot(
#             robot,
#             "SHEPHERD",
#             active_branch_id,
#         )

#     print(
#         "[BacktrackingShepherdActivated] "
#         f"branch="
#         f"{active_branch_id} "
#         f"count="
#         f"{len(wall_ids)} "
#         f"ids="
#         f"{sorted(wall_ids)}"
#     )

def estimate_current_corridor_width(
    scan: LidarScan,
) -> float:

    (
        left_range,
        right_range,
    ) = (
        extract_lateral_wall_ranges(
            scan
        )
    )

    if (
        math.isfinite(
            left_range
        )
        and
        math.isfinite(
            right_range
        )
        and
        left_range
        < scan.max_range - 1.0
        and
        right_range
        < scan.max_range - 1.0
    ):

        width = (
            left_range
            + right_range
        )

        # Junction / lateral opening을
        # corridor side wall로 잘못 보는 것을 방지
        if (
            0.75
            * KNOWN_CORRIDOR_WIDTH
            <= width
            <= 1.25
            * KNOWN_CORRIDOR_WIDTH
        ):

            return width

    # 유효한 corridor side-wall pair가 아니면
    # 기본 corridor width 사용
    return (
        KNOWN_CORRIDOR_WIDTH
    )

def register_outgoing_branches(
    lidar: AnchorLidar,
) -> list[dict]:

    """
    Anchor가 Junction 중앙에 도달한 뒤
    LiDAR opening center angle을 이용하여
    각 outgoing Branch의 Anchor-local entrance geometry를 생성한다.

    global map position은 사용하지 않는다.
    """

    if not lidar.opening_groups:
        return []

    openings = list(
        lidar.opening_groups
    )

    print(
        "[CenterOpeningGroups] "
        + " ".join(
            (
                f"center={opening['center_angle']:.1f} "
                f"sector="
                f"{opening['start_angle']:.1f}"
                f"~{opening['end_angle']:.1f}"
            )
            for opening in openings
        )
    )

    # =====================================================
    # 실제 rear opening이 보일 때만 parent corridor 제외
    # =====================================================

    rear_candidates = [
        opening
        for opening in openings
        if circular_error(
            opening["center_angle"],
            180.0,
        ) <= 45.0
    ]

    if rear_candidates:

        parent = min(
            rear_candidates,
            key=lambda opening:
            circular_error(
                opening["center_angle"],
                180.0,
            ),
        )

        outgoing_openings = [
            opening
            for opening in openings
            if opening is not parent
        ]

        print(
            "[ParentOpeningDetected] "
            f"center="
            f"{parent['center_angle']:.1f}"
        )

    else:

        # Root Junction에서 Base 쪽 rear opening이
        # threshold opening으로 보이지 않는 경우.
        outgoing_openings = openings

        print(
            "[ParentOpeningNotVisible] "
            "keep_all_current_openings=True"
        )

    # local bearing 순서
    outgoing_openings.sort(
        key=lambda opening:
        normalize_angle(
            opening["center_angle"]
        )
    )

    branches: list[dict] = []

    for index, opening in enumerate(
        outgoing_openings
    ):

        branch = dict(
            opening
        )

        branch["id"] = (
            f"B{index}"
        )

        # =================================================
        # Branch 중심 방향
        #
        # LiDAR opening의 mouth_a/mouth_b로 만든
        # 기울어진 axis를 사용하지 않고,
        # 검출된 center_angle을 직접 사용한다.
        # =================================================

        center_rad = math.radians(
            branch["center_angle"]
        )

        axis = pygame.Vector2(
            math.cos(center_rad),
            math.sin(center_rad),
        ).normalize()

        tangent = pygame.Vector2(
            -axis.y,
            axis.x,
        )

        # =================================================
        # Junction 중앙 기준 실제 Branch entrance
        # =================================================

        half_junction_depth = (
            0.5
            * KNOWN_JUNCTION_DEPTH
        )

        half_corridor_width = (
            0.5
            * KNOWN_CORRIDOR_WIDTH
        )

        entrance_midpoint = (
            axis
            * half_junction_depth
        )

        entrance_a = (
            entrance_midpoint
            - tangent
            * half_corridor_width
        )

        entrance_b = (
            entrance_midpoint
            + tangent
            * half_corridor_width
        )

        branch[
            "branch_axis"
        ] = axis

        branch[
            "entrance_tangent"
        ] = tangent

        branch[
            "entrance_midpoint"
        ] = entrance_midpoint

        branch[
            "entrance_a"
        ] = entrance_a

        branch[
            "entrance_b"
        ] = entrance_b

        print(
            "[TrueBranchEntrance] "
            f"id={branch['id']} "
            f"center_angle="
            f"{branch['center_angle']:.1f} "
            f"A=("
            f"{entrance_a.x:.2f},"
            f"{entrance_a.y:.2f}) "
            f"B=("
            f"{entrance_b.x:.2f},"
            f"{entrance_b.y:.2f})"
        )

        branches.append(
            branch
        )

    return branches


def assign_initial_shepherd_role(
    robot: environment.Robot,
    branch_id: str,
) -> None:

    """
    Initial Junction formation 중의 임시 Shepherd.

    역할만 SHEPHERD로 바꾼다.
    아직 입구 전체가 막히지 않았으므로 freeze하지 않는다.

    이동은 계속 SPH에 의해서만 결정된다.
    """

    robot.role = (
        "SHEPHERD"
    )

    robot.role_branch = (
        branch_id
    )

    robot.role_frozen = False

    robot.role_frozen_position = (
        None
    )

def freeze_role_robot(
    robot: environment.Robot,
    role: str,
    branch_id: str,
) -> None:
    """Assign a local role and retain the robot at its current physical pose."""
    robot.role = role
    robot.role_branch = branch_id
    robot.role_frozen = True
    robot.role_frozen_position = robot.position.copy()
    robot.previous_position.update(robot.position)
    robot.velocity.update(0.0, 0.0)
    robot.acceleration.update(0.0, 0.0)
    robot.filtered_acceleration.update(0.0, 0.0)
    robot.commanded_velocity.update(0.0, 0.0)
    robot.observed_velocity.update(0.0, 0.0)


def create_initial_branch_markers(
    observation: LocalObservation,
    branches: list[dict],
    branch_states: dict,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
) -> None:
    """
    모든 Initial Shepherd boundary가 완성된 뒤
    각 runtime Branch B0/B1/B2에 Marker 한 대를 남긴다.

    Marker 선정 기준:

    1. 해당 Branch의 Initial Shepherd 중
       Branch 내부 방향으로 가장 깊은 depth를 계산한다.

    2. 가장 깊은 robot과 depth 차이가
       GRID_SPACING 이하인 robot들을
       deepest candidate로 선택한다.

    3. 그 candidate들 중
       Branch 중심선에 가장 가까운 robot을
       Marker로 선택한다.

    global/world Branch 의미는 사용하지 않는다.
    Branch-local axis / tangent만 사용한다.
    """

    # =====================================================
    # Anchor 기준 local robot position table
    # =====================================================

    local_position_by_id = {
        robot_id: local_position

        for (
            robot_id,
            local_position,
        )

        in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    # =====================================================
    # 각 Branch마다 Marker 하나 생성
    # =====================================================

    for branch in branches:

        branch_id = (
            branch[
                "id"
            ]
        )

        state = (
            branch_states[
                branch_id
            ]
        )

        # =================================================
        # 이미 Marker가 있으면 다시 만들지 않는다.
        # =================================================

        if (
            state[
                "marker_id"
            ]
            is not None
        ):
            continue

        # =================================================
        # 현재 Branch의 Initial Shepherd 집합
        # =================================================

        shepherd_ids = set(
            state[
                "initial_shepherd_ids"
            ]
        )

        if not shepherd_ids:

            raise RuntimeError(
                f"No Initial Shepherds for "
                f"{branch_id}"
            )

        # =================================================
        # observation에 실제 존재하는 Shepherd만 사용
        # =================================================

        shepherd_ids = {
            robot_id
            for robot_id
            in shepherd_ids

            if (
                robot_id
                in local_position_by_id
            )
        }

        if not shepherd_ids:

            raise RuntimeError(
                f"No observable Initial Shepherds "
                f"for {branch_id}"
            )

        # =================================================
        # Branch-local geometry
        #
        # axis:
        # Branch 안쪽 진행 방향
        #
        # tangent:
        # Branch 폭 방향
        #
        # entrance_midpoint:
        # Branch 입구 중심
        # =================================================

        axis = (
            branch[
                "branch_axis"
            ]
        ).normalize()

        tangent = (
            branch[
                "entrance_tangent"
            ]
        ).normalize()

        entrance_midpoint = (
            branch[
                "entrance_midpoint"
            ]
        )

        # =================================================
        # Branch 안쪽 depth 계산
        #
        # depth가 클수록
        # Branch 안쪽으로 더 깊이 들어간 robot.
        # =================================================

        def branch_depth(
            robot_id: int,
        ) -> float:

            local_position = (
                local_position_by_id[
                    robot_id
                ]
            )

            offset = (
                local_position
                - entrance_midpoint
            )

            return (
                offset.dot(
                    axis
                )
            )

        # =================================================
        # Branch 중심선에서의 lateral offset 계산
        #
        # 값이 작을수록 Branch 가운데에 가깝다.
        # =================================================

        def lateral_offset(
            robot_id: int,
        ) -> float:

            local_position = (
                local_position_by_id[
                    robot_id
                ]
            )

            offset = (
                local_position
                - entrance_midpoint
            )

            return abs(
                offset.dot(
                    tangent
                )
            )

        # =================================================
        # 1. Shepherd 중 가장 깊은 depth
        # =================================================

        max_depth = max(
            branch_depth(
                robot_id
            )

            for robot_id
            in shepherd_ids
        )

        # =================================================
        # 2. 가장 깊은 영역 후보
        #
        # 가장 깊은 robot 딱 한 대만 보지 않고,
        # max_depth에서 GRID_SPACING 이내의 robot들은
        # 모두 같은 "최외곽 영역"으로 본다.
        # =================================================

        depth_margin = (
            environment.GRID_SPACING
        )

        deepest_candidates = {
            robot_id

            for robot_id
            in shepherd_ids

            if (
                branch_depth(
                    robot_id
                )
                >=
                max_depth
                - depth_margin
            )
        }

        if not deepest_candidates:

            raise RuntimeError(
                f"No deepest Marker candidates "
                f"for {branch_id}"
            )

        # =================================================
        # 3. 가장 깊은 후보들 중
        #    Branch 중심선에 가장 가까운 robot 선택
        # =================================================

        marker_id = min(
            deepest_candidates,
            key=lateral_offset,
        )

        marker = (
            robots_by_id[
                marker_id
            ]
        )

        marker_depth = (
            branch_depth(
                marker_id
            )
        )

        marker_lateral = (
            lateral_offset(
                marker_id
            )
        )

        # =================================================
        # Marker로 역할 전환 후 현재 위치에 고정
        # =================================================

        freeze_role_robot(
            marker,
            "MARKER",
            branch_id,
        )

        # =================================================
        # Marker는 이제 Shepherd boundary ID 집합에서 제외
        # =================================================

        state[
            "initial_shepherd_ids"
        ].discard(
            marker_id
        )

        # =================================================
        # Branch state에 Marker 등록
        # =================================================

        state[
            "marker_id"
        ] = (
            marker_id
        )

        state[
            "marker_state"
        ] = (
            "UNVISITED"
        )

        # =================================================
        # Debug
        # =================================================

        print(
            "[InitialMarkerCreated] "
            f"branch={branch_id} "
            f"marker_id={marker_id} "
            f"max_depth={max_depth:.3f} "
            f"marker_depth={marker_depth:.3f} "
            f"lateral_offset={marker_lateral:.3f} "
            f"deepest_candidates="
            f"{len(deepest_candidates)} "
            f"state=UNVISITED"
        )


def select_runtime_branch(
    branch_id: str,
    branch_states: dict,
) -> None:

    state = branch_states[
        branch_id
    ]

    if (
        state["visit_state"]
        != "UNVISITED"
    ):
        return

    state["visit_state"] = (
        "ACTIVE"
    )

    print(
        "[BranchSelected] "
        f"branch={branch_id} "
        "state=ACTIVE"
    )

    print(
        "[AnchorBroadcast] "
        f"EXPLORE {branch_id}"
    )

def open_selected_branch(
    branch_id: str,
    branch_states: dict,
    robots_by_id: dict[
        int,
        environment.Robot,
    ],
) -> None:

    state = branch_states[
        branch_id
    ]

    shepherd_ids = set(
        state[
            "initial_shepherd_ids"
        ]
    )

    marker_id = state[
        "marker_id"
    ]

    print(
        "[BranchOpenStart] "
        f"branch={branch_id} "
        f"shepherd_count="
        f"{len(shepherd_ids)} "
        f"marker_id={marker_id}"
    )

    print(
        "[AnchorBroadcast] "
        f"branch={branch_id} "
        "command=RELEASE_INITIAL_SHEPHERD "
        f"ids={sorted(shepherd_ids)}"
    )

    # Marker는 initial_shepherd_ids에서
    # 이미 빠져 있으므로 영향을 받지 않는다.
    release_shepherd_group(
        shepherd_ids,
        robots_by_id,
    )

    state["opened"] = True

    print(
        "[BranchOpened] "
        f"branch={branch_id} "
        f"marker_retained="
        f"{marker_id}"
    )


COLORS = {
    "background": (255, 255, 255),
    "panel": (255, 255, 255),
    "floor": (255, 255, 255),
    "wall": (76, 156, 196),
    "text": (226, 232, 240),
    "muted": (125, 139, 156),
    "stage_text": (35, 45, 60),

    "normal_beam": (242, 242, 242),
    "open_beam": (145, 60, 220),
    "group_edge": (255, 184, 76),

    "leader": (255, 225, 55),
    "detected": (255, 83, 92),
    "shepherd": (94, 191, 235),
    "marker": (255, 165, 0),
    "marker_border": (255, 245, 120),
}

WINDOW_SIZE = (640, 420)
MAP_PANEL = pygame.Rect(150, 25, 340, 320)


ROBOT_DRAW_RADIUS = 2
WALL_DRAW_WIDTH = 2
WALL_CONTACT_RADIUS = (
    ROBOT_DRAW_RADIUS
    + WALL_DRAW_WIDTH / 2.0
)


def draw_map(
    surface: pygame.Surface,
    font: pygame.font.Font,
) -> None:

    pygame.draw.rect(
        surface,
        COLORS["panel"],
        MAP_PANEL,
        border_radius=8,
    )

    pygame.draw.polygon(
        surface,
        COLORS["floor"],
        OUTER_BOUNDARY,
    )

    pygame.draw.polygon(
        surface,
        COLORS["wall"],
        OUTER_BOUNDARY,
        width=WALL_DRAW_WIDTH,
    )

    # Central obstacle
    pygame.draw.rect(
        surface,
        COLORS["background"],
        OBSTACLE_RECT,
    )

    pygame.draw.rect(
        surface,
        COLORS["wall"],
        OBSTACLE_RECT,
        width=WALL_DRAW_WIDTH,
    )

    label = font.render(
        "Base",
        True,
        COLORS["text"],
    )

    label_rect = label.get_rect(
        midtop=(
            round(BASE_POSITION.x),
            round(BASE_POSITION.y) + 6,
        )
    )

    surface.blit(
        label,
        label_rect,
    )


def draw_robots(
    surface: pygame.Surface,
    robots: list[environment.Robot],
    color_reference_density: float,
    anchor: environment.Robot,
    lidar: AnchorLidar,
    junction_detected: bool,
    anchor_yaw_degrees: float,
) -> None:
    robot_color = (220, 225, 230)
    display_radius = ROBOT_DRAW_RADIUS

    # Only render sparse rays that the LiDAR actually classifies as opening
    # support.  Wall/corridor rays deliberately remain invisible.
    open_ray_stride = 4
    for ray_index, (angle, measured_range, is_open) in enumerate(zip(
        lidar.angles,
        lidar.ranges,
        lidar.open_support,
    )):
        if not is_open or ray_index % open_ray_stride != 0:
            continue
        radians = math.radians(
            anchor_yaw_degrees + angle
        )
        endpoint = anchor.position + pygame.Vector2(
            math.cos(radians),
            math.sin(radians),
        ) * LIDAR_MAX_RANGE
        pygame.draw.line(
            surface,
            COLORS["open_beam"],
            anchor.position,
            endpoint,
            width=1,
        )

    for robot in robots:

        if robot is anchor:
            continue

        center = (
            round(robot.position.x),
            round(robot.position.y),
        )

        # =============================================
        # MARKER
        # 주황색 본체 + 밝은 외곽 테두리
        # =============================================

        if robot.role == "MARKER":

            pygame.draw.circle(
                surface,
                COLORS["marker_border"],
                center,
                5,
                width=2,
            )

            pygame.draw.circle(
                surface,
                COLORS["marker"],
                center,
                3,
            )

            continue

        # =============================================
        # Other robots
        # =============================================

        # ---------------------------------------------
        # NORMAL
        # 회색 본체 + 얇은 검은색 테두리
        # ---------------------------------------------

        if robot.role == "NORMAL":

            # 바깥쪽 검은 테두리
            pygame.draw.circle(
                surface,
                (25, 45, 90),
                center,
                3,
            )

            # 안쪽 일반 로봇 본체
            pygame.draw.circle(
                surface,
                robot_color,
                center,
                ROBOT_DRAW_RADIUS,
            )

            continue

        # ---------------------------------------------
        # SHEPHERD
        # 기존 하늘색 그대로
        # ---------------------------------------------

        if robot.role == "SHEPHERD":

            pygame.draw.circle(
                surface,
                COLORS["shepherd"],
                center,
                display_radius,
            )

            continue

        # ---------------------------------------------
        # 그 외 역할
        # ---------------------------------------------

        pygame.draw.circle(
            surface,
            robot_color,
            center,
            display_radius,
        )

    pygame.draw.circle(
        surface,
        COLORS["leader"],
        (round(anchor.position.x), round(anchor.position.y)),
        3.0,
    )
    if junction_detected:
        pygame.draw.circle(
            surface,
            COLORS["detected"],
            (round(anchor.position.x), round(anchor.position.y)),
            6.0,
            width=2,
        )

    if anchor.velocity.length_squared() > environment.EPSILON:
        direction = anchor.velocity.normalize()
        endpoint = anchor.position + direction * 16.0
        pygame.draw.line(
            surface,
            COLORS["open_beam"],
            anchor.position,
            endpoint,
            width=3,
        )

def draw_controls(
    surface: pygame.Surface,
    font: pygame.font.Font,
    paused: bool,
    show_comm_links: bool,
    anchor_motion_mode: str,
    active_branch_id: str | None,
) -> None:

    run_state = (
        "PAUSED"
        if paused
        else "RUNNING"
    )

    comm_state = (
        "ON"
        if show_comm_links
        else "OFF"
    )

    text = (
        f"SPACE: Pause  |  "
        f"R: Reset  |  "
        f"C: Communication [{comm_state}]  |  "
        f"{run_state}"
    )

    rendered = font.render(
        text,
        True,
        COLORS["text"],
    )

    surface.blit(
        rendered,
        (
            18,
            WINDOW_SIZE[1] - 28,
        ),
    )

    # Anchor의 현재 상태 머신 단계를 map 왼쪽에 항상 표시한다.
    # 긴 stage 이름은 underscore 단위로 줄바꿈한다.
    stage_lines: list[str] = []
    current_line = ""

    for word in anchor_motion_mode.split("_"):

        candidate = (
            word
            if not current_line
            else f"{current_line} {word}"
        )

        if (
            current_line
            and len(candidate) > 16
        ):

            stage_lines.append(current_line)
            current_line = word

        else:
            current_line = candidate

    if current_line:
        stage_lines.append(current_line)

    status_rect = pygame.Rect(
        12,
        42,
        MAP_PANEL.left - 24,
        132,
    )

    pygame.draw.rect(
        surface,
        COLORS["panel"],
        status_rect,
        border_radius=6,
    )

    pygame.draw.rect(
        surface,
        COLORS["wall"],
        status_rect,
        width=1,
        border_radius=6,
    )

    status_font = pygame.font.Font(
        None,
        16,
    )

    status_rows = [
        "ANCHOR STAGE",
        *stage_lines,
        f"BRANCH: {active_branch_id or '-'}",
    ]

    for row_index, row in enumerate(status_rows):

        row_color = (
            COLORS["open_beam"]
            if row_index == 0
            else COLORS["stage_text"]
        )

        row_surface = status_font.render(
            row,
            True,
            row_color,
        )

        surface.blit(
            row_surface,
            (
                status_rect.left + 8,
                status_rect.top + 8 + row_index * 17,
            ),
        )

def create_staggered_grid_robots(
    robot_count: int,
) -> list[environment.Robot]:
    """Create a lattice and preassign its deployment-topology centre Anchor."""
    global INITIAL_ANCHOR_ID

    INITIAL_ANCHOR_ID = None
    robots: list[environment.Robot] = []
    spawn_slots: list[tuple[int, int, int, int]] = []
    dx = environment.GRID_SPACING
    dy = math.sqrt(3.0) / 3.0 * dx

    usable_left = (
        environment.center_x
        - environment.half_width
        + environment.ROBOT_RADIUS
        + environment.INITIAL_GRID_SIDE_MARGIN
    )
    usable_right = (
        environment.center_x
        + environment.half_width
        - environment.ROBOT_RADIUS
        - environment.INITIAL_GRID_SIDE_MARGIN
    )
    top = (
        environment.center_y
        + environment.half_width
        + 12.0 * environment.MAP_SCALE
    )
    bottom = (
        environment.center_y
        + environment.half_width
        + environment.base_length
        - environment.ROBOT_RADIUS
        - 7.0 * environment.MAP_SCALE
    )
    max_columns = max(1, int((usable_right - usable_left) // dx) + 1)

    robot_id = 0
    row = 0
    while robot_id < robot_count:
        y = bottom - row * dy
        if y < top:
            print(f"Warning: only {len(robots)} robots fit.")
            break

        row_count = max_columns if row % 2 == 0 else max(1, max_columns - 1)
        row_width = (row_count - 1) * dx
        row_left = environment.center_x - 0.5 * row_width

        for column in range(row_count):
            if robot_id >= robot_count:
                break

            robot = environment.Robot(
                row_left + column * dx,
                y,
                robot_id,
            )
            robots.append(robot)
            spawn_slots.append((row, column, row_count, robot_id))
            robot_id += 1
        row += 1

    if not spawn_slots:
        return robots

    used_rows = sorted({
        row
        for row, _, _, _
        in spawn_slots
    })

    # 군집 정중앙보다 4 row 앞쪽에서
    # Anchor를 선택한다.
    center_index = (
        len(used_rows) // 2
    )

    anchor_index = min(
        len(used_rows) - 1,
        center_index + 16,
    )

    anchor_row = (
        used_rows[
            anchor_index
        ]
    )

    anchor_row_slots = [
        item
        for item in spawn_slots
        if item[0] == anchor_row
    ]

    _, _, _, INITIAL_ANCHOR_ID = min(
        anchor_row_slots,
        key=lambda item:
        abs(
            item[1]
            - (item[2] - 1) / 2.0
        ),
    )

    print(
        "[InitialAnchor] "
        f"id={INITIAL_ANCHOR_ID} "
        f"row={anchor_row} "
        "source=FORWARD_DEPLOYMENT_SLOT"
    )
    return robots


def compute_sph_only_forces(
    robots: list[environment.Robot],
    physics_grid,
    reference_density: float,
) -> None:
    """HydroSwarm Eqs. (12)-(14): f_SPH = f_press + f_vis."""
    h_sq = environment.SMOOTHING_LENGTH**2

    for robot_i in robots:
        pressure_force = pygame.Vector2()
        viscosity_force = pygame.Vector2()

        for robot_j in environment.iter_physics_neighbor_candidates(
            robot_i,
            physics_grid,
        ):
            if robot_i is robot_j:
                continue

            r_ij = robot_i.position - robot_j.position
            distance_sq = r_ij.length_squared()

            if (
                distance_sq <= environment.EPSILON
                or distance_sq > h_sq
            ):
                continue

            gradient = environment.spiky_gradient(
                r_ij,
                environment.SMOOTHING_LENGTH,
            )
            pressure_coefficient = (
                robot_i.pressure
                / max(robot_i.density**2, environment.EPSILON)
                + robot_j.pressure
                / max(robot_j.density**2, environment.EPSILON)
            )
            pressure_force += -pressure_coefficient * gradient

            v_ij = robot_i.velocity - robot_j.velocity
            approach = v_ij.dot(r_ij)
            if approach < 0.0:
                mu_ij = (
                    environment.SMOOTHING_LENGTH
                    * approach
                    / (
                        distance_sq
                        + 0.01 * environment.SMOOTHING_LENGTH**2
                    )
                )
                c_i_sq = (
                    robot_i.pressure
                    + environment.PRESSURE_GAIN * robot_i.density
                ) / max(robot_i.density, environment.EPSILON)
                c_j_sq = (
                    robot_j.pressure
                    + environment.PRESSURE_GAIN * robot_j.density
                ) / max(robot_j.density, environment.EPSILON)
                c_ij = 0.5 * (
                    math.sqrt(max(c_i_sq, 0.0))
                    + math.sqrt(max(c_j_sq, 0.0))
                )
                mean_density = 0.5 * (
                    robot_i.density + robot_j.density
                )
                pi_ij = (
                    -environment.VISCOSITY_XI1 * c_ij * mu_ij
                    + environment.VISCOSITY_XI2 * mu_ij**2
                ) / max(mean_density, environment.EPSILON)
                viscosity_force += -pi_ij * gradient

        pressure_force = environment.limit_vector(
            pressure_force,
            environment.SPH_PRESSURE_FORCE_LIMIT,
        )
        viscosity_force = environment.limit_vector(
            viscosity_force,
            environment.SPH_VISCOSITY_FORCE_LIMIT,
        )
        robot_i.acceleration = environment.limit_vector(
            pressure_force + viscosity_force,
            environment.MAX_ACCELERATION,
        )
        robot_i.filtered_acceleration.update(robot_i.acceleration)
        robot_i.last_sph_pressure_force = pressure_force.length()
        robot_i.last_goal_force = 0.0

def integrate_sph_only_robot(
    robot: environment.Robot,
    dt: float,
) -> None:

    old_position = (
        robot.position.copy()
    )

    # =============================================
    # 1. SPH acceleration -> velocity
    # =============================================

    robot.velocity += (
        robot.acceleration
        * dt
    )

    speed_limit = (
        environment.INITIAL_SAFE_MAX_SPEED
    )

    if (
        robot.velocity.length()
        > speed_limit
    ):
        robot.velocity.scale_to_length(
            speed_limit
        )

    # =============================================
    # 2. X movement
    # =============================================

    x_position = pygame.Vector2(
        robot.position.x
        + robot.velocity.x * dt,
        robot.position.y,
    )

    if physical_is_walkable(
        x_position,
        robot.radius,
    ):
        robot.position.x = (
            x_position.x
        )

    else:
        robot.velocity.x = (
            -robot.velocity.x
            * environment.INITIAL_WALL_RESTITUTION
        )

    # =============================================
    # 3. Y movement
    # =============================================

    y_position = pygame.Vector2(
        robot.position.x,
        robot.position.y
        + robot.velocity.y * dt,
    )

    if physical_is_walkable(
        y_position,
        robot.radius,
    ):
        robot.position.y = (
            y_position.y
        )

    else:
        robot.velocity.y = (
            -robot.velocity.y
            * environment.INITIAL_WALL_RESTITUTION
        )

    robot.commanded_velocity.update(
        robot.velocity
    )

    # =============================================
    # 4. 실제 관측 속도
    # =============================================

    realized_velocity = (
        robot.position
        - old_position
    ) / max(
        dt,
        environment.EPSILON,
    )

    robot.observed_velocity.update(
        realized_velocity.x,
        realized_velocity.y,
    )

    robot.previous_position.update(
        old_position
    )

    robot.acceleration.update(
        0.0,
        0.0,
    )

# =========================================================
# Main
# =========================================================

def main() -> None:

    pygame.init()

    if HEADLESS:
        print(
            "[HeadlessMode] "
            "enabled=True "
            "timestep=1/FPS"
        )

    pygame.display.set_caption(
        "Physical DFS | SPH Base Ingress"
    )

    screen = pygame.display.set_mode(
        WINDOW_SIZE
    )

    clock = pygame.time.Clock()

    font = pygame.font.Font(
        None,
        22,
    )

    # =====================================================
    # Environment / map setup
    # =====================================================

    environment.MAP_SCALE = MAP_SCALE

    environment.center_x = (
        BASE_CORRIDOR_RECT.left
        + BASE_CORRIDOR_RECT.right
    ) / 2.0

    environment.center_y = (
        BASE_CORRIDOR_RECT.top
        - BASE_CORRIDOR_RECT.width / 2.0
    )

    environment.corridor_width = (
        BASE_CORRIDOR_RECT.width
    )

    environment.half_width = (
        BASE_CORRIDOR_RECT.width // 2
    )

    environment.normal_length = (
        scale_point((208, 253))[1]
        - scale_point((208, 39))[1]
    )

    environment.right_length = (
        scale_point((524, 253))[0]
        - scale_point((302, 253))[0]
    )

    environment.base_length = (
        BASE_CORRIDOR_RECT.height
    )

    environment.cross_points = list(
        OUTER_BOUNDARY
    )

    environment.junction_rect = (
        pygame.Rect(
            scale_point((208, 253)),
            (
                BASE_CORRIDOR_RECT.width,
                BASE_CORRIDOR_RECT.width,
            ),
        )
    )

    environment.up_rect = pygame.Rect(
        scale_point((208, 39)),
        (
            BASE_CORRIDOR_RECT.width,
            environment.normal_length,
        ),
    )

    environment.left_rect = pygame.Rect(
        scale_point((78, 253)),
        (
            scale_point((208, 253))[0]
            - scale_point((78, 253))[0],

            BASE_CORRIDOR_RECT.width,
        ),
    )

    environment.right_rect = pygame.Rect(
        scale_point((302, 253)),
        (
            environment.right_length,
            BASE_CORRIDOR_RECT.width,
        ),
    )

    environment.bottom_rect = (
        BASE_CORRIDOR_RECT.copy()
    )

    environment.BRANCH_LENGTHS.update(
        UP=float(
            environment.normal_length
        ),
        LEFT=float(
            environment.left_rect.width
        ),
        RIGHT=float(
            environment.right_length
        ),
    )

    environment.BASE_POSITION = (
        BASE_POSITION.copy()
    )

    environment.BASE_COMPRESSION_CENTER = (
        BASE_POSITION.copy()
    )

    environment.JUNCTION_STAGING_POSITION = (
        pygame.Vector2(
            environment.junction_rect.center
        )
    )

    environment.INITIAL_INGRESS_TARGET_Y = (
        environment.junction_rect.top
        - 18.0 * environment.MAP_SCALE
    )

    environment.floor_surface = pygame.Surface(
        (
            environment.SCREEN_WIDTH,
            environment.SCREEN_HEIGHT,
        ),
        pygame.SRCALPHA,
    )

    environment.floor_surface.fill(
    (0, 0, 0, 0)
    )

    # =============================================
    # 실제 자유공간
    # =============================================

    pygame.draw.polygon(
        environment.floor_surface,
        (255, 255, 255, 255),
        environment.cross_points,
    )

    # =============================================
    # 실제 obstacle은 walkable 영역에서 제거
    # =============================================

    pygame.draw.rect(
        environment.floor_surface,
        (0, 0, 0, 0),
        OBSTACLE_RECT,
    )

    # =============================================
    # 실제 물리 map mask
    # =============================================

    environment.walkable_mask = (
        pygame.mask.from_surface(
            environment.floor_surface
        )
    )

    # =====================================================
    # Robot / SPH parameters
    # =====================================================

    # 로봇 수 증가
    environment.ROBOT_COUNT = (
        300
    )

    environment.ROBOT_RADIUS = (
        0.8 * SIMULATION_LENGTH_SCALE
    )

    environment.GRID_SPACING = (
        2.4 * SIMULATION_LENGTH_SCALE
    )

    environment.GRID_ROW_SPACING = (
        environment.GRID_SPACING
    )

    environment.INITIAL_GRID_SIDE_MARGIN = (
        2.0 * SIMULATION_LENGTH_SCALE
    )

    environment.SMOOTHING_LENGTH = (
        18.0 * SIMULATION_LENGTH_SCALE
    )

    environment.SAFE_RADIUS = (
        7.5 * MAP_SCALE
    )

    environment.SPH_CELL_SIZE = (
        environment.SMOOTHING_LENGTH
    )

    REFERENCE_DENSITY = 0.025

    # 이 값은 reference density 계산에는 더 이상 사용하지 않는다.
    # 다른 spacing 관련 로직이 사용할 수 있으므로 삭제하지 않음
    environment.REFERENCE_EQUILIBRIUM_SPACING = (
        5.0 * SIMULATION_LENGTH_SCALE
    )
    # 벽 충돌 후 더 강하게 반사
    environment.INITIAL_WALL_RESTITUTION = (
        0.20
    )

    # SPH pressure 증가
    environment.PRESSURE_GAIN = (
        750.0 * SIMULATION_LENGTH_SCALE**2
    )

    environment.VISCOSITY_XI1 = (
        0.7
    )

    environment.VISCOSITY_XI2 = (
        0.7
    )

    # pressure가 너무 일찍 clamp되지 않도록 증가
    environment.SPH_PRESSURE_FORCE_LIMIT = (
        100.0 * SIMULATION_LENGTH_SCALE
    )

    environment.SPH_VISCOSITY_FORCE_LIMIT = (
        45.0 * SIMULATION_LENGTH_SCALE
    )

    # 증가한 pressure를 실제 acceleration으로 허용
    environment.MAX_ACCELERATION = (
        110.0 * SIMULATION_LENGTH_SCALE
    )

    # 빨라진 swarm이 7.5에서 다시 잘리지 않도록 증가
    environment.INITIAL_SAFE_MAX_SPEED = (
        11.0 * SIMULATION_LENGTH_SCALE
    )

    environment.INITIAL_JUNCTION_MAX_SPEED = (
        8.0 * SIMULATION_LENGTH_SCALE
    )

    environment.COMM_RANGE = (
        24.0 * SIMULATION_LENGTH_SCALE
    )

    environment.COMM_SAFE_DISTANCE = (
        15.0 * SIMULATION_LENGTH_SCALE
    )



    environment.COMM_BARRIER_START = (
        15.0 * SIMULATION_LENGTH_SCALE
    )

    environment.COMM_GUARD_START = (
        12.0 * SIMULATION_LENGTH_SCALE
    )

    environment.COMM_GUARD_HARD_LIMIT = (
        float("inf")
    )

    environment.CELL_SIZE = max(
        environment.SMOOTHING_LENGTH,
        environment.COMM_RANGE,
    )

    environment.create_grid_robots = (
        create_staggered_grid_robots
    )

    (
        robots,
        spacing_based_reference_density,
        color_reference_density,
    ) = environment.initialize_simulation()

    # HydroSwarm rho_0를 직접 지정
    reference_density = REFERENCE_DENSITY

    print(
        "[SPHReferenceDensity] "
        f"spacing_based="
        f"{spacing_based_reference_density:.6f} "
        f"effective="
        f"{reference_density:.6f}"
    )

    # =====================================================
    # Anchor initialization
    # =====================================================

    if INITIAL_ANCHOR_ID is None:

        raise RuntimeError(
            "Initial Anchor was not assigned."
        )

    robots_by_id = {
        robot.robot_id: robot
        for robot in robots
    }

    anchor = robots_by_id[
        INITIAL_ANCHOR_ID
    ]

    swarm_robots = [
        robot
        for robot in robots
        if robot is not anchor
    ]

    anchor_lidar = (
        AnchorLidar()
    )

    # =====================================================
    # Runtime state
    # =====================================================

    junction_detected = False

    junction_entrance_stopped = (
        False
    )

    anchor_locally_centered = (
        False
    )

    anchor_center_stable_count = (
        0
    )

    entrance_target_samples: list[
        pygame.Vector2
    ] = []

    locked_center_target_local: (
        pygame.Vector2 | None
    ) = None

    anchor_center_traveled_local = (
        pygame.Vector2()
    )

    base_entrance_corners_local: (
        tuple[
            pygame.Vector2,
            pygame.Vector2,
        ]
        | None
    ) = None

    confirmed_branches: list[
    dict
    ] = []

    branch_states: dict = {}

    # =====================================================
    # Initial Junction formation runtime state
    # =====================================================

    initial_shepherds_ready = (
        False
    )

    # =====================================================
    # Initial Junction broadcast runtime state
    # =====================================================

    # Junction이 확정된 뒤
    # Initial Shepherd formation이 끝날 때까지
    # Anchor가 Junction message를 broadcast한다.
    junction_broadcast_active = (
        False
    )

    # JUNCTION_CONFIRMED message를
    # 한 번이라도 직접 수신한 robot IDs.
    junction_message_received_ids: set[int] = (
        set()
    )

    markers_ready = (
        False
    )

    branch_order: list[str] = []

    active_branch_id: str | None = None

    first_branch_opened = False

    # =============================================
    # Runtime Anchor heading / branch exploration
    # =============================================

    # 처음 시작할 때만 고정 초기 yaw를 사용하고,
    # 이후에는 Branch 선택에 따라 runtime으로 갱신한다.
    anchor_yaw_deg = (
        ANCHOR_YAW_DEG
    )

    # 현재 Junction에서 Branch angle이 정의된 기준 heading.
    # Branch B0/B1/B2는 이 heading 기준의 local angle이다.
    junction_reference_yaw_deg: float | None = (
        None
    )

    # ROOT_SETUP
    # BRANCH_ENTRY
    # BRANCH_EXPLORE
    # BACKTRACK_WAIT_SHEPHERD
    # PRESSURE_PUSH
    # FLOW_BACKTRACK
    # BACKTRACK_CORNER_TURN
    # RETURN_JUNCTION_ENTRANCE
    anchor_motion_mode = (
        "ROOT_SETUP"
    )

    # 동일 상태는 반복 출력하지 않고, 실제 state transition만 기록.
    last_logged_anchor_motion_mode: str | None = None

    final_push_branch_id: str | None = None

    final_push_ids: set[int] = set()

    final_side_branch_ids: set[str] = set()

    final_side_merge_stable_count = 0

    final_base_return_stable_count = 0

    final_anchor_return_traveled = 0.0

    # Junction center에서 선택 Branch mouth를
    # 지나가기 위한 local odometry.
    anchor_branch_entry_target = (
        0.0
    )

    anchor_branch_entry_traveled = (
        0.0
    )

    # Branch 입구 통과가 끝난 뒤,
# 실제 Branch exploration 동안 이동한 local odometry.
    branch_explore_traveled = (
        0.0
    )

    # =============================================
    # Backtracking runtime state
    # =============================================

    dead_end_stable_count = (
        0
    )

    backtrack_seed_id: (
        int | None
    ) = None

    backtrack_required_width = (
        KNOWN_CORRIDOR_WIDTH
    )

    backtrack_trigger_reason: (
        str | None
    ) = None

    # Pressure Push가 시작된 순간의 탐색 heading.
    #
    # Anchor가 FLOW_BACKTRACK에서 180도 회전하더라도
    # Shepherd push 방향은 이 heading을 계속 사용한다.
    backtrack_push_yaw_deg: (
        float | None
    ) = None

    # Wall detach가 끝난 뒤 돌아갈 state.
    #
    # 처음 backtracking:
    #   WALL_DETACH -> PRESSURE_PUSH
    #
    # Flow 중 벽 재접촉:
    #   WALL_DETACH -> FLOW_BACKTRACK
    backtrack_wall_detach_resume_mode = (
        "PRESSURE_PUSH"
    )

    # 한 번 선택한 wall-detach 방향은
# 해당 detach episode가 끝날 때까지 유지한다.
    backtrack_wall_detach_direction: str | None = (
        None
    )

    return_junction_entrance_reached = (
    False
    )
    # =============================================
    # Parent Junction return-centering state
    # =============================================

    return_base_entrance_corners_local: (
        tuple[
            pygame.Vector2,
            pygame.Vector2,
        ]
        | None
    ) = None

    return_center_target_samples: list[
        pygame.Vector2
    ] = []

    return_locked_center_target_local: (
        pygame.Vector2 | None
    ) = None

    return_center_traveled_local = (
        pygame.Vector2()
    )

    return_center_stable_count = (
        0
    )

    swarm_return_stable_count = (
        0
    )

    backtrack_event_valid = (
    False
    )

    detected_marker_id: (
        int | None
    ) = None

    origin_marker_cleared = False

    backtrack_required_count = (
        0
    )

    backtrack_formation_ids: set[int] = (
        set()
    )

    backtrack_chain_order: list[int] = (
        []
    )

    backtrack_formation_stable_count = (
        0
    )

    backtrack_reverse_stable_count = (
        0
    )

   
    # 새 corridor에 제대로 들어왔는지 연속 확인
    backtrack_corner_exit_stable_count = 0

    # 몇 번째 corner를 통과했는지 diagnostic
    backtrack_corner_count = 0


    # initial scan
    anchor_lidar.scan(
        anchor.position,
        anchor_yaw_deg,
    )

    paused = False

    # Communication link는 시작할 때부터 보이도록 한다.
    show_comm_links = True

    running = True

    # =====================================================
    # Main loop
    # =====================================================

    while running:

        # -------------------------------------------------
        # Events
        # -------------------------------------------------

        for event in (
            pygame.event.get()
        ):

            if (
                event.type
                == pygame.QUIT
            ):

                running = False

            elif (
                event.type
                == pygame.KEYDOWN
            ):

                # =========================================
                # SPACE : Pause / Resume
                # =========================================

                if (
                    event.key
                    == pygame.K_SPACE
                ):

                    paused = (
                        not paused
                    )

                    print(
                        "[Pause] "
                        f"paused={paused}"
                    )

                # =========================================
                # C : Communication links ON / OFF
                # =========================================

                elif (
                    event.key
                    == pygame.K_c
                ):

                    show_comm_links = (
                        not show_comm_links
                    )

                    print(
                        "[CommunicationView] "
                        f"visible="
                        f"{show_comm_links}"
                    )

                # =========================================
                # R : Full simulation reset
                # =========================================

                elif (
                    event.key
                    == pygame.K_r
                ):

                    print(
                        "[SimulationReset]"
                    )

                    pygame.quit()

                    os.execv(
                        sys.executable,
                        [
                            sys.executable,
                            *sys.argv,
                        ],
                    )

                elif (
                    event.key
                    == pygame.K_ESCAPE
                ):

                    running = False

        # -------------------------------------------------
        # Time step
        # -------------------------------------------------

        if HEADLESS:

            # 물리 timestep은 일반 실행과 같은 FPS 기준으로 유지하되,
            # 실시간 clock 대기는 하지 않는다.
            frame_dt = (
                1.0 / environment.FPS
            )

        else:

            frame_dt = max(
                clock.tick(
                    environment.FPS
                ) / 1000.0,
                1.0 / 240.0,
            )

        frame_dt = min(
            frame_dt,
            environment.INITIAL_INGRESS_MAX_DT,
        )

        substep_dt = (
            frame_dt / 8.0
        )

        # =================================================
        # Physics substeps
        # =================================================

        for substep_index in range(
            0 if paused else 8
        ):

            # =================================================
            # 이 physics substep에서 Shepherd가 NORMAL에게
            # 전달한 실제 physical contact response.
            #
            # 매 substep 새로 계산한다.
            # =================================================

            backtrack_contact_accelerations: dict[
                int,
                pygame.Vector2,
            ] = {}

            environment.simulation_time += (
                substep_dt
            )

            # ---------------------------------------------
            # Fresh Anchor-local sensing
            # ---------------------------------------------

            anchor_lidar.scan(
                anchor.position,
                anchor_yaw_deg,
            )

            observation = (
                LocalObservationBuilder.build(
                    anchor,
                    robots,
                    anchor_lidar,
                    anchor_yaw_deg,
                )
            )
            # =================================================
            # Junction message broadcast / reception
            # =================================================

            if (
                junction_broadcast_active
                and
                not initial_shepherds_ready
            ):

                received_now = (
                    update_junction_broadcast_receivers(
                        observation,
                        robots_by_id,
                    )
                )

                newly_received = (
                    received_now
                    - junction_message_received_ids
                )

                junction_message_received_ids.update(
                    received_now
                )

                if newly_received:

                    print(
                        "[JunctionMessageReceived] "
                        f"new_ids="
                        f"{sorted(newly_received)} "
                        f"total="
                        f"{len(junction_message_received_ids)}"
                    )

            # =================================================
            # 1. Anchor approaches Junction
            # =================================================

            if not junction_entrance_stopped:

                entrance_reached = (
                    False
                )

                if (
                    substep_index == 0
                ):

                    (
                        entrance_reached,
                        _,
                        _,
                        _,
                        _,
                    ) = (
                        update_junction_entrance_detector(
                            anchor_lidar
                        )
                    )

                if entrance_reached:

                    junction_entrance_stopped = (
                        True
                    )

                    stop_anchor(
                        anchor
                    )

                    freeze_stationary_threshold(
                        anchor_lidar
                    )

                    left_wall = (
                        anchor_lidar
                        .lateral_baseline_left
                    )

                    right_wall = (
                        anchor_lidar
                        .lateral_baseline_right
                    )

                    if (
                        left_wall is None
                        or right_wall is None
                    ):

                        raise RuntimeError(
                            "No valid Base corridor "
                            "wall calibration before Junction."
                        )

                    entrance_target_samples.clear()

                    locked_center_target_local = (
                        None
                    )

                    base_entrance_corners_local = (
                        pygame.Vector2(
                            0.0,
                            -left_wall,
                        ),
                        pygame.Vector2(
                            0.0,
                            right_wall,
                        ),
                    )

                    anchor_center_traveled_local.update(
                        0.0,
                        0.0,
                    )

                    anchor_center_stable_count = (
                        0
                    )

                    print(
                        "[JunctionEntranceDetected] "
                        f"W="
                        f"{anchor_lidar.locked_adaptive_w:.2f} "
                        f"T="
                        f"{anchor_lidar.locked_w_tau_threshold:.2f} "
                        f"BL=(0.00,{-left_wall:.2f}) "
                        f"BR=(0.00,{right_wall:.2f})"
                    )

                else:

                    follow_corridor_locally(
                        anchor,
                        observation,
                        anchor_yaw_deg,
                        substep_dt,
                    )

            # =================================================
            # 2. Stationary Junction confirmation
            # =================================================

            elif not junction_detected:

                stop_anchor(
                    anchor
                )

                if (
                    substep_index == 0
                    and
                    update_stationary_junction_confirmation(
                        anchor_lidar
                    )
                ):

                    junction_detected = (
                        True
                    )

                    # =============================================
                    # Junction confirmed
                    # → Anchor starts broadcast
                    # =============================================

                    junction_broadcast_active = (
                        True
                    )

                    junction_message_received_ids.clear()

                    print(
                        "[JunctionConfirmed]"
                    )

                    print(
                        "[AnchorBroadcast] "
                        "command=JUNCTION_CONFIRMED "
                        "active=True"
                    )

            # =================================================
            # 3. Estimate Junction center
            # =================================================

            elif (
                not anchor_locally_centered
                and
                locked_center_target_local
                is None
            ):

                stop_anchor(
                    anchor
                )

                if (
                    substep_index == 0
                    and
                    base_entrance_corners_local
                    is not None
                ):

                    (
                        base_left,
                        base_right,
                    ) = (
                        base_entrance_corners_local
                    )

                    front_corners = (
                        extract_front_mouth_corners(
                            anchor_lidar
                        )
                    )

                    if (
                        front_corners
                        is None
                    ):

                        candidate = (
                            None
                        )

                    else:

                        (
                            front_left,
                            front_right,
                        ) = front_corners

                        candidate = (
                            estimate_local_junction_center(
                                base_left,
                                base_right,
                                front_left,
                                front_right,
                            )
                        )

                    if candidate is None:

                        entrance_target_samples.clear()

                        print(
                            "[EntranceStationaryScan] "
                            "diagonal_center=NOT_AVAILABLE"
                        )

                    else:

                        if (
                            entrance_target_samples
                        ):

                            previous = (
                                entrance_target_samples[
                                    -1
                                ]
                            )

                            if (
                                candidate
                                - previous
                            ).length() > (
                                ANCHOR_TARGET_SAMPLE_TOLERANCE
                            ):

                                entrance_target_samples.clear()

                        entrance_target_samples.append(
                            candidate.copy()
                        )

                        print(
                            "[EntranceStationaryScan] "
                            f"center_candidate="
                            f"({candidate.x:.2f},"
                            f"{candidate.y:.2f}) "
                            f"samples="
                            f"{len(entrance_target_samples)}/"
                            f"{ANCHOR_ENTRANCE_STATIONARY_SCANS}"
                        )

                        if (
                            len(
                                entrance_target_samples
                            )
                            >=
                            ANCHOR_ENTRANCE_STATIONARY_SCANS
                        ):

                            total = (
                                pygame.Vector2()
                            )

                            for sample in (
                                entrance_target_samples
                            ):

                                total += sample

                            locked_center_target_local = (
                                total
                                / len(
                                    entrance_target_samples
                                )
                            )

                            anchor_center_traveled_local.update(
                                0.0,
                                0.0,
                            )

                            print(
                                "[JunctionCenterTargetLocked] "
                                f"target=("
                                f"{locked_center_target_local.x:.2f},"
                                f"{locked_center_target_local.y:.2f})"
                            )

            # =================================================
            # 4. Anchor moves to local Junction center
            # =================================================

            elif (
                not anchor_locally_centered
            ):

                (
                    centered_now,
                    local_delta,
                ) = (
                    move_anchor_to_locked_center(
                        anchor,
                        locked_center_target_local,
                        anchor_center_traveled_local,
                        anchor_yaw_deg,
                        substep_dt,
                    )
                )

                anchor_center_traveled_local += (
                    local_delta
                )

                if centered_now:

                    if (
                        substep_index == 0
                    ):

                        anchor_center_stable_count += (
                            1
                        )

                else:

                    anchor_center_stable_count = (
                        0
                    )

                if (
                    anchor_center_stable_count
                    >=
                    ANCHOR_CENTER_STABLE_SCANS
                ):

                    anchor_locally_centered = (
                        True
                    )

                    stop_anchor(
                        anchor
                    )

                    print(
                        "[AnchorLocallyCentered] "
                        f"target=("
                        f"{locked_center_target_local.x:.2f},"
                        f"{locked_center_target_local.y:.2f}) "
                        f"travel=("
                        f"{anchor_center_traveled_local.x:.2f},"
                        f"{anchor_center_traveled_local.y:.2f})"
                    )

            else:

                if (
                    anchor_motion_mode
                    == "ROOT_SETUP"
                ):

                    stop_anchor(
                        anchor
                    )

            # =================================================
            # 5. Register Branch geometry
            #
            # IMPORTANT:
            # substep 내부에서 등록.
            #
            # Anchor가 centered 된 다음 substep에서는
            # 위쪽에서 이미 fresh LiDAR scan을 수행했으므로
            # 이 위치에서 branch를 등록할 수 있다.
            # =================================================

            if (
                anchor_locally_centered
                and
                not confirmed_branches
            ):

                confirmed_branches = (
                    register_outgoing_branches(
                        anchor_lidar
                    )
                )
                if confirmed_branches:

                    junction_reference_yaw_deg = (
                        anchor_yaw_deg
                    )

                    print(
                        "[JunctionReferenceHeading] "
                        f"yaw={junction_reference_yaw_deg:.2f}"
                    )

                branch_states = {
                    branch["id"]: {

                        "initial_shepherd_ids":
                        set(),

                        "backtrack_shepherd_ids":
                        set(),

                        "initial_sealed":
                        False,

                        "marker_id":
                        None,

                        "marker_state":
                        "UNVISITED",

                        "visit_state":
                        "UNVISITED",
                    }

                    for branch
                    in confirmed_branches
                }

                if confirmed_branches:

                    print(
                        "[BranchSectorsRegistered] "
                        + " ".join(
                            (
                                f'{branch["id"]}:'
                                f'{branch["start_angle"]:.1f}'
                                f'~'
                                f'{branch["end_angle"]:.1f}'
                            )

                            for branch
                            in confirmed_branches
                        )
                    )
            # =================================================
            # 5.5. Independent Anchor Branch exploration
            # =================================================

            if (
                anchor_motion_mode
                == "BRANCH_ENTRY"
            ):

                # =============================================
                # Junction 내부에서는 좌/우 wall range가
                # 아직 Branch corridor wall을 의미하지 않을 수 있다.
                #
                # 따라서 선택 Branch의 heading으로
                # straight local motion만 수행한다.
                # =============================================

                local_delta = (
                    integrate_anchor_local_command(
                        anchor,
                        pygame.Vector2(
                            ANCHOR_FORWARD_SPEED,
                            0.0,
                        ),
                        anchor_yaw_deg,
                        substep_dt,
                    )
                )

                # Anchor-local odometry만 누적
                anchor_branch_entry_traveled += (
                    max(
                        0.0,
                        local_delta.x,
                    )
                )

                role_debug(
                    "anchor-branch-entry",
                    (
                        "[AnchorBranchEntry] "
                        f"branch={active_branch_id} "
                        f"travel="
                        f"{anchor_branch_entry_traveled:.2f}/"
                        f"{anchor_branch_entry_target:.2f}"
                    ),
                )

                if (
                    anchor_branch_entry_traveled
                    >=
                    anchor_branch_entry_target
                ):
                    origin_marker_cleared = (
                        False
                    )

                    anchor_motion_mode = (
                        "BRANCH_EXPLORE"
                    )

                    print(
                        "[AnchorBranchEntryComplete] "
                        f"branch={active_branch_id} "
                        f"travel="
                        f"{anchor_branch_entry_traveled:.2f}"
                    )


            elif (
                anchor_motion_mode
                == "BRANCH_EXPLORE"
            ):

                if (
                    active_branch_id
                    is None
                ):

                    raise RuntimeError(
                        "BRANCH_EXPLORE without active branch."
                    )

                active_state = (
                    branch_states[
                        active_branch_id
                    ]
                )

                # =============================================
                # 현재 Branch 입구에 있던 자기 Marker는
                # 탐색 시작 시 다시 감지하면 안 된다.
                # =============================================

                origin_marker_id = (
                    active_state[
                        "marker_id"
                    ]
                )

                # =============================================
                # 처음 Branch로 들어갈 때만
                # 자기 입구 Marker를 일시적으로 무시한다.
                #
                # Anchor가 실제로 Marker를 지나쳐
                # Marker가 Anchor 뒤쪽으로 넘어간 순간부터는
                # 더 이상 무시하지 않는다.
                #
                # 따라서 cycle을 돌아 다시 앞에서 만나면
                # 정상적으로 Marker가 감지된다.
                # =============================================

                if (
                    not origin_marker_cleared
                    and
                    origin_marker_id
                    is not None
                ):

                    origin_marker_local = next(
                        (
                            local_position
                            for robot_id, local_position
                            in zip(
                                observation.robot_ids,
                                observation.relative_positions,
                            )
                            if (
                                robot_id
                                == origin_marker_id
                            )
                        ),
                        None,
                    )

                    if (
                        origin_marker_local
                        is not None
                        and
                        origin_marker_local.x
                        < -2.0
                    ):

                        origin_marker_cleared = (
                            True
                        )

                        print(
                            "[OriginMarkerCleared] "
                            f"branch="
                            f"{active_branch_id} "
                            f"marker_id="
                            f"{origin_marker_id}"
                        )

                marker_to_ignore = (
                    None
                    if origin_marker_cleared
                    else origin_marker_id
                )

                # =====================================================
                # Branch 탐색을 충분히 진행하기 전에는
                # Root Junction 주변의 sibling Marker를 무시한다.
                #
                # map/world position은 사용하지 않고,
                # Branch exploration 이후 누적된 local odometry만 사용한다.
                # =====================================================

                if (
                    branch_explore_traveled
                    < MARKER_ACTIVATION_DISTANCE
                ):

                    visible_marker_id = (
                        None
                    )

                else:

                    visible_marker_id = (
                        detect_visible_marker_ahead(
                            observation,
                            robots_by_id,
                            anchor_lidar,
                            ignore_marker_id=
                            marker_to_ignore,
                        )
                    )
                # =============================================
                # 1. Marker가 보이면 최우선으로 STOP
                # =============================================

                if (
                    visible_marker_id
                    is not None
                ):

                    stop_anchor(
                        anchor
                    )

                    detected_marker_id = (
                        visible_marker_id
                    )

                    detected_marker_branch = (
                        find_branch_id_by_marker(
                            visible_marker_id,
                            branch_states,
                        )
                    )

                    # 반대편에서 만난 Marker가
                    # 아직 UNVISITED이면 VISITED.
                    if (
                            detected_marker_branch
                            is not None
                            and
                            detected_marker_branch
                            != active_branch_id
                        ):
                            mark_branch_visited(
                                detected_marker_branch,
                                branch_states,
                                reason=
                                "MARKER_REACHED_FROM_OPPOSITE_SIDE",
                            )



                    backtrack_required_width = (
                        estimate_current_corridor_width(
                            observation.lidar_scan
                        )
                    )

                    backtrack_seed_id = (
                        None
                    )

                    backtrack_trigger_reason = (
                        "MARKER"
                    )
                    backtrack_event_valid = (
                        True
                    )


                    dead_end_stable_count = (
                        0
                    )

                    anchor_motion_mode = (
                        "BACKTRACK_WAIT_SHEPHERD"
                    )

                    print(
                        "[BacktrackTrigger] "
                        f"reason=MARKER "
                        f"active_branch="
                        f"{active_branch_id} "
                        f"marker_id="
                        f"{visible_marker_id} "
                        f"corridor_width="
                        f"{backtrack_required_width:.2f}"
                    )

                else:

                    # =============================================
                    # 2. Marker가 없으면 LiDAR free gap 검사
                    # =============================================

                    target_gap_angle = (
                        find_branch_free_gap(
                            observation.lidar_scan
                        )
                    )

                    # =============================================
                    # 3. 진행 가능한 gap이 없음
                    #    → Dead-end candidate
                    # =============================================

                    if (
                        target_gap_angle
                        is None
                    ):

                        stop_anchor(
                            anchor
                        )

                        if (
                            substep_index
                            == 0
                        ):

                            dead_end_stable_count += (
                                1
                            )

                        if (
                            dead_end_stable_count
                            >=
                            DEAD_END_CONFIRM_FRAMES
                        ):

                            backtrack_required_width = (
                                estimate_current_corridor_width(
                                    observation.lidar_scan
                                )
                            )

                            backtrack_seed_id = (
                                None
                            )

                            backtrack_trigger_reason = (
                                "DEAD_END"
                            )
                            backtrack_event_valid = (
                                True
                            )

                            anchor_motion_mode = (
                                "BACKTRACK_WAIT_SHEPHERD"
                            )

                            print(
                                "[BacktrackTrigger] "
                                f"reason=DEAD_END "
                                f"active_branch="
                                f"{active_branch_id} "
                                f"corridor_width="
                                f"{backtrack_required_width:.2f}"
                            )

                    # =============================================
                    # 4. 진행 가능한 gap 존재
                    #    → dead-end 아님
                    #    → LiDAR 기반 독립 탐색
                    # =============================================

                    else:

                        dead_end_stable_count = (
                            0
                        )

                        # =====================================================
                        # Swarm의 가장 앞선 NORMAL을 찾고,
                        # 그 robot보다 일정 거리 앞을 유지하도록
                        # Anchor의 전진 속도를 결정한다.
                        # =====================================================

                        (
                            anchor_front_speed_cap,
                            front_robot_id,
                            front_gap,
                            front_robot_speed,
                        ) = (
                            compute_branch_explore_anchor_speed_cap(
                                observation,
                                robots_by_id,
                                KNOWN_CORRIDOR_WIDTH,
                            )
                        )

                        if (
                            front_robot_id
                            is None
                        ):

                            role_debug(
                                "branch-anchor-front-follow",
                                (
                                    "[BranchAnchorFrontFollow] "
                                    f"branch={active_branch_id} "
                                    "front_robot=None "
                                    "anchor_speed_cap=0.000 "
                                    "action=WAIT_FOR_SWARM"
                                ),
                            )

                        else:

                            target_gap = max(
                                3.0
                                * environment.ROBOT_RADIUS,
                                ANCHOR_FRONT_TARGET_GAP_ROWS
                                * environment.GRID_SPACING,
                            )

                            role_debug(
                                "branch-anchor-front-follow",
                                (
                                    "[BranchAnchorFrontFollow] "
                                    f"branch={active_branch_id} "
                                    f"front_robot="
                                    f"{front_robot_id} "
                                    f"front_gap="
                                    f"{front_gap:.3f} "
                                    f"target_gap="
                                    f"{target_gap:.3f} "
                                    f"front_speed="
                                    f"{front_robot_speed:.3f} "
                                    f"anchor_speed_cap="
                                    f"{anchor_front_speed_cap:.3f}"
                                ),
                            )

                        # =====================================================
                        # 기존 LiDAR free-gap navigation은 그대로 사용.
                        #
                        # 단, 실제 전진 속도만 swarm front 상태에 따라 제한.
                        # =====================================================

                        (
                            anchor_yaw_deg,
                            explore_delta,
                        ) = (
                            move_anchor_through_free_gap(
                                anchor,
                                observation.lidar_scan,
                                anchor_yaw_deg,
                                target_gap_angle,
                                substep_dt,
                                max_forward_speed=
                                anchor_front_speed_cap,
                            )
                        )

                        branch_explore_traveled += (
                            explore_delta.length()
                        )

            elif (
                anchor_motion_mode
                == "BACKTRACK_WAIT_SHEPHERD"
            ):

                # =============================================
                # Shepherd 모집 동안 Anchor 정지
                # =============================================

                stop_anchor(
                    anchor
                )

                # =============================================
                # Marker / Dead-end가 실제 발생한 경우만 허용
                # =============================================

                if (
                    not backtrack_event_valid
                ):

                    print(
                        "[BacktrackBlocked] "
                        "reason=NO_VALID_TERMINAL_EVENT"
                    )

                    backtrack_seed_id = (
                        None
                    )

                    backtrack_formation_ids.clear()

                    backtrack_chain_order.clear()

                    backtrack_formation_stable_count = (
                        0
                    )

                    anchor_motion_mode = (
                        "BRANCH_EXPLORE"
                    )

                else:

                    if (
                        active_branch_id
                        is None
                    ):

                        raise RuntimeError(
                            "BACKTRACK_WAIT_SHEPHERD "
                            "without active branch."
                        )

                    #=====================================================                  
                    # Backtracking Shepherd recruitment
                    #
                    # Terminal event:
                    #   DEAD_END or MARKER
                    #
                    # Hop 0:
                    #   Anchor에 가장 가까운 NORMAL seed
                    #
                    # Hop 1:
                    #   seed와 직접 COMM_RANGE 이웃
                    #
                    # Hop 2:
                    #   Hop 1과 직접 COMM_RANGE 이웃
                    #
                    # 한 번 capture한 cohort는
                    # Parent Junction 복귀 전까지 절대 재선정하지 않는다.
                    # =====================================================

                    if not backtrack_formation_ids:

                        # =============================================
                        # 1. Hop 0 seed 선정
                        # =============================================

                        if backtrack_seed_id is None:

                            backtrack_seed_id = (
                                find_backtracking_seed(
                                    observation,
                                    robots_by_id,
                                )
                            )

                        # 아직 Anchor 근처에 seed가 될 NORMAL이 없음.
                        # Anchor는 정지하고 NORMAL swarm은 SPH로 계속 이동.
                        if backtrack_seed_id is None:

                            role_debug(
                                "backtrack-seed-wait",
                                (
                                    "[BacktrackingSeedWait] "
                                    f"branch={active_branch_id} "
                                    f"reason={backtrack_trigger_reason} "
                                    "seed=None"
                                ),
                            )

                        else:

                            # =============================================
                            # 2. Seed 기준 Hop 0 / 1 / 2 수집
                            # =============================================

                            (
                                cohort_ids,
                                hop_layers,
                            ) = (
                                collect_backtracking_seed_one_two_hop(
                                    observation,
                                    robots_by_id,
                                    backtrack_seed_id,
                                )
                            )

                            hop_counts = [
                                len(layer)
                                for layer in hop_layers
                            ]

                            role_debug(
                                "backtrack-seed-hop",
                                (
                                    "[BacktrackingSeedHop] "
                                    f"branch={active_branch_id} "
                                    f"seed={backtrack_seed_id} "
                                    f"hop_counts={hop_counts} "
                                    f"cohort={len(cohort_ids)}"
                                ),
                            )

                            # seed가 사라졌거나 더 이상 유효하지 않으면
                            # 다음 frame에 다시 seed를 선정.
                            if not cohort_ids:

                                backtrack_seed_id = None

                            else:

                                # =============================================
                                # 3. Hop 0~2 전체를 Shepherd formation으로 확정
                                #
                                # 여기서 required_count로 잘라내지 않는다.
                                # 0/1/2 hop에 포함된 NORMAL 전체가 Shepherd.
                                # =============================================

                                formation_ready = (
                                    start_backtracking_shepherd_formation(
                                        observation,
                                        set(cohort_ids),
                                        robots_by_id,
                                        backtrack_seed_id,
                                        active_branch_id,
                                        backtrack_event_valid,
                                    )
                                )

                                if formation_ready:

                                    captured_ids = sorted(
                                        cohort_ids
                                    )

                                    # =========================================
                                    # 4. 같은 cohort를 즉시 PUSH mode로 전환
                                    # =========================================

                                    push_ready = (
                                        prepare_backtracking_shepherd_push(
                                            observation,
                                            captured_ids,
                                            robots_by_id,
                                            backtrack_seed_id,
                                            active_branch_id,
                                        )
                                    )

                                    if not push_ready:

                                        # FORM relay 일부만 성공한 비정상 상황이면
                                        # 다시 NORMAL로 복구하고 재시도.
                                        release_shepherd_group(
                                            set(cohort_ids),
                                            robots_by_id,
                                        )

                                        backtrack_seed_id = None

                                    else:

                                        # =====================================
                                        # 5. Shepherd membership LOCK
                                        #
                                        # 이 시점 이후에는
                                        # seed / hop / cohort 재선정 금지.
                                        # =====================================

                                        backtrack_formation_ids = set(
                                            captured_ids
                                        )

                                        # 기존 변수명을 유지하지만
                                        # 더 이상 line order가 아니라
                                        # locked Shepherd cohort ID container.
                                        backtrack_chain_order = list(
                                            captured_ids
                                        )

                                        branch_states[
                                            active_branch_id
                                        ][
                                            "backtrack_shepherd_ids"
                                        ] = set(
                                            captured_ids
                                        )

                                        backtrack_required_count = (
                                            len(captured_ids)
                                        )

                                        # =====================================
                                        # 6. Junction return direction
                                        #
                                        # Terminal event를 바라보고 있던
                                        # Anchor heading의 정확히 반대 방향.
                                        # =====================================

                                        backtrack_push_yaw_deg = (
                                            normalize_angle(
                                                anchor_yaw_deg
                                                + 180.0
                                            )
                                        )

                                        anchor_yaw_deg = (
                                            backtrack_push_yaw_deg
                                        )

                                        backtrack_reverse_stable_count = (
                                            0
                                        )

                                        # =====================================
                                        # 7. 기존 collision / wall recovery 유지
                                        # =====================================

                                        backtrack_wall_detach_resume_mode = (
                                            "PRESSURE_PUSH"
                                        )

                                        anchor_motion_mode = (
                                            "BACKTRACK_WALL_DETACH"
                                        )

                                        print(
                                            "[BacktrackingTopologyCaptured] "
                                            f"branch={active_branch_id} "
                                            f"reason={backtrack_trigger_reason} "
                                            f"seed={backtrack_seed_id} "
                                            f"hop_counts={hop_counts} "
                                            f"captured={len(captured_ids)} "
                                            f"ids={captured_ids} "
                                            f"return_yaw="
                                            f"{backtrack_push_yaw_deg:.2f} "
                                            "topology_locked=True "
                                            "reformation=False"
                                        )

            elif (
                anchor_motion_mode
                == "BACKTRACK_WALL_DETACH"
            ):

                if (
                    backtrack_push_yaw_deg
                    is None
                ):
                    raise RuntimeError(
                        "BACKTRACK_WALL_DETACH "
                        "without return heading."
                    )

                # detach 중 Anchor는 정지.
                stop_anchor(
                    anchor
                )

                (
                    detach_ready,
                    detach_blocked_ids,
                    detach_direction,
                ) = (
                    detach_backtracking_shepherd_group_step(
                        observation,
                        backtrack_chain_order,
                        robots_by_id,
                        backtrack_push_yaw_deg,
                        substep_dt,
                        backtrack_wall_detach_direction,
                    )
                )

                # detach episode의 첫 방향을 저장.
                # 이후 같은 BACKTRACK_WALL_DETACH 동안 계속 유지.
                if (
                    backtrack_wall_detach_direction
                    is None
                    and
                    detach_direction
                    in (
                        "LEFT",
                        "RIGHT",
                    )
                ):

                    backtrack_wall_detach_direction = (
                        detach_direction
                    )

                if detach_ready:

                    next_mode = (
                        backtrack_wall_detach_resume_mode
                    )

                    # detach episode 종료.
                    # 다음 wall contact에서는 다시 방향을 새로 선택할 수 있다.
                    backtrack_wall_detach_direction = (
                        None
                    )

                    anchor_motion_mode = (
                        next_mode
                    )

                    print(
                        "[BacktrackWallDetachComplete] "
                        f"branch={active_branch_id} "
                        f"count="
                        f"{len(backtrack_formation_ids)} "
                        f"resume={next_mode} "
                        "forward_ready=True"
                    )

                else:

                    role_debug(
                        "backtrack-wall-detach-progress",
                        (
                            "[BacktrackWallDetachProgress] "
                            f"branch={active_branch_id} "
                            "blocked_ids="
                            f"{sorted(detach_blocked_ids)} "
                            f"direction="
                            f"{detach_direction}"
                        ),
                    )


            elif (
                anchor_motion_mode
                == "PRESSURE_PUSH"
            ):

                # =============================================
                # Shepherd capture topology가 Junction 방향으로 이동
                #
                # NORMAL에게 별도 backtracking force는 주지 않음.
                # NORMAL은 계속 SPH만 사용.
                # =============================================

                if (
                    backtrack_push_yaw_deg
                    is None
                ):
                    raise RuntimeError(
                        "PRESSURE_PUSH without backtrack push heading."
                    )

                # =============================================
                # forward PUSH 전에 wall blocker 확인
                # =============================================

                pressure_wall_blockers = (
                    backtracking_forward_wall_blockers(
                        backtrack_chain_order,
                        robots_by_id,
                        backtrack_push_yaw_deg,
                        substep_dt,
                    )
                )

                if pressure_wall_blockers:

                    # 벽에 걸렸으면 현재 PUSH 중지
                    stop_anchor(
                        anchor
                    )

                    # detach 완료 후 다시 PRESSURE_PUSH로 복귀
                    backtrack_wall_detach_resume_mode = (
                        "PRESSURE_PUSH"
                    )



                    backtrack_wall_detach_direction = (
                        None
                    )

                    anchor_motion_mode = (
                        "BACKTRACK_WALL_DETACH"
                    )

                    print(
                        "[PressurePushWallRecovery] "
                        f"branch={active_branch_id} "
                        "blocked_ids="
                        f"{sorted(pressure_wall_blockers)} "
                        "resume=PRESSURE_PUSH"
                    )

                else:

                    # 벽 문제가 없을 때만 실제 Pressure Push 실행
                    backtrack_contact_accelerations = (
                        update_pressure_push(
                            observation,
                            backtrack_chain_order,
                            robots_by_id,
                            backtrack_push_yaw_deg,
                            substep_dt,
                        )
                    )

                    # Anchor도 Shepherd와 함께 return corridor 이동
                    follow_backtrack_corridor_with_comm_guard(
                        anchor,
                        robots,
                        anchor_lidar,
                        set(backtrack_formation_ids),
                        anchor_yaw_deg,
                        substep_dt,
                    )


            elif (
                anchor_motion_mode
                == "FLOW_BACKTRACK"
            ):

                if (
                    backtrack_push_yaw_deg
                    is None
                ):
                    raise RuntimeError(
                        "FLOW_BACKTRACK without "
                        "backtrack push heading."
                    )

                # =============================================
                # 2. Parent Junction entrance 검출
                #
                # 처음 Base -> Root에서 사용했던 것과 같은
                # local ±90° LiDAR lateral range jump 사용.
                # =============================================

                return_entrance_reached = (
                    False
                )

                return_left = (
                    0.0
                )

                return_right = (
                    0.0
                )

                return_delta_left = (
                    0.0
                )

                return_delta_right = (
                    0.0
                )

                # frame마다 한 번만 baseline / detector 갱신.
                if (
                    substep_index
                    == 0
                ):

                    (
                        return_entrance_reached,
                        return_left,
                        return_right,
                        return_delta_left,
                        return_delta_right,
                    ) = (
                        update_junction_entrance_detector(
                            anchor_lidar
                        )
                    )

                    print(
                        "[ReturnJunctionProbe] "
                        f"branch={active_branch_id} "
                        f"L={return_left:.2f} "
                        f"R={return_right:.2f} "
                        f"dL={return_delta_left:.2f} "
                        f"dR={return_delta_right:.2f}"
                    )

                # =============================================
                # 3. Parent Junction entrance 도착
                # =============================================

                if (
                    return_entrance_reached
                ):

                    stop_anchor(
                        anchor
                    )

                    return_junction_entrance_reached = (
                        True
                    )
                    # =============================================
                    # Return corridor에서 얻은 wall baseline을 사용해
                    # stationary Junction threshold를 새로 고정.
                    # =============================================

                    freeze_stationary_threshold(
                        anchor_lidar
                    )

                    return_left_wall = (
                        anchor_lidar
                        .lateral_baseline_left
                    )

                    return_right_wall = (
                        anchor_lidar
                        .lateral_baseline_right
                    )

                    if (
                        return_left_wall is None
                        or
                        return_right_wall is None
                    ):

                        raise RuntimeError(
                            "Missing return corridor wall calibration."
                        )

                    # 현재 Anchor가 멈춘 위치를 local origin으로 놓고
                    # 들어온 Branch의 mouth 좌우 끝점 저장.
                    return_base_entrance_corners_local = (
                        pygame.Vector2(
                            0.0,
                            -return_left_wall,
                        ),
                        pygame.Vector2(
                            0.0,
                            return_right_wall,
                        ),
                    )

                    return_center_target_samples.clear()

                    return_locked_center_target_local = (
                        None
                    )

                    return_center_traveled_local.update(
                        0.0,
                        0.0,
                    )

                    return_center_stable_count = (
                        0
                    )

                    anchor_motion_mode = (
                        "RETURN_JUNCTION_ENTRANCE"
                    )

                    print(
                        "[ParentJunctionEntranceReached] "
                        f"branch="
                        f"{active_branch_id} "
                        f"L={return_left:.2f} "
                        f"R={return_right:.2f} "
                        f"dL={return_delta_left:.2f} "
                        f"dR={return_delta_right:.2f}"
                    )

                # =============================================
                # 3. Parent Junction은 아직 아님.
                #    현재 return corridor의 corner 여부 확인.
                # =============================================

                else:

                    backtrack_corner_angle = (
                        detect_backtrack_corner(
                            observation.lidar_scan
                        )
                    )

                    # =========================================
                    # CORNER DETECTED
                    # =========================================

                    if (
                        backtrack_corner_angle
                        is not None
                    ):

                        # =====================================================
                        # 코너에서도 기존 Shepherd를 절대 release하지 않는다.
                        #
                        # 동일 ID / 동일 topology 유지.
                        # Anchor의 motion을 그대로 복사하면서 코너링한다.
                        # =====================================================

                        stop_anchor(
                            anchor
                        )

                        backtrack_corner_exit_stable_count = (
                            0
                        )

                        backtrack_corner_count += (
                            1
                        )

                        anchor_motion_mode = (
                            "BACKTRACK_CORNER_TURN"
                        )

                        print(
                            "[BacktrackCornerDetected] "
                            f"branch={active_branch_id} "
                            f"corner_angle="
                            f"{backtrack_corner_angle:.2f} "
                            f"shepherd_count="
                            f"{len(backtrack_formation_ids)} "
                            "release=False "
                            "same_cohort=True"
                        )
                    # =========================================
                    # STRAIGHT CORRIDOR
                    # Shepherd가 계속 밀고 Anchor는 LiDAR로
                    # Junction 쪽 corridor를 따라간다.
                    # =========================================

                    else:

                        # =========================================
                        # STRAIGHT CORRIDOR
                        #
                        # FLOW_BACKTRACK 중에도 Shepherd가
                        # wall에 다시 걸릴 수 있으므로
                        # forward PUSH 전에 wall blocker 확인.
                        # =========================================

                        flow_wall_blockers = (
                            backtracking_forward_wall_blockers(
                                backtrack_chain_order,
                                robots_by_id,
                                backtrack_push_yaw_deg,
                                substep_dt,
                            )
                        )

                        if flow_wall_blockers:

                            # 현재 return 이동 중지
                            stop_anchor(
                                anchor
                            )

                            backtrack_wall_detach_resume_mode = (
                                "FLOW_BACKTRACK"
                            )

                            # 새로운 detach episode이므로
                            # 방향 lock 초기화
                            backtrack_wall_detach_direction = (
                                None
                            )

                            anchor_motion_mode = (
                                "BACKTRACK_WALL_DETACH"
                            )

                            print(
                                "[FlowBacktrackWallRecovery] "
                                f"branch={active_branch_id} "
                                "blocked_ids="
                                f"{sorted(flow_wall_blockers)} "
                                "resume=FLOW_BACKTRACK"
                            )

                        else:

                            # 벽 문제가 없으면 기존 physical push 계속
                            backtrack_contact_accelerations = (
                                update_pressure_push(
                                    observation,
                                    backtrack_chain_order,
                                    robots_by_id,
                                    backtrack_push_yaw_deg,
                                    substep_dt,
                                )
                            )

                            # Anchor도 Shepherd와 함께 Junction 방향 이동
                            follow_backtrack_corridor_with_comm_guard(
                                anchor,
                                robots,
                                anchor_lidar,
                                set(backtrack_formation_ids),
                                anchor_yaw_deg,
                                substep_dt,
                            )

            elif (
                anchor_motion_mode
                == "BACKTRACK_CORNER_TURN"
            ):

                # =====================================================
                # SAME Shepherd cohort가 Anchor의 코너링을 그대로 복사.
                #
                # release 없음
                # re-recruit 없음
                # ID 변경 없음
                # =====================================================

                target_gap_angle = (
                    find_branch_free_gap(
                        observation.lidar_scan
                    )
                )

                if (
                    target_gap_angle
                    is None
                ):

                    stop_anchor(
                        anchor
                    )

                    backtrack_corner_exit_stable_count = (
                        0
                    )

                    role_debug(
                        "backtrack-corner-gap-wait",
                        (
                            "[BacktrackCornerTurnWait] "
                            f"branch={active_branch_id} "
                            "reason=NO_FREE_GAP"
                        ),
                    )

                else:

                    (
                        new_anchor_yaw,
                        _corner_delta,
                        corner_motion_ok,
                    ) = (
                        move_anchor_with_backtracking_shepherds(
                            anchor,
                            observation.lidar_scan,
                            set(
                                backtrack_formation_ids
                            ),
                            robots_by_id,
                            anchor_yaw_deg,
                            target_gap_angle,
                            substep_dt,
                        )
                    )

                    if corner_motion_ok:

                        anchor_yaw_deg = (
                            new_anchor_yaw
                        )

                    # 현재 heading 기준으로
                    # 새로운 straight corridor에 정렬됐는지 확인.
                    corridor_reacquired = (
                        backtrack_corridor_reacquired(
                            observation.lidar_scan,
                            target_gap_angle,
                        )
                    )

                    if (
                        substep_index
                        == 0
                    ):

                        if (
                            corridor_reacquired
                            and
                            corner_motion_ok
                        ):

                            backtrack_corner_exit_stable_count += (
                                1
                            )

                        else:

                            backtrack_corner_exit_stable_count = (
                                0
                            )

                        print(
                            "[BacktrackCornerTurn] "
                            f"branch={active_branch_id} "
                            f"corner_index="
                            f"{backtrack_corner_count} "
                            f"gap_angle="
                            f"{target_gap_angle:.2f} "
                            f"yaw="
                            f"{anchor_yaw_deg:.2f} "
                            f"aligned="
                            f"{corridor_reacquired} "
                            f"motion_ok="
                            f"{corner_motion_ok} "
                            f"stable="
                            f"{backtrack_corner_exit_stable_count}/"
                            f"{BACKTRACK_CORNER_EXIT_STABLE_SCANS} "
                            f"same_shepherds="
                            f"{len(backtrack_formation_ids)}"
                        )

                    # =================================================
                    # 코너 통과 완료
                    #
                    # 기존 Shepherd 그대로.
                    # 새 heading만 다음 PUSH 방향으로 갱신.
                    # =================================================

                    if (
                        backtrack_corner_exit_stable_count
                        >=
                        BACKTRACK_CORNER_EXIT_STABLE_SCANS
                    ):

                        # 코너 이후 새로운 Junction 복귀 방향
                        backtrack_push_yaw_deg = (
                            anchor_yaw_deg
                        )

                        backtrack_corner_exit_stable_count = (
                            0
                        )

                        # 새로운 corridor 기준으로
                        # Junction entrance detector를 다시 학습.
                        reset_junction_entrance_detector(
                            anchor_lidar
                        )

                        # 핵심:
                        # BACKTRACK_WAIT_SHEPHERD로 가지 않는다.
                        #
                        # 기존 Shepherd 그대로 FLOW_BACKTRACK 재개.
                        anchor_motion_mode = (
                            "FLOW_BACKTRACK"
                        )

                        print(
                            "[BacktrackCornerExit] "
                            f"branch={active_branch_id} "
                            f"corner_index="
                            f"{backtrack_corner_count} "
                            f"new_return_yaw="
                            f"{backtrack_push_yaw_deg:.2f} "
                            f"same_shepherd_count="
                            f"{len(backtrack_formation_ids)} "
                            "recruit=False"
                        )


            elif (
                anchor_motion_mode
                == "RETURN_JUNCTION_ENTRANCE"
            ):

                # Anchor는 Junction 입구에서 정지.
                stop_anchor(
                    anchor
                )

                if (
                    backtrack_push_yaw_deg
                    is None
                ):
                    raise RuntimeError(
                        "RETURN_JUNCTION_ENTRANCE "
                        "without push heading."
                    )

                # Backtracking swarm의 물리적 reverse flow는
                # stationary verification 동안에도 유지.
                backtrack_contact_accelerations = (
                    update_pressure_push(
                        observation,
                        backtrack_chain_order,
                        robots_by_id,
                        backtrack_push_yaw_deg,
                        substep_dt,
                    )
                )

                # =============================================
                # Parent Junction stationary confirmation
                # =============================================

                if (
                    substep_index
                    == 0
                    and
                    update_stationary_junction_confirmation(
                        anchor_lidar
                    )
                ):

                    return_center_target_samples.clear()

                    anchor_motion_mode = (
                        "RETURN_CENTER_ESTIMATE"
                    )

                    print(
                        "[ReturnJunctionConfirmed] "
                        f"branch="
                        f"{active_branch_id}"
                    )


            elif (
                anchor_motion_mode
                == "RETURN_CENTER_ESTIMATE"
            ):

                stop_anchor(
                    anchor
                )

                if (
                    backtrack_push_yaw_deg
                    is None
                ):
                    raise RuntimeError(
                        "RETURN_CENTER_ESTIMATE "
                        "without push heading."
                    )

                backtrack_contact_accelerations = (
                    update_pressure_push(
                        observation,
                        backtrack_chain_order,
                        robots_by_id,
                        backtrack_push_yaw_deg,
                        substep_dt,
                    )
                )

                if (
                    substep_index
                    == 0
                    and
                    return_base_entrance_corners_local
                    is not None
                ):

                    (
                        return_base_left,
                        return_base_right,
                    ) = (
                        return_base_entrance_corners_local
                    )

                    # 현재 진입 방향 기준 반대편 두 corner 검출
                    front_corners = (
                        extract_front_mouth_corners(
                            anchor_lidar
                        )
                    )

                    if (
                        front_corners
                        is None
                    ):

                        candidate = (
                            None
                        )

                    else:

                        (
                            return_front_left,
                            return_front_right,
                        ) = (
                            front_corners
                        )

                        candidate = (
                            estimate_local_junction_center(
                                return_base_left,
                                return_base_right,
                                return_front_left,
                                return_front_right,
                            )
                        )

                    if (
                        candidate
                        is None
                    ):

                        return_center_target_samples.clear()

                        print(
                            "[ReturnCenterScan] "
                            "center=NOT_AVAILABLE"
                        )

                    else:

                        if (
                            return_center_target_samples
                        ):

                            previous = (
                                return_center_target_samples[
                                    -1
                                ]
                            )

                            if (
                                candidate
                                - previous
                            ).length() > (
                                ANCHOR_TARGET_SAMPLE_TOLERANCE
                            ):

                                return_center_target_samples.clear()

                        return_center_target_samples.append(
                            candidate.copy()
                        )

                        print(
                            "[ReturnCenterScan] "
                            f"candidate="
                            f"({candidate.x:.2f},"
                            f"{candidate.y:.2f}) "
                            f"samples="
                            f"{len(return_center_target_samples)}/"
                            f"{ANCHOR_ENTRANCE_STATIONARY_SCANS}"
                        )

                        if (
                            len(
                                return_center_target_samples
                            )
                            >=
                            ANCHOR_ENTRANCE_STATIONARY_SCANS
                        ):

                            total = (
                                pygame.Vector2()
                            )

                            for sample in (
                                return_center_target_samples
                            ):

                                total += sample

                            return_locked_center_target_local = (
                                total
                                / len(
                                    return_center_target_samples
                                )
                            )

                            return_center_traveled_local.update(
                                0.0,
                                0.0,
                            )

                            return_center_stable_count = (
                                0
                            )

                            anchor_motion_mode = (
                                "RETURN_CENTER_TRANSIT"
                            )

                            print(
                                "[ReturnCenterTargetLocked] "
                                f"target=("
                                f"{return_locked_center_target_local.x:.2f},"
                                f"{return_locked_center_target_local.y:.2f})"
                            )




            elif (
                anchor_motion_mode
                == "RETURN_CENTER_TRANSIT"
            ):

                if (
                    return_locked_center_target_local
                    is None
                ):

                    raise RuntimeError(
                        "RETURN_CENTER_TRANSIT "
                        "without locked center."
                    )

                if (
                    backtrack_push_yaw_deg
                    is None
                ):

                    raise RuntimeError(
                        "RETURN_CENTER_TRANSIT "
                        "without push heading."
                    )

                # Shepherd reverse flow 유지
                backtrack_contact_accelerations = (
                    update_pressure_push(
                        observation,
                        backtrack_chain_order,
                        robots_by_id,
                        backtrack_push_yaw_deg,
                        substep_dt,
                    )
                )

                (
                    return_centered_now,
                    return_local_delta,
                ) = (
                    move_anchor_to_locked_center(
                        anchor,
                        return_locked_center_target_local,
                        return_center_traveled_local,
                        anchor_yaw_deg,
                        substep_dt,
                    )
                )

                return_center_traveled_local += (
                    return_local_delta
                )

                if (
                    return_centered_now
                ):

                    if (
                        substep_index
                        == 0
                    ):

                        return_center_stable_count += (
                            1
                        )

                else:

                    return_center_stable_count = (
                        0
                    )

                if (
                    return_center_stable_count
                    >=
                    ANCHOR_CENTER_STABLE_SCANS
                ):

                    stop_anchor(
                        anchor
                    )

                    anchor_motion_mode = (
                        "RETURN_JUNCTION_CENTERED"
                    )

                    print(
                        "[ReturnJunctionCentered] "
                        f"branch="
                        f"{active_branch_id} "
                        f"target=("
                        f"{return_locked_center_target_local.x:.2f},"
                        f"{return_locked_center_target_local.y:.2f}) "
                        f"travel=("
                        f"{return_center_traveled_local.x:.2f},"
                        f"{return_center_traveled_local.y:.2f})"
                    )

            elif (
                anchor_motion_mode
                == "RETURN_JUNCTION_CENTERED"
            ):

                stop_anchor(
                    anchor
                )

                if (
                    junction_reference_yaw_deg
                    is None
                ):

                    raise RuntimeError(
                        "RETURN_JUNCTION_CENTERED "
                        "without Junction reference heading."
                    )

                # 처음 Branch geometry를 등록했던
                # Junction-local 기준 heading으로 복귀.
                anchor_yaw_deg = (
                    junction_reference_yaw_deg
                )

                swarm_return_stable_count = (
                    0
                )

                anchor_motion_mode = (
                    "WAIT_SWARM_RETURN"
                )

                print(
                    "[WaitSwarmReturnStart] "
                    f"branch="
                    f"{active_branch_id}"
                )

            elif (
                anchor_motion_mode
                == "WAIT_SWARM_RETURN"
            ):

                stop_anchor(
                    anchor
                )

                if (
                    active_branch_id
                    is None
                ):

                    raise RuntimeError(
                        "WAIT_SWARM_RETURN "
                        "without active branch."
                    )

                if (
                    backtrack_push_yaw_deg
                    is None
                ):

                    raise RuntimeError(
                        "WAIT_SWARM_RETURN "
                        "without push heading."
                    )

                active_branch = next(
                    branch
                    for branch
                    in confirmed_branches
                    if (
                        branch["id"]
                        == active_branch_id
                    )
                )

                # 아직 Branch 내부에 남아 있는 NORMAL을
                # Junction 쪽으로 계속 밀기.
                backtrack_contact_accelerations = (
                    update_pressure_push(
                        observation,
                        backtrack_chain_order,
                        robots_by_id,
                        backtrack_push_yaw_deg,
                        substep_dt,
                    )
                )

                if (
                    substep_index
                    == 0
                ):

                    (
                        swarm_return_ready,
                        remaining_normals,
                        remaining_backtrack,
                    ) = (
                        branch_swarm_return_complete(
                            observation,
                            active_branch,
                            robots_by_id,
                            set(
                                backtrack_formation_ids
                            ),
                        )
                    )

                    if (
                        swarm_return_ready
                    ):

                        swarm_return_stable_count += (
                            1
                        )

                    else:

                        swarm_return_stable_count = (
                            0
                        )

                    print(
                        "[SwarmReturnCheck] "
                        f"branch="
                        f"{active_branch_id} "
                        f"remaining_normal="
                        f"{len(remaining_normals)} "
                        f"remaining_backtrack="
                        f"{len(remaining_backtrack)} "
                        f"stable="
                        f"{swarm_return_stable_count}/"
                        f"{SWARM_RETURN_STABLE_SCANS}"
                    )

                    if (
                        swarm_return_stable_count
                        >=
                        SWARM_RETURN_STABLE_SCANS
                    ):

                        anchor_motion_mode = (
                            "FINALIZE_BRANCH_RETURN"
                        )

                        print(
                            "[SwarmReturnComplete] "
                            f"branch="
                            f"{active_branch_id}"
                        )

            elif (
                anchor_motion_mode
                == "FINALIZE_BRANCH_RETURN"
            ):

                stop_anchor(
                    anchor
                )

                if (
                    active_branch_id
                    is None
                ):

                    raise RuntimeError(
                        "FINALIZE_BRANCH_RETURN "
                        "without active branch."
                    )

                # =============================================
                # 1. Backtracking Shepherd → NORMAL
                # =============================================

                release_shepherd_group(
                    set(
                        backtrack_formation_ids
                    ),
                    robots_by_id,
                )

                branch_states[
                    active_branch_id
                ][
                    "backtrack_shepherd_ids"
                ].clear()

                backtrack_formation_ids.clear()
                backtrack_chain_order.clear()

                # =============================================
                # 2. 이제서야 현재 Branch VISITED 확정
                # =============================================

                mark_branch_visited(
                    active_branch_id,
                    branch_states,
                    reason=
                    "PHYSICAL_RETURN_COMPLETE",
                )

                print(
                    "[PhysicalReturnComplete] "
                    f"branch="
                    f"{active_branch_id}"
                )

                # =============================================
                # 3. 탐색하면서 열었던 Branch를
                #    다시 Shepherd formation 가능 상태로 만듦
                # =============================================

                branch_states[
                    active_branch_id
                ][
                    "initial_shepherd_ids"
                ].clear()

                

                branch_states[
                    active_branch_id
                ][
                    "initial_sealed"
                ] = False

                branch_states[
                    active_branch_id
                ][
                    "opened"
                ] = False

                # Backtracking runtime reset
                backtrack_event_valid = (
                    False
                )

                backtrack_trigger_reason = (
                    None
                )

                backtrack_push_yaw_deg = (
                    None
                )

                backtrack_seed_id = (
                    None
                )

                backtrack_formation_stable_count = (
                    0
                )

                backtrack_reverse_stable_count = (
                    0
                )

                anchor_motion_mode = (
                    "REFORM_VISITED_BRANCH_SHEPHERD"
                )

            elif (
                anchor_motion_mode
                == "REFORM_VISITED_BRANCH_SHEPHERD"
            ):

                stop_anchor(
                    anchor
                )

                if (
                    active_branch_id
                    is None
                ):

                    raise RuntimeError(
                        "REFORM_VISITED_BRANCH_SHEPHERD "
                        "without branch."
                    )

                active_branch = next(
                    branch
                    for branch
                    in confirmed_branches
                    if (
                        branch["id"]
                        == active_branch_id
                    )
                )

                resealed = (
                    form_initial_junction_shepherd_boundaries(
                        observation,
                        robots_by_id,
                        [
                            active_branch
                        ],
                        branch_states,
                    )
                )

                if (
                    resealed
                ):

                    print(
                        "[VisitedBranchResealed] "
                        f"branch="
                        f"{active_branch_id}"
                    )

                    anchor_motion_mode = (
                        "SELECT_NEXT_BRANCH"
                    )

            elif (
                anchor_motion_mode
                == "SELECT_NEXT_BRANCH"
            ):

                stop_anchor(
                    anchor
                )

                next_branch_id = (
                    find_next_unvisited_branch(
                        branch_order,
                        branch_states,
                    )
                )

                if (
                    next_branch_id
                    is None
                ):

                    anchor_motion_mode = (
                        "ROOT_COMPLETE"
                    )

                    print(
                        "[RootDFSComplete]"
                    )

                else:

                    active_branch_id = (
                        next_branch_id
                    )

                    select_runtime_branch(
                        active_branch_id,
                        branch_states,
                    )

                    open_selected_branch(
                        active_branch_id,
                        branch_states,
                        robots_by_id,
                    )

                    active_branch = next(
                        branch
                        for branch
                        in confirmed_branches
                        if (
                            branch["id"]
                            == active_branch_id
                        )
                    )

                    if (
                        junction_reference_yaw_deg
                        is None
                    ):

                        raise RuntimeError(
                            "Missing Junction reference heading."
                        )

                    relative_turn_deg = (
                        active_branch[
                            "center_angle"
                        ]
                    )

                    anchor_yaw_deg = (
                        normalize_angle(
                            junction_reference_yaw_deg
                            + relative_turn_deg
                        )
                    )

                    anchor_branch_entry_target = (
                        active_branch[
                            "entrance_midpoint"
                        ].length()
                        + ANCHOR_BRANCH_ENTRY_MARGIN
                    )

                    anchor_branch_entry_traveled = (
                        0.0
                    )

                    branch_explore_traveled = (
                        0.0
                    )

                    # =============================================
                    # New Branch runtime reset
                    # =============================================

                    origin_marker_cleared = (
                        False
                    )

                    detected_marker_id = (
                        None
                    )

                    dead_end_stable_count = (
                        0
                    )

                    backtrack_event_valid = (
                        False
                    )

                    backtrack_trigger_reason = (
                        None
                    )

                    backtrack_seed_id = (
                        None
                    )

                    backtrack_push_yaw_deg = (
                        None
                    )

                    backtrack_required_count = (
                        0
                    )

                    backtrack_formation_ids.clear()

                    backtrack_chain_order.clear()

                    backtrack_formation_stable_count = (
                        0
                    )

                    backtrack_reverse_stable_count = (
                        0
                    )

                    swarm_return_stable_count = (
                        0
                    )

                    return_junction_entrance_reached = (
                        False
                    )

                    return_base_entrance_corners_local = (
                        None
                    )

                    return_center_target_samples.clear()

                    return_locked_center_target_local = (
                        None
                    )

                    return_center_traveled_local.update(
                        0.0,
                        0.0,
                    )

                    return_center_stable_count = (
                        0
                    )

                    anchor_motion_mode = (
                        "BRANCH_ENTRY"
                    )

                    print(
                        "[AnchorBranchStart] "
                        f"branch="
                        f"{active_branch_id} "
                        f"relative_turn="
                        f"{relative_turn_deg:.2f} "
                        f"runtime_yaw="
                        f"{anchor_yaw_deg:.2f} "
                        f"entry_target="
                        f"{anchor_branch_entry_target:.2f}"
                    )

            elif (
                anchor_motion_mode
                == "ROOT_COMPLETE"
            ):

                stop_anchor(
                    anchor
                )

                if (
                    junction_reference_yaw_deg
                    is None
                ):

                    raise RuntimeError(
                        "ROOT_COMPLETE "
                        "without Junction reference heading."
                    )

                anchor_yaw_deg = (
                    junction_reference_yaw_deg
                )

                anchor_motion_mode = (
                    "FINAL_RETURN_PREP"
                )

            elif (
                anchor_motion_mode
                == "FINAL_RETURN_PREP"
            ):

                stop_anchor(
                    anchor
                )

                final_push_branch = (
                    find_final_push_branch(
                        confirmed_branches
                    )
                )

                final_push_branch_id = (
                    final_push_branch["id"]
                )

                final_side_branch_ids = {
                    branch["id"]
                    for branch
                    in confirmed_branches

                    if branch["id"]
                    != final_push_branch_id
                }

                final_push_ids = set(
                    branch_states[
                        final_push_branch_id
                    ][
                        "initial_shepherd_ids"
                    ]
                )

                for robot_id in final_push_ids:

                    robot = robots_by_id[
                        robot_id
                    ]

                    robot.role = "SHEPHERD"
                    robot.role_branch = final_push_branch_id
                    robot.role_frozen = False
                    robot.role_frozen_position = None
                    robot.shepherd_mode = "FINAL_PUSH"

                marker_id = (
                    branch_states[
                        final_push_branch_id
                    ][
                        "marker_id"
                    ]
                )

                if (
                    marker_id
                    is not None
                ):

                    release_robot_to_normal(
                        robots_by_id[
                            marker_id
                        ]
                    )

                    print(
                        "[FinalMarkerReleased] "
                        f"branch={final_push_branch_id} "
                        f"marker_id={marker_id}"
                    )

                for branch_id in final_side_branch_ids:

                    state = branch_states[
                        branch_id
                    ]

                    for robot_id in state[
                        "initial_shepherd_ids"
                    ]:

                        release_robot_to_normal(
                            robots_by_id[
                                robot_id
                            ]
                        )

                    marker_id = state[
                        "marker_id"
                    ]

                    if marker_id is not None:

                        release_robot_to_normal(
                            robots_by_id[
                                marker_id
                            ]
                        )

                final_side_merge_stable_count = 0

                anchor_motion_mode = (
                    "FINAL_WAIT_SIDE_MERGE"
                )

            elif (
                anchor_motion_mode
                == "FINAL_WAIT_SIDE_MERGE"
            ):

                stop_anchor(
                    anchor
                )

                if (
                    substep_index
                    == 0
                ):

                    side_merge_ready = True

                    for branch_id in final_side_branch_ids:

                        side_branch = next(
                            branch
                            for branch
                            in confirmed_branches
                            if branch["id"] == branch_id
                        )

                        (
                            _side_return_ready,
                            remaining_normals,
                            _remaining_backtrack,
                        ) = branch_swarm_return_complete(
                            observation,
                            side_branch,
                            robots_by_id,
                            set(),
                        )

                        if remaining_normals:
                            side_merge_ready = False

                    if side_merge_ready:

                        final_side_merge_stable_count += 1

                    else:

                        final_side_merge_stable_count = 0

                    if (
                        final_side_merge_stable_count
                        >= FINAL_SIDE_MERGE_STABLE_SCANS
                    ):

                        final_base_return_stable_count = 0

                        anchor_motion_mode = (
                            "FINAL_BASE_PUSH"
                        )

            elif (
                anchor_motion_mode
                == "FINAL_BASE_PUSH"
            ):

                stop_anchor(
                    anchor
                )

                if (
                    junction_reference_yaw_deg
                    is None
                ):

                    raise RuntimeError(
                        "FINAL_BASE_PUSH "
                        "without Junction reference heading."
                    )

                update_final_base_push(
                    final_push_ids,
                    robots_by_id,
                    junction_reference_yaw_deg,
                    substep_dt,
                )

                if (
                    substep_index
                    == 0
                ):

                    if swarm_base_return_complete(
                        observation
                    ):

                        final_base_return_stable_count += 1

                    else:

                        final_base_return_stable_count = 0

                    if (
                        final_base_return_stable_count
                        >= FINAL_BASE_RETURN_STABLE_SCANS
                    ):

                        for robot_id in final_push_ids:

                            release_robot_to_normal(
                                robots_by_id[
                                    robot_id
                                ]
                            )

                        anchor_yaw_deg = normalize_angle(
                            junction_reference_yaw_deg
                            + 180.0
                        )

                        final_anchor_return_traveled = 0.0

                        anchor_motion_mode = (
                            "FINAL_ANCHOR_RETURN"
                        )

            elif (
                anchor_motion_mode
                == "FINAL_ANCHOR_RETURN"
            ):

                final_anchor_delta = follow_corridor_locally(
                    anchor,
                    observation,
                    anchor_yaw_deg,
                    substep_dt,
                )

                final_anchor_return_traveled += max(
                    0.0,
                    final_anchor_delta.x,
                )

                if (
                    final_anchor_return_traveled
                    >= ROOT_CENTER_TO_BASE_DISTANCE
                ):

                    stop_anchor(
                        anchor
                    )

                    anchor_motion_mode = (
                        "SYSTEM_COMPLETE"
                    )

            elif (
                anchor_motion_mode
                == "SYSTEM_COMPLETE"
            ):

                stop_anchor(
                    anchor
                )


            # =================================================
            # 6. SPH swarm motion
            # =================================================



            physics_grid = (
                environment.build_physics_grid(
                    swarm_robots
                )
            )

            environment.compute_densities(
                swarm_robots,
                physics_grid,
            )

            environment.compute_pressures(
                swarm_robots,
                reference_density,
            )

            compute_sph_only_forces(
                swarm_robots,
                physics_grid,
                reference_density,
            )

            # =================================================
            # Shepherd -> NORMAL physical contact response
            #
            # NORMAL의 controller는 여전히:
            #
            #     f_SPH = f_press + f_vis
            #
            # 이 항은 goal/backtracking controller가 아니라
            # 실제 robot-to-robot collision/contact response이다.
            # =================================================

            for (
                robot_id,
                contact_acceleration,
            ) in backtrack_contact_accelerations.items():

                robot = (
                    robots_by_id[
                        robot_id
                    ]
                )

                if (
                    robot.role
                    != "NORMAL"
                ):
                    continue

                robot.acceleration = (
                    environment.limit_vector(
                        robot.acceleration
                        + contact_acceleration,
                        environment.MAX_ACCELERATION,
                    )
                )

                robot.filtered_acceleration.update(
                    robot.acceleration
                )

            for robot in (
                swarm_robots
            ):

                # -----------------------------------------
                # Frozen Shepherd / Marker
                # -----------------------------------------

                if getattr(
                    robot,
                    "role_frozen",
                    False,
                ):

                    frozen_position = (
                        robot.role_frozen_position
                    )

                    robot.position.update(
                        frozen_position
                    )

                    robot.previous_position.update(
                        frozen_position
                    )

                    robot.velocity.update(
                        0.0,
                        0.0,
                    )

                    robot.acceleration.update(
                        0.0,
                        0.0,
                    )

                    robot.filtered_acceleration.update(
                        0.0,
                        0.0,
                    )

                    robot.commanded_velocity.update(
                        0.0,
                        0.0,
                    )

                    robot.observed_velocity.update(
                        0.0,
                        0.0,
                    )

                    continue

                if (
                    anchor_motion_mode
                    == "FINAL_BASE_PUSH"
                    and
                    robot.robot_id
                    in final_push_ids
                ):

                    robot.acceleration.update(
                        0.0,
                        0.0,
                    )

                    continue

                # -----------------------------------------
                # Backtracking Shepherd blob
                # -----------------------------------------

                if (
                    anchor_motion_mode
                    in (
                        "PRESSURE_PUSH",
                        "FLOW_BACKTRACK",
                        "RETURN_JUNCTION_ENTRANCE",
                        "RETURN_CENTER_ESTIMATE",
                        "RETURN_CENTER_TRANSIT",
                        "RETURN_JUNCTION_CENTERED",
                        "WAIT_SWARM_RETURN",
                        "FINALIZE_BRANCH_RETURN",
                    )
                    and
                    robot.robot_id
                    in backtrack_formation_ids
                ):

                    robot.acceleration.update(
                        0.0,
                        0.0,
                    )

                    continue

                # -----------------------------------------
                # NORMAL
                # -----------------------------------------

                integrate_sph_only_robot(
                    robot,
                    substep_dt,
                )




            # =================================================
            # 6.5. Pressure Push -> Flow Backtracking
            #
            # NORMAL robot들이 실제로 Junction 방향으로
            # 역류하기 시작했는지 observed velocity로 확인.
            #
            # pressure threshold는 사용하지 않는다.
            # =================================================

            if (
                anchor_motion_mode
                == "PRESSURE_PUSH"
            ):

                # ---------------------------------------------
                # SPH 이동이 끝난 뒤의 실제 velocity가 필요하므로
                # observation을 새로 만든다.
                # ---------------------------------------------

                flow_observation = (
                    LocalObservationBuilder.build(
                        anchor,
                        robots,
                        anchor_lidar,
                        anchor_yaw_deg,
                    )
                )

                (
                    reverse_flow_ready,
                    reverse_eligible_count,
                    reverse_count,
                    reverse_ratio,
                    mean_reverse_speed,
                ) = (
                    evaluate_normal_reverse_flow(
                        flow_observation,
                        robots_by_id,
                        backtrack_chain_order,
                        backtrack_required_width,
                    )
                )

                # ---------------------------------------------
                # stable count는 frame마다 한 번만 갱신
                # ---------------------------------------------

                if (
                    substep_index
                    == 0
                ):

                    if (
                        reverse_flow_ready
                    ):

                        backtrack_reverse_stable_count += (
                            1
                        )

                    else:

                        backtrack_reverse_stable_count = (
                            0
                        )

                    print(
                        "[ReverseFlowCheck] "
                        f"branch="
                        f"{active_branch_id} "
                        f"eligible="
                        f"{reverse_eligible_count} "
                        f"reverse="
                        f"{reverse_count} "
                        f"ratio="
                        f"{reverse_ratio:.3f} "
                        f"mean_reverse_speed="
                        f"{mean_reverse_speed:.3f} "
                        f"ready="
                        f"{reverse_flow_ready} "
                        f"stable="
                        f"{backtrack_reverse_stable_count}/"
                        f"{BACKTRACK_REVERSE_STABLE_SCANS}"
                    )

                # ---------------------------------------------
                # 실제 reverse flow가 연속적으로 확인되면
                # FLOW_BACKTRACK 상태로 전환
                # ---------------------------------------------

                if (
                    backtrack_reverse_stable_count
                    >=
                    BACKTRACK_REVERSE_STABLE_SCANS
                ):

                    if (
                        backtrack_push_yaw_deg
                        is None
                    ):
                        raise RuntimeError(
                            "FLOW_BACKTRACK without "
                            "backtrack push heading."
                        )

                    # B2 corridor를 새로운 lateral baseline으로
                    # 다시 학습하도록 기존 Root 진입 history 제거.
                    reset_junction_entrance_detector(
                        anchor_lidar
                    )

                    return_junction_entrance_reached = (
                        False
                    )

                    # Reverse flow가 확인되면 Anchor와 Shepherd가
                    # 같은 return corridor를 즉시 함께 따라간다.
                    anchor_motion_mode = (
                        "FLOW_BACKTRACK"
                    )

                    print(
                        "[FlowBacktrackStart] "
                        f"branch={active_branch_id} "
                        "anchor_yaw="
                        f"{anchor_yaw_deg:.2f}"
                    )

            # =================================================
            # 7. Initial Shepherd formation
            # =================================================

            if (
                anchor_locally_centered
                and confirmed_branches
                and not initial_shepherds_ready
            ):

                post_move_observation = (
                    LocalObservationBuilder.build(
                        anchor,
                        robots,
                        anchor_lidar,
                        anchor_yaw_deg,
                    )
                )

                initial_shepherds_ready = (
                    form_initial_junction_shepherd_boundaries(
                        post_move_observation,
                        robots_by_id,
                        confirmed_branches,
                        branch_states,
                        junction_message_received_ids,
                    )
                )
                if initial_shepherds_ready:

                    junction_broadcast_active = (
                        False
                    )

                    print(
                        "[AnchorBroadcast] "
                        "command=JUNCTION_CONFIRMED "
                        "active=False "
                        "reason=INITIAL_SHEPHERDS_READY"
                    )

                if initial_shepherds_ready:

                    print(
                        "[AllInitialShepherdBoundariesReady]"
                    )


            # =================================================
            # 8. Initial Marker 생성
            # =================================================

            if (
                initial_shepherds_ready
                and
                not markers_ready
            ):

                marker_observation = (
                    LocalObservationBuilder.build(
                        anchor,
                        robots,
                        anchor_lidar,
                        anchor_yaw_deg,
                    )
                )

                create_initial_branch_markers(
                    marker_observation,
                    confirmed_branches,
                    branch_states,
                    robots_by_id,
                )

                markers_ready = True

                print(
                    "[AllInitialMarkersReady]"
                )


            # =================================================
            # 9. Runtime Branch order 생성
            # =================================================

            if (
                markers_ready
                and
                not branch_order
            ):

                branch_order = [
                    branch["id"]
                    for branch
                    in reversed(
                        confirmed_branches
                    )
                ]

                print(
                    "[RuntimeBranchOrder] "
                    f"order={branch_order}"
                )


            # =================================================
            # 10. Branch exploration cycle 시작
            #
            # 첫 Branch도 이후 Branch와 동일하게
            # SELECT_NEXT_BRANCH를 통해 시작한다.
            # =================================================

            if (
                markers_ready
                and branch_order
                and not first_branch_opened
            ):

                first_branch_opened = (
                    True
                )

                active_branch_id = (
                    None
                )

                anchor_motion_mode = (
                    "SELECT_NEXT_BRANCH"
                )

                stop_anchor(
                    anchor
                )

                print(
                    "[BranchCycleStart] "
                    f"order={branch_order}"
                )


                # 여기까지가 if anchor_locally_centered: 내부


            
        # =====================================================
        # End of 8 physics substeps
        # =====================================================

        # -----------------------------------------------------
        # Fresh scan for rendering
        # -----------------------------------------------------

        anchor_lidar.scan(
            anchor.position,
            anchor_yaw_deg,
        )

        # =====================================================
        # Communication update
        # =====================================================

        if not paused:

            environment.update_communication_system(
                robots,
                environment.build_spatial_grid(
                    robots
                ),
            )

        # =====================================================
        # Terminal runtime trace: Anchor stage / command / roles
        # =====================================================

        if (
            anchor_motion_mode
            != last_logged_anchor_motion_mode
        ):

            normal_count = sum(
                robot.role == "NORMAL"
                for robot in swarm_robots
            )

            shepherd_count = sum(
                robot.role == "SHEPHERD"
                for robot in swarm_robots
            )

            marker_count = sum(
                robot.role == "MARKER"
                for robot in swarm_robots
            )

            print(
                "[AnchorStage] "
                f"t={environment.simulation_time:.2f} "
                f"stage={anchor_motion_mode} "
                f"command={anchor_command_name(anchor_motion_mode)} "
                f"branch={active_branch_id} "
                f"yaw={anchor_yaw_deg:.1f} "
                f"roles=NORMAL:{normal_count} "
                f"SHEPHERD:{shepherd_count} "
                f"MARKER:{marker_count}"
            )

            last_logged_anchor_motion_mode = (
                anchor_motion_mode
            )

        # =====================================================
        # Frozen-role audit
        # =====================================================

        if initial_shepherds_ready:

            frozen_robots = [
                robot
                for robot
                in swarm_robots

                if getattr(
                    robot,
                    "role_frozen",
                    False,
                )
            ]

            drifted = [
                (
                    robot.robot_id,
                    round(
                        robot.position.distance_to(
                            robot.role_frozen_position
                        ),
                        6,
                    ),
                )

                for robot
                in frozen_robots

                if (
                    robot.position.distance_to(
                        robot.role_frozen_position
                    )
                    > 1.0e-6
                )
            ]

            role_debug(
                "freeze-audit",
                (
                    "[FreezeAudit] "
                    f"shepherds="
                    f"{sum(robot.role == 'SHEPHERD' for robot in swarm_robots)} "
                    f"markers="
                    f"{sum(robot.role == 'MARKER' for robot in swarm_robots)} "
                    f"frozen="
                    f"{len(frozen_robots)} "
                    f"drifted="
                    f"{drifted[:10]}"
                ),
            )

        # =====================================================
        # Rendering
        # =====================================================

        screen.fill(
            COLORS["background"]
        )

        draw_map(
            screen,
            font,
        )

        # =====================================================
        # Communication links
        #
        # 로봇보다 먼저 그림.
        # 따라서 link 위에 robot이 다시 그려진다.
        # =====================================================

        if show_comm_links:

            environment.draw_communication_links(
                screen,
                robots,
            )

        draw_robots(
            screen,
            robots,
            color_reference_density,
            anchor,
            anchor_lidar,
            junction_detected,
            anchor_yaw_deg,
        )

        draw_controls(
            screen,
            font,
            paused,
            show_comm_links,
            anchor_motion_mode,
            active_branch_id,
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
