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


# 같은 key의 상태가 이전과 달라졌을 때만 디버그 메시지를 출력하여 중복 로그를 방지
def role_debug(key: str, message: str) -> None:
    """Print role diagnostics only when the state represented by a key changes."""
    if _ROLE_DEBUG_LAST.get(key) != message:
        _ROLE_DEBUG_LAST[key] = message
        print(message)



# left_range, right_range
#         ↓
# compute_adaptive_worst_wall_range()
#         ↓
# adaptive_worst_wall_range
#         ↓
# select_adaptive_w_tau_threshold(adaptive_w)
#         ↓
# 최종 adaptive threshold T



# 좌·우 벽까지의 LiDAR 거리로 현재 통로 폭을 반영한 worst wall range를 계산
# 이 값은 고정 임계값 대신 환경에 맞게 LiDAR의 adaptive threshold를 설정하는 기준으로 사용됨
# 입력 거리값이 유효하지 않으면 None을 반환
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
    # 앵커로부터 좌우 벽 거리의 합과 더 먼 쪽 벽 거리를 이용해 피타고라스 방식으로 worst wall range 계산
    return math.hypot(left_range + right_range, max(left_range, right_range))



# 현재 통로에서 계산된 adaptive_w를 이용해
# Opening 판별에 사용할 최종 LiDAR 거리 임계값 T를 계산한다.
#
# 1. adaptive_w가 없으면 현재 벽 구조에 기반한 임계값을 만들 수 없으므로 None 반환
# 2. lower는 adaptive_w에 margin을 추가한 값으로,
#    정상적인 벽의 LiDAR 측정값이 Opening으로 잘못 분류되는 것을 방지하기 위한 하한값
# 3. upper는 LiDAR 최대 측정거리에서 최대거리 부근의 noise 영역을 제외한 상한값
# 4. lower >= upper이면 유효한 threshold 선택 구간이 존재하지 않으므로 None 반환
# 5. 최종 threshold T는 lower와 upper 사이에서 W_TAU_ALPHA(α)의 비율로 선택
#
#    T = lower + α(upper - lower)
#
#    α가 0에 가까울수록 threshold는 lower에 가까워지고,
#    α가 1에 가까울수록 threshold는 upper에 가까워진다.
#    따라서 고정된 threshold가 아니라 현재 통로의 벽 거리 구조를 반영한
#    adaptive threshold를 생성하여 이후 LiDAR ray의 Opening 여부 판별에 사용한다.
def select_adaptive_w_tau_threshold(adaptive_w: float | None) -> float | None:

    # adaptive_w를 계산하지 못한 경우 threshold 계산 불가
    if adaptive_w is None:
        return None

    # 정상적인 벽 거리보다 일정 비율의 margin을 추가하여
    # threshold가 선택될 수 있는 최소값(lower)을 설정
    lower = adaptive_w * (1.0 + ADAPTIVE_W_MARGIN_RATIO)

    # LiDAR 최대 측정거리 부근의 noise 영역을 제외하여
    # threshold가 선택될 수 있는 최대값(upper)을 설정
    upper = LIDAR_MAX_RANGE * (1.0 - W_TAU_NOISE_FRACTION)

    # lower가 upper 이상이면 유효한 threshold 범위가 존재하지 않음
    if lower >= upper:
        return None

    # lower와 upper 사이에서 α 비율만큼 이동한 위치를
    # 최종 adaptive LiDAR threshold T로 선택
    #
    # T = lower + α(upper - lower)
    return lower + W_TAU_ALPHA * (upper - lower)



# 사각형 장애물(Rect)을 구성하는 4개의 벽을 각각 선분(segment) 형태로 변환한다.
# 각 벽 선분은 두 끝점 Vector2의 쌍으로 표현되며,
# 이후 LiDAR ray와 벽 사이의 교차점을 계산하여 LiDAR 측정 거리를 구할 때 사용한다.
def _rect_wall_segments(
    rectangle: pygame.Rect,
) -> tuple[tuple[pygame.Vector2, pygame.Vector2], ...]:

    # 사각형의 네 꼭짓점 좌표를 2차원 벡터로 변환
    top_left = pygame.Vector2(rectangle.topleft)
    top_right = pygame.Vector2(rectangle.topright)
    bottom_right = pygame.Vector2(rectangle.bottomright)
    bottom_left = pygame.Vector2(rectangle.bottomleft)

    # 사각형의 네 변을 (시작점, 끝점) 형태의 벽 선분으로 반환
    # 위쪽 → 오른쪽 → 아래쪽 → 왼쪽 벽 순서
    return (
        (top_left, top_right),
        (top_right, bottom_right),
        (bottom_right, bottom_left),
        (bottom_left, top_left),
    )



# LiDAR가 충돌 검사를 수행할 전체 벽 선분 목록을 생성한다.
# OUTER_BOUNDARY의 각 점을 다음 점과 연결하여 외곽 경계의 모든 벽 선분을 만들고,
# 마지막 점은 첫 번째 점과 연결하여 닫힌 경계를 구성한다.
# 여기에 _rect_wall_segments()로 생성한 사각형 장애물의 4개 벽 선분을 추가한다.
# 최종 LIDAR_WALL_SEGMENTS는 LiDAR ray와 벽의 교차점 및 측정 거리를 계산할 때 사용된다.
LIDAR_WALL_SEGMENTS = tuple(
    (
        pygame.Vector2(OUTER_BOUNDARY[index]),
        pygame.Vector2(OUTER_BOUNDARY[(index + 1) % len(OUTER_BOUNDARY)]),
    )
    for index in range(len(OUTER_BOUNDARY))
) + _rect_wall_segments(OBSTACLE_RECT)



# 한 번의 LiDAR scan 결과를 저장하는 데이터 구조이다.
# 각 ray의 Anchor 기준 상대 각도(angles_deg)와 측정 거리(ranges),
# 그리고 LiDAR의 최대 측정 거리(max_range)를 하나의 객체로 묶어 관리한다.
# map의 절대 좌표를 저장하지 않고 Anchor-local 관측값만 저장한다.
#
# frozen=True이므로 생성된 scan 데이터는 이후 수정할 수 없으며,
# 하나의 시점에서 얻은 LiDAR 관측값을 고정된 상태로 유지한다.
@dataclass(frozen=True)
class LidarScan:
    """Anchor-local angle/range observation, independent of map coordinates."""

    # 각 LiDAR ray의 Anchor 기준 상대 각도 [degree]
    angles_deg: tuple[float, ...]

    # 각 각도에서 측정된 벽/장애물까지의 거리
    # angles_deg와 같은 index끼리 하나의 LiDAR ray를 구성
    ranges: tuple[float, ...]

    # LiDAR가 측정할 수 있는 최대 거리
    max_range: float



# 한 시점에서 로봇 군집이 관측한 로컬 상태를 하나로 묶어 저장하는 데이터 구조이다.
# 각 로봇의 ID, 상대 위치, 속도와 함께 현재 LiDAR 담당 Anchor의 ID 및 LiDAR scan을 저장한다.
# 절대 map 좌표가 아니라 로봇 간 상대 위치와 Anchor-local LiDAR 정보만을 사용하므로,
# 이후 Junction/Branch 인식과 군집 상태 판단에 필요한 runtime observation으로 사용된다.
#
# frozen=True이므로 한 번 생성된 관측 데이터는 이후 수정되지 않는다.
@dataclass(frozen=True)
class LocalObservation:
    """The runtime boundary: relative swarm state plus one Anchor LiDAR scan."""

    # 해당 관측 데이터가 생성된 시점
    timestamp: float

    # 현재 관측에 포함된 로봇들의 ID
    robot_ids: tuple[int, ...]

    # 각 로봇의 상대 위치
    # robot_ids와 같은 index끼리 대응됨
    relative_positions: tuple[pygame.Vector2, ...]

    # 각 로봇의 현재 속도 벡터
    # robot_ids와 같은 index끼리 대응됨
    velocities: tuple[pygame.Vector2, ...]

    # 현재 LiDAR scan을 수행한 Anchor 로봇의 ID
    lidar_robot_id: int

    # 해당 Anchor가 측정한 한 시점의 LiDAR scan 데이터
    lidar_scan: LidarScan



# LiDAR의 각 ray 거리값에 원형 이동평균(Circular Moving Average)을 적용하여
# 순간적인 거리 측정 noise와 작은 변동을 완화하고 부드러운 range profile을 생성한다.
# 360° LiDAR는 첫 번째 ray와 마지막 ray가 서로 인접하므로,
# % 연산을 사용하여 배열의 양 끝도 연결된 것처럼 처리한다.
# 이렇게 smoothing된 거리값은 이후 wall boundary의 급격한 변화와
# Opening Candidate를 보다 안정적으로 검출하는 데 사용된다.
def smooth_ranges(
    ranges: tuple[float, ...] | list[float],
    window_size: int = 5,
) -> tuple[float, ...]:

    # 이동평균 window는 중심값을 기준으로 좌우에 같은 개수의 값을 사용해야 하므로
    # 0보다 큰 홀수만 허용한다.
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    # 중심 ray를 기준으로 좌우 몇 개의 ray를 평균에 포함할지 계산
    # 예: window_size=5이면 half=2 → [i-2, i-1, i, i+1, i+2] 사용
    half = window_size // 2

    # 전체 LiDAR ray 개수
    count = len(ranges)

    # 각 ray를 중심으로 주변 window_size개의 거리값을 평균낸다.
    # (index + offset) % count를 사용하여 마지막 ray 다음을 첫 번째 ray로 연결하므로
    # 360° LiDAR의 원형 구조를 유지한 채 moving average를 계산한다.
    return tuple(
        sum(
            ranges[(index + offset) % count]
            for offset in range(-half, half + 1)
        )
        / window_size
        for index in range(count)
    )




# 360° LiDAR의 인접한 두 ray 사이의 거리 변화량(range gradient)을 계산한다.
# 현재 ray의 거리값을 다음 ray의 거리값에서 빼서,
# 벽의 끝이나 Opening 시작/끝에서 나타나는 급격한 거리 변화를 찾는 데 사용한다.
#
# gradient > 0 : 다음 ray에서 측정 거리가 갑자기 증가
#                → 가까운 벽이 끝나고 먼 공간/Opening이 시작될 가능성
# gradient < 0 : 다음 ray에서 측정 거리가 갑자기 감소
#                → 먼 공간/Opening이 끝나고 가까운 벽이 시작될 가능성
#
# 마지막 ray와 첫 번째 ray도 서로 인접한 360° scan이므로
# % count를 사용하여 원형(circular)으로 gradient를 계산한다.
def circular_range_gradient(scan: LidarScan) -> tuple[float, ...]:

    # 전체 LiDAR ray 개수
    count = len(scan.angles_deg)

    # 각 ray와 바로 다음 ray의 거리 차이를 계산
    # Δr[i] = range[i+1] - range[i]
    # 마지막 index에서는 다음 index를 0으로 연결하여 360° 경계를 처리
    return tuple(
        scan.ranges[(index + 1) % count] - scan.ranges[index]
        for index in range(count)
    )





# LiDAR scan에서 로봇 진행 방향 기준 좌측(-90°)과 우측(+90°) 벽까지의 대표 거리를 추출한다.
# 먼저 전체 range에 moving-average smoothing을 적용하여 LiDAR noise를 줄인다.
# 정확히 한 개의 ray만 사용하는 대신 목표 각도 ±2° 범위의 여러 ray를 선택하고,
# 그 거리값들의 중앙값(median)을 사용하여 좌·우 벽 거리를 안정적으로 계산한다.
# 이렇게 얻은 left_range와 right_range는 이후 현재 통로의 폭/벽 구조를 반영한
# adaptive worst wall range와 adaptive threshold를 계산하는 입력값으로 사용된다.
def extract_lateral_wall_ranges(
    scan: LidarScan,
    window_size: int = 5,
) -> tuple[float, float]:

    # Raw LiDAR range에 원형 이동평균을 적용하여 거리 측정 noise를 완화
    smoothed = smooth_ranges(scan.ranges, window_size)

    def lateral_median(target_angle: float) -> float:

        # 목표 방향(target_angle)을 중심으로 ±2° 이내에 있는 LiDAR ray들의 smoothing된 거리값만 선택한다.
        # 각도 차이는 360° wrap-around를 고려하여 -180°~180° 범위로 계산한다.
        nearby = [
            measured_range
            for angle, measured_range in zip(scan.angles_deg, smoothed)
            if abs(((angle - target_angle + 180.0) % 360.0) - 180.0) <= 2.0
        ]

        # 선택된 여러 거리값을 정렬한 뒤 중앙값을 대표 거리로 사용하여
        # 개별 ray의 순간적인 이상값에 덜 민감하도록 한다.
        return sorted(nearby)[len(nearby) // 2]

    # 로봇 진행 방향 기준 +90° 방향의 오른쪽 벽 거리 추출
    right_range = lateral_median(90.0)

    # 로봇 진행 방향 기준 -90° 방향의 왼쪽 벽 거리 추출
    left_range = lateral_median(-90.0)

    # 좌측 및 우측 벽까지의 대표 거리 반환
    return left_range, right_range


# Mobile Anchor가 사용하는 360° LiDAR의 전체 처리 과정을 담당하는 클래스
# 360° ray 생성 → 벽까지 거리 측정 → 좌우 벽 거리 추출 → adaptive threshold 계산
# → threshold보다 먼 ray를 Opening 후보로 분류 → gradient로 Opening 경계 보정
# → 각 Opening의 방향과 입구 geometry를 계산한다.
class AnchorLidar:
    """Local 360-degree LiDAR adapted from the attached LiDAR simulator."""

    def __init__(self) -> None:

        # 360°를 LIDAR_RAYS 개수만큼 균등하게 나누어
        # Anchor 기준 -180° ~ +180°의 LiDAR ray 각도를 생성한다.
        self.angles = tuple(
            -180.0 + 360.0 * index / LIDAR_RAYS
            for index in range(LIDAR_RAYS)
        )

        # 각 ray의 초기 측정 거리를 LiDAR 최대 측정거리로 설정
        self.ranges = [LIDAR_MAX_RANGE] * LIDAR_RAYS

        # 각 ray가 Opening 영역으로 판정되었는지를 저장
        # True = Opening을 지지하는 ray, False = 벽 영역
        self.open_support = [False] * LIDAR_RAYS

        # 연속된 Opening ray들을 하나의 Opening으로 묶어 저장
        self.opening_groups: list[dict] = []

        # 마지막으로 정상적으로 측정된 좌측/우측 벽 거리 저장
        # 현재 scan에서 벽 거리 계산이 불가능할 때 이전 유효값을 사용할 수 있다.
        self.last_valid_left_wall_range: float | None = None
        self.last_valid_right_wall_range: float | None = None

        # 마지막으로 정상적으로 계산된 adaptive worst wall range
        self.last_valid_adaptive_w: float | None = None

        # 현재 Opening 판별에 실제 사용 중인 adaptive_w
        self.active_adaptive_w: float | None = None

        # adaptive_w를 이용하여 계산된 현재 Opening 판별 threshold T
        self.selected_w_tau_threshold: float | None = None

        # 현재 threshold를 계산할 수 있는 유효한 범위가 존재하는지 저장
        self.threshold_interval_valid = False

        # Junction 접근 과정에서 threshold를 고정해서 사용할지 여부
        self.threshold_locked = False

        # threshold가 lock되었을 때 유지할 threshold와 adaptive_w
        self.locked_w_tau_threshold: float | None = None
        self.locked_adaptive_w: float | None = None

        # 현재 LiDAR 관측에서 Junction으로 볼 수 있는 Opening 증거가 있는지 저장
        self.junction_evidence = False

        # 좌우 벽 거리의 시간에 따른 관측 기록
        self.lateral_range_history: list[tuple[float, float]] = []

        # 정상 통로에서 기준으로 사용할 좌우 벽 거리
        self.lateral_baseline_left: float | None = None
        self.lateral_baseline_right: float | None = None

        # Anchor가 정지한 상태에서 누적한 LiDAR 관측 횟수
        self.stationary_samples = 0

        # 정지 관측 동안 지속적으로 확인되는 Opening track 저장
        self.stationary_tracks: list[dict] = []

        # 최초 LiDAR scan 상태 생성
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

        # 하나의 LiDAR ray와 하나의 벽 선분이 교차하는지 계산하고,
        # 교차한다면 ray 시작점에서 교차점까지의 거리를 반환한다.
        # 교차하지 않거나 서로 평행하면 None을 반환한다.

        # 벽 선분의 시작점과 끝점
        start, end = segment

        # 벽 선분의 방향 벡터
        edge = end - start

        # ray 방향과 벽 방향의 2D cross product
        # 0에 가까우면 두 선이 평행하므로 교차점을 계산할 수 없다.
        denominator = direction.x * edge.y - direction.y * edge.x

        if abs(denominator) < 1.0e-10:
            return None

        # LiDAR ray 시작점에서 벽 시작점까지의 벡터
        offset = start - origin

        # ray를 따라 얼마나 이동하면 교차점에 도달하는지 계산
        ray_t = (
            offset.x * edge.y - offset.y * edge.x
        ) / denominator

        # 교차점이 벽 선분 내부의 어느 위치에 있는지 계산
        # 0 <= segment_t <= 1이어야 실제 벽 선분 위에 존재한다.
        segment_t = (
            offset.x * direction.y - offset.y * direction.x
        ) / denominator

        # ray의 앞쪽에 있고 동시에 실제 벽 선분 내부에 교차점이 존재하면
        # LiDAR ray에서 교차점까지의 거리 ray_t 반환
        if (
            ray_t >= 0.0
            and -1.0e-9 <= segment_t <= 1.0 + 1.0e-9
        ):
            return ray_t

        return None

    def scan(
        self,
        position: pygame.Vector2,
        yaw_degrees: float,
    ) -> LidarScan:
        """Measure wall ranges using only the Anchor's local pose and rays."""

        # =====================================================
        # 1. 360° LiDAR ray casting
        # =====================================================

        ranges: list[float] = []

        for angle in self.angles:

            # Anchor의 현재 yaw와 각 LiDAR ray의 상대각도를 합쳐
            # 해당 ray가 향하는 방향 벡터를 계산
            radians = math.radians(yaw_degrees + angle)

            direction = pygame.Vector2(
                math.cos(radians),
                math.sin(radians),
            )

            # 현재 ray와 모든 벽 선분의 교차 여부를 검사
            hits = (
                self._ray_hit(
                    position,
                    direction,
                    wall,
                )
                for wall in LIDAR_WALL_SEGMENTS
            )

            # 여러 벽과 교차했다면 Anchor에서 가장 가까운 벽을
            # 실제 LiDAR 측정거리로 사용
            nearest = min(
                (
                    hit
                    for hit in hits
                    if hit is not None
                ),
                default=LIDAR_MAX_RANGE,
            )

            # 최대 측정거리보다 먼 값은 LIDAR_MAX_RANGE로 제한
            ranges.append(
                min(LIDAR_MAX_RANGE, nearest)
            )

        # 현재 360° LiDAR 거리 측정 결과 저장
        self.ranges = ranges

        self.last_scan = LidarScan(
            self.angles,
            tuple(ranges),
            LIDAR_MAX_RANGE,
        )

        # =====================================================
        # 2. 좌·우 벽 거리 추출
        # =====================================================

        # Anchor 진행방향 기준 -90°/+90° 주변 ray를 이용해
        # 현재 통로의 좌측/우측 벽까지의 대표 거리를 계산
        left_range, right_range = (
            extract_lateral_wall_ranges(
                self.last_scan,
                window_size=5,
            )
        )

        # =====================================================
        # 3. Adaptive worst wall range 계산
        # =====================================================

        current_w = None

        # 좌우 모두 실제 벽을 관측하고 있을 때만
        # 현재 통로 구조에 대한 adaptive_w를 계산한다.
        if (
            left_range < LIDAR_MAX_RANGE - 1.0
            and right_range < LIDAR_MAX_RANGE - 1.0
        ):
            current_w = (
                compute_adaptive_worst_wall_range(
                    left_range,
                    right_range,
                )
            )

        # threshold가 아직 lock되지 않았다면
        # 현재 정상적으로 계산된 좌우 벽 거리와 adaptive_w를 저장
        if (
            not self.threshold_locked
            and current_w is not None
        ):
            self.last_valid_left_wall_range = left_range
            self.last_valid_right_wall_range = right_range
            self.last_valid_adaptive_w = current_w

        # =====================================================
        # 4. Adaptive threshold 결정
        # =====================================================

        if self.threshold_locked:

            # threshold가 이미 lock된 경우 Junction 접근 중 측정값 변화에 따라
            # threshold가 계속 바뀌지 않도록 기존 값을 그대로 사용
            self.active_adaptive_w = (
                self.locked_adaptive_w
            )
            self.selected_w_tau_threshold = (
                self.locked_w_tau_threshold
            )

        else:

            # 현재 adaptive_w가 정상적으로 계산되면 현재 값을 사용하고,
            # 계산되지 않으면 마지막으로 유효했던 값을 유지한다.
            self.active_adaptive_w = (
                current_w
                if current_w is not None
                else self.last_valid_adaptive_w
            )

            # adaptive_w를 이용하여 Opening 판별용
            # 최종 adaptive threshold T를 계산
            self.selected_w_tau_threshold = (
                select_adaptive_w_tau_threshold(
                    self.active_adaptive_w
                )
            )

        # threshold가 정상적으로 계산되었는지 확인
        self.threshold_interval_valid = (
            self.selected_w_tau_threshold
            is not None
        )

        # =====================================================
        # 5. Opening 검출
        # =====================================================

        # threshold와 LiDAR range profile을 이용하여
        # 연속적인 Opening sector를 검출
        self._classify_openings()

        # 유효한 threshold가 존재하고 Opening이 3개 이상 관측되면
        # 현재 scan에 Junction 구조의 증거가 있다고 판단
        self.junction_evidence = (
            self.threshold_interval_valid
            and len(self.opening_groups) >= 3
        )

        # 이번 시점의 LiDAR scan 반환
        return self.last_scan

    def _classify_openings(self) -> None:
        """Infer opening sectors solely from the Anchor's local scan."""

        # =====================================================
        # 1. LiDAR range smoothing
        # =====================================================

        # Raw LiDAR range에 원형 이동평균을 적용하여
        # 작은 noise와 순간적인 거리 변동을 완화
        smoothed = smooth_ranges(
            self.last_scan.ranges,
            window_size=5,
        )

        # 현재 adaptive threshold T
        threshold = self.selected_w_tau_threshold

        # threshold를 계산할 수 없으면 Opening 판별을 수행하지 않는다.
        if threshold is None:
            self.open_support = [False] * LIDAR_RAYS
            self.opening_groups = []
            return

        # =====================================================
        # 2. Threshold 기반 Opening ray 분류
        # =====================================================

        # smoothing된 측정거리가 threshold 이상이면
        # 해당 ray를 "Opening을 지지하는 ray"로 분류
        #
        # True  → Opening 후보
        # False → 일반 벽 영역
        self.open_support = [
            value >= threshold
            for value in smoothed
        ]

        # =====================================================
        # 3. 인접 ray 사이의 range gradient 계산
        # =====================================================

        # Δr[i] = r[i+1] - r[i]
        #
        # 큰 양수 gradient → 가까운 벽에서 먼 공간으로 변화
        # 큰 음수 gradient → 먼 공간에서 가까운 벽으로 변화
        # 이를 이용해 Opening의 시작/끝 경계를 더 정확하게 찾는다.
        gradients = [
            smoothed[(index + 1) % LIDAR_RAYS]
            - smoothed[index]
            for index in range(LIDAR_RAYS)
        ]

        self.opening_groups = []

        # =====================================================
        # 4. 연속된 Opening ray를 하나의 Opening으로 grouping
        # =====================================================

        for start in range(LIDAR_RAYS):

            # 현재 ray가 Opening이고 바로 이전 ray는 Opening이 아닐 때만
            # 새로운 Opening sector의 시작점으로 판단
            if (
                not self.open_support[start]
                or self.open_support[
                    (start - 1) % LIDAR_RAYS
                ]
            ):
                continue

            indices = [start]

            cursor = (
                start + 1
            ) % LIDAR_RAYS

            # 연속적으로 threshold 이상인 ray들을 하나의 그룹으로 묶는다.
            while (
                self.open_support[cursor]
                and cursor != start
            ):
                indices.append(cursor)
                cursor = (
                    cursor + 1
                ) % LIDAR_RAYS

            # 너무 적은 ray로 구성된 영역은
            # 작은 noise일 가능성이 있으므로 Opening 후보에서 제외
            if len(indices) < 5:
                continue

            # =================================================
            # 5. Range gradient를 이용한 Opening 경계 보정
            # =================================================

            # threshold 기반으로 얻은 대략적인 Opening 시작/끝 index
            coarse_start = (
                indices[0] - 1
            ) % LIDAR_RAYS

            coarse_end = (
                indices[-1]
            ) % LIDAR_RAYS

            # 대략적인 시작 경계 주변 ±6 ray를 후보로 설정
            start_candidates = [
                (coarse_start + delta)
                % LIDAR_RAYS
                for delta in range(-6, 7)
            ]

            # 대략적인 끝 경계 주변 ±6 ray를 후보로 설정
            end_candidates = [
                (coarse_end + delta)
                % LIDAR_RAYS
                for delta in range(-6, 7)
            ]

            # 거리값이 가장 크게 증가하는 위치를
            # Opening의 실제 시작 wall boundary로 선택
            start_index = max(
                start_candidates,
                key=lambda index: gradients[index],
            )

            # 거리값이 가장 크게 감소하는 위치를
            # Opening의 실제 끝 wall boundary로 선택
            end_index = (
                min(
                    end_candidates,
                    key=lambda index: gradients[index],
                )
                + 1
            ) % LIDAR_RAYS

            # =================================================
            # 6. Opening의 각도와 폭 계산
            # =================================================

            start_angle = self.angles[start_index]
            end_angle = self.angles[end_index]

            # Opening의 angular width 계산
            # % 360을 사용하여 -180°/+180° 경계도 처리
            width = (
                end_angle - start_angle
            ) % 360.0

            # 지나치게 작은 영역이나 거의 360° 전체인 영역은 제거
            if (
                width < 5.0
                or width >= 359.0
            ):
                continue

            # Opening 중앙 방향 계산
            center_angle = (
                (
                    start_angle
                    + width / 2.0
                    + 180.0
                )
                % 360.0
            ) - 180.0

            # =================================================
            # 7. Opening mouth의 양쪽 끝점 계산
            # =================================================

            start_rad = math.radians(
                start_angle
            )
            end_rad = math.radians(
                end_angle
            )

            # Opening 시작 경계의 거리와 각도를
            # Anchor 기준 2D 상대좌표로 변환
            mouth_a = pygame.Vector2(
                smoothed[start_index]
                * math.cos(start_rad),
                smoothed[start_index]
                * math.sin(start_rad),
            )

            # Opening 끝 경계 역시 Anchor 기준
            # 2D 상대좌표로 변환
            mouth_b = pygame.Vector2(
                smoothed[end_index]
                * math.cos(end_rad),
                smoothed[end_index]
                * math.sin(end_rad),
            )

            # Opening 입구를 가로지르는 벡터
            mouth_vector = (
                mouth_b - mouth_a
            )

            # 두 mouth endpoint가 사실상 동일하면
            # 유효한 입구 방향을 정의할 수 없으므로 제외
            if (
                mouth_vector.length_squared()
                <= environment.EPSILON
            ):
                continue

            # =================================================
            # 8. Branch-local 좌표축 생성
            # =================================================

            # mouth_a → mouth_b 방향의 단위벡터
            # 즉 Branch 입구를 가로지르는 tangent 방향
            mouth_tangent = (
                mouth_vector.normalize()
            )

            # mouth tangent에 수직인 벡터를 만들어
            # Branch 내부로 향하는 axis 후보 생성
            branch_axis = pygame.Vector2(
                -mouth_tangent.y,
                mouth_tangent.x,
            )

            # LiDAR에서 관측된 Opening 중앙 방향 벡터
            center_rad = math.radians(
                center_angle
            )

            opening_direction = pygame.Vector2(
                math.cos(center_rad),
                math.sin(center_rad),
            )

            # branch_axis가 Opening 반대 방향을 가리키고 있다면
            # 방향을 뒤집어 실제 Branch 내부 방향과 일치시킨다.
            if (
                branch_axis.dot(
                    opening_direction
                )
                < 0.0
            ):
                branch_axis *= -1.0

            # =================================================
            # 9. 최종 Opening descriptor 저장
            # =================================================

            # 검출된 Branch 입구의 각도와 geometry를 저장한다.
            # 모든 좌표는 Anchor 기준 상대 geometry이다.
            self.opening_groups.append(
                {
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                    "center_angle": center_angle,

                    # Branch 입구의 양쪽 wall endpoint
                    "mouth_a": mouth_a,
                    "mouth_b": mouth_b,

                    # Branch 입구 중앙점
                    "mouth_midpoint":
                        (mouth_a + mouth_b) * 0.5,

                    # Branch 입구를 가로지르는 방향
                    "mouth_tangent":
                        mouth_tangent,

                    # Branch 내부로 향하는 방향
                    "branch_axis":
                        branch_axis,
                }
            )



# world 좌표계에서 표현된 위치 차이(offset)를 Anchor 기준의 로컬 좌표계로 변환한다.
# Anchor의 yaw(진행 방향)를 기준으로 forward(전방)와 right(우측) 단위벡터를 만들고,
# offset을 각 방향에 투영(dot product)하여 전방/우측 상대거리를 계산한다.
# 반환값의 x는 Anchor 기준 앞(+)/뒤(-) 방향 거리,
# y는 Anchor 기준 오른쪽(+)/왼쪽(-) 방향 거리를 의미한다.
# 따라서 map상의 절대 위치를 직접 사용하는 것이 아니라,
# 두 위치 사이의 offset을 Anchor 방향 기준의 상대 좌표로 표현할 때 사용한다.
def world_offset_to_anchor_local(
    offset: pygame.Vector2,
    yaw_degrees: float,
) -> pygame.Vector2:

    # Anchor의 yaw를 degree에서 radian으로 변환
    yaw = math.radians(yaw_degrees)

    # Anchor가 바라보는 전방 방향의 단위벡터
    forward = pygame.Vector2(
        math.cos(yaw),
        math.sin(yaw),
    )

    # Anchor의 전방 방향에서 90° 회전한 우측 방향의 단위벡터
    right = pygame.Vector2(
        -math.sin(yaw),
        math.cos(yaw),
    )

    # offset을 forward와 right 방향에 각각 투영하여
    # world-frame offset → Anchor-local (전방, 우측) 상대좌표로 변환
    return pygame.Vector2(
        offset.dot(forward),
        offset.dot(right),
    )



# Anchor 기준 로컬 좌표계로 표현된 벡터를 다시 world 좌표계의 방향 벡터로 변환한다.
# local_vector.x는 Anchor 기준 전방(+)/후방(-) 성분,
# local_vector.y는 Anchor 기준 우측(+)/좌측(-) 성분을 의미한다.
# Anchor의 yaw를 이용해 world 좌표계에서의 forward와 right 단위벡터를 만든 뒤,
# 두 방향 성분을 결합하여 실제 world-frame 벡터를 계산한다.
# world_offset_to_anchor_local()과 반대 방향의 좌표 변환에 해당한다.
def anchor_local_to_world(
    local_vector: pygame.Vector2,
    yaw_degrees: float,
) -> pygame.Vector2:

    # Anchor의 yaw를 degree에서 radian으로 변환
    yaw = math.radians(yaw_degrees)

    # world 좌표계에서 Anchor가 바라보는 전방 방향 단위벡터
    forward = pygame.Vector2(
        math.cos(yaw),
        math.sin(yaw),
    )

    # world 좌표계에서 Anchor 기준 우측 방향 단위벡터
    right = pygame.Vector2(
        -math.sin(yaw),
        math.cos(yaw),
    )

    # 로컬 벡터의 전방 성분(x)과 우측 성분(y)을
    # 각각 world-frame의 forward/right 방향으로 변환한 뒤 합산
    return (
        forward * local_vector.x
        + right * local_vector.y
    )


# 로봇의 현재 위치에서 로봇 반경까지 포함한 영역이 실제 이동 가능한 공간인지 검사한다.
# 사전 정의된 UP/LEFT/RIGHT/BOTTOM 영역은 사용하지 않고,
# 실제 map의 walkable_mask만 이용하여 벽 및 obstacle과의 물리적 충돌 여부를 판단한다.
# 로봇 중심과 상·하·좌·우·대각선 총 9개 지점을 검사하며,
# 하나라도 map 밖이거나 이동 불가능한 pixel이면 False, 모두 안전하면 True를 반환한다.
def physical_is_walkable(
    position: pygame.Vector2,
    radius: float,
) -> bool:

    # 검사할 로봇 중심 위치를 pixel 좌표의 정수값으로 변환
    x = int(round(position.x))

    y = int(round(position.y))

    # 실제 충돌 판정에 사용할 반경 결정
    # 로봇 자체 radius와 벽 접촉을 고려한 WALL_CONTACT_RADIUS 중
    # 더 큰 값을 사용하여 로봇 외곽이 벽을 침범하지 않도록 한다.
    contact_radius = max(radius, WALL_CONTACT_RADIUS)

    # 실제 pixel 단위로 검사할 반경 계산
    # 최소 1 pixel 이상이며 소수점 반경은 ceil하여 보수적으로 처리
    pixel_radius = max(1, int(math.ceil(contact_radius)))

    # 로봇 중심에서 대각선 방향으로도 같은 반경만큼 떨어진 점을 검사하기 위해
    # x, y 각각의 대각선 offset을 계산한다.
    # diagonal² + diagonal² ≈ pixel_radius²
    diagonal = int(round(pixel_radius/ math.sqrt(2.0)))

    # 로봇이 차지하는 원형 영역을 근사하기 위해 총 9개 지점을 검사
    # 중심 1개 + 상하좌우 4개 + 대각선 4개
    test_points = [
        (x, y),                         # 중심
        (x + pixel_radius, y),          # 오른쪽
        (x - pixel_radius, y),          # 왼쪽
        (x, y + pixel_radius),          # 아래쪽
        (x, y - pixel_radius),          # 위쪽
        (x + diagonal, y + diagonal),   # 오른쪽 아래
        (x + diagonal, y - diagonal),   # 오른쪽 위
        (x - diagonal, y + diagonal),   # 왼쪽 아래
        (x - diagonal, y - diagonal),   # 왼쪽 위
    ]

    # 실제 map geometry를 기반으로 생성된 이동 가능 영역 mask
    mask = environment.walkable_mask

    # mask의 가로/세로 크기
    mask_width, mask_height = mask.get_size()

    # 9개의 검사점이 모두 이동 가능한 영역에 있는지 확인
    for px, py in test_points:

        # 검사점이 map 범위를 벗어나면 이동 불가능
        if not (
            0 <= px < mask_width
            and
            0 <= py < mask_height
        ):
            return False

        # 해당 pixel이 walkable_mask에서 0이면
        # 벽/obstacle 등 이동 불가능한 영역이므로 충돌로 판단
        if (
            mask.get_at(
                (px, py)
            )
            == 0
        ):
            return False

    # 로봇 중심과 외곽의 모든 검사점이 이동 가능한 영역이면
    # 해당 위치에 로봇이 물리적으로 존재할 수 있다고 판단
    return True


# 현재 시점의 로봇 군집 상태를 Anchor 기준의 LocalObservation으로 생성한다.
# 각 로봇의 world 좌표 자체를 저장하지 않고,
# Anchor와 각 로봇 사이의 위치 차이 및 속도를 Anchor 기준 로컬 좌표계로 변환하여 저장한다.
# 여기에 Anchor의 ID와 가장 최근 360° LiDAR scan을 함께 묶어,
# 이후 Junction/Branch 인식 및 로봇 군집 제어가 사용하는 로컬 관측 데이터를 만든다.
def build_local_observation(
    anchor: environment.Robot,
    robots: list[environment.Robot],
    lidar: AnchorLidar,
    anchor_yaw_degrees: float,
) -> LocalObservation:

    # 현재 simulation 시점의 모든 로컬 관측 정보를 하나의 LocalObservation으로 생성
    return LocalObservation(

        # 현재 관측이 생성된 simulation 시간
        timestamp=environment.simulation_time,

        # 현재 관측 대상인 모든 로봇의 ID 저장
        # 이후 상대 위치/속도와 동일한 index로 대응된다.
        robot_ids=tuple(
            robot.robot_id
            for robot in robots
        ),

        # 각 로봇의 위치에서 Anchor 위치를 빼서
        # "Anchor → 해당 로봇"의 상대 위치 벡터를 계산한다.
        #
        # 이후 world_offset_to_anchor_local()을 통해
        # world 방향으로 표현된 상대 벡터를
        # Anchor 기준 (전방/후방, 우측/좌측) 좌표로 변환한다.
        relative_positions=tuple(
            world_offset_to_anchor_local(
                robot.position - anchor.position,
                anchor_yaw_degrees,
            )
            for robot in robots
        ),

        # 각 로봇의 관측 속도 벡터도 Anchor 기준 로컬 좌표계로 변환한다.
        # 따라서 로봇이 Anchor 기준으로 어느 방향으로 움직이고 있는지를
        # 로컬 정보만으로 판단할 수 있다.
        velocities=tuple(
            world_offset_to_anchor_local(
                robot.observed_velocity,
                anchor_yaw_degrees,
            )
            for robot in robots
        ),

        # 현재 LiDAR 관측을 수행하고 있는 Anchor 로봇의 ID
        lidar_robot_id=anchor.robot_id,

        # Anchor가 가장 최근에 측정한 360° LiDAR scan
        lidar_scan=lidar.last_scan,
    )



# 시뮬레이터 내부의 실제 로봇 상태를 알고리즘이 사용할 수 있는
# Anchor 기준 LocalObservation으로 변환하는 명확한 경계(interface) 역할을 하는 클래스이다.
# 실제 변환 작업은 build_local_observation()에 위임하며,
# 이후 알고리즘이 simulator의 world 상태에 직접 접근하지 않고
# LocalObservation만 전달받도록 구조를 분리하는 역할을 한다.
class LocalObservationBuilder:
    """Explicit simulator-ground-truth to local-runtime observation boundary."""

    @staticmethod
    def build(
        anchor: environment.Robot,
        robots: list[environment.Robot],
        lidar: AnchorLidar,
        anchor_yaw_degrees: float,
    ) -> LocalObservation:

        # 별도의 객체 생성 없이 호출할 수 있는 static method로,
        # Anchor와 전체 로봇 상태, LiDAR, Anchor yaw를 전달하여
        # Anchor-relative LocalObservation을 생성하고 반환한다.
        return build_local_observation(
            anchor,
            robots,
            lidar,
            anchor_yaw_degrees,
        )



# Adaptive W-τ 기반 LiDAR detector가 현재 관측에서 Junction 증거를 검출했는지 반환한다.
# AnchorLidar.scan() 과정에서 adaptive threshold로 Opening들을 검출하고,
# 유효한 threshold가 존재하면서 필요한 Opening 조건이 만족되면
# lidar.junction_evidence가 True가 된다.
# 이 함수는 해당 판정 결과만 가져와 Junction 검출 여부를 True/False로 반환한다.
def lidar_detects_junction(lidar: AnchorLidar) -> bool:

    # AnchorLidar에서 이미 계산된 Junction evidence를 boolean 값으로 반환
    return bool(lidar.junction_evidence)



# 두 각도(first, second) 사이의 최소 각도 차이를 계산한다.
# 360° 원형 각도를 고려하여 -180°~+180° 범위로 차이를 정규화한 뒤 절댓값을 반환한다.
# 따라서 0°와 359°처럼 숫자상 차이는 크지만 실제 방향은 가까운 경우에도
# 실제 최소 각도 차이인 1°로 계산할 수 있다.
# LiDAR Opening의 방향이나 center angle이 서로 얼마나 가까운지 비교할 때 사용한다.
def circular_error(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


# LiDAR scan에서 원하는 로컬 각도(target_angle)에 가장 가까운 ray를 찾아
# 해당 방향에서 측정된 LiDAR 거리값을 반환한다.
# circular_error()를 사용하므로 -180°/+180° 경계를 포함한 360° 원형 각도를 올바르게 비교한다.
# 정확히 target_angle과 일치하는 ray가 없어도 가장 가까운 LiDAR ray의 측정값을 사용할 수 있다.
def lidar_range_at_local_angle(scan: LidarScan, target_angle: float) -> float:
    # 모든 LiDAR ray 중 target_angle과의 원형 각도 차이가 가장 작은 ray의 index를 선택
    index = min(
        range(len(scan.angles_deg)),
        key=lambda item: circular_error(
            scan.angles_deg[item],
            target_angle,
        ),
    )

    # 선택된 ray에서 측정된 거리값 반환
    return float(scan.ranges[index])


# Dead-end에서 Anchor의 로컬 LiDAR 관측만을 이용하여
# 전방 막힌 벽(dead-end wall)의 왼쪽/오른쪽 끝점을 Anchor-local 좌표로 생성한다.
# 전방 0° 거리와 좌·우 ±90° 벽 거리를 이용하며,
# +x는 Anchor 전방, -y는 왼쪽, +y는 오른쪽을 의미한다.
# 유효한 벽 거리값을 얻지 못하면 None을 반환한다.
# 생성된 wall_left와 wall_right는 이후 Dead-end wall의 위치/폭을 표현하고,
# wall 앞의 로봇 분포를 확인하거나 Backtracking Shepherd 형성에 사용할 수 있다.
def extract_dead_end_wall_segment(
    scan: LidarScan,
) -> tuple[
    pygame.Vector2,
    pygame.Vector2,
] | None:

    # Anchor 정면(0°) LiDAR ray의 거리값을 가져와
    # Dead-end 전방 벽까지의 거리를 얻는다.
    front_range = (
        lidar_range_at_local_angle(
            scan,
            0.0,
        )
    )

    # Anchor 기준 좌측(-90°)과 우측(+90°) 벽까지의
    # 대표 거리값을 추출한다.
    (
        left_range,
        right_range,
    ) = (
        extract_lateral_wall_ranges(
            scan
        )
    )

    # 전방/좌측/우측 거리 중 하나라도
    # NaN 또는 무한대 등 유효하지 않은 값이면
    # Dead-end wall geometry를 만들 수 없으므로 None 반환
    if (
        not math.isfinite(front_range)
        or not math.isfinite(left_range)
        or not math.isfinite(right_range)
    ):
        return None

    # 전방/좌측/우측 중 하나라도 LiDAR 최대 측정거리에 가까우면
    # 해당 방향에서 실제 벽을 안정적으로 관측했다고 볼 수 없으므로
    # Dead-end wall segment를 생성하지 않는다.
    if (
        front_range
        >= scan.max_range - 1.0
        or left_range
        >= scan.max_range - 1.0
        or right_range
        >= scan.max_range - 1.0
    ):
        return None

    # Anchor-local 좌표에서 Dead-end 전방 벽의 왼쪽 끝점 생성
    #
    # x = front_range → Anchor 전방의 벽까지 거리
    # y = -left_range → Anchor 기준 왼쪽 방향
    #
    # wall_left = (전방 거리, 왼쪽 거리)
    wall_left = pygame.Vector2(
        front_range,
        -left_range,
    )

    # Anchor-local 좌표에서 Dead-end 전방 벽의 오른쪽 끝점 생성
    #
    # x = front_range → 왼쪽 끝점과 동일한 전방 벽 위치
    # y = +right_range → Anchor 기준 오른쪽 방향
    #
    # wall_right = (전방 거리, 오른쪽 거리)
    wall_right = pygame.Vector2(
        front_range,
        right_range,
    )

    # Dead-end wall을 구성하는 왼쪽/오른쪽 끝점을 반환
    return (
        wall_left,
        wall_right,
    )


# 새로운 corridor에 진입했을 때 Junction entrance 검출을 새로 시작할 수 있도록
# 이전 corridor에서 누적한 좌·우 lateral LiDAR 거리 기록과 baseline을 초기화한다.
# 이전 통로의 폭/벽 거리 기준이 새로운 통로의 Junction 검출에 영향을 주는 것을 방지한다.
# world/map상의 절대 위치는 사용하지 않고 LiDAR의 local lateral 관측 상태만 reset한다.
def reset_junction_entrance_detector(
    lidar: AnchorLidar,
) -> None:

    # 이전 corridor에서 누적한 좌·우 벽 거리 관측 기록 삭제
    lidar.lateral_range_history.clear()

    # 이전 corridor에서 설정된 좌측 벽 거리 baseline 제거
    lidar.lateral_baseline_left = None

    # 이전 corridor에서 설정된 우측 벽 거리 baseline 제거
    lidar.lateral_baseline_right = None



# Anchor가 corridor를 이동하는 동안 ±90° LiDAR 거리 변화를 추적하여 Junction entrance 도달 여부를 판단한다.
# 최근 lateral 거리의 중앙값을 현재 corridor의 좌·우 baseline으로 사용하고,
# 현재 거리와 baseline의 차이(delta)가 양쪽 모두 threshold를 초과하면 통로가 양쪽으로 넓어진 것으로 판단한다.
def update_junction_entrance_detector(
    lidar: AnchorLidar,
) -> tuple[bool, float, float, float, float]:

    # 현재 Anchor 기준 왼쪽(-90°), 오른쪽(+90°) 벽 거리 측정
    left = lidar_range_at_local_angle(lidar.last_scan, -90.0)
    right = lidar_range_at_local_angle(lidar.last_scan, 90.0)
    history = lidar.lateral_range_history

    # baseline 계산에 필요한 lateral 관측값이 충분하지 않으면 현재 값을 누적
    if len(history) < LATERAL_BASELINE_SAMPLES:
        history.append((left, right))
        return False, left, right, 0.0, 0.0

    # 최근 lateral 관측값을 사용하여 현재 corridor의 좌·우 벽 거리 baseline 계산
    window = history[-LATERAL_BASELINE_SAMPLES:]
    baseline_left = sorted(value[0] for value in window)[LATERAL_BASELINE_SAMPLES // 2]
    baseline_right = sorted(value[1] for value in window)[LATERAL_BASELINE_SAMPLES // 2]

    # 현재 좌·우 벽 거리가 평소 corridor보다 얼마나 멀어졌는지 계산
    delta_left, delta_right = left - baseline_left, right - baseline_right

    # 좌우 벽 거리가 동시에 threshold 이상 증가하면 Junction entrance에 도달했다고 판단
    reached = (
        delta_left > LATERAL_RANGE_JUMP_THRESHOLD
        and delta_right > LATERAL_RANGE_JUMP_THRESHOLD
    )

    if reached:
        # Junction entrance 검출 당시의 corridor baseline 저장
        lidar.lateral_baseline_left = baseline_left
        lidar.lateral_baseline_right = baseline_right
    else:
        # 아직 Junction이 아니면 현재 값을 추가하여 baseline을 계속 갱신
        history.append((left, right))
        if len(history) > 30:
            del history[:-30]  # 최근 30개의 관측값만 유지

    # Junction 도달 여부, 현재 좌우 거리, baseline 대비 좌우 거리 변화량 반환
    return reached, left, right, delta_left, delta_right


# Junction entrance가 검출되어 Anchor가 정지하면, 직전 corridor의 좌·우 baseline을 이용해
# adaptive_w와 W-τ threshold를 다시 계산하고 고정한다.
# 정지 후 Junction 내부의 넓어진 거리값 때문에 threshold가 변하는 것을 방지하여,
# 동일한 기준으로 stationary LiDAR scan을 누적하고 Opening들을 검증하기 위해 사용한다.
def freeze_stationary_threshold(lidar: AnchorLidar) -> None:

    # Junction 진입 직전 corridor의 좌·우 baseline으로 adaptive worst wall range 계산
    adaptive_w = compute_adaptive_worst_wall_range(
        lidar.lateral_baseline_left,
        lidar.lateral_baseline_right,
    )

    # 계산된 adaptive_w를 이용해 stationary Opening 검출에 사용할 W-τ threshold 계산
    threshold = select_adaptive_w_tau_threshold(adaptive_w)

    # 유효한 corridor baseline이 없어 threshold를 만들 수 없으면 실행 중단
    if adaptive_w is None or threshold is None:
        raise RuntimeError("Missing lateral corridor baseline.")

    # 이후 LiDAR scan에서 threshold가 다시 계산되지 않도록 현재 값을 고정
    lidar.threshold_locked = True
    lidar.locked_adaptive_w = lidar.active_adaptive_w = adaptive_w
    lidar.locked_w_tau_threshold = lidar.selected_w_tau_threshold = threshold

    # 새로운 stationary verification을 시작하기 위해 이전 누적 상태 초기화
    lidar.stationary_samples = 0
    lidar.stationary_tracks.clear()



# Anchor가 Junction entrance에서 정지한 동안 반복적으로 검출되는 Opening을 추적하여
# 일시적인 LiDAR noise가 아닌 지속적인 Branch Opening인지 확인하고 Junction을 최종 확정한다.
# 각 Opening의 center angle을 이전 track과 매칭하여 관측 횟수와 persistence ratio를 계산하고,
# 지속 조건을 만족하는 Opening track이 3개 이상이면 Junction으로 확인한다.
def update_stationary_junction_confirmation(
    lidar: AnchorLidar,
) -> bool:

    # Anchor 정지 후 수행된 LiDAR 관측 횟수 증가
    lidar.stationary_samples += 1

    # 현재 scan에서 검출된 각 Opening을 기존 stationary track과 비교
    for opening in lidar.opening_groups:
        center = opening["center_angle"]

        # center angle이 허용 오차 이내인 기존 track을 찾아 같은 Opening으로 association
        match = next(
            (
                track
                for track in lidar.stationary_tracks
                if circular_error(center, track["center"])
                <= STATIONARY_ASSOCIATION_TOLERANCE_DEG
            ),
            None,
        )

        # 대응되는 기존 track이 없으면 새로운 Opening track 생성
        if match is None:
            lidar.stationary_tracks.append(
                {
                    "center": center,
                    "observations": 1,
                }
            )

        # 기존 track과 매칭되면 해당 Opening의 반복 관측 횟수 증가
        else:
            match["observations"] += 1

    # 충분한 횟수로 관측되었고 전체 stationary scan 중 일정 비율 이상 계속 나타난
    # Opening만 persistent Opening으로 인정
    persistent_tracks = [
        track
        for track in lidar.stationary_tracks
        if (
            track["observations"] >= STATIONARY_MIN_PERSISTENT_OBSERVATIONS
            and
            track["observations"] / lidar.stationary_samples
            >= STATIONARY_PERSISTENCE_RATIO
        )
    ]

    # 10 scan마다 현재 threshold, 좌/정면/우 거리, Opening 및 track 상태를 출력하여 디버깅
    if lidar.stationary_samples % 10 == 0:
        left_range = lidar_range_at_local_angle(lidar.last_scan, -90.0)
        front_range = lidar_range_at_local_angle(lidar.last_scan, 0.0)
        right_range = lidar_range_at_local_angle(lidar.last_scan, 90.0)

        print(
            "[StationaryJunctionDebug] "
            f"samples={lidar.stationary_samples} "
            f"T={lidar.selected_w_tau_threshold:.2f} "
            f"ranges=L:{left_range:.2f},F:{front_range:.2f},R:{right_range:.2f} "
            f"opening_count={len(lidar.opening_groups)} "
            f"opening_angles={[round(o['center_angle'], 1) for o in lidar.opening_groups]} "
            f"tracks={[(round(t['center'], 1), t['observations']) for t in lidar.stationary_tracks]} "
            f"persistent={len(persistent_tracks)}"
        )

    # 지속적으로 관측되는 Opening이 3개 이상이면 실제 Junction으로 최종 확인
    return len(persistent_tracks) >= 3


# 독립적으로 제어되는 Anchor를 완전히 정지시키기 위해 모든 동적 상태를 0으로 초기화한다.
# 실제 velocity뿐만 아니라 acceleration, 제어 명령 속도(commanded_velocity),
# 관측 속도(observed_velocity)까지 제거하여 이전 이동 명령이나 운동 상태가 남지 않도록 한다.
# Junction entrance 도달 후 Anchor를 정지시키고 stationary LiDAR verification을 수행할 때 사용한다.
def stop_anchor(anchor: environment.Robot) -> None:

    # Anchor의 실제 현재 속도를 0으로 설정
    anchor.velocity.update(0.0, 0.0)

    # 이전 제어에서 남아 있는 가속도를 제거
    anchor.acceleration.update(0.0, 0.0)

    # Anchor에 내려진 이동 명령 속도를 제거하여 다시 움직이지 않도록 함
    anchor.commanded_velocity.update(0.0, 0.0)

    # 알고리즘이 관측하는 Anchor의 속도 역시 0으로 초기화
    anchor.observed_velocity.update(0.0, 0.0)


# 현재 Physical DFS state machine의 motion_mode를
# Anchor가 실제로 수행해야 하는 명령(command) 이름으로 변환한다.
# 즉 내부 상태 이름을 사람이 이해하기 쉬운 Anchor 행동 이름으로 매핑하여,
# 현재 Anchor가 통로 이동, Branch 탐색, Shepherd 모집/밀기, Backtracking,
# Junction 복귀, 다음 Branch 선택, 최종 Base 복귀 중 어떤 동작을 수행하는지 나타낸다.
# 정의되지 않은 motion_mode가 들어오면 "UNKNOWN"을 반환한다.
def anchor_command_name(
    motion_mode: str,
) -> str:
    """Return the command that the Anchor is currently executing."""

    return {
        # 초기 Root corridor를 따라 Junction 방향으로 이동
        "ROOT_SETUP": "FOLLOW_CORRIDOR",

        # 선택된 Branch 입구로 진입
        "BRANCH_ENTRY": "ENTER_BRANCH",

        # 현재 Branch 내부를 전진하며 탐색
        "BRANCH_EXPLORE": "EXPLORE_BRANCH",

        # Backtracking에 사용할 Shepherd 로봇 모집
        "BACKTRACK_WAIT_SHEPHERD": "RECRUIT_SHEPHERD",

        # Shepherd가 NORMAL 군집을 물리적으로 밀어 Backtracking 시작
        "PRESSURE_PUSH": "PUSH_SHEPHERD",

        # 형성된 역방향 flow를 따라 Junction으로 복귀
        "FLOW_BACKTRACK": "RETURN_TO_JUNCTION",

        # Dead-end wall에서 Backtracking Shepherd를 분리
        "BACKTRACK_WALL_DETACH": "DETACH_SHEPHERD_FROM_WALL",

        # 복귀 과정에서 corridor corner 방향으로 회전
        "BACKTRACK_CORNER_TURN": "TURN_RETURN_CORNER",

        # 복귀한 Junction의 Branch 입구에서 대기
        "RETURN_JUNCTION_ENTRANCE": "HOLD_AT_JUNCTION_MOUTH",

        # 로컬 관측으로 Junction center를 추정
        "RETURN_CENTER_ESTIMATE": "ESTIMATE_JUNCTION_CENTER",

        # 추정된 Junction center 방향으로 이동
        "RETURN_CENTER_TRANSIT": "MOVE_TO_JUNCTION_CENTER",

        # Junction center에 도착한 상태로 대기
        "RETURN_JUNCTION_CENTERED": "HOLD_AT_JUNCTION_CENTER",

        # 나머지 swarm이 Junction으로 복귀할 때까지 대기
        "WAIT_SWARM_RETURN": "WAIT_FOR_SWARM_RETURN",

        # Branch 복귀 완료 후 Backtracking Shepherd 역할 해제
        "FINALIZE_BRANCH_RETURN": "RELEASE_BACKTRACK_SHEPHERD",

        # 탐색 완료된 Branch 입구의 Shepherd 경계를 다시 형성
        "REFORM_VISITED_BRANCH_SHEPHERD": "REFORM_ENTRANCE_SHEPHERD",

        # DFS 상태에 따라 다음 탐색 Branch 선택
        "SELECT_NEXT_BRANCH": "SELECT_NEXT_BRANCH",

        # Root Junction의 모든 탐색 완료 후 최종 복귀 준비
        "ROOT_COMPLETE": "PREPARE_FINAL_RETURN",

        # Base 방향 최종 복귀를 위한 push 역할 할당
        "FINAL_RETURN_PREP": "ASSIGN_FINAL_PUSH",

        # 양쪽에 분산된 swarm이 합류할 때까지 대기
        "FINAL_WAIT_SIDE_MERGE": "WAIT_SIDE_SWARM_MERGE",

        # 군집을 Base 방향으로 최종적으로 밀어 복귀
        "FINAL_BASE_PUSH": "FINAL_BASE_PUSH",

        # Anchor 자체를 Base로 최종 복귀
        "FINAL_ANCHOR_RETURN": "RETURN_ANCHOR_TO_BASE",

        # 전체 Physical DFS 탐색 및 복귀 완료
        "SYSTEM_COMPLETE": "SYSTEM_STOP",

    # 등록되지 않은 motion_mode이면 UNKNOWN 반환
    }.get(
        motion_mode,
        "UNKNOWN",
    )


# Anchor-local 좌표계에서 주어진 속도 명령을 실제 시뮬레이션 이동으로 적용하고,
# 실제로 이동한 거리를 다시 Anchor-local 좌표계의 odometry로 반환한다.
# 이동 전에는 충돌 여부를 검사하며, 이동할 수 없는 위치라면 Anchor를 정지시킨다.
def integrate_anchor_local_command(
    anchor: environment.Robot,
    local_velocity: pygame.Vector2,
    yaw_degrees: float,
    dt: float,
) -> pygame.Vector2:
    """Actuate a local command and return actual local-frame odometry."""

    # 이동 전 Anchor 위치 저장 → 이후 실제 이동량(odometry) 계산에 사용
    previous_position = anchor.position.copy()

    # Anchor-local 속도 명령을 시뮬레이션 world 방향의 속도 벡터로 변환
    world_velocity = anchor_local_to_world(local_velocity, yaw_degrees)

    # 속도 × 시간으로 이번 step에서 이동할 다음 위치 계산
    next_position = anchor.position + world_velocity * dt
    anchor.previous_position.update(anchor.position)

    # 다음 위치가 벽과 충돌하지 않고 실제로 이동 가능한지 검사
    if physical_is_walkable(next_position, anchor.radius):
        # 이동 가능하면 계산된 위치와 속도를 Anchor에 실제 적용
        anchor.position.update(next_position)
        anchor.velocity.update(world_velocity)
    else:
        # 벽과 충돌할 경우 위치는 변경하지 않고 속도를 0으로 설정
        anchor.velocity.update(0.0, 0.0)

    # 독립 Anchor 제어에서는 별도의 가속도 적분 없이 현재 이동 결과를 상태에 반영
    anchor.acceleration.update(0.0, 0.0)
    anchor.commanded_velocity.update(anchor.velocity)
    anchor.observed_velocity.update(anchor.velocity)

    # 실제로 이동한 world-frame 변위를 다시 Anchor-local 좌표계로 변환하여 반환
    # 따라서 명령한 이동량이 아니라 충돌 검사까지 반영된 실제 odometry가 반환됨
    return world_offset_to_anchor_local(
        anchor.position - previous_position,
        yaw_degrees,
    )


# Anchor가 통로를 전진하면서 좌·우 LiDAR 거리만 이용해 통로 중앙을 유지하도록 제어한다.
# 좌우 벽 거리 차이를 lateral error로 계산하고, 그 차이에 비례해 좌우 보정 속도를 생성한다.
# 최종적으로 일정한 전진 속도 + lateral 보정 속도를 Anchor-local 명령으로 전달한다.
def follow_corridor_locally(
    anchor: environment.Robot,
    observation: LocalObservation,
    yaw_degrees: float,
    dt: float,
) -> pygame.Vector2:
    """Move forward while centring from only local left/right LiDAR returns."""

    # 현재 LiDAR scan에서 Anchor 기준 좌측(-90°), 우측(+90°) 벽 거리 추출
    left_range, right_range = extract_lateral_wall_ranges(observation.lidar_scan)

    # 좌우 벽 거리 차이로 통로 중앙에서 얼마나 벗어났는지 계산
    # 두 거리가 같으면 error=0이므로 중앙에 위치한 상태
    lateral_error = right_range - left_range

    # lateral error에 gain을 곱해 좌우 보정 속도를 계산하고,
    # 너무 큰 횡방향 이동을 막기 위해 최대 속도 범위로 제한
    lateral_speed = max(-ANCHOR_MAX_LATERAL_SPEED, min(ANCHOR_MAX_LATERAL_SPEED, ANCHOR_LATERAL_GAIN * lateral_error))

    # local x축으로 일정한 속도로 전진하면서,
    # local y축으로 lateral_speed만큼 보정하여 통로 중앙을 따라 이동
    # 실제 이동 후 Anchor-local odometry를 반환
    return integrate_anchor_local_command(
        anchor,
        pygame.Vector2(ANCHOR_FORWARD_SPEED, lateral_speed),
        yaw_degrees,
        dt,
    )


# Backtracking 중 Anchor가 NORMAL swarm보다 혼자 앞서가지 않도록 선두 cohort를 따라 이동시킨다.
# NORMAL 선두의 위치·속도로 전진 속도를 결정하고 LiDAR로 통로 중앙을 유지하며,
# 이동 후 captured Shepherd와의 직접 통신 거리(COMM_RANGE)가 유지되는지도 검사한다.
def follow_backtrack_corridor_with_comm_guard(
    anchor: environment.Robot,
    robots: list[environment.Robot],
    lidar: AnchorLidar,
    shepherd_ids: set[int],
    yaw_degrees: float,
    dt: float,
) -> pygame.Vector2:

    # 현재 로봇들의 위치·속도를 Anchor-local 관측으로 생성
    observation = LocalObservationBuilder.build(anchor, robots, lidar, yaw_degrees)
    robots_by_id = {robot.robot_id: robot for robot in robots}
    local_position_by_id = {robot_id: local_position for robot_id, local_position in zip(observation.robot_ids, observation.relative_positions)}
    local_velocity_by_id = {robot_id: local_velocity for robot_id, local_velocity in zip(observation.robot_ids, observation.velocities)}

    # 현재 return corridor 폭 안에 있고 Anchor에서 일정 통신 범위 내에 있는 NORMAL 로봇만 후보로 선택
    half_width = 0.5 * KNOWN_CORRIDOR_WIDTH + environment.GRID_SPACING
    normal_candidates: list[tuple[int, pygame.Vector2]] = []

    for robot_id, local_position in local_position_by_id.items():
        if robot_id == observation.lidar_robot_id:
            continue  # Anchor 자신 제외

        robot = robots_by_id[robot_id]
        if robot.role != "NORMAL":
            continue  # NORMAL 로봇만 사용
        if abs(local_position.y) > half_width:
            continue  # 현재 corridor 폭 밖의 로봇 제외
        if abs(local_position.x) > BACKTRACK_FLOW_EVAL_HOPS * environment.COMM_RANGE:
            continue  # Anchor에서 너무 멀리 떨어진 로봇 제외

        normal_candidates.append((robot_id, local_position))

    # 따라갈 NORMAL swarm이 없으면 Anchor가 혼자 선행하지 않고 정지
    if not normal_candidates:
        stop_anchor(anchor)
        role_debug("backtrack-anchor-front-hold", "[BacktrackAnchorFrontHold] reason=NO_NORMAL_SWARM")
        return pygame.Vector2()

    # +x(Junction 복귀 방향) 기준으로 가장 앞쪽 NORMAL들을 선두 cohort로 선정
    normal_candidates.sort(key=lambda item: item[1].x, reverse=True)
    front_count = min(len(normal_candidates), max(3, len(normal_candidates) // 5))
    front_cohort = normal_candidates[:front_count]

    # 한 대의 outlier 대신 선두 cohort의 중앙값을 대표 선두 위치로 사용
    front_x_values = sorted(local_position.x for _, local_position in front_cohort)
    front_x = front_x_values[len(front_x_values) // 2]

    # 선두 cohort의 실제 Junction 방향 속도 중앙값을 대표 전진 속도로 사용
    front_speed_values = sorted(max(0.0, local_velocity_by_id[robot_id].x) for robot_id, _ in front_cohort)
    front_speed = front_speed_values[len(front_speed_values) // 2]

    # Anchor가 NORMAL 선두보다 일정 거리 뒤를 유지하도록 목표 간격 계산
    target_front_gap = max(2.0 * environment.ROBOT_RADIUS, environment.GRID_SPACING)
    front_gap_error = front_x - target_front_gap

    # Anchor가 선두에 너무 가까우면 정지하고, 뒤처진 경우에만 선두 속도에 catch-up 속도를 추가
    if front_gap_error <= 0.0:
        forward_speed = 0.0
    else:
        catchup_speed = 1.5 * front_gap_error
        forward_speed = min(ANCHOR_FORWARD_SPEED, front_speed + catchup_speed)
        forward_speed = max(0.0, forward_speed)

    # 좌·우 LiDAR 거리 차이를 이용해 Backtracking 중에도 corridor 중앙 유지
    left_range, right_range = extract_lateral_wall_ranges(observation.lidar_scan)
    lateral_error = right_range - left_range
    lateral_speed = max(-ANCHOR_MAX_LATERAL_SPEED, min(ANCHOR_MAX_LATERAL_SPEED, ANCHOR_LATERAL_GAIN * lateral_error))

    # 계산한 전진·횡방향 속도로 Anchor를 실제 이동시키고 local odometry 저장
    before_position = anchor.position.copy()
    before_previous = anchor.previous_position.copy()
    local_delta = integrate_anchor_local_command(anchor, pygame.Vector2(forward_speed, lateral_speed), yaw_degrees, dt)

    # 이동 후 새로운 Anchor-local 관측을 생성하여 captured Shepherd와의 통신 상태 확인
    post_observation = LocalObservationBuilder.build(anchor, robots, lidar, yaw_degrees)
    post_local_by_id = {robot_id: local_position for robot_id, local_position in zip(post_observation.robot_ids, post_observation.relative_positions)}

    # captured ID 중 현재 실제 SHEPHERD 역할을 유지하고 있는 로봇만 추출
    active_shepherd_ids = {
        robot_id for robot_id in shepherd_ids
        if robot_id in post_local_by_id
        and robot_id in robots_by_id
        and robots_by_id[robot_id].role == "SHEPHERD"
    }

    # 모든 active Shepherd가 Anchor와 직접 COMM_RANGE 안에 있는지 검사
    communication_ok = bool(active_shepherd_ids) and all(
        post_local_by_id[robot_id].length() <= environment.COMM_RANGE
        for robot_id in active_shepherd_ids
    )

    # Anchor 이동으로 Shepherd와의 통신이 끊겼다면 방금 이동을 취소하고 정지
    if not communication_ok:
        anchor.position.update(before_position)
        anchor.previous_position.update(before_previous)
        stop_anchor(anchor)
        role_debug(
            "backtrack-anchor-comm-hold",
            f"[BacktrackAnchorCommHold] shepherds={len(active_shepherd_ids)} reason=COMM_RANGE",
        )
        return pygame.Vector2()

    # 정상적으로 swarm 선두를 따라가고 있는 현재 제어 상태 출력
    role_debug(
        "backtrack-anchor-front-follow",
        f"[BacktrackAnchorFrontFollow] front_count={front_count} front_x={front_x:.2f} "
        f"target_gap={target_front_gap:.2f} front_speed={front_speed:.2f} anchor_speed={forward_speed:.2f}",
    )

    # 충돌 및 통신 조건까지 반영된 Anchor의 실제 local 이동량 반환
    return local_delta


# Anchor가 특정 로봇 그룹 전체에 직접 명령을 전달할 수 있는 통신 범위 내에 있는지 확인한다.
# 각 로봇의 Anchor-local 상대 위치를 이용해 Anchor와의 거리를 계산하고,
# group_ids의 모든 로봇이 COMM_RANGE 이내에 있을 때만 True를 반환한다.
def anchor_can_command_group(
    observation: LocalObservation,
    group_ids: set[int],
) -> bool:

    # robot ID별 Anchor-local 상대 위치를 빠르게 조회할 수 있도록 dictionary 생성
    local_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(observation.robot_ids, observation.relative_positions)
    }

    # 그룹이 비어 있지 않고, 모든 로봇이 관측되며 Anchor와 직접 통신 가능한 거리 안에 있는지 확인
    return bool(
        group_ids
        and all(
            robot_id in local_by_id
            and local_by_id[robot_id].length() <= environment.COMM_RANGE
            for robot_id in group_ids
        )
    )


# Anchor 전방 반구의 LiDAR에서 충분히 멀리 열린 ray들이 연속되는 free gap을 찾고,
# 실제 진행 가능한 폭을 가진 gap 중 가장 적합한 하나를 선택하여 그 중심 각도를 진행 방향으로 반환한다.
def find_branch_free_gap(
    scan: LidarScan,
) -> float | None:

    # LiDAR range의 순간적인 변동을 줄이기 위해 먼저 smoothing
    smoothed = smooth_ranges(scan.ranges, window_size=5)

    # 전체 360° scan 중 Anchor가 탐색할 전방 FOV 안의 LiDAR ray만 추출
    samples = [
        (angle, measured_range)
        for angle, measured_range in zip(scan.angles_deg, smoothed)
        if -ANCHOR_EXPLORE_HALF_FOV_DEG <= angle <= ANCHOR_EXPLORE_HALF_FOV_DEG
    ]

    gaps: list[list[tuple[float, float]]] = []
    current_gap: list[tuple[float, float]] = []

    # 측정 거리가 traversable threshold 이상인 연속 ray들을 하나의 free gap으로 grouping
    for angle, measured_range in samples:
        if measured_range >= ANCHOR_TRAVERSABLE_RANGE:
            current_gap.append((angle, measured_range))
        else:
            if current_gap:
                gaps.append(current_gap)
                current_gap = []

    # 마지막까지 열린 상태로 끝난 gap도 저장
    if current_gap:
        gaps.append(current_gap)

    # 각도 폭이 너무 좁은 gap은 Anchor가 통과하기 어렵다고 보고 제외
    valid_gaps = [
        gap for gap in gaps
        if gap[-1][0] - gap[0][0] >= ANCHOR_MIN_FREE_GAP_WIDTH_DEG
    ]

    # 실제 진행 가능한 free gap이 없으면 탐색 방향 없음
    if not valid_gaps:
        return None

    # 각 free gap의 폭, 평균 LiDAR 거리, 정면과의 각도 차이를 이용해 우선순위 계산
    def gap_score(gap):
        start_angle = gap[0][0]
        end_angle = gap[-1][0]
        width = end_angle - start_angle
        center_angle = 0.5 * (start_angle + end_angle)
        mean_range = sum(measured_range for _, measured_range in gap) / len(gap)

        # max() 기준: 넓은 gap → 평균 거리가 먼 gap → 정면에 가까운 gap 순으로 우선
        return (width, mean_range, -abs(center_angle))

    # 가장 높은 우선순위를 가진 free gap 선택
    best_gap = max(valid_gaps, key=gap_score)

    # 선택된 free gap의 중심 각도를 Anchor-local 진행 목표 방향으로 반환
    return 0.5 * (best_gap[0][0] + best_gap[-1][0])


# Branch 탐색 중 Anchor가 NORMAL swarm 선두보다 일정 거리 앞을 유지하면서 함께 이동하도록 전진 속도 상한을 계산한다.
# Anchor-relative 위치·속도만 사용하며, 선두와의 간격이 너무 크면 감속하고 너무 가까우면 가속한다.
def compute_branch_explore_anchor_speed_cap(
    observation: LocalObservation,
    robots_by_id: dict[int, environment.Robot],
    corridor_width: float,
) -> tuple[float, int | None, float, float]:

    # 현재 Branch corridor 폭에 여유 margin을 추가하여 선두 NORMAL 후보를 찾을 횡방향 범위 설정
    half_width = 0.5 * corridor_width + ANCHOR_FRONT_LATERAL_MARGIN_ROWS * environment.GRID_SPACING

    # Anchor 주변에서 swarm 선두를 탐색할 종방향 관측 거리 설정
    observation_depth = ANCHOR_FRONT_OBSERVATION_HOPS * environment.COMM_RANGE

    candidates: list[tuple[int, pygame.Vector2, pygame.Vector2]] = []

    # Anchor-local 관측에서 현재 Branch 내부의 NORMAL 로봇만 선두 후보로 추출
    for robot_id, local_position, local_velocity in zip(
        observation.robot_ids,
        observation.relative_positions,
        observation.velocities,
    ):
        if robot_id == observation.lidar_robot_id:
            continue  # Anchor 자신 제외

        robot = robots_by_id[robot_id]
        if robot.role != "NORMAL":
            continue  # swarm 선두 판단에는 NORMAL만 사용
        if abs(local_position.y) > half_width:
            continue  # 현재 Branch corridor 폭 밖의 로봇 제외
        if abs(local_position.x) > observation_depth:
            continue  # Anchor에서 너무 멀리 떨어진 로봇 제외

        candidates.append((robot_id, local_position, local_velocity))

    # 주변에 NORMAL swarm이 없으면 Anchor가 혼자 선행하지 않도록 전진 속도를 0으로 제한
    if not candidates:
        return 0.0, None, float("inf"), 0.0

    # Anchor-local +x가 진행 방향이므로 x가 가장 큰 NORMAL을 현재 swarm의 선두 로봇으로 선택
    front_robot_id, front_position, front_velocity = max(candidates, key=lambda item: item[1].x)

    # 선두 NORMAL이 Anchor 뒤에 있는 거리 계산: front_position.x=-4이면 Anchor가 4만큼 앞에 있음
    front_gap = -front_position.x

    # Anchor가 NORMAL 선두보다 유지해야 할 목표 거리 설정
    target_gap = max(
        3.0 * environment.ROBOT_RADIUS,
        ANCHOR_FRONT_TARGET_GAP_ROWS * environment.GRID_SPACING,
    )

    # 선두 NORMAL의 실제 진행 방향(+x) 속도 사용
    front_speed = max(0.0, front_velocity.x)

    # 목표 간격과 현재 간격의 오차 계산
    # 양수: Anchor와 선두가 너무 가까움 → Anchor 속도 증가
    # 음수: Anchor가 너무 앞서 있음 → Anchor 속도 감소
    gap_error = target_gap - front_gap

    # 선두 NORMAL의 실제 속도를 기준으로 gap error에 비례하여 Anchor 속도 상한 조절
    speed_cap = front_speed + ANCHOR_FRONT_GAP_GAIN * gap_error

    # 최종 속도 상한을 0 ~ ANCHOR_FORWARD_SPEED 범위로 제한
    speed_cap = max(0.0, min(ANCHOR_FORWARD_SPEED, speed_cap))

    # 속도 상한, 선두 NORMAL ID, 현재 선두 간격, 선두 속도 반환
    return speed_cap, front_robot_id, front_gap, front_speed


# LiDAR가 찾은 free gap 방향으로 Anchor의 yaw를 조금씩 회전시키면서 전진시킨다.
# 정면 장애물 거리와 회전량에 따라 전진 속도를 조절하고, 필요하면 swarm 선두 기준 속도 상한도 적용한다.
def move_anchor_through_free_gap(
    anchor: environment.Robot,
    scan: LidarScan,
    yaw_degrees: float,
    target_local_angle: float,
    dt: float,
    max_forward_speed: float | None = None,
) -> tuple[float, pygame.Vector2]:

    # 한 step에서 Anchor가 회전할 수 있는 최대 각도 계산
    max_turn_this_step = ANCHOR_MAX_TURN_RATE_DEG * dt

    # free gap 방향으로 회전하되 최대 회전 속도를 넘지 않도록 제한
    applied_turn = max(-max_turn_this_step, min(max_turn_this_step, target_local_angle))

    # 실제 적용된 회전량을 현재 yaw에 더해 새로운 Anchor heading 계산
    new_yaw = normalize_angle(yaw_degrees + applied_turn)

    # Anchor 정면(0°)의 LiDAR 거리 확인
    front_range = lidar_range_at_local_angle(scan, 0.0)

    # 정면이 막혀 있고 목표 free gap이 옆쪽에 있으면 전진하지 않고 먼저 제자리 회전
    if front_range < ANCHOR_TRAVERSABLE_RANGE and abs(target_local_angle) > 30.0:
        forward_speed = 0.0

    else:
        # 정면 공간이 충분히 열려 있는 정도를 0~1 비율로 계산
        clearance_ratio = max(0.0, min(1.0, front_range / ANCHOR_TRAVERSABLE_RANGE))

        # 회전각이 클수록 전진 속도를 낮추고, 최소 20%의 비율은 유지
        turn_ratio = max(0.20, 1.0 - abs(target_local_angle) / 90.0)

        # 장애물 여유와 회전 정도 중 더 보수적인 값을 속도 결정에 사용
        speed_ratio = min(clearance_ratio, turn_ratio)

        # speed_ratio에 따라 최소 탐색 속도~최대 전진 속도 사이에서 실제 전진 속도 계산
        forward_speed = ANCHOR_MIN_EXPLORE_SPEED + (ANCHOR_FORWARD_SPEED - ANCHOR_MIN_EXPLORE_SPEED) * speed_ratio

    # swarm 선두를 기준으로 계산된 속도 상한이 있으면 Anchor가 그보다 빠르게 전진하지 못하도록 제한
    # LiDAR가 계산한 안전 속도를 증가시키지는 않고 필요할 때 감소시키기만 한다.
    if max_forward_speed is not None:
        forward_speed = min(forward_speed, max(0.0, max_forward_speed))

    # 갱신된 yaw 방향으로 Anchor를 실제 전진시키고 실제 local 이동량을 얻음
    local_delta = integrate_anchor_local_command(
        anchor,
        pygame.Vector2(forward_speed, 0.0),
        new_yaw,
        dt,
    )

    # 갱신된 Anchor yaw와 실제 Anchor-local 이동량 반환
    return new_yaw, local_delta


# Backtracking 중 코너를 돌 때 Anchor의 이동·회전을 현재 PUSH Shepherd들이 그대로 따라가도록 한다.
# Shepherd를 새로 선정하지 않고 기존 Shepherd들의 Anchor-relative 대형을 유지한 채 rigid motion으로 함께 이동시킨다.
def move_anchor_with_backtracking_shepherds(
    anchor: environment.Robot,
    scan: LidarScan,
    shepherd_ids: set[int],
    robots_by_id: dict[int, environment.Robot],
    yaw_degrees: float,
    target_local_angle: float,
    dt: float,
) -> tuple[float, pygame.Vector2, bool]:

    # 기존 shepherd_ids 중 현재 실제 SHEPHERD이면서 PUSH 모드인 로봇만 사용
    active_ids = {
        robot_id for robot_id in shepherd_ids
        if robots_by_id[robot_id].role == "SHEPHERD"
        and getattr(robots_by_id[robot_id], "shepherd_mode", None) == "PUSH"
    }

    # 따라올 PUSH Shepherd가 없으면 Anchor를 정지시키고 코너 이동 실패 반환
    if not active_ids:
        stop_anchor(anchor)
        return yaw_degrees, pygame.Vector2(), False

    # 이동 실패 시 복원할 수 있도록 이동 전 Anchor 위치와 yaw 저장
    old_anchor_position = anchor.position.copy()
    old_anchor_previous = anchor.previous_position.copy()
    old_yaw = yaw_degrees

    # 각 Shepherd의 현재 Anchor-local 상대 위치를 저장하여 기존 대형(capture topology)을 보존
    local_offsets = {}
    old_shepherd_positions = {}

    for robot_id in active_ids:
        robot = robots_by_id[robot_id]
        old_shepherd_positions[robot_id] = robot.position.copy()
        local_offsets[robot_id] = world_offset_to_anchor_local(
            robot.position - old_anchor_position,
            old_yaw,
        )

    # LiDAR free gap controller로 Anchor를 먼저 이동·회전시킴
    new_yaw, anchor_delta = move_anchor_through_free_gap(
        anchor,
        scan,
        old_yaw,
        target_local_angle,
        dt,
    )
    new_anchor_position = anchor.position.copy()

    # 저장했던 Anchor-local offset을 새로운 yaw에 적용하여 각 Shepherd의 목표 위치 계산
    target_positions = {}

    for robot_id in active_ids:
        target_position = new_anchor_position + anchor_local_to_world(
            local_offsets[robot_id],
            new_yaw,
        )

        # 한 Shepherd라도 목표 위치에서 벽과 충돌하면 Anchor의 이번 코너 이동 전체를 취소
        if not physical_is_walkable(target_position, robots_by_id[robot_id].radius):
            anchor.position.update(old_anchor_position)
            anchor.previous_position.update(old_anchor_previous)
            stop_anchor(anchor)

            role_debug(
                "backtrack-corner-rigid-block",
                f"[BacktrackCornerRigidBlock] robot_id={robot_id}",
            )
            return old_yaw, pygame.Vector2(), False

        target_positions[robot_id] = target_position

    # 모든 Shepherd가 이동 가능하면 동일한 상대 대형을 유지한 채 실제 위치를 함께 이동
    for robot_id in active_ids:
        robot = robots_by_id[robot_id]
        old_position = old_shepherd_positions[robot_id]
        target_position = target_positions[robot_id]

        robot.previous_position.update(old_position)
        robot.position.update(target_position)

        # 실제 이동 변위/dt로 Shepherd의 실현 속도를 계산하여 동적 상태에 반영
        realized_velocity = (target_position - old_position) / max(dt, environment.EPSILON)
        robot.velocity.update(realized_velocity)
        robot.observed_velocity.update(realized_velocity)
        robot.commanded_velocity.update(realized_velocity)
        robot.acceleration.update(0.0, 0.0)

    # Anchor와 Shepherd가 함께 수행한 코너 이동 상태 출력
    role_debug(
        "backtrack-anchor-copy",
        f"[BacktrackAnchorCopy] count={len(active_ids)} yaw={old_yaw:.2f}->{new_yaw:.2f} "
        f"anchor_move={anchor_delta.length():.4f}",
    )

    # 새로운 yaw, Anchor의 실제 이동량, 코너 이동 성공 여부 반환
    return new_yaw, anchor_delta, True


# 코너링을 했을 때 Shepherd 전체를 다시 뽑는 게 아니라 새 통로에 들어갈 수 있는 기존 Shepherd만 유지하고, 걸림을 만드는 바깥쪽 Shepherd만 NORMAL로 풀어주기
# Backtracking Shepherd들이 코너를 돌아 새 corridor 방향으로 정렬된 뒤,
# 새 corridor의 LiDAR 폭 안에 들어오는 Shepherd만 유지하고 바깥쪽 Shepherd는 NORMAL로 해제한다.
def trim_backtracking_shepherds_to_corridor(
    observation: LocalObservation,
    active_shepherd_ids: set[int],
    robots_by_id: dict[int, environment.Robot],
) -> tuple[set[int], set[int]]:

    # 현재 활성 Shepherd가 없으면 유지/해제할 로봇 없이 종료
    if not active_shepherd_ids:
        return set(), set()

    # 새 corridor 방향 기준 좌·우 벽까지의 LiDAR 거리 측정
    left_range = lidar_range_at_local_angle(observation.lidar_scan, +90.0)
    right_range = lidar_range_at_local_angle(observation.lidar_scan, -90.0)

    # Anchor가 corridor 정중앙이 아닐 수 있으므로 더 가까운 벽을 기준으로 안전한 사용 폭 계산
    # 로봇 반지름과 추가 여유 공간을 빼서 Shepherd가 벽에 닿지 않는 범위만 허용
    safe_half_width = (
        min(left_range, right_range)
        - environment.ROBOT_RADIUS
        - 0.5 * environment.GRID_SPACING
    )
    safe_half_width = max(environment.ROBOT_RADIUS, safe_half_width)

    # 각 로봇 ID에 대응하는 Anchor-local 상대 위치 생성
    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position in zip(
            observation.robot_ids,
            observation.relative_positions,
        )
    }

    kept_ids: set[int] = set()
    released_ids: set[int] = set()

    # 각 Shepherd의 횡방향(local y) 위치가 새 corridor의 안전 폭 안에 있는지 검사
    for robot_id in active_shepherd_ids:
        local_position = local_position_by_id.get(robot_id)

        # 현재 local observation에서 찾을 수 없는 Shepherd는 임의로 해제하지 않고 일단 유지
        if local_position is None:
            kept_ids.add(robot_id)
            continue

        # corridor 안전 폭 안이면 Shepherd 유지, 폭 밖이면 해제 대상으로 분류
        if abs(local_position.y) <= safe_half_width:
            kept_ids.add(robot_id)
        else:
            released_ids.add(robot_id)

    # 새 corridor 폭 밖에 있는 Shepherd의 역할을 NORMAL로 전환
    for robot_id in released_ids:
        robot = robots_by_id[robot_id]
        release_robot_to_normal(robot)

    # trimming 전후 Shepherd 수와 해제된 ID를 디버그 출력
    print(
        "[BacktrackCorridorTrim] "
        f"left={left_range:.2f} right={right_range:.2f} "
        f"safe_half_width={safe_half_width:.2f} "
        f"before={len(active_shepherd_ids)} kept={len(kept_ids)} "
        f"released={len(released_ids)} released_ids={sorted(released_ids)}"
    )

    # 계속 사용할 Shepherd와 NORMAL로 해제한 Shepherd ID를 각각 반환
    return kept_ids, released_ids


# “코너인가 + 어느 방향으로 꺾어야 하는가”를 판단해서 각도를 반환하는 함수 (직접 앵커를 움직이는 함수는 아님)
# Backtracking 중 Anchor가 Junction 방향으로 복귀할 때 정면 corridor가 끝나고
# 좌/우로 꺾어야 하는 corner에 도달했는지 LiDAR free gap을 이용해 판단한다.
def detect_backtrack_corner(
    scan: LidarScan,
) -> float | None:

    # 전방 LiDAR에서 현재 진행 가능한 free gap의 중심 방향 탐색
    target_gap_angle = find_branch_free_gap(scan)

    # 진행 가능한 free gap 자체가 없으면 corner 방향을 결정할 수 없으므로 종료
    if target_gap_angle is None:
        return None

    # 현재 Anchor 정면(0°)의 장애물까지 거리 측정
    front_range = lidar_range_at_local_angle(scan, 0.0)

    # 정면 진행 공간이 막혀 있으면서 free gap이 충분히 큰 좌/우 각도에 존재하면 corner로 판단
    corner_detected = (
        front_range < ANCHOR_TRAVERSABLE_RANGE
        and abs(target_gap_angle) >= BACKTRACK_CORNER_MIN_TURN_DEG
    )

    # 조건을 만족하지 않으면 아직 직선 corridor라고 판단
    if not corner_detected:
        return None

    # corner가 검출되면 Anchor가 회전해야 할 local free-gap 방향 반환
    return target_gap_angle


# "코너링 중 → 새 통로가 좌우에 제대로 잡힘 → Anchor 방향도 새 통로와 나란해짐 → 코너링 종료 → 다시 직선 Backtracking" 를 판단하는 함수
# Backtracking 코너를 돈 뒤 Anchor가 새로운 직선 corridor 방향으로 완전히 진입했는지 판단한다.
# 좌·우 LiDAR로 측정한 통로 폭이 기존 corridor 폭과 유사하고,
# 목표 free-gap 방향이 Anchor 정면과 충분히 정렬되었을 때 새 corridor에 진입했다고 판단한다.
def backtrack_corridor_reacquired(
    scan: LidarScan,
    target_gap_angle: float | None,
) -> bool:

    # 아직 코너에서 따라갈 free-gap 방향이 없으면 새 corridor 진입 여부를 판단하지 않음
    if target_gap_angle is None:
        return False

    # 현재 Anchor 기준 좌·우 벽까지의 LiDAR 거리 측정
    left_range, right_range = extract_lateral_wall_ranges(scan)

    # 좌우 벽이 정상적으로 측정되지 않았거나 LiDAR 최대 거리 부근이면 corridor로 인정하지 않음
    if (
        not math.isfinite(left_range)
        or not math.isfinite(right_range)
        or left_range >= scan.max_range - 1.0
        or right_range >= scan.max_range - 1.0
    ):
        return False

    # 좌측 벽 거리 + 우측 벽 거리로 현재 통로 폭 계산
    measured_width = left_range + right_range

    # 측정된 폭이 기존 corridor 폭의 75~125% 범위인지 확인
    corridor_width_valid = (
        0.75 * KNOWN_CORRIDOR_WIDTH
        <= measured_width
        <= 1.25 * KNOWN_CORRIDOR_WIDTH
    )

    # free gap 방향이 거의 정면(0°)으로 들어왔는지 확인하여 코너 회전이 끝났는지 판단
    heading_aligned = abs(target_gap_angle) <= BACKTRACK_CORNER_EXIT_ANGLE_DEG

    # 정상적인 corridor 폭 + Anchor heading 정렬을 모두 만족하면 새 직선 corridor 진입 완료
    return corridor_width_valid and heading_aligned


# Branch를 탐색하다가 앞에 실제로 접근 가능한 Marker가 보이는지 확인하는 함수 (옆 Branch의 Marker나 벽 너머 Marker를 잘못 인식하지 않도록 걸러냄)
# Anchor 전방에서 실제로 보이는 Marker를 탐색하고, 조건을 만족하는 Marker 중 가장 가까운 Marker ID를 반환한다.
# 현재 Branch 진입 시 통과하는 자기 Branch Marker는 제외하고, 현재 corridor 안·전방 시야각·탐지 거리·LiDAR line-of-sight를 모두 만족하는 Marker만 인정한다.
def detect_visible_marker_ahead(
    observation: LocalObservation,
    robots_by_id: dict[int, environment.Robot],
    lidar: AnchorLidar,
    ignore_marker_id: int | None,
) -> int | None:

    # 조건을 만족하는 Marker 중 가장 가까운 Marker를 찾기 위한 초기값
    nearest_marker_id: int | None = None
    nearest_distance = float("inf")

    # Marker 탐지 거리를 설정하되 LiDAR 최대 측정 거리를 넘지 않도록 제한
    detection_range = min(MARKER_DETECTION_RANGE, observation.lidar_scan.max_range)

    # Anchor-local 관측에 포함된 모든 로봇을 검사
    for robot_id, local_position in zip(observation.robot_ids, observation.relative_positions):

        # 현재 Branch 진입 시 통과하는 자기 Branch Marker는 탐지 대상에서 제외
        if robot_id == ignore_marker_id:
            continue

        robot = robots_by_id[robot_id]

        # MARKER 역할인 로봇만 탐지
        if robot.role != "MARKER":
            continue

        # Anchor 뒤쪽(x<=0)에 있는 Marker는 제외
        if local_position.x <= 0.0:
            continue

        # 현재 진행 중인 corridor 중앙 영역 밖의 Marker는 다른 Branch/Junction Marker로 보고 제외
        if abs(local_position.y) > 0.45 * KNOWN_CORRIDOR_WIDTH:
            continue

        # Anchor와 Marker 사이의 상대 거리 계산 후 탐지 범위를 벗어나면 제외
        distance = local_position.length()
        if distance > detection_range:
            continue

        # Marker가 Anchor 정면에서 어느 각도에 있는지 계산
        angle = math.degrees(math.atan2(local_position.y, local_position.x))

        # Marker가 Anchor의 Marker 탐지 시야각 밖에 있으면 제외
        if abs(angle) > MARKER_HALF_ANGLE_DEG:
            continue

        # Marker 방향의 LiDAR 벽 거리를 확인하여 Marker와 Anchor 사이에 벽이 있는지 검사
        wall_range = lidar_range_at_local_angle(lidar.last_scan, angle)

        # Marker보다 가까운 위치에 벽이 있으면 가려진 Marker이므로 제외
        if distance > wall_range + MARKER_LINE_OF_SIGHT_MARGIN:
            continue

        # 모든 조건을 만족한 Marker 중 Anchor와 가장 가까운 Marker를 저장
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_marker_id = robot_id

    # 실제로 보이는 가장 가까운 Marker ID 반환, 없으면 None
    return nearest_marker_id


# Marker ID를 이용해 해당 Marker가 어느 Branch에 속해 있는지 찾는 함수
def find_branch_id_by_marker(
    marker_id: int,
    branch_states: dict,
) -> str | None:

    # 현재 Junction에 등록된 모든 Branch 상태를 순회
    for branch_id, state in branch_states.items():

        # 해당 Branch가 가지고 있는 marker_id가 찾는 Marker ID와 같으면
        if state.get("marker_id") == marker_id:

            # 그 Marker가 속한 Branch ID 반환
            return branch_id

    # 일치하는 Marker가 없으면 None 반환
    return None


# Branch 탐색이 완료되었을 때 해당 Branch의 상태를 VISITED로 변경하고,
# 그 Branch에 Marker가 존재하면 Marker 상태도 함께 VISITED로 변경한다. (이미 탐색한 곳을 다시 선택하지 않도록 기록)
def mark_branch_visited(
    branch_id: str,
    branch_states: dict,
    reason: str,
) -> None:

    # 탐색이 완료된 Branch의 상태 정보 가져오기
    state = branch_states[branch_id]

    # 해당 Branch 자체를 탐색 완료(VISITED) 상태로 변경
    state["visit_state"] = "VISITED"

    # Branch에 Marker가 존재하면 Marker 역시 탐색 완료 상태로 변경
    if state.get("marker_id") is not None:
        state["marker_state"] = "VISITED"

    # 어떤 Branch가 어떤 이유로 VISITED 처리되었는지 로그 출력
    print("[BranchVisited] " f"branch={branch_id} " f"reason={reason}")


# 현재 Junction에 정의된 DFS 탐색 순서(branch_order)를 따라
# 아직 탐색하지 않은 UNVISITED Branch 중 가장 먼저 나오는 Branch를 선택한다.
# Branch의 실제 LEFT/RIGHT/UP 방향은 사용하지 않고 DFS에 저장된 순서만 사용한다.
def find_next_unvisited_branch(
    branch_order: list[str],
    branch_states: dict,
) -> str | None:

    # DFS에 저장된 Branch 탐색 순서를 앞에서부터 확인
    for branch_id in branch_order:

        # 아직 방문하지 않은 Branch를 발견하면 다음 탐색 Branch로 반환
        if branch_states[branch_id]["visit_state"] == "UNVISITED":
            return branch_id

    # 모든 Branch가 이미 탐색되었다면 더 이상 선택할 Branch가 없음
    return None


# 최종 Base 복귀를 위해 어느 Branch의 Shepherd들을 FINAL_PUSH에 사용할지 선택한다.
# 모든 Branch 탐색이 끝난 뒤 Anchor heading은 처음 Root Junction에서
# Branch geometry를 등록할 때 저장한 junction_reference_yaw_deg로 복원된다.
# 따라서 그 기준에서 center_angle이 0°에 가장 가까운 Branch를 선택하고,
# 이후 해당 Branch의 Initial Shepherd들을 FINAL_PUSH Shepherd로 사용한다.
def find_final_push_branch(branches: list[dict]) -> dict:

    # Junction 기준 정면 0°와 중심각 차이가 가장 작은 Branch 선택
    return min(
        branches,
        key=lambda branch: abs(normalize_angle(branch["center_angle"])),
    )


# 사각형 Junction 입구에서 Anchor의 360° LiDAR Opening 정보를 이용해
# 정면 Branch 입구의 양쪽 벽 모서리(corner)를 LiDAR range profile로부터 찾아 Anchor-local 좌표로 반환한다.
# 이후 Base 쪽 입구의 두 corner와 함께 Junction center를 추정하는 데 사용한다.
def extract_front_mouth_corners(lidar: AnchorLidar) -> tuple[pygame.Vector2, pygame.Vector2] | None:

    # Junction이라면 최소 LEFT / FRONT / RIGHT의 3개 Opening이 필요하다.
    if len(lidar.opening_groups) < 3:
        print("[FrontCornerFail] " f"reason=NOT_ENOUGH_OPENINGS " f"count={len(lidar.opening_groups)}")
        return None

    scan = lidar.last_scan

    # LiDAR range noise 때문에 잘못된 corner가 선택되는 것을 줄이기 위해 range profile을 smoothing한다.
    smoothed = smooth_ranges(scan.ranges, window_size=5)

    # =====================================================
    # 1. LEFT / FRONT / RIGHT Opening 구분
    # =====================================================

    # Anchor 정면 0°에 가장 가까운 Opening을 FRONT로 선택
    front_opening = min(
        lidar.opening_groups,
        key=lambda opening: abs(normalize_angle(opening["center_angle"])),
    )

    # -90°에 가장 가까운 Opening을 LEFT로 선택
    left_opening = min(
        lidar.opening_groups,
        key=lambda opening: circular_error(opening["center_angle"], -90.0),
    )

    # +90°에 가장 가까운 Opening을 RIGHT로 선택
    right_opening = min(
        lidar.opening_groups,
        key=lambda opening: circular_error(opening["center_angle"], 90.0),
    )

    # 같은 Opening이 두 방향으로 중복 선택되면 정상적인 L/F/R 구조로 볼 수 없으므로 실패
    if front_opening is left_opening or front_opening is right_opening or left_opening is right_opening:
        print("[FrontCornerFail] reason=DUPLICATED_OPENING")
        return None

    # 이후 corner 탐색에 사용할 각 Opening의 Anchor-local 중심각
    left_angle = normalize_angle(left_opening["center_angle"])
    front_angle = normalize_angle(front_opening["center_angle"])
    right_angle = normalize_angle(right_opening["center_angle"])

    # =====================================================
    # 2. 두 Opening 사이의 실제 Junction corner 탐색
    # =====================================================

    # 두 Opening 중심각 사이의 LiDAR ray 중 range가 가장 짧은 지점을 찾는다.
    # Opening 사이에는 Junction 벽 모서리가 존재하므로 가장 가까운 wall point를
    # 두 Branch 사이의 물리적인 Junction corner로 사용한다.
    def find_corner_between(angle_a: float, angle_b: float) -> tuple[pygame.Vector2, float, float] | None:

        lower = min(angle_a, angle_b)
        upper = max(angle_a, angle_b)

        # 두 Opening 중심각 사이에 포함되는 LiDAR ray만 선택
        indices = [
            index for index, angle in enumerate(scan.angles_deg)
            if lower <= angle <= upper
        ]

        if not indices:
            return None

        # 해당 angular interval에서 가장 가까운 wall point를 corner로 선택
        corner_index = min(indices, key=lambda index: smoothed[index])

        corner_angle = scan.angles_deg[corner_index]
        corner_range = smoothed[corner_index]
        radians = math.radians(corner_angle)

        # polar LiDAR 측정값 (angle, range)을 Anchor-local Cartesian 좌표로 변환
        corner = pygame.Vector2(
            corner_range * math.cos(radians),
            corner_range * math.sin(radians),
        )

        return corner, corner_angle, corner_range

    # LEFT Opening과 FRONT Opening 사이의 Junction corner
    left_result = find_corner_between(left_angle, front_angle)

    # FRONT Opening과 RIGHT Opening 사이의 Junction corner
    right_result = find_corner_between(front_angle, right_angle)

    # 양쪽 corner를 모두 찾지 못하면 FRONT mouth geometry를 만들 수 없음
    if left_result is None or right_result is None:
        print("[FrontCornerFail] reason=NO_CORNER")
        return None

    front_left, front_left_angle, front_left_range = left_result
    front_right, front_right_angle, front_right_range = right_result

    # local y값을 기준으로 두 corner의 순서를 일관되게 정렬
    if front_left.y > front_right.y:
        front_left, front_right = front_right, front_left
        front_left_angle, front_right_angle = front_right_angle, front_left_angle
        front_left_range, front_right_range = front_right_range, front_left_range

    # 검출된 Opening 방향과 실제 Junction corner 위치를 diagnostic으로 출력
    print(
        "[FrontMouthCorners] "
        f"opening_angles=L:{left_angle:.1f},F:{front_angle:.1f},R:{right_angle:.1f} "
        f"corner_angles=FL:{front_left_angle:.1f},FR:{front_right_angle:.1f} "
        f"FL=({front_left.x:.2f},{front_left.y:.2f}) "
        f"FR=({front_right.x:.2f},{front_right.y:.2f}) "
        f"ranges={front_left_range:.2f},{front_right_range:.2f}"
    )

    # FRONT Branch mouth 양쪽의 물리적 Junction corner를 Anchor-local 좌표로 반환
    return front_left, front_right


# 이미 탐색 순서가 정해졌으므로, 정해진 Branch 방향을 target_angle로 받아
# 실제 LiDAR가 검출한 Opening 중 해당 방향에 가장 가까운 Opening을 선택한다.
# target_angle과 ±60° 이내에 있는 Opening만 후보로 인정하고,
# 그중 target_angle과 중심각 차이가 가장 작은 Opening을 선택한다.
def get_lateral_opening(lidar: AnchorLidar, target_angle: float) -> dict | None:

    # target_angle 주변 ±60° 범위에 들어오는 Opening만 후보로 선택
    candidates = [
        opening for opening in lidar.opening_groups
        if abs(normalize_angle(opening["center_angle"] - target_angle)) <= 60.0
    ]

    # 해당 방향에서 Opening을 찾지 못하면 None 반환
    if not candidates:
        return None

    # 후보 중 target_angle과 중심각이 가장 가까운 Opening을 최종 선택
    return min(
        candidates,
        key=lambda opening: abs(normalize_angle(opening["center_angle"] - target_angle)),
    )


# 정지한 Anchor의 좌·우 Opening geometry를 이용해
# Anchor가 Junction으로 들어온 Base 쪽 입구의 좌·우 corner를 추출한다.
# 추출한 두 corner는 FRONT 쪽 corner와 함께 Junction center를 추정하는 데 사용한다.
def extract_base_entrance_corners(lidar: AnchorLidar) -> tuple[pygame.Vector2, pygame.Vector2] | None:

    # 정해진 좌(-90°), 우(+90°) 방향에 가장 가까운 lateral Opening을 각각 선택
    left_opening = get_lateral_opening(lidar, -90.0)
    right_opening = get_lateral_opening(lidar, 90.0)

    # 좌·우 Opening 중 하나라도 찾지 못하면 Base 입구 corner를 결정할 수 없음
    if left_opening is None or right_opening is None:
        return None

    # LEFT Opening의 두 mouth 끝점 중 local x가 더 작은 점을 Base 입구의 한쪽 corner로 선택
    base_left = min(
        (left_opening["mouth_a"], left_opening["mouth_b"]),
        key=lambda point: point.x,
    ).copy()

    # RIGHT Opening에서도 local x가 더 작은 mouth 끝점을 반대쪽 Base corner로 선택
    base_right = min(
        (right_opening["mouth_a"], right_opening["mouth_b"]),
        key=lambda point: point.x,
    ).copy()

    # local y 기준으로 두 점의 좌·우 순서를 일관되게 정렬
    if base_left.y > base_right.y:
        base_left, base_right = base_right, base_left

    # 두 corner 사이 거리 = 검출된 Base 입구 폭
    base_width = (base_right - base_left).length()

    # 두 corner의 중점 = Base 입구의 중심 위치
    base_midpoint = (base_left + base_right) * 0.5

    # 검출된 입구 폭이 알려진 corridor width와 지나치게 다르면 잘못된 corner로 판단
    if not (KNOWN_CORRIDOR_WIDTH * 0.65 <= base_width <= KNOWN_CORRIDOR_WIDTH * 1.35):
        print("[RejectBaseCorners] " f"reason=width width={base_width:.2f}")
        return None

    # Base 입구의 중점이 Anchor-local x=0 부근에 있어야 하므로
    # 너무 앞/뒤로 떨어져 있으면 잘못된 Base corner 조합으로 판단
    if abs(base_midpoint.x) > 8.0:
        print("[RejectBaseCorners] " f"reason=depth mid_x={base_midpoint.x:.2f}")
        return None

    # 검증을 통과한 Base 쪽 입구의 두 corner 반환
    return base_left, base_right


# 두 대각선 선분의 실제 교차점을 계산한다.
# 두 선분이 평행하거나, 교차점이 두 선분의 범위 밖에 있으면 None을 반환한다.
# Junction의 네 corner로 만든 두 대각선의 교차점을 Junction center로 구할 때 사용한다.
def diagonal_segment_intersection(
    first_start: pygame.Vector2,
    first_end: pygame.Vector2,
    second_start: pygame.Vector2,
    second_end: pygame.Vector2,
) -> pygame.Vector2 | None:

    # 각 대각선의 시작점→끝점 방향 벡터
    first_direction = first_end - first_start
    second_direction = second_end - second_start

    # 두 방향 벡터의 cross product로 평행 여부 확인
    determinant = first_direction.cross(second_direction)

    # cross product가 거의 0이면 두 대각선이 평행하므로 교차점 계산 불가
    if abs(determinant) <= environment.EPSILON:
        return None

    # 두 대각선 시작점 사이의 상대 벡터
    offset = second_start - first_start

    # 각 선분에서 교차점이 시작점으로부터 어느 비율(t)에 위치하는지 계산
    first_t = offset.cross(second_direction) / determinant
    second_t = offset.cross(first_direction) / determinant

    # t가 0~1 범위를 벗어나면 무한 직선끼리는 만나더라도
    # 실제 두 대각선 '선분' 내부에서 교차하는 것은 아니므로 제외
    if not (0.0 <= first_t <= 1.0 and 0.0 <= second_t <= 1.0):
        return None

    # 첫 번째 대각선을 따라 교차점의 실제 좌표 계산
    return first_start + first_direction * first_t


# Base 쪽 2개 corner와 Front 쪽 2개 corner로 Junction의 두 대각선을 만들고,
# 두 대각선의 교차점을 Anchor-local 좌표계의 Junction center로 추정한다.
def estimate_local_junction_center(
    base_left: pygame.Vector2,
    base_right: pygame.Vector2,
    front_left: pygame.Vector2,
    front_right: pygame.Vector2,
) -> pygame.Vector2 | None:

    # BL→FR 대각선과 BR→FL 대각선의 교차점을 계산
    # 사각형 Junction에서는 이 두 대각선의 교차점을 Junction center로 사용
    center = diagonal_segment_intersection(
        base_left,
        front_right,
        base_right,
        front_left,
    )

    # 두 대각선이 실제 선분 범위 안에서 만나지 않으면
    # 정상적인 Junction center를 추정할 수 없으므로 실패
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

    # 검출된 네 corner와 최종 Junction center를 diagnostic으로 출력
    print(
        "[JunctionDiagonalGeometry] "
        f"BL=({base_left.x:.2f},{base_left.y:.2f}) "
        f"BR=({base_right.x:.2f},{base_right.y:.2f}) "
        f"FL=({front_left.x:.2f},{front_left.y:.2f}) "
        f"FR=({front_right.x:.2f},{front_right.y:.2f}) "
        f"C=({center.x:.2f},{center.y:.2f})"
    )

    # Anchor에서 바라본 Junction center의 local 좌표 반환
    return center


# 처음 계산한 Junction Center까지 가려면 지금부터 얼마나 더 이동해야 하는가? 를 계산
# 정지 LiDAR scan으로 미리 계산하여 고정한 Junction center까지
# Anchor의 누적 local 이동량(odometry)만 이용하여 Anchor를 이동시킨다.
# 현재 위치를 다시 추정하지 않고, 목표 local 좌표 - 누적 이동량으로 남은 이동 벡터를 계산한다.
def move_anchor_to_locked_center(
    anchor: environment.Robot,
    locked_target_local: pygame.Vector2,
    traveled_local: pygame.Vector2,
    yaw_degrees: float,
    dt: float,
) -> tuple[bool, pygame.Vector2]:

    # 지금까지의 누적 이동량을 빼서 Junction center까지 얼마나 더 이동해야 하는지 계산
    remaining = locked_target_local - traveled_local

    print(
        "[AnchorCenterTransit] "
        f"target=({locked_target_local.x:.2f},{locked_target_local.y:.2f}) "
        f"travel=({traveled_local.x:.2f},{traveled_local.y:.2f}) "
        f"remaining=({remaining.x:.2f},{remaining.y:.2f})"
    )

    # 목표 center와의 남은 거리가 허용 오차 이내이면 Anchor를 정지시키고 이동 완료
    if remaining.length() <= ANCHOR_LOCAL_CENTER_TOLERANCE:
        stop_anchor(anchor)
        return True, pygame.Vector2()

    # 남은 전후방 오차(remaining.x)에 비례해 전진/후진 속도를 계산하고 최대 속도로 제한
    forward_speed = max(
        -ANCHOR_CENTERING_SPEED,
        min(ANCHOR_CENTERING_SPEED, ANCHOR_LOCAL_CENTER_GAIN * remaining.x),
    )

    # 남은 좌우 오차(remaining.y)에 비례해 lateral 속도를 계산하고 최대 속도로 제한
    lateral_speed = max(
        -ANCHOR_CENTER_MAX_LATERAL_SPEED,
        min(ANCHOR_CENTER_MAX_LATERAL_SPEED, ANCHOR_LOCAL_CENTER_GAIN * remaining.y),
    )

    # 계산된 local 속도로 Anchor를 실제 이동시키고 이번 step의 실제 local 이동량을 반환
    local_delta = integrate_anchor_local_command(
        anchor,
        pygame.Vector2(forward_speed, lateral_speed),
        yaw_degrees,
        dt,
    )

    # 아직 center에 도착하지 않았으므로 False와 이번 step의 local 이동량 반환
    return False, local_delta


# 같은 방향을 나타내는 여러 각도 표현을 -180° ~ +180° 범위 하나로 통일하는 함수
# 입력된 각도를 -180° ~ +180° 범위로 정규화한다.
# LiDAR Opening 방향이나 Anchor 회전 방향의 각도 차이를 일관되게 비교할 때 사용한다.
def normalize_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


# 특정 각도가 Branch/Opening의 angular sector 안에 들어오는지 검사하는 함수
# 주어진 angle이 start~end로 정의된 각도 구간(sector) 안에 포함되는지 판단한다.
# 0°/360° 경계를 넘어가는 sector도 올바르게 처리한다.
def angle_inside_sector(angle: float, start: float, end: float) -> bool:

    # 비교하기 쉽도록 모든 각도를 0°~360° 범위로 변환
    angle = angle % 360.0
    start = start % 360.0
    end = end % 360.0

    # 일반적인 sector: start → end가 0° 경계를 지나지 않는 경우
    if start <= end:
        return start <= angle <= end

    # 0°/360° 경계를 넘어가는 sector는
    # start 이상이거나 end 이하이면 sector 내부로 판단
    return angle >= start or angle <= end


# Anchor에서 특정 방향으로 ray를 쐈을 때
# Branch 입구와 만나는 지점까지의 거리를 계산한다.
def ray_to_branch_mouth_range(
    direction: pygame.Vector2,
    mouth_a: pygame.Vector2,
    mouth_b: pygame.Vector2,
) -> float | None:

    # Branch mouth의 한쪽 끝점에서 다른 끝점으로 향하는 선분 벡터
    mouth_vector = mouth_b - mouth_a

    # ray 방향과 mouth 선분의 cross product 계산
    # 두 벡터가 평행하면 교차점을 계산할 수 없음
    denominator = direction.cross(mouth_vector)

    if abs(denominator) <= environment.EPSILON:
        return None

    # Anchor 원점에서 direction을 따라 얼마만큼 이동하면
    # mouth 선분과 만나는지 나타내는 ray상의 거리
    ray_range = mouth_a.cross(mouth_vector) / denominator

    # 교차점이 mouth_a→mouth_b 선분의 어느 위치에 있는지 나타내는 비율
    # 0 = mouth_a, 1 = mouth_b, 0~1 = 실제 mouth 선분 내부
    segment_ratio = mouth_a.cross(direction) / denominator

    # 교차점이 Anchor 뒤쪽에 있으면 유효한 ray 교차가 아니므로 제외
    if ray_range < 0.0:
        return None

    # 무한 직선끼리는 만나더라도 실제 Branch mouth 선분 밖에서 만나면 제외
    if not (0.0 <= segment_ratio <= 1.0):
        return None

    # Anchor에서 Branch mouth와의 교차점까지의 local range 반환
    return float(ray_range)


# 로봇이 특정 Branch 입구 영역에 들어왔는지 판단하여 Initial Shepherd 후보로 선택한다.
# Branch 입구 폭 안에 있으면서 입구보다 너무 뒤쪽에 있지 않으면 True를 반환한다.
# 한 번 입구를 지나 Branch 안쪽으로 깊게 들어간 로봇도 계속 후보로 인정한다.
# ① 입구 폭 안에 있는가(lateral) → ② 입구에 도달했거나 안쪽으로 들어갔는가(depth) → 둘 다 만족하면 Initial Shepherd 후보
def robot_is_at_branch_entrance(local_position: pygame.Vector2, branch: dict) -> bool:

    # Branch 입구의 중심점
    entrance_midpoint = branch["entrance_midpoint"]

    # Branch 안쪽으로 향하는 진행 방향
    axis = branch["branch_axis"].normalize()

    # Branch 입구를 가로지르는 좌우 방향
    tangent = branch["entrance_tangent"].normalize()

    # Branch 입구의 실제 폭과 절반 폭
    entrance_width = (branch["entrance_b"] - branch["entrance_a"]).length()
    half_width = 0.5 * entrance_width

    radius = environment.ROBOT_RADIUS

    # 입구 중심을 기준으로 현재 로봇이 어디에 있는지 나타내는 상대 벡터
    offset = local_position - entrance_midpoint

    # 로봇이 입구에서 Branch 안쪽 방향으로 얼마나 들어갔는지
    depth = offset.dot(axis)

    # 로봇이 입구 중심에서 좌우로 얼마나 벗어나 있는지
    lateral = offset.dot(tangent)

    # 로봇이 Branch 입구 폭 안에 들어와 있는지 검사
    inside_width = abs(lateral) <= half_width + radius + ENTRANCE_CAPTURE_MARGIN

    if not inside_width:
        return False

    # 입구보다 너무 뒤쪽에 있는 로봇은 제외한다.
    # 허용 경계를 통과한 이후에는 Branch 안쪽으로 아무리 깊게 들어가도 후보로 유지한다.
    return depth >= -(radius + ENTRANCE_DEPTH_CAPTURE_MARGIN)


# Anchor에서 로봇 방향으로 LiDAR를 쐈을 때, 벽을 만나기 전에 로봇이 있는가?”를 검사
# 특정 로봇과 Anchor 사이에 벽이 없는지 확인하여 직접 LOS(Line of Sight)가 확보되는지 판단한다.
# 로봇의 Anchor-relative 위치와 해당 방향의 LiDAR wall range만 사용하며 절대좌표는 사용하지 않는다.
def robot_has_direct_anchor_los(robot_id: int, observation: LocalObservation) -> bool:

    # observation에서 해당 robot의 Anchor-relative local 위치를 찾는다.
    local_position = next(
        (position for rid, position in zip(observation.robot_ids, observation.relative_positions) if rid == robot_id),
        None,
    )

    # 해당 로봇의 local 위치 정보가 없으면 LOS를 판단할 수 없음
    if local_position is None:
        return False

    # Anchor에서 로봇까지의 직선거리
    distance = local_position.length()

    if distance <= environment.EPSILON:
        return False

    # Anchor에서 봤을 때 로봇이 위치한 방향(bearing)을 각도로 계산
    bearing_deg = math.degrees(math.atan2(local_position.y, local_position.x))

    # 로봇이 있는 방향으로 Anchor LiDAR를 봤을 때 가장 가까운 벽까지의 거리
    wall_range = lidar_range_at_local_angle(observation.lidar_scan, bearing_deg)

    # Anchor→robot 거리보다 Anchor→wall 거리가 더 길면 로봇보다 앞에 벽이 없으므로 직접 LOS 확보
    # 반대로 wall이 robot보다 먼저 나타나면 벽에 가려져 있으므로 LOS가 차단된 것으로 판단
    visible = distance <= wall_range + environment.ROBOT_RADIUS

    return visible


# Anchor가 Junction을 확인한 뒤 JUNCTION_CONFIRMED 메시지를 broadcast했을 때, 그 메시지를 Anchor로부터 직접 받을 수 있는 NORMAL 로봇들을 찾는 함수
# Anchor가 Junction 확인 메시지(JUNCTION_CONFIRMED)를 broadcast했을 때,
# Anchor의 직접 통신거리(COMM_RANGE) 안에 있는 NORMAL 로봇들의 ID를 반환한다.
# 여기서는 통신거리만 검사하며, 실제 Shepherd 선정 시 필요한 LOS는 이후 별도로 검사한다.
def update_junction_broadcast_receivers(
    observation: LocalObservation,
    robots_by_id: dict[int, environment.Robot],
) -> set[int]:

    # Junction broadcast를 직접 수신한 로봇 ID를 저장
    received_ids: set[int] = set()

    # Anchor-relative 관측에 포함된 모든 로봇을 검사
    for robot_id, local_position in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):

        # LiDAR Anchor 자신은 broadcast 수신 대상에서 제외
        if robot_id == observation.lidar_robot_id:
            continue

        robot = robots_by_id[robot_id]

        # 현재 NORMAL 역할인 로봇만 수신 후보로 사용
        if robot.role != "NORMAL":
            continue

        # Anchor와 로봇 사이의 상대거리가 직접 통신거리보다 크면 수신 불가
        if local_position.length() > environment.COMM_RANGE:
            continue

        # NORMAL이면서 Anchor의 직접 통신거리 안에 있으면 broadcast 수신 로봇으로 등록
        received_ids.add(robot_id)

    # JUNCTION_CONFIRMED를 Anchor로부터 직접 수신할 수 있는 NORMAL 로봇 ID 반환
    return received_ids



# [Initial Shepherd 부분_1]
# "어떤 로봇들을 multi-hop으로 모집할 것인가?" -> INITIAL_SHEPHERD_HOPS에 설정된 값만큼 확장
# 각 Branch에서 가장 안쪽까지 진입한 후보 로봇 1대를 Hop 0 seed로 선택하고,
# 그 로봇을 시작점으로 robot-to-robot COMM_RANGE 연결을 따라
# Hop 1 → Hop 2 → ... → INITIAL_SHEPHERD_HOPS까지 통신 영역을 확장한다.
# 이렇게 연결된 로봇들을 해당 Branch의 Initial Shepherd 모집 영역으로 반환한다.
# 모든 계산은 Anchor-relative local position을 사용하며 절대좌표는 사용하지 않는다.
def collect_initial_shepherd_hop_region(
    observation: LocalObservation,
    branch: dict,
    candidate_ids: set[int],
) -> tuple[set[int], list[set[int]]]:

    # observation에서 각 로봇의 Anchor-relative local 위치를 ID별로 저장
    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(observation.robot_ids, observation.relative_positions)
    }

    # Branch 입구 중심, Branch 진행 방향(axis), 입구 가로 방향(tangent)
    entrance_midpoint = branch["entrance_midpoint"]
    axis = branch["branch_axis"].normalize()
    tangent = branch["entrance_tangent"].normalize()

    # =================================================
    # 1. 각 후보 로봇의 Branch-local depth / lateral 계산
    # =================================================

    depth_by_id: dict[int, float] = {}
    lateral_by_id: dict[int, float] = {}

    for robot_id in candidate_ids:
        position = local_position_by_id.get(robot_id)

        if position is None:
            continue

        offset = position - entrance_midpoint

        # Branch 입구에서 안쪽으로 얼마나 깊이 들어갔는지
        depth_by_id[robot_id] = offset.dot(axis)

        # Branch centerline에서 좌우로 얼마나 떨어져 있는지
        lateral_by_id[robot_id] = offset.dot(tangent)

    # 사용할 수 있는 후보 로봇이 없으면 빈 Hop region 반환
    if not depth_by_id:
        return set(), [set()]

    # =================================================
    # 2. Hop 0 seed 선택
    # =================================================

    # Branch 안쪽으로 가장 깊이 들어간 로봇을 Hop 0 seed로 선택한다.
    # depth가 같으면 Branch centerline에 더 가까운 로봇을 우선한다.
    seed_id = max(
        depth_by_id,
        key=lambda robot_id: (
            depth_by_id[robot_id],
            -abs(lateral_by_id[robot_id]),
        ),
    )

    # Hop 0에는 seed robot 한 대만 존재
    layers: list[set[int]] = [{seed_id}]
    visited = {seed_id}

    # =================================================
    # 3. Hop 1 → Hop 2 → ... 통신 graph 확장
    # =================================================

    # 이전 Hop의 로봇들과 COMM_RANGE 이내에 있는 새로운 후보 로봇들을
    # 다음 Hop으로 추가하여 robot-to-robot communication chain을 확장한다.
    for _hop in range(1, INITIAL_SHEPHERD_HOPS + 1):

        previous_layer = layers[-1]
        next_layer: set[int] = set()

        for source_id in previous_layer:
            source_position = local_position_by_id[source_id]

            for candidate_id in candidate_ids:

                # 이미 이전 Hop에서 포함된 로봇은 중복 추가하지 않음
                if candidate_id in visited:
                    continue

                candidate_position = local_position_by_id.get(candidate_id)

                if candidate_position is None:
                    continue

                # source robot과 직접 통신할 수 없는 후보는 현재 Hop에서 제외
                if source_position.distance_to(candidate_position) > environment.COMM_RANGE:
                    continue

                # COMM_RANGE 이내이면 다음 Hop의 통신 연결 로봇으로 추가
                next_layer.add(candidate_id)

        layers.append(next_layer)
        visited.update(next_layer)

        # 더 이상 연결되는 새로운 로봇이 없으면 Hop 확장을 조기 종료
        if not next_layer:
            break

    # Hop 0부터 마지막 Hop까지 포함된 모든 로봇 ID를 하나의 집합으로 통합
    hop_region_ids = set().union(*layers)

    # 전체 multi-hop 영역과 Hop별 로봇 집합을 함께 반환
    return hop_region_ids, layers


# def branch_mouth_is_fully_covered(
#     branch: dict,
#     robot_local_positions: list[pygame.Vector2],
# ) -> tuple[bool, float]:

#     # Branch 입구 양 끝점, 중심점, 입구 가로 방향, Branch 진행 방향
#     entrance_a = branch["entrance_a"]
#     entrance_b = branch["entrance_b"]
#     entrance_midpoint = branch["entrance_midpoint"]
#     tangent = branch["entrance_tangent"].normalize()
#     axis = branch["branch_axis"].normalize()

#     # 실제 Branch 입구 전체 폭
#     entrance_width = (entrance_b - entrance_a).length()

#     # 정상적인 입구 폭을 만들 수 없으면 sealing 실패
#     if entrance_width <= environment.EPSILON:
#         return False, float("inf")

#     radius = environment.ROBOT_RADIUS

#     # 입구에서 Branch 안쪽 약 4열 깊이까지만 실제 sealing 판정에 사용
#     seal_depth = 4.0 * environment.GRID_SPACING

#     # 로봇 한 대가 입구의 좌우 방향으로 덮는다고 인정할 범위
#     coverage_half_span = max(radius, 0.5 * environment.GRID_SPACING)

#     # 각 로봇이 Branch 입구 폭에서 덮고 있는 구간을 저장
#     intervals: list[tuple[float, float]] = []

#     for position in robot_local_positions:

#         # 입구 중심 기준 로봇의 상대 위치
#         offset = position - entrance_midpoint

#         # 로봇이 입구 기준 Branch 진행 방향으로 얼마나 들어가 있는지 계산
#         depth = offset.dot(axis)

#         # 입구보다 너무 뒤에 있거나 Branch 안쪽으로 너무 깊이 들어간 로봇은
#         # Shepherd 후보일 수는 있어도 실제 입구 sealing 판정에서는 제외
#         if depth < -(radius + ENTRANCE_CAPTURE_MARGIN):
#             continue
#         if depth > seal_depth:
#             continue

#         # entrance_a를 0으로 두었을 때 입구 가로 방향에서 로봇의 위치
#         lateral = (position - entrance_a).dot(tangent)

#         # 해당 로봇이 입구 폭에서 실제로 덮는 구간 [left, right] 계산
#         left = max(0.0, lateral - coverage_half_span)
#         right = min(entrance_width, lateral + coverage_half_span)

#         if right >= left:
#             intervals.append((left, right))

#     # 입구 근처에서 입구를 덮는 로봇이 하나도 없으면 sealing 실패
#     if not intervals:
#         return False, entrance_width

#     # 입구 왼쪽부터 coverage 구간을 순서대로 검사하기 위해 정렬
#     intervals.sort()

#     covered_until = 0.0
#     max_gap = 0.0

#     # 로봇들의 coverage 구간을 이어 붙이면서 가장 큰 빈틈을 계산
#     for left, right in intervals:
#         if left > covered_until:
#             max_gap = max(max_gap, left - covered_until)

#         covered_until = max(covered_until, right)

#     # 마지막 로봇 coverage 이후 입구 끝까지 남은 빈틈도 검사
#     if covered_until < entrance_width:
#         max_gap = max(max_gap, entrance_width - covered_until)

#     # 이 크기 이하의 작은 빈틈은 물리적으로 막힌 것으로 허용
#     gap_tolerance = 0.25 * environment.GRID_SPACING

#     # 가장 큰 빈틈이 허용 오차 이하이면 Branch 입구 전체가 막혔다고 판단
#     sealed = max_gap <= gap_tolerance

#     # 입구 봉쇄 여부와 실제 가장 큰 빈틈 크기를 반환
#     return sealed, max_gap


# [Initial Shepherd 부분_2]
# Multi-hop으로 형성된 특정 Hop layer의 로봇들이 Branch 입구 전체 폭을 충분히 덮고 있는지 검사한다.
# 해당 layer 로봇들의 lateral coverage를 합친 뒤 가장 큰 빈틈(max_gap)을 계산하고,
# 그 빈틈이 로봇 직경 이하이면 해당 Hop layer가 입구를 충분히 덮었다고 판단한다.
def shepherd_hop_layer_is_fully_covered(
    observation: LocalObservation,
    branch: dict,
    layer_ids: set[int],
) -> tuple[bool, float]:

    # 해당 Hop layer에 로봇이 없으면 coverage 실패
    if not layer_ids:
        return False, float("inf")

    # 각 로봇의 Anchor-relative local 위치를 ID별로 저장
    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(observation.robot_ids, observation.relative_positions)
    }

    # Branch 입구 중심, 입구 가로 방향, 실제 입구 폭
    entrance_midpoint = branch["entrance_midpoint"]
    tangent = branch["entrance_tangent"].normalize()
    entrance_width = (branch["entrance_b"] - branch["entrance_a"]).length()
    half_width = 0.5 * entrance_width

    # 로봇 한 대가 입구 가로 방향에서 덮는 것으로 인정할 반경
    coverage_half_span = max(
        environment.ROBOT_RADIUS,
        WALL_CONTACT_RADIUS,
    )

    intervals = []

    # 현재 Hop layer에 속한 로봇들만 검사
    for robot_id in layer_ids:

        position = local_position_by_id.get(robot_id)

        if position is None:
            continue

        # 입구 중심 기준으로 로봇의 좌우 위치를 계산
        lateral = (position - entrance_midpoint).dot(tangent)

        # midpoint 기준 좌표를 입구 시작점 0 ~ entrance_width 범위로 변환
        center = lateral + half_width

        # 해당 로봇이 입구 폭에서 덮는 구간 계산
        left = max(0.0, center - coverage_half_span)
        right = min(entrance_width, center + coverage_half_span)

        if right >= left:
            intervals.append((left, right))

    # 해당 Hop layer가 입구를 덮는 구간이 하나도 없으면 실패
    if not intervals:
        return False, entrance_width

    intervals.sort()

    covered_until = 0.0
    max_gap = 0.0

    # 각 로봇의 coverage를 연결하면서 입구에 남아 있는 가장 큰 빈틈 계산
    for left, right in intervals:
        if left > covered_until:
            max_gap = max(max_gap, left - covered_until)

        covered_until = max(covered_until, right)

    # 마지막 coverage 이후 입구 끝까지 남은 빈틈도 검사
    if covered_until < entrance_width:
        max_gap = max(max_gap, entrance_width - covered_until)

    # 최대 빈틈이 로봇 직경 이하이면 해당 Hop layer가 입구 폭을 충분히 덮은 것으로 인정
    gap_tolerance = 2.0 * environment.ROBOT_RADIUS
    covered = max_gap <= gap_tolerance

    # 해당 Hop layer의 coverage 성공 여부와 가장 큰 빈틈 반환
    return covered, max_gap



# =========================================================
# Initial Shepherd Formation - 함수 관계 및 전체 알고리즘
#
# 1. form_initial_junction_shepherd_boundaries()
#    → Initial Shepherd 형성 전체 과정을 관리하는 상위 함수
#    → 각 Branch entrance로 자연스럽게 유입된 NORMAL 로봇들을 candidate_ids로 수집
#
# 2. collect_initial_shepherd_hop_region()
#    → candidate 중 Branch 안쪽으로 가장 깊이 진입한 1대를 Hop 0 seed로 선택
#    → robot-to-robot COMM_RANGE 연결을 따라 Hop 1 → Hop 2 → ... → 설정된 최대 Hop까지 확장
#    → 각 Hop에는 여러 로봇이 포함될 수 있으며, 모든 Hop의 합집합을 hop_region_ids로 반환
#    → 최대 Hop 수는 고정되며, 입구가 안 막혔다고 Hop 4, Hop 5로 계속 증가하지 않음
#
# 3. shepherd_hop_layer_is_fully_covered()
#    → 현재 구현에서는 개별 Hop 하나가 아니라 hop_region_ids 전체를 입력받음
#    → 전체 multi-hop cohort의 lateral coverage를 계산하여 Branch 입구의 최대 빈틈(max_gap)을 검사
#    → 입구가 충분히 막혔으면 cohort_covered=True
#
# 4. 아직 입구가 막히지 않은 경우
#    → Initial Shepherd를 확정하지 않고 다음 frame까지 기다림
#    → 그동안 새로운 NORMAL 로봇이 Branch entrance로 유입될 수 있음
#    → 다음 frame에서 candidate와 Hop 0~최대 Hop cohort를 다시 구성하여 coverage 재검사
#    → 즉 Hop 수를 늘리는 것이 아니라, 같은 Hop 범위 안에 포함되는 로봇 수가 증가할 수 있음
#
# 5. 입구가 충분히 막힌 경우
#    → hop_region_ids 전체를 Initial Shepherd로 확정
#    → NORMAL → SHEPHERD 역할 변경
#    → 현재 위치에서 freeze
#    → state["initial_sealed"] = True
#
# 전체 흐름:
# Branch entrance로 NORMAL 유입
# → candidate_ids 수집
# → Hop 0 seed 선정
# → COMM_RANGE 기반 multi-hop cohort 구성
# → hop_region_ids의 physical coverage 검사
# → 미봉쇄: 다음 frame에서 로봇 유입을 기다린 뒤 다시 검사
# → 봉쇄: hop_region_ids 전체를 Initial Shepherd로 확정 및 freeze
#
# 핵심:
# Hop = Initial Shepherd 후보를 구성하는 robot-to-robot 통신 범위
# Coverage = 그 후보들이 실제 Branch 입구를 물리적으로 막았는지 확인하는 조건
# Hop 수는 고정되지만 각 Hop에 포함되는 로봇 수는 여러 대이며,
# 입구가 물리적으로 봉쇄될 때까지 새로운 로봇의 유입과 coverage 검사를 반복한다.
# =========================================================
def form_initial_junction_shepherd_boundaries(
    observation: LocalObservation,
    robots_by_id: dict[int, environment.Robot],
    branches: list[dict],
    branch_states: dict,
    junction_message_received_ids: set[int],
) -> bool:

    # 각 로봇의 Anchor-relative local 위치를 ID별로 저장
    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position in zip(observation.robot_ids, observation.relative_positions)
    }

    # 각 Branch의 Initial Shepherd boundary를 독립적으로 형성
    for branch in branches:
        branch_id = branch["id"]
        state = branch_states[branch_id]

        # 이미 Initial Shepherd wall이 완성된 Branch는 다시 처리하지 않는다.
        if state["initial_sealed"]:
            continue

        # =================================================
        # 1. Branch entrance에 진입한 NORMAL 후보 수집
        # =================================================
        current_ids: set[int] = set()

        for robot_id, local_position in zip(observation.robot_ids, observation.relative_positions):

            # Anchor 자신은 후보에서 제외
            if robot_id == observation.lidar_robot_id:
                continue

            robot = robots_by_id[robot_id]

            # Initial Shepherd 형성 전에는 NORMAL 로봇만 후보로 사용
            if robot.role != "NORMAL":
                continue

            # Branch-local 위치 및 통신/LOS 상태 확인
            entrance_midpoint = branch["entrance_midpoint"]
            axis = branch["branch_axis"].normalize()
            tangent = branch["entrance_tangent"].normalize()

            offset = local_position - entrance_midpoint
            depth = offset.dot(axis)          # Branch 안쪽으로 진입한 깊이
            lateral = offset.dot(tangent)     # Branch centerline 기준 좌우 위치

            entrance_width = (branch["entrance_b"] - branch["entrance_a"]).length()
            half_width = 0.5 * entrance_width

            print(
                "[InitialShepherdCandidateDebug] "
                f"id={robot_id} branch={branch_id} "
                f"depth={depth:.2f} lateral={lateral:.2f} "
                f"limit={half_width + environment.ROBOT_RADIUS + ENTRANCE_CAPTURE_MARGIN:.2f} "
                f"received={robot_id in junction_message_received_ids} "
                f"los={robot_has_direct_anchor_los(robot_id, observation)}"
            )

            # 실제 Branch entrance 영역에 진입한 로봇만 후보로 인정
            if not robot_is_at_branch_entrance(local_position, branch):
                continue

            current_ids.add(robot_id)

        # 현재 Branch entrance에 존재하는 NORMAL 로봇들을 multi-hop 모집 후보로 사용
        candidate_ids = set(current_ids)

        candidate_positions = [
            local_position_by_id[robot_id]
            for robot_id in candidate_ids
            if robot_id in local_position_by_id
        ]

        # =================================================
        # 2. Multi-hop Initial Shepherd cohort 구성
        #
        # 가장 깊이 진입한 로봇 = Hop 0
        # → COMM_RANGE를 따라 Hop 1 → Hop 2 → ... 확장
        # hop_layers = Hop별 로봇 집합
        # hop_region_ids = 모든 Hop을 합친 전체 cohort
        # =================================================
        hop_region_ids, hop_layers = collect_initial_shepherd_hop_region(
            observation, branch, candidate_ids
        )

        # Hop 0을 포함하므로 필요한 전체 layer 수 = 최대 Hop 수 + 1
        required_layer_count = INITIAL_SHEPHERD_HOPS + 1

        # 설정된 Hop까지 모든 layer가 실제로 존재하는지 확인
        required_hop_ready = (
            len(hop_layers) >= required_layer_count
            and all(len(hop_layers[hop_index]) > 0 for hop_index in range(required_layer_count))
        )

        # DEBUG용 Hop별 로봇 수
        hop_counts = [len(layer) for layer in hop_layers]

        # =================================================
        # 3. Multi-hop cohort의 최종 물리적 봉쇄 확인
        #
        # 통신상 연결만으로는 입구 봉쇄를 보장할 수 없으므로,
        # 전체 cohort의 lateral coverage를 이용해 Branch 입구의 최대 빈틈(max_gap)을 검사한다.
        # =================================================
        cohort_covered, max_gap = shepherd_hop_layer_is_fully_covered(
            observation, branch, hop_region_ids
        )

        # 현재 구현의 최종 봉쇄 조건은 전체 cohort의 physical coverage
        sealed = cohort_covered

        role_debug(
            f"initial-shepherd-{branch_id}",
            (
                "[InitialShepherdCoverage] "
                f"branch={branch_id} candidate_count={len(candidate_ids)} "
                f"hop_counts={hop_counts} cohort_count={len(hop_region_ids)} "
                f"cohort_covered={cohort_covered} max_gap={max_gap:.3f} sealed={sealed}"
            ),
        )

        # 아직 입구가 충분히 봉쇄되지 않았으면 다음 frame에서 다시 검사
        if not sealed:
            continue

        # =================================================
        # 4. Initial Shepherd 확정
        #
        # Physical coverage가 만족된 순간 전체 multi-hop cohort를
        # Initial Shepherd로 확정하고 현재 위치에서 freeze한다.
        # =================================================
        frozen_ids = set(hop_region_ids)

        # NORMAL → SHEPHERD 역할 변경
        for robot_id in frozen_ids:
            robot = robots_by_id[robot_id]

            if robot.role != "NORMAL":
                raise RuntimeError(
                    "Initial Shepherd cohort contains "
                    f"non-NORMAL robot: id={robot_id} role={robot.role}"
                )

            assign_initial_shepherd_role(robot, branch_id)

        # 확정된 Initial Shepherd들을 현재 위치에 고정
        for robot_id in frozen_ids:
            robot = robots_by_id[robot_id]
            freeze_role_robot(robot, "SHEPHERD", branch_id)

        # 해당 Branch의 Initial Shepherd ID와 봉쇄 완료 상태 저장
        state["initial_shepherd_ids"] = set(frozen_ids)
        state["initial_sealed"] = True

        print(
            "[InitialShepherdBoundaryLocked] "
            f"branch={branch_id} count={len(frozen_ids)} "
            f"hop_counts={hop_counts} max_gap={max_gap:.3f} "
            f"ids={sorted(frozen_ids)}"
        )

    # 모든 Branch의 Initial Shepherd boundary가 완성되어야 True
    return all(
        branch_states[branch["id"]]["initial_sealed"]
        for branch in branches
    )



# Backtracking 과정에서 Anchor 바로 뒤에 있으면서 Anchor와 직접 통신 가능한 NORMAL 로봇들을 찾는 함수
# Backtracking 중 Anchor 주변에서 직접 통신 가능한 NORMAL 로봇을 탐색한다.
# Anchor 자신과 NORMAL이 아닌 로봇은 제외하고, Anchor-local 좌표 기준 진행방향 뒤쪽(x < 0)에 있으면서, Anchor와의 거리가 COMM_RANGE 이내인 로봇만 선택한다.
# 즉, multi-hop이 아니라 Anchor와 직접 1-hop communication 가능한
# 후방 NORMAL 로봇들의 ID를 반환한다.
def get_anchor_nearby_backtracking_normals(
    observation: LocalObservation,
    robots_by_id: dict[int, environment.Robot],
) -> list[int]:

    nearby_ids = []

    for robot_id, local_position in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):
        # Anchor 자신 제외
        if robot_id == observation.lidar_robot_id:
            continue

        robot = robots_by_id[robot_id]

        # Backtracking 대상은 NORMAL 로봇만
        if robot.role != "NORMAL":
            continue

        # Anchor-local x축 기준 Anchor 진행방향 뒤쪽에 있는 로봇만 선택
        if local_position.x >= -environment.ROBOT_RADIUS:
            continue

        # Anchor와 직접 1-hop communication 가능한 거리인지 확인
        if local_position.length() > environment.COMM_RANGE:
            continue

        nearby_ids.append(robot_id)

    return nearby_ids



# def detect_complete_dead_end_wall_layer(
#     observation: LocalObservation,
#     robots_by_id: dict[
#         int,
#         environment.Robot,
#     ],
# ) -> tuple[
#     bool,
#     set[int],
#     float,
#     float,
# ]:
#     """
#     LiDAR가 얻은 dead-end wall WL~WR를 기준으로,

#     1. wall 바로 앞 1-hop depth 안에 있는 NORMAL을 찾고
#     2. 그 robot body가 WL~WR 전체 폭을 실제로 덮는지 검사한다.

#     Returns:
#         ready
#         wall_robot_ids
#         wall_width
#         max_gap
#     """

#     wall_segment = (
#         extract_dead_end_wall_segment(
#             observation.lidar_scan
#         )
#     )

#     if wall_segment is None:

#         return (
#             False,
#             set(),
#             0.0,
#             float("inf"),
#         )

#     (
#         wall_left,
#         wall_right,
#     ) = wall_segment

#     wall_vector = (
#         wall_right
#         - wall_left
#     )

#     wall_width = (
#         wall_vector.length()
#     )

#     if (
#         wall_width
#         <= environment.EPSILON
#     ):

#         return (
#             False,
#             set(),
#             wall_width,
#             float("inf"),
#         )

#     # WL -> WR 방향 unit vector
#     tangent = (
#         wall_vector
#         / wall_width
#     )

#     local_position_by_id = {
#         robot_id: local_position

#         for (
#             robot_id,
#             local_position,
#         )

#         in zip(
#             observation.robot_ids,
#             observation.relative_positions,
#         )
#     }

#     wall_robot_ids: set[int] = set()

#     intervals: list[
#         tuple[
#             float,
#             float,
#         ]
#     ] = []

#     # =====================================================
#     # Wall 바로 앞 1-hop layer만 선택
#     # =====================================================

#     for (
#         robot_id,
#         position,
#     ) in local_position_by_id.items():

#         if (
#             robot_id
#             == observation.lidar_robot_id
#         ):
#             continue

#         robot = (
#             robots_by_id[
#                 robot_id
#             ]
#         )

#         if (
#             robot.role
#             != "NORMAL"
#         ):
#             continue

#         # ---------------------------------------------
#         # WL 기준 wall tangent 방향 위치
#         # ---------------------------------------------

#         relative = (
#             position
#             - wall_left
#         )

#         along_wall = (
#             relative.dot(
#                 tangent
#             )
#         )

#         # wall segment 좌우 바깥 robot 제외
#         if (
#             along_wall
#             <
#             -environment.ROBOT_RADIUS
#             or
#             along_wall
#             >
#             wall_width
#             + environment.ROBOT_RADIUS
#         ):
#             continue

#         # ---------------------------------------------
#         # robot center와 wall line 사이 수직거리
#         # ---------------------------------------------

#         projected_point = (
#             wall_left
#             + tangent
#             * min(
#                 wall_width,
#                 max(
#                     0.0,
#                     along_wall,
#                 ),
#             )
#         )

#         distance_from_wall = (
#             position
#             - projected_point
#         ).length()

#         # wall 바로 앞 1-hop layer만.
#         if (
#             distance_from_wall
#             >
#             BACKTRACK_DEAD_END_ONE_HOP_DEPTH
#         ):
#             continue

#         wall_robot_ids.add(
#             robot_id
#         )

#         # ---------------------------------------------
#         # 이 robot body가 wall 폭에서 덮는 interval
#         # ---------------------------------------------

#         left = max(
#             0.0,
#             along_wall
#             - environment.ROBOT_RADIUS,
#         )

#         right = min(
#             wall_width,
#             along_wall
#             + environment.ROBOT_RADIUS,
#         )

#         if (
#             right
#             >= left
#         ):

#             intervals.append(
#                 (
#                     left,
#                     right,
#                 )
#             )

#     # =====================================================
#     # WL -> WR 전체 coverage 검사
#     # =====================================================

#     if not intervals:

#         return (
#             False,
#             wall_robot_ids,
#             wall_width,
#             wall_width,
#         )

#     intervals.sort()

#     covered_until = 0.0
#     max_gap = 0.0

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

#     # 마지막 robot -> WR 끝점까지 gap
#     if (
#         covered_until
#         < wall_width
#     ):

#         max_gap = max(
#             max_gap,
#             wall_width
#             - covered_until,
#         )

#     ready = (
#         max_gap
#         <=
#         BACKTRACK_DEAD_END_GAP_TOLERANCE
#     )

#     return (
#         ready,
#         wall_robot_ids,
#         wall_width,
#         max_gap,
#     )

# def collect_dead_end_wall_shepherd_cohort(
#     observation: LocalObservation,
#     robots_by_id: dict[
#         int,
#         environment.Robot,
#     ],
#     max_hops: int,
# ) -> tuple[
#     set[int],
#     list[set[int]],
# ]:
#     """
#     Dead-end LiDAR wall 바로 앞에 자연스럽게 쌓인 NORMAL을
#     seed layer로 잡고, 그 layer로부터 max_hops만큼
#     robot-to-robot physical adjacency를 확장한다.

#     사용 정보:
#         - Anchor-local robot relative position
#         - Anchor-local LiDAR wall range
#         - robot-to-robot relative distance

#     global/map robot position은 사용하지 않는다.
#     """

#     local_position_by_id = {
#         robot_id: local_position

#         for (
#             robot_id,
#             local_position,
#         )

#         in zip(
#             observation.robot_ids,
#             observation.relative_positions,
#         )
#     }

#     # =====================================================
#     # 1. Dead-end wall 바로 앞 NORMAL = seed layer
#     # =====================================================

#     wall_seed_ids: set[int] = set()

#     for (
#         robot_id,
#         local_position,
#     ) in local_position_by_id.items():

#         if (
#             robot_id
#             == observation.lidar_robot_id
#         ):
#             continue

#         robot = (
#             robots_by_id[
#                 robot_id
#             ]
#         )

#         if (
#             robot.role
#             != "NORMAL"
#         ):
#             continue

#         # Dead-end를 바라보고 있는 Anchor 기준
#         # 앞쪽 robot만 후보.
#         if (
#             local_position.x
#             <= 0.0
#         ):
#             continue

#         # 현재 corridor 폭 바깥 robot 제외.
#         if (
#             abs(local_position.y)
#             >
#             (
#                 0.5
#                 * KNOWN_CORRIDOR_WIDTH
#                 + environment.GRID_SPACING
#             )
#         ):
#             continue

#         robot_distance = (
#             local_position.length()
#         )

#         if (
#             robot_distance
#             <= environment.EPSILON
#         ):
#             continue

#         # Anchor에서 해당 robot이 보이는 local bearing.
#         robot_bearing_deg = (
#             math.degrees(
#                 math.atan2(
#                     local_position.y,
#                     local_position.x,
#                 )
#             )
#         )

#         # 뒤쪽 robot은 제외.
#         if (
#             abs(robot_bearing_deg)
#             >
#             ANCHOR_EXPLORE_HALF_FOV_DEG
#         ):
#             continue

#         # 같은 bearing 방향으로 LiDAR가 본 실제 wall 거리.
#         wall_range = (
#             lidar_range_at_local_angle(
#                 observation.lidar_scan,
#                 robot_bearing_deg,
#             )
#         )

#         # robot center에서 wall까지 남은 local 거리.
#         wall_gap = (
#             wall_range
#             - robot_distance
#         )

#         # LiDAR가 본 wall 바로 앞에 있는 NORMAL이면
#         # wall-contact seed.
#         if (
#             -environment.ROBOT_RADIUS
#             <= wall_gap
#             <= BACKTRACK_WALL_SEED_GAP
#         ):

#             wall_seed_ids.add(
#                 robot_id
#             )

#     if not wall_seed_ids:

#         return (
#             set(),
#             [],
#         )

#     # =====================================================
#     # 2. seed layer에서 robot-hop 확장
#     # =====================================================

#     layers: list[
#         set[int]
#     ] = [
#         set(
#             wall_seed_ids
#         )
#     ]

#     selected_ids = set(
#         wall_seed_ids
#     )

#     frontier = set(
#         wall_seed_ids
#     )

#     for _hop in range(
#         1,
#         max_hops + 1,
#     ):

#         next_frontier: set[int] = set()

#         for source_id in frontier:

#             source_position = (
#                 local_position_by_id[
#                     source_id
#                 ]
#             )

#             for (
#                 candidate_id,
#                 candidate_position,
#             ) in local_position_by_id.items():

#                 if (
#                     candidate_id
#                     in selected_ids
#                 ):
#                     continue

#                 if (
#                     candidate_id
#                     == observation.lidar_robot_id
#                 ):
#                     continue

#                 candidate_robot = (
#                     robots_by_id[
#                         candidate_id
#                     ]
#                 )

#                 if (
#                     candidate_robot.role
#                     != "NORMAL"
#                 ):
#                     continue

#                 # Anchor 앞쪽의 dead-end compressed crowd만.
#                 if (
#                     candidate_position.x
#                     <= 0.0
#                 ):
#                     continue

#                 if (
#                     abs(candidate_position.y)
#                     >
#                     (
#                         0.5
#                         * KNOWN_CORRIDOR_WIDTH
#                         + environment.GRID_SPACING
#                     )
#                 ):
#                     continue

#                 relative_position = (
#                     candidate_position
#                     - source_position
#                 )

#                 if (
#                     relative_position.length()
#                     <= BACKTRACK_WALL_HOP_RADIUS
#                 ):

#                     next_frontier.add(
#                         candidate_id
#                     )

#         if not next_frontier:
#             break

#         layers.append(
#             next_frontier
#         )

#         selected_ids.update(
#             next_frontier
#         )

#         frontier = (
#             next_frontier
#         )

#     return (
#         selected_ids,
#         layers,
#     )



# =========================================================
# Backtracking Shepherd - Seed 선택
#
# VISITED Marker 또는 Dead-end 감지 후 Backtracking Shepherd를 형성하기 위해
# Anchor 바로 뒤쪽에 있는 NORMAL 로봇 중 seed 1대를 선택한다.
#
# 1. Anchor 자신과 NORMAL이 아닌 로봇은 제외
# 2. Anchor-local 좌표에서 Anchor 진행방향 뒤쪽에 있는 로봇만 후보로 사용
# 3. Anchor와 직접 통신 가능한 COMM_RANGE 이내의 로봇만 후보로 사용
# 4. 조건을 만족하는 후보 중 Anchor와 가장 가까운 NORMAL 1대를 seed로 선택
# 5. 선택된 seed는 이후 multi-hop 모집의 기준점이며,
#    collect_backtracking_seed_one_two_hop()에서 seed-relative Hop 0으로 사용된다.
#
# 이후 robot-to-robot COMM_RANGE를 따라
# seed 기준 Hop 1 → Hop 2까지 확장하여
# Backtracking Shepherd cohort를 구성한다.
#
# 흐름:
# Backtracking 시작
# → Anchor 뒤쪽 NORMAL 탐색
# → Anchor와 직접 통신 가능한 후보 추출
# → Anchor와 가장 가까운 NORMAL 1대 선택
# → Backtracking seed
# → seed-relative Hop 0으로 설정
# → Hop 1~2 multi-hop Shepherd 모집
#
# 모든 위치 판단은 Anchor-relative local position을 사용한다.
# =========================================================
def find_backtracking_seed(
    observation: LocalObservation,
    robots_by_id: dict[int, environment.Robot],
) -> int | None:

    candidates: list[tuple[float, int]] = []

    for robot_id, local_position in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):
        # Anchor 자신 제외
        if robot_id == observation.lidar_robot_id:
            continue

        robot = robots_by_id[robot_id]

        # NORMAL만 seed 후보
        if robot.role != "NORMAL":
            continue

        # Anchor 진행방향 기준 뒤쪽 로봇만 선택
        if local_position.x >= -environment.ROBOT_RADIUS:
            continue

        distance = local_position.length()

        # Anchor와 직접 통신 가능한 범위만 선택
        if distance > environment.COMM_RANGE:
            continue

        candidates.append((distance, robot_id))

    if not candidates:
        return None

    # Anchor와 직접 통신 가능한 NORMAL 중 가장 가까운 로봇을
    # Backtracking seed로 선택
    candidates.sort()
    return candidates[0][1]



# =========================================================
# Backtracking Shepherd - Hop 0~2 Multi-hop Cohort 모집
#
# find_backtracking_seed()에서 선택한 NORMAL 1대를 Hop 0 seed로 사용하고,
# Anchor 진행방향 뒤쪽의 NORMAL 로봇들을 대상으로 robot-to-robot COMM_RANGE 기반 Hop 1~2 cohort를 구성한다.
#
# Hop 0 = find_backtracking_seed()에서 선택된 seed 1대
# Hop 1 = Hop 0 seed와 COMM_RANGE 이내에서 직접 통신 가능한 NORMAL
# Hop 2 = Hop 1 로봇과 COMM_RANGE 이내에서 직접 통신 가능하며 Hop 0/1에 포함되지 않은 추가 NORMAL
#
# 최종적으로 Hop 0 ∪ Hop 1 ∪ Hop 2를 Backtracking Shepherd 후보 cohort_ids로 반환한다.
# 모든 공간 판단은 Anchor-relative local position과 robot-to-robot relative distance만 사용한다.
#
# 흐름:
# Hop 0 seed → 뒤쪽 NORMAL 후보 추출 → Hop 1 모집 → Hop 2 모집 → 전체 cohort 반환
# =========================================================
def collect_backtracking_seed_one_two_hop(
    observation: LocalObservation,
    robots_by_id: dict[int, environment.Robot],
    seed_id: int,
) -> tuple[set[int], list[set[int]]]:

    # 각 로봇의 Anchor-relative local 위치를 ID별로 저장
    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position in zip(observation.robot_ids, observation.relative_positions)
    }

    # find_backtracking_seed()에서 선택된 Hop 0 seed의 local 위치 확인
    seed_position = local_position_by_id.get(seed_id)
    if seed_position is None:
        return set(), [set(), set(), set()]

    # Hop 0 seed는 반드시 NORMAL이어야 함
    seed_robot = robots_by_id[seed_id]
    if seed_robot.role != "NORMAL":
        return set(), [set(), set(), set()]

    # Anchor 뒤쪽에 위치한 NORMAL만 multi-hop 모집 대상으로 사용 (로봇 ID, 로봇의 2차원 상대위치(Anchor 기준 상대좌표))
    eligible: dict[int, pygame.Vector2] = {}

    for robot_id, local_position in local_position_by_id.items():
        if robot_id == observation.lidar_robot_id:  # Anchor 자신 제외
            continue

        robot = robots_by_id[robot_id]
        if robot.role != "NORMAL":  # NORMAL만 모집
            continue

        if local_position.x >= -environment.ROBOT_RADIUS:  # Anchor 진행방향 뒤쪽만 모집
            continue

        eligible[robot_id] = local_position

    # seed가 현재 모집 가능 영역에 없으면 cohort 형성 실패
    if seed_id not in eligible:
        return set(), [set(), set(), set()]

    # Hop 0 = find_backtracking_seed()에서 선택된 NORMAL seed 1대
    seed_layer = {seed_id}

    # Hop 1 = Hop 0 seed와 robot-to-robot COMM_RANGE 이내에서 직접 통신 가능한 NORMAL
    one_hop: set[int] = set()

    for candidate_id, candidate_position in eligible.items():
        if candidate_id == seed_id:
            continue

        # "seed와 candidate 사이 거리가 COMM_RANGE 이하인가?"
        if (candidate_position - seed_position).length() <= environment.COMM_RANGE:
            one_hop.add(candidate_id)


    # seed/Hop1 제외한 나머지 eligible 로봇
    # → 각 Hop1 로봇과 거리 계산
    # → COMM_RANGE 이내면
    # → Hop2

    # Hop 2 = Hop 1과 직접 통신 가능하며 Hop 0/1에 아직 포함되지 않은 추가 NORMAL
    two_hop: set[int] = set()

    # 각 Hop 1 로봇을 통신거리 검사 기준점으로 사용 (source_position 이 hop1 로봇의 상대위치)
    for source_id in one_hop:
        source_position = eligible[source_id]

        # eligible.items()에는 이미 앞 단계에서 걸러진 “Anchor 뒤쪽의 NORMAL 로봇들”이 들어있음
        # Anchor 뒤쪽의 eligible NORMAL 전체를 Hop 2 후보로 다시 검사
        for candidate_id, candidate_position in eligible.items():
            # 이미 seed(hop 0)이거나 hop1에 이미 들어간 로봇은 제외 -> 남는건 Anchor 뒤쪽 NORMAL 중에서 아직 Hop 0도 아니고 Hop 1도 아닌 로봇들
            if candidate_id == seed_id or candidate_id in one_hop:
                continue

            # candidate_position = 현재 검사 중인 Hop 2 후보 로봇의 Anchor-relative 위치
            # source_position = 현재 기준이 되는 Hop 1 로봇의 Anchor-relative 위치  
            relative_position = candidate_position - source_position

            # 현재 Hop 1 로봇과 통신거리 이내인 로봇만 Hop 2에 추가
            if relative_position.length() <= environment.COMM_RANGE:
                two_hop.add(candidate_id)

    # Hop 0~2의 합집합을 최종 Backtracking Shepherd 후보 cohort로 구성
    cohort_ids = seed_layer | one_hop | two_hop

    # 전체 cohort와 Hop별 robot ID 집합을 함께 반환
    return cohort_ids, [seed_layer, one_hop, two_hop]



# =========================================================
# Backtracking Shepherd - Backtracking을 시작하기에 필요한 최소 crowd 수 계산
#
# Backtracking Shepherd cohort를 형성하기 전에 Anchor 주변에 충분한 NORMAL crowd가 모였는지
# 판단하기 위한 최소 robot 수를 corridor_width에 따라 계산한다.
#
# 1. 로봇 직경(2*ROBOT_RADIUS)과 설정된 Shepherd 간격(GRID_SPACING*SPACING_RATIO) 중
#    더 큰 값을 target_spacing으로 사용한다.
# 2. corridor_width / target_spacing으로 통로 폭을 기준으로 필요한 robot 수를 계산한다.
# 3. ceil()로 올림하고 +1하여 full_width_count를 계산한다.
# 4. 최소 2대 이상이 되도록 max(2, full_width_count)를 반환한다.
#
# 주의: 이 값은 실제 통로를 한 줄로 완전히 봉쇄했는지 검사하는 physical coverage 조건이 아니라,
# Backtracking을 시작하기에 Anchor 주변 crowd의 로봇 수가 충분한지 판단하는 최소 개수 기준이다.
# 위치 정보는 사용하지 않고 corridor_width와 robot/formation 크기 파라미터만 사용한다.
# =========================================================
def required_backtracking_crowd_count(corridor_width: float) -> int:

    # Backtracking crowd에서 로봇 사이에 필요한 기준 간격
    target_spacing = max(
        2.0 * environment.ROBOT_RADIUS,
        environment.GRID_SPACING * BACKTRACK_SHEPHERD_SPACING_RATIO,
    )

    # 통로 폭을 기준으로 필요한 최소 crowd robot 수 계산
    full_width_count = math.ceil(corridor_width / target_spacing) + 1

    # 최소 2대 이상을 요구
    return max(2, full_width_count)



# # =========================================================
# # Backtracking Shepherd - Role Activation In Place
# #
# # Hop 0~2로 선택된 Backtracking Shepherd cohort를 현재 위치 그대로
# # NORMAL → SHEPHERD(PUSH)로 전환하여 이후 Pressure Push에 사용한다.
# # =========================================================
# def activate_backtracking_shepherds_in_place(
#     robot_ids: list[int],
#     robots_by_id: dict[int, environment.Robot],
#     branch_id: str,
# ) -> None:

#     for robot_id in robot_ids:
#         robot = robots_by_id[robot_id]

#         # Backtracking Shepherd로 선택된 로봇은 반드시 NORMAL이어야 함
#         if robot.role != "NORMAL":
#             raise RuntimeError(
#                 f"Captured backtracking robot is not NORMAL: "
#                 f"id={robot_id} role={robot.role}"
#             )

#         # 역할 전환 전 위치 저장: 전환 과정에서 위치가 바뀌지 않는지 확인
#         before = robot.position.copy()

#         # NORMAL → Backtracking SHEPHERD로 역할 변경하고 현재 Branch와 PUSH mode 지정
#         robot.role = "SHEPHERD"
#         robot.role_branch = branch_id
#         robot.shepherd_mode = "PUSH"

#         # 이후 Junction 방향으로 이동해야 하므로 위치를 freeze하지 않음
#         robot.role_frozen = False
#         robot.role_frozen_position = None

#         # 기존 NORMAL의 motion state만 초기화하며 현재 위치는 변경하지 않음
#         robot.velocity.update(0.0, 0.0)
#         robot.acceleration.update(0.0, 0.0)
#         robot.filtered_acceleration.update(0.0, 0.0)
#         robot.commanded_velocity.update(0.0, 0.0)
#         robot.observed_velocity.update(0.0, 0.0)

#         # 역할 전환으로 위치가 바뀌었다면 오류
#         if robot.position.distance_to(before) > environment.EPSILON:
#             raise RuntimeError(
#                 "Backtracking Shepherd role transition "
#                 "caused a position jump."
#             )



# =========================================================
# Backtracking Shepherd - Formation Cohort Selection
#
# Hop 0~2로 모집된 cohort에서 seed를 반드시 포함하고,
# seed와 가까운 순서로 required_count만큼 최종 Backtracking Shepherd를 선택한다.
# =========================================================
def select_backtracking_formation_cohort(
    observation: LocalObservation,
    cohort_ids: set[int],
    seed_id: int,
    required_count: int,
) -> set[int]:

    # 각 로봇의 Anchor-relative local 위치를 ID별로 저장
    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(observation.robot_ids, observation.relative_positions)
    }

    # seed가 모집된 cohort 또는 현재 observation에 없으면 선택 실패
    if seed_id not in cohort_ids or seed_id not in local_position_by_id:
        return set()

    # Hop 0 seed의 Anchor-relative local 위치
    seed_position = local_position_by_id[seed_id]

    # 전체 Hop 0~2 cohort에서 seed를 제외한 선택 후보 구성
    other_candidates = [
        robot_id
        for robot_id in cohort_ids
        if robot_id != seed_id and robot_id in local_position_by_id
    ]

    # 절대 위치가 아닌 seed와 각 후보 사이의 상대거리 기준으로 가까운 순서 정렬
    other_candidates.sort(
        key=lambda robot_id:
        (local_position_by_id[robot_id] - seed_position).length()
    )

    # Hop 0 seed는 최종 Shepherd formation에 반드시 포함
    selected = {seed_id}

    # required_count를 맞추기 위해 seed 외에 추가로 필요한 로봇 수 계산
    remaining_count = max(0, required_count - 1)

    # seed와 가까운 후보부터 필요한 수만큼 최종 Shepherd로 선택
    selected.update(other_candidates[:remaining_count])

    return selected


# =========================================================
# Backtracking Shepherd - Formation Command Relay
#
# Hop 0 seed에서 시작하여 선택된 Shepherd formation 내부의 COMM_RANGE 연결을 따라
# FORM / PUSH / RELEASE 명령을 multi-hop으로 전달하고, 수신한 로봇에 해당 상태를 적용한다.
# =========================================================
def relay_backtracking_shepherd_command(
    observation: LocalObservation,
    robots_by_id: dict[int, environment.Robot],
    seed_id: int,
    formation_ids: set[int],
    branch_id: str,
    command: str,
    left_wall_range: float | None = None,
    right_wall_range: float | None = None,
) -> set[int]:

    # 각 로봇의 Anchor-relative local 위치를 ID별로 저장
    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position in zip(observation.robot_ids, observation.relative_positions)
    }

    # Hop 0 seed가 formation 또는 현재 observation에 없으면 relay 실패
    if seed_id not in formation_ids or seed_id not in local_position_by_id:
        return set()

    # Hop 0 seed가 최초 명령 수신자이며, 여기서부터 relay 시작
    received = {seed_id}
    frontier = {seed_id}

    # formation 내부에서 robot-to-robot COMM_RANGE를 따라 명령을 multi-hop 전파
    while frontier:
        next_frontier: set[int] = set()

        for source_id in frontier:
            source_position = local_position_by_id.get(source_id)
            if source_position is None:
                continue

            for candidate_id in formation_ids:
                # 이미 명령을 받은 Shepherd는 중복 relay 대상에서 제외
                if candidate_id in received:
                    continue

                candidate_position = local_position_by_id.get(candidate_id)
                if candidate_position is None:
                    continue

                # 현재 송신 Shepherd와 직접 통신 가능하면 다음 relay 수신자로 추가
                if (candidate_position - source_position).length() <= environment.COMM_RANGE:
                    received.add(candidate_id)
                    next_frontier.add(candidate_id)

        # 이번 Hop에서 새로 수신한 Shepherd들이 다음 Hop의 송신자가 됨
        frontier = next_frontier

    # 실제로 relay 명령을 수신한 Shepherd들에게 command 적용
    for robot_id in received:
        robot = robots_by_id[robot_id]

        if command == "FORM":
            # Backtracking Shepherd로 전환하고 formation 형성 단계 시작
            robot.role = "SHEPHERD"
            robot.role_branch = branch_id
            robot.shepherd_mode = "FORM"
            robot.role_frozen = False
            robot.role_frozen_position = None

            # formation 유지에 사용할 좌/우 wall range 저장
            if left_wall_range is not None:
                robot.shepherd_left_wall_range = float(left_wall_range)
            if right_wall_range is not None:
                robot.shepherd_right_wall_range = float(right_wall_range)

            # 기존 NORMAL의 motion state 제거
            robot.velocity.update(0.0, 0.0)
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
            robot.commanded_velocity.update(0.0, 0.0)
            robot.observed_velocity.update(0.0, 0.0)

        elif command == "PUSH":
            # formation 완료 후 Junction 방향 Pressure Push 단계로 전환
            robot.shepherd_mode = "PUSH"
            robot.role_frozen = False
            robot.role_frozen_position = None

        elif command == "RELEASE":
            # Backtracking 종료 후 Shepherd 역할을 해제하고 NORMAL로 복귀
            release_robot_to_normal(robot)

    # 명령을 실제로 수신한 formation robot 수와 ID 출력
    print(
        "[ShepherdRelay] "
        f"branch={branch_id} seed={seed_id} command={command} "
        f"received={len(received)}/{len(formation_ids)} "
        f"ids={sorted(received)}"
    )

    # 실제로 relay 명령을 수신한 robot ID 반환
    return received


# =========================================================
# Backtracking Shepherd - Formation Start
#
# Branch 탐색이 정상적으로 끝났을 때, 
# Backtracking에 사용할 Shepherd 로봇 전체에게 FORM 명령이 전달됐는지 확인하고, 
# 모두 받았을 때만 formation 시작을 승인
# =========================================================
def start_backtracking_shepherd_formation(
    observation: LocalObservation,
    formation_ids: set[int],
    robots_by_id: dict[int, environment.Robot],
    seed_id: int,
    branch_id: str,
    event_valid: bool,
) -> bool:

    # VISITED Marker / Dead-end 등 유효한 탐색 종료 이벤트가 아니면 Backtracking 시작 거부
    if not event_valid:
        print(
            "[BacktrackingFormationRejected] "
            "reason=NO_VALID_TERMINAL_EVENT"
        )
        return False

    # Hop 0 seed에서 formation 전체로 FORM 명령을 multi-hop relay
    received_ids = relay_backtracking_shepherd_command(
        observation,
        robots_by_id,
        seed_id,
        formation_ids,
        branch_id,
        command="FORM",
    )

    # 선택된 formation의 모든 로봇이 FORM 명령을 수신했는지 확인
    all_received = (received_ids == formation_ids)

    # 한 대라도 FORM 명령을 받지 못했다면 formation 시작 실패
    if not all_received:
        print(
            "[BacktrackingFormationRelayIncomplete] "
            f"branch={branch_id} "
            f"received={len(received_ids)} "
            f"required={len(formation_ids)}"
        )
        return False

    # formation 전체에 FORM 명령 전달이 완료되었음을 기록
    print(
        "[BacktrackingFormationBroadcast] "
        f"branch={branch_id} "
        f"seed={seed_id} "
        f"count={len(formation_ids)} "
        f"ids={sorted(formation_ids)}"
    )

    return True



# =========================================================
# Backtracking Shepherd - Forward Wall Blocker Check
#
# PUSH 중인 Shepherd가 다음 한 step을 return direction으로 이동할 때
# 벽에 충돌할 로봇을 실제 이동 전에 검사한다.
# =========================================================
def backtracking_forward_wall_blockers(
    shepherd_ids: list[int],
    robots_by_id: dict[int, environment.Robot],
    return_yaw_deg: float,
    dt: float,
) -> set[int]:

    # 현재 SHEPHERD이면서 PUSH mode인 로봇만 검사
    active_ids = [
        robot_id for robot_id in shepherd_ids
        if robots_by_id[robot_id].role == "SHEPHERD"
        and getattr(robots_by_id[robot_id], "shepherd_mode", None) == "PUSH"
    ]

    # PUSH 중인 Shepherd가 없으면 blocker도 없음
    if not active_ids:
        return set()

    # Backtracking PUSH 속도 계산
    push_speed = BACKTRACK_PUSH_SPEED_RATIO * environment.INITIAL_SAFE_MAX_SPEED

    # 다음 dt 동안 return direction으로 이동할 예상 변위 계산
    forward_delta = anchor_local_to_world(
        pygame.Vector2(push_speed * dt, 0.0), return_yaw_deg
    )

    # 예상 다음 위치가 이동 불가능한 Shepherd를 wall blocker로 반환
    # 실제 robot.position은 변경하지 않음
    return {
        robot_id for robot_id in active_ids
        if not physical_is_walkable(
            robots_by_id[robot_id].position + forward_delta,
            robots_by_id[robot_id].radius,
        )
    }



# =========================================================
# Backtracking Shepherd - Wall Detach (벽 회피용 보정 동작 -> Shepherd 대형 전체를 통째로 옆으로 미는 것)
#
# return-direction PUSH가 벽에 막히면 Shepherd formation 전체를
# LEFT 또는 RIGHT로 같은 거리만큼 이동시켜 topology를 유지한 채 벽에서 분리한다.
# detach 방향은 episode 시작 시 한 번 선택하고 PUSH가 가능해질 때까지 유지한다.
# =========================================================
def detach_backtracking_shepherd_group_step(
    observation: LocalObservation,
    shepherd_ids: list[int],
    robots_by_id: dict[int, environment.Robot],
    return_yaw_deg: float,
    dt: float,
    locked_direction: str | None,
) -> tuple[bool, set[int], str]:

    # 현재 SHEPHERD이면서 PUSH mode인 로봇만 detach 대상으로 선택
    active_ids = [
        robot_id for robot_id in shepherd_ids
        if robots_by_id[robot_id].role == "SHEPHERD"
        and getattr(robots_by_id[robot_id], "shepherd_mode", None) == "PUSH"
    ]

    # PUSH Shepherd가 없으면 별도의 detach 없이 완료
    if not active_ids:
        return True, set(), "NONE"

    # 현재 상태에서 바로 return-direction PUSH가 가능한지 검사
    blocked_ids = backtracking_forward_wall_blockers(
        active_ids, robots_by_id, return_yaw_deg, dt
    )

    # 벽에 걸리는 Shepherd가 없으면 detach 완료
    if not blocked_ids:
        return True, set(), locked_direction if locked_direction is not None else "NONE"

    # Shepherd들의 Anchor-relative local 위치 저장
    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(observation.robot_ids, observation.relative_positions)
    }

    # LiDAR에서 Anchor 기준 좌/우 벽까지의 거리 추출
    left_range, right_range = extract_lateral_wall_ranges(observation.lidar_scan)
    contact_radius = max(WALL_CONTACT_RADIUS, environment.ROBOT_RADIUS)

    # formation 전체가 LEFT/RIGHT로 이동할 수 있는 최소 여유 공간 계산
    # Anchor-local 기준 y<0은 LEFT, y>0은 RIGHT
    left_rooms: list[float] = []
    right_rooms: list[float] = []

    for robot_id in active_ids:
        local_position = local_position_by_id.get(robot_id)
        if local_position is None:
            continue
        
        # 각 Shepherd의 Anchor-relative 위치를 이용하여 계산
        # Anchor 를 기준점으로 삼아서, Anchor 와 벽 사이의 거리 & Anchor 와 Shepherd의 상대위치 정보를 결합해 Shepherd 와 벽 사이의 거리를 추정
        left_room = local_position.y + left_range - contact_radius
        right_room = right_range - local_position.y - contact_radius
        left_rooms.append(left_room)
        right_rooms.append(right_room)

    # formation에서 가장 여유가 적은 로봇을 기준으로 전체 이동 가능 공간 결정
    minimum_left_room = min(left_rooms) if left_rooms else 0.0
    minimum_right_room = min(right_rooms) if right_rooms else 0.0

    # 이미 detach 방향이 정해졌으면 그대로 유지하고, 처음이라면 여유가 큰 방향 선택
    if locked_direction in ("LEFT", "RIGHT"):
        chosen_direction = locked_direction
    else:
        chosen_direction = "LEFT" if minimum_left_room > minimum_right_room else "RIGHT"

    # Anchor-local lateral 이동 방향 부호 결정
    lateral_sign = -1.0 if chosen_direction == "LEFT" else +1.0

    # 이번 dt 동안 lateral 방향으로 이동할 거리 계산
    detach_speed = BACKTRACK_WALL_DETACH_SPEED_RATIO * environment.INITIAL_SAFE_MAX_SPEED
    local_detach_delta = pygame.Vector2(0.0, lateral_sign * detach_speed * dt)
    world_detach_delta = anchor_local_to_world(local_detach_delta, return_yaw_deg)

    # formation 전체가 동일한 lateral delta로 이동 가능한지 미리 검사
    group_can_move = all(
        physical_is_walkable(
            robots_by_id[robot_id].position + world_detach_delta,
            robots_by_id[robot_id].radius,
        )
        for robot_id in active_ids
    )

    # 한 대라도 이동할 수 없다면 현재 방향을 유지한 채 detach 실패 반환
    if not group_can_move:
        role_debug(
            "backtrack-detach-no-room",
            (
                "[BacktrackDetachNoRoom] "
                f"direction={chosen_direction} blocked_ids={sorted(blocked_ids)} "
                f"left_room={minimum_left_room:.3f} right_room={minimum_right_room:.3f}"
            ),
        )
        return False, blocked_ids, chosen_direction

    # 실제 적용된 lateral 이동량으로 Shepherd 속도 계산
    realized_velocity = world_detach_delta / max(dt, environment.EPSILON)

    # 모든 Shepherd에 동일한 lateral delta를 적용하여 기존 formation topology 유지
    for robot_id in active_ids:
        robot = robots_by_id[robot_id]
        old_position = robot.position.copy()
        robot.previous_position.update(old_position)
        robot.position += world_detach_delta
        robot.velocity.update(realized_velocity)
        robot.observed_velocity.update(realized_velocity)
        robot.commanded_velocity.update(realized_velocity)
        robot.acceleration.update(0.0, 0.0)

    # lateral 이동 후 다시 return-direction PUSH가 가능한지 재검사
    remaining_blocked_ids = backtracking_forward_wall_blockers(
        active_ids, robots_by_id, return_yaw_deg, dt
    )
    detach_ready = len(remaining_blocked_ids) == 0

    # detach 결과 기록
    role_debug(
        "backtrack-wall-detach",
        (
            "[BacktrackWallDetach] "
            f"blocked_ids={sorted(remaining_blocked_ids)} direction={chosen_direction} "
            f"count={len(active_ids)} delta={world_detach_delta.length():.4f} "
            f"ready={detach_ready}"
        ),
    )

    # PUSH 가능 여부, 아직 막힌 Shepherd, 현재 lock된 detach 방향 반환
    return detach_ready, remaining_blocked_ids, chosen_direction



# =========================================================
# Backtracking Shepherd - Physical Pressure Push
#
# Backtracking Shepherd formation 전체를 Junction return direction으로 함께 이동시키고,
# 접촉한 NORMAL에 물리적 PUSH를 전달하여 reverse flow를 유도한다.
# =========================================================
def update_pressure_push(
    observation: LocalObservation,
    shepherd_ids: list[int],
    robots_by_id: dict[int, environment.Robot],
    return_yaw_deg: float,
    dt: float,
) -> dict[int, pygame.Vector2]:

    # Shepherd와 접촉하여 PUSH를 받은 NORMAL별 acceleration 저장
    contact_accelerations: dict[int, pygame.Vector2] = {}

    # Shepherd가 없으면 전달할 contact acceleration도 없음
    if not shepherd_ids:
        return contact_accelerations

    # 현재 실제 SHEPHERD이면서 PUSH mode인 로봇만 Pressure Push에 사용
    active_ids = [
        robot_id for robot_id in shepherd_ids
        if robots_by_id[robot_id].role == "SHEPHERD"
        and getattr(robots_by_id[robot_id], "shepherd_mode", None) == "PUSH"
    ]

    if not active_ids:
        return contact_accelerations

    # Anchor가 지정한 Junction return direction의 PUSH 속도와 이번 step의 목표 이동량 계산
    push_speed = BACKTRACK_PUSH_SPEED_RATIO * environment.INITIAL_SAFE_MAX_SPEED
    desired_world_velocity = anchor_local_to_world(
        pygame.Vector2(push_speed, 0.0), return_yaw_deg
    )
    desired_delta = desired_world_velocity * dt

    # 이동 속도가 사실상 0이면 PUSH하지 않음
    if desired_world_velocity.length_squared() <= environment.EPSILON:
        return contact_accelerations

    # NORMAL을 어느 방향으로 밀어야 하는지 판단하기 위한 Junction return 단위 방향
    return_direction = desired_world_velocity.normalize()
    active_set = set(active_ids)

    # 접촉한 NORMAL에 Shepherd의 물리적 PUSH acceleration을 누적
    def add_normal_contact(
        shepherd: environment.Robot,
        other_id: int,
        other: environment.Robot,
    ) -> None:

        # NORMAL swarm robot에만 PUSH 전달
        if other.role != "NORMAL":
            return

        # Shepherd → NORMAL 방향 계산
        contact_direction = other.position - shepherd.position
        if contact_direction.length_squared() <= environment.EPSILON:
            contact_direction = return_direction.copy()
        else:
            contact_direction = contact_direction.normalize()

        # Shepherd의 Junction return 방향 앞쪽에 있는 NORMAL만 밀어냄
        if contact_direction.dot(return_direction) <= 0.0:
            return

        # 접촉 방향으로 물리적 PUSH acceleration 생성
        contact_acceleration = (
            contact_direction
            * (BACKTRACK_CONTACT_ACCEL_RATIO * environment.MAX_ACCELERATION)
        )

        # 여러 Shepherd가 같은 NORMAL을 밀 수 있으므로 contact acceleration 누적
        contact_accelerations[other_id] = (
            contact_accelerations.get(other_id, pygame.Vector2())
            + contact_acceleration
        )

    # Shepherd formation 전체가 같은 delta로 이동할 때 벽을 통과하지 않는지 검사
    def wall_safe(fraction: float) -> bool:
        delta = desired_delta * fraction
        return all(
            physical_is_walkable(
                robots_by_id[robot_id].position + delta,
                robots_by_id[robot_id].radius,
            )
            for robot_id in active_ids
        )

    # 기본적으로 목표 displacement 전체를 이동
    safe_fraction = 1.0

    # 전체 이동 시 벽에 걸리면 이분 탐색으로 formation 전체가 이동 가능한 최대 비율 계산
    if not wall_safe(1.0):
        low, high = 0.0, 1.0
        for _ in range(14):
            middle = 0.5 * (low + high)
            if wall_safe(middle):
                low = middle
            else:
                high = middle
        safe_fraction = low

    # Shepherd와 다른 physical robot 사이의 충돌을 검사할 목표 movement
    movement = desired_delta
    movement_sq = movement.length_squared()

    if movement_sq > environment.EPSILON:
        for shepherd_id in active_ids:
            shepherd = robots_by_id[shepherd_id]

            for other_id, other in robots_by_id.items():

                # 같은 Backtracking Shepherd formation 내부 로봇끼리는 충돌 검사에서 제외
                if other_id in active_set:
                    continue

                # 현재 Shepherd와 다른 로봇 사이의 상대 위치 및 최소 접촉 거리 계산
                relative_start = shepherd.position - other.position
                minimum_distance = shepherd.radius + other.radius
                c = relative_start.length_squared() - minimum_distance**2

                # 이미 두 로봇이 접촉 또는 약간 overlap된 상태
                if c <= 0.0:
                    moving_into_robot = relative_start.dot(movement) < 0.0

                    # 상대 로봇 쪽으로 더 이동하려 하면 관통을 막고 NORMAL에는 PUSH 전달
                    if moving_into_robot:
                        safe_fraction = 0.0
                        add_normal_contact(shepherd, other_id, other)
                    continue

                # 현재는 떨어져 있지만 이번 movement 중 충돌할지 연속적으로 계산
                a = movement_sq
                b = 2.0 * relative_start.dot(movement)
                discriminant = b * b - 4.0 * a * c

                # 충돌 궤적이 아니면 다음 로봇 검사
                if discriminant < 0.0:
                    continue

                # 현재 step에서 충돌이 발생하는 이동 비율 계산
                hit_fraction = (
                    -b - math.sqrt(discriminant)
                ) / (2.0 * a)

                # 이번 이동 범위 안에서 충돌한다면 충돌 직전까지만 이동하도록 제한
                if 0.0 <= hit_fraction <= safe_fraction:
                    safe_fraction = max(0.0, hit_fraction - 1.0e-4)

                    # 충돌 대상이 NORMAL이면 실제 physical contact acceleration 전달
                    add_normal_contact(shepherd, other_id, other)

    # 벽/로봇 충돌을 고려한 formation 전체의 실제 이동량과 속도 계산
    actual_delta = desired_delta * safe_fraction
    actual_velocity = actual_delta / max(dt, environment.EPSILON)

    # 모든 Shepherd에 동일한 delta를 적용하여 capture 당시 formation topology 유지
    for robot_id in active_ids:
        robot = robots_by_id[robot_id]
        old_position = robot.position.copy()
        robot.previous_position.update(old_position)
        robot.position += actual_delta
        robot.velocity.update(actual_velocity)
        robot.observed_velocity.update(actual_velocity)

        # 실제 이동이 막혀도 motor command 자체는 계속 Junction return direction을 요구
        robot.commanded_velocity.update(desired_world_velocity)

        # Shepherd는 SPH acceleration이 아니라 별도의 rigid PUSH controller로 이동
        robot.acceleration.update(0.0, 0.0)

    # 실제 Pressure Push 결과 출력
    role_debug(
        "physical-pressure-push",
        (
            "[PhysicalPressurePush] "
            f"count={len(active_ids)} requested={desired_delta.length():.4f} "
            f"actual={actual_delta.length():.4f} safe_fraction={safe_fraction:.4f} "
            f"contact_normals={len(contact_accelerations)} topology_locked=True"
        ),
    )

    # 접촉으로 물리적 PUSH를 받은 NORMAL별 acceleration 반환
    return contact_accelerations



# [Final Base Push] 단계
# 전체 탐색 종료 후 final Shepherd들을 Base 방향으로 이동시켜
# 남아 있는 swarm을 최종적으로 Base 쪽으로 밀어낸다.
def update_final_base_push(
    final_push_ids: set[int],
    robots_by_id: dict[int, environment.Robot],
    reference_yaw_deg: float,
    dt: float,
) -> None:

    # Final Push 이동 속도 설정
    push_speed = 0.35 * environment.INITIAL_SAFE_MAX_SPEED

    # Junction 기준 진행방향의 반대(-x), 즉 Base 방향의 local velocity 생성
    local_command = pygame.Vector2(-push_speed, 0.0)

    # Base 방향 local velocity를 world 좌표계의 velocity로 변환
    world_velocity = anchor_local_to_world(local_command, reference_yaw_deg)

    # Final Push에 선택된 Shepherd들을 Base 방향으로 이동
    for robot_id in final_push_ids:
        robot = robots_by_id[robot_id]

        # 현재 SHEPHERD 역할인 로봇만 Final Push 수행
        if robot.role != "SHEPHERD":
            continue

        # 이번 dt 동안 이동했을 때의 다음 위치 계산
        next_position = robot.position + world_velocity * dt

        # 다음 위치가 벽과 충돌하지 않을 때만 실제 이동
        if physical_is_walkable(next_position, robot.radius):
            robot.previous_position.update(robot.position)
            robot.position.update(next_position)
            robot.velocity.update(world_velocity)

        # 벽에 막히면 해당 Shepherd 정지
        else:
            robot.velocity.update(0.0, 0.0)


# Backtracking - Reverse Flow Evaluation
# Shepherd의 물리적 Pressure Push로 인해 앞쪽 NORMAL swarm에
# 실제 Junction 방향 reverse flow가 형성되었는지 observed velocity로 판정한다.
# NORMAL에 별도의 backtracking/goal motion을 부여하지 않는다.
def evaluate_normal_reverse_flow(
    observation: LocalObservation,
    robots_by_id: dict[int, environment.Robot],
    chain_order: list[int],
    corridor_width: float,
) -> tuple[bool, int, int, float, float]:

    # 각 로봇의 Anchor-relative local 위치와 실제 관측 속도를 ID별로 저장
    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(observation.robot_ids, observation.relative_positions)
    }
    local_velocity_by_id = {
        robot_id: local_velocity
        for robot_id, local_velocity
        in zip(observation.robot_ids, observation.velocities)
    }

    # Shepherd formation의 local x값을 모아 중앙값을 현재 Shepherd 위치 기준으로 사용
    shepherd_x_values = sorted(
        local_position_by_id[robot_id].x
        for robot_id in chain_order
        if robot_id in local_position_by_id
    )

    # 관측 가능한 Shepherd가 없으면 reverse flow 판정 불가
    if not shepherd_x_values:
        return False, 0, 0, 0.0, 0.0

    shepherd_x = shepherd_x_values[len(shepherd_x_values) // 2]

    # Shepherd 앞쪽에서 reverse flow를 검사할 범위를 COMM_RANGE 기반으로 설정
    evaluation_depth = BACKTRACK_FLOW_EVAL_HOPS * environment.COMM_RANGE
    half_width = 0.5 * corridor_width + environment.GRID_SPACING

    eligible_ids: list[int] = []
    reverse_speeds: list[float] = []

    # Shepherd 앞쪽의 NORMAL들만 대상으로 실제 observed velocity 검사
    for robot_id, local_position in local_position_by_id.items():
        robot = robots_by_id[robot_id]

        # NORMAL만 평가
        if robot.role != "NORMAL":
            continue

        # Shepherd보다 Dead-end 쪽에 있는 로봇은 제외
        if local_position.x < shepherd_x - environment.GRID_SPACING:
            continue

        # Shepherd 앞쪽이라도 평가 범위보다 너무 멀리 있는 로봇은 제외
        if local_position.x > shepherd_x + evaluation_depth:
            continue

        # 현재 corridor 폭 밖의 로봇은 제외
        if abs(local_position.y) > half_width:
            continue

        # 해당 NORMAL의 실제 Anchor-local observed velocity 확인
        local_velocity = local_velocity_by_id.get(robot_id)
        if local_velocity is None:
            continue

        eligible_ids.append(robot_id)

        # Backtracking 중 +x가 Junction 복귀 방향이므로 x속도를 return speed로 사용
        return_speed = local_velocity.x

        # 일정 속도 이상 Junction 방향으로 움직이면 reverse-flow robot으로 인정
        if return_speed >= BACKTRACK_REVERSE_SPEED_THRESHOLD:
            reverse_speeds.append(return_speed)

    # 평가 대상 NORMAL 수와 실제 Junction 방향으로 움직이는 NORMAL 수 계산
    eligible_count = len(eligible_ids)
    reverse_count = len(reverse_speeds)

    # 평가할 NORMAL이 없으면 reverse flow가 형성되지 않은 것으로 판정
    if eligible_count == 0:
        return False, 0, 0, 0.0, 0.0

    # 평가 대상 중 실제 reverse flow를 보이는 NORMAL의 비율 계산
    reverse_ratio = reverse_count / eligible_count

    # Junction 방향으로 움직이는 NORMAL들의 평균 복귀 속도 계산
    mean_reverse_speed = (
        sum(reverse_speeds) / len(reverse_speeds)
        if reverse_speeds else 0.0
    )

    # 최소 로봇 수와 reverse-flow 비율을 모두 만족하면 Flow Backtracking 시작 가능
    ready = (
        eligible_count >= BACKTRACK_REVERSE_MIN_ROBOTS
        and reverse_ratio >= BACKTRACK_REVERSE_RATIO_THRESHOLD
    )

    # 판정 결과와 진단값 반환
    return ready, eligible_count, reverse_count, reverse_ratio, mean_reverse_speed



# Backtracking Shepherd - Prepare Pressure Push
# 이미 형성이 끝난 Backtracking Shepherd 전체에게 PUSH 명령을 relay하고, 
# 모두 PUSH 명령을 받은 경우에만 Pressure Push를 시작하도록 허용하는 함수
def prepare_backtracking_shepherd_push(
    observation: LocalObservation,
    chain_order: list[int],
    robots_by_id: dict[int, environment.Robot],
    seed_id: int,
    active_branch_id: str,
) -> bool:

    # 현재 Backtracking Shepherd formation 전체 ID 구성
    formation_ids = set(chain_order)

    # Hop 0 seed에서 formation 전체로 PUSH 명령을 multi-hop relay
    received_ids = relay_backtracking_shepherd_command(
        observation,
        robots_by_id,
        seed_id,
        formation_ids,
        active_branch_id,
        command="PUSH",
    )

    # formation의 모든 Shepherd가 PUSH 명령을 받았는지 확인
    all_received = (received_ids == formation_ids)

    # 한 대라도 PUSH 명령을 받지 못했다면 Pressure Push 시작 보류
    if not all_received:
        print(
            "[BacktrackingPushRelayIncomplete] "
            f"branch={active_branch_id} "
            f"received={len(received_ids)} "
            f"required={len(formation_ids)}"
        )
        return False

    # formation 전체가 PUSH mode로 전환되어 실제 물리적 밀기 준비 완료
    print(
        "[BacktrackingShepherdReadyForPush] "
        f"branch={active_branch_id} "
        f"seed={seed_id} "
        f"count={len(chain_order)}"
    )

    return True



# Shepherd - Release to Normal
# 역할을 마친 Shepherd를 NORMAL로 복귀시키고,
# 기존 Shepherd 역할 및 motion state를 모두 초기화한다.
def release_robot_to_normal(
    robot: environment.Robot,
) -> None:

    # Shepherd 역할을 해제하고 일반 swarm robot으로 복귀
    robot.role = "NORMAL"
    robot.role_branch = None

    # 역할에 의해 고정된 상태가 있다면 해제
    robot.role_frozen = False
    robot.role_frozen_position = None

    # FORM/PUSH 등 Shepherd 전용 동작 mode 제거
    robot.shepherd_mode = None

    # 이전 Shepherd 동작이 NORMAL 움직임에 남지 않도록 motion state 초기화
    robot.velocity.update(0.0, 0.0)
    robot.acceleration.update(0.0, 0.0)
    robot.filtered_acceleration.update(0.0, 0.0)
    robot.commanded_velocity.update(0.0, 0.0)
    robot.observed_velocity.update(0.0, 0.0)



# Shepherd Group - Release to Normal
# 지정된 Shepherd 그룹 전체를 NORMAL로 복귀시키고,
# Shepherd 동작을 제거하여 이후 다시 SPH에 따라 움직이도록 한다.
def release_shepherd_group(
    robot_ids: set[int],
    robots_by_id: dict[int, environment.Robot],
) -> None:

    # 지정된 그룹의 모든 로봇을 순회
    for robot_id in robot_ids:
        robot = robots_by_id[robot_id]

        # 현재 SHEPHERD인 로봇만 release
        if robot.role != "SHEPHERD":
            continue

        # SHEPHERD → NORMAL로 복귀하고 Branch 및 고정 상태 제거
        robot.role = "NORMAL"
        robot.role_branch = None
        robot.role_frozen = False
        robot.role_frozen_position = None

        # Shepherd의 기존 움직임이 남지 않도록 motion state 초기화
        # 별도의 복귀 방향은 주지 않으며 이후 움직임은 다시 SPH가 결정
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
        robot.commanded_velocity.update(0.0, 0.0)
        robot.observed_velocity.update(0.0, 0.0)



# Branch Swarm Return Completion 확인
# 현재 Branch 내부에 Backtracking Shepherd가 남아 있는지 검사하여, 해당 Branch의 물리적 Backtracking 완료 여부를 판정한다.
# NORMAL 잔류 로봇 수도 함께 수집해 진단값으로 반환한다.
def branch_swarm_return_complete(
    observation: LocalObservation,
    branch: dict,
    robots_by_id: dict[int, environment.Robot],
    backtrack_ids: set[int],
) -> tuple[bool, list[int], list[int]]:

    # Branch 안쪽으로 향하는 진행축과 입구를 가로지르는 횡방향 축
    axis = branch["branch_axis"].normalize()
    tangent = branch["entrance_tangent"].normalize()

    # Branch entrance의 중심과 실제 입구 폭
    entrance_midpoint = branch["entrance_midpoint"]
    entrance_width = (branch["entrance_b"] - branch["entrance_a"]).length()
    half_width = 0.5 * entrance_width

    # 로봇 반경과 오차를 고려해 Branch 내부 판정 범위에 margin 추가
    depth_margin = environment.ROBOT_RADIUS + ENTRANCE_CAPTURE_MARGIN
    lateral_margin = environment.ROBOT_RADIUS + ENTRANCE_CAPTURE_MARGIN

    remaining_normals: list[int] = []
    remaining_backtrack: list[int] = []

    # Anchor-local 관측 위치를 이용해 아직 Branch 내부에 있는 로봇 검사
    for robot_id, local_position in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):
        # LiDAR Anchor 자신은 검사 대상에서 제외
        if robot_id == observation.lidar_robot_id:
            continue

        robot = robots_by_id[robot_id]

        # Branch entrance 중심을 기준으로 한 상대 위치
        offset = local_position - entrance_midpoint

        # Branch 안쪽 방향 거리와 입구 횡방향 거리 계산
        depth = offset.dot(axis)
        lateral = offset.dot(tangent)

        # 현재 로봇이 해당 Branch corridor 폭 안에 있는지 검사
        inside_branch_width = abs(lateral) <= half_width + lateral_margin
        if not inside_branch_width:
            continue

        # entrance를 지나 아직 Branch 안쪽에 남아 있는지 검사
        still_inside_branch = depth > depth_margin
        if not still_inside_branch:
            continue

        # Branch 내부에 남아 있는 NORMAL 기록
        if robot.role == "NORMAL":
            remaining_normals.append(robot_id)

        # Branch 내부에 남아 있는 Backtracking Shepherd 기록
        if robot_id in backtrack_ids:
            remaining_backtrack.append(robot_id)

    # Backtracking Shepherd가 모두 Branch를 빠져나오면 물리적 복귀 완료
    complete = len(remaining_backtrack) == 0

    return complete, remaining_normals, remaining_backtrack



# Swarm - Final Base Return Completion Check
# Root center 기준 Anchor-local 좌표에서 모든 swarm robot이 Base corridor의 최종 회수 영역까지 복귀했는지 판정한다.
def swarm_base_return_complete(
    observation: LocalObservation,
) -> bool:

    # Root center에서 Base 최종 회수 영역까지 필요한 -x 방향 복귀 깊이 계산
    base_return_depth = (
        ROOT_CENTER_TO_BASE_DISTANCE
        - 0.5 * KNOWN_BASE_CORRIDOR_LENGTH
    )

    # Base corridor 폭에 로봇 반경을 더해 허용 가능한 횡방향 범위 설정
    base_lateral_limit = (
        0.5 * KNOWN_CORRIDOR_WIDTH
        + environment.ROBOT_RADIUS
    )

    # 관측된 모든 swarm robot의 Anchor-local 위치 검사
    for robot_id, local_position in zip(
        observation.robot_ids,
        observation.relative_positions,
    ):

        # Root center에 있는 LiDAR Anchor는 swarm 복귀 검사에서 제외
        if robot_id == observation.lidar_robot_id:
            continue

        # 아직 Base 방향으로 충분히 복귀하지 않았거나
        # Base corridor 폭을 벗어난 로봇이 한 대라도 있으면 미완료
        if (
            local_position.x > -base_return_depth
            or abs(local_position.y) > base_lateral_limit
        ):
            return False

    # Anchor를 제외한 모든 swarm robot이 Base 회수 영역에 들어오면 최종 복귀 완료
    return True




# Corridor Width Estimation
# LiDAR로 측정한 좌·우 side wall 거리를 이용해 현재 corridor 폭을 추정한다.
# Junction opening 등으로 측정값이 비정상적이면 기본 corridor 폭을 사용한다.
def estimate_current_corridor_width(
    scan: LidarScan,
) -> float:

    # LiDAR에서 현재 로봇의 좌측/우측 벽까지의 거리 추출
    left_range, right_range = extract_lateral_wall_ranges(scan)

    # 양쪽 모두 실제 side wall로 측정 가능한 유효한 거리인지 확인
    if (
        math.isfinite(left_range)
        and math.isfinite(right_range)
        and left_range < scan.max_range - 1.0
        and right_range < scan.max_range - 1.0
    ):

        # 좌측 벽 거리 + 우측 벽 거리로 현재 corridor 폭 추정
        width = left_range + right_range

        # Junction의 lateral opening을 벽으로 잘못 사용하지 않도록
        # 알려진 corridor 폭의 75~125% 범위에 있는 측정값만 인정
        if (
            0.75 * KNOWN_CORRIDOR_WIDTH
            <= width
            <= 1.25 * KNOWN_CORRIDOR_WIDTH
        ):
            return width

    # 정상적인 좌·우 side wall pair를 얻지 못하면 기본 corridor 폭 사용
    return KNOWN_CORRIDOR_WIDTH



# Outgoing Branch Registration
# LiDAR로 opening 방향을 찾고 → 뒤쪽 parent corridor를 제외하고 → 나머지 opening마다 Branch ID와 입구 geometry를 만드는 함수
# Junction 중앙에서 검출된 LiDAR opening들을 이용해 각 outgoing Branch의 Anchor-local entrance geometry를 생성한다.
def register_outgoing_branches(
    lidar: AnchorLidar,
) -> list[dict]:

    # 검출된 opening이 없으면 등록할 Branch 없음
    if not lidar.opening_groups:
        return []

    # 현재 Junction에서 검출된 모든 opening 가져오기
    openings = list(lidar.opening_groups)

    # 검출된 opening의 중심각과 angular sector 출력
    print(
        "[CenterOpeningGroups] "
        + " ".join(
            f"center={opening['center_angle']:.1f} "
            f"sector={opening['start_angle']:.1f}~{opening['end_angle']:.1f}"
            for opening in openings
        )
    )

    # Anchor 뒤쪽(180° ±45°) opening을 parent corridor 후보로 탐색
    rear_candidates = [
        opening for opening in openings
        if circular_error(opening["center_angle"], 180.0) <= 45.0
    ]

    # 실제 rear opening이 있으면 180°에 가장 가까운 opening을 parent corridor로 판단하여 제외
    if rear_candidates:
        parent = min(
            rear_candidates,
            key=lambda opening: circular_error(opening["center_angle"], 180.0),
        )
        outgoing_openings = [opening for opening in openings if opening is not parent]

        print(
            "[ParentOpeningDetected] "
            f"center={parent['center_angle']:.1f}"
        )

    # Root Junction처럼 rear opening이 검출되지 않으면 모든 opening을 outgoing으로 유지
    else:
        outgoing_openings = openings
        print("[ParentOpeningNotVisible] keep_all_current_openings=True")

    # Branch ID를 일관되게 부여하기 위해 local bearing 순서로 정렬
    outgoing_openings.sort(
        key=lambda opening: normalize_angle(opening["center_angle"])
    )

    branches: list[dict] = []

    # 각 outgoing opening을 하나의 Branch로 등록
    for index, opening in enumerate(outgoing_openings):
        branch = dict(opening)
        branch["id"] = f"B{index}"

        # LiDAR opening center angle로 Branch 진행방향 axis 생성
        center_rad = math.radians(branch["center_angle"])
        axis = pygame.Vector2(
            math.cos(center_rad),
            math.sin(center_rad),
        ).normalize()

        # Branch axis에 수직인 entrance 횡방향 tangent 생성
        tangent = pygame.Vector2(-axis.y, axis.x)

        # Junction 중앙에서 Branch entrance까지의 거리와 entrance 반폭 설정
        half_junction_depth = 0.5 * KNOWN_JUNCTION_DEPTH
        half_corridor_width = 0.5 * KNOWN_CORRIDOR_WIDTH

        # Junction 중앙 기준 Branch entrance 중심 계산
        entrance_midpoint = axis * half_junction_depth

        # entrance 중심에서 tangent 양쪽으로 이동하여 실제 입구 양 끝점 생성
        entrance_a = entrance_midpoint - tangent * half_corridor_width
        entrance_b = entrance_midpoint + tangent * half_corridor_width

        # 생성한 Anchor-local Branch geometry 저장
        branch["branch_axis"] = axis
        branch["entrance_tangent"] = tangent
        branch["entrance_midpoint"] = entrance_midpoint
        branch["entrance_a"] = entrance_a
        branch["entrance_b"] = entrance_b

        # 등록된 Branch entrance geometry 출력
        print(
            "[TrueBranchEntrance] "
            f"id={branch['id']} "
            f"center_angle={branch['center_angle']:.1f} "
            f"A=({entrance_a.x:.2f},{entrance_a.y:.2f}) "
            f"B=({entrance_b.x:.2f},{entrance_b.y:.2f})"
        )

        branches.append(branch)

    # 등록된 모든 outgoing Branch 반환
    return branches



# Initial Shepherd Role Assignment
# “이 로봇은 이제 이 Branch의 Initial Shepherd야”라고 역할만 바꾸는 함수 (SHEPHERD 역할만 주고 계속 SPH로 움직이게 하는 것)
# Junction 입구 형성 중 선택된 로봇을 임시 SHEPHERD로 전환한다.
# 아직 입구가 완성되지 않았으므로 고정하지 않고 SPH에 따라 계속 이동한다.
def assign_initial_shepherd_role(
    robot: environment.Robot,
    branch_id: str,
) -> None:

    # 해당 로봇을 Initial Shepherd로 전환
    robot.role = "SHEPHERD"

    # 이 Shepherd가 담당하는 Branch 저장
    robot.role_branch = branch_id

    # 아직 입구 형성 중이므로 위치를 고정하지 않음
    robot.role_frozen = False
    robot.role_frozen_position = None



# Role Robot Freeze
# “이 로봇에게 역할을 부여한 다음, 지금 서 있는 바로 그 자리에 고정해라”
# 로봇에 지정된 역할과 Branch를 부여하고, 현재 물리적 위치에서 완전히 정지·고정한다.
def freeze_role_robot(
    robot: environment.Robot,
    role: str,
    branch_id: str,
) -> None:

    # 지정된 역할과 담당 Branch 부여
    robot.role = role
    robot.role_branch = branch_id

    # 현재 위치를 역할 수행 위치로 고정
    robot.role_frozen = True
    robot.role_frozen_position = robot.position.copy()
    robot.previous_position.update(robot.position)

    # 기존 움직임이 남지 않도록 모든 motion state 초기화
    robot.velocity.update(0.0, 0.0)
    robot.acceleration.update(0.0, 0.0)
    robot.filtered_acceleration.update(0.0, 0.0)
    robot.commanded_velocity.update(0.0, 0.0)
    robot.observed_velocity.update(0.0, 0.0)



# Initial Branch Marker Creation
# 모든 Initial Shepherd boundary가 완성된 뒤, 각 Branch마다 가장 안쪽에 있으면서 중심선에 가까운 Shepherd 1대를 Marker로 남긴다.
def create_initial_branch_markers(
    observation: LocalObservation,
    branches: list[dict],
    branch_states: dict,
    robots_by_id: dict[int, environment.Robot],
) -> None:

    # Anchor-relative local 위치를 robot ID별로 저장
    local_position_by_id = {
        robot_id: local_position
        for robot_id, local_position
        in zip(observation.robot_ids, observation.relative_positions)
    }

    # 각 runtime Branch마다 Marker 1대 생성
    for branch in branches:
        branch_id = branch["id"]
        state = branch_states[branch_id]

        # 이미 Marker가 생성된 Branch는 제외
        if state["marker_id"] is not None:
            continue

        # 해당 Branch의 Initial Shepherd 집합 가져오기
        shepherd_ids = set(state["initial_shepherd_ids"])
        if not shepherd_ids:
            raise RuntimeError(f"No Initial Shepherds for {branch_id}")

        # 현재 Anchor observation에서 실제 관측되는 Shepherd만 사용
        shepherd_ids = {
            robot_id for robot_id in shepherd_ids
            if robot_id in local_position_by_id
        }
        if not shepherd_ids:
            raise RuntimeError(f"No observable Initial Shepherds for {branch_id}")

        # Branch-local 진행방향(axis), 폭방향(tangent), 입구 중심 가져오기
        axis = branch["branch_axis"].normalize()
        tangent = branch["entrance_tangent"].normalize()
        entrance_midpoint = branch["entrance_midpoint"]

        # Branch 입구에서 안쪽으로 얼마나 깊이 들어가 있는지 계산
        def branch_depth(robot_id: int) -> float:
            offset = local_position_by_id[robot_id] - entrance_midpoint
            return offset.dot(axis)

        # Branch 중심선에서 좌우로 얼마나 떨어져 있는지 계산
        def lateral_offset(robot_id: int) -> float:
            offset = local_position_by_id[robot_id] - entrance_midpoint
            return abs(offset.dot(tangent))

        # Initial Shepherd 중 가장 깊은 위치 확인
        max_depth = max(branch_depth(robot_id) for robot_id in shepherd_ids)

        # 최심부 로봇 주변 GRID_SPACING 이내의 Shepherd들을 Marker 후보로 선정
        depth_margin = environment.GRID_SPACING
        deepest_candidates = {
            robot_id for robot_id in shepherd_ids
            if branch_depth(robot_id) >= max_depth - depth_margin
        }

        if not deepest_candidates:
            raise RuntimeError(f"No deepest Marker candidates for {branch_id}")

        # 최심부 후보 중 Branch 중심선에 가장 가까운 Shepherd 1대를 Marker로 선택
        marker_id = min(deepest_candidates, key=lateral_offset)
        marker = robots_by_id[marker_id]

        # Debug용 Marker의 depth와 중심선 거리 저장
        marker_depth = branch_depth(marker_id)
        marker_lateral = lateral_offset(marker_id)

        # 선택된 Shepherd를 MARKER로 전환하고 현재 위치에 고정
        freeze_role_robot(marker, "MARKER", branch_id)

        # Marker가 된 로봇은 Initial Shepherd 집합에서 제거
        state["initial_shepherd_ids"].discard(marker_id)

        # 해당 Branch의 Marker로 등록하고 초기 상태를 UNVISITED로 설정
        state["marker_id"] = marker_id
        state["marker_state"] = "UNVISITED"

        # 생성된 Marker 정보 출력
        print(
            "[InitialMarkerCreated] "
            f"branch={branch_id} "
            f"marker_id={marker_id} "
            f"max_depth={max_depth:.3f} "
            f"marker_depth={marker_depth:.3f} "
            f"lateral_offset={marker_lateral:.3f} "
            f"deepest_candidates={len(deepest_candidates)} "
            f"state=UNVISITED"
        )



# Runtime Branch Selection
# DFS가 선택한 UNVISITED Branch를 ACTIVE로 전환하고, 해당 Branch 탐색 시작 명령을 출력한다.
def select_runtime_branch(
    branch_id: str,
    branch_states: dict,
) -> None:

    # 선택하려는 Branch의 현재 상태 가져오기
    state = branch_states[branch_id]

    # UNVISITED Branch가 아니면 다시 선택하지 않음
    if state["visit_state"] != "UNVISITED":
        return

    # 선택된 Branch를 현재 탐색 중인 ACTIVE 상태로 전환
    state["visit_state"] = "ACTIVE"

    # Branch 선택 상태 출력
    print(
        "[BranchSelected] "
        f"branch={branch_id} "
        "state=ACTIVE"
    )

    # Anchor가 해당 Branch 탐색 시작 명령을 전달했음을 출력
    print(
        "[AnchorBroadcast] "
        f"EXPLORE {branch_id}"
    )



# Selected Branch Opening
# 선택된 Branch를 막고 있던 Initial Shepherd들을 NORMAL로 해제하여 입구를 열고, Marker는 그대로 남겨 Branch의 물리적 표시를 유지한다.
def open_selected_branch(
    branch_id: str,
    branch_states: dict,
    robots_by_id: dict[int, environment.Robot],
) -> None:

    # 선택된 Branch의 현재 상태 가져오기
    state = branch_states[branch_id]

    # 해당 Branch 입구를 막고 있는 Initial Shepherd ID 집합
    shepherd_ids = set(state["initial_shepherd_ids"])

    # Branch에 남겨둘 Marker ID
    marker_id = state["marker_id"]

    # Branch 개방 시작 상태 출력
    print(
        "[BranchOpenStart] "
        f"branch={branch_id} "
        f"shepherd_count={len(shepherd_ids)} "
        f"marker_id={marker_id}"
    )

    # Initial Shepherd들에게 역할 해제 명령을 전달했음을 출력
    print(
        "[AnchorBroadcast] "
        f"branch={branch_id} "
        "command=RELEASE_INITIAL_SHEPHERD "
        f"ids={sorted(shepherd_ids)}"
    )

    # Initial Shepherd들을 NORMAL로 되돌려 Branch 입구를 개방
    # Marker는 이미 shepherd_ids에서 제외되어 있으므로 그대로 유지됨
    release_shepherd_group(shepherd_ids, robots_by_id)

    # 해당 Branch가 물리적으로 개방되었음을 상태에 기록
    state["opened"] = True

    # Marker를 유지한 채 Branch 개방 완료
    print(
        "[BranchOpened] "
        f"branch={branch_id} "
        f"marker_retained={marker_id}"
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


# Map Drawing
# 시뮬레이션의 맵 영역, 외벽, 중앙 장애물과 Base 위치를 화면에 그린다.
# 로봇의 탐색/제어 알고리즘에는 관여하지 않는 시각화 함수이다.
def draw_map(
    surface: pygame.Surface,
    font: pygame.font.Font,
) -> None:

    # 전체 맵이 표시되는 패널 배경 그리기
    pygame.draw.rect(
        surface,
        COLORS["panel"],
        MAP_PANEL,
        border_radius=8,
    )

    # 로봇이 이동할 수 있는 맵 내부 영역 그리기
    pygame.draw.polygon(
        surface,
        COLORS["floor"],
        OUTER_BOUNDARY,
    )

    # 맵의 외곽 벽 그리기
    pygame.draw.polygon(
        surface,
        COLORS["wall"],
        OUTER_BOUNDARY,
        width=WALL_DRAW_WIDTH,
    )

    # 중앙 장애물의 내부 영역 그리기
    pygame.draw.rect(
        surface,
        COLORS["background"],
        OBSTACLE_RECT,
    )

    # 중앙 장애물의 벽 경계 그리기
    pygame.draw.rect(
        surface,
        COLORS["wall"],
        OBSTACLE_RECT,
        width=WALL_DRAW_WIDTH,
    )

    # Base 위치를 표시할 텍스트 생성
    label = font.render(
        "Base",
        True,
        COLORS["text"],
    )

    # Base 좌표를 기준으로 텍스트 위치 설정
    label_rect = label.get_rect(
        midtop=(
            round(BASE_POSITION.x),
            round(BASE_POSITION.y) + 6,
        )
    )

    # 화면에 Base 텍스트 표시
    surface.blit(
        label,
        label_rect,
    )



# Robot / LiDAR Visualization
# LiDAR가 opening으로 판단한 ray와 각 역할별 로봇,
# Anchor의 위치·Junction 검출 상태·이동 방향을 화면에 표시한다.
# 탐색/제어에는 관여하지 않는 시각화 함수이다.
def draw_robots(
    surface: pygame.Surface,
    robots: list[environment.Robot],
    color_reference_density: float,
    anchor: environment.Robot,
    lidar: AnchorLidar,
    junction_detected: bool,
    anchor_yaw_degrees: float,
) -> None:

    # 일반 NORMAL 로봇의 기본 색상과 표시 크기
    robot_color = (220, 225, 230)
    display_radius = ROBOT_DRAW_RADIUS

    # LiDAR가 opening support로 판단한 ray만 4개 간격으로 화면에 표시
    open_ray_stride = 4
    for ray_index, (angle, measured_range, is_open) in enumerate(
        zip(lidar.angles, lidar.ranges, lidar.open_support)
    ):
        # Opening ray가 아니거나 표시 간격에 해당하지 않으면 생략
        if not is_open or ray_index % open_ray_stride != 0:
            continue

        # Anchor-local LiDAR 각도를 화면상의 world 방향으로 변환
        radians = math.radians(anchor_yaw_degrees + angle)

        # 해당 방향으로 LiDAR 최대거리까지 ray 끝점 계산
        endpoint = anchor.position + pygame.Vector2(
            math.cos(radians),
            math.sin(radians),
        ) * LIDAR_MAX_RANGE

        # Opening을 나타내는 LiDAR ray 그리기
        pygame.draw.line(
            surface,
            COLORS["open_beam"],
            anchor.position,
            endpoint,
            width=1,
        )

    # Anchor를 제외한 모든 로봇을 역할에 따라 표시
    for robot in robots:
        if robot is anchor:
            continue

        # 로봇의 화면 표시 중심 좌표
        center = (
            round(robot.position.x),
            round(robot.position.y),
        )

        # MARKER: 주황색 본체와 외곽 테두리로 표시
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

        # NORMAL: 회색 본체와 진한 외곽선으로 표시
        if robot.role == "NORMAL":
            pygame.draw.circle(
                surface,
                (25, 45, 90),
                center,
                3,
            )
            pygame.draw.circle(
                surface,
                robot_color,
                center,
                ROBOT_DRAW_RADIUS,
            )
            continue

        # SHEPHERD: Shepherd 전용 색상으로 표시
        if robot.role == "SHEPHERD":
            pygame.draw.circle(
                surface,
                COLORS["shepherd"],
                center,
                display_radius,
            )
            continue

        # 그 외 역할의 로봇은 기본 색상으로 표시
        pygame.draw.circle(
            surface,
            robot_color,
            center,
            display_radius,
        )

    # Anchor를 leader 전용 색상으로 표시
    pygame.draw.circle(
        surface,
        COLORS["leader"],
        (round(anchor.position.x), round(anchor.position.y)),
        3.0,
    )

    # Junction이 검출된 경우 Anchor 주변에 검출 표시 추가
    if junction_detected:
        pygame.draw.circle(
            surface,
            COLORS["detected"],
            (round(anchor.position.x), round(anchor.position.y)),
            6.0,
            width=2,
        )

    # Anchor가 이동 중이면 현재 velocity 방향을 선으로 표시
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



# Control / Anchor State Visualization
# 조작키·실행/통신 상태와 현재 Anchor 상태 머신 단계, 탐색 중인 Branch를 화면에 표시하는 시각화 함수이다.
def draw_controls(
    surface: pygame.Surface,
    font: pygame.font.Font,
    paused: bool,
    show_comm_links: bool,
    anchor_motion_mode: str,
    active_branch_id: str | None,
) -> None:

    # 현재 실행 상태와 Communication 표시 상태
    run_state = "PAUSED" if paused else "RUNNING"
    comm_state = "ON" if show_comm_links else "OFF"

    # 화면 하단에 조작키와 현재 상태 표시
    text = (
        f"SPACE: Pause  |  R: Reset  |  "
        f"C: Communication [{comm_state}]  |  {run_state}"
    )
    rendered = font.render(text, True, COLORS["text"])
    surface.blit(rendered, (18, WINDOW_SIZE[1] - 28))

    # 긴 Anchor 상태 이름을 underscore 기준으로 나누어 여러 줄로 구성
    stage_lines: list[str] = []
    current_line = ""

    for word in anchor_motion_mode.split("_"):
        candidate = word if not current_line else f"{current_line} {word}"

        # 한 줄이 너무 길면 현재 줄을 저장하고 다음 줄 시작
        if current_line and len(candidate) > 16:
            stage_lines.append(current_line)
            current_line = word
        else:
            current_line = candidate

    # 마지막 상태 문자열 추가
    if current_line:
        stage_lines.append(current_line)

    # Anchor 상태를 표시할 왼쪽 패널 영역 생성
    status_rect = pygame.Rect(12, 42, MAP_PANEL.left - 24, 132)

    # 상태 패널의 배경과 테두리 표시
    pygame.draw.rect(surface, COLORS["panel"], status_rect, border_radius=6)
    pygame.draw.rect(surface, COLORS["wall"], status_rect, width=1, border_radius=6)

    # 현재 Anchor stage와 ACTIVE Branch 정보 구성
    status_font = pygame.font.Font(None, 16)
    status_rows = [
        "ANCHOR STAGE",
        *stage_lines,
        f"BRANCH: {active_branch_id or '-'}",
    ]

    # 상태 정보를 패널에 한 줄씩 표시
    for row_index, row in enumerate(status_rows):
        row_color = COLORS["open_beam"] if row_index == 0 else COLORS["stage_text"]
        row_surface = status_font.render(row, True, row_color)
        surface.blit(
            row_surface,
            (status_rect.left + 8, status_rect.top + 8 + row_index * 17),
        )



# Initial Robot Grid / Anchor Creation
# Base 영역에 로봇들을 staggered grid 형태로 배치하고, 군집 중앙보다 앞쪽 row의 중앙 로봇을 초기 Anchor로 선택한다.
def create_staggered_grid_robots(
    robot_count: int,
) -> list[environment.Robot]:

    global INITIAL_ANCHOR_ID

    # 초기 Anchor와 로봇/배치정보 초기화
    INITIAL_ANCHOR_ID = None
    robots: list[environment.Robot] = []
    spawn_slots: list[tuple[int, int, int, int]] = []

    # 로봇 간 가로/세로 배치 간격
    dx = environment.GRID_SPACING
    dy = math.sqrt(3.0) / 3.0 * dx

    # Base 내부에서 로봇을 배치할 수 있는 좌우/상하 범위 계산
    usable_left = (
        environment.center_x - environment.half_width
        + environment.ROBOT_RADIUS + environment.INITIAL_GRID_SIDE_MARGIN
    )
    usable_right = (
        environment.center_x + environment.half_width
        - environment.ROBOT_RADIUS - environment.INITIAL_GRID_SIDE_MARGIN
    )
    top = (
        environment.center_y + environment.half_width
        + 12.0 * environment.MAP_SCALE
    )
    bottom = (
        environment.center_y + environment.half_width + environment.base_length
        - environment.ROBOT_RADIUS - 7.0 * environment.MAP_SCALE
    )

    # 한 row에 배치할 수 있는 최대 로봇 수
    max_columns = max(1, int((usable_right - usable_left) // dx) + 1)

    # Base 아래쪽부터 위쪽으로 staggered grid 형태의 로봇 생성
    robot_id = 0
    row = 0

    while robot_id < robot_count:
        y = bottom - row * dy

        # Base 배치 영역을 벗어나면 생성 종료
        if y < top:
            print(f"Warning: only {len(robots)} robots fit.")
            break

        # 홀수 row는 한 칸 줄여 staggered 형태 생성
        row_count = max_columns if row % 2 == 0 else max(1, max_columns - 1)
        row_width = (row_count - 1) * dx
        row_left = environment.center_x - 0.5 * row_width

        # 현재 row의 로봇 생성 및 배치정보 저장
        for column in range(row_count):
            if robot_id >= robot_count:
                break

            robot = environment.Robot(row_left + column * dx, y, robot_id)
            robots.append(robot)
            spawn_slots.append((row, column, row_count, robot_id))
            robot_id += 1

        row += 1

    # 생성된 로봇이 없으면 그대로 반환
    if not spawn_slots:
        return robots

    # 실제 로봇이 배치된 row 목록
    used_rows = sorted({
        row for row, _, _, _ in spawn_slots
    })

    # 전체 군집 중앙보다 앞쪽에 있는 row를 Anchor row로 선택
    center_index = len(used_rows) // 2
    anchor_index = min(len(used_rows) - 1, center_index + 16)
    anchor_row = used_rows[anchor_index]

    # 선택된 Anchor row에 속한 로봇들
    anchor_row_slots = [
        item for item in spawn_slots
        if item[0] == anchor_row
    ]

    # 해당 row에서 좌우 중앙에 가장 가까운 로봇을 초기 Anchor로 선택
    _, _, _, INITIAL_ANCHOR_ID = min(
        anchor_row_slots,
        key=lambda item: abs(item[1] - (item[2] - 1) / 2.0),
    )

    # 선택된 초기 Anchor 정보 출력
    print(
        "[InitialAnchor] "
        f"id={INITIAL_ANCHOR_ID} "
        f"row={anchor_row} "
        "source=FORWARD_DEPLOYMENT_SLOT"
    )

    return robots




# [1] 주변 로봇 탐색
#     ↓
# Spatial Grid에서 주변 cell의 로봇들을 후보로 탐색
#     ↓
# 실제 거리 ≤ Smoothing Length h 인 로봇만 SPH 이웃으로 사용

# [2] SPH Pressure Force 계산
#     ↓
# 각 이웃의 density와 pressure를 이용해 압력력 계산
#     ↓
# 밀도 차이에 따라 로봇들이 퍼지거나 모이는 힘 발생

# [3] SPH Viscosity Force 계산
#     ↓
# 상대 위치 + 상대 속도 확인
#     ↓
# 서로 접근 중인 이웃에 Artificial Viscosity 적용
#     ↓
# 급격한 상대 운동을 완화

# [4] 최종 SPH 가속도
#     ↓
# f_SPH = f_press + f_vis
#     ↓
# force limit / acceleration limit
#     ↓
# robot.acceleration 저장

# [5] 가속도 → 속도
#     ↓
# v ← v + a·dt
#     ↓
# 최대 속도 제한

# [6] 속도 → 위치
#     ↓
# x, y 방향 이동 시도
#     ↓
# physical_is_walkable()로 벽 충돌 검사
#     ↓
# 이동 가능 → 위치 갱신
# 벽 충돌 → 해당 방향 속도 반사

# [7] 실제 이동 결과 저장
#     ↓
# commanded_velocity = 계산된 속도
# observed_velocity = 실제 위치 변화 / dt
#     ↓
# acceleration 초기화
#     ↓
# 다음 simulation step


# =========================================================
# SPH Internal Force
#
# 각 로봇의 주변 이웃에 대해 압력력과 점성력을 계산하고,
# f_SPH = f_press + f_vis 로 최종 SPH 가속도를 결정한다.

# limit_vector() 는 수식 자체의 구성 요소가 아닌, 
# 시뮬레이션 구현 시 힘이나 가속도가 비정상적으로 커져 로봇이 튀는 것을 방지하기 위해 최대값에서 잘라서 더 이상 커지지 않도록 상한을 두는 것
# =========================================================
def compute_sph_only_forces(
    robots: list[environment.Robot],
    physics_grid,
    reference_density: float,
) -> None:
    """HydroSwarm Eqs. (12)-(14): f_SPH = f_press + f_vis."""
    # SPH 이웃 범위를 제곱거리 형태로 저장
    h_sq = environment.SMOOTHING_LENGTH**2

    # 각 로봇 i에 대해 SPH force 계산
    for robot_i in robots:
        pressure_force = pygame.Vector2()
        viscosity_force = pygame.Vector2()

        # Spatial grid의 주변 cell에서 이웃 후보 robot j 탐색
        for robot_j in environment.iter_physics_neighbor_candidates(robot_i, physics_grid):

            # 자기 자신은 이웃 계산에서 제외
            if robot_i is robot_j:
                continue

            # 두 로봇 사이의 상대 위치와 거리 계산
            r_ij = robot_i.position - robot_j.position
            distance_sq = r_ij.length_squared()

            # 너무 가깝거나 smoothing length h 밖이면 SPH 이웃에서 제외
            if distance_sq <= environment.EPSILON or distance_sq > h_sq:
                continue

            # 두 로봇 사이의 SPH kernel gradient 계산
            gradient = environment.spiky_gradient(r_ij, environment.SMOOTHING_LENGTH)

            # 두 로봇의 압력과 밀도를 이용한 대칭형 pressure coefficient 계산
            pressure_coefficient = (
                robot_i.pressure / max(robot_i.density**2, environment.EPSILON)
                + robot_j.pressure / max(robot_j.density**2, environment.EPSILON)
            )

            # 압력력 누적
            pressure_force += -pressure_coefficient * gradient

            # 두 로봇의 상대속도로 서로 접근 중인지 확인
            v_ij = robot_i.velocity - robot_j.velocity
            approach = v_ij.dot(r_ij)

            # 서로 접근하는 경우에만 artificial viscosity 적용
            if approach < 0.0:

                # 상대 위치와 상대속도로 접근률 mu_ij 계산
                mu_ij = (
                    environment.SMOOTHING_LENGTH * approach
                    / (distance_sq + 0.01 * environment.SMOOTHING_LENGTH**2)
                )

                # 각 로봇의 압력/밀도로 local sound-speed 항 계산
                c_i_sq = (
                    robot_i.pressure + environment.PRESSURE_GAIN * robot_i.density
                ) / max(robot_i.density, environment.EPSILON)

                c_j_sq = (
                    robot_j.pressure + environment.PRESSURE_GAIN * robot_j.density
                ) / max(robot_j.density, environment.EPSILON)

                # 두 로봇의 평균 sound-speed와 평균 밀도 계산
                c_ij = 0.5 * (
                    math.sqrt(max(c_i_sq, 0.0))
                    + math.sqrt(max(c_j_sq, 0.0))
                )
                mean_density = 0.5 * (robot_i.density + robot_j.density)

                # Monaghan 형태의 artificial viscosity coefficient 계산
                pi_ij = (
                    -environment.VISCOSITY_XI1 * c_ij * mu_ij
                    + environment.VISCOSITY_XI2 * mu_ij**2
                ) / max(mean_density, environment.EPSILON)

                # 점성력 누적
                viscosity_force += -pi_ij * gradient

        # 압력력과 점성력이 지나치게 커지지 않도록 각각 제한
        pressure_force = environment.limit_vector(
            pressure_force, environment.SPH_PRESSURE_FORCE_LIMIT
        )
        viscosity_force = environment.limit_vector(
            viscosity_force, environment.SPH_VISCOSITY_FORCE_LIMIT
        )

        # 최종 SPH force = pressure force + viscosity force → 로봇 가속도로 적용
        robot_i.acceleration = environment.limit_vector(
            pressure_force + viscosity_force,
            environment.MAX_ACCELERATION,
        )

        # 계산된 가속도와 디버깅용 SPH force 정보 저장
        robot_i.filtered_acceleration.update(robot_i.acceleration)
        robot_i.last_sph_pressure_force = pressure_force.length()
        robot_i.last_goal_force = 0.0


# =========================================================
# SPH Robot Motion Integration
#
# 계산된 SPH 가속도를 실제 속도와 위치 변화로 반영한다.
# 벽 충돌을 X/Y 방향으로 각각 검사하고 실제 이동 결과를 관측 속도로 저장한다.
# =========================================================
def integrate_sph_only_robot(
    robot: environment.Robot,
    dt: float,
) -> None:

    # 이동 전 위치 저장
    old_position = robot.position.copy()

    # SPH acceleration을 dt 동안 적분하여 velocity 갱신
    robot.velocity += robot.acceleration * dt

    # 로봇의 최대 이동 속도 제한
    speed_limit = environment.INITIAL_SAFE_MAX_SPEED
    if robot.velocity.length() > speed_limit:
        robot.velocity.scale_to_length(speed_limit)

    # X 방향으로 먼저 이동을 시도하고 벽 충돌 여부 확인
    x_position = pygame.Vector2(
        robot.position.x + robot.velocity.x * dt,
        robot.position.y,
    )

    if physical_is_walkable(x_position, robot.radius):
        # 이동 가능하면 X 위치 갱신
        robot.position.x = x_position.x
    else:
        # 벽에 충돌하면 X 속도를 반대 방향으로 반사
        robot.velocity.x = (
            -robot.velocity.x * environment.INITIAL_WALL_RESTITUTION
        )

    # Y 방향으로 이동을 시도하고 벽 충돌 여부 확인
    y_position = pygame.Vector2(
        robot.position.x,
        robot.position.y + robot.velocity.y * dt,
    )

    if physical_is_walkable(y_position, robot.radius):
        # 이동 가능하면 Y 위치 갱신
        robot.position.y = y_position.y
    else:
        # 벽에 충돌하면 Y 속도를 반대 방향으로 반사
        robot.velocity.y = (
            -robot.velocity.y * environment.INITIAL_WALL_RESTITUTION
        )

    # 제어상 계산된 현재 velocity 저장
    robot.commanded_velocity.update(robot.velocity)

    # 실제 위치 변화량 / dt로 실제 이동한 observed velocity 계산
    realized_velocity = (
        robot.position - old_position
    ) / max(dt, environment.EPSILON)

    robot.observed_velocity.update(
        realized_velocity.x,
        realized_velocity.y,
    )

    # 다음 step 계산을 위해 이전 위치 저장
    robot.previous_position.update(old_position)

    # 이번 step에서 사용한 acceleration 초기화
    robot.acceleration.update(0.0, 0.0)





# =========================================================
# Main 함수
# =========================================================
#
# Overall Algorithm Flow
#
# 1. INITIALIZATION
#    Environment / SPH parameter 설정
#    → Staggered grid 형태로 swarm 생성
#    → 초기 Anchor 선택
#    → Anchor를 제외한 나머지 로봇은 NORMAL swarm으로 시작
#    → Anchor LiDAR / DFS / Branch / Backtracking runtime state 초기화
#
# 2. ROOT JUNCTION DETECTION
#    Anchor가 corridor를 따라 전진하면서 매 physics substep마다 360° LiDAR scan
#    → ±90° 주변 LiDAR range로 좌·우 wall distance 추출
#    → 좌·우 wall distance로 adaptive worst wall range W 계산
#    → W와 LiDAR maximum range를 이용해 adaptive OPEN/WALL threshold T 계산
#    → 전체 360° range profile을 smoothing
#    → threshold T로 OPEN/WALL ray 분류
#    → 연속 OPEN ray grouping
#    → range gradient로 Opening boundary refinement
#    → Opening geometry 생성
#    → 유효 Opening이 3개 이상이면 lidar.junction_evidence = True
#
#    동시에 이동 중:
#    → update_junction_entrance_detector()
#    → ±90° lateral range history로 정상 corridor baseline 형성
#    → 좌·우 lateral range가 baseline보다 동시에 크게 증가하면 Junction entrance 검출
#
#    Junction entrance 검출
#    → Anchor 정지
#    → 진입 직전 corridor baseline으로 W와 T 재계산
#    → freeze_stationary_threshold()로 W와 T 고정
#    → stationary LiDAR observation 시작
#    → 동일 Opening을 center_angle 기준으로 association
#    → observation count + persistence ratio 검사
#    → persistent Opening이 3개 이상이면 Junction 최종 확정
#
# 3. LOCAL JUNCTION GEOMETRY
#    Junction entrance에서 얻은 Base-side corner geometry
#    + stationary LiDAR의 front-side corner geometry 사용
#    → Anchor-local 좌표계에서 Junction center 추정
#    → 여러 scan에서 center estimate를 누적
#    → 안정적으로 동일한 center가 관측되면 local target 고정
#    → Anchor가 local odometry를 이용해 추정된 Junction center로 이동
#
# 4. BRANCH REGISTRATION
#    Anchor가 Junction center에 도달한 뒤 현재 LiDAR opening들을 이용해 Branch 등록
#    → parent/rear corridor에 해당하는 Opening 제외
#    → 나머지 outgoing Opening마다 runtime Branch 생성
#    → Branch ID 부여
#    → center angle / mouth geometry / tangent / branch axis 저장
#    → Branch runtime state 생성
#    → 초기 visit_state = UNVISITED
#
# 5. INITIAL SHEPHERD / MARKER FORMATION
#    Junction 최종 확정 후 Anchor가 JUNCTION_CONFIRMED message broadcast 시작
#    → Branch entrance로 자연스럽게 유입된 NORMAL들을 candidate로 수집
#    → 각 Branch에서 가장 깊이 들어간 NORMAL 1대를 Hop 0 seed로 선택
#    → robot-to-robot COMM_RANGE를 따라
#         Hop 1 → Hop 2 → ... → INITIAL_SHEPHERD_HOPS까지 확장
#    → Hop 0~max-hop 전체 cohort의 lateral coverage 검사
#
#    Branch 입구가 아직 충분히 막히지 않음
#    → Shepherd 확정하지 않고 다음 frame까지 대기
#    → 새로운 NORMAL 유입 후 cohort와 coverage 다시 계산
#
#    Branch 입구가 충분히 막힘
#    → multi-hop cohort 전체를 SHEPHERD로 전환
#    → 현재 위치에서 freeze
#    → 해당 Branch의 initial_sealed = True
#
#    모든 Branch의 Initial Shepherd boundary가 완성되면
#    → Junction broadcast 종료
#    → 각 Branch Initial Shepherd 중 1대를 Marker로 전환
#    → Marker state = UNVISITED
#
# 6. PHYSICAL DFS BRANCH SELECTION
#    Initial Marker 생성 완료 후 confirmed Branch 순서를 뒤집어
#    runtime branch_order 생성
#
#    → find_next_unvisited_branch()
#    → branch_order 앞에서부터 UNVISITED Branch 탐색
#    → 선택 Branch: UNVISITED → ACTIVE
#    → 해당 Branch의 Initial Shepherd boundary를 release하여 NORMAL로 복귀
#    → Branch Marker는 상태 표시용으로 유지
#    → Anchor heading을 선택 Branch center_angle 방향으로 변경
#
# 7. BRANCH ENTRY / EXPLORATION
#    Anchor가 선택 Branch 입구를 통과
#    → Branch entry local odometry 누적
#    → Branch 내부 탐색 상태 BRANCH_EXPLORE 진입
#
#    탐색 중:
#    → Anchor 전방 LiDAR에서 traversable free gap 탐색
#    → 가장 적합한 free gap 방향으로 Anchor yaw / forward motion 제어
#    → NORMAL swarm은 계속 SPH로 이동
#    → Anchor는 가장 앞선 NORMAL과 목표 간격을 유지하도록 speed cap 적용
#
#    또한:
#    → Branch 시작 지점의 자기 Marker는 일정 거리까지 ignore
#    → 충분히 전진한 뒤 전방 Marker 탐색 활성화
#    → Marker 또는 Dead-end를 Branch 탐색 종료 조건으로 검사
#
# 8. TERMINAL EVENT
#    [A] Visible Marker 발견
#        → 현재 Branch 진입 시 사용한 자기 Marker는 제외
#        → 전방 / corridor 내부 / detection range / LiDAR LOS 조건을 모두 만족한 Marker 탐지
#        → Marker가 다른 Branch의 Marker라면
#             해당 Marker가 속한 Branch를 VISITED 처리
#        → 현재 Branch의 탐색 종료 event를 MARKER로 설정
#
#    [B] 진행 가능한 LiDAR free gap이 없음
#        → Dead-end candidate
#        → Anchor 정지
#        → DEAD_END_CONFIRM_FRAMES 동안 연속적으로 free gap이 없는지 확인
#        → 연속 조건 만족 시 Dead-end 확정
#        → 탐색 종료 event를 DEAD_END로 설정
#
#    Marker 또는 Dead-end가 확정되면
#    → backtrack_event_valid = True
#    → BACKTRACK_WAIT_SHEPHERD 진입
#
# 9. BACKTRACKING SHEPHERD FORMATION
#    Anchor 정지
#    → Anchor 뒤쪽 NORMAL 중
#      Anchor와 직접 COMM_RANGE 안에 있는 가장 가까운 1대를 Hop 0 seed로 선택
#
#    → Hop 1:
#      Hop 0과 COMM_RANGE로 연결된 NORMAL 모집
#
#    → Hop 2:
#      Hop 1과 COMM_RANGE로 연결된 NORMAL 모집
#
#    → Hop 0~2 전체 cohort를 Backtracking Shepherd로 사용
#    → selected cohort의 ID와 relative topology 유지
#    → Shepherd command relay를 통해 FORM 상태 확인
#    → 이후 PUSH 상태로 전환
#
#    복귀 heading은 Branch 탐색 종료 시 Anchor heading의 반대 방향 사용
#
# 10. PRESSURE PUSH
#    Backtracking Shepherd cohort가 return heading 방향으로 rigid하게 이동
#    → Shepherd가 NORMAL과 물리적으로 접촉하면 contact acceleration 전달
#
#    NORMAL에는 별도의 Junction goal force / backtracking goal force를 추가하지 않음
#    → NORMAL은 계속 기본 SPH controller 사용
#
#       f_SPH = f_press + f_vis
#
#    → Shepherd의 물리적 접촉과 밀어내기에 의해
#      NORMAL swarm에 Junction 방향 역류 유도
#
# 11. PRESSURE PUSH → FLOW BACKTRACKING
#    SPH integration 이후 NORMAL의 실제 observed velocity 검사
#    → Junction 복귀 방향 velocity 성분 계산
#    → 충분한 수의 NORMAL이 reverse direction으로 이동
#    → reverse ratio / minimum robot count / stable scan 조건 만족
#    → FLOW_BACKTRACK으로 전환
#
# 12. FLOW BACKTRACKING
#    Anchor + 동일 Backtracking Shepherd cohort가 Parent Junction 방향으로 복귀
#    → Anchor는 NORMAL swarm front를 따라가면서 corridor 중앙 유지
#
#    Shepherd가 wall 때문에 return 방향으로 이동할 수 없음
#        → BACKTRACK_WALL_DETACH
#        → 전체 cohort를 동일한 lateral 방향으로 shift
#        → 기존 Shepherd topology 유지
#        → 이동 가능해지면 복귀 계속
#
#    corridor corner 발견
#        → BACKTRACK_CORNER_TURN
#        → Anchor와 동일 Shepherd cohort를 rigid motion으로 함께 회전
#        → 새 corridor에 들어온 뒤 corridor width / heading 조건 확인
#        → 필요하면 새 corridor 폭 밖의 Shepherd만 NORMAL로 release
#        → FLOW_BACKTRACK 재개
#
# 13. PARENT JUNCTION RETURN
#    복귀 중 update_junction_entrance_detector()를 다시 사용
#    → return corridor의 lateral baseline 형성
#    → 좌·우 lateral range 증가로 Parent Junction entrance 재검출
#
#    Parent Junction entrance 도착
#    → Anchor 정지
#    → return corridor baseline으로 adaptive W / T 고정
#    → stationary Junction confirmation 수행
#    → persistence 조건으로 Parent Junction 재확인
#
#    → 들어온 Branch-side corner + 반대편/front-side corner geometry 이용
#    → Parent Junction center를 Anchor-local 좌표로 재추정
#    → 안정된 local center target 고정
#    → Anchor가 Junction center로 이동
#
#    이 동안 Backtracking Shepherd는
#    남은 NORMAL swarm을 계속 Junction 방향으로 push
#
# 14. PHYSICAL BRANCH RETURN COMPLETE
#    Parent Junction center 도달 후
#    → Branch 내부 swarm의 실제 물리적 복귀 완료 여부 확인
#
#    복귀 완료
#    → Backtracking Shepherd들을 NORMAL로 release
#    → 현재 ACTIVE Branch를 VISITED로 변경
#    → 해당 Branch Marker state도 VISITED로 변경
#
#    → 해당 Branch의 Initial Shepherd state 초기화
#    → 방금 탐색한 Branch 입구를 다시 Initial Shepherd boundary로 형성
#    → physical sealing이 다시 완료될 때까지 대기
#
# 15. DFS CONTINUATION
#    탐색 완료 Branch의 입구가 다시 봉쇄되면
#    → SELECT_NEXT_BRANCH
#
#    branch_order에서 다음 UNVISITED Branch 탐색
#
#    UNVISITED Branch 존재
#        → Branch = ACTIVE
#        → Branch Shepherd release
#        → BRANCH_ENTRY
#        → BRANCH_EXPLORE
#        → Marker / Dead-end
#        → Backtracking
#        → Parent Junction Return
#        → Branch VISITED
#        → Branch reseal
#        → 반복
#
#    UNVISITED Branch 없음
#        → ROOT_COMPLETE
#
# 16. FINAL BASE RETURN
#    Root Junction의 모든 Branch 탐색 완료
#    → Junction reference heading 복원
#    → final push에 사용할 Branch / Shepherd group 선택
#    → side Branch의 Shepherd / Marker release
#    → side에 남아 있던 swarm이 Root Junction으로 합류할 때까지 대기
#
#    → Final Shepherd group이 swarm을 Base 방향으로 push
#    → swarm의 Base 복귀 완료 확인
#    → Final Shepherd release
#
#    → Anchor heading을 Base 방향으로 변경
#    → Anchor가 corridor-following으로 Base 방향 최종 복귀
#    → SYSTEM_COMPLETE
#
# 17. SPH SWARM MOTION (EVERY PHYSICS SUBSTEP)
#    Spatial grid에서 주변 이웃 후보 탐색
#    → density ρ_i 계산
#    → pressure P_i 계산
#    → pressure force + viscosity force 계산
#
#       f_SPH = f_press + f_vis
#
#    → acceleration
#    → velocity
#    → wall collision을 고려한 position integration
#    → observed velocity 갱신
#
#    Backtracking 중 Shepherd와 NORMAL이 실제 접촉하면
#    → contact acceleration을 NORMAL acceleration에 추가
#
#    단:
#    → Marker / role_frozen Shepherd는 위치 고정
#    → Backtracking PUSH Shepherd는 SPH integration 대상에서 제외
#    → Final Base Push Shepherd 역시 별도 motion 사용
#
# 전체 핵심 흐름:
#
# Base
# → Corridor Following + 360° LiDAR
# → Adaptive W / Threshold T
# → Junction Entrance Detection
# → Threshold Freeze
# → Stationary Persistence Verification
# → Junction Confirmed
# → Junction Center Estimation
# → Branch Registration
# → Initial Shepherd Formation
# → Initial Marker Creation
# → Runtime Branch Order
# → Select UNVISITED Branch
# → Branch Entry
# → Branch Exploration
# → Marker / Dead-end
# → Hop 0-2 Backtracking Shepherd Formation
# → Pressure Push
# → Reverse Flow Confirmation
# → Flow Backtracking
# → Parent Junction Entrance Detection
# → Parent Junction Stationary Confirmation
# → Parent Junction Center Return
# → Physical Swarm Return Complete
# → ACTIVE Branch = VISITED
# → Branch Reseal
# → Next UNVISITED Branch
# → 모든 Branch 완료
# → Final Swarm Base Push
# → Anchor Base Return
# → SYSTEM_COMPLETE
# =========================================================


def main() -> None:


    # =========================================================
    # Pygame Simulation Initialization
    # Pygame 실행 환경과 화면, 시간 제어, 출력용 Font를 초기화한다.
    # HEADLESS 모드에서는 화면 대기 없이 FPS 기준 고정 timestep을 사용한다.
    # =========================================================

    pygame.init()

    # HEADLESS 실행 여부 출력
    if HEADLESS:
        print("[HeadlessMode] enabled=True timestep=1/FPS")

    # 시뮬레이션 화면 초기화
    pygame.display.set_caption("Physical DFS | SPH Base Ingress")
    screen = pygame.display.set_mode(WINDOW_SIZE)

    # FPS 제어용 Clock 및 화면 출력용 Font 생성
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)


    # =====================================================
    # Environment / Map Setup
    # 시뮬레이션에서 사용할 십자가형 Junction 환경의 크기와 물리 영역을 설정한다.
    # 실제 탐색 알고리즘이 환경의 절대 위치를 사용하는 것이 아니라, 시뮬레이션 map 자체를 구성하기 위한 설정이다.
    # =====================================================

    environment.MAP_SCALE = MAP_SCALE

    # Base corridor를 기준으로 map 중심, 통로 폭, 길이 계산
    environment.center_x = (BASE_CORRIDOR_RECT.left + BASE_CORRIDOR_RECT.right) / 2.0
    environment.center_y = BASE_CORRIDOR_RECT.top - BASE_CORRIDOR_RECT.width / 2.0
    environment.corridor_width = BASE_CORRIDOR_RECT.width
    environment.half_width = BASE_CORRIDOR_RECT.width // 2
    environment.normal_length = scale_point((208, 253))[1] - scale_point((208, 39))[1]
    environment.right_length = scale_point((524, 253))[0] - scale_point((302, 253))[0]
    environment.base_length = BASE_CORRIDOR_RECT.height
    environment.cross_points = list(OUTER_BOUNDARY)

    # Junction 및 UP / LEFT / RIGHT / BASE corridor의 물리 영역 정의
    environment.junction_rect = pygame.Rect(scale_point((208, 253)), (BASE_CORRIDOR_RECT.width, BASE_CORRIDOR_RECT.width))
    environment.up_rect = pygame.Rect(scale_point((208, 39)), (BASE_CORRIDOR_RECT.width, environment.normal_length))
    environment.left_rect = pygame.Rect(scale_point((78, 253)), (scale_point((208, 253))[0] - scale_point((78, 253))[0], BASE_CORRIDOR_RECT.width))
    environment.right_rect = pygame.Rect(scale_point((302, 253)), (environment.right_length, BASE_CORRIDOR_RECT.width))
    environment.bottom_rect = BASE_CORRIDOR_RECT.copy()

    # 시뮬레이션 map의 각 Branch 길이 저장
    environment.BRANCH_LENGTHS.update(UP=float(environment.normal_length), LEFT=float(environment.left_rect.width), RIGHT=float(environment.right_length))

    # 초기 swarm 위치 및 Junction 진입 관련 시뮬레이션 기준점 설정
    environment.BASE_POSITION = BASE_POSITION.copy()
    environment.BASE_COMPRESSION_CENTER = BASE_POSITION.copy()
    environment.JUNCTION_STAGING_POSITION = pygame.Vector2(environment.junction_rect.center)
    environment.INITIAL_INGRESS_TARGET_Y = environment.junction_rect.top - 18.0 * environment.MAP_SCALE

    # 실제 이동 가능한 map 영역을 그릴 투명 Surface 생성
    environment.floor_surface = pygame.Surface((environment.SCREEN_WIDTH, environment.SCREEN_HEIGHT), pygame.SRCALPHA)
    environment.floor_surface.fill((0, 0, 0, 0))

    # =============================================
    # Walkable Map 생성
    # 실제 자유공간을 그리고 obstacle 영역을 제거한 뒤, 로봇의 물리적 이동 가능 여부를 검사할 walkable mask를 생성한다.
    # =============================================

    # 십자가형 corridor 영역을 실제 이동 가능한 자유공간으로 생성
    pygame.draw.polygon(environment.floor_surface, (255, 255, 255, 255), environment.cross_points)

    # Obstacle 영역을 자유공간에서 제거
    pygame.draw.rect(environment.floor_surface, (0, 0, 0, 0), OBSTACLE_RECT)

    # 최종 자유공간을 물리적 충돌/이동 가능 여부 판단용 mask로 변환
    environment.walkable_mask = pygame.mask.from_surface(environment.floor_surface)
    
    
    
    # =====================================================
    # Robot / SPH Parameters
    # 로봇의 물리 크기와 초기 배치, SPH 상호작용, 이동 제한 및 통신 범위를 설정하고 swarm을 생성한다.
    # =====================================================

    # 로봇 수와 물리 크기 및 초기 staggered-grid 배치 간격
    environment.ROBOT_COUNT = 300
    environment.ROBOT_RADIUS = 0.8 * SIMULATION_LENGTH_SCALE
    environment.GRID_SPACING = 2.4 * SIMULATION_LENGTH_SCALE
    environment.GRID_ROW_SPACING = environment.GRID_SPACING
    environment.INITIAL_GRID_SIDE_MARGIN = 2.0 * SIMULATION_LENGTH_SCALE

    # SPH 이웃 탐색 반경과 spatial grid 크기 설정
    environment.SMOOTHING_LENGTH = 18.0 * SIMULATION_LENGTH_SCALE
    environment.SAFE_RADIUS = 7.5 * MAP_SCALE
    environment.SPH_CELL_SIZE = environment.SMOOTHING_LENGTH

    # SPH 기준 밀도 rho_0 직접 지정 -> 핵심 파라미터값
    REFERENCE_DENSITY = 0.025

    # 기존 spacing 관련 로직을 위해 유지하며 reference density 계산에는 사용하지 않음
    environment.REFERENCE_EQUILIBRIUM_SPACING = 5.0 * SIMULATION_LENGTH_SCALE

    # 벽 충돌 반사 계수 및 SPH pressure / viscosity 계수
    environment.INITIAL_WALL_RESTITUTION = 0.20
    environment.PRESSURE_GAIN = 750.0 * SIMULATION_LENGTH_SCALE**2
    environment.VISCOSITY_XI1 = 0.7
    environment.VISCOSITY_XI2 = 0.7

    # Pressure와 viscosity force의 최대 크기 및 최종 가속도 saturation 설정
    environment.SPH_PRESSURE_FORCE_LIMIT = 100.0 * SIMULATION_LENGTH_SCALE
    environment.SPH_VISCOSITY_FORCE_LIMIT = 45.0 * SIMULATION_LENGTH_SCALE
    environment.MAX_ACCELERATION = 110.0 * SIMULATION_LENGTH_SCALE

    # 일반 swarm 및 Junction 진입 시 최대 이동 속도
    environment.INITIAL_SAFE_MAX_SPEED = 11.0 * SIMULATION_LENGTH_SCALE
    environment.INITIAL_JUNCTION_MAX_SPEED = 8.0 * SIMULATION_LENGTH_SCALE

    # 로봇 간 통신 가능 거리와 통신 유지 관련 거리 기준
    environment.COMM_RANGE = 24.0 * SIMULATION_LENGTH_SCALE
    environment.COMM_SAFE_DISTANCE = 15.0 * SIMULATION_LENGTH_SCALE
    environment.COMM_BARRIER_START = 15.0 * SIMULATION_LENGTH_SCALE
    environment.COMM_GUARD_START = 12.0 * SIMULATION_LENGTH_SCALE
    environment.COMM_GUARD_HARD_LIMIT = float("inf")

    # SPH 이웃 탐색과 통신 탐색을 모두 포함할 수 있도록 spatial grid cell 크기 설정
    environment.CELL_SIZE = max(environment.SMOOTHING_LENGTH, environment.COMM_RANGE)

    # staggered-grid 방식으로 초기 robot swarm 생성
    environment.create_grid_robots = create_staggered_grid_robots
    robots, spacing_based_reference_density, color_reference_density = environment.initialize_simulation()

    # 실제 SPH 계산에서 사용할 HydroSwarm 기준 밀도 rho_0
    reference_density = REFERENCE_DENSITY

    # 자동 계산된 밀도와 실제 적용되는 기준 밀도 출력
    print("[SPHReferenceDensity] " f"spacing_based={spacing_based_reference_density:.6f} " f"effective={reference_density:.6f}")

    

    
    # =====================================================
    # Anchor Initialization
    # 초기 Anchor를 확인하고 전체 robot swarm에서 Anchor와 일반 swarm을 분리한 뒤 LiDAR를 생성한다.
    # =====================================================

    # 초기 배치 과정에서 Anchor가 지정되지 않았으면 실행 중단
    if INITIAL_ANCHOR_ID is None:
        raise RuntimeError("Initial Anchor was not assigned.")

    # Robot ID 기반 접근을 위한 dictionary 생성
    robots_by_id = {robot.robot_id: robot for robot in robots}

    # 지정된 로봇을 Mobile LiDAR Anchor로 설정하고 나머지를 swarm으로 분리
    anchor = robots_by_id[INITIAL_ANCHOR_ID]
    swarm_robots = [robot for robot in robots if robot is not anchor]

    # Anchor의 360° LiDAR 생성
    anchor_lidar = AnchorLidar()

    # =====================================================
    # Runtime State
    # Junction 검출 및 Anchor의 Junction entrance 정지/center 이동 과정을 관리하는 상태 변수
    # =====================================================

    # Junction 검출 및 entrance 정지 상태
    junction_detected = False
    junction_entrance_stopped = False

    # Anchor의 Junction center 도달 및 안정화 상태
    anchor_locally_centered = False
    anchor_center_stable_count = 0

    # LiDAR로 추정한 Junction center 후보와 최종 고정된 Anchor-local center target
    entrance_target_samples: list[pygame.Vector2] = []
    locked_center_target_local: pygame.Vector2 | None = None

    # Junction center로 이동하는 동안 Anchor가 이동한 local displacement 누적
    anchor_center_traveled_local = pygame.Vector2()

    # LiDAR로 추출한 Base entrance의 두 corner
    base_entrance_corners_local: tuple[pygame.Vector2, pygame.Vector2] | None = None

    # 검출·등록된 Branch 정보와 각 Branch의 DFS 상태 저장
    confirmed_branches: list[dict] = []
    branch_states: dict = {}

    # =====================================================
    # Initial Junction Formation State
    # 모든 Branch 입구의 Initial Shepherd 형성이 완료되었는지 관리
    # =====================================================

    initial_shepherds_ready = False



    # =====================================================
    # Initial Junction Broadcast State
    # Junction 확정 후 Initial Shepherd formation이 완료될 때까지 Anchor의 Junction message broadcast와 수신 상태를 관리한다.
    # =====================================================

    # Junction message broadcast 활성 여부와 직접 수신한 robot ID 저장
    junction_broadcast_active = False
    junction_message_received_ids: set[int] = set()

    # Initial Marker 생성 및 DFS Branch 순서/현재 탐색 Branch 상태
    markers_ready = False
    branch_order: list[str] = []
    active_branch_id: str | None = None
    first_branch_opened = False

    # =====================================================
    # Runtime Anchor Heading / Branch Exploration
    # Anchor의 현재 heading과 Physical DFS 탐색 state machine 및 이동 odometry를 관리한다.
    # =====================================================

    # 초기 yaw로 시작하며 이후 선택된 Branch 방향에 따라 runtime으로 갱신
    anchor_yaw_deg = ANCHOR_YAW_DEG

    # 현재 Junction에서 Branch local angle을 정의하는 기준 heading
    junction_reference_yaw_deg: float | None = None

    # Anchor state machine:
    # ROOT_SETUP → BRANCH_ENTRY → BRANCH_EXPLORE → BACKTRACK_WAIT_SHEPHERD → PRESSURE_PUSH → FLOW_BACKTRACK → BACKTRACK_CORNER_TURN → RETURN_JUNCTION_ENTRANCE
    anchor_motion_mode = "ROOT_SETUP"

    # 동일 상태의 반복 로그를 방지하고 실제 state transition만 기록
    last_logged_anchor_motion_mode: str | None = None

    # 모든 Branch 탐색 완료 후 최종 Base 복귀에 사용할 Shepherd/Branch 상태
    final_push_branch_id: str | None = None
    final_push_ids: set[int] = set()
    final_side_branch_ids: set[str] = set()
    final_side_merge_stable_count = 0
    final_base_return_stable_count = 0
    final_anchor_return_traveled = 0.0

    # Junction center → 선택 Branch mouth 통과까지의 목표 거리와 이동 거리
    anchor_branch_entry_target = 0.0
    anchor_branch_entry_traveled = 0.0

    # Branch 입구 통과 후 실제 Branch exploration 동안의 Anchor 이동 거리
    branch_explore_traveled = 0.0

    # =====================================================
    # Backtracking Runtime State
    # Marker/Dead-end 탐색 종료 후 Shepherd 형성, Pressure Push, Flow Backtracking 및 Junction 복귀 상태를 관리한다.
    # =====================================================

    # Dead-end가 일시적인 LiDAR 관측이 아닌지 확인하기 위한 연속 검출 횟수
    dead_end_stable_count = 0

    # Anchor 뒤에서 Backtracking Hop0 seed로 선택된 NORMAL robot
    backtrack_seed_id: int | None = None

    # Backtracking Shepherd formation에 필요한 corridor width와 탐색 종료 원인
    backtrack_required_width = KNOWN_CORRIDOR_WIDTH
    backtrack_trigger_reason: str | None = None

    # Pressure Push 시작 시의 탐색 heading을 고정하여 Anchor가 회전해도 Shepherd의 push 방향을 유지
    backtrack_push_yaw_deg: float | None = None

    # Wall detach 완료 후 복귀할 Backtracking state
    # 최초 충돌: WALL_DETACH → PRESSURE_PUSH / Flow 중 재충돌: WALL_DETACH → FLOW_BACKTRACK
    backtrack_wall_detach_resume_mode = "PRESSURE_PUSH"

    # 한 번 선택된 wall-detach 방향을 해당 detach episode 동안 고정
    backtrack_wall_detach_direction: str | None = None

    # Flow Backtracking을 통해 Parent Junction entrance에 실제 도달했는지 저장
    return_junction_entrance_reached = False



    # =====================================================
    # Parent Junction Return / Backtracking State
    # Backtracking 후 Parent Junction entrance와 center로 복귀하는 과정 및 Backtracking Shepherd formation 상태를 관리한다.
    # =====================================================

    # 복귀 시 LiDAR로 다시 추출한 Parent Junction의 Base-side entrance corner
    return_base_entrance_corners_local: tuple[pygame.Vector2, pygame.Vector2] | None = None

    # Parent Junction center 후보를 누적하고 최종 local center target을 고정
    return_center_target_samples: list[pygame.Vector2] = []
    return_locked_center_target_local: pygame.Vector2 | None = None

    # Parent Junction center까지의 Anchor local 이동량과 center 도달 안정성 관리
    return_center_traveled_local = pygame.Vector2()
    return_center_stable_count = 0

    # Swarm 전체가 Parent Junction으로 충분히 복귀했는지 확인하기 위한 안정화 count
    swarm_return_stable_count = 0

    # Marker 또는 Dead-end에 의해 유효한 Backtracking event가 발생했는지 저장
    backtrack_event_valid = False
    detected_marker_id: int | None = None
    origin_marker_cleared = False

    # Backtracking Shepherd 형성에 필요한 수와 실제 선택된 Hop0-2 cohort 정보
    backtrack_required_count = 0
    backtrack_formation_ids: set[int] = set()
    backtrack_chain_order: list[int] = []

    # Shepherd formation과 NORMAL reverse flow가 안정적으로 형성되었는지 확인하는 count
    backtrack_formation_stable_count = 0
    backtrack_reverse_stable_count = 0

    # Corner turn 후 새로운 corridor에 정상적으로 진입했는지 연속 확인
    backtrack_corner_exit_stable_count = 0

    # Backtracking 중 통과한 corner 수를 기록하는 diagnostic 값
    backtrack_corner_count = 0

    # 시뮬레이션 시작 시 Anchor의 초기 360° LiDAR scan 수행
    anchor_lidar.scan(anchor.position, anchor_yaw_deg)

    # 시뮬레이션 실행 및 화면 표시 상태 초기화
    paused = False
    show_comm_links = True
    running = True




    # =====================================================
    # Main loop
    # =====================================================

    while running:

        # =====================================================
        # Event Handling / Time Step
        # 사용자 입력을 처리하고 각 simulation frame에서 사용할 물리 timestep을 계산한다.
        # =====================================================

        # Pygame event 처리
        for event in pygame.event.get():

            # 창을 닫으면 simulation 종료
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                # SPACE: simulation 일시정지 / 재개
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print("[Pause] " f"paused={paused}")

                # C: robot 간 communication link 시각화 ON / OFF
                elif event.key == pygame.K_c:
                    show_comm_links = not show_comm_links
                    print("[CommunicationView] " f"visible={show_comm_links}")

                # R: 현재 Python process를 다시 실행하여 simulation 전체 초기화
                elif event.key == pygame.K_r:
                    print("[SimulationReset]")
                    pygame.quit()
                    os.execv(sys.executable, [sys.executable, *sys.argv])

                # ESC: simulation 종료
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # =====================================================
        # Time Step
        # HEADLESS 여부에 따라 frame timestep을 계산하고, 물리 계산 안정성을 위해 8개의 substep으로 분할한다.
        # =====================================================

        # HEADLESS에서는 실시간 대기 없이 FPS 기준의 고정 timestep 사용
        if HEADLESS:
            frame_dt = 1.0 / environment.FPS
        else:
            # 일반 실행에서는 실제 frame 시간을 사용하되 지나치게 작은 dt를 방지
            frame_dt = max(clock.tick(environment.FPS) / 1000.0, 1.0 / 240.0)

        # 지나치게 큰 timestep으로 인한 물리 simulation 불안정 방지
        frame_dt = min(frame_dt, environment.INITIAL_INGRESS_MAX_DT)

        # 한 frame을 8개의 작은 물리 timestep으로 나누어 계산
        substep_dt = frame_dt / 8.0



        # =====================================================
        # Physics Substep / Fresh Local Sensing
        # 각 frame을 8개의 physics substep으로 나누어 SPH, sensing, communication 및 물리 상호작용을 갱신한다.
        # Pause 상태에서는 substep을 실행하지 않는다.
        # =====================================================
        for substep_index in range(0 if paused else 8):

            # 이번 substep에서 Backtracking Shepherd가 NORMAL과 접촉하여 전달한 물리적 가속도를 저장
            backtrack_contact_accelerations: dict[int, pygame.Vector2] = {}

            # simulation 시간 갱신
            environment.simulation_time += substep_dt

            # 현재 Anchor 위치/방향에서 새로운 360° LiDAR scan 수행
            anchor_lidar.scan(anchor.position, anchor_yaw_deg)

            # LiDAR와 로봇 간 상대 위치/속도를 이용해 현재 Anchor-local observation 생성
            observation = LocalObservationBuilder.build(anchor, robots, anchor_lidar, anchor_yaw_deg)

            # =====================================================
            # Junction Message Broadcast
            # Junction 확정 후 Initial Shepherd formation이 끝날 때까지
            # Anchor의 Junction message를 수신한 robot들을 계속 누적한다.
            # =====================================================

            if junction_broadcast_active and not initial_shepherds_ready:

                # 현재 substep에서 Junction message를 수신할 수 있는 robot 확인
                received_now = update_junction_broadcast_receivers(observation, robots_by_id)

                # 이전까지 수신하지 않았던 새 robot만 추출하여 누적
                newly_received = received_now - junction_message_received_ids
                junction_message_received_ids.update(received_now)

                # 새롭게 message를 수신한 robot이 있을 때만 diagnostic 출력
                if newly_received:
                    print("[JunctionMessageReceived] " f"new_ids={sorted(newly_received)} " f"total={len(junction_message_received_ids)}")



            # =================================================
            # 1. Anchor Approaches Junction
            # Anchor가 통로를 따라 이동하면서 LiDAR의 좌·우 wall range 변화를 이용해 Junction entrance 도달 여부를 판단한다.
            # =================================================

            if not junction_entrance_stopped:

                entrance_reached = False

                # 한 frame의 첫 substep에서만 Junction entrance detection 수행
                if substep_index == 0:
                    entrance_reached, _, _, _, _ = update_junction_entrance_detector(anchor_lidar)

                if entrance_reached:

                    # Junction entrance가 검출되면 Anchor를 즉시 정지
                    junction_entrance_stopped = True
                    stop_anchor(anchor)

                    # 이동 중 계산된 adaptive threshold를 고정하여 stationary Junction detection에 사용
                    freeze_stationary_threshold(anchor_lidar)

                    # Junction 진입 직전 Base corridor에서 측정한 좌·우 wall range 확보
                    left_wall = anchor_lidar.lateral_baseline_left
                    right_wall = anchor_lidar.lateral_baseline_right

                    if left_wall is None or right_wall is None:
                        raise RuntimeError("No valid Base corridor wall calibration before Junction.")

                    # Junction center 추정을 위한 runtime 상태 초기화
                    entrance_target_samples.clear()
                    locked_center_target_local = None

                    # 진입 직전 좌·우 wall range로 Base entrance의 두 corner를 Anchor-local 좌표에 설정
                    base_entrance_corners_local = (pygame.Vector2(0.0, -left_wall), pygame.Vector2(0.0, right_wall))

                    # Junction center까지의 Anchor 이동량 및 안정화 count 초기화
                    anchor_center_traveled_local.update(0.0, 0.0)
                    anchor_center_stable_count = 0

                    print("[JunctionEntranceDetected] " f"W={anchor_lidar.locked_adaptive_w:.2f} " f"T={anchor_lidar.locked_w_tau_threshold:.2f} " f"BL=(0.00,{-left_wall:.2f}) " f"BR=(0.00,{right_wall:.2f})")

                else:
                    # Junction이 아직 검출되지 않았으면 local sensing만으로 corridor를 따라 계속 전진
                    follow_corridor_locally(anchor, observation, anchor_yaw_deg, substep_dt)



            # =================================================
            # 2. Stationary Junction Confirmation
            # Junction entrance에서 Anchor를 정지시킨 상태로 LiDAR opening이 지속적으로 관측되는지 확인하여 실제 Junction인지 검증한다.
            # =================================================

            elif not junction_detected:

                # Stationary confirmation 동안 Anchor 정지 유지
                stop_anchor(anchor)

                # 한 frame의 첫 substep에서만 stationary Junction confirmation 수행
                if substep_index == 0 and update_stationary_junction_confirmation(anchor_lidar):

                    # 지속적인 opening 관측이 확인되면 Junction을 최종 확정
                    junction_detected = True

                    # Junction 확정 후 Initial Shepherd formation을 위해 Anchor의 JUNCTION_CONFIRMED broadcast 시작
                    junction_broadcast_active = True
                    junction_message_received_ids.clear()

                    print("[JunctionConfirmed]")
                    print("[AnchorBroadcast] command=JUNCTION_CONFIRMED active=True")


            # =================================================
            # 3. Estimate Junction Center
            # 정지 상태의 Anchor LiDAR로 Base 입구와 정면 Branch 입구의 네 corner를 구하고, 대각선 교점을 이용해 Junction center를 Anchor-local 좌표에서 추정한다.
            # 여러 scan에서 안정적으로 같은 center가 관측되면 평균값을 최종 이동 target으로 고정한다.
            # =================================================

            elif not anchor_locally_centered and locked_center_target_local is None:

                # Junction center 추정 동안 Anchor 정지 유지
                stop_anchor(anchor)

                if substep_index == 0 and base_entrance_corners_local is not None:

                    # Junction 진입 시 저장한 Base 입구의 좌·우 corner
                    base_left, base_right = base_entrance_corners_local

                    # 현재 LiDAR opening으로 정면 Branch 입구의 좌·우 physical corner 추출
                    front_corners = extract_front_mouth_corners(anchor_lidar)

                    if front_corners is None:
                        candidate = None
                    else:
                        front_left, front_right = front_corners

                        # Base와 Front의 네 corner를 이용해 Junction center 후보 계산
                        candidate = estimate_local_junction_center(base_left, base_right, front_left, front_right)

                    # 유효한 center를 구하지 못하면 누적 sample 초기화
                    if candidate is None:
                        entrance_target_samples.clear()
                        print("[EntranceStationaryScan] diagonal_center=NOT_AVAILABLE")

                    else:
                        # 이전 center 후보와 차이가 너무 크면 안정적인 연속 관측이 아니므로 다시 누적
                        if entrance_target_samples:
                            previous = entrance_target_samples[-1]
                            if (candidate - previous).length() > ANCHOR_TARGET_SAMPLE_TOLERANCE:
                                entrance_target_samples.clear()

                        # 안정적으로 관측된 Junction center 후보 누적
                        entrance_target_samples.append(candidate.copy())

                        print("[EntranceStationaryScan] " f"center_candidate=({candidate.x:.2f},{candidate.y:.2f}) " f"samples={len(entrance_target_samples)}/{ANCHOR_ENTRANCE_STATIONARY_SCANS}")

                        # 필요한 횟수만큼 안정적으로 관측되면 평균 center를 최종 target으로 고정
                        if len(entrance_target_samples) >= ANCHOR_ENTRANCE_STATIONARY_SCANS:
                            total = pygame.Vector2()
                            for sample in entrance_target_samples:
                                total += sample

                            locked_center_target_local = total / len(entrance_target_samples)

                            # 이후 center까지 실제 이동할 local odometry 초기화
                            anchor_center_traveled_local.update(0.0, 0.0)

                            print("[JunctionCenterTargetLocked] " f"target=({locked_center_target_local.x:.2f},{locked_center_target_local.y:.2f})")


            # =================================================
            # 4. Anchor Moves to Local Junction Center
            # 고정된 Anchor-local Junction center target까지 Anchor를 이동시키고,
            # 연속적으로 center 도달이 확인되면 정지하여 Root Junction setup 단계로 전환한다.
            # =================================================

            elif not anchor_locally_centered:

                # 현재까지의 local odometry를 기준으로 Junction center 방향으로 Anchor 이동
                centered_now, local_delta = move_anchor_to_locked_center(
                    anchor, locked_center_target_local, anchor_center_traveled_local,
                    anchor_yaw_deg, substep_dt
                )

                # 이번 substep에서 실제 이동한 Anchor-local displacement 누적
                anchor_center_traveled_local += local_delta

                # center 도달 상태가 연속적으로 유지되는지 확인
                if centered_now:
                    if substep_index == 0:
                        anchor_center_stable_count += 1
                else:
                    anchor_center_stable_count = 0

                # 필요한 횟수만큼 center 도달이 유지되면 Junction center 도착 확정
                if anchor_center_stable_count >= ANCHOR_CENTER_STABLE_SCANS:
                    anchor_locally_centered = True
                    stop_anchor(anchor)

                    print("[AnchorLocallyCentered] "
                        f"target=({locked_center_target_local.x:.2f},{locked_center_target_local.y:.2f}) "
                        f"travel=({anchor_center_traveled_local.x:.2f},{anchor_center_traveled_local.y:.2f})")

            else:

                # Junction center 도착 후 Root Junction의 Branch/Shepherd/Marker setup이 진행되는 동안 Anchor 정지 유지
                if anchor_motion_mode == "ROOT_SETUP":
                    stop_anchor(anchor)


            # =================================================
            # 5. Register Branch Geometry
            # Anchor가 Junction center에 도달한 뒤 최신 LiDAR opening으로 outgoing Branch들을 등록하고,
            # 현재 Anchor heading을 Branch angle의 기준 heading으로 고정한 뒤 각 Branch의 DFS 상태를 초기화한다.
            # =================================================
            if anchor_locally_centered and not confirmed_branches:

                # Junction center에서 관측된 LiDAR opening으로 outgoing Branch geometry 등록
                confirmed_branches = register_outgoing_branches(anchor_lidar)

                if confirmed_branches:
                    # 현재 Anchor heading을 이 Junction의 Branch local angle 기준으로 저장
                    junction_reference_yaw_deg = anchor_yaw_deg
                    print("[JunctionReferenceHeading] " f"yaw={junction_reference_yaw_deg:.2f}")

                # 각 Branch의 Initial/Backtracking Shepherd, Marker 및 DFS visit state 초기화
                branch_states = {
                    branch["id"]: {
                        "initial_shepherd_ids": set(),
                        "backtrack_shepherd_ids": set(),
                        "initial_sealed": False,
                        "marker_id": None,
                        "marker_state": "UNVISITED",
                        "visit_state": "UNVISITED",
                    }
                    for branch in confirmed_branches
                }

                # 등록된 각 Branch의 LiDAR angular sector 출력
                if confirmed_branches:
                    print(
                        "[BranchSectorsRegistered] "
                        + " ".join(
                            f'{branch["id"]}:{branch["start_angle"]:.1f}~{branch["end_angle"]:.1f}'
                            for branch in confirmed_branches
                        )
                    )





            # =================================================
            # 6. Independent Anchor Branch exploration
            # =================================================

            # =================================================
            # 6-1. Branch Entry
            # DFS에서 선택된 Branch 방향으로 Anchor가 Junction을 빠져나가 Branch corridor 내부로 진입한다.
            # Junction 내부에서는 corridor wall이 아직 안정적으로 관측되지 않으므로 wall-following 없이 선택 Branch heading으로 직진한다.
            # =================================================
            if anchor_motion_mode == "BRANCH_ENTRY":

                # 선택 Branch heading 기준으로 Anchor를 직진
                local_delta = integrate_anchor_local_command(
                    anchor, pygame.Vector2(ANCHOR_FORWARD_SPEED, 0.0),
                    anchor_yaw_deg, substep_dt
                )

                # 실제 전진한 Anchor-local 거리만 odometry에 누적
                anchor_branch_entry_traveled += max(0.0, local_delta.x)

                # Branch 입구 통과 진행 상태 출력
                role_debug(
                    "anchor-branch-entry",
                    "[AnchorBranchEntry] "
                    f"branch={active_branch_id} "
                    f"travel={anchor_branch_entry_traveled:.2f}/{anchor_branch_entry_target:.2f}",
                )

                # 목표 진입 거리만큼 이동하면 Branch corridor 진입 완료 → 실제 탐색 시작
                if anchor_branch_entry_traveled >= anchor_branch_entry_target:
                    origin_marker_cleared = False
                    anchor_motion_mode = "BRANCH_EXPLORE"

                    print(
                        "[AnchorBranchEntryComplete] "
                        f"branch={active_branch_id} "
                        f"travel={anchor_branch_entry_traveled:.2f}"
                    )


            # =================================================
            # 6-2. Branch Exploration
            # Anchor가 LiDAR free gap을 따라 Branch를 탐색한다.
            # Marker 또는 Dead-end 발견 시 탐색을 종료하고 Backtracking 단계로 전환한다.
            # =================================================
            elif anchor_motion_mode == "BRANCH_EXPLORE":

                # 6-2-1. 현재 ACTIVE Branch 상태 확인
                if active_branch_id is None:
                    raise RuntimeError("BRANCH_EXPLORE without active branch.")

                active_state = branch_states[active_branch_id]
                origin_marker_id = active_state["marker_id"]

                # 6-2-2. 탐색 시작 시 출발 Branch의 자기 Marker는 일시적으로 무시
                # Anchor가 Marker를 지나 Marker가 뒤쪽으로 넘어가면 ignore를 해제한다.
                if not origin_marker_cleared and origin_marker_id is not None:
                    origin_marker_local = next(
                        (local_position for robot_id, local_position in zip(observation.robot_ids, observation.relative_positions)
                        if robot_id == origin_marker_id), None
                    )

                    if origin_marker_local is not None and origin_marker_local.x < -2.0:
                        origin_marker_cleared = True
                        print("[OriginMarkerCleared] " f"branch={active_branch_id} " f"marker_id={origin_marker_id}")

                marker_to_ignore = None if origin_marker_cleared else origin_marker_id

                # 6-2-3. Branch를 충분히 전진하기 전에는 Junction 주변 Marker 오검출 방지
                # 절대 map 위치가 아닌 Branch 탐색 이후 누적된 local odometry만 사용한다.
                if branch_explore_traveled < MARKER_ACTIVATION_DISTANCE:
                    visible_marker_id = None
                else:
                    visible_marker_id = detect_visible_marker_ahead(
                        observation, robots_by_id, anchor_lidar, ignore_marker_id=marker_to_ignore
                    )

                # 6-2-4. Marker 발견 → Branch 탐색 종료 및 Backtracking 준비
                if visible_marker_id is not None:
                    stop_anchor(anchor)
                    detected_marker_id = visible_marker_id
                    detected_marker_branch = find_branch_id_by_marker(visible_marker_id, branch_states)

                    # 다른 Branch의 Marker를 반대편에서 만났다면 해당 Branch를 VISITED 처리
                    if detected_marker_branch is not None and detected_marker_branch != active_branch_id:
                        mark_branch_visited(
                            detected_marker_branch, branch_states,
                            reason="MARKER_REACHED_FROM_OPPOSITE_SIDE",
                        )

                    # 현재 LiDAR로 corridor width를 추정하고 Backtracking event 설정
                    backtrack_required_width = estimate_current_corridor_width(observation.lidar_scan)
                    backtrack_seed_id = None
                    backtrack_trigger_reason = "MARKER"
                    backtrack_event_valid = True
                    dead_end_stable_count = 0
                    anchor_motion_mode = "BACKTRACK_WAIT_SHEPHERD"

                    print(
                        "[BacktrackTrigger] "
                        f"reason=MARKER active_branch={active_branch_id} "
                        f"marker_id={visible_marker_id} corridor_width={backtrack_required_width:.2f}"
                    )

                else:
                    # 6-2-5. Marker가 없으면 LiDAR에서 전방의 진행 가능한 free gap 탐색
                    target_gap_angle = find_branch_free_gap(observation.lidar_scan)

                    # 6-2-6. 진행 가능한 gap이 없으면 Dead-end candidate로 판단
                    if target_gap_angle is None:
                        stop_anchor(anchor)

                        # 일시적인 LiDAR 관측으로 Dead-end가 확정되지 않도록 연속 관측
                        if substep_index == 0:
                            dead_end_stable_count += 1

                        # 일정 frame 동안 계속 gap이 없으면 Dead-end 최종 확정
                        if dead_end_stable_count >= DEAD_END_CONFIRM_FRAMES:
                            backtrack_required_width = estimate_current_corridor_width(observation.lidar_scan)
                            backtrack_seed_id = None
                            backtrack_trigger_reason = "DEAD_END"
                            backtrack_event_valid = True
                            anchor_motion_mode = "BACKTRACK_WAIT_SHEPHERD"

                            print(
                                "[BacktrackTrigger] "
                                f"reason=DEAD_END active_branch={active_branch_id} "
                                f"corridor_width={backtrack_required_width:.2f}"
                            )

                    else:
                        # 6-2-7. Free gap이 존재하면 Dead-end count를 초기화하고 Branch 탐색 계속
                        dead_end_stable_count = 0

                        # Swarm에서 가장 앞선 NORMAL을 기준으로 Anchor가 너무 멀리 앞서지 않도록 속도 제한
                        anchor_front_speed_cap, front_robot_id, front_gap, front_robot_speed = (
                            compute_branch_explore_anchor_speed_cap(
                                observation, robots_by_id, KNOWN_CORRIDOR_WIDTH
                            )
                        )

                        if front_robot_id is None:
                            role_debug(
                                "branch-anchor-front-follow",
                                "[BranchAnchorFrontFollow] "
                                f"branch={active_branch_id} front_robot=None "
                                "anchor_speed_cap=0.000 action=WAIT_FOR_SWARM",
                            )
                        else:
                            target_gap = max(
                                3.0 * environment.ROBOT_RADIUS,
                                ANCHOR_FRONT_TARGET_GAP_ROWS * environment.GRID_SPACING,
                            )

                            role_debug(
                                "branch-anchor-front-follow",
                                "[BranchAnchorFrontFollow] "
                                f"branch={active_branch_id} front_robot={front_robot_id} "
                                f"front_gap={front_gap:.3f} target_gap={target_gap:.3f} "
                                f"front_speed={front_robot_speed:.3f} anchor_speed_cap={anchor_front_speed_cap:.3f}",
                            )

                        # 6-2-8. LiDAR free-gap 방향으로 이동하되 swarm front 상태에 따라 전진 속도를 제한
                        anchor_yaw_deg, explore_delta = move_anchor_through_free_gap(
                            anchor, observation.lidar_scan, anchor_yaw_deg,
                            target_gap_angle, substep_dt,
                            max_forward_speed=anchor_front_speed_cap,
                        )

                        # 절대 위치 대신 실제 Anchor-local 이동량을 Branch exploration odometry에 누적
                        branch_explore_traveled += explore_delta.length()


            # =================================================
            # 6-3. Backtracking Shepherd Formation
            # Marker/Dead-end로 Branch 탐색이 종료되면 Anchor를 정지시키고,
            # Anchor 주변 NORMAL에서 Hop 0→1→2 통신 cohort를 구성해 Backtracking Shepherd로 전환한다.
            #
            #find_backtracking_seed()
            # ↓
            # Hop (Anchor에 가장 가까운 NORMAL을 seed로 선정)
            #
            # collect_backtracking_seed_one_two_hop()
            # ↓
            # Hop0 + Hop1 + Hop2 (hop 구조를 만듦)
            #
            # start_backtracking_shepherd_formation()
            # ↓
            # FORM
            #
            # relay_backtracking_shepherd_command()
            # ↓
            # 명령 전파 (실제 명령 전달)
            #
            # prepare_backtracking_shepherd_push()
            # ↓
            # PUSH
            # =================================================
            elif anchor_motion_mode == "BACKTRACK_WAIT_SHEPHERD":

                # 6-3-1. Shepherd 모집 동안 Anchor 정지
                stop_anchor(anchor)

                # Marker/Dead-end가 실제 확인된 경우에만 Backtracking 허용
                if not backtrack_event_valid:
                    print("[BacktrackBlocked] reason=NO_VALID_TERMINAL_EVENT")
                    backtrack_seed_id = None
                    backtrack_formation_ids.clear()
                    backtrack_chain_order.clear()
                    backtrack_formation_stable_count = 0
                    anchor_motion_mode = "BRANCH_EXPLORE"

                else:
                    if active_branch_id is None:
                        raise RuntimeError("BACKTRACK_WAIT_SHEPHERD without active branch.")

                    # 6-3-2. 아직 Shepherd cohort가 확정되지 않았다면 Hop 기반 모집 시작
                    if not backtrack_formation_ids:

                        # Hop 0: Anchor에 가장 가까운 NORMAL을 seed로 선정
                        if backtrack_seed_id is None:
                            backtrack_seed_id = find_backtracking_seed(observation, robots_by_id)

                            # 새 seed가 실제로 선정된 경우 로그 출력
                            if backtrack_seed_id is not None:
                                print(
                                    "[BacktrackingSeedSelected] "
                                    f"branch={active_branch_id} "
                                    f"seed={backtrack_seed_id} "
                                    f"reason={backtrack_trigger_reason}"
                                )

                        # Seed가 없으면 Anchor는 정지하고 NORMAL swarm이 가까워질 때까지 대기
                        if backtrack_seed_id is None:
                            role_debug(
                                "backtrack-seed-wait",
                                "[BacktrackingSeedWait] "
                                f"branch={active_branch_id} reason={backtrack_trigger_reason} seed=None",
                            )

                        else:
                            # 6-3-3. Seed를 Hop 0으로 두고 COMM_RANGE를 따라 Hop 1, Hop 2까지 NORMAL 모집
                            cohort_ids, hop_layers = collect_backtracking_seed_one_two_hop(
                                observation, robots_by_id, backtrack_seed_id
                            )
                            hop_counts = [len(layer) for layer in hop_layers]

                            role_debug(
                                "backtrack-seed-hop",
                                "[BacktrackingSeedHop] "
                                f"branch={active_branch_id} seed={backtrack_seed_id} "
                                f"hop_counts={hop_counts} cohort={len(cohort_ids)}",
                            )

                            # Seed/cohort가 유효하지 않으면 다음 frame에서 seed부터 다시 선정
                            if not cohort_ids:
                                backtrack_seed_id = None

                            else:
                                # 6-3-4. Hop 0~2 전체 cohort를 Backtracking Shepherd formation으로 전환
                                # required_count로 자르지 않고 모집된 NORMAL 전체를 사용
                                formation_ready = start_backtracking_shepherd_formation(
                                    observation, set(cohort_ids), robots_by_id,
                                    backtrack_seed_id, active_branch_id, backtrack_event_valid,
                                )

                                if formation_ready:
                                    captured_ids = sorted(cohort_ids)

                                    # 같은 Shepherd cohort에 PUSH 명령 전달
                                    push_ready = prepare_backtracking_shepherd_push(
                                        observation, captured_ids, robots_by_id,
                                        backtrack_seed_id, active_branch_id,
                                    )

                                    # PUSH 전달 실패 시 NORMAL로 복구하고 다음 frame에서 재시도
                                    if not push_ready:
                                        release_shepherd_group(set(cohort_ids), robots_by_id)
                                        backtrack_seed_id = None

                                    else:
                                        # 6-3-5. Shepherd membership LOCK
                                        # Parent Junction 복귀 전까지 seed/hop/cohort를 재선정하지 않는다.
                                        backtrack_formation_ids = set(captured_ids)
                                        backtrack_chain_order = list(captured_ids)
                                        branch_states[active_branch_id]["backtrack_shepherd_ids"] = set(captured_ids)
                                        backtrack_required_count = len(captured_ids)

                                        # 6-3-6. 현재 Anchor heading의 반대 방향을 Junction 복귀 방향으로 설정
                                        backtrack_push_yaw_deg = normalize_angle(anchor_yaw_deg + 180.0)
                                        anchor_yaw_deg = backtrack_push_yaw_deg
                                        backtrack_reverse_stable_count = 0

                                        # 6-3-7. 실제 Pressure Push 전에 wall/collision recovery 단계 수행
                                        backtrack_wall_detach_resume_mode = "PRESSURE_PUSH"
                                        anchor_motion_mode = "BACKTRACK_WALL_DETACH"

                                        print(
                                            "[BacktrackingTopologyCaptured] "
                                            f"branch={active_branch_id} reason={backtrack_trigger_reason} "
                                            f"seed={backtrack_seed_id} hop_counts={hop_counts} "
                                            f"captured={len(captured_ids)} ids={captured_ids} "
                                            f"return_yaw={backtrack_push_yaw_deg:.2f} "
                                            "topology_locked=True reformation=False"
                                        )


            # =================================================
            # 6-4. Backtracking Shepherd Wall Detach
            # Backtracking Shepherd가 Junction 방향으로 PUSH하기 전에 wall과 충돌하는지 확인하고,
            # 충돌 시 formation 전체를 좌/우로 이동시켜 전진 가능한 위치로 복구한다.
            # =================================================
            elif anchor_motion_mode == "BACKTRACK_WALL_DETACH":

                # Junction 복귀 방향이 설정되지 않은 경우 실행 불가
                if backtrack_push_yaw_deg is None:
                    raise RuntimeError("BACKTRACK_WALL_DETACH without return heading.")

                # Wall detach 동안 Anchor는 정지
                stop_anchor(anchor)

                # Shepherd formation 전체의 전진 가능 여부 확인
                # 막혀 있으면 topology를 유지한 채 LEFT/RIGHT 방향으로 함께 이동
                detach_ready, detach_blocked_ids, detach_direction = detach_backtracking_shepherd_group_step(
                    observation, backtrack_chain_order, robots_by_id,
                    backtrack_push_yaw_deg, substep_dt, backtrack_wall_detach_direction,
                )

                # 6-4-1. Detach가 시작되면 처음 선택한 LEFT/RIGHT 방향을 episode 동안 유지
                if backtrack_wall_detach_direction is None and detach_direction in ("LEFT", "RIGHT"):
                    backtrack_wall_detach_direction = detach_direction

                # 6-4-2. 모든 Shepherd가 Junction 방향으로 다시 전진 가능하면 detach 종료
                if detach_ready:
                    next_mode = backtrack_wall_detach_resume_mode

                    # Detach 방향을 초기화하여 다음 wall contact에서는 방향을 새로 선택
                    backtrack_wall_detach_direction = None
                    anchor_motion_mode = next_mode

                    print(
                        "[BacktrackWallDetachComplete] "
                        f"branch={active_branch_id} count={len(backtrack_formation_ids)} "
                        f"resume={next_mode} forward_ready=True"
                    )

                else:
                    # 아직 막힌 Shepherd가 있으면 같은 방향으로 wall detach 계속 수행
                    role_debug(
                        "backtrack-wall-detach-progress",
                        "[BacktrackWallDetachProgress] "
                        f"branch={active_branch_id} blocked_ids={sorted(detach_blocked_ids)} "
                        f"direction={detach_direction}",
                    )


            # =================================================
            # 6-5. Pressure Push
            # Backtracking Shepherd formation이 Junction 복귀 방향으로 이동하며 NORMAL을 물리적으로 밀어낸다.
            # NORMAL에는 별도 Backtracking force를 주지 않고 기존 SPH만 적용한다.
            # =================================================
            elif anchor_motion_mode == "PRESSURE_PUSH":

                # Junction 복귀 방향이 설정되지 않은 경우 실행 불가
                if backtrack_push_yaw_deg is None:
                    raise RuntimeError("PRESSURE_PUSH without backtrack push heading.")

                # 6-5-1. 실제 PUSH 전에 Shepherd가 진행 방향의 wall과 충돌하는지 확인
                pressure_wall_blockers = backtracking_forward_wall_blockers(
                    backtrack_chain_order, robots_by_id, backtrack_push_yaw_deg, substep_dt
                )

                if pressure_wall_blockers:
                    # Wall에 막히면 PUSH를 중단하고 6-4 Wall Detach로 전환
                    stop_anchor(anchor)
                    backtrack_wall_detach_resume_mode = "PRESSURE_PUSH"
                    backtrack_wall_detach_direction = None
                    anchor_motion_mode = "BACKTRACK_WALL_DETACH"

                    print(
                        "[PressurePushWallRecovery] "
                        f"branch={active_branch_id} blocked_ids={sorted(pressure_wall_blockers)} "
                        "resume=PRESSURE_PUSH"
                    )

                else:
                    # 6-5-2. Wall 문제가 없으면 Shepherd formation을 유지하며 Junction 방향으로 Pressure Push
                    # NORMAL과 접촉하면 물리적 contact acceleration을 전달
                    backtrack_contact_accelerations = update_pressure_push(
                        observation, backtrack_chain_order, robots_by_id,
                        backtrack_push_yaw_deg, substep_dt,
                    )

                    # 6-5-3. Anchor도 Backtracking Shepherd와 함께 return corridor를 따라 이동
                    follow_backtrack_corridor_with_comm_guard(
                        anchor, robots, anchor_lidar, set(backtrack_formation_ids),
                        anchor_yaw_deg, substep_dt,
                    )


            # =================================================
            # 6-6. Flow Backtracking
            # 역방향 흐름이 형성된 뒤 기존 Backtracking Shepherd cohort를 유지한 채
            # NORMAL swarm을 밀면서 Parent Junction 방향으로 실제 복귀한다.
            # =================================================
            elif anchor_motion_mode == "FLOW_BACKTRACK":

                # Junction 복귀 방향이 설정되지 않은 경우 실행 불가
                if backtrack_push_yaw_deg is None:
                    raise RuntimeError("FLOW_BACKTRACK without backtrack push heading.")

                # 6-6-1. ±90° LiDAR lateral range 변화로 Parent Junction entrance 탐지
                return_entrance_reached = False
                return_left = return_right = 0.0
                return_delta_left = return_delta_right = 0.0

                # 같은 frame의 substep마다 중복 갱신하지 않고 frame당 한 번만 detector 갱신
                if substep_index == 0:
                    return_entrance_reached, return_left, return_right, return_delta_left, return_delta_right = (
                        update_junction_entrance_detector(anchor_lidar)
                    )
                    print(
                        "[ReturnJunctionProbe] "
                        f"branch={active_branch_id} L={return_left:.2f} R={return_right:.2f} "
                        f"dL={return_delta_left:.2f} dR={return_delta_right:.2f}"
                    )

                # 6-6-2. Parent Junction entrance 도착 → Anchor 정지 후 Junction 재진입 준비
                if return_entrance_reached:
                    stop_anchor(anchor)
                    return_junction_entrance_reached = True

                    # Return corridor에서 관측한 wall baseline으로 stationary Junction threshold 고정
                    freeze_stationary_threshold(anchor_lidar)
                    return_left_wall = anchor_lidar.lateral_baseline_left
                    return_right_wall = anchor_lidar.lateral_baseline_right

                    if return_left_wall is None or return_right_wall is None:
                        raise RuntimeError("Missing return corridor wall calibration.")

                    # 현재 Anchor 위치를 local origin으로 하여 들어온 Branch mouth의 좌/우 corner 저장
                    return_base_entrance_corners_local = (
                        pygame.Vector2(0.0, -return_left_wall),
                        pygame.Vector2(0.0, return_right_wall),
                    )

                    # Parent Junction center 재추정을 위한 상태 초기화
                    return_center_target_samples.clear()
                    return_locked_center_target_local = None
                    return_center_traveled_local.update(0.0, 0.0)
                    return_center_stable_count = 0
                    anchor_motion_mode = "RETURN_JUNCTION_ENTRANCE"

                    print(
                        "[ParentJunctionEntranceReached] "
                        f"branch={active_branch_id} L={return_left:.2f} R={return_right:.2f} "
                        f"dL={return_delta_left:.2f} dR={return_delta_right:.2f}"
                    )

                else:
                    # 6-6-3. 아직 Parent Junction이 아니면 return corridor의 corner 탐지
                    backtrack_corner_angle = detect_backtrack_corner(observation.lidar_scan)

                    if backtrack_corner_angle is not None:
                        # Corner에서도 기존 Shepherd ID와 topology를 그대로 유지
                        stop_anchor(anchor)
                        backtrack_corner_exit_stable_count = 0
                        backtrack_corner_count += 1
                        anchor_motion_mode = "BACKTRACK_CORNER_TURN"

                        print(
                            "[BacktrackCornerDetected] "
                            f"branch={active_branch_id} corner_angle={backtrack_corner_angle:.2f} "
                            f"shepherd_count={len(backtrack_formation_ids)} "
                            "release=False same_cohort=True"
                        )

                    else:
                        # 6-6-4. Straight corridor에서는 PUSH 전에 Shepherd wall blocker 확인
                        flow_wall_blockers = backtracking_forward_wall_blockers(
                            backtrack_chain_order, robots_by_id,
                            backtrack_push_yaw_deg, substep_dt,
                        )

                        if flow_wall_blockers:
                            # Wall에 막히면 6-4 Wall Detach로 이동 후 다시 FLOW_BACKTRACK으로 복귀
                            stop_anchor(anchor)
                            backtrack_wall_detach_resume_mode = "FLOW_BACKTRACK"
                            backtrack_wall_detach_direction = None
                            anchor_motion_mode = "BACKTRACK_WALL_DETACH"

                            print(
                                "[FlowBacktrackWallRecovery] "
                                f"branch={active_branch_id} blocked_ids={sorted(flow_wall_blockers)} "
                                "resume=FLOW_BACKTRACK"
                            )

                        else:
                            # 6-6-5. Wall 문제가 없으면 동일 Shepherd formation으로 physical push 계속
                            backtrack_contact_accelerations = update_pressure_push(
                                observation, backtrack_chain_order, robots_by_id,
                                backtrack_push_yaw_deg, substep_dt,
                            )

                            # Anchor도 Shepherd와 함께 Parent Junction 방향으로 이동
                            follow_backtrack_corridor_with_comm_guard(
                                anchor, robots, anchor_lidar, set(backtrack_formation_ids),
                                anchor_yaw_deg, substep_dt,
                            )


            # =================================================
            # 6-7. Backtracking Corner Turn
            # Backtracking 중 corner를 만나면 기존 Shepherd cohort를 그대로 유지한 채
            # Anchor의 LiDAR free-gap 코너링을 Shepherd들이 함께 따라가며 새로운 return corridor에 정렬한다.
            # =================================================
            elif anchor_motion_mode == "BACKTRACK_CORNER_TURN":

                # 6-7-1. 현재 LiDAR에서 코너를 빠져나갈 수 있는 free gap 탐색
                target_gap_angle = find_branch_free_gap(observation.lidar_scan)

                if target_gap_angle is None:
                    # 진행 가능한 gap이 없으면 Anchor와 Shepherd를 이동시키지 않고 대기
                    stop_anchor(anchor)
                    backtrack_corner_exit_stable_count = 0

                    role_debug(
                        "backtrack-corner-gap-wait",
                        f"[BacktrackCornerTurnWait] branch={active_branch_id} reason=NO_FREE_GAP",
                    )

                else:
                    # 6-7-2. 동일 Shepherd cohort가 Anchor의 코너링 motion을 함께 수행
                    # Shepherd release/re-recruit/ID 변경 없이 기존 topology 유지
                    new_anchor_yaw, _corner_delta, corner_motion_ok = move_anchor_with_backtracking_shepherds(
                        anchor, observation.lidar_scan, set(backtrack_formation_ids),
                        robots_by_id, anchor_yaw_deg, target_gap_angle, substep_dt,
                    )

                    if corner_motion_ok:
                        anchor_yaw_deg = new_anchor_yaw

                    # 6-7-3. 코너 이후 새로운 straight corridor에 정렬되었는지 확인
                    corridor_reacquired = backtrack_corridor_reacquired(
                        observation.lidar_scan, target_gap_angle
                    )

                    # 연속된 안정 관측으로 corner 탈출 여부 확인
                    if substep_index == 0:
                        if corridor_reacquired and corner_motion_ok:
                            backtrack_corner_exit_stable_count += 1
                        else:
                            backtrack_corner_exit_stable_count = 0

                        print(
                            "[BacktrackCornerTurn] "
                            f"branch={active_branch_id} corner_index={backtrack_corner_count} "
                            f"gap_angle={target_gap_angle:.2f} yaw={anchor_yaw_deg:.2f} "
                            f"aligned={corridor_reacquired} motion_ok={corner_motion_ok} "
                            f"stable={backtrack_corner_exit_stable_count}/{BACKTRACK_CORNER_EXIT_STABLE_SCANS} "
                            f"same_shepherds={len(backtrack_formation_ids)}"
                        )

                    # 6-7-4. 새로운 corridor 정렬이 안정적으로 확인되면 corner 통과 완료
                    if backtrack_corner_exit_stable_count >= BACKTRACK_CORNER_EXIT_STABLE_SCANS:

                        # 현재 Anchor heading을 새로운 Junction 복귀 방향으로 갱신
                        backtrack_push_yaw_deg = anchor_yaw_deg
                        backtrack_corner_exit_stable_count = 0

                        # 새로운 corridor 기준으로 Parent Junction entrance detector 재학습
                        reset_junction_entrance_detector(anchor_lidar)

                        # Shepherd를 다시 모집하지 않고 동일 cohort로 Flow Backtracking 재개
                        anchor_motion_mode = "FLOW_BACKTRACK"

                        print(
                            "[BacktrackCornerExit] "
                            f"branch={active_branch_id} corner_index={backtrack_corner_count} "
                            f"new_return_yaw={backtrack_push_yaw_deg:.2f} "
                            f"same_shepherd_count={len(backtrack_formation_ids)} recruit=False"
                        )


            # =================================================
            # 6-8. Parent Junction Stationary Confirmation
            # Parent Junction 입구에 도착한 Anchor를 정지시키고 LiDAR로 Junction을 stationary verification한다.
            # 확인 중에도 기존 Shepherd의 Pressure Push는 유지하여 swarm의 reverse flow를 계속시킨다.
            # =================================================
            elif anchor_motion_mode == "RETURN_JUNCTION_ENTRANCE":

                # 6-8-1. Junction 입구에서 Anchor 정지
                stop_anchor(anchor)

                if backtrack_push_yaw_deg is None:
                    raise RuntimeError("RETURN_JUNCTION_ENTRANCE without push heading.")

                # 6-8-2. Anchor가 정지해 있는 동안에도 동일 Shepherd cohort의 Pressure Push 유지
                backtrack_contact_accelerations = update_pressure_push(
                    observation, backtrack_chain_order, robots_by_id,
                    backtrack_push_yaw_deg, substep_dt,
                )

                # 6-8-3. 정지 상태의 LiDAR 관측으로 Parent Junction 최종 확인
                if substep_index == 0 and update_stationary_junction_confirmation(anchor_lidar):
                    # Junction 확인 완료 → center 추정 단계 준비
                    return_center_target_samples.clear()
                    anchor_motion_mode = "RETURN_CENTER_ESTIMATE"

                    print(
                        "[ReturnJunctionConfirmed] "
                        f"branch={active_branch_id}"
                    )


            # =================================================
            # 6-9. Parent Junction Center Estimation
            # stationary LiDAR에서 진입 Branch와 반대편 Branch의 mouth corner를 이용해
            # Parent Junction center를 Anchor-local 좌표로 반복 추정하고 안정된 평균 위치를 고정한다.
            # =================================================
            elif anchor_motion_mode == "RETURN_CENTER_ESTIMATE":

                # 6-9-1. Center 추정 동안 Anchor는 정지
                stop_anchor(anchor)

                if backtrack_push_yaw_deg is None:
                    raise RuntimeError("RETURN_CENTER_ESTIMATE without push heading.")

                # Anchor가 정지해 있어도 기존 Shepherd의 Pressure Push는 계속 유지
                backtrack_contact_accelerations = update_pressure_push(
                    observation, backtrack_chain_order, robots_by_id,
                    backtrack_push_yaw_deg, substep_dt,
                )

                # 6-9-2. Frame당 한 번씩 Junction center 후보 추정
                if substep_index == 0 and return_base_entrance_corners_local is not None:
                    return_base_left, return_base_right = return_base_entrance_corners_local

                    # 현재 진입 Branch 반대편의 두 physical corner를 LiDAR로 검출
                    front_corners = extract_front_mouth_corners(anchor_lidar)

                    if front_corners is None:
                        candidate = None
                    else:
                        return_front_left, return_front_right = front_corners

                        # 진입측 2개 + 반대편 2개 corner로 Anchor-local Junction center 추정
                        candidate = estimate_local_junction_center(
                            return_base_left, return_base_right,
                            return_front_left, return_front_right,
                        )

                    # 6-9-3. 유효한 center를 얻지 못하면 누적 sample 초기화
                    if candidate is None:
                        return_center_target_samples.clear()
                        print("[ReturnCenterScan] center=NOT_AVAILABLE")

                    else:
                        # 이전 sample과 차이가 너무 크면 불안정한 추정으로 보고 다시 수집
                        if return_center_target_samples:
                            previous = return_center_target_samples[-1]
                            if (candidate - previous).length() > ANCHOR_TARGET_SAMPLE_TOLERANCE:
                                return_center_target_samples.clear()

                        return_center_target_samples.append(candidate.copy())

                        print(
                            "[ReturnCenterScan] "
                            f"candidate=({candidate.x:.2f},{candidate.y:.2f}) "
                            f"samples={len(return_center_target_samples)}/{ANCHOR_ENTRANCE_STATIONARY_SCANS}"
                        )

                        # 6-9-4. 충분한 수의 안정된 center sample이 모이면 평균 위치를 최종 target으로 고정
                        if len(return_center_target_samples) >= ANCHOR_ENTRANCE_STATIONARY_SCANS:
                            total = pygame.Vector2()
                            for sample in return_center_target_samples:
                                total += sample

                            return_locked_center_target_local = total / len(return_center_target_samples)

                            # Center 이동을 위한 Anchor-local odometry 초기화
                            return_center_traveled_local.update(0.0, 0.0)
                            return_center_stable_count = 0
                            anchor_motion_mode = "RETURN_CENTER_TRANSIT"

                            print(
                                "[ReturnCenterTargetLocked] "
                                f"target=({return_locked_center_target_local.x:.2f},"
                                f"{return_locked_center_target_local.y:.2f})"
                            )


            # =================================================
            # 6-10. Parent Junction Center Transit
            # 고정된 Anchor-local Junction center target까지 Anchor를 이동시키고,
            # 이동 중에도 기존 Shepherd의 Pressure Push를 유지한다.
            # =================================================
            elif anchor_motion_mode == "RETURN_CENTER_TRANSIT":

                # 고정된 Junction center target이 없으면 실행 불가
                if return_locked_center_target_local is None:
                    raise RuntimeError("RETURN_CENTER_TRANSIT without locked center.")

                if backtrack_push_yaw_deg is None:
                    raise RuntimeError("RETURN_CENTER_TRANSIT without push heading.")

                # 6-10-1. Anchor가 Junction center로 이동하는 동안에도 Shepherd reverse flow 유지
                backtrack_contact_accelerations = update_pressure_push(
                    observation, backtrack_chain_order, robots_by_id,
                    backtrack_push_yaw_deg, substep_dt,
                )

                # 6-10-2. 고정된 Anchor-local center target을 향해 Anchor 이동
                return_centered_now, return_local_delta = move_anchor_to_locked_center(
                    anchor, return_locked_center_target_local,
                    return_center_traveled_local, anchor_yaw_deg, substep_dt,
                )

                # 실제 Anchor-local 이동량을 누적하여 target까지 남은 거리를 추적
                return_center_traveled_local += return_local_delta

                # 6-10-3. Center 도달 상태가 연속적으로 유지되는지 확인
                if return_centered_now:
                    if substep_index == 0:
                        return_center_stable_count += 1
                else:
                    return_center_stable_count = 0

                # 6-10-4. 충분한 횟수 동안 center 도달이 유지되면 Parent Junction center 복귀 완료
                if return_center_stable_count >= ANCHOR_CENTER_STABLE_SCANS:
                    stop_anchor(anchor)
                    anchor_motion_mode = "RETURN_JUNCTION_CENTERED"

                    print(
                        "[ReturnJunctionCentered] "
                        f"branch={active_branch_id} "
                        f"target=({return_locked_center_target_local.x:.2f},"
                        f"{return_locked_center_target_local.y:.2f}) "
                        f"travel=({return_center_traveled_local.x:.2f},"
                        f"{return_center_traveled_local.y:.2f})"
                    )


            # =================================================
            # 6-11. Wait for Swarm Return Preparation
            # Anchor가 Parent Junction center에 도착하면 정지하고,
            # 처음 Branch geometry를 등록했던 Junction-local 기준 heading으로 복원한 뒤
            # Backtracking swarm의 실제 Junction 복귀 완료를 기다리는 단계로 전환한다.
            # =================================================
            elif anchor_motion_mode == "RETURN_JUNCTION_CENTERED":

                # 6-11-1. Parent Junction center에서 Anchor 정지
                stop_anchor(anchor)

                # 처음 Branch geometry 등록 시 저장한 Junction 기준 heading이 없으면 실행 불가
                if junction_reference_yaw_deg is None:
                    raise RuntimeError(
                        "RETURN_JUNCTION_CENTERED without Junction reference heading."
                    )

                # 6-11-2. Anchor heading을 처음 Branch geometry를 등록했던 Junction-local 기준으로 복원
                anchor_yaw_deg = junction_reference_yaw_deg

                # 6-11-3. Swarm 복귀 완료를 판단하기 위한 안정 count 초기화
                swarm_return_stable_count = 0

                # Anchor는 Junction center에서 대기하며 swarm의 실제 복귀 완료를 확인
                anchor_motion_mode = "WAIT_SWARM_RETURN"

                print(
                    "[WaitSwarmReturnStart] "
                    f"branch={active_branch_id}"
                )


            # =================================================
            # 6-12. Wait for Swarm Return
            # Anchor는 Parent Junction center에서 정지한 채 기다리고,
            # 기존 Backtracking Shepherd가 남은 NORMAL을 계속 밀어 Branch 내부 swarm의 실제 복귀를 완료시킨다.
            # =================================================
            elif anchor_motion_mode == "WAIT_SWARM_RETURN":

                # 6-12-1. Swarm 복귀를 기다리는 동안 Anchor는 Junction center에서 정지
                stop_anchor(anchor)

                if active_branch_id is None:
                    raise RuntimeError("WAIT_SWARM_RETURN without active branch.")

                if backtrack_push_yaw_deg is None:
                    raise RuntimeError("WAIT_SWARM_RETURN without push heading.")

                # 현재 탐색을 마치고 복귀 중인 ACTIVE Branch 정보
                active_branch = next(
                    branch for branch in confirmed_branches
                    if branch["id"] == active_branch_id
                )

                # 6-12-2. 기존 Backtracking Shepherd의 Pressure Push를 계속 유지하여
                # 아직 Branch 내부에 남아 있는 NORMAL을 Parent Junction 방향으로 밀어냄
                backtrack_contact_accelerations = update_pressure_push(
                    observation, backtrack_chain_order, robots_by_id,
                    backtrack_push_yaw_deg, substep_dt,
                )

                # 6-12-3. Frame당 한 번씩 Branch 내부 swarm의 복귀 완료 여부 확인
                if substep_index == 0:
                    swarm_return_ready, remaining_normals, remaining_backtrack = (
                        branch_swarm_return_complete(
                            observation, active_branch, robots_by_id,
                            set(backtrack_formation_ids),
                        )
                    )

                    # 복귀 완료 상태가 연속적으로 유지되는지 확인
                    if swarm_return_ready:
                        swarm_return_stable_count += 1
                    else:
                        swarm_return_stable_count = 0

                    print(
                        "[SwarmReturnCheck] "
                        f"branch={active_branch_id} "
                        f"remaining_normal={len(remaining_normals)} "
                        f"remaining_backtrack={len(remaining_backtrack)} "
                        f"stable={swarm_return_stable_count}/{SWARM_RETURN_STABLE_SCANS}"
                    )

                    # 6-12-4. Swarm 복귀가 충분한 frame 동안 안정적으로 확인되면 Branch 복귀 마무리 단계로 전환
                    if swarm_return_stable_count >= SWARM_RETURN_STABLE_SCANS:
                        anchor_motion_mode = "FINALIZE_BRANCH_RETURN"

                        print(
                            "[SwarmReturnComplete] "
                            f"branch={active_branch_id}"
                        )


            # =================================================
            # 6-13. Finalize Branch Return
            # Branch 내부 swarm의 물리적 복귀가 완료되면 Backtracking Shepherd를 해제하고,
            # 현재 Branch를 VISITED로 확정한 뒤 입구 Shepherd를 다시 형성할 수 있도록 상태를 초기화한다.
            # =================================================
            elif anchor_motion_mode == "FINALIZE_BRANCH_RETURN":

                # 6-13-1. Parent Junction center에서 Anchor 정지
                stop_anchor(anchor)

                if active_branch_id is None:
                    raise RuntimeError("FINALIZE_BRANCH_RETURN without active branch.")

                # 6-13-2. 물리적 복귀에 사용한 Backtracking Shepherd를 NORMAL로 복구
                release_shepherd_group(
                    set(backtrack_formation_ids), robots_by_id
                )
                branch_states[active_branch_id]["backtrack_shepherd_ids"].clear()
                backtrack_formation_ids.clear()
                backtrack_chain_order.clear()

                # 6-13-3. 실제 Branch 복귀가 완료된 시점에서 현재 Branch를 VISITED로 확정
                mark_branch_visited(
                    active_branch_id,
                    branch_states,
                    reason="PHYSICAL_RETURN_COMPLETE",
                )

                print(
                    "[PhysicalReturnComplete] "
                    f"branch={active_branch_id}"
                )

                # 6-13-4. 탐색하면서 열었던 Branch 입구를 다시 Shepherd로 막을 수 있도록 상태 초기화
                branch_states[active_branch_id]["initial_shepherd_ids"].clear()
                branch_states[active_branch_id]["initial_sealed"] = False
                branch_states[active_branch_id]["opened"] = False

                # 6-13-5. 현재 Branch에서 사용한 Backtracking runtime 상태 초기화
                backtrack_event_valid = False
                backtrack_trigger_reason = None
                backtrack_push_yaw_deg = None
                backtrack_seed_id = None
                backtrack_formation_stable_count = 0
                backtrack_reverse_stable_count = 0

                # VISITED Branch 입구의 Shepherd formation을 다시 형성하는 단계로 전환
                anchor_motion_mode = "REFORM_VISITED_BRANCH_SHEPHERD"


            # =================================================
            # 6-14. Reform Visited Branch Shepherd
            # 물리적 복귀가 완료되어 VISITED가 된 Branch의 입구에
            # Shepherd formation을 다시 형성해 해당 Branch를 물리적으로 다시 막는다.
            # =================================================
            elif anchor_motion_mode == "REFORM_VISITED_BRANCH_SHEPHERD":

                # 6-14-1. Shepherd를 다시 형성하는 동안 Anchor는 Junction center에서 정지
                stop_anchor(anchor)

                if active_branch_id is None:
                    raise RuntimeError("REFORM_VISITED_BRANCH_SHEPHERD without branch.")

                # 방금 탐색을 완료한 VISITED Branch 정보 가져오기
                active_branch = next(
                    branch for branch in confirmed_branches
                    if branch["id"] == active_branch_id
                )

                # 6-14-2. 해당 Branch 입구에 Initial Shepherd formation을 다시 형성
                resealed = form_initial_junction_shepherd_boundaries(
                    observation, robots_by_id,
                    [active_branch], branch_states,
                )

                # 6-14-3. Branch 입구가 다시 완전히 막히면 다음 DFS Branch 선택 단계로 전환
                if resealed:
                    print(
                        "[VisitedBranchResealed] "
                        f"branch={active_branch_id}"
                    )

                    anchor_motion_mode = "SELECT_NEXT_BRANCH"


            # =================================================
            # 6-15. Select Next Branch
            # 현재 Branch의 탐색·복귀·재봉쇄가 끝나면 DFS branch_order에서
            # 다음 UNVISITED Branch를 선택하고 새로운 Branch exploration을 준비한다.
            # =================================================
            elif anchor_motion_mode == "SELECT_NEXT_BRANCH":

                # 6-15-1. 다음 Branch를 선택하는 동안 Anchor는 Junction center에서 정지
                stop_anchor(anchor)

                # DFS branch_order에서 다음 UNVISITED Branch 탐색
                next_branch_id = find_next_unvisited_branch(
                    branch_order, branch_states,
                )

                # 6-15-2. 더 이상 UNVISITED Branch가 없으면 Root Junction DFS 완료
                if next_branch_id is None:
                    anchor_motion_mode = "ROOT_COMPLETE"
                    print("[RootDFSComplete]")

                else:
                    # 6-15-3. 다음 Branch를 ACTIVE로 선택하고 해당 입구의 Shepherd를 열어 탐색 가능하게 함
                    active_branch_id = next_branch_id
                    select_runtime_branch(active_branch_id, branch_states)
                    open_selected_branch(
                        active_branch_id, branch_states, robots_by_id,
                    )

                    active_branch = next(
                        branch for branch in confirmed_branches
                        if branch["id"] == active_branch_id
                    )

                    if junction_reference_yaw_deg is None:
                        raise RuntimeError("Missing Junction reference heading.")

                    # 6-15-4. Junction 기준 heading + 선택 Branch의 상대 angle로 새 탐색 heading 설정
                    relative_turn_deg = active_branch["center_angle"]
                    anchor_yaw_deg = normalize_angle(
                        junction_reference_yaw_deg + relative_turn_deg
                    )

                    # Branch corridor 내부로 진입하기 위한 목표 거리 설정 및 odometry 초기화
                    anchor_branch_entry_target = (
                        active_branch["entrance_midpoint"].length()
                        + ANCHOR_BRANCH_ENTRY_MARGIN
                    )
                    anchor_branch_entry_traveled = 0.0
                    branch_explore_traveled = 0.0

                    # 6-15-5. 새로운 Branch 탐색을 위해 이전 Branch의 runtime 상태 초기화
                    origin_marker_cleared = False
                    detected_marker_id = None
                    dead_end_stable_count = 0
                    backtrack_event_valid = False
                    backtrack_trigger_reason = None
                    backtrack_seed_id = None
                    backtrack_push_yaw_deg = None
                    backtrack_required_count = 0
                    backtrack_formation_ids.clear()
                    backtrack_chain_order.clear()
                    backtrack_formation_stable_count = 0
                    backtrack_reverse_stable_count = 0
                    swarm_return_stable_count = 0

                    # Parent Junction 복귀 과정에서 사용했던 상태도 초기화
                    return_junction_entrance_reached = False
                    return_base_entrance_corners_local = None
                    return_center_target_samples.clear()
                    return_locked_center_target_local = None
                    return_center_traveled_local.update(0.0, 0.0)
                    return_center_stable_count = 0

                    # 6-15-6. 새 Branch의 입구 진입 단계부터 다시 시작
                    anchor_motion_mode = "BRANCH_ENTRY"

                    print(
                        "[AnchorBranchStart] "
                        f"branch={active_branch_id} "
                        f"relative_turn={relative_turn_deg:.2f} "
                        f"runtime_yaw={anchor_yaw_deg:.2f} "
                        f"entry_target={anchor_branch_entry_target:.2f}"
                    )


            # =================================================
            # 6-16. Root DFS Complete
            # Root Junction의 모든 Branch 탐색이 완료되면 Anchor를 정지시키고,
            # Junction-local 기준 heading으로 복원한 뒤 최종 Base 복귀 준비 단계로 전환한다.
            # =================================================
            elif anchor_motion_mode == "ROOT_COMPLETE":

                # 6-16-1. Root Junction DFS가 완료된 상태에서 Anchor 정지
                stop_anchor(anchor)

                if junction_reference_yaw_deg is None:
                    raise RuntimeError(
                        "ROOT_COMPLETE without Junction reference heading."
                    )

                # 6-16-2. Anchor heading을 처음 Branch geometry를 등록했던 Junction-local 기준으로 복원
                anchor_yaw_deg = junction_reference_yaw_deg

                # 6-16-3. 모든 Branch 탐색 완료 → 최종 Base 복귀 준비 단계로 전환
                anchor_motion_mode = "FINAL_RETURN_PREP"


            # =================================================
            # 6-17. Final Return Preparation
            # Root DFS 완료 후 Base 최종 복귀에 사용할 Shepherd group을 선택하고,
            # 나머지 Branch의 Shepherd/Marker를 NORMAL로 해제하여 전체 swarm의 합류를 준비한다.
            # =================================================
            elif anchor_motion_mode == "FINAL_RETURN_PREP":

                # 6-17-1. 최종 복귀 준비 동안 Anchor는 Root Junction center에서 정지
                stop_anchor(anchor)

                # 최종 Base 복귀 시 PUSH에 사용할 Branch 선택
                final_push_branch = find_final_push_branch(confirmed_branches)
                final_push_branch_id = final_push_branch["id"]

                # Final Push Branch를 제외한 나머지 side Branch 구분
                final_side_branch_ids = {
                    branch["id"] for branch in confirmed_branches
                    if branch["id"] != final_push_branch_id
                }

                # 6-17-2. 선택된 Branch의 Initial Shepherd들을 Final Push Shepherd로 전환
                final_push_ids = set(
                    branch_states[final_push_branch_id]["initial_shepherd_ids"]
                )

                for robot_id in final_push_ids:
                    robot = robots_by_id[robot_id]
                    robot.role = "SHEPHERD"
                    robot.role_branch = final_push_branch_id
                    robot.role_frozen = False
                    robot.role_frozen_position = None
                    robot.shepherd_mode = "FINAL_PUSH"

                # 6-17-3. Final Push Branch에 남아 있던 Marker도 NORMAL로 해제
                marker_id = branch_states[final_push_branch_id]["marker_id"]

                if marker_id is not None:
                    release_robot_to_normal(robots_by_id[marker_id])

                    print(
                        "[FinalMarkerReleased] "
                        f"branch={final_push_branch_id} marker_id={marker_id}"
                    )

                # 6-17-4. 나머지 side Branch의 Initial Shepherd와 Marker를 모두 NORMAL로 해제
                # 더 이상 Branch를 막을 필요가 없으므로 전체 swarm에 다시 합류시킴
                for branch_id in final_side_branch_ids:
                    state = branch_states[branch_id]

                    for robot_id in state["initial_shepherd_ids"]:
                        release_robot_to_normal(robots_by_id[robot_id])

                    marker_id = state["marker_id"]
                    if marker_id is not None:
                        release_robot_to_normal(robots_by_id[marker_id])

                # 6-17-5. Side Branch에서 풀린 NORMAL들의 합류 안정성을 확인하기 위한 count 초기화
                final_side_merge_stable_count = 0

                # 전체 swarm 합류를 기다리는 단계로 전환
                anchor_motion_mode = "FINAL_WAIT_SIDE_MERGE"


            # =================================================
            # 6-18. Wait for Side Swarm Merge
            # Final Push를 시작하기 전에 side Branch에서 해제된 NORMAL들이
            # Branch 내부를 빠져나와 Junction swarm에 합류했는지 확인한다.
            # =================================================
            elif anchor_motion_mode == "FINAL_WAIT_SIDE_MERGE":

                # 6-18-1. Side swarm 합류를 기다리는 동안 Anchor는 Root Junction center에서 정지
                stop_anchor(anchor)

                # Frame당 한 번씩 side Branch의 잔류 NORMAL 확인
                if substep_index == 0:
                    side_merge_ready = True

                    # 6-18-2. Final Push Branch를 제외한 모든 side Branch 검사
                    for branch_id in final_side_branch_ids:
                        side_branch = next(
                            branch for branch in confirmed_branches
                            if branch["id"] == branch_id
                        )

                        _side_return_ready, remaining_normals, _remaining_backtrack = (
                            branch_swarm_return_complete(
                                observation, side_branch, robots_by_id, set(),
                            )
                        )

                        # 하나의 side Branch라도 NORMAL이 남아 있으면 아직 합류 미완료
                        if remaining_normals:
                            side_merge_ready = False

                    # 6-18-3. 모든 side Branch가 비어 있는 상태가 연속적으로 유지되는지 확인
                    if side_merge_ready:
                        final_side_merge_stable_count += 1
                    else:
                        final_side_merge_stable_count = 0

                    # 6-18-4. Side swarm 합류가 안정적으로 확인되면 최종 Base Push 시작
                    if final_side_merge_stable_count >= FINAL_SIDE_MERGE_STABLE_SCANS:
                        final_base_return_stable_count = 0
                        anchor_motion_mode = "FINAL_BASE_PUSH"


            # =================================================
            # 6-19. Final Base Push
            # Side swarm 합류가 완료되면 Final Push Shepherd들이 swarm을 Base 방향으로 밀어 보내고,
            # swarm의 Base 복귀가 확인되면 Shepherd를 해제한 뒤 Anchor의 최종 복귀를 준비한다.
            # =================================================
            elif anchor_motion_mode == "FINAL_BASE_PUSH":

                # 6-19-1. Final Push 동안 Anchor는 Root Junction center에서 정지
                stop_anchor(anchor)

                if junction_reference_yaw_deg is None:
                    raise RuntimeError("FINAL_BASE_PUSH without Junction reference heading.")

                # 6-19-2. 선택된 Final Push Shepherd들을 이용해 swarm을 Base 방향으로 이동
                update_final_base_push(
                    final_push_ids, robots_by_id,
                    junction_reference_yaw_deg, substep_dt,
                )

                # 6-19-3. Frame당 한 번씩 swarm의 Base 복귀 완료 여부 확인
                if substep_index == 0:

                    if swarm_base_return_complete(observation):
                        final_base_return_stable_count += 1
                    else:
                        final_base_return_stable_count = 0

                    # 6-19-4. Base 복귀가 안정적으로 확인되면 Final Push 종료
                    if final_base_return_stable_count >= FINAL_BASE_RETURN_STABLE_SCANS:

                        # Final Push Shepherd들을 NORMAL로 복구
                        for robot_id in final_push_ids:
                            release_robot_to_normal(robots_by_id[robot_id])

                        # Anchor가 Base로 돌아갈 수 있도록 Junction 기준 heading의 반대 방향으로 회전
                        anchor_yaw_deg = normalize_angle(
                            junction_reference_yaw_deg + 180.0
                        )

                        # Anchor의 최종 복귀 odometry 초기화
                        final_anchor_return_traveled = 0.0

                        # Anchor 자신의 Base 복귀 단계로 전환
                        anchor_motion_mode = "FINAL_ANCHOR_RETURN"


            # =================================================
            # 6-20. Final Anchor Return
            # Swarm의 Base 복귀가 완료된 뒤 Anchor가 마지막으로 corridor를 따라 Base로 복귀하고,
            # 목표 복귀 거리에 도달하면 전체 Physical DFS exploration을 종료한다.
            # =================================================
            elif anchor_motion_mode == "FINAL_ANCHOR_RETURN":

                # 6-20-1. LiDAR/local observation을 이용해 Base 방향 corridor를 따라 Anchor 이동
                final_anchor_delta = follow_corridor_locally(
                    anchor, observation, anchor_yaw_deg, substep_dt,
                )

                # 실제 Anchor-local 전진 거리만 최종 복귀 odometry에 누적
                final_anchor_return_traveled += max(
                    0.0, final_anchor_delta.x,
                )

                # 6-20-2. Root Junction center에서 Base까지의 목표 거리에 도달하면 Anchor 복귀 완료
                if final_anchor_return_traveled >= ROOT_CENTER_TO_BASE_DISTANCE:
                    stop_anchor(anchor)

                    # 모든 Branch 탐색과 swarm/Anchor의 Base 복귀가 완료된 최종 상태로 전환
                    anchor_motion_mode = "SYSTEM_COMPLETE"


            # =================================================
            # 6-21. System Complete
            # 모든 Branch의 Physical DFS 탐색과 swarm 및 Anchor의 Base 복귀가 완료된 최종 상태.
            # 이후 Anchor를 정지 상태로 유지하며 전체 exploration을 종료한다.
            # =================================================
            elif anchor_motion_mode == "SYSTEM_COMPLETE":

                # 전체 탐색 완료 → Anchor를 계속 정지 상태로 유지
                stop_anchor(anchor)





            # =================================================
            # 7. SPH Swarm Motion
            # 각 physics substep에서 swarm의 밀도와 압력을 계산하고
            # NORMAL 로봇에 f_SPH = f_press + f_vis를 적용하여 실제 위치를 갱신한다.
            # Backtracking 중 Shepherd와의 물리적 접촉이 발생한 경우에만 contact acceleration을 추가한다.
            # =================================================

            # 7-1. 주변 로봇 탐색을 위한 physics spatial grid 생성
            physics_grid = environment.build_physics_grid(swarm_robots)

            # 각 로봇의 주변 밀도 ρ_i 계산
            environment.compute_densities(
                swarm_robots, physics_grid,
            )

            # 현재 밀도와 기준 밀도로부터 각 로봇의 압력 P_i 계산
            environment.compute_pressures(
                swarm_robots, reference_density,
            )

            # 7-2. 주변 로봇과의 상호작용으로 SPH pressure + viscosity force 계산
            # NORMAL의 기본 운동은 f_SPH = f_press + f_vis만 사용
            compute_sph_only_forces(
                swarm_robots, physics_grid, reference_density,
            )

            # 7-3. Backtracking Shepherd가 NORMAL과 물리적으로 접촉한 경우 contact acceleration 추가
            # 별도의 goal/backtracking force가 아니라 실제 robot-to-robot contact response
            for robot_id, contact_acceleration in backtrack_contact_accelerations.items():
                robot = robots_by_id[robot_id]

                # NORMAL에게만 Shepherd의 물리적 접촉 효과 적용
                if robot.role != "NORMAL":
                    continue

                robot.acceleration = environment.limit_vector(
                    robot.acceleration + contact_acceleration,
                    environment.MAX_ACCELERATION,
                )
                robot.filtered_acceleration.update(robot.acceleration)

            # 7-4. 역할에 따라 각 swarm robot의 실제 motion 처리
            for robot in swarm_robots:

                # Frozen Shepherd / Marker는 지정된 위치에 고정
                if getattr(robot, "role_frozen", False):
                    frozen_position = robot.role_frozen_position
                    robot.position.update(frozen_position)
                    robot.previous_position.update(frozen_position)
                    robot.velocity.update(0.0, 0.0)
                    robot.acceleration.update(0.0, 0.0)
                    robot.filtered_acceleration.update(0.0, 0.0)
                    robot.commanded_velocity.update(0.0, 0.0)
                    robot.observed_velocity.update(0.0, 0.0)
                    continue

                # FINAL_BASE_PUSH Shepherd는 별도 Final Push controller가 직접 이동시키므로 SPH 적분 제외
                if anchor_motion_mode == "FINAL_BASE_PUSH" and robot.robot_id in final_push_ids:
                    robot.acceleration.update(0.0, 0.0)
                    continue

                # Backtracking Shepherd는 update_pressure_push()가 formation을 직접 이동시키므로 SPH 적분 제외
                if (
                    anchor_motion_mode in (
                        "PRESSURE_PUSH",
                        "FLOW_BACKTRACK",
                        "RETURN_JUNCTION_ENTRANCE",
                        "RETURN_CENTER_ESTIMATE",
                        "RETURN_CENTER_TRANSIT",
                        "RETURN_JUNCTION_CENTERED",
                        "WAIT_SWARM_RETURN",
                        "FINALIZE_BRANCH_RETURN",
                    )
                    and robot.robot_id in backtrack_formation_ids
                ):
                    robot.acceleration.update(0.0, 0.0)
                    continue

                # 7-5. 나머지 NORMAL 로봇은 SPH 가속도를 실제 속도와 위치에 적분
                integrate_sph_only_robot(robot, substep_dt)


            # 7-6. Pressure Push to Flow Backtracking Transition
            # Pressure Push 이후 NORMAL들의 실제 observed velocity를 확인하여 충분한 로봇이 Junction 방향으로 역류하면 FLOW_BACKTRACK으로 전환한다.
            # Pressure 값 자체를 전환 조건으로 사용하지 않는다.
            if anchor_motion_mode == "PRESSURE_PUSH":

                # 7-6-1. SPH 이동이 반영된 실제 velocity를 사용하기 위해 local observation 갱신
                flow_observation = LocalObservationBuilder.build(
                    anchor, robots, anchor_lidar, anchor_yaw_deg,
                )

                # NORMAL들이 실제 Junction 복귀 방향으로 이동하고 있는지 평가
                reverse_flow_ready, reverse_eligible_count, reverse_count, reverse_ratio, mean_reverse_speed = (
                    evaluate_normal_reverse_flow(
                        flow_observation, robots_by_id, backtrack_chain_order,
                        backtrack_required_width,
                    )
                )

                # 7-6-2. Reverse flow가 연속적으로 유지되는지 frame마다 한 번씩 확인
                if substep_index == 0:
                    if reverse_flow_ready:
                        backtrack_reverse_stable_count += 1
                    else:
                        backtrack_reverse_stable_count = 0

                    print(
                        "[ReverseFlowCheck] "
                        f"branch={active_branch_id} eligible={reverse_eligible_count} "
                        f"reverse={reverse_count} ratio={reverse_ratio:.3f} "
                        f"mean_reverse_speed={mean_reverse_speed:.3f} "
                        f"ready={reverse_flow_ready} "
                        f"stable={backtrack_reverse_stable_count}/{BACKTRACK_REVERSE_STABLE_SCANS}"
                    )

                # 7-6-3. 실제 reverse flow가 충분한 frame 동안 유지되면 FLOW_BACKTRACK 시작
                if backtrack_reverse_stable_count >= BACKTRACK_REVERSE_STABLE_SCANS:

                    if backtrack_push_yaw_deg is None:
                        raise RuntimeError(
                            "FLOW_BACKTRACK without backtrack push heading."
                        )

                    # 현재 return corridor를 기준으로 Junction entrance detector를 새로 학습
                    # 이전 Root 진입 과정에서 저장된 lateral baseline/history는 제거
                    reset_junction_entrance_detector(anchor_lidar)

                    return_junction_entrance_reached = False

                    # Anchor와 동일 Shepherd cohort가 return corridor를 따라 실제 복귀 시작
                    anchor_motion_mode = "FLOW_BACKTRACK"

                    print(
                        "[FlowBacktrackStart] "
                        f"branch={active_branch_id} "
                        f"anchor_yaw={anchor_yaw_deg:.2f}"
                    )



            # =================================================
            # 8. Initial Shepherd Formation
            # Root Junction center에서 Branch geometry 등록이 완료된 뒤,
            # 각 Branch 입구의 NORMAL 로봇들을 통신 기반으로 모집하여
            # Initial Shepherd boundary를 형성한다.
            # 모든 Branch의 Shepherd boundary가 완성되면 Junction broadcast를 종료한다.
            # =================================================
            if anchor_locally_centered and confirmed_branches and not initial_shepherds_ready:

                # 8-1. SPH 이동 이후의 최신 로봇 상대 위치를 반영하기 위해 local observation 갱신
                post_move_observation = LocalObservationBuilder.build(anchor, robots, anchor_lidar, anchor_yaw_deg)

                # 각 Branch 입구에서 Initial Shepherd cohort를 모집하고
                # 물리적인 Shepherd boundary가 완성되었는지 확인
                initial_shepherds_ready = form_initial_junction_shepherd_boundaries(post_move_observation, robots_by_id, confirmed_branches, branch_states, junction_message_received_ids)

                # 8-2. 모든 Branch의 Initial Shepherd boundary가 완성되면
                # 더 이상 Junction confirmed message를 broadcast하지 않음
                if initial_shepherds_ready:
                    junction_broadcast_active = False

                    print("[AnchorBroadcast] " "command=JUNCTION_CONFIRMED " "active=False " "reason=INITIAL_SHEPHERDS_READY")

                # Initial Shepherd formation 전체 완료 로그
                if initial_shepherds_ready:
                    print("[AllInitialShepherdBoundariesReady]")



            # =================================================
            # 9. Initial Marker Creation
            # 모든 Branch의 Initial Shepherd boundary가 완성된 뒤,
            # 각 Branch에서 Initial Shepherd 한 대를 선택하여
            # 해당 Branch의 초기 UNVISITED Marker로 생성한다.
            # =================================================
            if initial_shepherds_ready and not markers_ready:
                
                # 9-1. Marker 선택에 사용할 최신 Anchor-local 상대 위치 갱신
                marker_observation = LocalObservationBuilder.build(anchor, robots, anchor_lidar, anchor_yaw_deg)

                # 각 Branch의 Initial Shepherd 중 한 대를 선택하여
                # 해당 Branch의 UNVISITED Marker로 고정
                create_initial_branch_markers(marker_observation, confirmed_branches, branch_states, robots_by_id)

                # 모든 Branch의 초기 Marker 생성 완료
                markers_ready = True

                print("[AllInitialMarkersReady]")



            # =================================================
            # 10. Runtime Branch Order
            # Initial Marker 생성이 완료된 뒤 confirmed Branch들의 순서를 뒤집어
            # 실제 Physical DFS 탐색에 사용할 Branch 방문 순서를 생성한다.
            # =================================================
            if markers_ready and not branch_order:
                # LiDAR로 등록된 confirmed Branch 순서를 역순으로 배치하여
                # DFS에서 사용할 runtime Branch 탐색 순서 생성
                branch_order = [branch["id"] for branch in reversed(confirmed_branches)]
                print("[RuntimeBranchOrder] " f"order={branch_order}")


            # =================================================
            # 11. Branch Exploration Cycle Start
            # Marker와 Branch order가 준비되면 첫 Branch도 이후 Branch와 동일하게
            # SELECT_NEXT_BRANCH를 통해 Physical DFS 탐색 cycle을 시작한다.
            # =================================================
            if markers_ready and branch_order and not first_branch_opened:
                # Branch exploration cycle 최초 시작
                first_branch_opened = True
                active_branch_id = None

                # 첫 Branch도 공통 DFS Branch 선택 로직으로 선택
                anchor_motion_mode = "SELECT_NEXT_BRANCH"
                stop_anchor(anchor)

                print("[BranchCycleStart] " f"order={branch_order}")


        # =====================================================
        # 12. End of Physics Substeps
        # 한 frame의 8개 physics substep이 모두 완료된 이후
        # rendering과 communication에 사용할 최신 상태를 갱신한다.
        # =====================================================

        # 12-1. Rendering에 사용할 최신 Anchor LiDAR scan 갱신
        anchor_lidar.scan(anchor.position, anchor_yaw_deg)


        # =====================================================
        # 13. Communication Update
        # 현재 로봇 배치를 기준으로 robot-to-robot communication 상태를 갱신한다.
        # =====================================================
        if not paused:
            environment.update_communication_system(
                robots, environment.build_spatial_grid(robots),
            )


        # =====================================================
        # 14. Runtime State Trace
        # Anchor의 state가 변경될 때만 현재 command, Branch 및 역할별 로봇 수를 출력한다.
        # =====================================================
        if anchor_motion_mode != last_logged_anchor_motion_mode:
            normal_count = sum(robot.role == "NORMAL" for robot in swarm_robots)
            shepherd_count = sum(robot.role == "SHEPHERD" for robot in swarm_robots)
            marker_count = sum(robot.role == "MARKER" for robot in swarm_robots)

            print(
                "[AnchorStage] "
                f"t={environment.simulation_time:.2f} stage={anchor_motion_mode} "
                f"command={anchor_command_name(anchor_motion_mode)} branch={active_branch_id} "
                f"yaw={anchor_yaw_deg:.1f} roles=NORMAL:{normal_count} "
                f"SHEPHERD:{shepherd_count} MARKER:{marker_count}"
            )
            last_logged_anchor_motion_mode = anchor_motion_mode


        # =====================================================
        # 15. Frozen Role Audit
        # 고정된 Shepherd/Marker가 지정된 위치에서 움직이지 않았는지 검사한다.
        # =====================================================
        if initial_shepherds_ready:
            frozen_robots = [
                robot for robot in swarm_robots
                if getattr(robot, "role_frozen", False)
            ]

            drifted = [
                (robot.robot_id, round(robot.position.distance_to(robot.role_frozen_position), 6))
                for robot in frozen_robots
                if robot.position.distance_to(robot.role_frozen_position) > 1.0e-6
            ]

            role_debug(
                "freeze-audit",
                (
                    "[FreezeAudit] "
                    f"shepherds={sum(robot.role == 'SHEPHERD' for robot in swarm_robots)} "
                    f"markers={sum(robot.role == 'MARKER' for robot in swarm_robots)} "
                    f"frozen={len(frozen_robots)} drifted={drifted[:10]}"
                ),
            )


        # =====================================================
        # 16. Rendering
        # 현재 simulation map, communication link, robots 및 control 정보를 화면에 출력한다.
        # =====================================================
        screen.fill(COLORS["background"])
        draw_map(screen, font)

        # Communication link를 robot보다 먼저 그려 robot이 link 위에 표시되도록 함
        if show_comm_links:
            environment.draw_communication_links(screen, robots)

        draw_robots(
            screen, robots, color_reference_density, anchor,
            anchor_lidar, junction_detected, anchor_yaw_deg,
        )

        draw_controls(
            screen, font, paused, show_comm_links,
            anchor_motion_mode, active_branch_id,
        )

        # 현재 frame을 실제 화면에 표시
        pygame.display.flip()


    # =====================================================
    # 17. Simulation Shutdown
    # main loop 종료 후 Pygame을 종료한다.
    # =====================================================
    pygame.quit()


if __name__ == "__main__":
    main()
