from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, order=True)
class SE2GridState:
    """A collision-checked base pose on an orientation-aware grid."""

    row: int
    column: int
    yaw_index: int


@dataclass(frozen=True)
class SE2GridPlan:
    states: tuple[SE2GridState, ...]
    actions: tuple[str, ...]
    translation_length_cells: float
    rotation_angle_rad: float
    expanded_state_count: int

    @property
    def contains_stationary_turn(self) -> bool:
        return any(action.startswith("rotate_") for action in self.actions)


def circular_yaw_indices(
    start: int,
    stop: int,
    yaw_bins: int,
    *,
    direction: int,
) -> tuple[int, ...]:
    """Return every discrete orientation touched by one directed rotation."""
    if yaw_bins < 4 or direction not in (-1, 1):
        raise ValueError("yaw_bins must be >= 4 and direction must be -1 or 1")
    current = int(start) % yaw_bins
    target = int(stop) % yaw_bins
    result = [current]
    for _ in range(yaw_bins):
        if current == target:
            return tuple(result)
        current = (current + direction) % yaw_bins
        result.append(current)
    raise ValueError("directed rotation did not reach its target")


def swept_rotation_is_safe(
    safe_masks: BoolArray,
    row: int,
    column: int,
    start_yaw_index: int,
    stop_yaw_index: int,
    *,
    direction: int,
) -> bool:
    """Validate all intermediate yaw bins, not only rotation endpoints."""
    masks = np.asarray(safe_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("safe_masks must have shape [yaw,row,column]")
    _, height, width = masks.shape
    if not (0 <= row < height and 0 <= column < width):
        return False
    touched = circular_yaw_indices(
        start_yaw_index, stop_yaw_index, len(masks), direction=direction
    )
    return bool(np.all(masks[np.asarray(touched), row, column]))


def default_heading_steps(yaw_bins: int) -> IntArray:
    """Map mathematical yaw bins to 8-connected (row, column) moves.

    This convention assumes increasing columns are +x and increasing rows are
    +y. Simulator adapters should pass calibrated steps when their map axes
    differ.
    """
    angles = 2.0 * np.pi * np.arange(yaw_bins, dtype=np.float64) / yaw_bins
    columns = np.rint(np.cos(angles)).astype(np.int64)
    rows = np.rint(np.sin(angles)).astype(np.int64)
    steps = np.column_stack((rows, columns))
    if np.any(np.all(steps == 0, axis=1)):
        raise ValueError("yaw discretization produced a zero forward step")
    return steps


def _octile_distance(left: tuple[int, int], right: tuple[int, int]) -> float:
    row_distance = abs(left[0] - right[0])
    column_distance = abs(left[1] - right[1])
    diagonal = min(row_distance, column_distance)
    return float(diagonal * np.sqrt(2.0) + abs(row_distance - column_distance))


def _translation_is_safe(
    masks: BoolArray,
    state: SE2GridState,
    target_row: int,
    target_column: int,
) -> bool:
    _, height, width = masks.shape
    if not (0 <= target_row < height and 0 <= target_column < width):
        return False
    yaw = state.yaw_index
    if not masks[yaw, target_row, target_column]:
        return False
    row_delta = target_row - state.row
    column_delta = target_column - state.column
    if abs(row_delta) == 1 and abs(column_delta) == 1:
        # Prevent a diagonal primitive from cutting through a one-cell corner.
        return bool(
            masks[yaw, state.row + row_delta, state.column]
            and masks[yaw, state.row, state.column + column_delta]
        )
    return True


def sample_reachable_waypoint_cells(
    safe_masks: BoolArray,
    start_cell: Sequence[int],
    *,
    segment_count: int,
    minimum_length_m: float,
    maximum_length_m: float,
    map_resolution_m: float,
    maximum_turn_rad: float,
    rng: np.random.Generator,
    heading_steps: IntArray | None = None,
    forward_enabled_by_yaw: BoolArray | None = None,
    maximum_attempts: int = 48,
) -> IntArray | None:
    """Propose waypoint cells from collision-safe SE(2) motion primitives.

    Random XY endpoints from a 2-D connected component are often unreachable
    for a non-circular robot. This sampler traces valid forward primitives and
    permits a waypoint turn only when the complete in-place yaw sweep is safe.
    The returned controls are still replanned by the shortest-path planner.
    """
    masks = np.asarray(safe_masks, dtype=bool)
    if masks.ndim != 3 or len(masks) < 4:
        raise ValueError("safe_masks must have shape [B,H,W] with B >= 4")
    if segment_count < 1 or maximum_attempts < 1:
        raise ValueError("segment_count and maximum_attempts must be positive")
    if not (
        0.0 < minimum_length_m <= maximum_length_m
        and map_resolution_m > 0.0
        and 0.0 < maximum_turn_rad <= np.pi
    ):
        raise ValueError("invalid reachable-waypoint sampling limits")

    yaw_bins, height, width = masks.shape
    start = np.asarray(start_cell, dtype=np.int64)
    if start.shape != (2,) or not (
        0 <= start[0] < height and 0 <= start[1] < width
    ):
        return None
    steps = (
        default_heading_steps(yaw_bins)
        if heading_steps is None
        else np.asarray(heading_steps, dtype=np.int64)
    )
    if steps.shape != (yaw_bins, 2):
        raise ValueError("heading_steps must have shape [yaw_bins,2]")
    if forward_enabled_by_yaw is None:
        step_angles = np.arctan2(steps[:, 0], steps[:, 1])
        yaw_angles = 2.0 * np.pi * np.arange(yaw_bins) / yaw_bins
        forward_enabled = np.abs(
            (yaw_angles - step_angles + np.pi) % (2.0 * np.pi) - np.pi
        ) <= 1.0e-7
    else:
        forward_enabled = np.asarray(forward_enabled_by_yaw, dtype=bool)
    if forward_enabled.shape != (yaw_bins,):
        raise ValueError("forward_enabled_by_yaw must have shape [yaw_bins]")

    maximum_steps = max(1, int(np.ceil(maximum_length_m / map_resolution_m)))
    yaw_radians = 2.0 * np.pi / yaw_bins

    def rotation_safe(row: int, column: int, source: int, target: int) -> bool:
        left_steps = (source - target) % yaw_bins
        right_steps = (target - source) % yaw_bins
        delta_bins = min(left_steps, right_steps)
        if delta_bins == 0 or delta_bins * yaw_radians > maximum_turn_rad + 1.0e-9:
            return False
        directions = []
        if left_steps == delta_bins:
            directions.append(-1)
        if right_steps == delta_bins:
            directions.append(1)
        return any(
            swept_rotation_is_safe(
                masks, row, column, source, target, direction=direction
            )
            for direction in directions
        )

    for _ in range(maximum_attempts):
        desired_total = float(rng.uniform(minimum_length_m, maximum_length_m))
        desired_segments = (
            np.asarray([desired_total], dtype=np.float64)
            if segment_count == 1
            else desired_total * rng.dirichlet(np.full(segment_count, 2.0))
        )
        controls = [start.copy()]
        current = start.copy()
        current_yaw: int | None = None
        accumulated = 0.0
        feasible = True
        for segment_index, desired_length in enumerate(desired_segments):
            yaw_candidates = np.flatnonzero(
                forward_enabled & masks[:, current[0], current[1]]
            )
            if current_yaw is not None:
                yaw_candidates = np.asarray(
                    [
                        int(yaw)
                        for yaw in yaw_candidates
                        if rotation_safe(
                            int(current[0]),
                            int(current[1]),
                            current_yaw,
                            int(yaw),
                        )
                    ],
                    dtype=np.int64,
                )
            if not len(yaw_candidates):
                feasible = False
                break
            yaw_candidates = rng.permutation(yaw_candidates)
            chosen: tuple[int, np.ndarray, float] | None = None
            for yaw_value in yaw_candidates:
                yaw = int(yaw_value)
                state = SE2GridState(int(current[0]), int(current[1]), yaw)
                ray: list[np.ndarray] = []
                ray_lengths: list[float] = []
                distance = 0.0
                for _step in range(maximum_steps):
                    row_step, column_step = map(int, steps[yaw])
                    target_row = state.row + row_step
                    target_column = state.column + column_step
                    if not _translation_is_safe(
                        masks, state, target_row, target_column
                    ):
                        break
                    distance += float(np.hypot(row_step, column_step)) * map_resolution_m
                    state = SE2GridState(target_row, target_column, yaw)
                    ray.append(np.asarray([target_row, target_column], dtype=np.int64))
                    ray_lengths.append(distance)
                if not ray:
                    continue
                lengths = np.asarray(ray_lengths, dtype=np.float64)
                remaining_minimum = max(
                    0.0,
                    minimum_length_m
                    - accumulated
                    - float(np.sum(desired_segments[segment_index + 1 :])),
                )
                eligible = np.flatnonzero(
                    (lengths >= max(map_resolution_m, remaining_minimum))
                    & (accumulated + lengths <= maximum_length_m + 1.0e-9)
                )
                if not len(eligible):
                    continue
                errors = np.abs(lengths[eligible] - float(desired_length))
                best_error = float(np.min(errors))
                near = eligible[errors <= best_error + 0.5 * map_resolution_m]
                selected = int(rng.choice(near))
                chosen = (yaw, ray[selected], float(lengths[selected]))
                break
            if chosen is None:
                feasible = False
                break
            current_yaw, current, segment_length = chosen
            accumulated += segment_length
            controls.append(current.copy())
        if (
            feasible
            and minimum_length_m <= accumulated <= maximum_length_m
            and len(controls) == segment_count + 1
        ):
            return np.asarray(controls, dtype=np.int64)
    return None


def plan_se2_grid(
    safe_masks: BoolArray,
    start_cell: Sequence[int],
    goal_cell: Sequence[int],
    *,
    start_yaw_index: int | None = None,
    goal_yaw_index: int | None = None,
    heading_steps: IntArray | None = None,
    forward_enabled_by_yaw: BoolArray | None = None,
    rotation_cost_cells: float = 0.12,
    yaw_freedom_penalty: float = 0.05,
    maximum_expansions: int = 250_000,
) -> SE2GridPlan | None:
    """Plan forward and stop-and-turn primitives in (x, y, yaw_bin).

    Pose validity is decided solely by the exact orientation mask. Yaw freedom
    contributes only a small ranking penalty and can never invalidate a pose.
    """
    masks = np.asarray(safe_masks, dtype=bool)
    if masks.ndim != 3 or len(masks) < 4:
        raise ValueError("safe_masks must have shape [B,H,W] with B >= 4")
    yaw_bins, height, width = masks.shape
    start = tuple(map(int, start_cell))
    goal = tuple(map(int, goal_cell))
    if len(start) != 2 or len(goal) != 2:
        raise ValueError("start_cell and goal_cell must be (row, column)")
    if not (
        0 <= start[0] < height
        and 0 <= start[1] < width
        and 0 <= goal[0] < height
        and 0 <= goal[1] < width
    ):
        return None
    steps = default_heading_steps(yaw_bins) if heading_steps is None else np.asarray(
        heading_steps, dtype=np.int64
    )
    if steps.shape != (yaw_bins, 2):
        raise ValueError("heading_steps must have shape [yaw_bins,2]")
    if forward_enabled_by_yaw is None:
        step_angles = np.arctan2(steps[:, 0], steps[:, 1])
        yaw_angles = 2.0 * np.pi * np.arange(yaw_bins) / yaw_bins
        forward_enabled = np.abs(
            (yaw_angles - step_angles + np.pi) % (2.0 * np.pi) - np.pi
        ) <= 1.0e-7
    else:
        forward_enabled = np.asarray(forward_enabled_by_yaw, dtype=bool)
        if forward_enabled.shape != (yaw_bins,):
            raise ValueError("forward_enabled_by_yaw must have shape [yaw_bins]")
    if rotation_cost_cells <= 0.0 or maximum_expansions < 1:
        raise ValueError("rotation cost and expansion budget must be positive")

    allowed_starts = (
        [int(start_yaw_index) % yaw_bins]
        if start_yaw_index is not None
        else np.flatnonzero(masks[:, start[0], start[1]]).tolist()
    )
    allowed_starts = [
        yaw for yaw in allowed_starts if masks[yaw, start[0], start[1]]
    ]
    if not allowed_starts or not np.any(masks[:, goal[0], goal[1]]):
        return None

    serial = count()
    frontier: list[tuple[float, float, int, SE2GridState]] = []
    cost: dict[SE2GridState, float] = {}
    parent: dict[SE2GridState, tuple[SE2GridState, str]] = {}
    yaw_freedom = np.mean(masks, axis=0)
    for yaw in allowed_starts:
        state = SE2GridState(start[0], start[1], yaw)
        cost[state] = 0.0
        heapq.heappush(
            frontier,
            (_octile_distance(start, goal), 0.0, next(serial), state),
        )

    reached: SE2GridState | None = None
    expanded = 0
    while frontier and expanded < maximum_expansions:
        _, current_cost, _, state = heapq.heappop(frontier)
        if current_cost > cost.get(state, np.inf) + 1.0e-12:
            continue
        expanded += 1
        if (state.row, state.column) == goal and (
            goal_yaw_index is None
            or state.yaw_index == int(goal_yaw_index) % yaw_bins
        ):
            reached = state
            break

        neighbors: list[tuple[SE2GridState, str, float]] = []
        row_step, column_step = map(int, steps[state.yaw_index])
        target_row = state.row + row_step
        target_column = state.column + column_step
        if forward_enabled[state.yaw_index] and _translation_is_safe(
            masks, state, target_row, target_column
        ):
            distance = float(np.hypot(row_step, column_step))
            target = SE2GridState(target_row, target_column, state.yaw_index)
            rank_penalty = yaw_freedom_penalty * (
                1.0 - float(yaw_freedom[target_row, target_column])
            )
            neighbors.append((target, "forward", distance + rank_penalty))
        for direction, action in ((-1, "rotate_right"), (1, "rotate_left")):
            target_yaw = (state.yaw_index + direction) % yaw_bins
            if swept_rotation_is_safe(
                masks,
                state.row,
                state.column,
                state.yaw_index,
                target_yaw,
                direction=direction,
            ):
                target = SE2GridState(state.row, state.column, target_yaw)
                neighbors.append((target, action, rotation_cost_cells))
        for target, action, action_cost in neighbors:
            proposed = current_cost + action_cost
            if proposed + 1.0e-12 >= cost.get(target, np.inf):
                continue
            cost[target] = proposed
            parent[target] = (state, action)
            heuristic = _octile_distance((target.row, target.column), goal)
            heapq.heappush(
                frontier,
                (proposed + heuristic, proposed, next(serial), target),
            )
    if reached is None:
        return None

    states = [reached]
    actions: list[str] = []
    while states[-1] in parent:
        previous, action = parent[states[-1]]
        actions.append(action)
        states.append(previous)
    states.reverse()
    actions.reverse()
    translation = sum(
        float(np.hypot(right.row - left.row, right.column - left.column))
        for left, right, action in zip(states[:-1], states[1:], actions, strict=True)
        if action == "forward"
    )
    rotation = sum(action.startswith("rotate_") for action in actions) * (
        2.0 * np.pi / yaw_bins
    )
    return SE2GridPlan(
        states=tuple(states),
        actions=tuple(actions),
        translation_length_cells=float(translation),
        rotation_angle_rad=float(rotation),
        expanded_state_count=expanded,
    )


def plan_se2_waypoints(
    safe_masks: BoolArray,
    waypoint_cells: Sequence[Sequence[int]],
    **kwargs: object,
) -> SE2GridPlan | None:
    """Plan every waypoint segment while preserving the reached orientation."""
    cells = [tuple(map(int, cell)) for cell in waypoint_cells]
    if len(cells) < 2:
        raise ValueError("at least two waypoint cells are required")
    combined_states: list[SE2GridState] = []
    combined_actions: list[str] = []
    translation = 0.0
    rotation = 0.0
    expanded = 0
    current_yaw = kwargs.pop("start_yaw_index", None)
    for source, target in zip(cells[:-1], cells[1:], strict=True):
        segment = plan_se2_grid(
            safe_masks,
            source,
            target,
            start_yaw_index=current_yaw,
            **kwargs,
        )
        if segment is None:
            return None
        combined_states.extend(
            segment.states if not combined_states else segment.states[1:]
        )
        combined_actions.extend(segment.actions)
        translation += segment.translation_length_cells
        rotation += segment.rotation_angle_rad
        expanded += segment.expanded_state_count
        current_yaw = segment.states[-1].yaw_index
    return SE2GridPlan(
        states=tuple(combined_states),
        actions=tuple(combined_actions),
        translation_length_cells=translation,
        rotation_angle_rad=rotation,
        expanded_state_count=expanded,
    )


def se2_plan_is_safe(plan: SE2GridPlan, safe_masks: BoolArray) -> bool:
    """Revalidate every planned pose and every swept rotation."""
    masks = np.asarray(safe_masks, dtype=bool)
    for state in plan.states:
        if not masks[state.yaw_index, state.row, state.column]:
            return False
    for left, right, action in zip(
        plan.states[:-1], plan.states[1:], plan.actions, strict=True
    ):
        if action.startswith("rotate_"):
            direction = 1 if action == "rotate_left" else -1
            if not swept_rotation_is_safe(
                masks,
                left.row,
                left.column,
                left.yaw_index,
                right.yaw_index,
                direction=direction,
            ):
                return False
        elif action == "forward" and not _translation_is_safe(
            masks, left, right.row, right.column
        ):
            return False
    return True
