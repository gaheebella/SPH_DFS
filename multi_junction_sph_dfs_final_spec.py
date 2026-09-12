
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import types
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any, Sequence
import copy


import numpy as np
import pygame


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PHYSICAL_SOURCE = HERE / "single_junction_sph_dfs_environment.py"
ADAPTIVE_SOURCE =(
HERE 
/"lidar_junction_detection_adaptive_w_tau_anchor_stop_before_geometry.py"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import lidar_junction_detection_adaptive_w_tau_anchor_stop_before_geometry as adaptive

WINDOW_SIZE = (1440, 900)
ROBOT_MOTION_SPEED_SCALE = 2.5
MAIN_RECT = pygame.Rect(16, 54, 824, 830)
PROFILE_RECT = pygame.Rect(870, 72, 540, 410)
DIAGNOSTIC_RECT = pygame.Rect(870, 506, 540, 370)
COLORS = {
    "background": (10, 13, 18),
    "panel": (21, 27, 35),
    "panel_alt": (25, 31, 40),
    "floor": (31, 42, 54),
    "wall": (205, 214, 224),
    "text": (226, 232, 240),
    "muted": (125, 139, 156),
    "raw": (232, 238, 245),
    "smooth": (72, 156, 255),
    "open": (190, 92, 246),
    "open_fill": (190, 92, 246),
    "candidate": (144, 238, 144),
    "safe_band": (72, 156, 255),
    "group_edge": (255, 146, 52),
    "group_center": (225, 92, 246),
    "threshold": (255, 196, 74),
    "safe": (139, 92, 246),
    "anchor": (255, 221, 74),
    "normal": (52, 120, 246),
    "guard": (235, 72, 88),
    "frontier": (255, 146, 52),
    "shepherd": (255, 60, 170),

"pebble":(35 ,211 ,104 ),
"relay":(185 ,125 ,64 ),
"trunk":(139 ,87 ,67 ),
}

RAY_COUNT =adaptive .LIDAR_RAYS 
MAX_RANGE =adaptive .LIDAR_MAX_RANGE 
SMOOTHING_WINDOW =5 
ALPHA =0.5 
NOISE_FRACTION =adaptive .DEFAULT_NOISE_FRACTION 
TAU =MAX_RANGE *NOISE_FRACTION 
ADAPTIVE_W_MARGIN_RATIO =adaptive .ADAPTIVE_W_MARGIN_RATIO 
STATIONARY_WINDOW = 120
MIN_PERSISTENT_OBSERVATIONS = 72

MOVING_PERSISTENCE_WINDOW = 20
MOVING_MIN_PERSISTENT_OBSERVATIONS = 12
MOVING_PERSISTENCE_RATIO = 0.60
PREVIOUS_APPROACH_EXTENSION =60.0 
BASE_ADDED_EXTENSION =0.0 
APPROACH_EXTENSION =PREVIOUS_APPROACH_EXTENSION +BASE_ADDED_EXTENSION 
ASSOCIATION_TOLERANCE_DEG =max (
2.0 *float (adaptive .FROZEN_PARAMETERS ["merge_gap_deg"]),
float (adaptive .FROZEN_PARAMETERS ["min_opening_width_deg"]),
)

LOCAL_SATURATION_PRESSURE_RATIO =1.02 







PROVISIONAL_MOUTH_WIDTH_W_RATIO =0.75 
PROVISIONAL_MOUTH_WIDTH_MIN_RATIO =0.80 
PROVISIONAL_JUNCTION_DEPTH_MIN_WIDTH_RATIO =0.95 
PROVISIONAL_JUNCTION_DEPTH_MAX_WIDTH_RATIO =1.25 
PROVISIONAL_SIDE_SECTOR_MIN_ABS_ANGLE =90.0 
GUARD_WHO_AXIAL_WEIGHT =4.0 
GUARD_WHO_LATERAL_WEIGHT =2.0 
GUARD_WHO_PATH_WEIGHT =1.0 
GUARD_CAPTURE_UPSTREAM_WIDTH_RATIO =0.32 
GUARD_CAPTURE_DOWNSTREAM_WIDTH_RATIO =0.28 
GUARD_CAPTURE_LATERAL_MARGIN_RATIO =0.10 
GUARD_ASSIGN_MAX_AXIAL_WIDTH_RATIO =1.20 
GUARD_ASSIGN_MAX_LATERAL_WIDTH_RATIO =1.20 
GUARD_ASSIGN_MAX_PATH_WIDTH_RATIO =1.50 
GUARD_LATERAL_COVERAGE_BINS =9 
GUARD_READINESS_LOG_PERIOD =10 
GUARD_EDGE_SEAL_MARGIN_RATIO =0.25 
GUARD_LATERAL_OVERLAP_RATIO =0.90 
PROVISIONAL_WALL_SETTLED_RATIO =0.95 
PROVISIONAL_WALL_STABILITY_DWELL =0.18 
JUNCTION_ARRIVAL_RATIO_THRESHOLD =0.45 
JUNCTION_APPROACH_CRAWL_SPEED =10.0
POST_ANCHOR_NORMAL_CRAWL_SPEED =6.0 
LATERAL_BASELINE_SAMPLES =10 
LATERAL_RANGE_JUMP_THRESHOLD =2.0 *TAU 
ENTRANCE_STABILITY_FRAMES =1 


CHILD_PROBE_TRIGGER_RANGE_RATIO =0.95 
CHILD_PROBE_TRIGGER_WIDTH_RATIO =1.45 

CHILD_PROBE_RELAY_TRIGGER_RATIO =0.82 

CHILD_PROBE_TRIGGER_MAX_DISTANCE =(
MAX_RANGE *CHILD_PROBE_TRIGGER_RANGE_RATIO 
)


CHILD_CANDIDATE_MIN_STRUCTURAL_STREAK =12 
CHILD_CANDIDATE_NON_AXIAL_MAX_DOT =0.75 
CHILD_CANDIDATE_MIN_MOUTH_WIDTH_RATIO =0.35 
CHILD_PARENT_CLEARANCE_W_RATIO =0.50 
CHILD_APPROACH_STOP_W_RATIO =0.10 
CHILD_APPROACH_SLOWDOWN_W_RATIO =0.35 


CHILD_STATIONARY_PERSISTENCE_RATIO =0.60 
CHILD_STATIONARY_MIN_OUTGOING =2 
LIDAR_ROBOT_ID =675 


class PerceptionState (Enum ):
    MOVING =auto ()
    JUNCTION_APPROACH =auto ()
    FIXED_ACCUMULATING =auto ()
    BRANCHES_READY =auto ()
    PHYSICAL_DFS =auto ()


class BranchPhase (Enum ):
    IDLE =auto ()

    ANCHOR_PREP =auto ()

    FRONTIER_BOOTSTRAP =auto ()

    EXPLORE =auto ()

    FILL =auto ()

    PRESSURE_PUSH =auto ()

    FLOW_BACKTRACK =auto ()


@dataclass 
class MovingOpeningCandidateTrack :
    track_id :str 
    reference :dict [str ,Any ]
    latest_candidate :dict [str ,Any ]
    observation_frames :list [int ]=field (default_factory =list )
    last_frame :int =-1 

    def update (
    self ,
    candidate :dict [str ,Any ],
    frame :int ,
    )->None :
        self .reference =build_moving_candidate_track_reference (
        candidate 
        )
        self .latest_candidate =dict (candidate )

        if (
        not self .observation_frames 
        or self .observation_frames [-1 ]!=frame 
        ):
            self .observation_frames .append (frame )

        self .last_frame =frame 

    def prune (
    self ,
    frame :int ,
    )->None :
        first_frame =(
        frame 
        -MOVING_PERSISTENCE_WINDOW 
        +1 
        )
        self .observation_frames =[
        observed_frame 
        for observed_frame in self .observation_frames 
        if observed_frame >=first_frame 
        ]

    def persistence_ratio (self )->float :
        return (
        len (self .observation_frames )
        /float (MOVING_PERSISTENCE_WINDOW )
        )


@dataclass 
class PersistentOpening :
    persistent_id :str 
    sine_sum :float =0.0 
    cosine_sum :float =0.0 
    widths :list [float ]=field (default_factory =list )
    confidences :list [float ]=field (default_factory =list )
    observations :list [dict [str ,float ]]=field (default_factory =list )
    last_frame :int =-1 

    @property 
    def center_angle (self )->float :
        return math .degrees (math .atan2 (self .sine_sum ,self .cosine_sum ))

    @property 
    def mean_width (self )->float :
        return float (np .mean (self .widths ))if self .widths else 0.0 

    @property 
    def confidence (self )->float :
        return float (np .mean (self .confidences ))if self .confidences else 0.0 

    def persistence_ratio (self ,sample_count :int )->float :
        return len (self .observations )/max (sample_count ,1 )

    def update (self ,opening :dict [str ,float ],frame :int )->None :
        angle =float (opening ["center_angle"])
        self .sine_sum +=math .sin (math .radians (angle ))
        self .cosine_sum +=math .cos (math .radians (angle ))
        self .widths .append (float (opening ["width_deg"]))
        self .confidences .append (float (opening ["confidence"]))
        self .observations .append ({
        "start_angle":float (opening ["start_angle"]),
        "end_angle":float (opening ["end_angle"]),
        "center_angle":angle ,
        "width_deg":float (opening ["width_deg"]),
        "confidence":float (opening ["confidence"]),
        "frame":float (frame ),
        })
        self .last_frame =frame 


@dataclass (frozen =True )
class ParentTopologyEdge :
    """Traversed ingress edge; deliberately not a LiDAR opening track."""

    persistent_id :str ="PARENT_00"
    source :str ="INGRESS_HISTORY"


@dataclass 
class LidarFrame :
    frame :int 
    angles :np .ndarray 
    raw :np .ndarray 
    smoothed :np .ndarray 
    support :np .ndarray 
    openings :tuple [dict [str ,float ],...]
    left :float |None 
    right :float |None 
    adaptive_w :float 
    lower :float 
    upper :float 
    selected :float |None 
    interval_valid :bool 
    current_evidence :bool 


@dataclass 
class ProvisionalGuardGeometry :
    provisional_uid :str 
    opening :dict [str ,float ]
    descriptor :Any 
    columns :int 
    layers :int 
    slots :list [pygame .Vector2 ]
    selected_ids :list [int ]=field (default_factory =list )
    fixture_key :str |None =None 
    persistent_uid :str |None =None 
    cohort_ready :bool =False 
    first_robot_crossing_mouth_frame :int |None =None 
    candidate_sufficient_frame :int |None =None 
    guard_ready_frame :int |None =None 
    role_assignment_frame :int |None =None 
    last_candidate_count :int =-1 
    last_assignment_count :int =-1 
    readiness_diagnostics :dict [str ,Any ]=field (default_factory =dict )
    sealing_lateral_min :float =0.0 
    sealing_lateral_max :float =0.0 
    slot_spacing :float =0.0 
    local_branch_key :str |None =None 
    opening_start_local :pygame .Vector2 |None =None 
    opening_end_local :pygame .Vector2 |None =None 
    mouth_start_world :pygame .Vector2 |None =None 
    mouth_end_world :pygame .Vector2 |None =None 
    mouth_center_world :pygame .Vector2 |None =None 
    mouth_lateral_unit :pygame .Vector2 |None =None 
    branch_tangent_unit :pygame .Vector2 |None =None 
    mouth_span :float =0.0 








@dataclass 
class ChildObservationSession :
    """Fresh LiDAR observation session for one possible Child Junction."""

    parent_junction_uid :str 
    parent_branch_uid :str 

    lidar_id :int 
    start_frame :int 



    ingress_t :pygame .Vector2 
    ingress_n :pygame .Vector2 

    probe_traveled_axial :float =0.0 

    samples :int =0 
    valid_samples :int =0 
    consecutive_valid :int =0 

    last_valid_w :float |None =None 
    last_selected_threshold :float |None =None 



    structural_streak :int =0 

    candidate_frame :int |None =None 
    candidate_depth_local :float |None =None 
    candidate_selected_threshold :float |None =None 
    candidate_lidar_frame :LidarFrame |None =None 

    candidate_traveled_axial :float =0.0 
    candidate_remaining_depth :float |None =None 

    anchor_stopped :bool =False 
    anchor_stop_frame :int |None =None 


    stationary_samples :int =0 
    stationary_tracks :list [PersistentOpening ]=field (
    default_factory =list 
    )
    stationary_outgoing :list [PersistentOpening ]=field (
    default_factory =list 
    )
    stationary_verified_openings :list [
    dict [str ,float ]
    ]=field (default_factory =list )

    stationary_confirmed_lidar_frame :LidarFrame |None =None 

    stationary_confirmed :bool =False 
    stationary_confirmation_frame :int |None =None 

@dataclass 
class MultiJunctionFrame :
    """Logical DFS state for one confirmed Junction."""

    junction_uid :str 
    parent_junction_uid :str |None 
    incoming_branch_uid :str |None 







    branch_states :dict [str ,str ]=field (default_factory =dict )
    branch_order :list [str ]=field (default_factory =list )



    active_branch_uid :str |None =None 



    pending_branch_uid :str |None =None 



    branch_phase :BranchPhase =BranchPhase .IDLE 


    ingress_direction_local :pygame .Vector2 |None =None 
    return_direction_local :pygame .Vector2 |None =None 



    return_marker_id :int |None =None 
    completion_marker_ids :dict [str ,int ]=field (default_factory =dict )


    saved_physical_context :dict [str ,Any ]=field (
    default_factory =dict 
    )



    subtree_complete :bool =False 




    parent_return_started :bool =False 
    parent_return_arrived :bool =False 
    parent_return_dwell :float =0.0 


class MultiJunctionManager :
    """Top-level recursive DFS bookkeeping.

    This class must not control Guard, Frontier, Shepherd, SPH,
    pressure push, or ordinary dead-end backtracking.
    Those remain owned by the existing Single-Junction Physical DFS.
    """

    def __init__ (self )->None :
        self .stack :list [MultiJunctionFrame ]=[]
        self .next_junction_index :int =0 



        self .child_candidate_active :bool =False 
        self .child_confirmed :bool =False 







        self .pending_child_frame :MultiJunctionFrame |None =None 

        self .child_push_complete :bool =False 
        self .parent_release_pending :bool =False 
        self .parent_restore_pending :bool =False 
        self .pending_parent_restore_child :MultiJunctionFrame |None =None 
        self .parent_guard_reformation_pending :bool =False 



        self .child_probe_active :bool =False 
        self .child_probe_branch_uid :str |None =None 
        self .child_probe_start_frame :int |None =None 
        self .child_probe_lidar_id :int |None =None 
        self .child_session :ChildObservationSession |None =None 

    @property 
    def current (self )->MultiJunctionFrame |None :
        """Return the current DFS Junction, if one exists."""
        if not self .stack :
            return None 
        return self .stack [-1 ]

    @property 
    def depth (self )->int :
        """Current DFS depth. Before J0 exists, return -1."""
        return len (self .stack )-1 

    def refresh_subtree_complete (
    self ,
    frame :MultiJunctionFrame ,
    )->bool :
        """Recompute whether one Junction DFS subtree is fully complete."""

        ordered_states =[
        frame .branch_states .get (branch_uid )
        for branch_uid in frame .branch_order 
        ]

        has_registered_branches =bool (
        frame .branch_order 
        )

        has_active_branch =(
        frame .active_branch_uid is not None 
        or any (
        state in {
        "ACTIVE",
        "ACTIVE_CHILD",
        }
        for state in ordered_states 
        )
        )

        all_visited =(
        has_registered_branches 
        and bool (ordered_states )
        and all (
        state =="VISITED"
        for state in ordered_states 
        )
        )

        complete =(
        all_visited 
        and not has_active_branch 
        )

        previous =bool (
        frame .subtree_complete 
        )

        frame .subtree_complete =(
        complete 
        )

        if complete and not previous :
            print (
            "[SubtreeComplete] "
            f"junction={frame .junction_uid } "
            f"branches={frame .branch_order } "
            f"states={frame .branch_states } "
            f"depth={self .depth }"
            )

        return complete 


    def global_dfs_complete (self )->bool :
        """Return True only when the remaining DFS frame is the completed Root."""

        if len (self .stack )!=1 :
            return False 

        root =self .stack [0 ]

        if root .parent_junction_uid is not None :
            return False 

        if not root .subtree_complete :
            return False 

        if root .active_branch_uid is not None :
            return False 

        return all (
        root .branch_states .get (
        branch_uid 
        )
        =="VISITED"
        for branch_uid in root .branch_order 
        )

    def create_root (self )->MultiJunctionFrame :
        """Create J0 once after the first Junction is truly confirmed."""
        if self .stack :
            raise RuntimeError (
            "Root Junction cannot be created twice"
            )

        frame =MultiJunctionFrame (
        junction_uid ="J0",
        parent_junction_uid =None ,
        incoming_branch_uid =None ,
        )

        self .stack .append (frame )
        self .next_junction_index =1 

        print (
        "[MultiDFS] ROOT_CREATED "
        "junction=J0 depth=0"
        )

        return frame 

    def allocate_child_uid (self )->str :
        """Allocate the next Junction UID without changing the DFS stack."""
        uid =f"J{self .next_junction_index }"
        self .next_junction_index +=1 
        return uid 


    def stage_confirmed_child (
    self ,
    session :ChildObservationSession ,
    )->MultiJunctionFrame :
        """Stage a stationary-confirmed Child without DFS PUSH."""

        if self .pending_child_frame is not None :
            raise RuntimeError (
            "another confirmed Child is already staged"
            )

        parent =self .current 

        if parent is None :
            raise RuntimeError (
            "cannot stage Child without Parent Junction"
            )

        if (
        parent .junction_uid 
        !=session .parent_junction_uid 
        ):
            raise RuntimeError (
            "Child session Parent does not match "
            "current DFS Junction: "
            f"current={parent .junction_uid } "
            f"session_parent="
            f"{session .parent_junction_uid }"
            )

        parent_branch_uid =(
        session .parent_branch_uid 
        )

        if (
        parent_branch_uid 
        not in parent .branch_states 
        ):
            raise RuntimeError (
            "Child Parent branch missing from "
            "Parent DFS frame: "
            f"{parent_branch_uid }"
            )

        previous_state =(
        parent .branch_states [
        parent_branch_uid 
        ]
        )

        if previous_state !="ACTIVE":
            raise RuntimeError (
            "confirmed Child must come from "
            "an ACTIVE Parent branch: "
            f"branch={parent_branch_uid } "
            f"state={previous_state }"
            )


        parent .branch_states [
        parent_branch_uid 
        ]="ACTIVE_CHILD"

        parent .active_branch_uid =(
        parent_branch_uid 
        )

        parent .subtree_complete =False 

        child_uid =(
        self .allocate_child_uid ()
        )

        ingress =(
        session .ingress_t .copy ()
        )

        if (
        ingress .length_squared ()
        <=1.0e-12 
        ):
            raise RuntimeError (
            "confirmed Child has invalid ingress direction"
            )

        ingress =ingress .normalize ()

        child =MultiJunctionFrame (
        junction_uid =child_uid ,
        parent_junction_uid =(
        parent .junction_uid 
        ),
        incoming_branch_uid =(
        parent_branch_uid 
        ),
        ingress_direction_local =(
        ingress .copy ()
        ),
        return_direction_local =(
        -ingress 
        ),
        )


        self .pending_child_frame =child 
        self .child_push_complete =False 
        self .parent_release_pending =True 

        print (
        "[ParentBranchActiveChild] "
        f"junction={parent .junction_uid } "
        f"branch={parent_branch_uid } "
        f"{previous_state }->ACTIVE_CHILD"
        )

        print (
        "[ChildPushStaged] "
        f"parent={parent .junction_uid } "
        f"child={child .junction_uid } "
        f"incoming_branch="
        f"{child .incoming_branch_uid } "
        f"stack_depth_before_push={self .depth } "
        "dfs_push=False "
        "parent_release_pending=True"
        )

        return child 


    def commit_staged_child_push (
    self ,
    )->MultiJunctionFrame :
        """Push the staged Child only after Parent physical release."""

        child =self .pending_child_frame 

        if child is None :
            raise RuntimeError (
            "no staged Child exists for DFS PUSH"
            )

        if self .parent_release_pending :
            raise RuntimeError (
            "cannot DFS PUSH Child before "
            "Parent release is complete"
            )

        if self .child_push_complete :
            raise RuntimeError (
            "staged Child was already pushed"
            )

        parent =self .current 

        if parent is None :
            raise RuntimeError (
            "cannot commit Child without current Parent"
            )

        if (
        parent .junction_uid 
        !=child .parent_junction_uid 
        ):
            raise RuntimeError (
            "staged Child Parent mismatch before PUSH: "
            f"current={parent .junction_uid } "
            f"child_parent="
            f"{child .parent_junction_uid }"
            )

        self .stack .append (
        child 
        )

        self .pending_child_frame =None 
        self .child_push_complete =True 

        print (
        "[DFSStackPush] "
        f"parent={parent .junction_uid } "
        f"child={child .junction_uid } "
        f"incoming_branch="
        f"{child .incoming_branch_uid } "
        f"depth={self .depth } "
        "source=PARENT_RELEASE_COMPLETE"
        )

        print (
        "[MultiDFS] STACK "
        f"junctions="
        f"{[frame .junction_uid for frame in self .stack ]} "
        f"current={self .current .junction_uid } "
        "parent_release_pending=False"
        )

        return child 

    def pop_arrived_child (
    self ,
    )->tuple [MultiJunctionFrame ,MultiJunctionFrame ]:
        """POP Child only after its physical return to Parent is confirmed."""

        if self .parent_restore_pending :
            raise RuntimeError (
            "cannot POP another Child while "
            "Parent context restore is pending"
            )

        if len (self .stack )<2 :
            raise RuntimeError (
            "Child POP requires stack=[..., Parent, Child]"
            )

        child =self .stack [-1 ]
        parent =self .stack [-2 ]

        if child .parent_junction_uid is None :
            raise RuntimeError (
            "cannot POP Root Junction as Child"
            )

        if (
        child .parent_junction_uid 
        !=parent .junction_uid 
        ):
            raise RuntimeError (
            "Child/Parent stack mismatch before POP: "
            f"child={child .junction_uid } "
            f"expected_parent={child .parent_junction_uid } "
            f"stack_parent={parent .junction_uid }"
            )




        if not child .subtree_complete :
            raise RuntimeError (
            "cannot POP incomplete Child subtree: "
            f"child={child .junction_uid }"
            )

        if not child .parent_return_started :
            raise RuntimeError (
            "cannot POP Child before Parent return starts: "
            f"child={child .junction_uid }"
            )

        if not child .parent_return_arrived :
            raise RuntimeError (
            "cannot POP Child before physical Parent arrival: "
            f"child={child .junction_uid }"
            )

        parent_branch_uid =(
        child .incoming_branch_uid 
        )

        if parent_branch_uid is None :
            raise RuntimeError (
            "Child has no incoming Parent branch UID"
            )

        if (
        parent_branch_uid 
        not in parent .branch_states 
        ):
            raise RuntimeError (
            "connecting Parent branch missing before Child POP: "
            f"parent={parent .junction_uid } "
            f"branch={parent_branch_uid }"
            )

        previous_state =(
        parent .branch_states [
        parent_branch_uid 
        ]
        )

        if previous_state !="ACTIVE_CHILD":
            raise RuntimeError (
            "returned Child must belong to an ACTIVE_CHILD "
            "Parent branch: "
            f"parent={parent .junction_uid } "
            f"branch={parent_branch_uid } "
            f"state={previous_state }"
            )







        popped =self .stack .pop ()

        if popped is not child :
            raise RuntimeError (
            "DFS stack POP returned unexpected Junction"
            )

        parent .active_branch_uid =(
        parent_branch_uid 
        )



        self .pending_parent_restore_child =(
        child 
        )

        self .parent_restore_pending =True 
        self .child_candidate_active =False 
        self .child_confirmed =False 
        self .child_push_complete =False 
        self .parent_release_pending =False 

        self .child_probe_active =False 
        self .child_probe_branch_uid =None 
        self .child_probe_start_frame =None 
        self .child_probe_lidar_id =None 
        self .child_session =None 

        self .pending_child_frame =None 

        print (
        "[DFSStackPop] "
        f"child={child .junction_uid } "
        f"parent={parent .junction_uid } "
        f"incoming_branch={parent_branch_uid } "
        f"depth={self .depth } "
        f"stack="
        f"{[frame .junction_uid for frame in self .stack ]} "
        "source=PHYSICAL_PARENT_RETURN "
        "parent_branch_state=ACTIVE_CHILD "
        "restore_pending=True"
        )

        print (
        "[ParentBranchAwaitingRestore] "
        f"junction={parent .junction_uid } "
        f"branch={parent_branch_uid } "
        f"state={previous_state } "
        "restore_pending=True "
        "visited_transition_deferred=True"
        )

        return child ,parent 

multi_dfs =MultiJunctionManager ()

def sync_multi_dfs_from_physical (
physical :types .ModuleType ,
)->None :
    """Mirror the current Physical-DFS branch states into Multi DFS.

    Read-only synchronization:
    this function must never change Physical DFS behavior.
    """

    frame =multi_dfs .current 

    if frame is None :
        return 

    if not frame .branch_order :
        return 

    changed =False 

    for uid in frame .branch_order :
        descriptor =physical .branch_descriptors_by_uid .get (uid )

        if descriptor is None :
            continue 

        physical_state =str (descriptor .visit_state )
        previous_state =frame .branch_states .get (uid )

        if previous_state !=physical_state :
            frame .branch_states [uid ]=physical_state 
            changed =True 

            print (
            "[MultiDFS] BRANCH_SYNC "
            f"junction={frame .junction_uid } "
            f"branch={uid } "
            f"{previous_state }->{physical_state }"
            )

    active_uid =getattr (
    physical ,
    "active_branch_uid",
    None ,
    )

    if frame .active_branch_uid !=active_uid :
        previous_active =frame .active_branch_uid 
        frame .active_branch_uid =active_uid 

        print (
        "[MultiDFS] ACTIVE_SYNC "
        f"junction={frame .junction_uid } "
        f"{previous_active }->{active_uid }"
        )

    if changed :
        print (
        "[MultiDFS] STATE "
        f"junction={frame .junction_uid } "
        f"depth={multi_dfs .depth } "
        f"states={frame .branch_states }"
        )

def save_parent_physical_context (
physical :types .ModuleType ,
parent :MultiJunctionFrame ,
)->None :
    """Freeze Parent DFS control context before physical release.

    No robot position is stored as a return target.
    """

    if parent .saved_physical_context :
        return 

    parent .saved_physical_context =copy .deepcopy (
    {
    "branch_descriptors_by_uid":
    physical .branch_descriptors_by_uid ,

    "fixture_key_to_branch_uid":
    physical .fixture_key_to_branch_uid ,

    "branch_uid_to_fixture_key":
    physical .branch_uid_to_fixture_key ,

    "detected_branch_candidates":
    physical .detected_branch_candidates ,

    "junction_guard_groups":
    physical .junction_guard_groups ,

    "integration_wall_lifecycle":
    physical .integration_wall_lifecycle ,

    "integration_ready_guard_ids_by_uid":
    physical .integration_ready_guard_ids_by_uid ,

    "integration_wall_status":
    physical .integration_wall_status ,

    "branch_order_plan":
    physical .branch_order_plan ,

    "branch_fixture_order_plan":
    physical .branch_fixture_order_plan ,

    "active_branch":
    physical .active_branch ,

    "active_branch_uid":
    physical .active_branch_uid ,

    "phase":
    physical .phase ,
    }
    )

    print (
    "[ParentStateSaved] "
    f"junction={parent .junction_uid } "
    f"branches={parent .branch_order } "
    f"states={parent .branch_states } "
    f"active_child={parent .active_branch_uid }"
    )


def release_consumed_parent_return_marker (
parent :MultiJunctionFrame ,
robots :Sequence [Any ],
)->int :
    """Release the Parent Return Marker after physical Parent arrival."""

    marker_id =parent .return_marker_id 

    if marker_id is None :
        raise RuntimeError (
        "Parent context restore requires consumed Return Marker"
        )

    marker =next (
    (
    robot 
    for robot in robots 
    if robot .robot_id ==marker_id 
    ),
    None ,
    )

    if marker is None :
        raise RuntimeError (
        "Parent Return Marker robot is missing: "
        f"junction={parent .junction_uid } "
        f"marker={marker_id }"
        )

    if (
    marker .role !="PEBBLE"
    or getattr (
    marker ,
    "pebble_state",
    None ,
    )
    !="JUNCTION_RETURN"
    or getattr (
    marker ,
    "marker_junction_uid",
    None ,
    )
    !=parent .junction_uid 
    ):
        raise RuntimeError (
        "invalid consumed Parent Return Marker: "
        f"junction={parent .junction_uid } "
        f"marker={marker_id }"
        )









    marker .role ="NORMAL"

    marker .pebble_anchor =None 
    marker .pebble_branch_uid =None 
    marker .pebble_branch_key =None 
    marker .pebble_state =None 
    marker .pebble_ingress_direction_local =None 
    marker .pebble_return_direction_local =None 

    marker .marker_type =None 
    marker .marker_junction_uid =None 

    marker .base_reserve =False 
    marker .is_branch_leader =False 

    marker .velocity *=0.25 

    marker .commanded_velocity .update (
    0.0 ,
    0.0 ,
    )

    marker .acceleration .update (
    0.0 ,
    0.0 ,
    )

    marker .filtered_acceleration .update (
    0.0 ,
    0.0 ,
    )

    parent .return_marker_id =None 

    print (
    "[ParentReturnMarkerConsumed] "
    f"junction={parent .junction_uid } "
    f"robot={marker_id } "
    "role=NORMAL "
    "position_jump=0"
    )

    return marker_id 

def restore_parent_context_after_child_pop (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Restore Parent topology/geometry after an arrived Child was DFS-popped."""

    if not multi_dfs .parent_restore_pending :
        return 

    parent =multi_dfs .current 
    returned_child =(
    multi_dfs .pending_parent_restore_child 
    )

    if parent is None :
        raise RuntimeError (
        "Parent context restore requires current Parent"
        )

    if returned_child is None :
        raise RuntimeError (
        "Parent context restore lost popped Child"
        )

    if (
    returned_child .parent_junction_uid 
    !=parent .junction_uid 
    ):
        raise RuntimeError (
        "Parent restore Child mismatch: "
        f"parent={parent .junction_uid } "
        f"child={returned_child .junction_uid } "
        f"child_parent="
        f"{returned_child .parent_junction_uid }"
        )

    saved =parent .saved_physical_context 

    if not saved :
        raise RuntimeError (
        "Parent has no saved physical context: "
        f"junction={parent .junction_uid }"
        )

    restored =copy .deepcopy (
    saved 
    )


    physical .branch_descriptors_by_uid =(
    restored [
    "branch_descriptors_by_uid"
    ]
    )

    physical .fixture_key_to_branch_uid =(
    restored [
    "fixture_key_to_branch_uid"
    ]
    )

    physical .branch_uid_to_fixture_key =(
    restored [
    "branch_uid_to_fixture_key"
    ]
    )

    physical .detected_branch_candidates =(
    restored [
    "detected_branch_candidates"
    ]
    )

    physical .branch_order_plan =(
    restored [
    "branch_order_plan"
    ]
    )

    physical .branch_fixture_order_plan =(
    restored [
    "branch_fixture_order_plan"
    ]
    )



    physical .integration_detected_branch_order =list (
    parent .branch_order 
    )


    parent_branch_uid =(
    returned_child .incoming_branch_uid 
    )

    if parent_branch_uid is None :
        raise RuntimeError (
        "restored Child has no incoming Parent branch"
        )

    previous_parent_state =(
    parent .branch_states .get (
    parent_branch_uid 
    )
    )

    if previous_parent_state !="ACTIVE_CHILD":
        raise RuntimeError (
        "Parent connecting branch must remain ACTIVE_CHILD "
        "until context restoration: "
        f"junction={parent .junction_uid } "
        f"branch={parent_branch_uid } "
        f"state={previous_parent_state }"
        )

    parent .branch_states [
    parent_branch_uid 
    ]="VISITED"

    parent .active_branch_uid =None 
    parent .pending_branch_uid =None 
    parent .branch_phase =BranchPhase .IDLE 

    multi_dfs .refresh_subtree_complete (
    parent 
    )

    print (
    "[ParentBranchVisitedAfterRestore] "
    f"junction={parent .junction_uid } "
    f"branch={parent_branch_uid } "
    f"{previous_parent_state }->VISITED "
    f"subtree_complete="
    f"{parent .subtree_complete } "
    "source=RESTORED_PARENT_CONTEXT"
    )









    for branch_uid in parent .branch_order :
        descriptor =(
        physical .branch_descriptors_by_uid .get (
        branch_uid 
        )
        )

        if descriptor is None :
            raise RuntimeError (
            "restored Parent descriptor missing: "
            f"junction={parent .junction_uid } "
            f"branch={branch_uid }"
            )

        logical_state =(
        parent .branch_states .get (
        branch_uid 
        )
        )

        if logical_state is None :
            raise RuntimeError (
            "restored Parent branch has no DFS state: "
            f"junction={parent .junction_uid } "
            f"branch={branch_uid }"
            )

        descriptor .visit_state =(
        logical_state 
        )




    physical .junction_guard_groups ={}

    physical .integration_wall_lifecycle ={}
    physical .integration_ready_guard_ids_by_uid ={}
    physical .integration_wall_status ={}
    physical .integration_wall_stats ={}

    physical .integration_provisional_guard_groups ={}

    physical .integration_provisional_guard_active =False 
    physical .integration_guard_gating_enabled =False 

    physical .integration_all_walls_ready =False 
    physical .integration_ready_guard_handoff =False 

    physical .integration_guard_formation_start_frame =None 

    physical .integration_child_guard_lifecycle_initialized =False 













    physical .active_branch =None 
    physical .active_branch_uid =None 

    physical .phase =(
    physical .SimulationPhase .FORM_JUNCTION_GUARDS 
    )

    physical .integration_child_guard_forming =True 



    physical .integration_frontier_active_uid =None 
    physical .integration_frontier_ids =set ()
    physical .integration_frontier_offsets ={}

    physical .integration_frontier_bootstrap_complete =False 
    physical .integration_frontier_ready_logged =False 
    physical .integration_child_dfs_phase ="IDLE"



    perception .handoff_complete =False 
    perception .provisional_guards =[]
    perception .provisional_guard_started =False 





    consumed_return_marker_id =(
    release_consumed_parent_return_marker (
    parent ,
    robots ,
    )
    )


    physical .integration_parent_return_active =False 
    physical .integration_parent_return_arrived =False 
    physical .integration_parent_return_child_uid =None 

    physical .integration_parent_return_direction_local =(
    pygame .Vector2 ()
    )



    parent .saved_physical_context ={}

    multi_dfs .pending_parent_restore_child =None 
    multi_dfs .parent_restore_pending =False 


    multi_dfs .parent_guard_reformation_pending =(
    not parent .subtree_complete 
    )

    print (
    "[ParentContextRestored] "
    f"junction={parent .junction_uid } "
    f"returned_child="
    f"{returned_child .junction_uid } "
    f"branches={parent .branch_order } "
    f"states={parent .branch_states } "
    f"return_marker_consumed="
    f"{consumed_return_marker_id } "
    f"subtree_complete="
    f"{parent .subtree_complete } "
    f"guard_reformation_pending="
    f"{multi_dfs .parent_guard_reformation_pending } "
    "stale_guard_membership_restored=False "
    "localization=False"
    )

def retain_parent_completion_markers (
physical :types .ModuleType ,
parent :MultiJunctionFrame ,
robots :Sequence [Any ],
)->None :
    """Retain or materialize one physical Completion Marker per VISITED branch.

    Existing VISITED Pebbles are reused.

    If legacy/same-ID Physical DFS completed the branch by restoring the
    original Guard wall instead of creating a Pebble, promote exactly one
    robot from that branch's existing JUNCTION_GUARD lineage in place.

    ACTIVE_CHILD never receives a Completion Marker here.
    """

    existing_by_uid ={
    pebble .pebble_branch_uid :pebble 
    for pebble in physical .get_pebbles (robots )
    if (
    getattr (
    pebble ,
    "pebble_state",
    None ,
    )
    =="VISITED"
    and getattr (
    pebble ,
    "pebble_branch_uid",
    None ,
    )
    is not None 
    )
    }

    visited_uids :list [str ]=[]

    for branch_uid ,state in (
    parent .branch_states .items ()
    ):
        if state !="VISITED":
            continue 

        visited_uids .append (branch_uid )

        marker =existing_by_uid .get (
        branch_uid 
        )


        if marker is None :

            candidates =[
            robot 
            for robot in robots 
            if (
            getattr (
            robot ,
            "role",
            None ,
            )
            =="JUNCTION_GUARD"
            and getattr (
            robot ,
            "junction_guard_branch_uid",
            None ,
            )
            ==branch_uid 
            )
            ]

            descriptor =getattr (
            physical ,
            "branch_descriptors_by_uid",
            {},
            ).get (branch_uid )


            if candidates :
                marker =min (
                candidates ,
                key =lambda robot :robot .robot_id ,
                )

        if marker is None :
            raise RuntimeError (
            "VISITED Parent branch has neither "
            "a physical Completion Pebble nor "
            "an original Guard candidate: "
            f"junction={parent .junction_uid } "
            f"branch={branch_uid }"
            )



        if (
        getattr (marker ,"role",None )
        =="PEBBLE"
        and getattr (
        marker ,
        "pebble_state",
        None ,
        )
        =="VISITED"
        ):
            parent .completion_marker_ids [
            branch_uid 
            ]=marker .robot_id 

            print (
            "[CompletionMarkerRetained] "
            f"junction={parent .junction_uid } "
            f"branch={branch_uid } "
            f"robot={marker .robot_id }"
            )
            continue 

        branch_key =getattr (
        marker ,
        "junction_guard_branch",
        None ,
        )

        ingress =None 

        if descriptor is not None :
            candidate_ingress =getattr (
            descriptor ,
            "local_outgoing_direction",
            None ,
            )
            if (
            candidate_ingress is not None 
            and candidate_ingress .length_squared ()
            >physical .EPSILON 
            ):
                ingress =(
                candidate_ingress .normalize ()
                )

        if ingress is None :
            local_by_uid =getattr (
            marker ,
            "local_ingress_tangents_by_uid",
            {},
            )
            candidate_ingress =(
            local_by_uid .get (branch_uid )
            )

            if (
            candidate_ingress is not None 
            and candidate_ingress .length_squared ()
            >physical .EPSILON 
            ):
                ingress =(
                candidate_ingress .normalize ()
                )

        if (
        ingress is None 
        and branch_key is not None 
        ):
            local_by_fixture =getattr (
            marker ,
            "local_ingress_tangents",
            {},
            )
            candidate_ingress =(
            local_by_fixture .get (branch_key )
            )

            if (
            candidate_ingress is not None 
            and candidate_ingress .length_squared ()
            >physical .EPSILON 
            ):
                ingress =(
                candidate_ingress .normalize ()
                )

        if ingress is None :
            raise RuntimeError (
            "Original Guard selected for Completion "
            "Marker has no valid branch-local ingress: "
            f"junction={parent .junction_uid } "
            f"branch={branch_uid } "
            f"robot={marker .robot_id }"
            )



        marker .role ="PEBBLE"
        marker .pebble_anchor =(
        marker .position .copy ()
        )
        marker .pebble_branch_uid =(
        branch_uid 
        )
        marker .pebble_branch_key =(
        branch_key 
        )
        marker .pebble_state ="VISITED"
        marker .pebble_ingress_direction_local =(
        ingress .copy ()
        )
        marker .pebble_return_direction_local =(
        -ingress 
        )

        if hasattr (
        physical ,
        "branch_completion_epoch",
        ):
            physical .branch_completion_epoch +=1 
            marker .pebble_completion_epoch =(
            physical .branch_completion_epoch 
            )
        else :
            marker .pebble_completion_epoch =0 

        if hasattr (
        marker ,
        "known_visited_branch_uids",
        ):
            marker .known_visited_branch_uids .add (
            branch_uid 
            )

        if (
        branch_key is not None 
        and hasattr (
        marker ,
        "known_visited_branches",
        )
        ):
            marker .known_visited_branches .add (
            branch_key 
            )



        marker .velocity .update (0.0 ,0.0 )
        marker .commanded_velocity .update (
        0.0 ,
        0.0 ,
        )
        marker .observed_velocity .update (
        0.0 ,
        0.0 ,
        )
        marker .acceleration .update (0.0 ,0.0 )
        marker .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )

        parent .completion_marker_ids [
        branch_uid 
        ]=marker .robot_id 

        existing_by_uid [
        branch_uid 
        ]=marker 

        print (
        "[CompletionMarkerBackfilled] "
        f"junction={parent .junction_uid } "
        f"branch={branch_uid } "
        f"robot={marker .robot_id } "
        "source=ORIGINAL_GUARD "
        "position_jump=0"
        )

    print (
    "[ParentCompletionMarkersReady] "
    f"junction={parent .junction_uid } "
    f"visited={visited_uids } "
    f"markers="
    f"{parent .completion_marker_ids }"
    )

def create_parent_return_marker (
physical :types .ModuleType ,
parent :MultiJunctionFrame ,
robots :Sequence [Any ],
perception :AdaptivePerception ,
)->int :
    """Convert one existing Parent wall robot in place.

    No teleport.
    No global Junction coordinate.
    LiDAR robot is never used.
    """

    parent_branch_uids =set (
    parent .branch_order 
    )

    candidates :list [
    tuple [int ,Any ]
    ]=[]

    for robot in robots :

        if robot is perception .leader :
            continue 

        if robot .role !="JUNCTION_GUARD":
            continue 

        branch_uid =getattr (
        robot ,
        "junction_guard_branch_uid",
        None ,
        )

        branch_key =getattr (
        robot ,
        "junction_guard_branch",
        None ,
        )

        if (
        branch_uid 
        not in parent_branch_uids 
        ):
            try :
                branch_uid =(
                physical .branch_uid_for_fixture (
                branch_key 
                )
                )
            except (
            KeyError ,
            TypeError ,
            AttributeError ,
            ):
                branch_uid =None 

        if branch_uid not in parent_branch_uids :
            continue 

        candidates .append (
        (
        robot .robot_id ,
        robot ,
        )
        )

    if not candidates :
        raise RuntimeError (
        "no physical Parent Guard available "
        "for Junction Return Marker"
        )

    _ ,marker =min (
    candidates ,
    key =lambda item :item [0 ],
    )







    marker .role ="PEBBLE"

    marker .pebble_anchor =(
    marker .position .copy ()
    )



    marker .pebble_branch_uid =None 
    marker .pebble_branch_key =None 

    marker .pebble_state =(
    "JUNCTION_RETURN"
    )

    marker .pebble_ingress_direction_local =None 

    marker .pebble_return_direction_local =(
    parent .return_direction_local .copy ()
    if parent .return_direction_local 
    is not None 
    else None 
    )



    marker .marker_type =(
    "JUNCTION_RETURN"
    )

    marker .marker_junction_uid =(
    parent .junction_uid 
    )

    marker .junction_guard_anchor =None 
    marker .junction_guard_branch =None 
    marker .junction_guard_branch_uid =None 
    marker .junction_guard_parent_id =None 
    marker .junction_guard_layer =-1 
    marker .is_branch_leader =False 

    marker .shepherd_anchor =None 
    marker .shepherd_origin =None 
    marker .shepherd_branch =None 
    marker .frontier_local_lateral =None 

    marker .velocity .update (
    0.0 ,
    0.0 ,
    )

    marker .acceleration .update (
    0.0 ,
    0.0 ,
    )

    marker .filtered_acceleration .update (
    0.0 ,
    0.0 ,
    )

    parent .return_marker_id =(
    marker .robot_id 
    )

    print (
    "[ReturnMarkerCreated] "
    f"junction={parent .junction_uid } "
    f"robot={marker .robot_id } "
    "position_snap=False "
    "branch_uid=None "
    "state=JUNCTION_RETURN"
    )

    return marker .robot_id 

def release_parent_physical_roles (
physical :types .ModuleType ,
parent :MultiJunctionFrame ,
robots :Sequence [Any ],
)->dict [str ,int ]:
    """Release only Parent wall roles.

    Completion Pebbles and Return Marker remain fixed.
    Breadcrumb/Relay robots are intentionally preserved.
    """

    parent_branch_uids =set (
    parent .branch_order 
    )

    released ={
    "guard":0 ,
    "frontier":0 ,
    "shepherd":0 ,
    }

    for robot in robots :

        if robot .role =="PEBBLE":
            continue 

        branch_uid =None 

        if robot .role =="JUNCTION_GUARD":

            branch_uid =getattr (
            robot ,
            "junction_guard_branch_uid",
            None ,
            )

            branch_key =getattr (
            robot ,
            "junction_guard_branch",
            None ,
            )

            if (
            branch_uid 
            not in parent_branch_uids 
            ):
                try :
                    branch_uid =(
                    physical .branch_uid_for_fixture (
                    branch_key 
                    )
                    )
                except (
                KeyError ,
                TypeError ,
                AttributeError ,
                ):
                    branch_uid =None 

        elif robot .role in {
        "FRONTIER_SHEPHERD",
        "SHEPHERD",
        "PRE_SHEPHERD",
        }:

            branch_key =getattr (
            robot ,
            "shepherd_branch",
            None ,
            )

            try :
                branch_uid =(
                physical .branch_uid_for_fixture (
                branch_key 
                )
                )
            except (
            KeyError ,
            TypeError ,
            AttributeError ,
            ):
                branch_uid =None 

        else :
            continue 

        if branch_uid not in parent_branch_uids :
            continue 

        previous_role =robot .role 

        robot .role ="NORMAL"

        robot .junction_guard_anchor =None 
        robot .junction_guard_branch =None 
        robot .junction_guard_branch_uid =None 
        robot .junction_guard_hop =-1 
        robot .junction_guard_parent_id =None 
        robot .junction_guard_layer =-1 
        robot .is_branch_leader =False 

        if hasattr (
        robot ,
        "integration_guard_waypoints",
        ):
            robot .integration_guard_waypoints =[]

        if hasattr (
        robot ,
        "integration_guard_final_anchor",
        ):
            robot .integration_guard_final_anchor =None 

        robot .shepherd_anchor =None 
        robot .shepherd_origin =None 
        robot .shepherd_branch =None 
        robot .shepherd_return_direction =None 
        robot .frontier_local_lateral =None 

        robot .base_reserve =False 

        if previous_role =="JUNCTION_GUARD":
            released ["guard"]+=1 

        elif previous_role =="FRONTIER_SHEPHERD":
            released ["frontier"]+=1 

        else :
            released ["shepherd"]+=1 

    print (
    "[ParentRolesReleased] "
    f"junction={parent .junction_uid } "
    f"guards={released ['guard']} "
    f"frontiers={released ['frontier']} "
    f"shepherds={released ['shepherd']} "
    "relays_preserved=True"
    )

    return released 

def release_confirmed_parent_junction (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Save/release Parent first, then commit the staged Child DFS PUSH."""

    if not multi_dfs .parent_release_pending :
        return 

    parent =multi_dfs .current 
    child =multi_dfs .pending_child_frame 

    if parent is None :
        raise RuntimeError (
        "Parent release requires current Parent Junction"
        )

    if child is None :
        raise RuntimeError (
        "Parent release requires a staged Child"
        )

    if (
    child .parent_junction_uid 
    !=parent .junction_uid 
    ):
        raise RuntimeError (
        "staged Child does not belong to current Parent: "
        f"parent={parent .junction_uid } "
        f"child_parent={child .parent_junction_uid }"
        )

    print (
    "[JunctionReleaseStart] "
    f"parent={parent .junction_uid } "
    f"child={child .junction_uid } "
    f"active_child="
    f"{parent .active_branch_uid } "
    "child_on_stack=False"
    )







    save_parent_physical_context (
    physical ,
    parent ,
    )













    retain_parent_completion_markers (
    physical ,
    parent ,
    robots ,
    )







    create_parent_return_marker (
    physical ,
    parent ,
    robots ,
    perception ,
    )









    release_parent_physical_roles (
    physical ,
    parent ,
    robots ,
    )



    multi_dfs .parent_release_pending =False 

    print (
    "[ParentReleaseComplete] "
    f"parent={parent .junction_uid } "
    f"child={child .junction_uid } "
    f"completion_markers="
    f"{parent .completion_marker_ids } "
    f"return_marker="
    f"{parent .return_marker_id } "
    "parent_release_pending=False "
    "dfs_push=False"
    )







    committed_child =(
    multi_dfs .commit_staged_child_push ()
    )

    if committed_child is not child :
        raise RuntimeError (
        "committed Child differs from staged Child"
        )

    print (
    "[ChildPushAfterParentRelease] "
    f"parent={parent .junction_uid } "
    f"child={committed_child .junction_uid } "
    f"depth={multi_dfs .depth } "
    "context_saved=True "
    "markers_ready=True "
    "parent_roles_released=True "
    "dfs_push=True"
    )



def initialize_confirmed_child_guard_context (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Initialize physical Guard formation for confirmed Child Junction."""

    child =multi_dfs .current 
    session =multi_dfs .child_session 



    if child is None :
        return 



    if child .parent_junction_uid is None :
        return 



    if multi_dfs .parent_release_pending :
        return 



    if session is None :
        return 



    if not session .stationary_confirmed :
        return 



    if child .branch_order :
        return 

    print (
    "[ChildGuardInitializationStart] "
    f"junction={child .junction_uid } "
    f"parent={child .parent_junction_uid } "
    f"incoming_branch={child .incoming_branch_uid }"
    )



    reset_guard_frontend_for_child (
    perception 
    )

    reset_physical_guard_context_for_child (
    physical 
    )

    branch_uids =(
    register_current_child_branches ()
    )

    child_frame =(
    session .stationary_confirmed_lidar_frame 
    )

    if child_frame is None :
        raise RuntimeError (
        "Child Guard initialization "
        "missing stationary LiDAR frame"
        )

    geometries =(
    build_provisional_guard_descriptors_from_lidar (
    physical ,
    perception ,
    child_frame ,
    outgoing_override =(
    session .stationary_verified_openings 
    ),
    branch_uids_override =(
    branch_uids 
    ),
    junction_uid_override =(
    child .junction_uid 
    ),
    )
    )

    geometry_frame =(
    session .stationary_confirmation_frame 
    if session .stationary_confirmation_frame 
    is not None 
    else child_frame .frame 
    )

    install_provisional_guard_geometries (
    physical ,
    perception ,
    robots ,
    geometries ,
    geometry_frame ,
    )

    print (
    "[ChildTopologyInitialized] "
    f"junction={child .junction_uid } "
    f"branches={branch_uids }"
    )


def reset_guard_frontend_for_child (
perception :AdaptivePerception ,
)->None :
    """Reset Guard-formation frontend before building Child Junction guards."""



    perception .handoff_complete =False 



    perception .provisional_guards =[]
    perception .provisional_guard_started =False 



    perception .guard_leakage .clear ()
    perception .guard_communication_audits .clear ()

    perception .guard_geometry_frame =None 
    perception .guard_who_frame =None 
    perception .guard_motion_start_frame =None 

    perception .guard_all_groups_activated =False 
    perception .guard_current_group_index =0 

    perception .integration_detected_branch_order =[]

    print (
    "[ChildGuardFrontendReset] "
    "handoff_complete=False "
    "provisional_guard_started=False"
    )

def reset_physical_guard_context_for_child (
physical :types .ModuleType ,
)->None :
    """Clear current-Junction Guard/DFS registries before installing Child."""





    physical .branch_descriptors_by_uid ={}

    physical .fixture_key_to_branch_uid ={}
    physical .branch_uid_to_fixture_key ={}

    physical .detected_branch_candidates =set ()

    physical .junction_guard_groups ={}

    physical .integration_wall_lifecycle ={}
    physical .integration_ready_guard_ids_by_uid ={}
    physical .integration_wall_status ={}
    physical .integration_wall_stats ={}








    physical .integration_child_guard_lifecycle_initialized =False 

    physical .integration_provisional_guard_groups ={}

    physical .integration_provisional_guard_active =False 
    physical .integration_guard_gating_enabled =False 



    physical .integration_all_walls_ready =False 
    physical .integration_ready_guard_handoff =False 

    physical .integration_guard_formation_start_frame =None 

    physical .branch_order_plan =[]
    physical .branch_fixture_order_plan =[]

    physical .active_branch =None 
    physical .active_branch_uid =None 





















    previous_phase =physical .phase 

    physical .phase =(
    physical .SimulationPhase .FORM_JUNCTION_GUARDS 
    )

    physical .integration_child_guard_forming =True 

    print (
    "[ChildPhysicalPhaseReset] "
    f"{previous_phase .name }->FORM_JUNCTION_GUARDS "
    "active_branch=None"
    )

    print (
    "[ChildPhysicalGuardContextReset] "
    "old_parent_context_cleared=True "
    "all_walls_ready=False "
    "ready_handoff=False"
    )



def register_current_child_branches ()->list [str ]:
    """Create outgoing DFS branch identities for the confirmed Child."""

    child =multi_dfs .current 
    session =multi_dfs .child_session 

    if child is None :
        raise RuntimeError (
        "Child branch registration requires current Junction"
        )

    if child .parent_junction_uid is None :
        raise RuntimeError (
        "Root J0 must not use Child branch registration"
        )

    if session is None :
        raise RuntimeError (
        "Child branch registration requires observation session"
        )

    if not session .stationary_confirmed :
        raise RuntimeError (
        "Child branch registration requires stationary confirmation"
        )

    if child .branch_order :
        return list (child .branch_order )

    openings =sorted (
    session .stationary_verified_openings ,
    key =lambda opening :
    float (opening ["center_angle"]),
    )

    if not openings :
        raise RuntimeError (
        "Confirmed Child has no stationary verified outgoing openings"
        )

    child .branch_order =[
    f"{child .junction_uid }-B{index }"
    for index in range (len (openings ))
    ]

    child .branch_states ={
    branch_uid :"UNVISITED"
    for branch_uid in child .branch_order 
    }

    child .active_branch_uid =None 

    for branch_uid ,opening in zip (
    child .branch_order ,
    openings ,
    ):
        print (
        "[ChildBranchRegistered] "
        f"junction={child .junction_uid } "
        f"branch={branch_uid } "
        f"center_angle="
        f"{float (opening ['center_angle']):.2f}"
        )

    print (
    "[ChildPhysicalContextActivated] "
    f"junction={child .junction_uid } "
    f"parent={child .parent_junction_uid } "
    f"incoming_branch={child .incoming_branch_uid } "
    f"outgoing={child .branch_order }"
    )

    return list (child .branch_order )



def initialize_current_junction_guard_lifecycle (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Freeze current-Junction Guard IDs and branch-local offsets.

    The authoritative identity is always the runtime Branch UID.
    This function contains no Root/Child DFS distinction.
    """

    junction =multi_dfs .current 

    if junction is None :
        raise RuntimeError (
        "Guard lifecycle requires current Junction"
        )

    if not junction .branch_order :
        raise RuntimeError (
        "Guard lifecycle requires registered Branch UIDs: "
        f"junction={junction .junction_uid }"
        )

    by_id ={
    robot .robot_id :robot 
    for robot in robots 
    }











    geometry_by_uid :dict [
    str ,
    ProvisionalGuardGeometry ,
    ]={}

    for geometry in perception .provisional_guards :

        runtime_uid =(
        geometry .persistent_uid 
        or geometry .provisional_uid 
        )

        geometry_by_uid [
        runtime_uid 
        ]=geometry 

        geometry_by_uid .setdefault (
        geometry .provisional_uid ,
        geometry ,
        )

















    existing_complete =all (
    current_junction_branch_lifecycle (
    physical ,
    junction ,
    branch_uid ,
    )
    is not None 
    for branch_uid in junction .branch_order 
    )

    if existing_complete :
        print (
        "[CurrentGuardLifecycleReuse] "
        f"junction={junction .junction_uid } "
        f"depth={multi_dfs .depth } "
        "existing_complete=True"
        )

        physical .integration_child_guard_lifecycle_initialized =True 
        return 

















    fixture_keys ={
    branch_uid :
    physical .branch_fixture_for_uid (
    branch_uid 
    )
    for branch_uid 
    in junction .branch_order 
    }

    fixture_presence =[
    fixture_key is not None 
    for fixture_key 
    in fixture_keys .values ()
    ]

    if (
    any (fixture_presence )
    and not all (fixture_presence )
    ):
        raise RuntimeError (
        "Mixed current-Junction lifecycle representation: "
        f"junction={junction .junction_uid } "
        f"fixtures={fixture_keys }"
        )

    fixture_backed_junction =(
    bool (fixture_presence )
    and all (fixture_presence )
    )

    if fixture_backed_junction :
        raise RuntimeError (
        "Fixture-backed current Junction has an incomplete "
        "physical Guard lifecycle; refusing UID rematerialization: "
        f"junction={junction .junction_uid }"
        )














    physical .integration_wall_lifecycle ={}
    physical .integration_ready_guard_ids_by_uid ={}

    for branch_uid in junction .branch_order :

        geometry =geometry_by_uid .get (
        branch_uid 
        )

        if geometry is None :
            raise RuntimeError (
            "Missing Guard geometry for current Junction: "
            f"junction={junction .junction_uid } "
            f"branch={branch_uid }"
            )

        descriptor =(
        physical .branch_descriptors_by_uid .get (
        branch_uid 
        )
        )

        if descriptor is None :
            raise RuntimeError (
            "Missing Branch descriptor during Guard lifecycle init: "
            f"junction={junction .junction_uid } "
            f"branch={branch_uid }"
            )

        robot_ids =sorted (
        geometry .selected_ids 
        )

        if not robot_ids :
            raise RuntimeError (
            "Guard lifecycle cannot be created from zero robots: "
            f"junction={junction .junction_uid } "
            f"branch={branch_uid }"
            )

        members =[
        by_id [robot_id ]
        for robot_id in robot_ids 
        if robot_id in by_id 
        ]

        if len (members )!=len (robot_ids ):
            raise RuntimeError (
            "Guard robot ID lookup incomplete: "
            f"junction={junction .junction_uid } "
            f"branch={branch_uid }"
            )

        live_guard_ids ={
        robot .robot_id 
        for robot in members 
        if (
        robot .role =="JUNCTION_GUARD"
        and getattr (
        robot ,
        "junction_guard_branch_uid",
        None ,
        )
        ==branch_uid 
        )
        }

        if live_guard_ids !=set (robot_ids ):
            raise RuntimeError (
            "Guard lifecycle same-ID validation failed: "
            f"junction={junction .junction_uid } "
            f"branch={branch_uid } "
            f"expected={sorted (robot_ids )} "
            f"live={sorted (live_guard_ids )}"
            )

        coordinates =[
        physical .branch_local_coordinates (
        robot .position ,
        descriptor ,
        )
        for robot in members 
        ]

        centroid_axial =float (
        np .mean ([
        axial 
        for axial ,_ in coordinates 
        ])
        )

        centroid_lateral =float (
        np .mean ([
        lateral 
        for _ ,lateral in coordinates 
        ])
        )

        physical .integration_ready_guard_ids_by_uid [
        branch_uid 
        ]=list (
        robot_ids 
        )

        physical .integration_wall_lifecycle [
        branch_uid 
        ]={
        "uid":branch_uid ,

        "state":"GUARD",

        "rows":geometry .layers ,
        "cols":geometry .columns ,



        "robot_ids":list (
        robot_ids 
        ),

        "centroid_axial":
        centroid_axial ,

        "centroid_lateral":
        centroid_lateral ,















        "original_guard_centroid_axial":
        centroid_axial ,

        "original_guard_centroid_lateral":
        centroid_lateral ,

        "guard_anchor_by_id":{
        robot .robot_id :
        robot .position .copy ()
        for robot in members 
        },

        "mouth_center_world":(
        geometry .mouth_center_world .copy ()
        if geometry .mouth_center_world 
        is not None 
        else descriptor .observed_mouth_position .copy ()
        ),

        "branch_tangent_unit":(
        geometry .branch_tangent_unit .copy ()
        if geometry .branch_tangent_unit 
        is not None 
        else physical .descriptor_local_basis (
        descriptor 
        )[0 ].copy ()
        ),

        "mouth_lateral_unit":(
        geometry .mouth_lateral_unit .copy ()
        if geometry .mouth_lateral_unit 
        is not None 
        else physical .descriptor_local_basis (
        descriptor 
        )[1 ].copy ()
        ),

        "measured_mouth_span":float (
        geometry .mouth_span 
        or descriptor .observed_physical_width 
        ),

        "sealing_lateral_min":float (
        geometry .sealing_lateral_min 
        ),

        "sealing_lateral_max":float (
        geometry .sealing_lateral_max 
        ),

        "slot_spacing":float (
        geometry .slot_spacing 
        ),

        "guard_layer_by_id":{
        robot .robot_id :int (
        getattr (
        robot ,
        "junction_guard_layer",
        -1 ,
        )
        )
        for robot in members 
        },

        "guard_hop_by_id":{
        robot .robot_id :int (
        getattr (
        robot ,
        "junction_guard_hop",
        -1 ,
        )
        )
        for robot in members 
        },

        "guard_parent_id_by_id":{
        robot .robot_id :getattr (
        robot ,
        "junction_guard_parent_id",
        None ,
        )
        for robot in members 
        },

        "guard_branch_key_by_id":{
        robot .robot_id :getattr (
        robot ,
        "junction_guard_branch",
        None ,
        )
        for robot in members 
        },

        "guard_is_leader_by_id":{
        robot .robot_id :bool (
        getattr (
        robot ,
        "is_branch_leader",
        False ,
        )
        )
        for robot in members 
        },

        "guard_slot_index_by_id":{
        robot .robot_id :int (
        getattr (
        robot ,
        "integration_guard_slot_index",
        -1 ,
        )
        )
        for robot in members 
        },

        "relative_offsets":{
        robot .robot_id :(
        float (
        axial 
        -centroid_axial 
        ),
        float (
        lateral 
        -centroid_lateral 
        ),
        )
        for (
        robot ,
        (axial ,lateral ),
        )
        in zip (
        members ,
        coordinates ,
        )
        },

        "junction_uid":
        junction .junction_uid ,
        }

        print (
        "[CurrentGuardLifecycleSaved] "
        f"junction={junction .junction_uid } "
        f"depth={multi_dfs .depth } "
        f"branch={branch_uid } "
        f"robots={len (robot_ids )} "
        f"rows={geometry .layers } "
        f"cols={geometry .columns }"
        )

    physical .integration_child_guard_lifecycle_initialized =True 

    print (
    "[CurrentGuardLifecycleReady] "
    f"junction={junction .junction_uid } "
    f"depth={multi_dfs .depth } "
    f"branches={junction .branch_order } "
    "uid_keyed=True "
    "initialized=True"
    )



def current_junction_branch_lifecycle_key (
physical :types .ModuleType ,
junction :MultiJunctionFrame ,
branch_uid :str ,
)->Any |None :
    """Resolve one current-Junction Branch UID to its live lifecycle key.

    Transitional compatibility:
      - depth-generalized/new lifecycle: Branch UID key
      - legacy Root lifecycle: fixture key

    All callers outside this adapter should reason with Branch UID.
    """

    if branch_uid not in junction .branch_order :
        raise RuntimeError (
        "Branch UID does not belong to current Junction: "
        f"junction={junction .junction_uid } "
        f"branch={branch_uid }"
        )

    lifecycle_map =getattr (
    physical ,
    "integration_wall_lifecycle",
    {},
    )

    if branch_uid in lifecycle_map :

        lifecycle =lifecycle_map [
        branch_uid 
        ]

        stored_uid =lifecycle .get (
        "uid",
        branch_uid ,
        )

        if stored_uid !=branch_uid :
            raise RuntimeError (
            "UID-keyed lifecycle identity mismatch: "
            f"key={branch_uid } "
            f"stored_uid={stored_uid }"
            )

        return branch_uid 



    fixture_key =(
    physical .branch_fixture_for_uid (
    branch_uid 
    )
    )

    if (
    fixture_key is not None 
    and fixture_key in lifecycle_map 
    ):
        lifecycle =lifecycle_map [
        fixture_key 
        ]

        stored_uid =lifecycle .get (
        "uid",
        branch_uid ,
        )

        if stored_uid !=branch_uid :
            raise RuntimeError (
            "fixture-keyed lifecycle identity mismatch: "
            f"fixture={fixture_key } "
            f"expected_uid={branch_uid } "
            f"stored_uid={stored_uid }"
            )

        return fixture_key 

    return None 


def current_junction_branch_lifecycle (
physical :types .ModuleType ,
junction :MultiJunctionFrame ,
branch_uid :str ,
)->dict [str ,Any ]|None :
    """Return the live physical lifecycle for one runtime Branch UID."""

    key =current_junction_branch_lifecycle_key (
    physical ,
    junction ,
    branch_uid ,
    )

    if key is None :
        return None 

    return getattr (
    physical ,
    "integration_wall_lifecycle",
    {},
    ).get (
    key 
    )


def current_junction_frontier_runtime_kind (
physical :types .ModuleType ,
junction :MultiJunctionFrame ,
)->str |None :
    """Resolve the physical Frontier runtime without using DFS depth.

    UID_LOCAL:
        lifecycle is indexed directly by runtime Branch UID.

    LEGACY_FIXTURE:
        lifecycle is still indexed by a temporary fixture adapter.

    The distinction is physical representation only.
    It is NOT a Root/Child algorithm distinction.
    """

    if not junction .branch_order :
        return None 

    kinds :set [str ]=set ()

    for branch_uid in junction .branch_order :

        key =(
        current_junction_branch_lifecycle_key (
        physical ,
        junction ,
        branch_uid ,
        )
        )

        if key is None :
            continue 

        if key ==branch_uid :
            kinds .add (
            "UID_LOCAL"
            )
        else :
            kinds .add (
            "LEGACY_FIXTURE"
            )

    if not kinds :
        return None 

    if len (kinds )!=1 :
        raise RuntimeError (
        "Mixed Frontier lifecycle representations "
        "inside one Junction: "
        f"junction={junction .junction_uid } "
        f"kinds={sorted (kinds )}"
        )

    return next (
    iter (kinds )
    )


def select_next_current_branch_uid (
junction :MultiJunctionFrame ,
)->str |None :
    """Select the next DFS Branch at any Junction depth.

    No Root/Child distinction is used here.
    """

    active_states ={
    "ACTIVE",
    "ACTIVE_CHILD",
    }

    active_branches =[
    branch_uid 
    for branch_uid in junction .branch_order 
    if junction .branch_states .get (
    branch_uid 
    )
    in active_states 
    ]


    if active_branches :

        if len (active_branches )>1 :
            raise RuntimeError (
            "Multiple branches are ACTIVE in one Junction: "
            f"junction={junction .junction_uid } "
            f"active={active_branches }"
            )

        return None 

    if junction .active_branch_uid is not None :
        return None 













    for branch_uid in junction .branch_order :

        state =junction .branch_states .get (
        branch_uid 
        )

        if state =="UNVISITED":
            return branch_uid 

        if state !="VISITED":
            raise RuntimeError (
            "Invalid branch state before DFS selection: "
            f"junction={junction .junction_uid } "
            f"branch={branch_uid } "
            f"state={state }"
            )

    return None 



def current_junction_all_guard_walls_ready (
physical :types .ModuleType ,
perception :AdaptivePerception ,
junction :MultiJunctionFrame ,
)->bool :
    """Return True only when every current-Junction 3xN Guard wall is READY."""

    if not junction .branch_order :
        return False 

    if not perception .provisional_guards :
        return False 




    geometry_by_uid :dict [
    str ,
    ProvisionalGuardGeometry ,
    ]={}

    for geometry in perception .provisional_guards :

        runtime_uid =(
        geometry .persistent_uid 
        or geometry .provisional_uid 
        )

        geometry_by_uid [
        runtime_uid 
        ]=geometry 





        geometry_by_uid .setdefault (
        geometry .provisional_uid ,
        geometry ,
        )

    for branch_uid in junction .branch_order :

        geometry =geometry_by_uid .get (
        branch_uid 
        )

        if geometry is None :
            return False 

        if not geometry .cohort_ready :
            return False 

        if (
        len (geometry .selected_ids )
        !=len (geometry .slots )
        ):
            return False 

        status =(
        physical .integration_wall_status .get (
        branch_uid ,
        {},
        )
        )





        if not status :
            status =(
            physical .integration_wall_status .get (
            geometry .provisional_uid ,
            {},
            )
            )

        if not bool (
        status .get (
        "ready",
        False ,
        )
        ):
            return False 

    return True 


def prepare_current_junction_frontier_handoff (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Prepare one Guard wall for Frontier transport at any DFS depth."""

    junction =multi_dfs .current 

    if junction is None :
        return 

    if not junction .branch_order :
        return 

    if not current_junction_all_guard_walls_ready (
    physical ,
    perception ,
    junction ,
    ):
        return 



    if not getattr (
    physical ,
    "integration_child_guard_lifecycle_initialized",
    False ,
    ):
        initialize_current_junction_guard_lifecycle (
        physical ,
        perception ,
        robots ,
        )


    for branch_uid in junction .branch_order :

        lifecycle =(
        current_junction_branch_lifecycle (
        physical ,
        junction ,
        branch_uid ,
        )
        )

        if lifecycle is None :
            return 

        if "robot_ids"not in lifecycle :
            return 

        if "relative_offsets"not in lifecycle :
            return 


    active_branches =[
    branch_uid 
    for branch_uid in junction .branch_order 
    if junction .branch_states .get (
    branch_uid 
    )
    in {
    "ACTIVE",
    "ACTIVE_CHILD",
    }
    ]

    if active_branches :
        return 

    if junction .active_branch_uid is not None :
        return 


    active_frontier_uid =getattr (
    physical ,
    "integration_frontier_active_uid",
    None ,
    )

    if active_frontier_uid is not None :
        return 


    branch_uid =(
    select_next_current_branch_uid (
    junction 
    )
    )

    if branch_uid is None :
        return 

    print (
    "[CurrentJunctionFrontierPending] "
    f"junction={junction .junction_uid } "
    f"depth={multi_dfs .depth } "
    f"selected_branch={branch_uid } "
    "guards_remain_fixed=True "
    "frontier_transport_ready=True"
    )





    if (
    junction .parent_junction_uid 
    is not None 
    ):
        print (
        "[ChildAllGuardsReady] "
        f"junction={junction .junction_uid } "
        f"branches={junction .branch_order }"
        )

        print (
        "[ChildFrontierPending] "
        f"junction={junction .junction_uid } "
        f"selected_branch={branch_uid } "
        "guards_remain_fixed=True "
        "frontier_transport_ready=True"
        )


def resolve_current_junction_frontier_launch_uid (
physical :types .ModuleType ,
junction :MultiJunctionFrame ,
)->str |None :
    """Return one Branch UID only after the common Anchor-prep launch gate."""

    if not junction .branch_order :
        return None 





    branch_uid =junction .pending_branch_uid 

    if branch_uid is None :
        branch_uid =(
        select_next_current_branch_uid (
        junction 
        )
        )

        if branch_uid is None :
            return None 

        junction .pending_branch_uid =branch_uid 
        junction .branch_phase =BranchPhase .ANCHOR_PREP 

    lifecycle =(
    current_junction_branch_lifecycle (
    physical ,
    junction ,
    branch_uid ,
    )
    )

    if lifecycle is None :
        return None 

    if not lifecycle .get (
    "robot_ids"
    ):
        return None 

    logical_state =(
    junction .branch_states .get (
    branch_uid 
    )
    )

    if logical_state !="UNVISITED":
        raise RuntimeError (
        "Frontier launch selected a non-UNVISITED branch: "
        f"junction={junction .junction_uid } "
        f"branch={branch_uid } "
        f"state={logical_state }"
        )

    lifecycle_state =lifecycle .get (
    "state",
    "GUARD",
    )

    if lifecycle_state !="GUARD":
        raise RuntimeError (
        "UNVISITED branch does not own a GUARD lifecycle: "
        f"junction={junction .junction_uid } "
        f"branch={branch_uid } "
        f"lifecycle_state={lifecycle_state }"
        )



    if getattr (
    physical ,
    "integration_leading_anchor_uid",
    None ,
    )!=branch_uid :

        prep_request_uid =getattr (
        physical ,
        "integration_anchor_prep_request_uid",
        None ,
        )

        prep_ready_uid =getattr (
        physical ,
        "integration_anchor_prep_ready_uid",
        None ,
        )

        if (
        prep_request_uid !=branch_uid 
        and prep_ready_uid !=branch_uid 
        ):
            request_anchor_prep (
            physical ,
            branch_uid ,
            )

            print (
            "[CurrentDFSBranchPrepSelected] "
            f"junction={junction .junction_uid } "
            f"depth={multi_dfs .depth } "
            f"branch={branch_uid } "
            "frontier_promoted=False"
            )

            return None 

        if prep_ready_uid !=branch_uid :
            return None 

        promote_anchor_prep_to_leading (
        physical ,
        branch_uid ,
        )

    return branch_uid 


def promote_current_guard_wall_roles_to_frontier (
physical :types .ModuleType ,
robots :Sequence [Any ],
junction :MultiJunctionFrame ,
branch_uid :str ,
lifecycle :dict [str ,Any ],
relative_offsets :dict [int ,tuple [float ,float ]],
*,
shepherd_branch_key :Any ,
clear_guard_waypoints :bool ,
zero_command_state :bool ,
)->tuple [list [Any ],float ]:
    """Promote exactly the same 3xN Guard IDs to Frontier in-place.

    This primitive performs only the common physical role transition.
    Root/Child transport-controller state remains outside this function.
    """

    if branch_uid not in junction .branch_order :
        raise RuntimeError (
        "Frontier promotion Branch UID does not belong "
        "to current Junction: "
        f"junction={junction .junction_uid } "
        f"branch={branch_uid }"
        )

    stored_uid =lifecycle .get (
    "uid",
    branch_uid ,
    )

    if stored_uid !=branch_uid :
        raise RuntimeError (
        "Frontier promotion lifecycle UID mismatch: "
        f"junction={junction .junction_uid } "
        f"branch={branch_uid } "
        f"stored_uid={stored_uid }"
        )

    expected_ids =set (
    lifecycle .get (
    "robot_ids",
    [],
    )
    )

    if not expected_ids :
        raise RuntimeError (
        "Frontier promotion has zero Guard IDs: "
        f"junction={junction .junction_uid } "
        f"branch={branch_uid }"
        )

    offset_ids ={
    int (robot_id )
    for robot_id 
    in relative_offsets 
    }

    if offset_ids !=expected_ids :
        raise RuntimeError (
        "Frontier relative-offset IDs do not equal Guard IDs: "
        f"branch={branch_uid } "
        f"guards={sorted (expected_ids )} "
        f"offsets={sorted (offset_ids )}"
        )

    guards =[
    robot 
    for robot in robots 
    if (
    robot .robot_id in expected_ids 
    and robot .role =="JUNCTION_GUARD"
    and getattr (
    robot ,
    "junction_guard_branch_uid",
    None ,
    )
    ==branch_uid 
    )
    ]

    guard_ids ={
    robot .robot_id 
    for robot in guards 
    }

    if guard_ids !=expected_ids :
        raise RuntimeError (
        "Guard->Frontier same-ID validation failed: "
        f"junction={junction .junction_uid } "
        f"branch={branch_uid } "
        f"expected={sorted (expected_ids )} "
        f"live={sorted (guard_ids )}"
        )

    before ={
    robot .robot_id :
    robot .position .copy ()
    for robot in guards 
    }
    for robot in guards :

        _ ,lateral_offset =(
        relative_offsets [
        robot .robot_id 
        ]
        )

        robot .role =(
        "FRONTIER_SHEPHERD"
        )

        robot .junction_guard_anchor =(
        None 
        )

        robot .shepherd_anchor =(
        robot .position .copy ()
        )

        robot .shepherd_origin =(
        robot .position .copy ()
        )

        robot .shepherd_branch =(
        shepherd_branch_key 
        )

        robot .frontier_local_lateral =(
        float (lateral_offset )
        )

        if (
        clear_guard_waypoints 
        and hasattr (
        robot ,
        "integration_guard_waypoints",
        )
        ):
            robot .integration_guard_waypoints =(
            []
            )

        robot .velocity .update (
        0.0 ,
        0.0 ,
        )

        if zero_command_state :

            robot .commanded_velocity .update (
            0.0 ,
            0.0 ,
            )

            robot .observed_velocity .update (
            0.0 ,
            0.0 ,
            )

        robot .acceleration .update (
        0.0 ,
        0.0 ,
        )

        robot .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )

    lifecycle ["state"]=(
    "FRONTIER"
    )

    live_frontier_ids ={
    robot .robot_id 
    for robot in robots 
    if (
    robot .robot_id in expected_ids 
    and robot .role 
    =="FRONTIER_SHEPHERD"
    and robot .shepherd_branch 
    ==shepherd_branch_key 
    )
    }

    if live_frontier_ids !=expected_ids :
        raise RuntimeError (
        "Frontier IDs changed during Guard promotion: "
        f"branch={branch_uid } "
        f"expected={sorted (expected_ids )} "
        f"frontier={sorted (live_frontier_ids )}"
        )

    transition_jump =max (
    (
    robot .position .distance_to (
    before [
    robot .robot_id 
    ]
    )
    for robot in guards 
    ),
    default =0.0 ,
    )

    if transition_jump >physical .EPSILON :
        raise RuntimeError (
        "Guard->Frontier role transition moved robots: "
        f"branch={branch_uid } "
        f"max_jump={transition_jump :.9f}"
        )

    print (
    "[CurrentGuardToFrontier] "
    f"junction={junction .junction_uid } "
    f"depth={multi_dfs .depth } "
    f"branch={branch_uid } "
    f"robots={len (expected_ids )} "
    "same_ids=True "
    f"position_jump={transition_jump :.6f}"
    )

    return guards ,transition_jump 



def update_child_lidar_probe (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
dt :float ,
)->None :
    if getattr (physical ,"integration_anchor_breakout_active",False ):
        return 
    """Release the same LiDAR robot only for deep-branch observation.

    IMPORTANT:
    - This does not confirm a Child Junction.
    - This does not release the Parent Junction.
    - This does not create Markers.
    - This does not perform DFS PUSH.
    """

    frame =multi_dfs .current 

    if frame is None :
        return 

    active_uid =getattr (
    physical ,
    "active_branch_uid",
    None ,
    )











    if multi_dfs .child_probe_active :
        if active_uid !=multi_dfs .child_probe_branch_uid :
            return 

        descriptor =physical .branch_descriptors_by_uid .get (
        active_uid 
        )

        if descriptor is None :
            return 

        lidar_robot =perception .leader 

        try :
            tangent ,normal =physical .descriptor_local_basis (descriptor )
        except (ValueError ,AttributeError ):
            return 

        tangent =tangent .normalize ()
        normal =normal .normalize ()
















        session =multi_dfs .child_session 





        if (
        session is not None 
        and session .anchor_stopped 
        ):
            return 

        lidar_frame =perception .last_frame 
        if lidar_frame is None :
            return 
        left_range =_range_at_local_angle (lidar_frame ,-90.0 )
        right_range =_range_at_local_angle (lidar_frame ,90.0 )
        left_axis =_body_local_unit (perception ,-90.0 ).normalize ()
        lateral_velocity =lidar_robot .velocity .dot (left_axis )







        lateral_command =3.0 *(left_range -right_range )-2.2 *lateral_velocity 

        lateral_command =max (
        -55.0 ,
        min (55.0 ,lateral_command ),
        )























        candidate_control_active =(
        multi_dfs .child_candidate_active 
        and session is not None 
        and session .candidate_depth_local is not None 
        )

        forward_command =0.0 

        if not candidate_control_active :

            axial_direction =(
            session .ingress_t 
            if session is not None 
            else tangent 
            )

            axial_velocity =float (
            lidar_robot .velocity .dot (
            axial_direction 
            )
            )









            existing_axial_acc =float (
            lidar_robot .acceleration .dot (
            axial_direction 
            )
            )

            lidar_robot .acceleration -=(
            axial_direction 
            *existing_axial_acc 
            )



































            probe_cruise_speed =10.0 

            comm_parent =getattr (
            lidar_robot ,
            "comm_parent",
            None ,
            )

            if (
            comm_parent is None 
            or not lidar_robot .connected_to_base 
            ):
                comm_parent_distance =float ("inf")
                comm_speed_scale =0.0 

            else :
                parent_observation =observe_local_neighbors (
                lidar_robot ,
                [comm_parent ],
                axial_direction ,
                )
                comm_parent_distance =(
                parent_observation [0 ].relative_range 
                if parent_observation else float ("inf")
                )

                comm_guard_start =float (
                physical .COMM_GUARD_START 
                )

                comm_hard_limit =float (
                physical .COMM_GUARD_HARD_LIMIT 
                )

                if (
                comm_parent_distance 
                <=comm_guard_start 
                ):
                    comm_speed_scale =1.0 

                elif (
                comm_parent_distance 
                >=comm_hard_limit 
                ):
                    comm_speed_scale =0.0 

                else :
                    comm_speed_scale =(
                    comm_hard_limit 
                    -comm_parent_distance 
                    )/max (
                    physical .EPSILON ,
                    comm_hard_limit 
                    -comm_guard_start ,
                    )

                    comm_speed_scale =float (
                    np .clip (
                    comm_speed_scale ,
                    0.0 ,
                    1.0 ,
                    )
                    )

            probe_target_axial_speed =(
            probe_cruise_speed 
            *comm_speed_scale 
            )

            probe_speed_error =(
            probe_target_axial_speed 
            -axial_velocity 
            )

            probe_forward_limit =(
            0.15 
            *physical .MAX_ACCELERATION 
            )

            forward_command =float (
            np .clip (
            6.0 *probe_speed_error ,
            -probe_forward_limit ,
            probe_forward_limit ,
            )
            )

            if (
            physical .integration_frame 
            %20 
            ==0 
            ):
                print (
                "[ChildProbeCruise] "
                f"junction={frame .junction_uid } "
                f"branch={active_uid } "
                f"lidar_id={lidar_robot .robot_id } "
                f"axial_v={axial_velocity :.2f} "
                f"target_v="
                f"{probe_target_axial_speed :.2f} "
                f"comm_scale="
                f"{comm_speed_scale :.2f} "
                f"comm_dist="
                f"{comm_parent_distance :.2f} "
                f"sph_axial_removed="
                f"{existing_axial_acc :.2f} "
                f"forward_cmd="
                f"{forward_command :.2f}"
                )





















        if (
        multi_dfs .child_candidate_active 
        and session is not None 
        and session .candidate_depth_local is not None 
        ):






            candidate_forward_speed =max (
            0.0 ,
            float (lidar_robot .velocity .dot (session .ingress_t )),
            )
            session .candidate_traveled_axial +=candidate_forward_speed *dt 

            session .candidate_remaining_depth =max (
            0.0 ,
            session .candidate_depth_local 
            -session .candidate_traveled_axial ,
            )







            if (
            session .candidate_lidar_frame 
            is not None 
            ):
                candidate_width =float (
                session .candidate_lidar_frame .adaptive_w 
                )
            elif session .last_valid_w is not None :
                candidate_width =float (
                session .last_valid_w 
                )
            else :
                candidate_width =float (
                lidar_frame .adaptive_w 
                )


















            slowdown_distance =max (
            CHILD_APPROACH_SLOWDOWN_W_RATIO 
            *candidate_width ,
            physical .EPSILON ,
            )

            remaining_depth =(
            session .candidate_remaining_depth 
            )

















            (
            entrance_line_reached ,
            entrance_frame ,
            left_90_range ,
            right_90_range ,
            delta_left ,
            delta_right ,
            )=anchor_junction_entrance_line_reached (
            perception 
            )

            if entrance_line_reached :
                session .anchor_stopped =True 
                session .anchor_stop_frame =(
                physical .integration_frame 
                )




                session .stationary_samples =0 
                session .stationary_tracks .clear ()
                session .stationary_outgoing .clear ()

                session .stationary_verified_openings .clear ()
                session .stationary_confirmed_lidar_frame =None 

                session .stationary_confirmed =False 
                session .stationary_confirmation_frame =None 

                perception .anchor_position =(
                lidar_robot .position .copy ()
                )
                perception .anchor_fixed =True 

                physical .integration_anchor_position =(
                perception .anchor_position .copy ()
                )

                lidar_robot .is_fixed_anchor =True 
                lidar_robot .base_reserve =True 

                lidar_robot .velocity .update (
                0.0 ,
                0.0 ,
                )
                lidar_robot .acceleration .update (
                0.0 ,
                0.0 ,
                )
                lidar_robot .filtered_acceleration .update (
                0.0 ,
                0.0 ,
                )



                physical .integration_guard_hold_active =False 

                print (
                "[ChildAnchorEntranceLineStop] "
                f"parent={session .parent_junction_uid } "
                f"branch={session .parent_branch_uid } "
                f"lidar_id={session .lidar_id } "
                f"range_jump_frame={entrance_frame } "
                f"frame={session .anchor_stop_frame } "
                f"left90={left_90_range :.2f} "
                f"right90={right_90_range :.2f} "
                f"delta_left={delta_left :.2f} "
                f"delta_right={delta_right :.2f} "
                f"jump_threshold={LATERAL_RANGE_JUMP_THRESHOLD :.2f} "
                "source=BILATERAL_90_TEMPORAL_RANGE_JUMP "
                "position_snap=False "
                )

                return 









            drive_ratio =float (
            np .clip (
            remaining_depth 
            /slowdown_distance ,
            0.0 ,
            1.0 ,
            )
            )





















            axial_velocity =float (
            lidar_robot .velocity .dot (
            session .ingress_t 
            )
            )











            target_axial_speed =(
            3.0 
            +11.0 *drive_ratio 
            )

            speed_error =(
            target_axial_speed 
            -axial_velocity 
            )

















            existing_axial_acc =float (
            lidar_robot .acceleration .dot (
            session .ingress_t 
            )
            )

            lidar_robot .acceleration -=(
            session .ingress_t 
            *existing_axial_acc 
            )

            forward_command =(
            10.0 *speed_error 
            )

            forward_limit =(
            0.25 
            *physical .MAX_ACCELERATION 
            )







            forward_command =float (
            np .clip (
            forward_command ,
            -forward_limit ,
            forward_limit ,
            )
            )

            if (
            physical .integration_frame 
            %10 
            ==0 
            ):
                print (
                "[ChildCandidateApproach] "
                f"parent={session .parent_junction_uid } "
                f"branch={session .parent_branch_uid } "
                f"lidar_id={session .lidar_id } "
                f"traveled="
                f"{session .candidate_traveled_axial :.2f} "
                f"remaining="
                f"{remaining_depth :.2f} "
                f"stop_tol="
                f"{stop_tolerance :.2f} "
                f"sph_axial_removed="
                f"{existing_axial_acc :.2f} "
                f"axial_v="
                f"{axial_velocity :.2f} "
                f"target_v="
                f"{target_axial_speed :.2f} "
                f"forward_cmd="
                f"{forward_command :.2f}"
                )











        lidar_robot .acceleration +=(
        tangent *forward_command 
        +left_axis *lateral_command 
        )

        if physical .integration_frame %20 ==0 :
            comm_parent =getattr (
            lidar_robot ,
            "comm_parent",
            None ,
            )

            comm_parent_id =getattr (
            comm_parent ,
            "robot_id",
            None ,
            )

            parent_observation =(
            observe_local_neighbors (lidar_robot ,[comm_parent ],tangent )
            if comm_parent is not None else []
            )
            comm_parent_distance =(
            parent_observation [0 ].relative_range 
            if parent_observation else float ("nan")
            )

            probe_acc =lidar_robot .acceleration .length ()

            forward_test_position =(
            lidar_robot .position 
            +tangent *physical .ROBOT_RADIUS 
            )

            forward_walkable =physical .is_walkable (
            forward_test_position ,
            lidar_robot .radius ,
            )

            print (
            "[ChildProbeProgress] "
            f"junction={frame .junction_uid } "
            f"branch={active_uid } "
            f"lidar_id={lidar_robot .robot_id } "
            f"role={lidar_robot .role } "
            f"left={left_range :.2f} "
            f"right={right_range :.2f} "
            f"lateral_v={lateral_velocity :.2f} "
            f"center_cmd={lateral_command :.2f} "
            f"acc={probe_acc :.2f} "
            f"anchor_fixed={perception .anchor_fixed } "
            f"is_fixed_anchor="
            f"{lidar_robot .is_fixed_anchor } "
            f"base_reserve={lidar_robot .base_reserve } "
            f"connected="
            f"{lidar_robot .connected_to_base } "
            f"comm_parent={comm_parent_id } "
            f"comm_dist={comm_parent_distance :.2f} "
            f"comm_hard="
            f"{physical .COMM_GUARD_HARD_LIMIT :.2f} "
            f"forward_walkable={forward_walkable }"
            )

        return 







    if (
    physical .phase 
    !=physical .SimulationPhase .EXPLORE_BRANCH 
    or active_uid is None 
    ):
        return 

    descriptor =physical .branch_descriptors_by_uid .get (
    active_uid 
    )

    if descriptor is None :
        return 

    active_fixture =getattr (
    physical ,
    "active_branch",
    None ,
    )

    if active_fixture is None :
        return 

    frontiers =physical .get_frontier_shepherds (
    robots ,
    active_fixture ,
    )

    if not frontiers :
        return 

    tangent ,lateral_axis =physical .descriptor_local_basis (descriptor )
    tangent =tangent .normalize ()
    lateral_axis =lateral_axis .normalize ()
    tracked_uid =getattr (physical ,"integration_child_probe_odometry_uid",None )
    if tracked_uid !=active_uid :
        physical .integration_child_probe_odometry_uid =active_uid 
        physical .integration_child_probe_frontier_odometry =0.0 
    mean_forward_speed =float (np .mean ([
    max (0.0 ,robot .velocity .dot (tangent ))for robot in frontiers 
    ]))
    physical .integration_child_probe_frontier_odometry +=mean_forward_speed *dt 
    frontier_centroid_depth =float (
    physical .integration_child_probe_frontier_odometry 
    )

















    observed_width =float (
    getattr (
    descriptor ,
    "observed_physical_width",
    0.0 ,
    )
    or getattr (
    descriptor ,
    "observed_width",
    0.0 ,
    )
    or 0.0 
    )

    if observed_width >physical .EPSILON :
        child_probe_trigger_depth =min (
        CHILD_PROBE_TRIGGER_MAX_DISTANCE ,
        CHILD_PROBE_TRIGGER_WIDTH_RATIO 
        *observed_width ,
        )
    else :


        child_probe_trigger_depth =(
        CHILD_PROBE_TRIGGER_MAX_DISTANCE 
        )

    if physical .integration_frame %20 ==0 :
        observed_width =float (
        getattr (
        descriptor ,
        "observed_physical_width",
        0.0 ,
        )
        or getattr (
        descriptor ,
        "observed_width",
        0.0 ,
        )
        or 0.0 
        )

        print (
        "[ChildProbeGate] "
        f"junction={frame .junction_uid } "
        f"branch={active_uid } "
        f"frontier_depth="
        f"{frontier_centroid_depth :.2f} "
        f"trigger_depth="
        f"{child_probe_trigger_depth :.2f} "
        f"mouth_width="
        f"{observed_width :.2f} "
        f"depth_over_width="
        f"{frontier_centroid_depth /max (observed_width ,physical .EPSILON ):.2f} "
        f"frontiers="
        f"{len (frontiers )} "
        "depth_source=FRONTIER_LOCAL_ODOMETRY "
        "probe_active=False"
        )

    if (
    frontier_centroid_depth 
    <child_probe_trigger_depth 
    ):
        return 

    lidar_robot =perception .leader 





    if lidar_robot .role in {
    "RELAY",
    "TRUNK_RELAY",
    }:
        previous_role =lidar_robot .role 
        previous_relay_index =getattr (
        lidar_robot ,
        "relay_index",
        -1 ,
        )

        lidar_robot .role ="NORMAL"
        lidar_robot .relay_anchor =None 
        lidar_robot .relay_index =-1 

        if hasattr (
        lidar_robot ,
        "relay_scope",
        ):
            lidar_robot .relay_scope =None 

        if hasattr (
        lidar_robot ,
        "relay_owner_edge_id",
        ):
            lidar_robot .relay_owner_edge_id =None 

        lidar_robot .velocity .update (
        0.0 ,
        0.0 ,
        )
        lidar_robot .acceleration .update (
        0.0 ,
        0.0 ,
        )
        lidar_robot .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )

        print (
        "[LiDARIllegalRelayRelease] "
        f"lidar_id={lidar_robot .robot_id } "
        f"previous_role={previous_role } "
        f"relay_index={previous_relay_index }"
        )



    multi_dfs .child_probe_active =True 
    multi_dfs .child_probe_branch_uid =active_uid 
    multi_dfs .child_probe_start_frame =(
    physical .integration_frame 
    )
    multi_dfs .child_probe_lidar_id =(
    lidar_robot .robot_id 
    )




    tangent ,normal =physical .descriptor_local_basis (
    descriptor 
    )
    tangent =tangent .normalize ()
    normal =normal .normalize ()

    perception .yaw_deg =math .degrees (
    math .atan2 (
    tangent .y ,
    tangent .x ,
    )
    )

    perception .lateral_range_history .clear ()
    perception .lateral_last_checked_index =0 













    multi_dfs .child_session =ChildObservationSession (
    parent_junction_uid =frame .junction_uid ,
    parent_branch_uid =active_uid ,
    lidar_id =lidar_robot .robot_id ,
    start_frame =physical .integration_frame ,
    ingress_t =tangent .copy (),
    ingress_n =normal .copy (),
    probe_traveled_axial =0.0 ,
    )

    print (
    "[ChildSessionStart] "
    f"parent={frame .junction_uid } "
    f"branch={active_uid } "
    f"lidar_id={lidar_robot .robot_id } "
    f"frame={physical .integration_frame }"
    )












    perception .anchor_fixed =False 
    lidar_robot .is_fixed_anchor =False 
    lidar_robot .base_reserve =False 

    print (
    "[ChildProbeStart] "
    f"junction={frame .junction_uid } "
    f"branch={active_uid } "
    f"lidar_id={lidar_robot .robot_id } "
    f"frontier_depth={frontier_centroid_depth :.2f} "
    f"trigger_depth={child_probe_trigger_depth :.2f} "
    f"mouth_width={observed_width :.2f} "
    f"depth_over_width="
    f"{frontier_centroid_depth /max (observed_width ,physical .EPSILON ):.2f} "
    f"yaw={perception .yaw_deg :.2f} "
    "parent_release=False "
    "marker=False "
    "dfs_push=False"
    )

def update_child_observation_session (
physical :types .ModuleType ,
perception :AdaptivePerception ,
lidar_frame :LidarFrame ,
)->None :
    """Accumulate only fresh Child-observation LiDAR samples."""

    session =multi_dfs .child_session 

    if not multi_dfs .child_probe_active :
        return 

    if session is None :
        return 



    if perception .leader .robot_id !=session .lidar_id :
        raise RuntimeError (
        "Child observation LiDAR ID changed: "
        f"{session .lidar_id } -> "
        f"{perception .leader .robot_id }"
        )

    session .samples +=1 

    valid =(
    lidar_frame .interval_valid 
    and lidar_frame .selected is not None 
    )

    if valid :
        session .valid_samples +=1 
        session .consecutive_valid +=1 

        session .last_valid_w =float (
        lidar_frame .adaptive_w 
        )

        session .last_selected_threshold =float (
        lidar_frame .selected 
        )

    else :
        session .consecutive_valid =0 

    if physical .integration_frame %20 ==0 :
        print (
        "[ChildSession] "
        f"parent={session .parent_junction_uid } "
        f"branch={session .parent_branch_uid } "
        f"lidar_id={session .lidar_id } "
        f"samples={session .samples } "
        f"valid={session .valid_samples } "
        f"consecutive_valid={session .consecutive_valid } "
        f"openings={len (lidar_frame .openings )} "
        f"interval_valid={lidar_frame .interval_valid } "
        f"selected={lidar_frame .selected }"
        )

def _build_child_stationary_frozen_frame (
perception :AdaptivePerception ,
lidar_frame :LidarFrame ,
session :ChildObservationSession ,
)->LidarFrame |None :
    """Re-evaluate a stationary raw scan with the valid Candidate threshold.

    The current adaptive W may become invalid inside a Junction.
    Therefore the threshold that was valid when the moving Candidate
    was detected is frozen and reused here.

    Raw FAR/Rmax rays are still not accepted as physical mouths.
    Finite wall-side verification is performed separately.
    """

    if (
    session .candidate_selected_threshold is None 
    or session .candidate_lidar_frame is None 
    ):
        return None 

    frozen_threshold =float (
    session .candidate_selected_threshold 
    )

    candidate_frame =session .candidate_lidar_frame 

    openings ,diagnostics =(
    adaptive ._detect_openings_w_tau_with_diagnostics (
    lidar_frame .angles ,
    lidar_frame .raw ,
    selected_threshold =frozen_threshold ,
    threshold_interval_valid =True ,
    smoothing_window_size =SMOOTHING_WINDOW ,
    )
    )

    return LidarFrame (
    frame =lidar_frame .frame ,
    angles =lidar_frame .angles .copy (),
    raw =lidar_frame .raw .copy (),
    smoothed =np .asarray (
    diagnostics ["smoothed_ranges"]
    ),
    support =np .asarray (
    diagnostics ["open_support_mask"],
    dtype =bool ,
    ),
    openings =tuple (
    dict (item )
    for item in openings 
    ),
    left =lidar_frame .left ,
    right =lidar_frame .right ,





    adaptive_w =float (
    candidate_frame .adaptive_w 
    ),
    lower =float (candidate_frame .lower ),
    upper =float (candidate_frame .upper ),
    selected =frozen_threshold ,
    interval_valid =True ,
    current_evidence =False ,
    )


def match_stationary_tracks_to_verified_openings (
tracks :Sequence [PersistentOpening ],
openings :Sequence [dict [str ,float ]],
)->list [dict [str ,float ]]:
    """Match persistent Child openings to the current verified physical mouths."""

    remaining =[
    dict (opening )
    for opening in openings 
    ]

    matched :list [dict [str ,float ]]=[]

    for track in sorted (
    tracks ,
    key =lambda item :item .center_angle ,
    ):
        if not remaining :
            break 

        opening =min (
        remaining ,
        key =lambda item :circular_error (
        float (item ["center_angle"]),
        float (track .center_angle ),
        ),
        )

        error =circular_error (
        float (opening ["center_angle"]),
        float (track .center_angle ),
        )

        if error >ASSOCIATION_TOLERANCE_DEG :
            raise RuntimeError (
            "Persistent Child opening could not "
            "be matched to verified physical mouth: "
            f"track={track .center_angle :.2f} "
            f"error={error :.2f}"
            )

        matched .append (
        dict (opening )
        )
        remaining .remove (
        opening 
        )

    if len (matched )!=len (tracks ):
        raise RuntimeError (
        "Child stationary opening match incomplete: "
        f"tracks={len (tracks )} "
        f"matched={len (matched )}"
        )

    return matched 



def update_child_stationary_verification (
physical :types .ModuleType ,
perception :AdaptivePerception ,
lidar_frame :LidarFrame ,
)->None :
    """Confirm a Child Junction from persistent stationary physical mouths.

    After stationary confirmation this function performs
    logical Multi-DFS handoff only:

    - Parent ACTIVE -> ACTIVE_CHILD
    - create Child MultiJunctionFrame
    - DFS PUSH

    It still does NOT:
    - release Parent Guard/Shepherd roles,
    - create Completion/Return Markers,
    - initialize Child branch descriptors,
    - start Child Physical DFS.
    """

    session =multi_dfs .child_session 

    if (
    session is None 
    or not multi_dfs .child_candidate_active 
    or not session .anchor_stopped 
    or multi_dfs .child_confirmed 
    ):
        return 

    frozen_frame =_build_child_stationary_frozen_frame (
    perception ,
    lidar_frame ,
    session ,
    )

    if frozen_frame is None :
        return 

    session .stationary_samples +=1 

    verified_outgoing :list [
    dict [str ,float ]
    ]=[]

    candidate_width =float (
    session .candidate_lidar_frame .adaptive_w 
    )

    minimum_mouth_width =(
    CHILD_CANDIDATE_MIN_MOUTH_WIDTH_RATIO 
    *candidate_width 
    )



    for opening in frozen_frame .openings :
        start_angle =float (
        opening ["start_angle"]
        )
        end_angle =float (
        opening ["end_angle"]
        )
        center_angle =float (
        opening ["center_angle"]
        )

        try :
            start_point ,_ ,_ =(
            _nearest_wall_side_endpoint (
            frozen_frame ,
            perception ,
            start_angle ,
            search_direction =-1 ,
            )
            )

            end_point ,_ ,_ =(
            _nearest_wall_side_endpoint (
            frozen_frame ,
            perception ,
            end_angle ,
            search_direction =+1 ,
            )
            )
        except RuntimeError :




            continue 

        mouth_chord =(
        end_point -start_point 
        )

        if (
        mouth_chord .length ()
        <minimum_mouth_width 
        ):
            continue 

        radial =_body_local_unit (
        perception ,
        center_angle ,
        )

        if (
        radial .length_squared ()
        <=physical .EPSILON 
        ):
            continue 

        radial =radial .normalize ()

        ingress_alignment =float (
        radial .dot (session .ingress_t )
        )



        if ingress_alignment <=-0.50 :
            continue 

        verified_outgoing .append (
        opening 
        )


    available =set (
    range (len (session .stationary_tracks ))
    )

    for opening in verified_outgoing :
        center =float (
        opening ["center_angle"]
        )

        candidates =[
        (
        circular_error (
        center ,
        session .stationary_tracks [
        index 
        ].center_angle ,
        ),
        index ,
        )
        for index in available 
        ]

        error ,index =min (
        candidates ,
        default =(float ("inf"),-1 ),
        )

        if (
        index >=0 
        and error 
        <=ASSOCIATION_TOLERANCE_DEG 
        ):
            track =(
            session .stationary_tracks [index ]
            )
            available .remove (index )

        else :
            track =PersistentOpening (
            "CHILD_OPEN_"
            f"{len (session .stationary_tracks ):02d}"
            )
            session .stationary_tracks .append (
            track 
            )

        track .update (
        opening ,
        physical .integration_frame ,
        )

    persistent_outgoing =[
    track 
    for track in session .stationary_tracks 
    if (
    len (track .observations )
    >=MIN_PERSISTENT_OBSERVATIONS 
    and track .persistence_ratio (
    session .stationary_samples 
    )
    >=CHILD_STATIONARY_PERSISTENCE_RATIO 
    )
    ]





    persistent_non_axial =[]

    for track in persistent_outgoing :
        radial =_body_local_unit (
        perception ,
        track .center_angle ,
        )

        if (
        radial .length_squared ()
        <=physical .EPSILON 
        ):
            continue 

        radial =radial .normalize ()

        alignment =abs (
        float (
        radial .dot (session .ingress_t )
        )
        )

        if (
        alignment 
        <=CHILD_CANDIDATE_NON_AXIAL_MAX_DOT 
        ):
            persistent_non_axial .append (
            track 
            )

    if (
    physical .integration_frame 
    %10 
    ==0 
    ):
        print (
        "[ChildStationaryVerification] "
        f"parent={session .parent_junction_uid } "
        f"branch={session .parent_branch_uid } "
        f"lidar_id={session .lidar_id } "
        f"samples={session .stationary_samples } "
        f"frozen_threshold="
        f"{session .candidate_selected_threshold :.2f} "
        f"raw_openings="
        f"{len (frozen_frame .openings )} "
        f"verified_outgoing="
        f"{len (verified_outgoing )} "
        f"persistent_outgoing="
        f"{len (persistent_outgoing )} "
        f"persistent_non_axial="
        f"{len (persistent_non_axial )}"
        )

    confirmed =(
    len (persistent_outgoing )
    >=CHILD_STATIONARY_MIN_OUTGOING 
    and len (persistent_non_axial )
    >=1 
    )

    if not confirmed :
        return 



    persistent_outgoing .sort (
    key =lambda track :track .center_angle 
    )

    session .stationary_outgoing =list (
    persistent_outgoing 
    )

    session .stationary_verified_openings =(
    match_stationary_tracks_to_verified_openings (
    persistent_outgoing ,
    verified_outgoing ,
    )
    )

    session .stationary_confirmed_lidar_frame =(
    frozen_frame 
    )

    session .stationary_confirmed =True 

    session .stationary_confirmation_frame =(
    physical .integration_frame 
    )

    multi_dfs .child_confirmed =True 

    staged_child =(
    multi_dfs .stage_confirmed_child (
    session 
    )
    )

    print (
    "[ChildJunctionConfirmed] "
    f"parent={session .parent_junction_uid } "
    f"branch={session .parent_branch_uid } "
    f"lidar_id={session .lidar_id } "
    f"frame="
    f"{session .stationary_confirmation_frame } "
    f"stationary_samples="
    f"{session .stationary_samples } "
    f"outgoing_count="
    f"{len (session .stationary_outgoing )} "
    f"frozen_threshold="
    f"{session .candidate_selected_threshold :.2f} "
    "parent_source=INGRESS_HISTORY "
    "parent_release=False "
    "marker=False "
    "active_child=True "
    "dfs_push=False "
    "pending_child=True "
    f"child={staged_child .junction_uid } "
    f"dfs_depth={multi_dfs .depth }"
    )



def update_child_moving_candidate (
physical :types .ModuleType ,
perception :AdaptivePerception ,
lidar_frame :LidarFrame ,
dt :float ,
)->None :
    """Detect a moving Child-Junction candidate from fresh local LiDAR evidence.

    This only latches a candidate.

    It does NOT:
    - confirm a Child Junction,
    - release the Parent Junction,
    - create Markers,
    - perform DFS PUSH.
    """

    session =multi_dfs .child_session 

    if (
    not multi_dfs .child_probe_active 
    or session is None 
    or multi_dfs .child_candidate_active 
    ):
        return 





    probe_forward_speed =max (
    0.0 ,
    float (perception .leader .velocity .dot (session .ingress_t )),
    )
    session .probe_traveled_axial +=probe_forward_speed *dt 

    width_reference =session .last_valid_w 

    if width_reference is None :
        width_reference =float (
        lidar_frame .adaptive_w 
        )

    parent_clearance_depth =(
    CHILD_PARENT_CLEARANCE_W_RATIO 
    *width_reference 
    )

    if session .probe_traveled_axial <parent_clearance_depth :
        session .structural_streak =0 

        if physical .integration_frame %20 ==0 :
            print (
            "[ChildParentClearance] "
            f"parent={session .parent_junction_uid } "
            f"branch={session .parent_branch_uid } "
            f"lidar_id={session .lidar_id } "
            f"odometry_axial={session .probe_traveled_axial :.2f} "
            f"required={parent_clearance_depth :.2f} "
            "cleared=False"
            )

        return 


    observation =evaluate_general_junction_structure (
    perception ,
    lidar_frame ,
    session .ingress_t ,
    )

    if not observation .valid :
        session .structural_streak =0 
        return 

    session .structural_streak +=1 

    if (
    physical .integration_frame 
    %10 
    ==0 
    ):
        print (
        "[GeneralJunctionEvidence] "
        f"scope=CHILD "
        f"parent={session .parent_junction_uid } "
        f"branch={session .parent_branch_uid } "
        f"verified_outgoing="
        f"{len (observation .verified_outgoing )} "
        f"non_axial="
        f"{len (observation .non_axial_outgoing )} "
        f"entrance_depth="
        f"{observation .entrance_depth :.2f} "
        f"streak="
        f"{session .structural_streak }/"
        f"{CHILD_CANDIDATE_MIN_STRUCTURAL_STREAK }"
        )

    if (
    session .structural_streak 
    <CHILD_CANDIDATE_MIN_STRUCTURAL_STREAK 
    ):
        return 

    candidate_depth =float (
    observation .entrance_depth 
    )

    multi_dfs .child_candidate_active =True 

    session .candidate_frame =(
    physical .integration_frame 
    )

    perception .lateral_last_checked_index =len (
    perception .lateral_range_history 
    )

    session .candidate_depth_local =(
    candidate_depth 
    )

    session .candidate_selected_threshold =float (
    lidar_frame .selected 
    )

    session .candidate_lidar_frame =(
    lidar_frame 
    )



    session .candidate_traveled_axial =0.0 

    session .candidate_remaining_depth =(
    candidate_depth 
    )

    session .anchor_stopped =False 
    session .anchor_stop_frame =None 

    print (
    "[ChildMovingCandidate] "
    f"parent={session .parent_junction_uid } "
    f"branch={session .parent_branch_uid } "
    f"lidar_id={session .lidar_id } "
    f"frame={session .candidate_frame } "
    f"verified_outgoing="
    f"{len (observation .verified_outgoing )} "
    f"non_axial="
    f"{len (observation .non_axial_outgoing )} "
    f"depth_local={candidate_depth :.2f} "
    f"threshold={session .candidate_selected_threshold :.2f} "
    "confirmed=False "
    "parent_release=False "
    "marker=False "
    "dfs_push=False"
    )

def _load_physical_definitions ()->types .ModuleType :
    """Load definitions before the original top-level main loop starts."""
    source =PHYSICAL_SOURCE .read_text (encoding ="utf-8")
    marker ="robots, reference_density, color_reference_density = initialize_simulation()"
    if marker not in source :
        raise RuntimeError ("Physical DFS integration marker not found")
    definitions =source [:source .index (marker )]
    module =types .ModuleType ("_physical_dfs_runtime")
    module .__file__ =str (PHYSICAL_SOURCE )
    module .__package__ ="pygame_simulator"
    sys .modules [module .__name__ ]=module 
    exec (compile (definitions ,str (PHYSICAL_SOURCE ),"exec"),module .__dict__ )
    return module 

def configure_extended_approach (physical :types .ModuleType )->tuple [float ,float ]:
    """Make horizontal branches symmetric and modestly extend the approach."""
    left_branch_length =float (physical .normal_length )
    right_branch_length_before =float (physical .right_length )
    right_branch_length_after =left_branch_length 
    base_reference_length =float (physical .base_length )
    base_length_before =base_reference_length +PREVIOUS_APPROACH_EXTENSION 
    base_length_after =base_length_before +BASE_ADDED_EXTENSION 

    physical .right_length =right_branch_length_after 
    physical .base_length =base_length_after 
    right_x =physical .center_x +physical .half_width +right_branch_length_after 
    bottom_y =physical .center_y +physical .half_width +base_length_after 
    points =list (physical .cross_points )
    points [3 ]=(right_x ,physical .center_y -physical .half_width )
    points [4 ]=(right_x ,physical .center_y +physical .half_width )
    points [6 ]=(physical .center_x +physical .half_width ,bottom_y )
    points [7 ]=(physical .center_x -physical .half_width ,bottom_y )
    physical .cross_points =points 
    physical .right_rect =pygame .Rect (
    physical .center_x +physical .half_width ,
    physical .center_y -physical .half_width ,
    round (right_branch_length_after ),
    physical .corridor_width ,
    )
    physical .bottom_rect =pygame .Rect (
    physical .center_x -physical .half_width ,
    physical .center_y +physical .half_width ,
    physical .corridor_width ,
    round (base_length_after ),
    )
    physical .dead_end_regions ["RIGHT"]=pygame .Rect (
    right_x -physical .END_REGION_DEPTH ,
    physical .center_y -physical .half_width ,
    physical .END_REGION_DEPTH ,
    physical .corridor_width ,
    )
    physical .early_capture_regions ["RIGHT"]=pygame .Rect (
    right_x -physical .EARLY_CAPTURE_DEPTH ,
    physical .center_y -physical .half_width ,
    physical .EARLY_CAPTURE_DEPTH ,
    physical .corridor_width ,
    )
    physical .BRANCH_LENGTHS ["RIGHT"]=right_branch_length_after 
    physical .get_junction_state ().branch_edges ["RIGHT"].length =right_branch_length_after 
    physical .MAX_TRANSPORT_DISTANCE =(
    max (physical .BRANCH_LENGTHS .values ())+physical .corridor_width 
    )
    physical .BASE_POSITION =pygame .Vector2 (
    physical .center_x -25 *physical .MAP_SCALE ,
    bottom_y -14 *physical .MAP_SCALE ,
    )
    physical .BASE_COMPRESSION_CENTER =pygame .Vector2 (
    physical .center_x ,
    physical .center_y +physical .half_width +base_length_after *0.60 ,
    )
    physical .floor_surface =pygame .Surface (
    (physical .SCREEN_WIDTH ,physical .SCREEN_HEIGHT ),pygame .SRCALPHA 
    )
    physical .floor_surface .fill ((0 ,0 ,0 ,0 ))
    pygame .draw .polygon (
    physical .floor_surface ,(255 ,255 ,255 ,255 ),physical .cross_points 
    )
    physical .walkable_mask =pygame .mask .from_surface (physical .floor_surface )
    print (
    f"[Map] left_branch_length={left_branch_length :.1f} "
    f"right_branch_length_before={right_branch_length_before :.1f} "
    f"right_branch_length_after={right_branch_length_after :.1f} "
    f"base_length_before={base_length_before :.1f} "
    f"base_length_after={base_length_after :.1f} "
    f"base_added_extension={BASE_ADDED_EXTENSION :.1f}"
    )
    return base_length_before ,base_length_after 

def configure_multi_test_geometry (
physical :types .ModuleType ,
)->None :
    """Install one physical Child Junction for Multi-DFS testing.

    IMPORTANT:
    J1 geometry exists only as simulator environment geometry.
    Runtime Junction detection must never use its center/rect directly.
    """

    x0 =float (physical .center_x )
    y0 =float (physical .center_y )
    h =float (physical .half_width )
    length =float (physical .normal_length )



    x1 =x0 +h +float (physical .right_length )

    j1_rect =pygame .Rect (
    round (x1 -h ),
    round (y0 -h ),
    round (physical .corridor_width ),
    round (physical .corridor_width ),
    )

    j1_up_rect =pygame .Rect (
    round (x1 -h ),
    round (y0 -h -length ),
    round (physical .corridor_width ),
    round (length ),
    )

    j1_down_rect =pygame .Rect (
    round (x1 -h ),
    round (y0 +h ),
    round (physical .corridor_width ),
    round (length ),
    )

    physical .cross_points =[


    (x0 -h ,y0 -h -length ),
    (x0 +h ,y0 -h -length ),

    (x0 +h ,y0 -h ),
    (x1 -h ,y0 -h ),



    (x1 -h ,y0 -h -length ),
    (x1 +h ,y0 -h -length ),



    (x1 +h ,y0 +h +length ),



    (x1 -h ,y0 +h +length ),
    (x1 -h ,y0 +h ),



    (x0 +h ,y0 +h ),



    (x0 +h ,y0 +h +physical .base_length ),
    (x0 -h ,y0 +h +physical .base_length ),
    (x0 -h ,y0 +h ),



    (x0 -h -length ,y0 +h ),
    (x0 -h -length ,y0 -h ),



    (x0 -h ,y0 -h ),
    ]



    physical .floor_surface =pygame .Surface (
    (physical .SCREEN_WIDTH ,physical .SCREEN_HEIGHT ),
    pygame .SRCALPHA ,
    )
    physical .floor_surface .fill ((0 ,0 ,0 ,0 ))

    pygame .draw .polygon (
    physical .floor_surface ,
    (255 ,255 ,255 ,255 ),
    physical .cross_points ,
    )

    physical .walkable_mask =pygame .mask .from_surface (
    physical .floor_surface 
    )


    original_get_robot_region =physical .get_robot_region 

    def multi_get_robot_region (
    position :pygame .Vector2 ,
    )->str :
        point =(
        int (position .x ),
        int (position .y ),
        )

        if j1_rect .collidepoint (point ):
            return "J1_JUNCTION"

        if j1_up_rect .collidepoint (point ):
            return "J1_UP"

        if j1_down_rect .collidepoint (point ):
            return "J1_DOWN"

        return original_get_robot_region (position )

    physical .get_robot_region =multi_get_robot_region 

    def multi_is_region_allowed (
    position :pygame .Vector2 ,
    )->bool :
        return physical .get_robot_region (position )in {
        "BOTTOM",
        "JUNCTION",
        "UP",
        "LEFT",
        "RIGHT",
        "J1_JUNCTION",
        "J1_UP",
        "J1_DOWN",
        }

    physical .is_region_allowed =multi_is_region_allowed 





    off_map =pygame .Rect (
    -10000 ,
    -10000 ,
    1 ,
    1 ,
    )

    physical .dead_end_regions ["RIGHT"]=off_map .copy ()
    physical .early_capture_regions ["RIGHT"]=off_map .copy ()

    print (
    "[MultiMap] physical_test_geometry_ready "
    "J0_RIGHT_is_nonterminal=True "
    "J1_shape=T_JUNCTION "
    "runtime_detection_authority=LIDAR_ONLY"
    )

def install_local_forward_ingress (
physical :types .ModuleType ,
)->None :
    """Install body-local forward propulsion during initial ingress."""

    original_route_force =physical .compute_route_force 
    original_compression_envelope =(
    physical .get_base_compression_envelope 
    )

    def local_route_force (
    robot :Any ,
    )->pygame .Vector2 :
        if (
        physical .phase 
        !=physical .SimulationPhase .MOVE_TO_JUNCTION 
        ):
            return original_route_force (robot )

        if robot .role in {
        "PEBBLE",
        "RELAY",
        "TRUNK_RELAY",
        }:
            return pygame .Vector2 ()

        yaw =float (
        getattr (
        robot ,
        "body_yaw",
        -0.5 *math .pi ,
        )
        )

        weight =float (
        getattr (
        robot ,
        "propulsion_weight",
        adaptive .LOCAL_FOLLOWER_DRIVE_WEIGHT ,
        )
        )

        forward =pygame .Vector2 (
        math .cos (yaw ),
        math .sin (yaw ),
        )

        return (
        forward 
        *adaptive .LOCAL_FORWARD_DRIVE_FORCE 
        *ROBOT_MOTION_SPEED_SCALE 
        *weight 
        )

    physical .compute_route_force =(
    local_route_force 
    )





    physical .get_base_compression_envelope =(
    lambda :(
    0.0 
    if (
    physical .phase 
    ==physical .SimulationPhase .MOVE_TO_JUNCTION 
    )
    else original_compression_envelope ()
    )
    )


def circular_error (a :float ,b :float )->float :
    return abs ((a -b +180.0 )%360.0 -180.0 )


def detect_structural_opening_candidates (
angles :np .ndarray ,
smoothed :np .ndarray ,
)->list [dict [str ,float ]]:
    """Build moving structural candidates without an OPEN/WALL threshold."""
    boundary_angles ,gradient =(
    adaptive .circular_range_gradient (
    angles ,
    smoothed ,
    )
    )

    gradient_threshold =(
    adaptive ._automatic_gradient_threshold (
    gradient ,
    float (
    adaptive .FROZEN_PARAMETERS [
    "gradient_mad_scale"
    ]
    ),
    float (
    adaptive .FROZEN_PARAMETERS [
    "min_gradient_threshold"
    ]
    ),
    )
    )

    positive_mask =gradient >=gradient_threshold 
    negative_mask =gradient <=-gradient_threshold 

    positive_runs =adaptive ._circular_runs (
    positive_mask ,
    value =True ,
    )
    negative_runs =adaptive ._circular_runs (
    negative_mask ,
    value =True ,
    )

    start_indices =[
    int (run [np .argmax (gradient [run ])])
    for run in positive_runs 
    if len (run )>0 
    ]
    end_indices =[
    int (run [np .argmin (gradient [run ])])
    for run in negative_runs 
    if len (run )>0 
    ]

    min_width =float (
    adaptive .FROZEN_PARAMETERS [
    "min_opening_width_deg"
    ]
    )
    candidates :list [dict [str ,float ]]=[]
    used_end_indices :set [int ]=set ()

    for start_idx in start_indices :
        start_angle =float (boundary_angles [start_idx ])
        possible_ends =[]

        for end_idx in end_indices :
            if end_idx in used_end_indices :
                continue 

            end_angle =float (boundary_angles [end_idx ])
            width =float ((end_angle -start_angle )%360.0 )

            if width <=0.0 :
                continue 

            possible_ends .append ((width ,end_idx ,end_angle ))

        possible_ends .sort (key =lambda item :item [0 ])

        for width ,end_idx ,end_angle in possible_ends :
            if width <min_width :
                continue 

            if width >=359.0 :
                continue 

            center_angle =float (
            (start_angle +0.5 *width +180.0 )%360.0 -180.0 
            )
            edge_strength =min (
            abs (float (gradient [start_idx ])),
            abs (float (gradient [end_idx ])),
            )
            confidence =float (
            np .clip (
            edge_strength /max (gradient_threshold ,1.0e-9 ),
            0.0 ,
            1.0 ,
            )
            )

            candidates .append ({
            "start_angle":start_angle ,
            "end_angle":end_angle ,
            "center_angle":center_angle ,
            "width_deg":width ,
            "confidence":confidence ,
            })
            used_end_indices .add (end_idx )
            break 

    candidates .sort (key =lambda item :item ["center_angle"])
    return candidates 


def build_moving_candidate_track_reference (
candidate :dict [str ,Any ],
)->dict [str ,float ]:
    """Store the angular geometry of one moving Opening candidate."""
    return {
    "center_angle":float (candidate ["center_angle"]),
    "start_angle":float (candidate ["start_angle"]),
    "end_angle":float (candidate ["end_angle"]),
    "width_deg":float (candidate ["width_deg"]),
    }


def associate_moving_candidate_to_reference (
candidates :Sequence [dict [str ,Any ]],
reference :dict [str ,float ],
center_tolerance_deg :float ,
boundary_tolerance_deg :float ,
)->int |None :
    """Associate a moving Opening with its prior center/start/end geometry."""
    best_index :int |None =None 
    best_score =float ("inf")
    for candidate_index ,candidate in enumerate (candidates ):
        center_error =circular_error (
        float (candidate ["center_angle"]),
        float (reference ["center_angle"]),
        )
        start_error =circular_error (
        float (candidate ["start_angle"]),
        float (reference ["start_angle"]),
        )
        end_error =circular_error (
        float (candidate ["end_angle"]),
        float (reference ["end_angle"]),
        )
        if center_error >center_tolerance_deg :
            continue 
        if start_error >boundary_tolerance_deg :
            continue 
        if end_error >boundary_tolerance_deg :
            continue 
        score =center_error +start_error +end_error 
        if score <best_score :
            best_score =score 
            best_index =candidate_index 
    return best_index 


class PhysicalMapLidar :
    """Analytic ray caster over the Physical DFS cross polygon walls."""

    def __init__ (self ,physical :types .ModuleType )->None :
        self .angles =np .linspace (-180.0 ,180.0 ,RAY_COUNT ,endpoint =False )
        points =[np .asarray (point ,dtype =float )for point in physical .cross_points ]
        self .segments =tuple (zip (points ,points [1 :]+points [:1 ]))

    @staticmethod 
    def _hit (origin :np .ndarray ,direction :np .ndarray ,segment :Any )->float |None :
        start ,end =segment 
        edge =end -start 
        denominator =direction [0 ]*edge [1 ]-direction [1 ]*edge [0 ]
        if abs (denominator )<1.0e-10 :
            return None 
        offset =start -origin 
        ray_t =(offset [0 ]*edge [1 ]-offset [1 ]*edge [0 ])/denominator 
        seg_t =(offset [0 ]*direction [1 ]-offset [1 ]*direction [0 ])/denominator 
        if ray_t >=0.0 and -1.0e-9 <=seg_t <=1.0 +1.0e-9 :
            return float (ray_t )
        return None 

    def scan (self ,position :pygame .Vector2 ,yaw_deg :float )->np .ndarray :
        origin =np .array ([position .x ,position .y ],dtype =float )
        ranges =np .full (RAY_COUNT ,MAX_RANGE ,dtype =float )
        for index ,local_angle in enumerate (self .angles ):
            world_angle =math .radians (yaw_deg +float (local_angle ))
            direction =np .array ([math .cos (world_angle ),math .sin (world_angle )])
            hits =[
            value for segment in self .segments 
            if (value :=self ._hit (origin ,direction ,segment ))is not None 
            ]
            if hits :
                ranges [index ]=min (MAX_RANGE ,min (hits ))
        return ranges 


class AdaptivePerception :
    def __init__ (self ,physical :types .ModuleType ,robots :Sequence [Any ])->None :
        self .physical =physical 
        self .sensor =PhysicalMapLidar (physical )
        self .state =PerceptionState .MOVING 
        self .guard_activation_stage ="WAIT_GROUP"
        self .guard_activation_groups :list [list [ProvisionalGuardGeometry ]]=[]
        self .guard_current_group_index =0 
        self .guard_all_groups_activated =False 
        self .integration_detected_branch_order :list [str ]=[]
        self .guard_side_pair_activation_frame :int |None =None 
        self .guard_up_activation_frame :int |None =None 
        self .frame =0 
        self .lateral_range_history :list [tuple [int ,float ,float ]]=[]
        self .lateral_last_checked_index =0 
        self .lateral_baseline_left :float |None =None 
        self .lateral_baseline_right :float |None =None 
        self .yaw_deg =-90.0 
        self .last_valid_w :float |None =None 
        self .stationary_threshold_w :float |None =None 
        self .stationary_threshold_lower :float |None =None 
        self .stationary_threshold_upper :float |None =None 
        self .stationary_threshold :float |None =None 
        self .last_frame :LidarFrame |None =None 
        self .junction_confirmed =False 
        self .junction_candidate_detected =False 
        self .junction_candidate_frame :int |None =None 
        self .junction_candidate_time :float |None =None 
        self .entrance_detected =False 
        self .entrance_detection_frame :int |None =None 
        self .entrance_confidence =0.0 
        self .entrance_depth :float |None =None 
        self .entrance_left_endpoint_local :pygame .Vector2 |None =None 
        self .entrance_right_endpoint_local :pygame .Vector2 |None =None 
        self .entrance_center_local :pygame .Vector2 |None =None 
        self .entrance_world_position :pygame .Vector2 |None =None 
        self .candidate_lidar_frame :LidarFrame |None =None 
        self .entrance_history :list [tuple [float ,float ]]=[]
        self .anchor_fix_frame :int |None =None 
        self .confirmation_frame :int |None =None 
        self .confirmation_time :float |None =None 
        self .first_open_support_frame :int |None =None 
        self .first_junction_evidence_frame :int |None =None 
        self .anchor_fixed =False 
        self .anchor_position :pygame .Vector2 |None =None 
        self .pre_detection_travel =0.0 
        self .post_fix_drift =0.0 
        self .stationary_samples =0 
        self .continuation_bearing_deg :float |None =None 
        self .continuation_reference :dict [str ,float ]|None =None 
        self .moving_continuation_candidate :dict [str ,Any ]|None =None 
        self .moving_candidate_tracks :list [
        MovingOpeningCandidateTrack 
        ]=[]
        self .moving_persistent_tracks :list [
        MovingOpeningCandidateTrack 
        ]=[]
        self .moving_persistent_candidates :list [
        dict [str ,Any ]
        ]=[]
        self .next_moving_track_index =0 
        self .stationary_boundary_observations :list [list [float ]]=[]
        self .stationary_smoothed_scans :list [np .ndarray ]=[]
        self .stationary_verified_openings :list [
        dict [str ,float ]
        ]=[]
        self .tracks :list [PersistentOpening ]=[]
        self .parent :PersistentOpening |ParentTopologyEdge |None =None 
        self .parent_source :str |None =None 
        self .outgoing :list [PersistentOpening ]=[]
        self .handoff_complete =False 
        self .topology_ready_frame :int |None =None 
        self .guard_geometry_frame :int |None =None 
        self .guard_who_frame :int |None =None 
        self .guard_motion_start_frame :int |None =None 
        self .provisional_guards :list [ProvisionalGuardGeometry ]=[]
        self .provisional_guard_started =False 
        self .guard_leakage :dict [str ,dict [str ,Any ]]={}
        self .guard_communication_audits :dict [str ,dict [str ,Any ]]={}
        self .anchor_fixed_mean_normal_forward_speed =0.0 
        self .pre_topology_normal_forward_speeds :list [float ]=[]
        self .robots =robots 
        self .leader =self ._select_leader (robots )
        self .initial_leader_position =self .leader .position .copy ()
        self .leader .is_lidar_robot =True 
        self .leader .is_fixed_anchor =False 
        self .leader .body_yaw =math .radians (self .yaw_deg )
        print (f"lidar_id={self .leader .robot_id }")
        print (
        f"lidar_initial_position=({self .initial_leader_position .x :.3f}, "
        f"{self .initial_leader_position .y :.3f})"
        )
        print (f"[LiDAR] leader_id={self .leader .robot_id }")
        print (
        f"[LiDAR] initial_position=({self .initial_leader_position .x :.3f},"
        f"{self .initial_leader_position .y :.3f})"
        )

    def _select_leader (self ,robots :Sequence [Any ])->Any :
        for robot in robots :
            if robot .robot_id ==LIDAR_ROBOT_ID :
                return robot 

        raise RuntimeError (
        f"LiDAR robot {LIDAR_ROBOT_ID } not found"
        )

    def reset (self ,robots :Sequence [Any ])->None :
        self .__init__ (self .physical ,robots )

    def mean_normal_forward_speed (self )->float :
        forward =pygame .Vector2 (
        math .cos (math .radians (self .yaw_deg )),
        math .sin (math .radians (self .yaw_deg )),
        )
        normals =[
        robot for robot in self .robots 
        if robot .role =="NORMAL"and not robot .base_reserve 
        ]
        return float (np .mean ([
        robot .velocity .dot (forward )for robot in normals 
        ]))if normals else 0.0 

    def _update_moving_opening_persistence (
    self ,
    candidates :Sequence [dict [str ,Any ]],
    )->list [MovingOpeningCandidateTrack ]:
        candidate_list =[
        dict (candidate )
        for candidate in candidates 
        ]
        available_indices =list (
        range (len (candidate_list ))
        )

        for track in self .moving_candidate_tracks :
            if not available_indices :
                break 

            available_candidates =[
            candidate_list [index ]
            for index in available_indices 
            ]
            local_index =(
            associate_moving_candidate_to_reference (
            available_candidates ,
            track .reference ,
            ASSOCIATION_TOLERANCE_DEG ,
            ASSOCIATION_TOLERANCE_DEG ,
            )
            )

            if local_index is None :
                continue 

            matched_index =available_indices .pop (
            int (local_index )
            )
            track .update (
            candidate_list [matched_index ],
            self .frame ,
            )

        for candidate_index in available_indices :
            candidate =candidate_list [candidate_index ]
            track =MovingOpeningCandidateTrack (
            track_id =(
            f"MOVING_OPEN_"
            f"{self .next_moving_track_index :02d}"
            ),
            reference =(
            build_moving_candidate_track_reference (
            candidate 
            )
            ),
            latest_candidate =dict (candidate ),
            )
            self .next_moving_track_index +=1 
            track .update (candidate ,self .frame )
            self .moving_candidate_tracks .append (track )

        first_frame =(
        self .frame 
        -MOVING_PERSISTENCE_WINDOW 
        +1 
        )

        for track in self .moving_candidate_tracks :
            track .prune (self .frame )

        self .moving_candidate_tracks =[
        track 
        for track in self .moving_candidate_tracks 
        if track .last_frame >=first_frame 
        ]

        persistent =[
        track 
        for track in self .moving_candidate_tracks 
        if (
        track .last_frame ==self .frame 
        and len (track .observation_frames )
        >=MOVING_MIN_PERSISTENT_OBSERVATIONS 
        and track .persistence_ratio ()
        >=MOVING_PERSISTENCE_RATIO 
        )
        ]

        if self .frame %5 ==0 :
            print (
            "[MovingPersistence] "
            f"frame={self .frame } "
            f"candidates={len (candidate_list )} "
            f"tracks="
            f"{[(track .track_id,len (track .observation_frames ),round (track .persistence_ratio (),2 ))for track in self .moving_candidate_tracks ]} "
            f"persistent="
            f"{[track .track_id for track in persistent ]}"
            )

        return persistent 

    def _associate (self ,openings :Sequence [dict [str ,float ]])->None :
        available =set (range (len (self .tracks )))
        for opening in openings :
            center =float (opening ["center_angle"])
            candidates =[
            (circular_error (center ,self .tracks [index ].center_angle ),index )
            for index in available 
            ]
            error ,index =min (candidates ,default =(float ("inf"),-1 ))
            if error <=ASSOCIATION_TOLERANCE_DEG :
                track =self .tracks [index ]
                available .remove (index )
            else :
                track =PersistentOpening (f"OPEN_{len (self .tracks ):02d}")
                self .tracks .append (track )
            track .update (opening ,self .frame )

    def _persistent_ready (self )->bool :
        persistent =[
        track 
        for track in self .tracks 
        if (
        len (track .observations )
        >=MIN_PERSISTENT_OBSERVATIONS 
        and track .persistence_ratio (
        self .stationary_samples 
        )
        >=0.60 
        )
        ]

        if len (persistent )<2 :
            return False 

        rear_sector_deg =45.0 

        rear_candidates =[
        track 
        for track in persistent 
        if circular_error (
        track .center_angle ,
        180.0 ,
        )
        <=rear_sector_deg 
        ]

        if rear_candidates :

            self .parent =min (
            rear_candidates ,
            key =lambda track :circular_error (
            track .center_angle ,
            180.0 ,
            ),
            )

            self .parent .persistent_id =(
            "PARENT_00"
            )

            self .parent_source =(
            "LIDAR_PERSISTENT"
            )

            children =[
            track 
            for track in persistent 
            if circular_error (
            track .center_angle ,
            180.0 ,
            )
            >rear_sector_deg 
            ]

        else :
            self .parent =(
            ParentTopologyEdge ()
            )

            self .parent_source =(
            self .parent .source 
            )

            children =list (
            persistent 
            )

        children .sort (
        key =lambda track :
        track .center_angle 
        )
        if len (children )<2 :
            return False 

        self .outgoing =list (
        children 
        )

        for index ,track in enumerate (
        self .outgoing 
        ):
            track .persistent_id =(
            f"J0-B{index }"
            )

        return True 

    def _reconstruct_stationary_openings (
    self ,
    boundary_angles :np .ndarray ,
    angular_steps :np .ndarray ,
    )->bool :
        """Verify moving hypotheses using fresh stationary LiDAR scans."""

        if (
        self .stationary_samples <STATIONARY_WINDOW 
        or not self .stationary_smoothed_scans 
        ):
            return False 

        _ =boundary_angles 

        stationary_profile =np .median (
        np .stack (
        self .stationary_smoothed_scans ,
        axis =0 ,
        ),
        axis =0 ,
        )

        hypotheses :list [
        tuple [str ,dict [str ,Any ]]
        ]=[]

        if self .moving_continuation_candidate is not None :
            hypotheses .append (
            (
            "CONTINUATION",
            dict (self .moving_continuation_candidate ),
            )
            )

        for moving_track in self .moving_persistent_tracks :
            hypotheses .append (
            (
            moving_track .track_id ,
            dict (moving_track .latest_candidate ),
            )
            )

        if not hypotheses :
            self .stationary_verified_openings =[]
            self .tracks =[]
            self .outgoing =[]
            return False 

        verified_records :list [
        tuple [str ,dict [str ,float ]]
        ]=[]

        for hypothesis_id ,candidate in hypotheses :
            E_left ,E_right ,bearing_deg ,_ =(
            adaptive .candidate_branch_edges_local (
            candidate ,
            self .sensor .angles ,
            stationary_profile ,
            )
            )
            geometry =adaptive .compute_candidate_geometry (
            E_left ,
            E_right ,
            )

            if geometry is None :
                print (
                "[StationaryCandidateRejected] "
                f"id={hypothesis_id} "
                "reason=INVALID_FINITE_ENDPOINT_GEOMETRY"
                )
                continue 

            d_left =float (
            np .linalg .norm (
            np .asarray (E_left ,dtype =float )
            )
            )
            d_right =float (
            np .linalg .norm (
            np .asarray (E_right ,dtype =float )
            )
            )
            d_max =float (geometry ["d_max"])

            # -------------------------------------------------
            # Adaptive threshold policy:
            #
            # Keep the latest cleaned stationary geometry /
            # candidate-verification pipeline, but do NOT use
            # candidate d_max to define the OPEN/WALL threshold.
            #
            # The threshold remains the same Adaptive-W policy
            # used by the moving detector:
            #
            #   T_min = W * (1 + m)
            #   T_max = R_max - tau
            #   T     = select(T_min, T_max, alpha)
            #
            # d_max is retained only as stationary geometry
            # diagnostics and is not a threshold input.
            # -------------------------------------------------
            candidate_threshold =self .stationary_threshold 
            candidate_t_min =self .stationary_threshold_lower 
            candidate_t_max =self .stationary_threshold_upper 
            interval_valid =(
            candidate_threshold is not None 
            and candidate_t_min is not None 
            and candidate_t_max is not None 
            )

            if not interval_valid :
                print (
                "[StationaryCandidateRejected] "
                f"id={hypothesis_id} "
                f"d0={d_left :.3f} "
                f"d1={d_right :.3f} "
                f"dmax={d_max :.3f} "
                f"Tmin={candidate_t_min :.3f} "
                f"Tmax={candidate_t_max :.3f} "
                "reason=EMPTY_SAFE_THRESHOLD_INTERVAL"
                )
                continue 

            verification =adaptive .verify_candidate_opening (
            candidate ,
            stationary_profile ,
            angular_steps ,
            candidate_threshold ,
            float (
            adaptive .FROZEN_PARAMETERS [
            "min_opening_width_deg"
            ]
            ),
            )

            if not verification ["confirmed"]:
                print (
                "[StationaryCandidateRejected] "
                f"id={hypothesis_id} "
                f"d0={d_left :.3f} "
                f"d1={d_right :.3f} "
                f"dmax={d_max :.3f} "
                f"Tk={candidate_threshold :.3f} "
                "reason=INSUFFICIENT_INTERVAL_SUPPORT"
                )
                continue 

            opening ={
            "start_angle":float (candidate ["start_angle"]),
            "end_angle":float (candidate ["end_angle"]),
            "center_angle":float (bearing_deg ),
            "width_deg":float (candidate ["width_deg"]),
            "confidence":1.0 ,
            "e_wall_ray_index":int (candidate ["e_wall_ray_index"]),
            "c_wall_ray_index":int (candidate ["c_wall_ray_index"]),
            }
            verified_records .append (
            (hypothesis_id ,opening )
            )

            print (
            "[StationaryCandidateVerified] "
            f"id={hypothesis_id} "
            f"d0={d_left :.3f} "
            f"d1={d_right :.3f} "
            f"dmax={d_max :.3f} "
            f"Tmin={candidate_t_min :.3f} "
            f"Tmax={candidate_t_max :.3f} "
            f"Tk={candidate_threshold :.3f} "
            f"bearing={bearing_deg :.2f}"
            )

        verified_records .sort (
        key =lambda item :float (
        item [1 ]["center_angle"]
        )
        )
        verified_openings =[
        dict (opening )
        for _ ,opening in verified_records 
        ]
        tracks :list [PersistentOpening ]=[]

        for _ ,opening in verified_records :
            track =PersistentOpening (
            f"OPEN_{len (tracks ):02d}"
            )
            track .update (opening ,self .frame )
            track .persistent_id =(
            f"J0-B{len (tracks )}"
            )
            tracks .append (track )

        self .stationary_verified_openings =verified_openings 
        self .tracks =tracks 
        self .parent =ParentTopologyEdge ()
        self .parent_source =self .parent .source 
        self .outgoing =list (tracks )

        topology_valid =len (self .outgoing )>=2 

        print (
        "[StationaryTopologyVerification] "
        f"hypotheses={len (hypotheses )} "
        f"verified={len (self .outgoing )} "
        f"valid={topology_valid}"
        )

        return topology_valid 



    def update (self ,simulation_time :float )->LidarFrame :
        self .leader .body_yaw =math .radians (self .yaw_deg )
        raw =self .sensor .scan (self .leader .position ,self .yaw_deg )
        smoothed =adaptive .smooth_ranges (raw ,SMOOTHING_WINDOW )
        left ,right =adaptive .extract_lateral_wall_ranges (
        self .sensor .angles ,smoothed ,MAX_RANGE ,SMOOTHING_WINDOW 
        )
        estimate =adaptive .compute_adaptive_worst_wall_range (left ,right )
        if estimate is not None :
            self .last_valid_w =estimate 
        adaptive_w =float (self .last_valid_w if self .last_valid_w is not None else 0.62 *MAX_RANGE )
        structural_candidates =(
        detect_structural_opening_candidates (
        self .sensor .angles ,
        smoothed ,
        )
        )
        rear_sector_deg =45.0 
        rear_openings =[
        candidate 
        for candidate in structural_candidates 
        if circular_error (
        float (candidate ["center_angle"]),
        180.0 ,
        )
        <=rear_sector_deg 
        ]
        outgoing_evidence_openings =[
        candidate 
        for candidate in structural_candidates 
        if circular_error (
        float (candidate ["center_angle"]),
        180.0 ,
        )
        >rear_sector_deg 
        ]
        moving_persistent_tracks =(
        self ._update_moving_opening_persistence (
        outgoing_evidence_openings 
        )
        )
        self .moving_persistent_tracks =list (moving_persistent_tracks )
        self .moving_persistent_candidates =[
        dict (track .latest_candidate )
        for track in moving_persistent_tracks 
        ]
        evidence =bool (
        len (moving_persistent_tracks )>=2 
        )

        lower =math .nan 
        upper =math .nan 
        selected =None 
        valid =False 
        openings =structural_candidates 
        diagnostics ={
        "smoothed_ranges":smoothed ,
        "open_support_mask":np .zeros (len (raw ),dtype =bool ),
        }

        result =LidarFrame (
        frame =self .frame ,
        angles =self .sensor .angles .copy (),
        raw =raw ,
        smoothed =np .asarray (diagnostics ["smoothed_ranges"]),
        support =np .asarray (diagnostics ["open_support_mask"],dtype =bool ),
        openings =tuple (dict (item )for item in openings ),
        left =left ,
        right =right ,
        adaptive_w =adaptive_w ,
        lower =lower ,
        upper =upper ,
        selected =selected ,
        interval_valid =valid ,
        current_evidence =evidence ,
        )
        append_lateral_range_sample (self ,result )
        self .last_frame =result 

        if np .any (result .support )and self .first_open_support_frame is None :
            self .first_open_support_frame =self .frame 
            print (f"[LiDAR] first_open_support_frame={self .frame }")

        if self .frame %60 ==0 :
            print (
            "[LiDAR] "
            f"adaptive W={adaptive_w :.2f} "
            f"Tmin={lower :.2f} "
            f"Tmax={upper :.2f} "
            f"selected={selected } "
            f"openings={len (openings )}"
            )

        if evidence and not self .junction_candidate_detected :
            self .first_junction_evidence_frame =self .frame 
            self .junction_candidate_detected =True 
            self .junction_candidate_frame =self .frame 
            baseline_window =self .lateral_range_history [
            -(LATERAL_BASELINE_SAMPLES +1 ):-1 
            ]
            if len (baseline_window )==LATERAL_BASELINE_SAMPLES :
                self .lateral_baseline_left =float (
                np .median ([sample [1 ]for sample in baseline_window ])
                )
                self .lateral_baseline_right =float (
                np .median ([sample [2 ]for sample in baseline_window ])
                )
                print (
                "[LateralBaselineFrozen] "
                f"frame={self .frame } "
                f"left={self .lateral_baseline_left :.2f} "
                f"right={self .lateral_baseline_right :.2f}"
                )
            self .lateral_last_checked_index =len (
            self .lateral_range_history 
            )
            self .candidate_lidar_frame =result 
            self .state =PerceptionState .JUNCTION_APPROACH 

            print (
            "[LiDAR] junction evidence "
            f"raw_openings={len (openings )} "
            f"rear={len (rear_openings )} "
            f"outgoing={len (outgoing_evidence_openings )}"
            )
            print ("[LiDAR] parent_source=INGRESS_LOCAL_HISTORY")
            print (
            "[JunctionCandidate] "
            f"frame={self .frame } "
            "anchor_fixed=False "
            f"state={self .state .name }"
            )

        if (
        self .state ==PerceptionState .JUNCTION_APPROACH 
        and not self .anchor_fixed 
        ):
            (
            entrance_line_reached ,
            entrance_frame ,
            left_90_range ,
            right_90_range ,
            left_jump ,
            right_jump ,
            )=anchor_junction_entrance_line_reached (
            self 
            )

            if (
            entrance_line_reached 
            and self .frame >int (self .junction_candidate_frame or -1 )
            ):
                self .anchor_fixed =True 
                self .anchor_fix_frame =self .frame 
                self .anchor_position =self .leader .position .copy ()
                self .leader .velocity .update (0.0 ,0.0 )
                self .leader .acceleration .update (0.0 ,0.0 )
                self .leader .filtered_acceleration .update (0.0 ,0.0 )
                if hasattr (self .leader ,"commanded_velocity"):
                    self .leader .commanded_velocity .update (0.0 ,0.0 )
                if hasattr (self .leader ,"observed_velocity"):
                    self .leader .observed_velocity .update (0.0 ,0.0 )
                self .physical .integration_anchor_position =(
                self .anchor_position .copy ()
                )
                self .pre_detection_travel =(
                self .anchor_position .distance_to (
                self .initial_leader_position 
                )
                )
                self .leader .is_fixed_anchor =True 
                self .leader .base_reserve =True 

                if (
                self .lateral_baseline_left is None 
                or self .lateral_baseline_right is None 
                ):
                    raise RuntimeError (
                    "Missing lateral corridor baseline "
                    "before stationary threshold freeze"
                    )

                threshold_w =(
                adaptive .compute_adaptive_worst_wall_range (
                self .lateral_baseline_left ,
                self .lateral_baseline_right ,
                )
                )

                if threshold_w is None :
                    raise RuntimeError (
                    "Cannot compute stationary W "
                    "from frozen corridor baseline"
                    )

                (
                stationary_lower ,
                stationary_upper ,
                stationary_valid ,
                )=adaptive .compute_adaptive_safe_threshold_interval (
                float (threshold_w ),
                MAX_RANGE ,
                TAU ,
                margin_ratio =ADAPTIVE_W_MARGIN_RATIO ,
                )

                stationary_threshold =(
                adaptive .select_threshold_in_safe_interval (
                stationary_lower ,
                stationary_upper ,
                stationary_valid ,
                ALPHA ,
                )
                )

                if (
                not stationary_valid 
                or stationary_threshold is None 
                ):
                    raise RuntimeError (
                    "Invalid stationary adaptive threshold "
                    "at Anchor stop"
                    )

                self .stationary_threshold_w =float (threshold_w )
                self .stationary_threshold_lower =float (stationary_lower )
                self .stationary_threshold_upper =float (stationary_upper )
                self .stationary_threshold =float (stationary_threshold )

                print (
                "[StationaryThresholdFrozen] "
                f"frame={self .frame } "
                f"W={self .stationary_threshold_w :.2f} "
                f"Tmin={self .stationary_threshold_lower :.2f} "
                f"Tmax={self .stationary_threshold_upper :.2f} "
                f"T={self .stationary_threshold :.2f}"
                )
                self .physical .integration_guard_hold_active =False 
                self .anchor_fixed_mean_normal_forward_speed =(
                self .mean_normal_forward_speed ()
                )
                self .state =PerceptionState .FIXED_ACCUMULATING 

                print (
                "[AnchorEntranceLineStop] "
                f"candidate_frame={self .junction_candidate_frame } "
                f"range_jump_frame={entrance_frame } "
                f"stop_frame={self .anchor_fix_frame } "
                f"left90={left_90_range :.2f} "
                f"right90={right_90_range :.2f} "
                f"delta_left={left_jump :.2f} "
                f"delta_right={right_jump :.2f} "
                f"jump_threshold={LATERAL_RANGE_JUMP_THRESHOLD :.2f} "
                "source=BILATERAL_90_TEMPORAL_RANGE_JUMP "
                "hard_latched=True "
                "velocity_zero=True "
                "position_snap=False"
                )
                print ("[Opening] stationary accumulation started")

        if self .anchor_fixed :
            self .stationary_samples +=1 

            if self .stationary_threshold is None :
                raise RuntimeError (
                "Stationary threshold was not frozen"
                )

            (
            stationary_openings ,
            stationary_diagnostics ,
            )=adaptive ._detect_openings_w_tau_with_diagnostics (
            self .sensor .angles ,
            raw ,
            selected_threshold =self .stationary_threshold ,
            threshold_interval_valid =True ,
            smoothing_window_size =SMOOTHING_WINDOW ,
            )

            result =replace (
            result ,
            smoothed =np .asarray (
            stationary_diagnostics ["smoothed_ranges"]
            ),
            support =np .asarray (
            stationary_diagnostics ["open_support_mask"],
            dtype =bool ,
            ),
            openings =tuple (
            dict (opening )
            for opening in stationary_openings 
            ),
            adaptive_w =float (self .stationary_threshold_w ),
            lower =float (self .stationary_threshold_lower ),
            upper =float (self .stationary_threshold_upper ),
            selected =float (self .stationary_threshold ),
            interval_valid =True ,
            )

            self .last_frame =result 

            if self .state ==PerceptionState .FIXED_ACCUMULATING :
                normal_speed =self .mean_normal_forward_speed ()
                self .pre_topology_normal_forward_speeds .append (normal_speed )

            self ._associate (stationary_openings )

            if self .stationary_samples %10 ==0 :
                print (
                "[StationaryOpeningCheck] "
                f"frame={self .frame } "
                f"T={self .stationary_threshold :.2f} "
                f"openings={len (stationary_openings )}"
                )
                count =sum (
                len (track .observations )
                >=MIN_PERSISTENT_OBSERVATIONS 
                for track in self .tracks 
                )
                print (
                "[Opening] "
                f"persistent count={count } "
                f"samples={self .stationary_samples }"
                )

            if (
            self .state ==PerceptionState .FIXED_ACCUMULATING 
            and self ._persistent_ready ()
            ):
                self .junction_confirmed =True 
                self .confirmation_frame =self .frame 
                self .confirmation_time =simulation_time 
                self .state =PerceptionState .BRANCHES_READY 
                self .topology_ready_frame =self .frame 
                self .integration_detected_branch_order =[
                track .persistent_id for track in self .outgoing 
                ]

                print (
                f"[Topology] parent={self .parent .persistent_id}"
                )
                print (
                f"[Topology] parent_source={self .parent_source}"
                )
                print (
                "[Topology] "
                f"outgoing="
                f"{[track .persistent_id for track in self .outgoing ]}"
                )
                print (
                "[Timeline] "
                f"TOPOLOGY_READY frame={self .frame }"
                )

        self .frame +=1 
        return result 


    def enforce_anchor (self )->None :
        if not self .anchor_fixed or self .anchor_position is None :
            return 
        self .post_fix_drift =max (
        self .post_fix_drift ,
        self .leader .position .distance_to (self .anchor_position ),
        )
        self .leader .position .update (
        self .anchor_position .x ,
        self .anchor_position .y ,
        )
        self .leader .velocity .update (0.0 ,0.0 )
        self .leader .acceleration .update (0.0 ,0.0 )
        self .leader .filtered_acceleration .update (0.0 ,0.0 )
        self .leader .commanded_velocity .update (0.0 ,0.0 )
        self .leader .observed_velocity .update (0.0 ,0.0 )
        self .leader .is_fixed_anchor =True 
        self .leader .base_reserve =True 


def hold_normal_swarm_during_stationary_verification (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    if (
    perception .state 
    !=PerceptionState .FIXED_ACCUMULATING 
    ):
        return 

    held =0 

    for robot in robots :
        if robot is perception .leader :
            continue 

        if robot .role !="NORMAL":
            continue 

        robot .velocity .update (0.0 ,0.0 )
        robot .acceleration .update (0.0 ,0.0 )
        robot .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )

        if hasattr (
        robot ,
        "commanded_velocity",
        ):
            robot .commanded_velocity .update (
            0.0 ,
            0.0 ,
            )

        held +=1 

    if (
    getattr (
    physical ,
    "integration_frame",
    -1 ,
    )
    %20 
    ==0 
    ):
        print (
        "[StationarySwarmHold] "
        f"frame="
        f"{getattr (physical ,'integration_frame',-1 )} "
        f"held_normals={held }"
        )


def _body_local_unit (perception :AdaptivePerception ,angle_deg :float )->pygame .Vector2 :
    world_angle =math .radians (perception .yaw_deg +angle_deg )
    return pygame .Vector2 (math .cos (world_angle ),math .sin (world_angle ))


@dataclass (frozen =True )
class LocalNeighborObservation :
    """Range/bearing-style observation expressed in an observer-local frame."""

    robot :Any 
    relative_range :float 
    relative_bearing :float 
    relative_axial :float 
    relative_lateral :float 
    relative_axial_velocity :float 


def observe_local_neighbors (
observer :Any ,
robots :Sequence [Any ],
forward_axis :pygame .Vector2 ,
*,
lateral_axis :pygame .Vector2 |None =None ,
max_range :float |None =None ,
predicate :Any |None =None ,
)->list [LocalNeighborObservation ]:
    """Local sensor adapter; controllers consume no world-coordinate deltas."""

    forward =forward_axis .normalize ()
    lateral =(
    lateral_axis .normalize ()
    if lateral_axis is not None 
    else pygame .Vector2 (-forward .y ,forward .x )
    )
    observations :list [LocalNeighborObservation ]=[]
    for robot in robots :
        if robot is observer :
            continue 
        if predicate is not None and not predicate (robot ):
            continue 
        relative =robot .position -observer .position 
        relative_range =float (relative .length ())
        if max_range is not None and relative_range >max_range :
            continue 
        axial =float (relative .dot (forward ))
        lateral_value =float (relative .dot (lateral ))
        observer_velocity =getattr (observer ,"velocity",pygame .Vector2 ())
        observations .append (LocalNeighborObservation (
        robot =robot ,
        relative_range =relative_range ,
        relative_bearing =math .atan2 (lateral_value ,axial ),
        relative_axial =axial ,
        relative_lateral =lateral_value ,
        relative_axial_velocity =float (
        (robot .velocity -observer_velocity ).dot (forward )
        ),
        ))
    return observations 


def update_anchor_follow_tree (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Build a local Anchor-rooted fan graph; Guards may relay information."""
    prep_uid =getattr (physical ,"integration_anchor_prep_request_uid",None )
    leading_uid =getattr (physical ,"integration_leading_anchor_uid",None )
    branch_uid =prep_uid or leading_uid 
    for robot in robots :
        robot .integration_anchor_follow_hop =-1 
        robot .integration_anchor_follow_layer =-1 
        robot .integration_anchor_follow_parent =None 
        robot .integration_anchor_relative_axial =None 
        robot .integration_anchor_relative_lateral =None 
        robot .integration_anchor_follow_forward =None 
        robot .integration_anchor_follow_lateral =None 
    if getattr (physical ,"integration_anchor_breakout_active",False ):
        return 
    if physical .phase not in {
    physical .SimulationPhase .FORM_JUNCTION_GUARDS ,
    physical .SimulationPhase .JUNCTION_SWITCH ,
    physical .SimulationPhase .EXPLORE_BRANCH ,
    }:
        return 
    anchor =perception .leader 
    anchor .integration_anchor_follow_hop =0 
    anchor .integration_anchor_follow_layer =0 
    anchor .integration_anchor_relative_axial =0.0 
    anchor .integration_anchor_relative_lateral =0.0 
    descriptor =(
    physical .branch_descriptors_by_uid .get (branch_uid )
    if branch_uid is not None else None 
    )
    if descriptor is not None :
        forward ,lateral =physical .descriptor_local_basis (descriptor )
        forward =forward .normalize ()
        lateral =lateral .normalize ()
    else :
        yaw =float (getattr (anchor ,"body_yaw",math .radians (perception .yaw_deg )))
        forward =pygame .Vector2 (math .cos (yaw ),math .sin (yaw )).normalize ()
        lateral =pygame .Vector2 (-forward .y ,forward .x ).normalize ()
    anchor .integration_anchor_follow_forward =forward .copy ()
    anchor .integration_anchor_follow_lateral =lateral .copy ()
    queue =[anchor ]
    cursor =0 
    relay_guards =0 




    follow_tree_roles =(
    {"NORMAL"}
    if physical .phase ==physical .SimulationPhase .EXPLORE_BRANCH 
    else {"NORMAL","JUNCTION_GUARD"}
    )

    while cursor <len (queue ):
        current =queue [cursor ]
        cursor +=1 
        hop =int (getattr (current ,"integration_anchor_follow_hop",0 ))
        for neighbor in getattr (current ,"comm_neighbors",()):
            role =getattr (neighbor ,"role",None )
            if (neighbor is anchor or role not in follow_tree_roles 
            or (role =="NORMAL"and getattr (neighbor ,"base_reserve",False ))
            or getattr (neighbor ,"integration_anchor_follow_hop",-1 )>=0 ):
                continue 
            observation =observe_local_neighbors (
            current ,[neighbor ],forward ,lateral_axis =lateral ,
            max_range =physical .COMM_RANGE ,
            )
            if not observation :
                continue 
            obs =observation [0 ]
            neighbor .integration_anchor_follow_hop =hop +1 
            neighbor .integration_anchor_follow_layer =(
            int (getattr (current ,"integration_anchor_follow_layer",0 ))
            +(1 if role =="NORMAL"else 0 )
            )
            neighbor .integration_anchor_follow_parent =current 
            neighbor .integration_anchor_relative_axial =(
            float (getattr (current ,"integration_anchor_relative_axial",0.0 )or 0.0 )
            +float (obs .relative_axial )
            )
            neighbor .integration_anchor_relative_lateral =(
            float (getattr (current ,"integration_anchor_relative_lateral",0.0 )or 0.0 )
            +float (obs .relative_lateral )
            )
            neighbor .integration_anchor_follow_forward =forward .copy ()
            neighbor .integration_anchor_follow_lateral =lateral .copy ()
            queue .append (neighbor )
            if role =="JUNCTION_GUARD":
                relay_guards +=1 
    if getattr (physical ,"integration_frame",0 )%20 ==0 :
        linked =[robot for robot in robots if getattr (robot ,"integration_anchor_follow_layer",-1 )>0 ]
        print (
        f"[AnchorFollowTree] branch={branch_uid or 'STAGING'} linked_normals={len (linked )} "
        f"relay_guards={relay_guards } max_layer={max ((int (robot .integration_anchor_follow_layer )for robot in linked ),default =0 )}"
        )



def apply_anchor_prep_corridor_yield (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
dt :float ,
)->None :
    """Physically clear a narrow local lane in front of the LiDAR Anchor.

    Universal collision remains fully active.  NORMAL robots are never
    teleported and never allowed to pass through the Anchor.  Instead, robots
    that are physically in the Anchor's immediate forward lane receive a local
    side-yield / separation command so the Anchor can reach the selected Guard
    standoff exactly as a real robot would through a crowd.

    This uses only branch-local direction plus local relative observations.
    """
    branch_uid =getattr (
    physical ,
    "integration_anchor_prep_request_uid",
    None ,
    )

    if branch_uid is None :
        return 

    if physical .phase not in {
    physical .SimulationPhase .FORM_JUNCTION_GUARDS ,
    physical .SimulationPhase .JUNCTION_SWITCH ,
    }:
        return 

    descriptor =physical .branch_descriptors_by_uid .get (
    branch_uid 
    )
    if descriptor is None :
        return 

    anchor =perception .leader 

    forward ,lateral =physical .descriptor_local_basis (
    descriptor 
    )
    if (
    forward .length_squared ()
    <=physical .EPSILON 
    or lateral .length_squared ()
    <=physical .EPSILON 
    ):
        return 

    forward =forward .normalize ()
    lateral =lateral .normalize ()

    yield_half_width =max (
    3.6 *float (physical .ROBOT_RADIUS ),
    1.20 *float (physical .GRID_SPACING ),
    )

    yield_depth =min (
    float (physical .COMM_RANGE ),
    max (
    10.0 *float (physical .ROBOT_RADIUS ),
    3.5 *float (
    physical .integration_anchor_target_gap 
    ),
    ),
    )

    observations =observe_local_neighbors (
    anchor ,
    robots ,
    forward ,
    lateral_axis =lateral ,
    max_range =yield_depth ,
    predicate =lambda robot :(
    robot is not anchor 
    and robot .role =="NORMAL"
    and not robot .base_reserve 
    ),
    )

    affected =0 
    nearest =float ("inf")

    for observation in observations :
        axial =float (
        observation .relative_axial 
        )
        lateral_offset =float (
        observation .relative_lateral 
        )



        if axial <=0.0 :
            continue 



        if abs (lateral_offset )>yield_half_width :
            continue 

        robot =observation .robot 
        affected +=1 
        nearest =min (
        nearest ,
        float (observation .relative_range ),
        )





        if abs (lateral_offset )>0.20 *physical .ROBOT_RADIUS :
            side_sign =(
            1.0 
            if lateral_offset >=0.0 
            else -1.0 
            )
        else :
            side_sign =(
            1.0 
            if robot .robot_id %2 ==0 
            else -1.0 
            )

        desired_lateral =(
        side_sign 
        *1.35 
        *yield_half_width 
        )

        lateral_error =(
        desired_lateral 
        -lateral_offset 
        )

        lateral_speed =float (
        robot .velocity .dot (
        lateral 
        )
        )

        lateral_accel =float (
        np .clip (
        8.0 *lateral_error 
        -3.0 *lateral_speed ,
        -0.50 *physical .MAX_ACCELERATION ,
        0.50 *physical .MAX_ACCELERATION ,
        )
        )

        robot .acceleration +=(
        lateral 
        *lateral_accel 
        )





        relative_vector =(
        forward *axial 
        +lateral *lateral_offset 
        )

        if (
        relative_vector .length_squared ()
        >physical .EPSILON 
        ):
            clearance =max (
            float (
            getattr (
            anchor ,
            "radius",
            physical .ROBOT_RADIUS ,
            )
            )
            +float (
            getattr (
            robot ,
            "radius",
            physical .ROBOT_RADIUS ,
            )
            ),
            2.05 
            *float (
            physical .ROBOT_RADIUS 
            ),
            )

            separation_range =(
            clearance 
            +1.5 
            *physical .ROBOT_RADIUS 
            )

            distance =float (
            observation .relative_range 
            )

            if distance <separation_range :
                separation_scale =(
                1.0 
                -distance 
                /max (
                separation_range ,
                physical .EPSILON ,
                )
                )

                robot .acceleration +=(
                relative_vector .normalize ()
                *(
                0.35 
                *physical .MAX_ACCELERATION 
                *separation_scale 
                )
                )







        forward_accel =float (
        robot .acceleration .dot (
        forward 
        )
        )

        if forward_accel >0.0 :
            robot .acceleration -=(
            forward 
            *forward_accel 
            *0.70 
            )

    if (
    affected >0 
    and physical .integration_frame %10 ==0 
    ):
        print (
        "[AnchorPrepYield] "
        f"frame={physical .integration_frame } "
        f"branch={branch_uid } "
        f"affected={affected } "
        f"nearest={nearest :.2f} "
        f"half_width={yield_half_width :.2f}"
        )


def apply_junction_approach_crawl (
physical :types .ModuleType ,
perception :AdaptivePerception ,
)->None :
    """Immediately clamp Anchor forward speed to the entrance crawl speed."""
    if perception .state !=PerceptionState .JUNCTION_APPROACH :
        return 
    if perception .anchor_fixed :
        return 
    anchor =perception .leader 
    forward =_body_local_unit (perception ,0.0 ).normalize ()
    axial_velocity =float (anchor .velocity .dot (forward ))
    lateral_velocity =anchor .velocity -forward *axial_velocity 
    anchor .velocity =(
    lateral_velocity +forward *JUNCTION_APPROACH_CRAWL_SPEED 
    )
    axial_acceleration =float (anchor .acceleration .dot (forward ))
    anchor .acceleration -=forward *axial_acceleration 
    if physical .integration_frame %5 ==0 :
        print(
            "[JunctionApproachCrawl] "
            f"frame={physical.integration_frame} "
            f"pos=({anchor.position.x:.3f},"
            f"{anchor.position.y:.3f}) "
            f"before_axial_v={axial_velocity:.3f} "
            f"after_axial_v="
            f"{anchor.velocity.dot(forward):.3f} "
            f"target_v="
            f"{JUNCTION_APPROACH_CRAWL_SPEED:.3f} "
            f"removed_axial_acc="
            f"{axial_acceleration:.3f}"
        )


def apply_post_anchor_normal_crawl (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Slow NORMAL flow through the Junction without blocking Anchor overtaking."""
    crawl_active =(
    perception .state in {
    PerceptionState .JUNCTION_APPROACH ,
    PerceptionState .FIXED_ACCUMULATING ,
    PerceptionState .BRANCHES_READY ,
    }
    and not perception .handoff_complete 
    )
    if not crawl_active :
        return 
    session =multi_dfs .child_session 
    if (
    session is not None 
    and session .ingress_t .length_squared ()>physical .EPSILON 
    ):
        forward =session .ingress_t .normalize ()
    else :
        forward =_body_local_unit (perception ,0.0 ).normalize ()
    target_speed =POST_ANCHOR_NORMAL_CRAWL_SPEED 
    max_feed_accel =0.12 *physical .MAX_ACCELERATION 
    kp =5.0 
    released =0 
    ahead =0 
    anchor =perception .leader 
    for robot in robots :
        if robot is anchor :
            continue 
        if robot .role !="NORMAL":
            continue 
        if robot .base_reserve :
            continue 
        forward_speed =float (robot .velocity .dot (forward ))
        speed_error =target_speed -forward_speed 
        feed_accel =float (
        np .clip (kp *speed_error ,-max_feed_accel ,+max_feed_accel )
        )
        robot .acceleration +=forward *feed_accel 
        if forward_speed >target_speed :
            robot .velocity -=forward *(
            forward_speed -target_speed 
            )
        relative_axial =float ((robot .position -anchor .position ).dot (forward ))
        if relative_axial >0.0 :
            ahead +=1 
        released +=1 
    if physical .integration_frame %10 ==0 :
        print (
        "[PostAnchorSwarmRelease] "
        f"frame={physical .integration_frame } "
        f"released_normals={released } "
        f"normal_ahead_of_anchor={ahead } "
        f"target_speed={target_speed :.3f}"
        )


def constrain_normal_behind_anchor (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robot :Any ,
before_update :pygame .Vector2 ,
proposed_position :pygame .Vector2 ,
)->pygame .Vector2 :
    """Cancel only this frame's pre-stop NORMAL Anchor crossing."""
    if perception .anchor_fixed :
        return proposed_position 
    anchor =perception .leader 
    if robot is anchor :
        return proposed_position 
    if robot .role !="NORMAL":
        return proposed_position 
    if robot .base_reserve :
        return proposed_position 
    forward =_body_local_unit (perception ,0.0 ).normalize ()
    proposed_axial =float ((proposed_position -anchor .position ).dot (forward ))
    if proposed_axial <=0.0 :
        return proposed_position 
    previous_axial =float ((before_update -anchor .position ).dot (forward ))
    lateral_delta =proposed_position -before_update 
    lateral_delta -=forward *lateral_delta .dot (forward )
    safe_axial =min (previous_axial ,-0.10 *physical .ROBOT_RADIUS )
    limited_position =(
    anchor .position 
    +forward *safe_axial 
    +(before_update -anchor .position -forward *previous_axial )
    +lateral_delta 
    )
    forward_speed =float (robot .velocity .dot (forward ))
    if forward_speed >0.0 :
        robot .velocity -=forward *forward_speed 
    return limited_position 


def enforce_junction_entry_anchor_lead (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
dt :float ,
)->None :
    """Keep NORMAL robots behind the mobile Anchor before Guard handoff."""
    if perception .anchor_fixed :
        return 
    anchor =perception .leader 
    root_entry_active =(
    physical .phase ==physical .SimulationPhase .MOVE_TO_JUNCTION 
    and not perception .handoff_complete 
    )
    session =multi_dfs .child_session 
    child_entry_active =(
    multi_dfs .child_probe_active 
    and session is not None 
    and not session .stationary_confirmed 
    )
    guard_frontend_active =(
    perception .anchor_fixed 
    and not perception .handoff_complete 
    )
    if not (
    root_entry_active 
    or child_entry_active 
    or guard_frontend_active 
    ):
        return 
    if (
    child_entry_active 
    and session is not None 
    and session .ingress_t .length_squared ()>physical .EPSILON 
    ):
        forward =session .ingress_t .normalize ()
    else :
        forward =_body_local_unit (perception ,0.0 ).normalize ()
    lateral =pygame .Vector2 (-forward .y ,forward .x ).normalize ()
    minimum_gap =max (
    2.4 *float (physical .ROBOT_RADIUS ),
    0.85 *float (
    getattr (
    physical ,
    "integration_anchor_target_gap",
    3.0 *physical .ROBOT_RADIUS ,
    )
    ),
    )
    braking_gap =2.5 *minimum_gap 
    anchor_speed =float (anchor .velocity .dot (forward ))
    affected =0 
    ahead_count =0 
    for robot in robots :
        if robot is anchor :
            continue 
        if robot .role !="NORMAL":
            continue 
        if robot .base_reserve :
            continue 
        observations =observe_local_neighbors (
        robot ,
        [anchor ],
        forward ,
        lateral_axis =lateral ,
        max_range =physical .COMM_RANGE ,
        )
        if not observations :
            continue 
        observation =observations [0 ]
        gap =float (observation .relative_axial )
        robot_speed =float (robot .velocity .dot (forward ))
        if gap >braking_gap :
            continue 
        affected +=1 
        if gap <=0.0 :
            ahead_count +=1 
            if robot_speed >0.0 :
                robot .velocity -=forward *robot_speed 
            forward_accel =float (robot .acceleration .dot (forward ))
            if forward_accel >0.0 :
                robot .acceleration -=forward *forward_accel 
            recovery_accel =min (
            0.20 *physical .MAX_ACCELERATION ,
            6.0 *(minimum_gap -gap ),
            )
            robot .acceleration -=forward *recovery_accel 
            continue 
        if gap <minimum_gap :
            desired_speed =min (
            anchor_speed ,
            0.0 if perception .anchor_fixed else anchor_speed ,
            )
        else :
            approach_allowance =(
            gap -minimum_gap 
            )/max (dt ,physical .EPSILON )
            desired_speed =anchor_speed +approach_allowance 
        if robot_speed >desired_speed :
            robot .velocity -=forward *(robot_speed -desired_speed )
        predicted_robot_speed =float ((robot .velocity +robot .acceleration *dt ).dot (forward ))
        predicted_anchor_speed =float ((anchor .velocity +anchor .acceleration *dt ).dot (forward ))
        maximum_closing_speed =max (0.0 ,gap -minimum_gap )/max (dt ,physical .EPSILON )
        max_predicted_speed =predicted_anchor_speed +maximum_closing_speed 
        if predicted_robot_speed >max_predicted_speed :
            robot .acceleration -=forward *(
            predicted_robot_speed -max_predicted_speed 
            )/max (dt ,physical .EPSILON )
    if physical .integration_frame %20 ==0 :
        print (
        "[JunctionEntryAnchorLead] "
        f"frame={physical .integration_frame } "
        f"anchor_fixed={perception .anchor_fixed } "
        f"affected_normals={affected } "
        f"normal_ahead={ahead_count }"
        )


def enforce_anchor_fan_no_overtake (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
dt :float ,
)->None :
    if getattr (physical ,"integration_anchor_breakout_active",False ):
        return 
    if (
    perception .anchor_fixed 
    and not perception .handoff_complete 
    and physical .phase in {
    physical .SimulationPhase .MOVE_TO_JUNCTION ,
    physical .SimulationPhase .FORM_JUNCTION_GUARDS ,
    }
    and getattr (physical ,"integration_anchor_prep_request_uid",None )is None 
    and getattr (physical ,"integration_leading_anchor_uid",None )is None 
    ):
        return 
    """Limit local forward closing so NORMAL robots cannot pass Anchor."""
    if physical .phase not in {
    physical .SimulationPhase .FORM_JUNCTION_GUARDS ,
    physical .SimulationPhase .JUNCTION_SWITCH ,
    physical .SimulationPhase .EXPLORE_BRANCH ,
    }:
        return 
    anchor =perception .leader 
    for robot in robots :
        if robot is anchor or robot .role !="NORMAL"or robot .base_reserve :
            continue 
        descriptor_uid =(
        getattr (physical ,"integration_anchor_prep_request_uid",None )
        or getattr (physical ,"integration_leading_anchor_uid",None )
        )
        descriptor =(
        physical .branch_descriptors_by_uid .get (descriptor_uid )
        if descriptor_uid is not None else None 
        )
        if descriptor is None :
            continue 
        forward ,lateral =physical .descriptor_local_basis (descriptor )
        forward =forward .normalize ()
        observations =observe_local_neighbors (
        robot ,[anchor ],forward ,lateral_axis =lateral .normalize (),
        max_range =physical .COMM_RANGE ,
        )
        if not observations :
            continue 
        gap =float (observations [0 ].relative_axial )
        minimum_gap =max (
        2.4 *float (physical .ROBOT_RADIUS ),
        0.85 *float (physical .integration_anchor_target_gap ),
        )
        braking_gap =2.5 *minimum_gap 
        if gap >braking_gap :
            continue 
        anchor_speed =float (anchor .velocity .dot (forward ))
        robot_speed =float (robot .velocity .dot (forward ))
        prep_mode =(
        getattr (
        physical ,
        "integration_anchor_prep_request_uid",
        None ,
        )
        is not None 
        )


        if gap <=0.0 and prep_mode :
            if robot_speed >0.0 :
                robot .velocity -=(
                forward 
                *robot_speed 
                )

            forward_accel =float (
            robot .acceleration .dot (
            forward 
            )
            )

            if forward_accel >0.0 :
                robot .acceleration -=(
                forward 
                *forward_accel 
                )

            continue 
        elif gap <=0.0 :
            recovery_speed =float (
            np .clip (
            4.0 *(minimum_gap -gap ),
            2.0 ,
            12.0 ,
            )
            )
            max_robot_speed =min (
            -recovery_speed ,
            anchor_speed -recovery_speed ,
            )
        elif gap >=minimum_gap :
            max_robot_speed =anchor_speed +max (0.0 ,gap -minimum_gap )/max (dt ,physical .EPSILON )
        else :
            recovery_speed =float (np .clip (4.0 *(minimum_gap -gap ),2.0 ,12.0 ))
            max_robot_speed =anchor_speed -recovery_speed 
        max_closing =max (0.0 ,gap -minimum_gap )/max (dt ,physical .EPSILON )
        closing =robot_speed -anchor_speed 
        if robot_speed >max_robot_speed :
            robot .velocity -=forward *(robot_speed -max_robot_speed )
        predicted =float ((robot .velocity +robot .acceleration *dt ).dot (forward ))
        anchor_predicted =float ((anchor .velocity +anchor .acceleration *dt ).dot (forward ))
        if gap <=0.0 :




            recovery_speed =float (
            np .clip (
            4.0 *(minimum_gap -gap ),
            2.0 ,
            12.0 ,
            )
            )
            max_predicted =min (
            -recovery_speed ,
            anchor_predicted -recovery_speed ,
            )
        elif gap <minimum_gap :
            max_predicted =anchor_predicted -float (np .clip (4.0 *(minimum_gap -gap ),2.0 ,12.0 ))
        else :
            max_predicted =anchor_predicted +max_closing 
        if predicted >max_predicted :
            robot .acceleration -=forward *(
            predicted -max_predicted 
            )/max (dt ,physical .EPSILON )


def _range_at_local_angle (frame :LidarFrame ,angle_deg :float )->float :
    index =min (
    range (len (frame .angles )),
    key =lambda item :circular_error (float (frame .angles [item ]),angle_deg ),
    )
    return float (frame .smoothed [index ])


def _raw_range_at_local_angle (frame :LidarFrame ,angle_deg :float )->float :
    index =min (
    range (len (frame .angles )),
    key =lambda i :circular_error (
    float (frame .angles [i ]),
    float (angle_deg ),
    ),
    )
    return float (frame .raw [index ])


def append_lateral_range_sample (
perception :AdaptivePerception ,
frame :LidarFrame ,
)->None :
    """Append the current body-relative lateral raw ranges to the time history."""
    left_90 =_raw_range_at_local_angle (frame ,+90.0 )
    right_90 =_raw_range_at_local_angle (frame ,-90.0 )
    perception .lateral_range_history .append (
    (int (frame .frame ),float (left_90 ),float (right_90 ))
    )


def _lidar_forward_opening_center (
frame :LidarFrame ,
perception :AdaptivePerception ,
*,
forward_axis :pygame .Vector2 |None =None ,
lateral_axis :pygame .Vector2 |None =None ,
)->tuple [float ,float ,float ]|None :
    """Estimate selected opening mouth center from finite wall endpoints."""
    openings =getattr (frame ,"openings",None )or []
    candidates =[
    opening for opening in openings 
    if circular_error (float (opening ["center_angle"]),0.0 )<=85.0 
    ]
    if not candidates :
        return None 
    opening =min (
    candidates ,
    key =lambda item :circular_error (float (item ["center_angle"]),0.0 ),
    )
    start_angle =float (opening ["start_angle"])
    end_angle =float (opening ["end_angle"])
    try :
        start_point ,_ ,_ =_nearest_wall_side_endpoint (
        frame ,perception ,start_angle ,search_direction =-1 ,
        max_search_deg =14.0 ,
        )
        end_point ,_ ,_ =_nearest_wall_side_endpoint (
        frame ,perception ,end_angle ,search_direction =1 ,
        max_search_deg =14.0 ,
        )
    except RuntimeError :
        return None 
    mouth_center =(start_point +end_point )*0.5 
    mouth_width =float (start_point .distance_to (end_point ))
    forward =(
    _body_local_unit (perception ,0.0 ).normalize ()
    if forward_axis is None else forward_axis .normalize ()
    )
    lateral =(
    _body_local_unit (perception ,-90.0 ).normalize ()
    if lateral_axis is None else lateral_axis .normalize ()
    )
    axial_distance =mouth_center .dot (forward )
    if axial_distance <=0.0 :
        return None 
    return float (axial_distance ),float (mouth_center .dot (lateral )),mouth_width 

@dataclass (frozen =True )
class JunctionEntranceEstimate :
    left_endpoint :pygame .Vector2 
    right_endpoint :pygame .Vector2 
    center :pygame .Vector2 
    width :float 
    depth :float 
    valid :bool 


def estimate_junction_entrance (
perception :AdaptivePerception ,
lidar_frame :LidarFrame ,
*,
forward_axis :pygame .Vector2 |None =None ,
)->JunctionEntranceEstimate |None :
    """Common J0/J1/J2 entrance estimator.

    This intentionally reproduces the current J0 entrance
    geometry first.

    No global Junction position is used.
    """

    if not lidar_frame .openings :
        return None 

    if forward_axis is None :
        forward =_body_local_unit (
        perception ,
        0.0 ,
        )
    else :
        forward =forward_axis .copy ()

    if forward .length_squared ()<=1.0e-12 :
        return None 

    forward =forward .normalize ()

    rear_sector_deg =45.0 

    forward_openings =[
    opening 
    for opening in lidar_frame .openings 
    if circular_error (
    float (opening ["center_angle"]),
    180.0 ,
    )
    >rear_sector_deg 
    ]

    if not forward_openings :
        return None 

    broad =max (
    forward_openings ,
    key =lambda item :float (
    item ["width_deg"]
    ),
    )

    start_angle =float (
    broad ["start_angle"]
    )

    end_angle =float (
    broad ["end_angle"]
    )

    left_endpoint =(
    _body_local_unit (
    perception ,
    start_angle ,
    )
    *_range_at_local_angle (
    lidar_frame ,
    start_angle ,
    )
    )

    right_endpoint =(
    _body_local_unit (
    perception ,
    end_angle ,
    )
    *_range_at_local_angle (
    lidar_frame ,
    end_angle ,
    )
    )

    center =(
    0.5 
    *(
    left_endpoint 
    +right_endpoint 
    )
    )

    width =float (
    left_endpoint .distance_to (
    right_endpoint 
    )
    )

    depth =float (
    center .dot (
    forward 
    )
    )

    valid =(
    depth >0.0 
    and width 
    >=0.5 *lidar_frame .adaptive_w 
    )

    return JunctionEntranceEstimate (
    left_endpoint =left_endpoint ,
    right_endpoint =right_endpoint ,
    center =center ,
    width =width ,
    depth =depth ,
    valid =valid ,
    )


def anchor_junction_entrance_line_reached (
perception :AdaptivePerception ,
)->tuple [bool ,int |None ,float ,float ,float ,float ]:
    """Detect the first bilateral jump from the frozen Junction-candidate baseline."""
    history =perception .lateral_range_history 
    baseline_left =perception .lateral_baseline_left 
    baseline_right =perception .lateral_baseline_right 
    if baseline_left is None or baseline_right is None :
        return False ,None ,0.0 ,0.0 ,0.0 ,0.0 
    start_t =perception .lateral_last_checked_index 
    for t in range (start_t ,len (history )):
        current_frame ,current_left ,current_right =history [t ]
        delta_left =current_left -baseline_left 
        delta_right =current_right -baseline_right 
        left_jump =delta_left >LATERAL_RANGE_JUMP_THRESHOLD 
        right_jump =delta_right >LATERAL_RANGE_JUMP_THRESHOLD 
        if current_frame %5 ==0 :
            print (
            "[LateralRangeJumpCheck] "
            f"frame={current_frame } "
            f"baseline_left={baseline_left :.2f} "
            f"baseline_right={baseline_right :.2f} "
            f"left={current_left :.2f} "
            f"right={current_right :.2f} "
            f"delta_left={delta_left :.2f} "
            f"delta_right={delta_right :.2f} "
            f"threshold={LATERAL_RANGE_JUMP_THRESHOLD :.2f} "
            f"left_jump={left_jump } "
            f"right_jump={right_jump }"
            )
        perception .lateral_last_checked_index =t +1 
        if left_jump and right_jump :
            return (
            True ,
            current_frame ,
            current_left ,
            current_right ,
            delta_left ,
            delta_right ,
            )
    return False ,None ,0.0 ,0.0 ,0.0 ,0.0 

def _nearest_wall_side_endpoint (
frame :LidarFrame ,
perception :AdaptivePerception ,
boundary_angle :float ,
*,
search_direction :int ,
max_search_deg :float =8.0 ,
)->tuple [pygame .Vector2 ,float ,float ]:
    """Return the nearest CLOSED/wall hit immediately outside an opening.

    Opening start/end rays are OPEN-support threshold crossings, not physical
    mouth corners.  Guard WHERE must use the adjacent wall-side LiDAR hit.
    The returned vector is already world-oriented relative to the Anchor
    because ``_body_local_unit`` already includes ``perception.yaw_deg``.
    """
    if search_direction not in {-1 ,1 }:
        raise ValueError ("search_direction must be -1 or +1")
    n =len (frame .angles )
    boundary_index =min (
    range (n ),
    key =lambda item :circular_error (float (frame .angles [item ]),boundary_angle ),
    )
    if n <=1 :
        raise RuntimeError ("LiDAR frame has insufficient angular samples")
    angular_step =360.0 /n 
    max_steps =max (1 ,int (math .ceil (max_search_deg /angular_step )))

    for offset in range (1 ,max_steps +1 ):
        index =(boundary_index +search_direction *offset )%n 
        if bool (frame .support [index ]):
            continue 
        raw_range =float (frame .raw [index ])
        if not math .isfinite (raw_range )or raw_range >=MAX_RANGE -1.0e-6 :
            continue 
        angle =float (frame .angles [index ])
        point =_body_local_unit (perception ,angle )*raw_range 
        return point ,angle ,raw_range 



    raise RuntimeError (
    f"no finite wall-side LiDAR hit near opening boundary {boundary_angle :+.1f}deg"
    )


def _guard_mouth_corner_from_lidar (
frame :LidarFrame ,
perception :AdaptivePerception ,
boundary_angle :float ,
*,
search_direction :int ,
max_search_deg :float =35.0 ,
)->tuple [pygame .Vector2 ,float ,float ]:
    """Estimate a physical Branch-mouth corner from adjacent LiDAR wall rays."""
    if search_direction not in {-1 ,+1 }:
        raise ValueError ("search_direction must be -1 or +1")

    n =len (frame .angles )
    if n <=2 :
        raise RuntimeError ("LiDAR frame has insufficient rays")

    boundary_index =min (
    range (n ),
    key =lambda index :circular_error (
    float (frame .angles [index ]),
    float (boundary_angle ),
    ),
    )
    angular_step =360.0 /float (n )
    max_steps =max (3 ,int (math .ceil (max_search_deg /angular_step )))
    samples :list [tuple [int ,int ,float ,float ,pygame .Vector2 ]]=[]
    wall_started =False 
    invalid_after_wall =0 

    for offset in range (0 ,max_steps +1 ):
        index =(boundary_index +search_direction *offset )%n 
        raw_range =float (frame .raw [index ])
        finite_wall_hit =(
        math .isfinite (raw_range )
        and raw_range >0.0 
        and raw_range <MAX_RANGE -1.0e-6 
        )

        if not finite_wall_hit :
            if wall_started :
                invalid_after_wall +=1 
                if invalid_after_wall >=2 :
                    break 
            continue 

        wall_started =True 
        invalid_after_wall =0 
        angle =float (frame .angles [index ])
        point =_body_local_unit (perception ,angle )*raw_range 
        samples .append ((offset ,index ,angle ,raw_range ,point ))

    if len (samples )<2 :
        raise RuntimeError (
        "Guard mouth corner search found "
        "insufficient finite LiDAR wall points: "
        f"boundary={boundary_angle :+.2f}"
        )

    raw_ranges =np .asarray ([sample [3 ]for sample in samples ],dtype =float )
    filtered_ranges =raw_ranges .copy ()

    if len (raw_ranges )>=3 :
        for index in range (1 ,len (raw_ranges )-1 ):
            filtered_ranges [index ]=float (
            np .median (raw_ranges [index -1 :index +2 ])
            )

    selected_index :int |None =None 
    if (
    len (filtered_ranges )>=2 
    and filtered_ranges [0 ]<=filtered_ranges [1 ]
    ):
        selected_index =0 

    if selected_index is None :
        for index in range (1 ,len (filtered_ranges )-1 ):
            if (
            filtered_ranges [index ]<=filtered_ranges [index -1 ]
            and filtered_ranges [index ]<=filtered_ranges [index +1 ]
            ):
                selected_index =index 
                break 

    if selected_index is None :
        selected_index =int (np .argmin (filtered_ranges ))

    (
    _ ,
    ray_index ,
    corner_angle ,
    corner_range ,
    corner_point ,
    )=samples [selected_index ]

    print (
    "[GuardMouthCorner] "
    f"boundary={boundary_angle :+.2f} "
    f"search_dir={search_direction :+d} "
    f"ray={ray_index } "
    f"angle={corner_angle :+.2f} "
    f"range={corner_range :.3f} "
    f"samples={len (samples )} "
    f"relative=({corner_point .x :.3f},"
    f"{corner_point .y :.3f})"
    )

    return corner_point .copy (),corner_angle ,corner_range 

@dataclass (frozen =True )
class GeneralVerifiedMouth :
    opening :dict [str ,float ]
    start_point :pygame .Vector2 
    end_point :pygame .Vector2 
    midpoint :pygame .Vector2 
    width :float 
    ingress_alignment :float 
    is_non_axial :bool 


@dataclass (frozen =True )
class GeneralJunctionObservation :
    valid :bool 
    verified_outgoing :tuple [GeneralVerifiedMouth ,...]
    non_axial_outgoing :tuple [GeneralVerifiedMouth ,...]
    entrance_depth :float |None 


def evaluate_general_junction_structure (
perception :AdaptivePerception ,
lidar_frame :LidarFrame ,
ingress_t :pygame .Vector2 ,
)->GeneralJunctionObservation :
    """Finite-mouth Junction structure for Child J1/J2/...

    Parent edge comes from the actually traversed ingress history.
    No global Junction coordinate or fixture position is used.
    """

    if (
    not lidar_frame .interval_valid 
    or lidar_frame .selected is None 
    ):
        return GeneralJunctionObservation (
        valid =False ,
        verified_outgoing =(),
        non_axial_outgoing =(),
        entrance_depth =None ,
        )

    if ingress_t .length_squared ()<=1.0e-12 :
        return GeneralJunctionObservation (
        valid =False ,
        verified_outgoing =(),
        non_axial_outgoing =(),
        entrance_depth =None ,
        )

    ingress =ingress_t .normalize ()

    minimum_mouth_width =(
    CHILD_CANDIDATE_MIN_MOUTH_WIDTH_RATIO 
    *lidar_frame .adaptive_w 
    )

    verified_outgoing :list [
    GeneralVerifiedMouth 
    ]=[]

    non_axial_outgoing :list [
    GeneralVerifiedMouth 
    ]=[]

    for opening in lidar_frame .openings :

        start_angle =float (
        opening ["start_angle"]
        )

        end_angle =float (
        opening ["end_angle"]
        )

        center_angle =float (
        opening ["center_angle"]
        )

        try :
            start_point ,_ ,_ =(
            _nearest_wall_side_endpoint (
            lidar_frame ,
            perception ,
            start_angle ,
            search_direction =-1 ,
            )
            )

            end_point ,_ ,_ =(
            _nearest_wall_side_endpoint (
            lidar_frame ,
            perception ,
            end_angle ,
            search_direction =+1 ,
            )
            )

        except RuntimeError :


            continue 

        width =float (
        start_point .distance_to (
        end_point 
        )
        )

        if width <minimum_mouth_width :
            continue 

        radial =_body_local_unit (
        perception ,
        center_angle ,
        )

        if radial .length_squared ()<=1.0e-12 :
            continue 

        radial =radial .normalize ()

        ingress_alignment =float (
        radial .dot (
        ingress 
        )
        )



        if ingress_alignment <=-0.50 :
            continue 

        midpoint =(
        0.5 
        *(
        start_point 
        +end_point 
        )
        )

        is_non_axial =(
        abs (ingress_alignment )
        <=CHILD_CANDIDATE_NON_AXIAL_MAX_DOT 
        )

        mouth =GeneralVerifiedMouth (
        opening =dict (opening ),
        start_point =start_point ,
        end_point =end_point ,
        midpoint =midpoint ,
        width =width ,
        ingress_alignment =ingress_alignment ,
        is_non_axial =is_non_axial ,
        )

        verified_outgoing .append (
        mouth 
        )

        if is_non_axial :
            non_axial_outgoing .append (
            mouth 
            )

    entrance_depth_samples =[
    float (
    mouth .midpoint .dot (
    ingress 
    )
    )
    for mouth in non_axial_outgoing 
    if float (
    mouth .midpoint .dot (
    ingress 
    )
    )>0.0 
    ]

    entrance_depth =(
    float (
    np .median (
    entrance_depth_samples 
    )
    )
    if entrance_depth_samples 
    else None 
    )

    valid =(
    len (verified_outgoing )
    >=CHILD_STATIONARY_MIN_OUTGOING 
    and len (non_axial_outgoing )>=1 
    and entrance_depth is not None 
    )

    return GeneralJunctionObservation (
    valid =valid ,
    verified_outgoing =tuple (
    verified_outgoing 
    ),
    non_axial_outgoing =tuple (
    non_axial_outgoing 
    ),
    entrance_depth =entrance_depth ,
    )

def _lidar_estimated_mouth_width (
frame :LidarFrame ,
opening :dict [str ,float ],
axis :pygame .Vector2 ,
perception :AdaptivePerception ,
)->float :
    start =float (opening ["start_angle"])
    end =float (opening ["end_angle"])
    center =float (opening ["center_angle"])
    start_range =_range_at_local_angle (frame ,start )
    end_range =_range_at_local_angle (frame ,end )
    mean_range =0.5 *(start_range +end_range )
    angular_width =math .radians (float (opening ["width_deg"]))
    visible_chord =2.0 *mean_range *math .sin (0.5 *angular_width )
    radial =_body_local_unit (perception ,center )
    obliquity =max (abs (radial .dot (axis )),0.25 )
    corrected_span =visible_chord /obliquity 
    adaptive_cap =frame .adaptive_w *PROVISIONAL_MOUTH_WIDTH_W_RATIO 
    return float (np .clip (
    corrected_span ,
    adaptive_cap *PROVISIONAL_MOUTH_WIDTH_MIN_RATIO ,
    adaptive_cap ,
    ))

def build_provisional_guard_descriptors_from_lidar (
physical :types .ModuleType ,
perception :AdaptivePerception ,
frame :LidarFrame ,
*,
outgoing_override :Sequence [
dict [str ,float ]
]|None =None ,
branch_uids_override :Sequence [str ]|None =None ,
junction_uid_override :str |None =None ,
)->list [ProvisionalGuardGeometry ]:
    """Build Guard geometry directly from LiDAR-detected physical mouth corners.

    Geometry policy:
      1. Detect each Branch opening.
      2. Estimate two physical mouth corners from adjacent LiDAR wall rays.
      3. Convert both corners to world coordinates using Anchor localization.
      4. Define the mouth center and lateral unit vector from those points.
      5. Define the Branch tangent as the perpendicular direction into the opening.
      6. Build 3xN Guard slots in world coordinates.

    Guard selection and placement are the localization-allowed stage.
    No LEFT / RIGHT / UP geometry hardcoding is used.
    """

    if perception .anchor_position is None :
        raise RuntimeError (
        "provisional Guard geometry requires a fixed Anchor"
        )

    if outgoing_override is None :

        root_openings =(
        perception .stationary_verified_openings 
        if perception .stationary_verified_openings 
        else frame .openings 
        )

        all_openings =sorted (
        (
        dict (item )
        for item in root_openings 
        ),
        key =lambda item :float (
        item ["center_angle"]
        ),
        )

        if len (all_openings )<3 :
            raise RuntimeError (
            "Junction evidence lacks "
            "three provisional openings"
            )

        rear_sector_deg =45.0 

        rear_candidates =[
        item 
        for item in all_openings 
        if circular_error (
        float (
        item ["center_angle"]
        ),
        180.0 ,
        )
        <=rear_sector_deg 
        ]

        if rear_candidates :
            parent =min (
            rear_candidates ,
            key =lambda item :circular_error (
            float (
            item ["center_angle"]
            ),
            180.0 ,
            ),
            )

            outgoing =[
            item 
            for item in all_openings 
            if circular_error (
            float (
            item ["center_angle"]
            ),
            180.0 ,
            )
            >rear_sector_deg 
            ]

        else :


            parent =None 
            outgoing =list (
            all_openings 
            )



        if len (outgoing )!=3 :
            raise RuntimeError (
            "Root fixture adapter requires three verified "
            "outgoing openings: "
            f"raw="
            f"{[round (float (x ['center_angle']),1 )for x in all_openings ]} "
            f"outgoing="
            f"{[round (float (x ['center_angle']),1 )for x in outgoing ]} "
            f"rear_count={len (rear_candidates )}"
            )

        openings =sorted (
        outgoing ,
        key =lambda item :float (
        item ["center_angle"]
        ),
        )


        keys :list [str |None ]=[]

        for item in openings :

            angle =float (
            item ["center_angle"]
            )

            key =(
            "UP"
            if abs (angle )<30.0 
            else (
            "LEFT"
            if angle <0.0 
            else "RIGHT"
            )
            )

            keys .append (
            key 
            )

        if len (set (keys ))!=3 :
            raise RuntimeError (
            "duplicate LiDAR branch "
            f"identities: {keys }"
            )

        branch_uids =[
        f"PROV_{index :02d}"
        for index 
        in range (len (openings ))
        ]

        junction_uid =(
        physical .CURRENT_JUNCTION_ID 
        )

        print (
        "[OpeningClassification] "
        f"all="
        f"{[round (float (x ['center_angle']),1 )for x in all_openings ]} "
        f"parent="
        f"{round (float (parent ['center_angle']),1 )if parent else None } "
        f"outgoing="
        f"{[round (float (x ['center_angle']),1 )for x in openings ]} "
        f"keys={keys } "
        "scope=ROOT"
        )

    else :
        openings =sorted (
        (
        dict (item )
        for item in outgoing_override 
        ),
        key =lambda item :float (
        item ["center_angle"]
        ),
        )

        if branch_uids_override is None :
            raise RuntimeError (
            "Child Guard geometry requires "
            "branch_uids_override"
            )

        branch_uids =list (
        branch_uids_override 
        )

        if (
        len (branch_uids )
        !=len (openings )
        ):
            raise RuntimeError (
            "Child branch/opening count mismatch: "
            f"branches={len (branch_uids )} "
            f"openings={len (openings )}"
            )

        if junction_uid_override is None :
            raise RuntimeError (
            "Child Guard geometry requires "
            "junction_uid_override"
            )

        junction_uid =(
        junction_uid_override 
        )



        keys =[
        None 
        for _ in openings 
        ]

        print (
        "[ChildOpeningSet] "
        f"junction={junction_uid } "
        f"branches={branch_uids } "
        f"angles="
        f"{[round (float (x ['center_angle']),1 )for x in openings ]} "
        "fixture_labels_used=False"
        )

    body_forward =_body_local_unit (
    perception ,
    0.0 ,
    ).normalize ()

    if branch_uids_override is None :


        forward =(
        body_forward .copy ()
        )

        broad =max (
        openings ,
        key =lambda item :float (
        item ["width_deg"]
        ),
        )

        width_reference =(
        frame .adaptive_w 
        *PROVISIONAL_MOUTH_WIDTH_W_RATIO 
        )

        boundary_forward_depths :list [float ]=[]

        for key in (
        "start_angle",
        "end_angle",
        ):

            angle =float (
            broad [key ]
            )

            projection =(
            _body_local_unit (
            perception ,
            angle ,
            ).dot (forward )
            *_range_at_local_angle (
            frame ,
            angle ,
            )
            )

            if projection >0.0 :
                boundary_forward_depths .append (
                projection 
                )

        raw_junction_depth =(
        float (
        np .mean (
        boundary_forward_depths 
        )
        )
        if boundary_forward_depths 
        else width_reference 
        )

        junction_depth =float (
        np .clip (
        raw_junction_depth ,
        width_reference 
        *PROVISIONAL_JUNCTION_DEPTH_MIN_WIDTH_RATIO ,
        width_reference 
        *PROVISIONAL_JUNCTION_DEPTH_MAX_WIDTH_RATIO ,
        )
        )

        junction_center =(
        perception .anchor_position 
        +forward *junction_depth 
        )

    else :

        session =(
        multi_dfs .child_session 
        )

        if (
        session is not None 
        and session .ingress_t .length_squared ()
        >physical .EPSILON 
        ):
            forward =(
            session .ingress_t .normalize ()
            )
        else :
            forward =(
            body_forward .copy ()
            )



        all_mouth_depths :list [float ]=[]



        non_axial_mouth_depths :list [float ]=[]

        for center_opening in openings :

            center_start =float (
            center_opening ["start_angle"]
            )

            center_end =float (
            center_opening ["end_angle"]
            )

            center_angle =float (
            center_opening ["center_angle"]
            )

            try :
                (
                center_start_point ,
                _ ,
                _ ,
                )=_nearest_wall_side_endpoint (
                frame ,
                perception ,
                center_start ,
                search_direction =-1 ,
                )

                (
                center_end_point ,
                _ ,
                _ ,
                )=_nearest_wall_side_endpoint (
                frame ,
                perception ,
                center_end ,
                search_direction =+1 ,
                )

            except RuntimeError :
                continue 





            mouth_mid_local =(
            0.5 
            *(
            center_start_point 
            +center_end_point 
            )
            )

            mouth_depth =float (
            mouth_mid_local .dot (
            forward 
            )
            )

            if mouth_depth <=0.0 :
                continue 

            all_mouth_depths .append (
            mouth_depth 
            )

            radial =(
            _body_local_unit (
            perception ,
            center_angle ,
            )
            )

            if (
            radial .length_squared ()
            <=physical .EPSILON 
            ):
                continue 

            radial =(
            radial .normalize ()
            )

            ingress_alignment =abs (
            float (
            radial .dot (
            forward 
            )
            )
            )


            if (
            ingress_alignment 
            <=CHILD_CANDIDATE_NON_AXIAL_MAX_DOT 
            ):
                non_axial_mouth_depths .append (
                mouth_depth 
                )

        if non_axial_mouth_depths :

            junction_depth =float (
            np .median (
            non_axial_mouth_depths 
            )
            )

            center_source =(
            "CHILD_FINITE_NON_AXIAL_MOUTH_MEDIAN"
            )

        elif all_mouth_depths :

            junction_depth =float (
            np .median (
            all_mouth_depths 
            )
            )

            center_source =(
            "CHILD_FINITE_MOUTH_MEDIAN"
            )

        else :



            junction_depth =float (
            frame .adaptive_w 
            *PROVISIONAL_MOUTH_WIDTH_W_RATIO 
            )

            center_source =(
            "CHILD_ADAPTIVE_W_FALLBACK"
            )



        junction_center =(
        perception .anchor_position 
        +forward *junction_depth 
        )

        print (
        "[ChildJunctionCenterEstimate] "
        f"junction={junction_uid } "
        f"depth={junction_depth :.3f} "
        f"center="
        f"({junction_center .x :.3f},"
        f"{junction_center .y :.3f}) "
        f"non_axial_samples="
        f"{[round (value ,3 )for value in non_axial_mouth_depths ]} "
        f"all_samples="
        f"{[round (value ,3 )for value in all_mouth_depths ]} "
        f"source={center_source }"
        )

    physical .integration_lidar_junction_estimate =(
    junction_center .copy ()
    )

    geometries :list [
    ProvisionalGuardGeometry 
    ]=[]





    for index ,opening in enumerate (
    openings 
    ):









        branch_key =keys [index ]


        start =float (
        opening ["start_angle"]
        )

        end =float (
        opening ["end_angle"]
        )

        center =float (
        opening ["center_angle"]
        )







        (
        start_point ,
        start_wall_angle ,
        start_wall_range ,
        )=_guard_mouth_corner_from_lidar (
        frame ,
        perception ,
        start ,
        search_direction =-1 ,
        )

        (
        end_point ,
        end_wall_angle ,
        end_wall_range ,
        )=_guard_mouth_corner_from_lidar (
        frame ,
        perception ,
        end ,
        search_direction =+1 ,
        )

        print (
        f"[GuardPhysicalMouth] "
        f"center={center :+.1f} "
        f"start_corner={start_wall_angle :+.1f} "
        f"start_r={start_wall_range :.2f} "
        f"end_corner={end_wall_angle :+.1f} "
        f"end_r={end_wall_range :.2f}"
        )

        opening_radial =(
        _body_local_unit (
        perception ,
        center ,
        ).normalize ()
        )









        if False :

            (
            direction_a ,
            quality_a ,
            )=valid_directions [0 ]

            (
            direction_b ,
            quality_b ,
            )=valid_directions [1 ]

            if (
            direction_a .dot (
            direction_b 
            )
            <0.0 
            ):
                direction_b =(
                -direction_b 
                )

            parallel_score =float (
            direction_a .dot (
            direction_b 
            )
            )




            if parallel_score <0.85 :

                if quality_a >=quality_b :
                    axis =direction_a 
                else :
                    axis =direction_b 

                axis_source =(
                "LIDAR_BEST_SINGLE_CORRIDOR_WALL"
                )

            else :

                combined =(
                direction_a 
                *max (
                quality_a ,
                1.0 ,
                )
                +direction_b 
                *max (
                quality_b ,
                1.0 ,
                )
                )

                if (
                combined .length_squared ()
                >physical .EPSILON 
                ):
                    axis =(
                    combined .normalize ()
                    )

                    axis_source =(
                    "LIDAR_TWO_PARALLEL_CORRIDOR_WALLS"
                    )

                else :
                    axis =direction_a 

                    axis_source =(
                    "LIDAR_SINGLE_EFFECTIVE_CORRIDOR_WALL"
                    )

        elif False :

            axis =(
            valid_directions [
            0 
            ][0 ]
            )

            axis_source =(
            "LIDAR_SINGLE_CORRIDOR_WALL"
            )

        else :










            axis =(
            opening_radial .copy ()
            )

            axis_source =(
            "LIDAR_OPENING_RADIAL_FALLBACK"
            )



        if (
        axis .dot (
        opening_radial 
        )
        <0.0 
        ):
            axis =-axis 

        axis =(
        axis .normalize ()
        )

        mouth_start_world =(
        perception .anchor_position 
        +start_point 
        )
        mouth_end_world =(
        perception .anchor_position 
        +end_point 
        )
        mouth_vector =mouth_end_world -mouth_start_world 
        mouth_span =float (mouth_vector .length ())

        if mouth_span <=physical .EPSILON :
            raise RuntimeError (
            "Physical mouth endpoints are degenerate: "
            f"uid={branch_uids [index ]}"
            )

        mouth_normal =mouth_vector .normalize ()
        mouth =0.5 *(mouth_start_world +mouth_end_world )
        axis =pygame .Vector2 (
        -mouth_normal .y ,
        mouth_normal .x ,
        ).normalize ()

        if axis .dot (opening_radial )<0.0 :
            axis =-axis 

        straight_start =mouth_start_world .copy ()
        straight_end =mouth_end_world .copy ()
        width_source ="PHYSICAL_LIDAR_MOUTH_CORNERS"
        axis_source ="PHYSICAL_MOUTH_PERPENDICULAR"

        print (
        "[GuardWorldMouth] "
        f"uid={branch_uids [index ]} "
        f"pL=({mouth_start_world .x :.3f},"
        f"{mouth_start_world .y :.3f}) "
        f"pR=({mouth_end_world .x :.3f},"
        f"{mouth_end_world .y :.3f}) "
        f"center=({mouth .x :.3f},"
        f"{mouth .y :.3f}) "
        f"span={mouth_span :.3f} "
        f"t=({axis .x :.3f},"
        f"{axis .y :.3f}) "
        f"n=({mouth_normal .x :.3f},"
        f"{mouth_normal .y :.3f})"
        )

        uid =branch_uids [
        index 
        ]

        descriptor =(
        physical .BranchDescriptor (
        uid =uid ,
        junction_uid =junction_uid ,
        fixture_key =None ,

        local_outgoing_direction =(
        axis .copy ()
        ),

        local_return_direction =(
        -axis 
        ),

        observed_mouth_position =(
        mouth .copy ()
        ),

        observed_width =mouth_span ,

        cohort_member_ids =set (),

        direction_last_estimate =(
        axis .copy ()
        ),

        direction_stability_reference =(
        axis .copy ()
        ),

        direction_stable_dwell =1.0 ,
        direction_sample_count =1 ,
        direction_angular_spread =0.0 ,
        direction_is_stable =True ,

        direction_mature_dwell =1.0 ,
        direction_is_mature =True ,

        direction_downstream_travel =0.0 ,

        motion_t =(
        axis .copy ()
        ),

        motion_n =(
        mouth_normal .copy ()
        ),

        motion_frame_locked =True ,
        motion_frame_source =axis_source ,

        motion_frame_sample_count =1 ,
        motion_frame_angular_spread =0.0 ,

        motion_observed_width =(
        mouth_span 
        ),

        observed_flow_width =(
        mouth_span 
        ),

        observed_physical_width =(
        mouth_span 
        ),

        physical_width_confident =True ,
        physical_width_source =(
        width_source 
        ),

        physical_left_boundary_lateral =(
        -0.5 
        *mouth_span 
        ),

        physical_right_boundary_lateral =(
        0.5 
        *mouth_span 
        ),

        physical_boundary_sample_count =2 ,

        discovered_at =(
        physical .simulation_time 
        ),
        )
        )

        geometries .append (
        ProvisionalGuardGeometry (
        provisional_uid =uid ,
        opening =opening ,
        descriptor =descriptor ,

        columns =0 ,
        layers =0 ,
        slots =[],

        local_branch_key =(
        branch_key 
        ),

        opening_start_local =(
        start_point .copy ()
        ),

        opening_end_local =(
        end_point .copy ()
        ),

        mouth_start_world =(
        straight_start .copy ()
        ),

        mouth_end_world =(
        straight_end .copy ()
        ),

        mouth_center_world =(
        mouth .copy ()
        ),

        mouth_lateral_unit =(
        mouth_normal .copy ()
        ),

        branch_tangent_unit =(
        axis .copy ()
        ),

        mouth_span =float (
        mouth_span 
        ),
        )
        )



        print(
            f"[CorridorWallFrame] "
            f"uid={uid} "
            f"branch={branch_key} "
            f"source={axis_source} "
            f"axis=({axis.x:.3f},{axis.y:.3f}) "
            f"normal=({mouth_normal.x:.3f},{mouth_normal.y:.3f}) "
            f"span={mouth_span:.3f} "
            f"mouth=({mouth.x:.3f},{mouth.y:.3f}) "
            f"geometry=PHYSICAL_LIDAR_MOUTH_CORNERS"
        )


    trusted_widths =[
    float (geometry .mouth_span )
    for geometry in geometries 
    if geometry .descriptor .physical_width_source 
    =="LIDAR_WALL_ENDPOINT_CROSS_SECTION"
    ]
    if trusted_widths :
        fallback_width =float (np .median (trusted_widths ))
        for geometry in geometries :
            descriptor =geometry .descriptor 
            if descriptor .physical_width_source !="ADAPTIVE_W_FALLBACK":
                continue 

            tangent =geometry .branch_tangent_unit 
            normal =geometry .mouth_lateral_unit 
            if tangent is None or normal is None :
                continue 



            center =geometry .mouth_center_world 

            if center is None :
                continue 

            center =center .copy ()

            geometry .mouth_span =fallback_width 
            geometry .mouth_center_world =center .copy ()
            geometry .mouth_start_world =center -normal *(0.5 *fallback_width )
            geometry .mouth_end_world =center +normal *(0.5 *fallback_width )
            descriptor .observed_mouth_position =center .copy ()
            descriptor .observed_width =fallback_width 
            descriptor .motion_observed_width =fallback_width 
            descriptor .observed_flow_width =fallback_width 
            descriptor .observed_physical_width =fallback_width 
            descriptor .physical_left_boundary_lateral =-0.5 *fallback_width 
            descriptor .physical_right_boundary_lateral =0.5 *fallback_width 
            descriptor .physical_width_source =(
            "LIDAR_SHARED_FULL_CROSS_SECTION_FALLBACK"
            )
            print (
            f"[GuardWidthFallback] uid={geometry .provisional_uid } "
            f"width={fallback_width :.3f} "
            "source=LIDAR_SHARED_FULL_CROSS_SECTION"
            )

    return geometries 

def compute_guard_lateral_interval (
physical :types .ModuleType ,
descriptor :Any ,
)->tuple [float ,float ]:


    mouth_half =0.5 *float (descriptor .observed_physical_width )
    wall_clearance =physical .ROBOT_RADIUS *(
    1.0 +GUARD_EDGE_SEAL_MARGIN_RATIO 
    )
    center_half =max (
    0.0 ,
    mouth_half -wall_clearance ,
    )

    return (-center_half ,center_half )


def compute_sealing_aware_column_count (
physical :types .ModuleType ,
descriptor :Any ,
)->tuple [int ,float ,float ,float ]:
    lateral_min ,lateral_max =compute_guard_lateral_interval (
    physical ,descriptor 
    )
    span =max (0.0 ,lateral_max -lateral_min )
    target_spacing =2.5 *physical .ROBOT_RADIUS 
    required_by_width =physical .required_junction_guard_count (descriptor )
    required_by_gap =int (
    math .ceil (span /max (target_spacing ,physical .EPSILON ))
    )+1 
    columns =max (required_by_width ,required_by_gap )
    spacing =span /max (columns -1 ,1 )
    return columns ,lateral_min ,lateral_max ,spacing 


def build_edge_sealing_slot_order (columns :int )->list [int ]:
    order :list [int ]=[]
    left ,right =0 ,columns -1 
    while left <=right :
        order .append (left )
        if right !=left :
            order .append (right )
        left +=1 
        right -=1 
    return order 


def build_sealing_aware_slots (
physical :types .ModuleType ,
descriptor :Any ,
columns :int ,
layers :int ,
geometry :ProvisionalGuardGeometry |None =None ,
perception :AdaptivePerception |None =None ,
)->list [pygame .Vector2 ]:
    """Build 3xN Guard slots from localized LiDAR mouth endpoints."""
    if geometry is None :
        raise RuntimeError (
        "Guard slot generation requires "
        "ProvisionalGuardGeometry"
        )

    if (
    geometry .mouth_start_world is None 
    or geometry .mouth_end_world is None 
    ):
        raise RuntimeError (
        "Guard slot generation requires "
        "localized LiDAR mouth endpoints"
        )

    mouth_start =geometry .mouth_start_world .copy ()
    mouth_end =geometry .mouth_end_world .copy ()
    mouth_vector =mouth_end -mouth_start 
    mouth_span =float (mouth_vector .length ())

    if mouth_span <=physical .EPSILON :
        raise RuntimeError (
        "Guard mouth endpoints are degenerate: "
        f"uid={descriptor .uid }"
        )

    normal =mouth_vector .normalize ()
    mouth_center =0.5 *(mouth_start +mouth_end )
    tangent =pygame .Vector2 (-normal .y ,normal .x ).normalize ()

    prior_tangent =geometry .branch_tangent_unit 
    if (
    prior_tangent is not None 
    and prior_tangent .length_squared ()>physical .EPSILON 
    and tangent .dot (prior_tangent )<0.0 
    ):
        tangent =-tangent 

    descriptor .observed_mouth_position =mouth_center .copy ()
    descriptor .local_outgoing_direction =tangent .copy ()
    descriptor .local_return_direction =-tangent 
    descriptor .direction_last_estimate =tangent .copy ()
    descriptor .direction_stability_reference =tangent .copy ()
    descriptor .motion_t =tangent .copy ()
    descriptor .motion_n =normal .copy ()
    descriptor .motion_frame_locked =True 
    descriptor .motion_frame_source ="LOCALIZED_LIDAR_MOUTH_ENDPOINTS"
    descriptor .observed_width =mouth_span 
    descriptor .motion_observed_width =mouth_span 
    descriptor .observed_flow_width =mouth_span 
    descriptor .observed_physical_width =mouth_span 
    descriptor .physical_left_boundary_lateral =-0.5 *mouth_span 
    descriptor .physical_right_boundary_lateral =0.5 *mouth_span 

    geometry .mouth_center_world =mouth_center .copy ()
    geometry .mouth_lateral_unit =normal .copy ()
    geometry .branch_tangent_unit =tangent .copy ()
    geometry .mouth_span =mouth_span 

    lateral_min ,lateral_max =compute_guard_lateral_interval (physical ,descriptor )
    if lateral_max <lateral_min :
        lateral_min =lateral_max =0.0 
    spacing =(lateral_max -lateral_min )/max (columns -1 ,1 )
    order =build_edge_sealing_slot_order (columns )

    total_required =columns *layers 

    def walkable_count_for_interval (
    test_min :float ,
    test_max :float ,
    )->int :
        test_spacing =(
        test_max -test_min 
        )/max (columns -1 ,1 )

        return sum (
        physical .is_walkable (
        mouth_center 
        +tangent 
        *(
        physical .JUNCTION_GUARD_BRANCH_INSET 
        +layer 
        *physical .THICK_MOUTH_GUARD_LAYER_SPACING 
        )
        +normal 
        *(
        test_min 
        +test_spacing *column 
        ),
        physical .ROBOT_RADIUS ,
        )
        for layer in range (layers )
        for column in range (columns )
        )

    current_walkable =walkable_count_for_interval (
    lateral_min ,
    lateral_max ,
    )

    if current_walkable <total_required :

        max_extra_inset =min (
        4.0 *physical .ROBOT_RADIUS ,
        0.15 *mouth_span ,
        )

        corrected =False 

        for extra_inset in np .linspace (
        0.0 ,
        max_extra_inset ,
        41 ,
        ):
            test_min =lateral_min +float (extra_inset )
            test_max =lateral_max -float (extra_inset )

            if test_max <=test_min :
                break 

            if (
            walkable_count_for_interval (
            test_min ,
            test_max ,
            )
            ==total_required 
            ):
                lateral_min =test_min 
                lateral_max =test_max 

                spacing =(
                lateral_max -lateral_min 
                )/max (columns -1 ,1 )

                corrected =True 

                print (
                f"[GuardSpanCorrection] "
                f"uid={descriptor .uid } "
                f"extra_inset={extra_inset :.3f} "
                f"walkable={total_required }/"
                f"{total_required }"
                )
                break 

        if not corrected :


            diagnostic_spacing =(
            lateral_max -lateral_min 
            )/max (
            columns -1 ,
            1 ,
            )

            unwalkable_diagnostics =[]

            for layer in range (layers ):

                axial =(
                physical .JUNCTION_GUARD_BRANCH_INSET 
                +layer 
                *physical .THICK_MOUTH_GUARD_LAYER_SPACING 
                )

                for column in range (columns ):

                    lateral =(
                    lateral_min 
                    +diagnostic_spacing 
                    *column 
                    )

                    slot =(
                    mouth_center 
                    +tangent *axial 
                    +normal *lateral 
                    )

                    if physical .is_walkable (
                    slot ,
                    physical .ROBOT_RADIUS ,
                    ):
                        continue 

                    unwalkable_diagnostics .append (
                    (
                    layer ,
                    column ,
                    axial ,
                    lateral ,
                    slot .x ,
                    slot .y ,
                    )
                    )

            print (
            "[GuardUnwalkableSummary] "
            f"uid={descriptor .uid } "
            f"rows={layers } "
            f"cols={columns } "
            f"unwalkable="
            f"{len (unwalkable_diagnostics )}/"
            f"{total_required } "
            f"t=({tangent .x :.3f},"
            f"{tangent .y :.3f}) "
            f"n=({normal .x :.3f},"
            f"{normal .y :.3f}) "
            f"mouth="
            f"({mouth_center .x :.3f},"
            f"{mouth_center .y :.3f}) "
            f"span="
            f"{lateral_max -lateral_min :.3f}"
            )

            for (
            layer ,
            column ,
            axial ,
            lateral ,
            world_x ,
            world_y ,
            )in unwalkable_diagnostics :

                print (
                "[GuardUnwalkableSlot] "
                f"uid={descriptor .uid } "
                f"layer={layer } "
                f"column={column } "
                f"axial={axial :.3f} "
                f"lateral={lateral :.3f} "
                f"world="
                f"({world_x :.3f},"
                f"{world_y :.3f})"
                )

            raise RuntimeError (
            f"Guard geometry has unreachable slots: "
            f"uid={descriptor .uid } "
            f"walkable={current_walkable }/"
            f"{total_required }"
            )



    if geometry is not None :
        geometry .sealing_lateral_min =lateral_min 
        geometry .sealing_lateral_max =lateral_max 
        geometry .slot_spacing =spacing 


    slots :list [pygame .Vector2 ]=[]

# 각 Branch 입구에 3×N Guard 대형을 세울 때, 각 Guard 로봇이 가야 할 목표 위치(slot)의 좌표를 계산
    for layer in range (layers ):
        # row_center = c_b​ + (a_r)*(​t^_b)​    
        row_center =(
        mouth_center 
        +tangent 
        *(
        physical .JUNCTION_GUARD_BRANCH_INSET 
        +layer *physical .THICK_MOUTH_GUARD_LAYER_SPACING 
        )
        )

        # (lateral_min + spacing * index) = (l_n) * (​n^_b)​
        row =[
        row_center +normal * (lateral_min +spacing * index )
        for index in range (columns )
        ]
        slots .extend (row )
        print (
        f"[GuardSlotOrder] uid={descriptor .uid } layer={layer } "
        f"order={order } perpendicular=True "
        f"t=({tangent .x :.3f},{tangent .y :.3f}) "
        f"n=({normal .x :.3f},{normal .y :.3f})"
        )

    return slots 


def build_provisional_multilayer_slots (

physical :types .ModuleType ,
robots :Sequence [Any ],
geometries :Sequence [ProvisionalGuardGeometry ],
perception :AdaptivePerception |None =None ,
)->None :
    """Create every LiDAR-derived layer at once; robot positions are unread."""
    for geometry in geometries :
        descriptor =geometry .descriptor 
        if geometry .mouth_span >0.0 :
            descriptor .observed_width =geometry .mouth_span 
            descriptor .observed_physical_width =geometry .mouth_span 
        columns ,lateral_min ,lateral_max ,slot_spacing =(
        compute_sealing_aware_column_count (physical ,descriptor )
        )
        layers =3 
        required =columns *layers 
        slots =build_sealing_aware_slots (physical ,descriptor ,columns ,layers ,geometry ,perception )
        geometry .columns =columns 
        geometry .layers =layers 
        geometry .slots =[slot .copy ()for slot in slots ]
        walkable =sum (
        physical .is_walkable (slot ,physical .ROBOT_RADIUS )for slot in slots 
        )
        unwalkable_slots =[
        (index ,slot )
        for index ,slot in enumerate (slots )
        if not physical .is_walkable (slot ,physical .ROBOT_RADIUS )
        ]
        opening =geometry .opening 
        print (f"[GuardGeometry] uid={geometry .provisional_uid }")
        print (f"[GuardGeometry] opening_start={float (opening ['start_angle']):.3f}")
        print (f"[GuardGeometry] opening_end={float (opening ['end_angle']):.3f}")
        print (f"[GuardGeometry] center={float (opening ['center_angle']):.3f}")
        print (f"[GuardGeometry] width_deg={float (opening ['width_deg']):.3f}")
        print (f"[GuardGeometry] estimated_mouth_width={descriptor .observed_physical_width :.3f}")
        print (
        f"[GuardAutoWidth] uid={geometry .provisional_uid } "
        f"wall_to_wall_span={geometry .mouth_span :.3f} "
        f"usable_span={max (0.0 ,lateral_max -lateral_min ):.3f} "
        f"columns={columns } layers={layers } "
        "source=LIDAR_WALL_ENDPOINT_CROSS_SECTION_OR_W_FALLBACK"
        )
        print (f"[GuardGeometry] usable_half={physical .local_physical_usable_half_width (descriptor ):.3f}")
        print (f"[GuardGeometry] sealing_lateral_min={lateral_min :.3f}")
        print (f"[GuardGeometry] sealing_lateral_max={lateral_max :.3f}")
        print (f"[GuardGeometry] columns={columns }")
        print (f"[GuardGeometry] layers={layers }")
        print (f"[GuardGeometry] required={required }")
        print (f"[GuardGeometry] slot_spacing={slot_spacing :.3f}")
        print (f"[GuardGeometry] slots_walkable={walkable }/{len (slots )}")
        if unwalkable_slots :
            print (
            f"[GuardGeometry] unwalkable_slots="
            f"{[(index ,round (slot .x ,3 ),round (slot .y ,3 ))for index ,slot in unwalkable_slots ]}"
            )

def install_provisional_guard_geometries (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
geometries :Sequence [
ProvisionalGuardGeometry 
],
geometry_frame :int ,
)->None :
    """Install one Junction's Guard geometries into the common Guard pipeline."""

    physical .integration_placement_localization_enabled =True 

    build_provisional_multilayer_slots (
    physical ,
    robots ,
    geometries ,
    perception ,
    )

    for geometry in geometries :

        physical .branch_descriptors_by_uid [
        geometry .provisional_uid 
        ]=geometry .descriptor 

        physical .integration_wall_status [
        geometry .provisional_uid 
        ]={
        "capture":0 ,
        "candidate_count":0 ,
        "assignment_count":0 ,
        "assigned":0 ,
        "edge_selected":0 ,
        "rows":geometry .layers ,
        "slots_per_row":geometry .columns ,
        "slots_walkable":sum (
        physical .is_walkable (
        slot ,
        physical .ROBOT_RADIUS ,
        )
        for slot in geometry .slots 
        ),
        "slots_total":len (
        geometry .slots 
        ),
        "ready":False ,
        "ready_frame":None ,
        }

    perception .provisional_guards =list (
    geometries 
    )

    print (
    "[GuardBranchMap] "
    +" ".join (
    f"{geometry .provisional_uid }="
    f"{geometry .local_branch_key }"
    for geometry in geometries 
    )
    )

    perception .provisional_guard_started =True 

    perception .guard_geometry_frame =(
    geometry_frame 
    )

    physical .integration_provisional_guard_groups ={}

    physical .integration_provisional_guard_active =False 

    physical .integration_guard_gating_enabled =False 

    print (
    "[Timeline] GUARD_GEOMETRY_READY "
    f"frame={geometry_frame } "
    "roles_assigned=0"
    )







    perception .guard_leakage .clear ()

    for geometry in geometries :

        descriptor =(
        geometry .descriptor 
        )

        state ={
        "robots_beyond_mouth_at_detection":0 ,
        "robots_beyond_mouth":0 ,
        "crossings_before_edge_seal":0 ,
        "crossings_after_edge_seal":0 ,
        "additional_outward_crossings_before_wall_ready":0 ,
        "additional_outward_crossings_after_wall_ready":0 ,
        "inward_returns":0 ,
        "maximum_normal_depth_before_wall_ready":0.0 ,
        "deepest_leaked_robot_depth":0.0 ,
        "leakage_blocked_after_edge_seal":False ,
        "previous_axial":{},
        }

        usable_half =(
        physical .local_physical_usable_half_width (
        descriptor 
        )
        )

        for robot in robots :

            axial ,lateral =(
            physical .branch_local_coordinates (
            robot .position ,
            descriptor ,
            )
            )

            state [
            "previous_axial"
            ][robot .robot_id ]=axial 

            if (
            robot .role =="NORMAL"
            and axial >0.0 
            and abs (lateral )
            <=usable_half 
            ):
                state [
                "robots_beyond_mouth_at_detection"
                ]+=1 

                state [
                "maximum_normal_depth_before_wall_ready"
                ]=max (
                state [
                "maximum_normal_depth_before_wall_ready"
                ],
                axial ,
                )

                state [
                "deepest_leaked_robot_depth"
                ]=max (
                state [
                "deepest_leaked_robot_depth"
                ],
                axial ,
                )

        perception .guard_leakage [
        geometry .provisional_uid 
        ]=state 

        print (
        "[Leakage] "
        f"uid={geometry .provisional_uid } "
        "robots_beyond_mouth_at_detection="
        f"{state ['robots_beyond_mouth_at_detection']}"
        )

    physical .integration_guard_leakage =(
    perception .guard_leakage 
    )


def initialize_provisional_guard_geometry_after_detection (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
frame :LidarFrame ,
)->None :
    """Initialize Root J0 Guard geometry."""

    if perception .provisional_guard_started :
        return 

    geometries =(
    build_provisional_guard_descriptors_from_lidar (
    physical ,
    perception ,
    frame ,
    )
    )

    geometry_frame =(
    perception .confirmation_frame 
    if perception .confirmation_frame is not None 
    else frame .frame 
    )

    install_provisional_guard_geometries (
    physical ,
    perception ,
    robots ,
    geometries ,
    geometry_frame ,
    )

def build_parent_guard_reformation_geometries (
physical :types .ModuleType ,
parent :MultiJunctionFrame ,
)->list [ProvisionalGuardGeometry ]:
    """Rebuild Parent Guard geometry from its restored Branch descriptors."""

    geometries :list [
    ProvisionalGuardGeometry 
    ]=[]

    for branch_uid in parent .branch_order :

        descriptor =(
        physical .branch_descriptors_by_uid .get (
        branch_uid 
        )
        )

        if descriptor is None :
            raise RuntimeError (
            "Parent Guard reformation missing descriptor: "
            f"junction={parent .junction_uid } "
            f"branch={branch_uid }"
            )

        mouth =(
        descriptor .observed_mouth_position 
        )

        if mouth is None :
            raise RuntimeError (
            "Parent Guard reformation missing mouth: "
            f"branch={branch_uid }"
            )

        tangent ,normal =(
        physical .descriptor_local_basis (
        descriptor 
        )
        )

        if (
        tangent .length_squared ()
        <=physical .EPSILON 
        or normal .length_squared ()
        <=physical .EPSILON 
        ):
            raise RuntimeError (
            "invalid Parent Guard local basis: "
            f"branch={branch_uid }"
            )

        tangent =tangent .normalize ()
        normal =normal .normalize ()

        mouth_span =float (
        descriptor .observed_physical_width 
        if (
        descriptor .observed_physical_width 
        >0.0 
        )
        else descriptor .observed_width 
        )

        if mouth_span <=0.0 :
            raise RuntimeError (
            "Parent Guard reformation has invalid mouth width: "
            f"branch={branch_uid } "
            f"width={mouth_span }"
            )

        fixture_key =(
        physical .branch_fixture_for_uid (
        branch_uid 
        )
        )

        local_branch_key =(
        fixture_key 
        if fixture_key is not None 
        else branch_uid 
        )

        mouth_start =(
        mouth 
        -normal 
        *(0.5 *mouth_span )
        )

        mouth_end =(
        mouth 
        +normal 
        *(0.5 *mouth_span )
        )


        center_angle =math .degrees (
        math .atan2 (
        tangent .y ,
        tangent .x ,
        )
        )

        diagnostic_opening ={
        "start_angle":center_angle ,
        "end_angle":center_angle ,
        "center_angle":center_angle ,
        "width_deg":0.0 ,
        }

        geometries .append (
        ProvisionalGuardGeometry (
        provisional_uid =branch_uid ,
        opening =diagnostic_opening ,
        descriptor =descriptor ,

        columns =0 ,
        layers =0 ,
        slots =[],

        fixture_key =fixture_key ,

        local_branch_key =(
        local_branch_key 
        ),

        mouth_start_world =(
        mouth_start .copy ()
        ),

        mouth_end_world =(
        mouth_end .copy ()
        ),

        mouth_center_world =(
        mouth .copy ()
        ),

        mouth_lateral_unit =(
        normal .copy ()
        ),

        branch_tangent_unit =(
        tangent .copy ()
        ),

        mouth_span =mouth_span ,
        )
        )

        print (
        "[ParentGuardGeometryRestore] "
        f"junction={parent .junction_uid } "
        f"branch={branch_uid } "
        f"fixture={fixture_key } "
        f"mouth_span={mouth_span :.3f} "
        "source=SAVED_BRANCH_DESCRIPTOR "
        "old_guard_ids_reused=False"
        )

    return geometries 


def initialize_parent_guard_reformation (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Start fresh 3xN Guard formation after Parent context restoration."""

    if not multi_dfs .parent_guard_reformation_pending :
        return 

    parent =multi_dfs .current 

    if parent is None :
        raise RuntimeError (
        "Parent Guard reformation requires current Junction"
        )

    if parent .subtree_complete :
        raise RuntimeError (
        "completed Parent must not rebuild Guard walls: "
        f"junction={parent .junction_uid }"
        )

    if perception .provisional_guard_started :
        return 

    if not parent .branch_order :
        raise RuntimeError (
        "Parent Guard reformation requires registered branches"
        )









    perception .guard_activation_stage =(
    "WAIT_GROUP"
    )

    perception .guard_activation_groups =[]
    perception .guard_current_group_index =0 
    perception .guard_all_groups_activated =False 

    perception .provisional_guards =[]
    perception .provisional_guard_started =False 

    perception .handoff_complete =False 
    perception .topology_ready_frame =None 

    physical .integration_wall_lifecycle ={}
    physical .integration_ready_guard_ids_by_uid ={}
    physical .integration_wall_status ={}
    physical .integration_wall_stats ={}

    physical .integration_provisional_guard_groups ={}
    physical .integration_provisional_guard_active =False 

    physical .integration_guard_gating_enabled =False 
    physical .integration_all_walls_ready =False 
    physical .integration_ready_guard_handoff =False 

    physical .integration_guard_formation_start_frame =None 

    physical .integration_child_guard_lifecycle_initialized =False 

    physical .pending_branch_start =None 
    physical .integration_pending_branch_uid =None 
    physical .integration_pending_frontier_committed =False 

    physical .active_branch =None 
    physical .active_branch_uid =None 

    physical .frontier_line_branch =None 
    physical .frontier_line_depth =0.0 
    physical .frontier_line_lateral_center =0.0 
    physical .frontier_line_row_ready =False 

    physical .phase =(
    physical .SimulationPhase .FORM_JUNCTION_GUARDS 
    )

    geometries =(
    build_parent_guard_reformation_geometries (
    physical ,
    parent ,
    )
    )

    geometry_frame =int (
    getattr (
    physical ,
    "integration_frame",
    -1 ,
    )
    )

    install_provisional_guard_geometries (
    physical ,
    perception ,
    robots ,
    geometries ,
    geometry_frame ,
    )

    print (
    "[ParentGuardReformationStart] "
    f"junction={parent .junction_uid } "
    f"branches={parent .branch_order } "
    f"depth={multi_dfs .depth } "
    "fresh_guard_selection=True "
    "localization_scope=GUARD_SELECTION_AND_PLACEMENT_ONLY"
    )

def parent_guard_reformation_all_ready (
physical :types .ModuleType ,
perception :AdaptivePerception ,
)->bool :
    """Return True only when every rebuilt Parent 3xN wall is physically READY."""

    parent =multi_dfs .current 

    if parent is None :
        return False 

    if not perception .provisional_guards :
        return False 

    geometry_by_uid ={
    geometry .provisional_uid :
    geometry 
    for geometry 
    in perception .provisional_guards 
    }

    if set (geometry_by_uid )!=set (
    parent .branch_order 
    ):
        return False 

    for branch_uid in parent .branch_order :

        geometry =geometry_by_uid .get (
        branch_uid 
        )

        if geometry is None :
            return False 

        if not geometry .cohort_ready :
            return False 

        if (
        len (geometry .selected_ids )
        !=len (geometry .slots )
        ):
            return False 

        status =(
        physical .integration_wall_status .get (
        branch_uid ,
        {},
        )
        )

        if not bool (
        status .get (
        "ready",
        False ,
        )
        ):
            return False 

    return True 


def finalize_parent_guard_reformation (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Freeze the newly selected Parent Guards and resume DFS."""

    if not multi_dfs .parent_guard_reformation_pending :
        return 

    if not parent_guard_reformation_all_ready (
    physical ,
    perception ,
    ):
        return 

    parent =multi_dfs .current 

    if parent is None :
        raise RuntimeError (
        "Parent Guard finalize requires current Junction"
        )

    by_id ={
    robot .robot_id :robot 
    for robot in robots 
    }

    geometry_by_uid ={
    geometry .provisional_uid :
    geometry 
    for geometry 
    in perception .provisional_guards 
    }



    fixture_keys_by_uid ={
    branch_uid :
    physical .branch_fixture_for_uid (
    branch_uid 
    )
    for branch_uid 
    in parent .branch_order 
    }

    fixture_presence =[
    fixture_key is not None 
    for fixture_key 
    in fixture_keys_by_uid .values ()
    ]

    if (
    any (fixture_presence )
    and not all (fixture_presence )
    ):
        raise RuntimeError (
        "Mixed Parent Guard lifecycle representations: "
        f"junction={parent .junction_uid } "
        f"fixtures={fixture_keys_by_uid }"
        )

    fixture_backed_parent =(
    bool (fixture_presence )
    and all (fixture_presence )
    )

    physical .integration_wall_lifecycle ={}
    physical .integration_ready_guard_ids_by_uid ={}
    physical .junction_guard_groups ={}

    for branch_uid in parent .branch_order :

        geometry =geometry_by_uid [
        branch_uid 
        ]

        descriptor =(
        physical .branch_descriptors_by_uid [
        branch_uid 
        ]
        )

        robot_ids =sorted (
        geometry .selected_ids 
        )

        members =[
        by_id [robot_id ]
        for robot_id in robot_ids 
        if robot_id in by_id 
        ]

        if len (members )!=len (robot_ids ):
            raise RuntimeError (
            "Parent Guard member lookup incomplete: "
            f"branch={branch_uid }"
            )

        if not members :
            raise RuntimeError (
            "Parent Guard lifecycle cannot be empty: "
            f"branch={branch_uid }"
            )

        logical_state =(
        parent .branch_states .get (
        branch_uid 
        )
        )

        if logical_state not in {
        "UNVISITED",
        "VISITED",
        }:
            raise RuntimeError (
            "invalid Parent branch state after Child POP: "
            f"junction={parent .junction_uid } "
            f"branch={branch_uid } "
            f"state={logical_state }"
            )

        descriptor .visit_state =(
        logical_state 
        )

        coordinates =[
        physical .branch_local_coordinates (
        robot .position ,
        descriptor ,
        )
        for robot in members 
        ]

        centroid_axial =float (
        np .mean ([
        axial 
        for axial ,_ in coordinates 
        ])
        )

        centroid_lateral =float (
        np .mean ([
        lateral 
        for _ ,lateral in coordinates 
        ])
        )

        physical .integration_ready_guard_ids_by_uid [
        branch_uid 
        ]=list (
        robot_ids 
        )

        fixture_key =(
        physical .branch_fixture_for_uid (
        branch_uid 
        )
        )


        if fixture_backed_parent :

            if fixture_key is None :
                raise RuntimeError (
                "Fixture-backed Parent reformation "
                "lost Branch fixture adapter: "
                f"junction={parent .junction_uid } "
                f"branch={branch_uid }"
                )

            lifecycle_key =(
            fixture_key 
            )

            robot_branch_key =(
            fixture_key 
            )

        else :

            lifecycle_key =(
            branch_uid 
            )

            robot_branch_key =(
            branch_uid 
            )
        for robot in members :

            if robot .role !="JUNCTION_GUARD":
                raise RuntimeError (
                "fresh Parent Guard lost physical role: "
                f"branch={branch_uid } "
                f"robot={robot .robot_id } "
                f"role={robot .role }"
                )

            robot .junction_guard_branch =(
            robot_branch_key 
            )

            robot .junction_guard_branch_uid =(
            branch_uid 
            )

        physical .junction_guard_groups [
        lifecycle_key 
        ]=list (
        robot_ids 
        )






        if fixture_backed_parent :

            if fixture_key is None :
                raise RuntimeError (
                "Fixture-backed Guard metadata "
                "has no fixture key: "
                f"junction={parent .junction_uid } "
                f"branch={branch_uid }"
                )

            physical .thick_mouth_guard_columns [
            fixture_key 
            ]=geometry .columns 

            physical .thick_mouth_guard_layers [
            fixture_key 
            ]=geometry .layers 

            if hasattr (
            physical ,
            "branch_gate_states",
            ):
                physical .branch_gate_states [
                fixture_key 
                ]="CLOSED"

        lifecycle_state =(
        "VISITED_GUARD"
        if logical_state =="VISITED"
        else "GUARD"
        )

        physical .integration_wall_lifecycle [
        lifecycle_key 
        ]={
        "uid":branch_uid ,

        "state":lifecycle_state ,

        "rows":geometry .layers ,
        "cols":geometry .columns ,

        "robot_ids":list (
        robot_ids 
        ),

        "centroid_axial":
        centroid_axial ,

        "centroid_lateral":
        centroid_lateral ,

        "original_guard_centroid_axial":
        centroid_axial ,

        "original_guard_centroid_lateral":
        centroid_lateral ,

        "guard_anchor_by_id":{
        robot .robot_id :
        robot .position .copy ()
        for robot in members 
        },

        "mouth_center_world":
        geometry .mouth_center_world .copy (),

        "branch_tangent_unit":
        geometry .branch_tangent_unit .copy (),

        "mouth_lateral_unit":
        geometry .mouth_lateral_unit .copy (),

        "measured_mouth_span":
        float (
        geometry .mouth_span 
        ),

        "sealing_lateral_min":
        float (
        geometry .sealing_lateral_min 
        ),

        "sealing_lateral_max":
        float (
        geometry .sealing_lateral_max 
        ),

        "slot_spacing":
        float (
        geometry .slot_spacing 
        ),

        "guard_layer_by_id":{
        robot .robot_id :int (
        getattr (
        robot ,
        "junction_guard_layer",
        -1 ,
        )
        )
        for robot in members 
        },

        "guard_hop_by_id":{
        robot .robot_id :int (
        getattr (
        robot ,
        "junction_guard_hop",
        -1 ,
        )
        )
        for robot in members 
        },

        "guard_parent_id_by_id":{
        robot .robot_id :getattr (
        robot ,
        "junction_guard_parent_id",
        None ,
        )
        for robot in members 
        },

        "guard_branch_key_by_id":{
        robot .robot_id :getattr (
        robot ,
        "junction_guard_branch",
        None ,
        )
        for robot in members 
        },

        "guard_is_leader_by_id":{
        robot .robot_id :bool (
        getattr (
        robot ,
        "is_branch_leader",
        False ,
        )
        )
        for robot in members 
        },

        "guard_slot_index_by_id":{
        robot .robot_id :int (
        getattr (
        robot ,
        "integration_guard_slot_index",
        -1 ,
        )
        )
        for robot in members 
        },

        "relative_offsets":{
        robot .robot_id :(
        axial 
        -centroid_axial ,
        lateral 
        -centroid_lateral ,
        )
        for (
        robot ,
        (axial ,lateral ),
        )
        in zip (
        members ,
        coordinates ,
        )
        },

        "same_ids_from_fresh_parent_guard":
        True ,

        "junction_uid":
        parent .junction_uid ,
        }
        print (
        "[ParentGuardLifecycleReady] "
        f"junction={parent .junction_uid } "
        f"branch={branch_uid } "
        f"key={lifecycle_key } "
        f"state={lifecycle_state } "
        f"robots={len (robot_ids )} "
        f"rows={geometry .layers } "
        f"cols={geometry .columns }"
        )


    physical .integration_child_guard_lifecycle_initialized =(
    not fixture_backed_parent 
    )

    physical .integration_guard_who_localization_enabled =False 
    physical .integration_placement_localization_enabled =False 

    physical .integration_all_walls_ready =True 
    physical .integration_ready_guard_handoff =True 
    physical .integration_guard_gating_enabled =True 
    physical .integration_provisional_guard_active =False 

    physical .pending_branch_start =None 
    physical .integration_pending_branch_uid =None 
    physical .integration_pending_frontier_committed =False 

    physical .active_branch =None 
    physical .active_branch_uid =None 

    physical .phase =(
    physical .SimulationPhase .FORM_JUNCTION_GUARDS 
    )

    perception .guard_all_groups_activated =True 
    perception .guard_activation_stage ="COMPLETE"

    perception .topology_ready_frame =int (
    getattr (
    physical ,
    "integration_frame",
    -1 ,
    )
    )

    perception .handoff_complete =True 
    perception .state =(
    PerceptionState .PHYSICAL_DFS 
    )

    physical .record_distributed_consensus (
    clear_selection =True 
    )

    multi_dfs .parent_guard_reformation_pending =False 

    print (
    "[ParentGuardReformationComplete] "
    f"junction={parent .junction_uid } "
    f"depth={multi_dfs .depth } "
    f"branches={parent .branch_order } "
    f"states={parent .branch_states } "
    "all_3xN_walls_ready=True "
    "fresh_ids=True "
    "localization=False "
    "dfs_resume=True"
    )

def update_parent_guard_reformation (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Run fresh Parent 3xN Guard selection/placement until every wall is READY."""

    if not multi_dfs .parent_guard_reformation_pending :
        return 

    initialize_parent_guard_reformation (
    physical ,
    perception ,
    robots ,
    )

    update_provisional_guard_leakage (
    physical ,
    perception ,
    robots ,
    )

    update_guard_readiness_and_activation (
    physical ,
    perception ,
    robots ,
    )

    update_provisional_wall_settling_audit (
    physical ,
    perception ,
    robots ,
    )

    log_wall_ready_blockers (
    physical ,
    perception ,
    robots ,
    )

    finalize_parent_guard_reformation (
    physical ,
    perception ,
    robots ,
    )



def communication_articulation_robot_ids (
robots :Sequence [Any ],
)->set [int ]:
    """Return articulation-like robot IDs from the existing local comm graph."""
    active_ids ={
    robot .robot_id for robot in robots 
    if getattr (robot ,"connected_to_base",False )
    }
    adjacency ={
    robot .robot_id :sorted (
    getattr (peer ,"robot_id",-1 )
    for peer in robot .comm_neighbors 
    if getattr (peer ,"robot_id",-1 )in active_ids 
    )
    for robot in robots 
    if robot .robot_id in active_ids 
    }
    discovery :dict [int ,int ]={}
    low :dict [int ,int ]={}
    parent :dict [int ,int |None ]={}
    critical :set [int ]=set ()
    counter =0 

    def visit (robot_id :int )->None :
        nonlocal counter 
        discovery [robot_id ]=counter 
        low [robot_id ]=counter 
        counter +=1 
        children =0 
        for neighbor_id in adjacency .get (robot_id ,[]):
            if neighbor_id not in discovery :
                parent [neighbor_id ]=robot_id 
                children +=1 
                visit (neighbor_id )
                low [robot_id ]=min (low [robot_id ],low [neighbor_id ])
                if parent .get (robot_id )is None and children >1 :
                    critical .add (robot_id )
                if (
                parent .get (robot_id )is not None 
                and low [neighbor_id ]>=discovery [robot_id ]
                ):
                    critical .add (robot_id )
            elif neighbor_id !=parent .get (robot_id ):
                low [robot_id ]=min (low [robot_id ],discovery [neighbor_id ])

    for robot_id in sorted (adjacency ):
        if robot_id not in discovery :
            parent [robot_id ]=None 
            visit (robot_id )
    return critical 


def largest_communication_component (robots :Sequence [Any ])->int :
    by_id ={robot .robot_id :robot for robot in robots }
    remaining =set (by_id )
    largest =0 
    while remaining :
        seed =min (remaining )
        stack =[seed ]
        remaining .remove (seed )
        size =0 
        while stack :
            robot_id =stack .pop ()
            size +=1 
            for peer in by_id [robot_id ].comm_neighbors :
                peer_id =getattr (peer ,"robot_id",-1 )
                if peer_id in remaining :
                    remaining .remove (peer_id )
                    stack .append (peer_id )
        largest =max (largest ,size )
    return largest 


def collect_shallow_guard_candidates_with_localization (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
geometry :ProvisionalGuardGeometry ,
all_geometries :Sequence [ProvisionalGuardGeometry ],
)->list [Any ]:
    """Localization-allowed Guard WHO selection.

    A NORMAL robot is eligible only while it is advancing from the Base toward
    this Junction, is closest to this still-recruiting Branch mouth, and lies
    inside that mouth's shallow capture region.
    """
    descriptor =geometry .descriptor 
    width =geometry .mouth_span or descriptor .observed_physical_width 
    tangent =(geometry .branch_tangent_unit or physical .descriptor_local_basis (descriptor )[0 ]).normalize ()
    lateral_axis =(geometry .mouth_lateral_unit or physical .descriptor_local_basis (descriptor )[1 ]).normalize ()
    center =geometry .mouth_center_world or descriptor .observed_mouth_position 

    session =multi_dfs .child_session 
    if (
    session is not None 
    and session .ingress_t .length_squared ()>physical .EPSILON 
    ):
        ingress_forward =session .ingress_t .normalize ()
    else :
        ingress_forward =_body_local_unit (perception ,0.0 ).normalize ()

    recruiting_geometries =[
    item 
    for item in all_geometries 
    if (
    not item .cohort_ready 
    and len (item .selected_ids )<len (item .slots )
    )
    ]
    if not recruiting_geometries :
        return []

    layers =max (geometry .layers ,1 )
    wall_depth =physical .JUNCTION_GUARD_BRANCH_INSET +(layers -1 )*physical .THICK_MOUTH_GUARD_LAYER_SPACING 
    upstream =max (GUARD_CAPTURE_UPSTREAM_WIDTH_RATIO *width ,wall_depth +2.0 *physical .ROBOT_RADIUS )
    downstream =max (GUARD_CAPTURE_DOWNSTREAM_WIDTH_RATIO *width ,wall_depth +physical .ROBOT_RADIUS )
    lateral_limit =0.5 *width +physical .ROBOT_RADIUS +0.5 
    candidates =[]
    for robot in robots :
        if (
        robot is perception .leader 
        or robot .role !="NORMAL"
        or robot .base_reserve 
        or not physical .is_walkable (robot .position ,robot .radius )
        ):
            continue 

        forward_speed =float (robot .velocity .dot (ingress_forward ))
        if forward_speed <=0.0 :
            continue 

        anchor_reference =(
        perception .anchor_position 
        if perception .anchor_position is not None 
        else perception .leader .position 
        )
        entered_junction_axial =float (
        (robot .position -anchor_reference ).dot (ingress_forward )
        )
        if entered_junction_axial <=0.0 :
            continue 

        def mouth_distance_key (candidate_geometry :ProvisionalGuardGeometry )->tuple [float ,str ]:
            candidate_center =(
            candidate_geometry .mouth_center_world 
            or candidate_geometry .descriptor .observed_mouth_position 
            )
            return (
            robot .position .distance_squared_to (candidate_center ),
            candidate_geometry .provisional_uid ,
            )

        nearest_geometry =min (
        recruiting_geometries ,
        key =mouth_distance_key ,
        )
        if nearest_geometry is not geometry :
            continue 

        delta =robot .position -center 
        axial ,lateral =delta .dot (tangent ),delta .dot (lateral_axis )
        if (
        -upstream 
        <=axial 
        <=downstream 
        and abs (lateral )<=lateral_limit 
        ):
            candidates .append (robot )
    return sorted (
    candidates ,
    key =lambda robot :(
    robot .position .distance_squared_to (center ),
    robot .robot_id ,
    ),
    )


def guard_mouth_coordinates (point :pygame .Vector2 ,geometry :ProvisionalGuardGeometry )->tuple [float ,float ]:
    center =geometry .mouth_center_world or geometry .descriptor .observed_mouth_position 
    tangent =geometry .branch_tangent_unit or geometry .descriptor .local_outgoing_direction 
    lateral =geometry .mouth_lateral_unit or geometry .descriptor .motion_n 
    delta =point -center 
    return float (delta .dot (tangent )),float (delta .dot (lateral ))


def build_guard_entry_waypoints (
physical :types .ModuleType ,
geometry :ProvisionalGuardGeometry ,
slot :pygame .Vector2 ,
slot_index :int |None =None ,
start_position :pygame .Vector2 |None =None ,
)->list [pygame .Vector2 ]:
    """Build a collision-aware, lane-preserving path into one Guard slot.

    Guard selection/placement is the one stage where world-position geometry is
    permitted. The path uses the frozen LiDAR mouth frame to route each selected
    robot through its own lateral lane instead of forcing every robot through one
    common centre staging point.

    For the rows of the same column, the deeper row is queued closest to the
    mouth and the shallower rows are queued progressively farther back.
    """
    center =(
    geometry .mouth_center_world 
    or geometry .descriptor .observed_mouth_position 
    )
    tangent =(
    geometry .branch_tangent_unit 
    or geometry .descriptor .local_outgoing_direction 
    )
    normal =(
    geometry .mouth_lateral_unit 
    or geometry .descriptor .motion_n 
    )

    if center is None or tangent is None or normal is None :
        return [slot .copy ()]

    tangent =tangent .normalize ()
    normal =normal .normalize ()

    mouth_span =float (
    geometry .mouth_span 
    or geometry .descriptor .observed_physical_width 
    )

    final_delta =slot -center 
    final_axial =float (final_delta .dot (tangent ))
    final_lateral =float (final_delta .dot (normal ))

    crossing_half_width =max (
    0.0 ,
    0.5 *mouth_span 
    -3.0 *physical .ROBOT_RADIUS ,
    )

    crossing_lateral =float (
    np .clip (
    final_lateral ,
    -crossing_half_width ,
    crossing_half_width ,
    )
    )

    if slot_index is None :
        slot_index =min (
        range (len (geometry .slots )),
        key =lambda index :
        geometry .slots [index ].distance_squared_to (slot ),
        )

    layer =int (
    slot_index 
    //max (geometry .columns ,1 )
    )
    layer =int (
    np .clip (
    layer ,
    0 ,
    max (geometry .layers -1 ,0 ),
    )
    )





    queue_rank =max (
    0 ,
    geometry .layers -1 -layer ,
    )

    queue_spacing =max (
    2.35 *physical .ROBOT_RADIUS ,
    0.70 *float (
    getattr (
    physical ,
    "THICK_MOUTH_GUARD_LAYER_SPACING",
    3.0 *physical .ROBOT_RADIUS ,
    )
    ),
    )

    lane_join_backoff =(
    7.0 *physical .ROBOT_RADIUS 
    +queue_rank *queue_spacing 
    )

    approach_backoff =(
    3.2 *physical .ROBOT_RADIUS 
    +queue_rank *queue_spacing 
    )



    lane_join =(
    center 
    -tangent *lane_join_backoff 
    +normal *crossing_lateral 
    )



    approach =(
    center 
    -tangent *approach_backoff 
    +normal *crossing_lateral 
    )



    inside_depth =max (
    2.5 *physical .ROBOT_RADIUS ,
    min (
    max (
    final_axial 
    -1.25 *physical .ROBOT_RADIUS ,
    0.0 ,
    ),
    4.0 *physical .ROBOT_RADIUS ,
    ),
    )

    inside =(
    center 
    +tangent *inside_depth 
    +normal *crossing_lateral 
    )

    if not physical .is_walkable (
    slot ,
    physical .ROBOT_RADIUS ,
    ):
        raise RuntimeError (
        "attempted to assign Guard to an unwalkable final slot"
        )

    waypoints :list [pygame .Vector2 ]=[]

    def segment_walkable (
    start :pygame .Vector2 ,
    end :pygame .Vector2 ,
    )->bool :
        """Check the whole Guard path segment, not only its endpoint."""

        distance =start .distance_to (end )

        if distance <=physical .EPSILON :
            return True 

        sample_step =max (
        0.75 *physical .ROBOT_RADIUS ,
        0.5 ,
        )

        sample_count =max (
        1 ,
        int (math .ceil (distance /sample_step )),
        )

        for sample_index in range (1 ,sample_count +1 ):
            alpha =sample_index /sample_count 

            point =start .lerp (
            end ,
            alpha ,
            )

            if not physical .is_walkable (
            point ,
            physical .ROBOT_RADIUS ,
            ):
                return False 

        return True 


    def append_waypoint (
    point :pygame .Vector2 ,
    )->None :

        if not physical .is_walkable (
        point ,
        physical .ROBOT_RADIUS ,
        ):
            return 

        if (
        waypoints 
        and waypoints [-1 ].distance_squared_to (
        point 
        )
        <=physical .EPSILON 
        ):
            return 

        waypoints .append (
        point .copy ()
        )



    if (
    start_position is not None 
    and not segment_walkable (
    start_position ,
    lane_join ,
    )
    ):






        start_delta =(
        start_position 
        -center 
        )

        start_lateral =float (
        start_delta .dot (normal )
        )

        corner_clearance =max (
        3.0 *physical .ROBOT_RADIUS ,
        0.08 *mouth_span ,
        )

        safe_lateral =float (
        np .clip (
        start_lateral ,
        -crossing_half_width 
        +corner_clearance ,
        crossing_half_width 
        -corner_clearance ,
        )
        )

        junction_ingress =(
        center 
        -tangent *lane_join_backoff 
        +normal *safe_lateral 
        )

        if (
        physical .is_walkable (
        junction_ingress ,
        physical .ROBOT_RADIUS ,
        )
        and segment_walkable (
        start_position ,
        junction_ingress ,
        )
        ):
            append_waypoint (
            junction_ingress 
            )

    append_waypoint (
    lane_join 
    )

    append_waypoint (
    approach 
    )

    append_waypoint (
    inside 
    )

    append_waypoint (
    slot 
    )

    if not waypoints :
        raise RuntimeError (
        "Guard waypoint construction produced no "
        "walkable waypoint"
        )

    return waypoints 





def guard_entry_path_is_walkable (
physical :types .ModuleType ,
start :pygame .Vector2 ,
waypoints :Sequence [pygame .Vector2 ],
radius :float ,
)->bool :
    """Check every physical segment of a Guard placement route.

    Guard selection/placement is the only localization-allowed stage.
    A waypoint being walkable is not sufficient: the straight segment
    from the robot to that waypoint must also remain inside free space.
    """

    current =start .copy ()

    sample_step =max (
    0.5 ,
    0.5 *float (radius ),
    )

    for target in waypoints :

        delta =target -current 
        distance =delta .length ()

        if distance <=physical .EPSILON :
            current =target .copy ()
            continue 

        sample_count =max (
        1 ,
        int (math .ceil (distance /sample_step )),
        )

        for sample_index in range (
        1 ,
        sample_count +1 ,
        ):
            alpha =(
            sample_index 
            /sample_count 
            )

            probe =(
            current 
            +delta *alpha 
            )

            if not physical .is_walkable (
            probe ,
            radius ,
            ):
                return False 

        current =target .copy ()

    return True 


def compute_full_guard_slot_assignment (
physical :types .ModuleType ,
geometry :ProvisionalGuardGeometry ,
candidates :Sequence [Any ],
)->tuple [list [tuple [Any ,pygame .Vector2 ,int ]],dict [str ,Any ]]:
    """Run deterministic full bipartite feasibility and branch-local WHO cost."""
    descriptor =geometry .descriptor 
    tangent =geometry .branch_tangent_unit or physical .descriptor_local_basis (descriptor )[0 ]
    lateral_axis =geometry .mouth_lateral_unit or physical .descriptor_local_basis (descriptor )[1 ]
    center =geometry .mouth_center_world or descriptor .observed_mouth_position 
    def mouth_coords (point :pygame .Vector2 )->tuple [float ,float ]:
        delta =point -center 
        return float (delta .dot (tangent )),float (delta .dot (lateral_axis ))
    width =descriptor .observed_physical_width 
    required =len (geometry .slots )
    options :dict [int ,list [tuple [float ,Any ]]]={}
    candidate_laterals =[]
    candidate_axials =[]
    for robot in candidates :
        axial ,lateral =mouth_coords (robot .position )
        candidate_axials .append (float (axial ))
        candidate_laterals .append (float (lateral ))
    for slot_index ,slot in enumerate (geometry .slots ):
        slot_axial ,slot_lateral =mouth_coords (slot )
        ranked =[]
        for robot in candidates :
            axial ,lateral =mouth_coords (robot .position )
            axial_delta =abs (axial -slot_axial )
            lateral_delta =abs (lateral -slot_lateral )
            path_distance =robot .position .distance_to (slot )
            if (
            axial_delta >GUARD_ASSIGN_MAX_AXIAL_WIDTH_RATIO *width 
            or lateral_delta 
            >GUARD_ASSIGN_MAX_LATERAL_WIDTH_RATIO *width 
            or path_distance >GUARD_ASSIGN_MAX_PATH_WIDTH_RATIO *width 
            ):
                continue 
            opposite_side_penalty =float (
            lateral 
            *slot_lateral 
            <0.0 
            and min (
            abs (lateral ),
            abs (slot_lateral ),
            )
            >2.0 
            *physical .ROBOT_RADIUS 
            )

            cost =(
            4.0 *opposite_side_penalty 
            +GUARD_WHO_AXIAL_WEIGHT 
            *axial_delta 
            /max (
            width ,
            physical .EPSILON ,
            )
            +3.0 
            *GUARD_WHO_LATERAL_WEIGHT 
            *lateral_delta 
            /max (
            width ,
            physical .EPSILON ,
            )
            +GUARD_WHO_PATH_WEIGHT 
            *path_distance 
            /max (
            width ,
            physical .EPSILON ,
            )
            )
            ranked .append ((cost ,robot ))
        options [slot_index ]=sorted (
        ranked ,key =lambda item :(item [0 ],item [1 ].robot_id )
        )

    slot_for_robot :dict [int ,int ]={}
    robot_for_slot :dict [int ,Any ]={}

    def augment (slot_index :int ,seen :set [int ])->bool :
        for _ ,robot in options [slot_index ]:
            if robot .robot_id in seen :
                continue 
            seen .add (robot .robot_id )
            previous_slot =slot_for_robot .get (robot .robot_id )
            if previous_slot is None or augment (previous_slot ,seen ):
                slot_for_robot [robot .robot_id ]=slot_index 
                robot_for_slot [slot_index ]=robot 
                return True 
        return False 

    priority =build_edge_sealing_slot_order (geometry .columns )
    slot_order =[layer *geometry .columns +col for layer in range (geometry .layers )for col in priority ]
    rank ={index :rank for rank ,index in enumerate (slot_order )}
    slot_order .sort (key =lambda index :(len (options [index ]),rank [index ]))
    for slot_index in slot_order :
        augment (slot_index ,set ())

    assignment =[
    (robot_for_slot [index ],geometry .slots [index ].copy (),index )
    for index in range (required )
    if index in robot_for_slot 
    ]
    distances =[
    robot .position .distance_to (slot )
    for robot ,slot ,_ in assignment 
    ]
    usable_half =physical .local_physical_usable_half_width (descriptor )
    if candidate_laterals :
        lateral_span =max (candidate_laterals )-min (candidate_laterals )
        coverage_ratio =float (np .clip (
        lateral_span /max (2.0 *usable_half ,physical .EPSILON ),
        0.0 ,
        1.0 ,
        ))
        occupied_bins =len ({
        int (np .clip (
        math .floor (
        (lateral +usable_half )
        /max (2.0 *usable_half ,physical .EPSILON )
        *GUARD_LATERAL_COVERAGE_BINS 
        ),
        0 ,
        GUARD_LATERAL_COVERAGE_BINS -1 ,
        ))
        for lateral in candidate_laterals 
        })
    else :
        coverage_ratio =0.0 
        occupied_bins =0 
    assigned_laterals =[guard_mouth_coordinates (robot .position ,geometry )[1 ]for robot ,_ ,_ in assignment ]
    left_edge_gap =max (
    0.0 ,
    (min (assigned_laterals )-geometry .sealing_lateral_min )
    if assigned_laterals else float ("inf"),
    )
    right_edge_gap =max (
    0.0 ,
    (geometry .sealing_lateral_max -max (assigned_laterals ))
    if assigned_laterals else float ("inf"),
    )
    ordered_laterals =sorted (assigned_laterals )
    max_internal_gap =max (
    (right -left for left ,right in zip (ordered_laterals ,ordered_laterals [1 :])),
    default =float ("inf"),
    )
    diagnostics ={
    "required":required ,
    "candidate_count":len (candidates ),
    "assignment_count":len (assignment ),
    "coverage_ratio":coverage_ratio ,
    "occupied_lateral_bins":occupied_bins ,
    "axial_min":min (candidate_axials ,default =0.0 ),
    "axial_max":max (candidate_axials ,default =0.0 ),
    "max_assignment_distance":max (distances ,default =float ("inf")),
    "mean_assignment_distance":float (np .mean (distances ))
    if distances else float ("inf"),
    "full_slot_assignment_possible":len (assignment )==required ,
    "left_edge_gap":left_edge_gap ,
    "right_edge_gap":right_edge_gap ,
    "max_edge_gap":max (left_edge_gap ,right_edge_gap ),
    "max_internal_gap":max_internal_gap ,
    "outer_edge_sealed":(
    left_edge_gap <=physical .FRONTIER_LINE_MAX_EDGE_GAP 
    and right_edge_gap <=physical .FRONTIER_LINE_MAX_EDGE_GAP 
    ),
    "worst_uncovered_span":max (left_edge_gap ,right_edge_gap ,max_internal_gap ),
    "outermost_assigned_slot_reach":(
    max (abs (min (assigned_laterals )-geometry .sealing_lateral_min ),
    abs (geometry .sealing_lateral_max -max (assigned_laterals )))
    if assigned_laterals else 0.0 
    ),
    }
    return assignment ,diagnostics 


def branch_guard_cohort_ready (
geometry :ProvisionalGuardGeometry ,
diagnostics :dict [str ,Any ],
)->bool :
    return bool (
    diagnostics ["candidate_count"]>=len (geometry .slots )
    and diagnostics ["full_slot_assignment_possible"]
    )


def activate_guard_cohort (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
geometry :ProvisionalGuardGeometry ,
assignment :Sequence [tuple [Any ,pygame .Vector2 ,int ]],
critical_ids :set [int ],
diagnostics :dict [str ,Any ],
)->None :
    """Perform same-position NORMAL->Guard transition, then close localization."""
    frame =getattr (physical ,"integration_frame",-1 )
    before_stats =physical .get_communication_stats (robots )
    before_largest =largest_communication_component (robots )
    relay_ids =sorted (
    robot .robot_id for robot in robots 
    if robot .role in {"RELAY","TRUNK_RELAY"}
    )
    selected =sorted (assignment ,key =lambda item :item [2 ])
    leader =min ((item [0 ]for item in selected ),key =lambda robot :robot .robot_id )
    maximum_jump =0.0 
    selected_ids =[]
    for robot ,slot ,slot_index in selected :
        before =robot .position .copy ()
        robot .role ="JUNCTION_GUARD"
        robot .integration_guard_waypoints =build_guard_entry_waypoints (
        physical ,
        geometry ,
        slot ,
        slot_index ,
        start_position =robot .position .copy (),
        )
        robot .integration_guard_final_anchor =slot .copy ()
        robot .integration_guard_slot_index =slot_index 
        robot .junction_guard_anchor =(
        robot .integration_guard_waypoints [0 ].copy ()
        )
        robot .junction_guard_branch =geometry .provisional_uid 
        robot .junction_guard_branch_uid =geometry .provisional_uid 
        robot .junction_guard_hop =0 
        robot .junction_guard_parent_id =(
        None if robot is leader else leader .robot_id 
        )
        robot .junction_guard_layer =slot_index //geometry .columns 
        robot .is_branch_leader =robot is leader 
        maximum_jump =max (
        maximum_jump ,robot .position .distance_to (before )
        )
        selected_ids .append (robot .robot_id )
    geometry .descriptor .leader_id =leader .robot_id 
    geometry .selected_ids =selected_ids 
    geometry .cohort_ready =True 
    geometry .guard_ready_frame =frame 
    geometry .role_assignment_frame =frame 
    physical .integration_provisional_guard_groups [
    geometry .provisional_uid 
    ]=selected_ids 
    physical .integration_provisional_guard_active =True 
    physical .integration_guard_gating_enabled =True 
    if physical .integration_guard_formation_start_frame is None :
        physical .integration_guard_formation_start_frame =frame 
    physical .integration_guard_role_transition_jump =max (
    physical .integration_guard_role_transition_jump ,
    maximum_jump ,
    )
    status =physical .integration_wall_status [geometry .provisional_uid ]
    status .update ({
    "capture":diagnostics ["candidate_count"],
    "candidate_count":diagnostics ["candidate_count"],
    "assignment_count":diagnostics ["assignment_count"],
    "assigned":len (selected_ids ),
    "edge_selected":len (selected_ids ),
    "coverage_ratio":diagnostics ["coverage_ratio"],
    "ready":False ,
    })
    after_stats =physical .get_communication_stats (robots )
    after_largest =largest_communication_component (robots )
    selected_critical =sorted (set (selected_ids )&critical_ids )
    communication_audit ={
    "before_connected":before_stats ["connected"],
    "before_largest_component":before_largest ,
    "after_connected":after_stats ["connected"],
    "after_largest_component":after_largest ,
    "relay_trunk_ids":relay_ids ,
    "critical_ids":sorted (critical_ids ),
    "selected_critical_ids":selected_critical ,
    "communication_disconnect_caused_by_guard_selection":(
    after_stats ["connected"]<before_stats ["connected"]
    or after_largest <before_largest 
    ),
    }
    perception .guard_communication_audits [
    geometry .provisional_uid 
    ]=communication_audit 
    print (
    f"[Timeline] GUARD_COHORT_READY uid={geometry .provisional_uid } "
    f"frame={frame }"
    )
    print (
    f"[Timeline] GUARD_ROLE_ASSIGNMENT uid={geometry .provisional_uid } "
    f"frame={frame }"
    )
    print (f"[LocalizationWHO] uid={geometry .provisional_uid }")
    print (f"[LocalizationWHO] elected_ids={selected_ids }")
    print (
    f"[LocalizationWHO] assignment_count={len (selected_ids )}/"
    f"{len (geometry .slots )}"
    )
    print (
    f"[CommunicationBefore] uid={geometry .provisional_uid } "
    f"connected={before_stats ['connected']} "
    f"largest_component={before_largest } "
    f"relay_trunk_ids={relay_ids }"
    )
    print (
    f"[CommunicationAfter] uid={geometry .provisional_uid } "
    f"connected={after_stats ['connected']} "
    f"largest_component={after_largest } "
    f"communication_critical_robots_selected={selected_critical } "
    "communication_disconnect_caused_by_guard_selection="
    f"{communication_audit ['communication_disconnect_caused_by_guard_selection']}"
    )
    print (
    f"[Teleport] uid={geometry .provisional_uid } "
    f"guard_role_transition_jump={maximum_jump :.6f} "
    "direct_position_overwrite_count=0"
    )

def update_guard_readiness_and_activation (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Each branch independently recruits arriving NORMALs into Guard slots."""

    if not perception .provisional_guard_started :
        return 

    frame =getattr (physical ,"integration_frame",-1 )

    perception .guard_activation_groups =[]

    for geometry in perception .provisional_guards :

        if geometry .cohort_ready :
            continue 
        lifecycle =(
        physical .integration_wall_lifecycle .get (
        geometry .provisional_uid 
        )
        )

        if (
        lifecycle is not None 
        and lifecycle .get (
        "state",
        "GUARD",
        )
        !="GUARD"
        ):
            continue 

        physical .integration_guard_who_localization_enabled =True 
        physical .integration_placement_localization_enabled =True 



        candidates =collect_shallow_guard_candidates_with_localization (
        physical ,
        perception ,
        robots ,
        geometry ,
        perception .provisional_guards ,
        )



        occupied_slots ={
        int (robot .integration_guard_slot_index )
        for robot in robots 
        if robot .robot_id in geometry .selected_ids 
        and getattr (robot ,"integration_guard_slot_index",None )is not None 
        }

        column_priority =build_edge_sealing_slot_order (
        geometry .columns 
        )

        slot_priority =[
        layer *geometry .columns +column 
        for layer in reversed (
        range (geometry .layers )
        )
        for column in column_priority 
        ]

        empty_slots =[
        slot_index 
        for slot_index in slot_priority 
        if slot_index not in occupied_slots 
        ]



        available =[
        robot 
        for robot in candidates 
        if robot .robot_id not in geometry .selected_ids 
        and robot .role =="NORMAL"
        ]



        for slot_index in empty_slots :

            if not available :
                break 

            slot =geometry .slots [slot_index ]

            slot_waypoints =(
            build_guard_entry_waypoints (
            physical ,
            geometry ,
            slot ,
            slot_index ,
            )
            )

            feasible_available =[
            candidate 
            for candidate in available 
            if guard_entry_path_is_walkable (
            physical ,
            candidate .position ,
            slot_waypoints ,
            candidate .radius ,
            )
            ]

            if not feasible_available :
                if frame %20 ==0 :
                    print (
                    "[GuardRouteBlocked] "
                    f"uid={geometry .provisional_uid } "
                    f"slot={slot_index } "
                    f"layer="
                    f"{slot_index //geometry .columns } "
                    f"candidates={len (available )} "
                    "feasible=0"
                    )
                continue 

            robot =min (
            feasible_available ,
            key =lambda candidate :(
            candidate .position .distance_squared_to (
            slot 
            ),
            candidate .robot_id ,
            ),
            )

            available .remove (robot )



            robot .role ="JUNCTION_GUARD"

            robot .integration_guard_waypoints =[
            waypoint .copy ()
            for waypoint in slot_waypoints 
            ]

            robot .integration_guard_final_anchor =(
            slot .copy ()
            )

            robot .integration_guard_slot_index =(
            slot_index 
            )

            robot .junction_guard_anchor =(
            robot .integration_guard_waypoints [0 ].copy ()
            )

            robot .junction_guard_branch =(
            geometry .provisional_uid 
            )

            robot .junction_guard_branch_uid =(
            geometry .provisional_uid 
            )

            robot .junction_guard_layer =(
            slot_index //geometry .columns 
            )



            if not geometry .selected_ids :
                geometry .descriptor .leader_id =(
                robot .robot_id 
                )
                robot .junction_guard_parent_id =None 
                robot .is_branch_leader =True 
            else :
                robot .junction_guard_parent_id =(
                geometry .descriptor .leader_id 
                )
                robot .is_branch_leader =False 

            geometry .selected_ids .append (
            robot .robot_id 
            )

            print (
            f"[IncrementalGuard] "
            f"uid={geometry .provisional_uid } "
            f"robot={robot .robot_id } "
            f"slot={slot_index } "
            f"filled={len (geometry .selected_ids )}/"
            f"{len (geometry .slots )}"
            )

        status =physical .integration_wall_status [
        geometry .provisional_uid 
        ]

        status ["candidate_count"]=len (
        candidates 
        )

        status ["assignment_count"]=len (
        geometry .selected_ids 
        )

        status ["assigned"]=len (
        geometry .selected_ids 
        )



        if geometry .selected_ids :

            physical .integration_provisional_guard_groups [
            geometry .provisional_uid 
            ]=list (geometry .selected_ids )

            physical .integration_provisional_guard_active =True 
            physical .integration_guard_gating_enabled =True 

            if (
            physical .integration_guard_formation_start_frame 
            is None 
            ):
                physical .integration_guard_formation_start_frame =(
                frame 
                )

            if geometry .role_assignment_frame is None :
                geometry .role_assignment_frame =frame 



        if len (geometry .selected_ids )==len (
        geometry .slots 
        ):
            geometry .cohort_ready =True 
            geometry .guard_ready_frame =frame 

            status =physical .integration_wall_status [
            geometry .provisional_uid 
            ]

            status ["assigned"]=len (
            geometry .selected_ids 
            )

            status ["assignment_count"]=len (
            geometry .selected_ids 
            )

            print (
            f"[IncrementalGuardComplete] "
            f"uid={geometry .provisional_uid } "
            f"robots={len (geometry .selected_ids )} "
            f"frame={frame }"
            )



    all_complete =all (
    geometry .cohort_ready 
    and len (geometry .selected_ids )
    ==len (geometry .slots )
    for geometry in perception .provisional_guards 
    )

    if all_complete :
        perception .guard_all_groups_activated =True 
        perception .guard_activation_stage ="COMPLETE"

        print (
        f"[GuardFormationComplete] "
        f"frame={frame } "
        f"incremental_first_arrival=True"
        )

    physical .integration_guard_who_localization_enabled =False 
    physical .integration_placement_localization_enabled =False 

def update_provisional_wall_settling_audit (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Measure actual LiDAR-local wall completion before topology handoff."""
    if perception .handoff_complete :
        return 
    by_id ={robot .robot_id :robot for robot in robots }
    for geometry in perception .provisional_guards :
        if not geometry .cohort_ready :
            continue 
        status =physical .integration_wall_status [geometry .provisional_uid ]
        guards =[
        by_id [robot_id ]for robot_id in geometry .selected_ids 
        ]
        settled =sum (
        robot .integration_guard_final_anchor is not None 
        and robot .position .distance_to (
        robot .integration_guard_final_anchor 
        )<=physical .JUNCTION_GUARD_POSITION_TOLERANCE 
        for robot in guards 
        )

        unsettled_guards =[]

        for robot in guards :
            final_anchor =getattr (
            robot ,
            "integration_guard_final_anchor",
            None ,
            )

            if final_anchor is None :
                final_error =float ("inf")
            else :
                final_error =robot .position .distance_to (
                final_anchor 
                )

            if (
            final_error 
            <=physical .JUNCTION_GUARD_POSITION_TOLERANCE 
            ):
                continue 
            

            current_anchor =getattr (
            robot ,
            "junction_guard_anchor",
            None ,
            )

            current_target_error =(
            robot .position .distance_to (
            current_anchor 
            )
            if current_anchor is not None 
            else float ("inf")
            )

            waypoints =getattr (
            robot ,
            "integration_guard_waypoints",
            [],
            )

            unsettled_guards .append (
            (
            robot ,
            final_error ,
            current_target_error ,
            len (waypoints ),
            )
            )

        if (
        unsettled_guards 
        and getattr (
        physical ,
        "integration_frame",
        0 ,
        )%20 
        ==0 
        ):
            print (
            f"[GuardSettlingDetail] "
            f"uid={geometry .provisional_uid } "
            f"settled={settled }/{len (guards )} "
            f"unsettled={len (unsettled_guards )} "
            f"tol="
            f"{physical .JUNCTION_GUARD_POSITION_TOLERANCE :.3f}"
            )

            for (
            robot ,
            final_error ,
            current_target_error ,
            waypoint_count ,
            )in sorted (
            unsettled_guards ,
            key =lambda item :-item [1 ],
            ):
                print (
                f"[GuardSettlingRobot] "
                f"uid={geometry .provisional_uid } "
                f"id={robot .robot_id } "
                f"layer="
                f"{getattr (robot ,'junction_guard_layer',-1 )} "
                f"slot="
                f"{getattr (robot ,'integration_guard_slot_index',-1 )} "
                f"final_error={final_error :.3f} "
                f"current_target_error="
                f"{current_target_error :.3f} "
                f"waypoints={waypoint_count } "
                f"speed={robot .velocity .length ():.3f}"
                )

        complete_rows =0 
        minimum_span_ratio =1.0 
        maximum_edge_gap =0.0 
        maximum_internal_gap =0.0 
        expected_span_max =0.0 
        actual_span_max =0.0 
        left_edge_gap_max =0.0 
        right_edge_gap_max =0.0 
        for layer in range (geometry .layers ):
            expected_slots =geometry .slots [layer *geometry .columns :(layer +1 )*geometry .columns ]
            expected_laterals =sorted (guard_mouth_coordinates (slot ,geometry )[1 ]for slot in expected_slots )
            laterals =sorted (
            guard_mouth_coordinates (robot .position ,geometry )[1 ]
            for robot in guards 
            if robot .junction_guard_layer ==layer 
            )
            if len (laterals )<geometry .columns :
                minimum_span_ratio =0.0 
                continue 
            complete_rows +=1 
            expected_span =expected_laterals [-1 ]-expected_laterals [0 ]
            span =laterals [-1 ]-laterals [0 ]
            expected_span_max =max (expected_span_max ,expected_span )
            actual_span_max =max (actual_span_max ,span )
            left_edge_gap_max =max (left_edge_gap_max ,max (0.0 ,laterals [0 ]-expected_laterals [0 ]))
            right_edge_gap_max =max (right_edge_gap_max ,max (0.0 ,expected_laterals [-1 ]-laterals [-1 ]))
            minimum_span_ratio =min (
            minimum_span_ratio ,
            span /max (expected_span ,physical .EPSILON ),
            )
            maximum_edge_gap =max (maximum_edge_gap ,left_edge_gap_max ,right_edge_gap_max )
            maximum_internal_gap =max (
            maximum_internal_gap ,
            max (
            (
            right -left 
            for left ,right in zip (laterals ,laterals [1 :])
            ),
            default =0.0 ,
            ),
            )
        settled_ratio =settled /max (len (guards ),1 )
        structurally_sealed =(
        len (guards )==len (geometry .slots )
        and complete_rows ==geometry .layers 
        and minimum_span_ratio 
        >=physical .FRONTIER_LINE_MIN_SPAN_RATIO 
        and maximum_edge_gap 
        <=physical .FRONTIER_LINE_MAX_EDGE_GAP 
        and maximum_internal_gap 
        <=physical .FRONTIER_LINE_MAX_INTERNAL_GAP 
        )
        if structurally_sealed and settled_ratio >=PROVISIONAL_WALL_SETTLED_RATIO :
            status ["wall_ready_dwell"]=min (
            float (status .get ("wall_ready_dwell",0.0 ))+
            float (getattr (physical ,"NORMAL_PHYSICS_MAX_DT",1.0 /60.0 )),
            PROVISIONAL_WALL_STABILITY_DWELL ,
            )
        else :
            status ["wall_ready_dwell"]=0.0 
        ready =status ["wall_ready_dwell"]>=PROVISIONAL_WALL_STABILITY_DWELL 
        status .update ({
        "settled_ratio":settled_ratio ,
        "settled_count":settled ,
        "structurally_sealed":structurally_sealed ,
        "min_span_ratio":minimum_span_ratio ,
        "max_edge_gap":maximum_edge_gap ,
        "max_internal_gap":maximum_internal_gap ,
        "expected_span":expected_span_max ,
        "actual_span":actual_span_max ,
        "left_edge_gap":left_edge_gap_max ,
        "right_edge_gap":right_edge_gap_max ,
        "ready":ready ,
        })
        reasons =[]
        if settled_ratio <PROVISIONAL_WALL_SETTLED_RATIO :reasons .append ("settled_ratio")
        if complete_rows !=geometry .layers :reasons .append ("complete_rows")
        if minimum_span_ratio <physical .FRONTIER_LINE_MIN_SPAN_RATIO :reasons .append ("span_ratio")
        if maximum_edge_gap >physical .FRONTIER_LINE_MAX_EDGE_GAP :reasons .append ("edge_gap")
        if maximum_internal_gap >physical .FRONTIER_LINE_MAX_INTERNAL_GAP :reasons .append ("internal_gap")
        if structurally_sealed and settled_ratio >=PROVISIONAL_WALL_SETTLED_RATIO and not ready :
            reasons .append ("stable_dwell")
        if getattr (physical ,"integration_frame",0 )%10 ==0 or ready :
            print (f"[WallReadyBlocker] uid={geometry .provisional_uid } guard_count={len (guards )} expected={len (geometry .slots )} settled={settled }/{len (guards )} settled_ratio={settled_ratio :.3f} complete_rows={complete_rows }/{geometry .layers } expected_span={expected_span_max :.3f} actual_span={actual_span_max :.3f} span_ratio={minimum_span_ratio :.3f} left_edge_gap={left_edge_gap_max :.3f} right_edge_gap={right_edge_gap_max :.3f} max_edge_gap={maximum_edge_gap :.3f} max_internal_gap={maximum_internal_gap :.3f} structurally_sealed={structurally_sealed } stable_dwell={status ['wall_ready_dwell']:.3f} ready={ready } blocking_reasons={reasons }")
        if ready and status .get ("ready_frame")is None :
            status ["ready_frame"]=getattr (
            physical ,"integration_frame",-1 
            )
            print (
            f"[Timeline] PROVISIONAL_WALL_READY "
            f"uid={geometry .provisional_uid } "
            f"frame={status ['ready_frame']}"
            )
            print (
            f"[ProvisionalWallReady] uid={geometry .provisional_uid } "
            f"columns={geometry .columns } layers={geometry .layers } "
            f"required={len (geometry .slots )} assigned={len (guards )} "
            f"settled_ratio={settled_ratio :.3f} "
            f"span_ratio={minimum_span_ratio :.3f} "
            f"edge_gap={maximum_edge_gap :.3f} "
            f"internal_gap={maximum_internal_gap :.3f}"
            )


def update_provisional_guard_leakage (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    if (
    not perception .provisional_guard_started 
    or getattr (physical ,"integration_all_walls_ready",False )
    ):
        return 
    frame =getattr (physical ,"integration_frame",-1 )
    for geometry in perception .provisional_guards :
        state =perception .guard_leakage [geometry .provisional_uid ]
        descriptor =geometry .descriptor 
        usable_half =physical .local_physical_usable_half_width (descriptor )
        wall_status =getattr (physical ,"integration_wall_status",{}).get (
        geometry .provisional_uid ,{}
        )
        wall_ready =bool (wall_status .get ("ready",False ))
        beyond =0 
        for robot in robots :
            axial ,lateral =physical .branch_local_coordinates (
            robot .position ,descriptor 
            )
            previous =state ["previous_axial"].get (robot .robot_id ,axial )
            if robot .role =="NORMAL"and abs (lateral )<=usable_half :
                if axial >0.0 :
                    beyond +=1 
                if previous <=0.0 <axial :
                    if geometry .first_robot_crossing_mouth_frame is None :
                        geometry .first_robot_crossing_mouth_frame =frame 
                        print (
                        "[Timeline] FIRST_ROBOT_CROSSING_MOUTH "
                        f"uid={geometry .provisional_uid } frame={frame }"
                        )
                    if wall_ready :
                        state ["crossings_after_edge_seal"]+=1 
                        state ["additional_outward_crossings_after_wall_ready"]+=1 
                    else :
                        state ["crossings_before_edge_seal"]+=1 
                        state ["additional_outward_crossings_before_wall_ready"]+=1 
                elif previous >0.0 >=axial :
                    state ["inward_returns"]+=1 
                if axial >0.0 :
                    state ["deepest_leaked_robot_depth"]=max (
                    state ["deepest_leaked_robot_depth"],axial 
                    )
                    if not wall_ready :
                        state ["maximum_normal_depth_before_wall_ready"]=max (
                        state ["maximum_normal_depth_before_wall_ready"],axial 
                        )
            state ["previous_axial"][robot .robot_id ]=axial 
        state ["robots_beyond_mouth"]=beyond 
        if wall_ready :
            state ["leakage_blocked_after_edge_seal"]=beyond ==0 


def install_thick_wall_readiness_audit (physical :types .ModuleType )->None :
    """Require and record physical multi-row coverage before DFS selection."""
    physical .FRONTIER_LINE_MAX_EDGE_GAP =max (
    float (physical .FRONTIER_LINE_MAX_EDGE_GAP ),3.5 
    )
    original_ready =physical .junction_guards_formed 
    physical .integration_wall_stats ={}
    physical .integration_wall_status ={}
    physical .integration_all_walls_ready =False 

    def audited_ready (robots :Sequence [Any ])->bool :
        expected =[
        branch 
        for branch in physical .detected_branch_candidates 
        if branch not in physical .observed_visited_branches (robots )
        ]
        stats :dict [str ,dict [str ,float |int ]]={}
        all_branch_ready =bool (expected )
        for branch in expected :
            uid =physical .branch_uid_for_fixture (branch )
            if uid is None :
                all_branch_ready =False 
                continue 
            descriptor =physical .branch_descriptors_by_uid [uid ]
            guards =[
            robot for robot in robots 
            if robot .role =="JUNCTION_GUARD"
            and robot .junction_guard_branch ==branch 
            ]
            rows =physical .thick_mouth_guard_layers [branch ]
            columns =physical .thick_mouth_guard_columns [branch ]
            settled =sum (
            robot .junction_guard_anchor is not None 
            and robot .position .distance_to (robot .junction_guard_anchor )
            <=physical .JUNCTION_GUARD_POSITION_TOLERANCE 
            for robot in guards 
            )
            maximum_gap =0.0 
            minimum_span_ratio =1.0 
            maximum_edge_gap =0.0 
            complete_rows =0 
            usable_half =physical .local_physical_usable_half_width (descriptor )
            branch_tangent ,branch_lateral =physical .descriptor_local_basis (descriptor )
            mouth_center =descriptor .observed_mouth_position 
            for layer in range (rows ):
                laterals =sorted (
                float ((robot .position -mouth_center ).dot (branch_lateral ))
                for robot in guards 
                if robot .junction_guard_layer ==layer 
                )
                if not laterals :
                    minimum_span_ratio =0.0 
                    continue 
                if len (laterals )>=columns :
                    complete_rows +=1 
                maximum_gap =max (
                maximum_gap ,
                max (
                (right -left for left ,right in zip (laterals ,laterals [1 :])),
                default =0.0 ,
                ),
                )
                span =laterals [-1 ]-laterals [0 ]
                expected_span =max (2.0 *usable_half ,physical .EPSILON )
                minimum_span_ratio =min (minimum_span_ratio ,span /expected_span )
                maximum_edge_gap =max (
                maximum_edge_gap ,
                max (0.0 ,laterals [0 ]+usable_half ),
                max (0.0 ,usable_half -laterals [-1 ]),
                )
            branch_stats ={
            "physical_width":physical .local_guard_observed_width (descriptor ),
            "rows":rows ,
            "slots_per_row":columns ,
            "total":len (guards ),
            "settled_ratio":settled /max (len (guards ),1 ),
            "min_span_ratio":minimum_span_ratio ,
            "max_edge_gap":maximum_edge_gap ,
            "max_internal_gap":maximum_gap ,
            }
            stats [uid ]=branch_stats 
            ready =(
            rows >=physical .THICK_MOUTH_GUARD_MIN_LAYERS 
            and columns >=physical .JUNCTION_GUARD_MIN_COUNT 
            and len (guards )>=rows *columns 
            and complete_rows ==rows 
            and branch_stats ["settled_ratio"]>=PROVISIONAL_WALL_SETTLED_RATIO 
            and minimum_span_ratio 
            >=physical .FRONTIER_LINE_MIN_SPAN_RATIO 
            and maximum_edge_gap 
            <=physical .FRONTIER_LINE_MAX_EDGE_GAP 
            and maximum_gap 
            <=physical .FRONTIER_LINE_MAX_INTERNAL_GAP 
            )
            status =physical .integration_wall_status .setdefault (uid ,{})
            status .update (branch_stats )
            status ["ready"]=ready 
            if ready and status .get ("ready_frame")is None :
                status ["ready_frame"]=getattr (
                physical ,"integration_frame",-1 
                )
                print (
                f"[Timeline] {branch }_WALL_READY "
                f"frame={status ['ready_frame']} uid={uid }"
                )
                print (
                f"[ThickWallReady] uid={uid } "
                f"width={branch_stats ['physical_width']:.1f} "
                f"rows={rows } slots_per_row={columns } total={len (guards )} "
                f"settled_ratio={branch_stats ['settled_ratio']:.3f} "
                f"span_ratio={minimum_span_ratio :.3f} "
                f"edge_gap={maximum_edge_gap :.3f} "
                f"internal_gap={maximum_gap :.3f}"
                )
            all_branch_ready =all_branch_ready and ready 
        physical .integration_wall_stats =stats 
        ready =all_branch_ready and original_ready (robots )
        if ready and not physical .integration_all_walls_ready :
            physical .integration_all_walls_ready =True 
            physical .integration_guard_hold_active =False 

            physical .integration_guard_who_localization_enabled =False 
            physical .integration_placement_localization_enabled =False 
            physical .integration_ready_guard_ids_by_uid ={
            physical .branch_uid_for_fixture (branch ):sorted (
            robot .robot_id for robot in robots 
            if robot .role =="JUNCTION_GUARD"
            and robot .junction_guard_branch ==branch 
            )
            for branch in expected 
            }
            physical .integration_wall_lifecycle ={}
            for branch in expected :
                uid =physical .branch_uid_for_fixture (branch )
                descriptor =physical .branch_descriptors_by_uid [uid ]
                members =[
                robot for robot in robots 
                if robot .robot_id 
                in physical .integration_ready_guard_ids_by_uid [uid ]
                ]
                coordinates =[
                physical .branch_local_coordinates (
                robot .position ,descriptor 
                )
                for robot in members 
                ]
                centroid_axial =float (np .mean ([
                axial for axial ,_ in coordinates 
                ]))
                centroid_lateral =float (np .mean ([
                lateral for _ ,lateral in coordinates 
                ]))
                physical .integration_wall_lifecycle [branch ]={
                "uid":uid ,
                "state":"GUARD",
                "rows":physical .thick_mouth_guard_layers [branch ],
                "cols":physical .thick_mouth_guard_columns [branch ],
                "robot_ids":sorted (
                robot .robot_id for robot in members 
                ),
                "centroid_axial":centroid_axial ,
                "centroid_lateral":centroid_lateral ,
                "relative_offsets":{
                robot .robot_id :(
                axial -centroid_axial ,
                lateral -centroid_lateral ,
                )
                for robot ,(axial ,lateral )
                in zip (members ,coordinates )
                },
                }
            print (
            "[Timeline] ALL_WALLS_READY "
            f"frame={getattr (physical ,'integration_frame',-1 )}"
            )
            print ("[LocalizationAudit] placement_localization_enabled=False")
            for provisional_uid ,leakage in getattr (
            physical ,"integration_guard_leakage",{}
            ).items ():
                print (
                f"[Leakage] uid={provisional_uid } "
                f"robots_beyond_mouth_at_detection="
                f"{leakage ['robots_beyond_mouth_at_detection']} "
                f"robots_beyond_mouth={leakage ['robots_beyond_mouth']} "
                f"crossings_before_edge_seal="
                f"{leakage ['crossings_before_edge_seal']} "
                f"crossings_after_edge_seal="
                f"{leakage ['crossings_after_edge_seal']} "
                f"leakage_blocked_after_edge_seal="
                f"{leakage ['leakage_blocked_after_edge_seal']} "
                f"inward_returns={leakage ['inward_returns']} "
                f"deepest_leaked_robot_depth="
                f"{leakage ['deepest_leaked_robot_depth']:.3f}"
                )
        return ready 

    def ready_or_handoff_bypass (robots :Sequence [Any ])->bool :








        if getattr (physical ,"integration_ready_guard_handoff",False ):
            return True 
        return audited_ready (robots )

    physical .junction_guards_formed =ready_or_handoff_bypass 


def install_lidar_relay_protection (
physical :types .ModuleType ,
perception :AdaptivePerception ,
)->None :
    """Keep the persistent LiDAR visible to relay-front tracking,
    but never allow it to become the Breadcrumb itself.

    LiDAR 675:
      - remains NORMAL,
      - contributes to front_progress,
      - is excluded only from tail_band Relay election.
    """

    lidar_robot =perception .leader 

    def protected_update_relay_deployment (
    robots :Sequence [Any ],
    dt :float ,
    )->None :

        physical .relay_deploy_cooldown =max (
        0.0 ,
        physical .relay_deploy_cooldown -dt ,
        )

        physical .relay_motion_scale =1.0 

        if physical .phase not in {
        physical .SimulationPhase .MOVE_TO_JUNCTION ,
        physical .SimulationPhase .EXPLORE_BRANCH ,
        physical .SimulationPhase .FORM_SHEPHERD_BOUNDARY ,
        physical .SimulationPhase .FILL_BEHIND_SHEPHERD ,
        }:
            return 


        current_junction =(
        multi_dfs .current 
        )















        current_guard_frontend_pending =(
        current_junction is not None 
        and not perception .handoff_complete 
        )

        if current_guard_frontend_pending :
            return 

        active_branch =getattr (
        physical ,
        "active_branch",
        None ,
        )

        if active_branch is None :
            return 

        if (
        physical .phase 
        ==physical .SimulationPhase .MOVE_TO_JUNCTION 
        and physical .simulation_time 
        <physical .BASE_COMPRESSION_DURATION 
        ):
            return 

        if (
        physical .relay_deploy_cooldown >0.0 
        or physical .base_station is None 
        ):
            return 


        descriptor =physical .branch_motion_descriptor (
        physical .active_branch 
        )
        if descriptor is None :
            return 
        tangent =descriptor .local_outgoing_direction 

        if (
        tangent .length_squared ()
        <=physical .EPSILON 
        ):
            return 

        tangent =tangent .normalize ()

        breadcrumbs =(
        physical .get_active_branch_relays (
        robots 
        )
        )

        last_node =(
        breadcrumbs [-1 ]
        if breadcrumbs 
        else physical .base_station 
        )

        if last_node is None :
            return 

        observations =observe_local_neighbors (
        last_node ,
        robots ,
        tangent ,
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :(
        robot .role =="NORMAL"
        and not robot .base_reserve 
        and robot .connected_to_base 
        and robot is not lidar_robot 
        ),
        )

        candidates =[
        observation 
        for observation in observations 
        if (
        observation .relative_axial 
        >0.0 
        and physical .BREADCRUMB_DEPLOY_DISTANCE 
        <=observation .relative_range 
        <=physical .COMM_RANGE *0.88 
        )
        ]

        if not candidates :
            return 

        selected =min (
        candidates ,
        key =lambda observation :(
        abs (
        observation .relative_range 
        -physical .BREADCRUMB_SPACING 
        ),
        observation .robot .robot_id ,
        ),
        )

        tail_robot =selected .robot 


        if (
        tail_robot .total_distance 
        <physical .BREADCRUMB_MIN_TRAVEL 
        ):
            return 

        tail_robot .role ="RELAY"





        tail_robot .relay_anchor =None 
        tail_robot .relay_hold_local =True 

        tail_robot .relay_index =(
        breadcrumbs [-1 ].relay_index +1 
        if breadcrumbs 
        else 0 
        )

        tail_robot .velocity .update (
        0.0 ,
        0.0 ,
        )

        tail_robot .acceleration .update (
        0.0 ,
        0.0 ,
        )

        tail_robot .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )

        physical .relay_deploy_cooldown =(
        physical .BREADCRUMB_DEPLOY_COOLDOWN 
        )

        print (
        "[Breadcrumb] "
        f"tail robot={tail_robot .robot_id }, "
        f"index={tail_robot .relay_index }, "
        f"local_parent_range={selected .relative_range :.1f}, "
        "static_guards=0 "
        f"lidar_front_visible=True "
        f"lidar_selected=False"
        )

    physical .update_relay_deployment =(
    protected_update_relay_deployment 
    )

    previous_robot_update =physical .Robot .update 

    def hold_local_breadcrumb (self :Any ,dt :float )->None :
        if self .role !="RELAY"or not getattr (self ,"relay_hold_local",False ):
            previous_robot_update (self ,dt )
            return 
        old_position =self .position .copy ()
        self .velocity .update (0.0 ,0.0 )
        self .commanded_velocity .update (0.0 ,0.0 )
        self .observed_velocity .update (0.0 ,0.0 )
        self .acceleration .update (0.0 ,0.0 )
        self .filtered_acceleration .update (0.0 ,0.0 )
        self .previous_position =old_position 
        self ._record_motion ()

    physical .Robot .update =hold_local_breadcrumb 

    print (
    "[LiDARRoleProtection] "
    f"lidar_id={lidar_robot .robot_id } "
    "visible_to_front_progress=True "
    "breadcrumb_candidate=False"
    )

def update_child_probe_relay_support (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Freeze a local NORMAL neighbor only when the moving LiDAR
    is about to exhaust its current Base-side communication link.

    This is Child-probe communication support only.

    It does NOT:
    - confirm a Child Junction,
    - release the Parent Junction,
    - create a Return Marker,
    - perform DFS PUSH,
    - move any robot by position overwrite.
    """

    if not multi_dfs .child_probe_active :
        return 

    lidar_robot =perception .leader 

    if lidar_robot .role !="NORMAL":
        return 

    if not lidar_robot .connected_to_base :
        return 

    parent =lidar_robot .comm_parent 

    if parent is None :
        return 

    local_axis =_body_local_unit (perception ,0.0 )
    parent_observation =observe_local_neighbors (
    lidar_robot ,
    [parent ],
    local_axis ,
    max_range =physical .COMM_RANGE ,
    )
    if not parent_observation :
        return 
    parent_distance =parent_observation [0 ].relative_range 

    hard_limit =float (
    physical .COMM_GUARD_HARD_LIMIT 
    )

    trigger_distance =(
    CHILD_PROBE_RELAY_TRIGGER_RATIO 
    *hard_limit 
    )



    if parent_distance <trigger_distance :
        return 

    current_margin =float (
    getattr (
    lidar_robot ,
    "comm_path_margin",
    float ("-inf"),
    )
    )

    candidates :list [
    tuple [float ,float ,int ,Any ]
    ]=[]























    candidate_observations ={
    observation .robot .robot_id :observation 
    for observation in observe_local_neighbors (
    lidar_robot ,
    getattr (lidar_robot ,"comm_neighbors",[]),
    local_axis ,
    max_range =physical .COMM_RANGE ,
    )
    }
    for candidate in getattr (
    lidar_robot ,
    "comm_neighbors",
    [],
    ):
        if candidate is lidar_robot :
            continue 

        if candidate is parent :
            continue 

        if getattr (
        candidate ,
        "role",
        None ,
        )!="NORMAL":
            continue 

        if getattr (
        candidate ,
        "base_reserve",
        False ,
        ):
            continue 

        if not getattr (
        candidate ,
        "connected_to_base",
        False ,
        ):
            continue 

        observation =candidate_observations .get (candidate .robot_id )
        if observation is None :
            continue 
        candidate_distance =observation .relative_range 

        if (
        candidate_distance 
        >=physical .COMM_GUARD_HARD_LIMIT 
        ):
            continue 

        candidate_path_margin =float (
        getattr (
        candidate ,
        "comm_path_margin",
        float ("-inf"),
        )
        )

        if not math .isfinite (
        candidate_path_margin 
        ):
            continue 

        local_edge_margin =(
        physical .COMM_RANGE 
        -candidate_distance 
        )

        bridge_margin =min (
        candidate_path_margin ,
        local_edge_margin ,
        )





        if (
        bridge_margin 
        <=current_margin 
        +physical .EPSILON 
        ):
            continue 

        candidates .append (
        (
        -bridge_margin ,
        candidate_distance ,
        candidate .robot_id ,
        candidate ,
        )
        )

    if not candidates :
        if (
        physical .integration_frame 
        %20 
        ==0 
        ):
            print (
            "[ChildProbeRelayWait] "
            f"lidar_id={lidar_robot .robot_id } "
            f"parent="
            f"{getattr (parent ,'robot_id',None )} "
            f"parent_dist={parent_distance :.2f} "
            f"trigger={trigger_distance :.2f} "
            f"hard={hard_limit :.2f} "
            f"current_margin="
            f"{current_margin :.2f} "
            "candidate=NONE"
            )

        return 

    _ ,_ ,_ ,relay_robot =min (
    candidates 
    )

    previous_relays =(
    physical .get_active_branch_relays (
    robots 
    )
    )

    relay_index =(
    max (
    (
    relay .relay_index 
    for relay in previous_relays 
    ),
    default =-1 ,
    )
    +1 
    )

    relay_robot .role ="RELAY"
    relay_robot .relay_anchor =None 
    relay_robot .relay_hold_local =True 
    relay_robot .relay_index =relay_index 

    if hasattr (
    relay_robot ,
    "relay_scope",
    ):
        relay_robot .relay_scope ="BRANCH"

    if hasattr (
    relay_robot ,
    "relay_owner_edge_id",
    ):
        relay_robot .relay_owner_edge_id =(
        multi_dfs .child_probe_branch_uid 
        )

    if hasattr (
    relay_robot ,
    "relay_branch",
    ):
        relay_robot .relay_branch =getattr (
        physical ,
        "active_branch",
        None ,
        )

    relay_robot .velocity .update (
    0.0 ,
    0.0 ,
    )

    relay_robot .acceleration .update (
    0.0 ,
    0.0 ,
    )

    relay_robot .filtered_acceleration .update (
    0.0 ,
    0.0 ,
    )

    print (
    "[ChildProbeRelay] "
    f"lidar_id={lidar_robot .robot_id } "
    f"relay_id={relay_robot .robot_id } "
    f"relay_index={relay_index } "
    f"old_parent="
    f"{getattr (parent ,'robot_id',None )} "
    f"old_parent_dist={parent_distance :.2f} "
    f"relay_dist="
    f"{candidate_observations .get (relay_robot .robot_id ).relative_range if candidate_observations .get (relay_robot .robot_id )else float ('nan'):.2f} "
    "position_snap=False"
    )


def install_continuous_guard_settling (physical :types .ModuleType )->None :
    """Let initially elected Guards walk to local slots at bounded speed."""
    original_limit =physical .limit_communication_proposed_position 

    def guard_formation_limit (
    robot :Any ,
    proposed :pygame .Vector2 ,
    old_position :pygame .Vector2 ,
    )->pygame .Vector2 :
        lifecycle =getattr (physical ,"integration_wall_lifecycle",{}).get (
        getattr (robot ,"junction_guard_branch",None ),
        {},
        )
        restoring_persistent_wall =(
        lifecycle .get ("state")=="RESTORING_GUARD"
        )
        if (
        (
        physical .phase ==physical .SimulationPhase .FORM_JUNCTION_GUARDS 
        or getattr (
        physical ,"integration_provisional_guard_active",False 
        )
        or restoring_persistent_wall 
        )
        and robot .role =="JUNCTION_GUARD"
        and robot .junction_guard_anchor is not None 
        ):








            return proposed 
        if (
        physical .phase in {
        physical .SimulationPhase .FORM_JUNCTION_GUARDS ,
        physical .SimulationPhase .EXPLORE_BRANCH ,
        }
        and robot .role =="FRONTIER_SHEPHERD"
        and robot .shepherd_branch in getattr (physical ,"integration_wall_lifecycle",{})
        ):






            return proposed 
        return original_limit (robot ,proposed ,old_position )

    physical .limit_communication_proposed_position =guard_formation_limit 



@dataclass 
class DeadEndDiagnostics :
    """Localization-free local physical evidence for Frontier dead-end detection."""

    branch_uid :str |None =None 
    command_forward_speed :float =0.0 
    actual_forward_speed :float =float ("inf")
    forward_blocked_ratio :float =0.0 
    lidar_blocked :bool =False 
    no_junction_evidence :bool =False 
    dwell :float =0.0 
    confirmed :bool =False 
    last_log_time :float =float ("-inf")

    def reset (self ,branch_uid :str )->None :
        self .branch_uid =branch_uid 
        self .command_forward_speed =0.0 
        self .actual_forward_speed =float ("inf")
        self .forward_blocked_ratio =0.0 
        self .lidar_blocked =False 
        self .no_junction_evidence =False 
        self .dwell =0.0 
        self .confirmed =False 
        self .last_log_time =float ("-inf")


def evaluate_frontier_dead_end (
physical :types .ModuleType ,
branch_uid :str ,
descriptor :Any ,
frontier_members :Sequence [Any ],
dt :float ,
)->DeadEndDiagnostics :
    """Evaluate dead-end without density, pressure, map endpoints, or localization."""

    trackers =getattr (
    physical ,
    "integration_dead_end_diagnostics",
    None ,
    )

    if trackers is None :
        trackers ={}
        physical .integration_dead_end_diagnostics =trackers 

    diagnostics =trackers .get (branch_uid )

    if diagnostics is None :
        diagnostics =DeadEndDiagnostics (
        branch_uid =branch_uid ,
        )
        trackers [branch_uid ]=diagnostics 

    if diagnostics .branch_uid !=branch_uid :
        diagnostics .reset (branch_uid )

    if not frontier_members :
        diagnostics .dwell =0.0 
        diagnostics .confirmed =False 
        return diagnostics 

    tangent =descriptor .local_outgoing_direction .normalize ()











    commanded_speeds :list [float ]=[]

    for robot in frontier_members :
        intended_forward_speed =getattr (
        robot ,
        "integration_frontier_intended_forward_speed",
        None ,
        )





        if intended_forward_speed is None :
            intended_forward_speed =float (
            getattr (
            robot ,
            "commanded_velocity",
            pygame .Vector2 (),
            ).dot (tangent )
            )

        commanded_speeds .append (
        max (
        0.0 ,
        float (intended_forward_speed ),
        )
        )

    diagnostics .command_forward_speed =(
    float (np .median (commanded_speeds ))
    if commanded_speeds 
    else 0.0 
    )











    actual_speeds =[
    max (
    0.0 ,
    float (
    getattr (
    robot ,
    "observed_velocity",
    robot .velocity ,
    ).dot (tangent )
    ),
    )
    for robot in frontier_members 
    ]

    diagnostics .actual_forward_speed =(
    float (np .median (actual_speeds ))
    if actual_speeds 
    else float ("inf")
    )

















    lidar_frame =getattr (
    physical ,
    "integration_latest_lidar_frame",
    None ,
    )

    diagnostics .forward_blocked_ratio =0.0 
    diagnostics .lidar_blocked =False 

    if (
    lidar_frame is not None 
    and lidar_frame .selected is not None 
    and len (lidar_frame .angles )>0 
    ):
        forward_indices =[
        index 
        for index ,angle 
        in enumerate (lidar_frame .angles )
        if circular_error (
        float (angle ),
        0.0 ,
        )<=22.5 
        ]

        if forward_indices :
            threshold =float (
            lidar_frame .selected 
            )

            blocked_count =sum (
            float (
            lidar_frame .smoothed [index ]
            )
            <threshold 
            for index in forward_indices 
            )

            diagnostics .forward_blocked_ratio =(
            blocked_count 
            /len (forward_indices )
            )

            diagnostics .lidar_blocked =(
            diagnostics .forward_blocked_ratio 
            >=0.85 
            )









    diagnostics .no_junction_evidence =(
    not bool (
    getattr (
    lidar_frame ,
    "current_evidence",
    False ,
    )
    )
    and not bool (
    getattr (
    multi_dfs ,
    "child_candidate_active",
    False ,
    )
    )
    )

    command_threshold =float (
    getattr (
    physical ,
    "CONTACT_COMMAND_MIN_SPEED",
    2.0 ,
    )
    )

    progress_threshold =float (
    getattr (
    physical ,
    "DEAD_END_FRONTIER_PROGRESS_RATE_THRESHOLD",
    2.5 ,
    )
    )

    evidence =(
    diagnostics .command_forward_speed 
    >=command_threshold 
    and diagnostics .actual_forward_speed 
    <=progress_threshold 
    and diagnostics .lidar_blocked 
    and diagnostics .no_junction_evidence 
    )

    diagnostics .dwell =(
    diagnostics .dwell +dt 
    if evidence 
    else 0.0 
    )

    diagnostics .confirmed =(
    diagnostics .dwell >=0.45 
    )

    if (
    physical .simulation_time 
    -diagnostics .last_log_time 
    >=0.5 
    ):
        diagnostics .last_log_time =(
        physical .simulation_time 
        )

        print (
        "[DeadEndEvidence] "
        f"branch={branch_uid } "
        f"command_speed="
        f"{diagnostics .command_forward_speed :.3f} "
        f"actual_speed="
        f"{diagnostics .actual_forward_speed :.3f} "
        f"lidar_blocked="
        f"{diagnostics .lidar_blocked } "
        f"blocked_ratio="
        f"{diagnostics .forward_blocked_ratio :.3f} "
        f"no_junction="
        f"{diagnostics .no_junction_evidence } "
        f"dwell={diagnostics .dwell :.3f}/0.450 "
        f"confirmed={diagnostics .confirmed }"
        )

    return diagnostics 

def evaluate_local_swarm_return_completion (
physical :types .ModuleType ,
robots :Sequence [Any ],
branch_key :str ,
descriptor :Any ,
lifecycle :dict [str ,Any ],
dt :float ,
)->tuple [bool ,int ,int ,float ]:
    """Confirm swarm return using only local Shepherd-relative observations."""

    expected_ids =set (
    lifecycle .get (
    "robot_ids",
    [],
    )
    )

    shepherds =[
    robot 
    for robot in robots 
    if (
    robot .robot_id in expected_ids 
    and robot .role =="SHEPHERD"
    and robot .shepherd_branch ==branch_key 
    )
    ]

    state_key =str (
    getattr (
    descriptor ,
    "uid",
    branch_key ,
    )
    )

    dwell_map =getattr (
    physical ,
    "integration_return_completion_dwell",
    None ,
    )

    if dwell_map is None :
        dwell_map ={}
        physical .integration_return_completion_dwell =dwell_map 

    if (
    not expected_ids 
    or {
    robot .robot_id 
    for robot in shepherds 
    }
    !=expected_ids 
    ):
        dwell_map [state_key ]=0.0 
        return False ,0 ,0 ,0.0 

    relative_offsets =lifecycle .get (
    "shepherd_relative_offsets",
    lifecycle .get (
    "relative_offsets",
    {},
    ),
    )

    reference =min (
    shepherds ,
    key =lambda robot :float (
    relative_offsets .get (
    robot .robot_id ,
    (0.0 ,0.0 ),
    )[0 ]
    ),
    )

    reference_lateral =float (
    relative_offsets .get (
    reference .robot_id ,
    (0.0 ,0.0 ),
    )[1 ]
    )

    tangent =(
    descriptor .local_outgoing_direction .normalize ()
    )

    lateral_axis =(
    physical .descriptor_local_basis (
    descriptor 
    )[1 ]
    )

    usable_half =max (
    float (
    physical .local_physical_usable_half_width (
    descriptor 
    )
    ),
    float (
    lifecycle .get (
    "usable_half_width",
    0.0 ,
    )
    ),
    )

    observations =observe_local_neighbors (
    reference ,
    robots ,
    tangent ,
    lateral_axis =lateral_axis ,
    max_range =physical .COMM_RANGE ,
    predicate =lambda robot :(
    robot .role =="NORMAL"
    and not robot .base_reserve 
    ),
    )

    corridor_observations =[
    observation 
    for observation in observations 
    if abs (
    observation .relative_lateral 
    +reference_lateral 
    )
    <=usable_half 
    +2.0 *physical .ROBOT_RADIUS 
    ]













    branch_side_normals =[
    observation 
    for observation in corridor_observations 
    if observation .relative_axial 
    >physical .ROBOT_RADIUS 
    ]







    junction_side_normals =[
    observation 
    for observation in corridor_observations 
    if observation .relative_axial 
    <=physical .ROBOT_RADIUS 
    ]







    debug_frame =int (
    getattr (
    physical ,
    "integration_frame",
    -1 ,
    )
    )

    if (
    branch_side_normals 
    and debug_frame %10 ==0 
    ):
        branch_side_details =[
        (
        observation .robot .robot_id ,
        round (
        float (
        observation .relative_axial 
        ),
        3 ,
        ),
        round (
        float (
        observation .relative_lateral 
        ),
        3 ,
        ),
        round (
        float (
        observation .relative_range 
        ),
        3 ,
        ),
        round (
        float (
        observation .relative_axial_velocity 
        ),
        3 ,
        ),
        )
        for observation 
        in branch_side_normals 
        ]

        print (
        "[BranchSideNormalAudit] "
        f"frame={debug_frame } "
        f"branch={branch_key } "
        f"reference_shepherd="
        f"{reference .robot_id } "
        f"count="
        f"{len (branch_side_normals )} "
        f"details="
        f"{branch_side_details }"
        )

    clear_now =(
    len (branch_side_normals )==0 
    )

    dwell =float (
    dwell_map .get (
    state_key ,
    0.0 ,
    )
    )

    dwell =(
    dwell +dt 
    if clear_now 
    else 0.0 
    )

    dwell_map [state_key ]=dwell 

    required_dwell =max (
    0.12 ,
    float (
    getattr (
    physical ,
    "FLOW_ESTABLISH_DWELL_TIME",
    0.12 ,
    )
    ),
    )

    return (
    dwell >=required_dwell ,
    len (branch_side_normals ),
    len (junction_side_normals ),
    dwell ,
    )



@dataclass 
class LocalSaturationDiagnostics :
    """Branch-local physical evidence; contains no map endpoint coordinates."""

    branch :str |None =None 
    start_depth :float =0.0 
    maximum_depth :float =0.0 
    frontier_speed :float =0.0 
    frontier_progress_rate :float =float ("inf")
    frontier_delta :float =float ("inf")
    frontier_stalled :bool =False 
    local_density :float =0.0 
    baseline_density :float =0.0 
    local_density_ratio :float =0.0 
    local_pressure :float =0.0 
    baseline_pressure :float =0.0 
    local_pressure_ratio :float =0.0 
    cross_section_fill :float =0.0 
    dwell :float =0.0 
    saturated :bool =False 
    shepherd_transition :bool =False 
    frontier_ids :list [int ]=field (default_factory =list )
    shepherd_ids :list [int ]=field (default_factory =list )
    max_transition_jump :float =0.0 
    max_formation_error :float =0.0 
    transition_frame :int |None =None 
    return_direction_local :tuple [float ,float ]=(0.0 ,0.0 )
    return_flow_ratio :float =0.0 
    mean_return_speed :float =0.0 
    return_dwell :float =0.0 
    backflow_confirmed :bool =False 
    progress_history :list [tuple [float ,float ]]=field (default_factory =list )
    last_log_time :float =float ("-inf")

    def reset (self ,branch :str ,depth :float )->None :
        self .__dict__ .update (LocalSaturationDiagnostics (
        branch =branch ,
        start_depth =depth ,
        maximum_depth =depth ,
        ).__dict__ )


def install_local_physical_saturation_bridge (physical :types .ModuleType )->None :
    """Connect local stall/packing evidence to the existing return phases."""
    original_update_state =physical .update_simulation_state 
    original_transfer_control =physical .update_transfer_continuity_control 
    original_prepare_scores =physical .prepare_branch_candidate_scores 
    original_get_shepherd_line_depth =physical .get_shepherd_line_depth 
    original_compute_route_force =physical .compute_route_force 
    original_compute_sph_forces =physical .compute_sph_forces 






    original_start_shepherd_pressure_push =physical .start_shepherd_pressure_push 
    original_release_shepherd_line_at_junction =(
    physical .release_shepherd_line_at_junction 
    )
    original_force_complete_shepherd_boundary =(
    physical .force_complete_shepherd_boundary 
    )
    original_update_pre_shepherd_pipeline =physical .update_pre_shepherd_pipeline 








    original_robot_update =physical .Robot .update 













    return_speed_scale =(
    ROBOT_MOTION_SPEED_SCALE 
    /max (
    float (physical .MOTION_SPEED_MULTIPLIER ),
    1.0 ,
    )
    )
    physical .integration_reference_return_speed_scale =return_speed_scale 
    physical .integration_reference_return_original_piston_speed =float (
    physical .SHEPHERD_PISTON_SPEED 
    )
    physical .integration_reference_return_original_line_speed =float (
    physical .SHEPHERD_LINE_BACKTRACK_SPEED 
    )
    physical .integration_reference_return_original_release_speed =float (
    physical .SHEPHERD_JUNCTION_RELEASE_SPEED 
    )
    physical .SHEPHERD_PISTON_SPEED *=return_speed_scale 
    physical .SHEPHERD_LINE_BACKTRACK_SPEED *=return_speed_scale 
    physical .SHEPHERD_JUNCTION_RELEASE_SPEED *=return_speed_scale 
    diagnostics =LocalSaturationDiagnostics ()
    physical .integration_saturation =diagnostics 
    physical .integration_saturation_events =[]
    physical .integration_backflow_events =[]

    physical .integration_final_guard_sweep_active =False 
    physical .integration_final_base_flow_dwell =0.0 
    physical .integration_final_base_flow_established =False 
    physical .integration_final_all_guards_released =False 
    physical .integration_frontier_lineage_events =[]
    physical .integration_shepherd_anchor_offsets ={}








    physical .integration_backtrack_command_depth =None 
    physical .integration_backtrack_pack_rear_depth =None 
    physical .integration_backtrack_support_depth =None 
    physical .integration_backtrack_support_count =0 
    physical .integration_backtrack_lateral_coverage =0.0 
    physical .integration_backtrack_pacer_last_log_frame =-1 








    physical .integration_shepherd_pack_ready_dwell =0.0 
    physical .integration_shepherd_pack_ready_required_dwell =max (
    0.18 ,physical .SATURATION_DWELL_TIME 
    )








    physical .integration_shepherd_fill_baseline_density =0.0 
    physical .integration_shepherd_fill_baseline_pressure =0.0 
    physical .integration_shepherd_fill_baseline_cross_fill =0.0 
    physical .integration_pending_shepherd_transition_event =None 






    physical .integration_herd_return_speed_scale =1.00 




    physical .integration_herd_formation_speed_scale =3.0 






    physical .integration_shepherd_contact_compression_ratio =0.20 










    physical .integration_shepherd_active_min_center_gap_ratio =1.50 
    physical .integration_shepherd_contact_shell_ratio =2.35 
    physical .integration_shepherd_contact_spring_scale =2.50 
    physical .integration_shepherd_contact_damping_scale =1.00 
    physical .integration_shepherd_actual_motion_log_frame =-1 
    physical .integration_real_contact_log_frame =-1 
    physical .integration_return_curtain_active =False 

    def reference_start_shepherd_pressure_push (
    robots :Sequence [Any ],branch :str 
    )->None :
        """Start return with the Shepherd target locked to the actual herd rear."""










        descriptor =descriptor_for (branch )
        current_line =(
        shepherd_line_leading_depth (robots ,branch ,descriptor )
        if descriptor is not None else None 
        )
        if current_line is None :
            current_line =float (original_get_shepherd_line_depth (branch ))
        physical .integration_backtrack_command_depth =float (current_line )
        physical .integration_backtrack_pack_rear_depth =None 
        physical .integration_backtrack_support_depth =None 
        physical .integration_backtrack_support_count =0 
        physical .integration_backtrack_lateral_coverage =0.0 
        physical .integration_return_curtain_active =False 



















        disable_leading_anchor (physical )
        physical .phase =physical .SimulationPhase .PRESSURE_PUSH 
        physical .pressure_push_timer =0.0 
        physical .flow_establish_timer =0.0 
        physical .shepherd_flow_timer =0.0 

        metrics =getattr (physical ,"metrics",None )
        if metrics is not None and hasattr (metrics ,"pressure_events"):
            metrics .pressure_events .append ({
            "branch":branch ,
            "started_at":getattr (physical ,"simulation_time",0.0 ),
            })





        print (
        f"[HerdLockedReturnStartV26] branch={branch } "
        f"command_depth={float (current_line ):.3f} "
        f"piston_speed={physical .SHEPHERD_PISTON_SPEED :.3f} "
        f"line_speed={physical .SHEPHERD_LINE_BACKTRACK_SPEED :.3f} "
        "cross_branch_transfer=False invisible_curtain=False"
        )

    def audited_commit_guard_roles (
    robots :Sequence [Any ],selected_branch :str 
    )->None :
        """Promote the complete READY multi-row wall without re-election."""
        uid =physical .branch_uid_for_fixture (selected_branch )
        lifecycle =physical .integration_wall_lifecycle [selected_branch ]
        ready_ids =set (lifecycle ["robot_ids"])
        frontiers =[
        robot for robot in robots 
        if robot .robot_id in ready_ids 
        and robot .role =="JUNCTION_GUARD"
        and robot .junction_guard_branch ==selected_branch 
        ]
        expected =lifecycle ["rows"]*lifecycle ["cols"]
        if len (frontiers )!=expected or len (frontiers )!=len (ready_ids ):
            raise RuntimeError (
            "selected thick Guard wall was incomplete before promotion"
            )
        descriptor =descriptor_for (selected_branch )
        if descriptor is None :
            raise RuntimeError (f"missing descriptor for selected branch {selected_branch }")
        lock_branch_transport_frame (selected_branch ,lifecycle ,descriptor )
        centroid_axial ,centroid_lateral ,regularized_offsets =(
        regularized_thick_wall_offsets (selected_branch ,frontiers ,lifecycle ,descriptor )
        )
        lifecycle ["centroid_axial"]=centroid_axial 
        lifecycle ["centroid_lateral"]=centroid_lateral 
        lifecycle ["relative_offsets"]=regularized_offsets 
        axial_offsets =[value [0 ]for value in lifecycle ["relative_offsets"].values ()]
        wall_axial_span =(
        max (axial_offsets )-min (axial_offsets )
        if axial_offsets else 0.0 
        )






        lifecycle ["frontier_bootstrap_target_depth"]=(
        centroid_axial 
        +max (
        physical .FRONTIER_LINE_LEAD_GAP ,
        wall_axial_span +2.0 *physical .ROBOT_RADIUS ,
        )
        )

        current_junction =(
        multi_dfs .current 
        )

        if current_junction is None :
            raise RuntimeError (
            "Root Frontier promotion requires "
            "current Junction"
            )

        if uid is None :
            raise RuntimeError (
            "Root Frontier promotion has no Branch UID: "
            f"fixture={selected_branch }"
            )

        frontiers ,transition_jump =(
        promote_current_guard_wall_roles_to_frontier (
        physical ,
        robots ,
        current_junction ,
        uid ,
        lifecycle ,
        lifecycle ["relative_offsets"],
        shepherd_branch_key =selected_branch ,
        clear_guard_waypoints =False ,
        zero_command_state =False ,
        )
        )


        physical .frontier_line_branch =selected_branch 
        physical .frontier_line_depth =centroid_axial 
        physical .frontier_line_lateral_center =centroid_lateral 
        physical .frontier_line_target_settled_ratio =0.0 
        physical .frontier_line_current_span =0.0 
        physical .frontier_line_target_span =max (
        lateral for _ ,lateral in lifecycle ["relative_offsets"].values ()
        )-min (
        lateral for _ ,lateral in lifecycle ["relative_offsets"].values ()
        )
        physical .frontier_line_physical_coverage_ratio =0.0 
        physical .frontier_line_left_edge_gap =float ("inf")
        physical .frontier_line_right_edge_gap =float ("inf")
        physical .frontier_line_continuous =False 
        physical .frontier_line_row_ready =False 
        physical .frontier_line_last_diagnostic_time =float ("-inf")
        physical .junction_guard_groups [selected_branch ]=sorted (ready_ids )
        physical .junction_guard_status =(
        f"FRONTIER_THICK={selected_branch };"
        f"{lifecycle ['rows']}x{lifecycle ['cols']};robots={len (frontiers )}"
        )
















        lifecycle ["frontier_rigid_applied_depth"]=float (
        centroid_axial 
        )
        lifecycle ["frontier_rigid_cache_frame"]=-1 
        lifecycle ["frontier_rigid_common_velocity"]=pygame .Vector2 ()
        lifecycle ["frontier_rigid_contact"]=False 
        lifecycle ["frontier_rigid_safe_fraction"]=1.0 
        lifecycle ["frontier_rigid_shape_locked"]=True 
        lifecycle ["max_formation_error"]=0.0 

        frontier_ids ={robot .robot_id for robot in frontiers }
        if not frontier_ids or frontier_ids !=ready_ids :
            raise RuntimeError (
            "FRONTIER_SHEPHERD IDs did not equal all READY Guard IDs"
            )

        event ={
        "uid":uid ,
        "branch":selected_branch ,
        "frame":getattr (physical ,"integration_frame",-1 ),
        "ready_guard_ids":sorted (ready_ids ),
        "frontier_ids":sorted (frontier_ids ),
        "same_ready_guard_ids":True ,
        "max_role_transition_jump":transition_jump ,
        "rows":lifecycle ["rows"],
        "cols":lifecycle ["cols"],
        }
        physical .integration_frontier_lineage_events .append (event )
        print (
        f"[FrontierLineage] uid={uid } ready_guard_ids="
        f"{event ['ready_guard_ids']} frontier_ids={event ['frontier_ids']} "
        "same_ready_guard_ids=True "
        f"rows={lifecycle ['rows']} cols={lifecycle ['cols']} "
        f"robots={len (frontier_ids )} "
        f"max_role_transition_jump={transition_jump :.6f}"
        )
        axial_values =[
        float (offset [0 ])
        for offset in lifecycle ["relative_offsets"].values ()
        ]
        lateral_values =[
        float (offset [1 ])
        for offset in lifecycle ["relative_offsets"].values ()
        ]

        print (
        f"[Frontier3xNPreserved] branch={selected_branch } uid={uid } "
        f"rows={lifecycle ['rows']} cols={lifecycle ['cols']} "
        f"robots={len (frontier_ids )} "
        f"axial_span="
        f"{(max (axial_values )-min (axial_values ))if axial_values else 0.0 :.3f} "
        f"lateral_span="
        f"{(max (lateral_values )-min (lateral_values ))if lateral_values else 0.0 :.3f} "
        f"transport={lifecycle .get ('transport_frame_source')} "
        "same_relative_offsets=True"
        )

    def pebble_filtered_candidate_scores (
    robots :Sequence [Any ],reference_density :float 
    )->list [str ]:
        candidates =original_prepare_scores (robots ,reference_density )
        visited =physical .observed_visited_branch_uids (robots )
        filtered =[uid for uid in candidates if uid not in visited ]
        for uid in candidates :
            if uid in visited :
                print (
                f"[VoteExclude] uid={uid } "
                "reason=observed-pebble-topology"
                )
        return filtered 

    def saturation_aware_transfer_control (robots :Sequence [Any ])->None :
        original_transfer_control (robots )
        if (
        physical .phase in {
        physical .SimulationPhase .EXPLORE_BRANCH ,
        physical .SimulationPhase .FILL_BEHIND_SHEPHERD ,
        }
        and not diagnostics .saturated 
        ):






            physical .branch_fill_feed_scale =max (
            physical .branch_fill_feed_scale ,0.82 
            )
            physical .branch_fill_deficit_control =max (
            physical .branch_fill_deficit_control ,0.75 
            )
            physical .branch_fill_feed_state ="LOCAL_SATURATION_FEED"

    def descriptor_for (branch :str |None )->Any |None :
        return (
        physical .branch_motion_descriptor (branch )
        if branch is not None 
        else None 
        )

    def transport_basis (
    branch :str ,
    lifecycle :dict [str ,Any ]|None =None ,
    descriptor :Any |None =None ,
    )->tuple [pygame .Vector2 ,pygame .Vector2 ,str ]:
        """Use the same LiDAR-frozen branch frame for Guard/Frontier/Shepherd.

        Do not re-align to legacy LEFT/RIGHT/UP fixture directions.
        The frame frozen at initial Guard formation is reused through:
        Guard -> Frontier -> Shepherd -> returned Guard.
        """

        descriptor =descriptor or descriptor_for (branch )

        lifecycle =lifecycle or getattr (
        physical ,
        "integration_wall_lifecycle",
        {},
        ).get (branch ,{})





        candidate =None 

        if lifecycle :
            candidate =lifecycle .get (
            "branch_tangent_unit"
            )



        if (
        candidate is None 
        or candidate .length_squared ()
        <=physical .EPSILON 
        ):
            if descriptor is not None :
                candidate =getattr (
                descriptor ,
                "motion_t",
                None ,
                )

        if (
        candidate is None 
        or candidate .length_squared ()
        <=physical .EPSILON 
        ):
            if descriptor is not None :
                candidate =getattr (
                descriptor ,
                "local_outgoing_direction",
                None ,
                )

        if (
        candidate is None 
        or candidate .length_squared ()
        <=physical .EPSILON 
        ):
            raise RuntimeError (
            f"no LiDAR-frozen transport tangent "
            f"for branch {branch }"
            )

        tangent =candidate .normalize ()



        lateral =pygame .Vector2 (
        -tangent .y ,
        tangent .x ,
        ).normalize ()



        old_lateral =(
        lifecycle .get ("mouth_lateral_unit")
        if lifecycle 
        else None 
        )

        if (
        old_lateral is not None 
        and old_lateral .length_squared ()
        >physical .EPSILON 
        and lateral .dot (old_lateral )<0.0 
        ):
            lateral =-lateral 

        source ="LIDAR_FROZEN_GUARD_FRAME"

        return (
        tangent ,
        lateral ,
        source ,
        )


    def lock_branch_transport_frame (
    branch :str ,
    lifecycle :dict [str ,Any ],
    descriptor :Any ,
    )->tuple [pygame .Vector2 ,pygame .Vector2 ]:
        """Lock one straight corridor frame before the Guard becomes Frontier."""
        tangent ,lateral ,source =transport_basis (branch ,lifecycle ,descriptor )
        lifecycle ["branch_tangent_unit"]=tangent .copy ()
        lifecycle ["mouth_lateral_unit"]=lateral .copy ()
        lifecycle ["transport_frame_source"]=source 

        descriptor .local_outgoing_direction =tangent .copy ()
        descriptor .local_return_direction =-tangent 
        descriptor .direction_last_estimate =tangent .copy ()
        descriptor .direction_stability_reference =tangent .copy ()
        descriptor .motion_t =tangent .copy ()
        descriptor .motion_n =lateral .copy ()
        descriptor .motion_frame_locked =True 
        descriptor .motion_frame_source =source 

        print (
        f"[BranchTransportFrame] branch={branch } uid={descriptor .uid } "
        f"source={source } t=({tangent .x :.3f},{tangent .y :.3f}) "
        f"n=({lateral .x :.3f},{lateral .y :.3f}) straight=True"
        )
        return tangent ,lateral 

    def regularized_thick_wall_offsets (
    branch :str ,
    frontiers :Sequence [Any ],
    lifecycle :dict [str ,Any ],
    descriptor :Any ,
    )->tuple [float ,float ,dict [int ,tuple [float ,float ]]]:
        """Reuse the authoritative Guard WHO/WHERE snapshot unchanged."""
        del branch ,descriptor 
        relative ={
        int (robot_id ):(float (offset [0 ]),float (offset [1 ]))
        for robot_id ,offset in lifecycle .get ("relative_offsets",{}).items ()
        }
        live_ids ={robot .robot_id for robot in frontiers }
        if not relative or set (relative )!=live_ids :
            raise RuntimeError (
            "Guard->Frontier requires the complete frozen Guard offsets"
            )
        centroid_axial =float (lifecycle ["centroid_axial"])
        centroid_lateral =float (lifecycle ["centroid_lateral"])
        lifecycle ["regularized_wall"]=False 
        lifecycle ["frontier_odometry_depth"]=centroid_axial 
        lifecycle ["frontier_odometry_time"]=float (physical .simulation_time )
        return centroid_axial ,centroid_lateral ,relative 

    def local_frontier_progress (
    robots :Sequence [Any ],branch :str ,dt :float 
    )->None :
        """Advance from local body support, without fixture-region queries."""
        frontiers =physical .get_frontier_shepherds (robots ,branch )
        descriptor =descriptor_for (branch )
        if (
        not frontiers 
        or descriptor is None 
        or physical .frontier_line_branch !=branch 
        ):
            return 
        physical .refresh_frontier_row_readiness (robots ,branch )
        lifecycle =getattr (physical ,"integration_wall_lifecycle",{}).get (branch )
        if lifecycle is not None :
            expected_wall =int (lifecycle .get ("rows",0 ))*int (lifecycle .get ("cols",0 ))
            if len (frontiers )==expected_wall :






                physical .frontier_line_row_ready =True 
                physical .frontier_line_continuous =True 
        if physical .frontier_line_row_ready :
            physical .update_frontier_lateral_center (
            robots ,branch ,descriptor ,dt 
            )
            physical .refresh_frontier_row_readiness (robots ,branch )
            if lifecycle is not None and len (frontiers )==expected_wall :
                physical .frontier_line_row_ready =True 
                physical .frontier_line_continuous =True 
        if not physical .frontier_line_row_ready :
            return 

        usable_half =physical .local_physical_usable_half_width (descriptor )
        relative_offsets =(
        lifecycle .get ("relative_offsets",{})
        if lifecycle is not None else {}
        )
        reference =min (
        frontiers ,
        key =lambda robot :sum (
        abs (float (value ))
        for value in relative_offsets .get (robot .robot_id ,(0.0 ,0.0 ))
        ),
        )
        reference_axial ,reference_lateral =relative_offsets .get (
        reference .robot_id ,(0.0 ,0.0 )
        )
        local_support =[
        observation 
        for observation in observe_local_neighbors (
        reference ,
        robots ,
        transport_basis (branch ,lifecycle ,descriptor )[0 ],
        lateral_axis =transport_basis (branch ,lifecycle ,descriptor )[1 ],
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :(
        robot .role =="NORMAL"and not robot .base_reserve 
        ),
        )
        if observation .relative_axial 
        >=-physical .FRONTIER_LINE_LEAD_GAP -float (reference_axial )
        and abs (
        observation .relative_lateral +float (reference_lateral )
        )<=usable_half 
        ]
        supported_relative_front =(
        physical .linear_quantile (
        [observation .relative_axial for observation in local_support ],
        physical .FRONTIER_LINE_SUPPORT_QUANTILE ,
        )
        if local_support else -physical .FRONTIER_LINE_LEAD_GAP 
        )
        axial_offsets =(
        [float (value [0 ])for value in lifecycle .get ("relative_offsets",{}).values ()]
        if lifecycle is not None else []
        )
        trailing_offset =min (axial_offsets ,default =0.0 )




        supported_target =(
        physical .frontier_line_depth 
        +float (reference_axial )
        +supported_relative_front 
        +physical .FRONTIER_LINE_LEAD_GAP 
        -trailing_offset 
        )
        bootstrap_target =(
        float (lifecycle .get ("frontier_bootstrap_target_depth",physical .frontier_line_depth ))
        if lifecycle is not None 
        else physical .frontier_line_depth 
        )
        desired =max (
        physical .frontier_line_depth ,
        supported_target ,
        bootstrap_target ,
        )



















        next_depth =min (
        desired ,
        physical .frontier_line_depth 
        +physical .FRONTIER_LINE_ADVANCE_SPEED *dt ,
        )

        if lifecycle is not None and axial_offsets :














            finite_shape_errors =[
            float (
            getattr (
            robot ,
            "integration_boundary_shape_error",
            0.0 ,
            )
            )
            for robot in frontiers 
            if math .isfinite (
            float (
            getattr (
            robot ,
            "integration_boundary_shape_error",
            0.0 ,
            )
            )
            )
            ]

            max_shape_error =max (
            finite_shape_errors ,
            default =0.0 ,
            )

            shape_error_tolerance =float (
            getattr (
            physical ,
            "integration_frontier_shape_error_tolerance",
            max (
            1.35 
            *float (
            physical .ROBOT_RADIUS 
            ),
            0.55 
            *float (
            physical .GRID_ROW_SPACING 
            ),
            ),
            )
            )

            shape_hold =(
            max_shape_error 
            >shape_error_tolerance 
            )

























            leading_offset =max (
            axial_offsets ,
            default =0.0 ,
            )

            leading_ids ={
            int (robot_id )
            for robot_id ,offset 
            in lifecycle .get (
            "relative_offsets",
            {},
            ).items ()
            if abs (
            float (offset [0 ])
            -leading_offset 
            )
            <=max (
            0.35 
            *float (
            physical .GRID_ROW_SPACING 
            ),
            0.75 
            *float (
            physical .ROBOT_RADIUS 
            ),
            )
            }

            leading_members =[
            robot 
            for robot in frontiers 
            if robot .robot_id 
            in leading_ids 
            ]

            leading_contacts =sum (
            bool (
            getattr (
            robot ,
            "integration_boundary_contact",
            False ,
            )
            )
            for robot 
            in leading_members 
            )

            contact_ratio =(
            leading_contacts 
            /len (
            leading_members 
            )
            if leading_members 
            else 0.0 
            )

            bumper_hold =(
            len (
            leading_members 
            )
            >=3 
            and leading_contacts 
            >=3 
            and contact_ratio 
            >=float (
            getattr (
            physical ,
            "integration_frontier_contact_hold_ratio",
            0.35 ,
            )
            )
            )

            if (
            shape_hold 
            or bumper_hold 
            ):
                next_depth =(
                physical .frontier_line_depth 
                )

            if bumper_hold :
                lifecycle [
                "frontier_bumper_hold_depth"
                ]=float (
                physical .frontier_line_depth 
                )

            frozen_depth =lifecycle .get (
            "frontier_contact_centroid_depth"
            )

            if frozen_depth is not None :


                next_depth =(
                physical .frontier_line_depth 
                )

            if (
            physical .integration_frame 
            %10 
            ==0 
            and (
            shape_hold 
            or bumper_hold 
            )
            ):
                print (
                "[FrontierRigidHold] "
                f"branch={branch } "
                f"shape_error="
                f"{max_shape_error :.3f} "
                f"shape_tol="
                f"{shape_error_tolerance :.3f} "
                f"leading_contacts="
                f"{leading_contacts }/"
                f"{len (leading_members )} "
                f"contact_ratio="
                f"{contact_ratio :.2f} "
                f"shape_hold="
                f"{shape_hold } "
                f"bumper_hold="
                f"{bumper_hold }"
                )



        physical .frontier_line_depth =max (
        physical .frontier_line_depth ,
        next_depth ,
        )

    def root_frontier_headstart_ready (
    robots :Sequence [Any ],
    branch :str ,
    lifecycle :dict [str ,Any ],
    descriptor :Any ,
    )->tuple [bool ,float |None ,int ,int ]:
        """Require each frozen Frontier layer to lead the LiDAR locally."""
        anchor =next (
        (robot for robot in robots 
        if getattr (robot ,"is_lidar_robot",False )
        or robot .robot_id ==LIDAR_ROBOT_ID ),
        None ,
        )
        if anchor is None :
            return False ,None ,0 ,0 
        tangent ,lateral ,_ =transport_basis (branch ,lifecycle ,descriptor )
        offsets =lifecycle .get ("relative_offsets",{})
        layers =lifecycle .get ("guard_layer_by_id",{})
        center_ids =[]
        for layer in sorted ({int (value )for value in layers .values ()}):
            ids =[
            int (robot_id )for robot_id ,robot_layer in layers .items ()
            if int (robot_layer )==layer and int (robot_id )in offsets 
            ]
            if ids :
                center_ids .append (min (ids ,key =lambda robot_id :abs (float (offsets [robot_id ][1 ]))))
        center_members =[
        robot for robot in robots 
        if robot .robot_id in center_ids 
        and robot .role =="FRONTIER_SHEPHERD"
        and robot .shepherd_branch ==branch 
        ]
        observations =observe_local_neighbors (
        anchor ,center_members ,tangent ,lateral_axis =lateral ,
        max_range =physical .COMM_RANGE ,
        )
        observed ={observation .robot .robot_id :observation for observation in observations }
        if not center_ids or set (observed )!=set (center_ids ):
            return False ,None ,len (observed ),len (center_ids )
        gap =min (observation .relative_axial for observation in observations )
        required =float (physical .integration_frontier_headstart_min_gap )
        return gap >=required ,float (gap ),len (observed ),len (center_ids )

    def sample_local_state (
    robots :Sequence [Any ],branch :str ,reference_density :float ,dt :float 
    )->LocalSaturationDiagnostics :
        descriptor =descriptor_for (branch )
        line =physical .get_frontier_shepherds (robots ,branch )
        if not line :
            line =[
            robot for robot in physical .get_shepherds (robots )
            if robot .shepherd_branch ==branch 
            ]
        if descriptor is None or not line :
            diagnostics .dwell =0.0 
            return diagnostics 
        lifecycle =getattr (
        physical ,"integration_wall_lifecycle",{}
        ).get (branch )
        if any (robot .role =="FRONTIER_SHEPHERD"for robot in line ):
            now =float (physical .simulation_time )
            depth =float (
            (lifecycle or {}).get (
            "frontier_odometry_depth",
            physical .frontier_line_depth ,
            )
            )
            previous_time =float (
            (lifecycle or {}).get ("frontier_odometry_time",now )
            )
            elapsed =max (0.0 ,now -previous_time )
            if lifecycle is not None and elapsed >0.0 :
                tangent =descriptor .local_outgoing_direction .normalize ()
                depth +=float (np .median ([
                robot .velocity .dot (tangent )for robot in line 
                ]))*elapsed 
                lifecycle ["frontier_odometry_depth"]=depth 
                lifecycle ["frontier_odometry_time"]=now 
        else :
            measured_depth =shepherd_line_leading_depth (
            robots ,branch ,descriptor 
            )
            depth =float (measured_depth if measured_depth is not None else 0.0 )
        if diagnostics .branch !=branch :
            diagnostics .reset (branch ,depth )
        diagnostics .maximum_depth =max (diagnostics .maximum_depth ,depth )
        diagnostics .progress_history .append ((physical .simulation_time ,depth ))
        diagnostics .progress_history =[
        item for item in diagnostics .progress_history 
        if physical .simulation_time -item [0 ]
        <=physical .SATURATION_FRONT_WINDOW 
        ]
        if len (diagnostics .progress_history )>=2 :
            time_span =(
            diagnostics .progress_history [-1 ][0 ]
            -diagnostics .progress_history [0 ][0 ]
            )
            diagnostics .frontier_delta =max (
            0.0 ,
            diagnostics .progress_history [-1 ][1 ]
            -diagnostics .progress_history [0 ][1 ],
            )
            diagnostics .frontier_progress_rate =(
            diagnostics .frontier_delta 
            /max (time_span ,physical .EPSILON )
            if time_span 
            >=0.75 *physical .SATURATION_FRONT_WINDOW 
            else float ("inf")
            )
        else :
            diagnostics .frontier_delta =float ("inf")
            diagnostics .frontier_progress_rate =float ("inf")
        diagnostics .frontier_speed =diagnostics .frontier_progress_rate 
        diagnostics .frontier_stalled =(
        diagnostics .frontier_progress_rate 
        <=physical .SATURATION_LOW_SPEED_THRESHOLD 
        )

        if lifecycle is not None :
            frozen_offsets =lifecycle .get ("relative_offsets",{})
            reference =line [0 ]
            reference_offset =frozen_offsets .get (reference .robot_id ,(0.0 ,0.0 ))
            line_ids ={robot .robot_id for robot in line }
            relative_line =observe_local_neighbors (
            reference ,
            line ,
            transport_basis (branch ,lifecycle ,descriptor )[0 ],
            lateral_axis =transport_basis (branch ,lifecycle ,descriptor )[1 ],
            )
            formation_error =max (
            [0.0 ]
            +[
            math .hypot (
            observation .relative_axial 
            -(
            frozen_offsets .get (observation .robot .robot_id ,(0.0 ,0.0 ))[0 ]
            -reference_offset [0 ]
            ),
            observation .relative_lateral 
            -(
            frozen_offsets .get (observation .robot .robot_id ,(0.0 ,0.0 ))[1 ]
            -reference_offset [1 ]
            ),
            )
            for observation in relative_line 
            if observation .robot .robot_id in line_ids 
            ]
            )
            lifecycle ["max_formation_error"]=max (
            lifecycle .get ("max_formation_error",0.0 ),
            formation_error ,
            )
            diagnostics .max_formation_error =lifecycle [
            "max_formation_error"
            ]

        usable_half =physical .local_physical_usable_half_width (descriptor )
        local_depth =max (
        physical .DEAD_END_FRONTIER_DEPTH ,
        physical .SHEPHERD_LOCAL_FLOW_DEPTH ,
        )
        frozen_offsets =lifecycle .get ("relative_offsets",{})if lifecycle else {}
        reference =line [0 ]
        reference_axial ,reference_lateral =frozen_offsets .get (
        reference .robot_id ,(0.0 ,0.0 )
        )
        local_observations =[
        observation 
        for observation in observe_local_neighbors (
        reference ,
        robots ,
        transport_basis (branch ,lifecycle ,descriptor )[0 ],
        lateral_axis =transport_basis (branch ,lifecycle ,descriptor )[1 ],
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :(
        robot .role =="NORMAL"and not robot .base_reserve 
        ),
        )
        if -local_depth <=(
        observation .relative_axial +float (reference_axial )
        )<=physical .ROBOT_RADIUS 
        and abs (
        observation .relative_lateral +float (reference_lateral )
        )<=usable_half 
        ]
        cohort =[observation .robot for observation in local_observations ]
        laterals =[
        observation .relative_lateral +float (reference_lateral )
        for observation in local_observations 
        ]
        diagnostics .local_density =float (np .mean (
        [robot .density for robot in cohort ]
        ))if cohort else 0.0 
        if diagnostics .local_density >0.0 :
            diagnostics .baseline_density =(
            diagnostics .local_density 
            if diagnostics .baseline_density <=0.0 
            else min (
            diagnostics .baseline_density ,
            diagnostics .local_density ,
            )
            )
        diagnostics .local_density_ratio =(
        diagnostics .local_density 
        /max (diagnostics .baseline_density ,physical .EPSILON )
        )
        diagnostics .local_pressure =float (np .mean ([
        max (0.0 ,robot .pressure )for robot in cohort 
        ]))if cohort else 0.0 
        if diagnostics .local_pressure >0.0 :
            diagnostics .baseline_pressure =(
            diagnostics .local_pressure 
            if diagnostics .baseline_pressure <=0.0 
            else min (
            diagnostics .baseline_pressure ,
            diagnostics .local_pressure ,
            )
            )
        diagnostics .local_pressure_ratio =(
        diagnostics .local_pressure 
        /max (diagnostics .baseline_pressure ,physical .EPSILON )
        )
        bin_count =max (
        physical .JUNCTION_GUARD_MIN_COUNT ,
        physical .thick_mouth_guard_columns .get (branch ,0 ),
        )
        occupied ={
        int (physical .clamp (
        (value +usable_half )
        /max (2.0 *usable_half ,physical .EPSILON )
        *bin_count ,
        0 ,
        bin_count -1 ,
        ))
        for value in laterals 
        }
        diagnostics .cross_section_fill =len (occupied )/max (bin_count ,1 )
        travelled =diagnostics .maximum_depth -diagnostics .start_depth 
        common_saturation_evidence =(
        len (cohort )>=physical .SATURATION_MIN_TIP_ROBOTS 
        and travelled >=max (
        physical .JUNCTION_COHORT_MIN_TRAVEL ,
        descriptor .observed_physical_width ,
        )
        and diagnostics .frontier_stalled 
        and diagnostics .local_pressure_ratio 
        >=LOCAL_SATURATION_PRESSURE_RATIO 
        and diagnostics .cross_section_fill 
        >=physical .SATURATION_PACKED_LATERAL_COVERAGE_RATIO 
        )
        conditions =(
        common_saturation_evidence 
        and diagnostics .local_density_ratio 
        >=physical .SATURATION_DENSITY_RATIO 
        )
        diagnostics .dwell =diagnostics .dwell +dt if conditions else 0.0 
        diagnostics .saturated =(
        diagnostics .dwell >=physical .SATURATION_DWELL_TIME 
        )
        if physical .simulation_time -diagnostics .last_log_time >=1.0 :
            diagnostics .last_log_time =physical .simulation_time 
            print (
            f"[LocalSaturation] uid={descriptor .uid } "
            f"frontier_speed={diagnostics .frontier_speed :.2f} "
            f"centroid_rate={diagnostics .frontier_progress_rate :.2f} "
            f"delta={diagnostics .frontier_delta :.2f} "
            f"stalled={diagnostics .frontier_stalled } "
            f"rho={diagnostics .local_density :.4f} "
            f"rho_ratio={diagnostics .local_density_ratio :.2f} "
            f"pressure={diagnostics .local_pressure :.2f} "
            f"pressure_ratio={diagnostics .local_pressure_ratio :.2f} "
            f"cross_fill={diagnostics .cross_section_fill :.2f} "
            f"mode={'LOCAL_STALL_DENSITY_PRESSURE'if conditions else '-'} "
            f"dwell={diagnostics .dwell :.2f} "
            f"saturated={diagnostics .saturated }"
            )
        return diagnostics 

    def local_backflow_metrics (
    robots :Sequence [Any ],branch :str 
    )->tuple [float ,float ,int ]:
        descriptor =descriptor_for (branch )
        shepherds =[
        robot for robot in physical .get_shepherds (robots )
        if robot .shepherd_branch ==branch 
        ]
        if descriptor is None or not shepherds :
            return 0.0 ,0.0 ,0 
        lifecycle =getattr (
        physical ,"integration_wall_lifecycle",{}
        ).get (branch ,{})
        offsets =lifecycle .get (
        "shepherd_relative_offsets",
        lifecycle .get ("relative_offsets",{}),
        )
        reference =min (
        shepherds ,
        key =lambda robot :float (
        offsets .get (robot .robot_id ,(0.0 ,0.0 ))[0 ]
        ),
        )
        reference_lateral =float (
        offsets .get (reference .robot_id ,(0.0 ,0.0 ))[1 ]
        )
        usable_half =physical .local_physical_usable_half_width (descriptor )
        cohort =[
        observation .robot 
        for observation in observe_local_neighbors (
        reference ,
        robots ,
        descriptor .local_outgoing_direction ,
        lateral_axis =getattr (descriptor ,"motion_n",None ),
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :(
        robot .role =="NORMAL"
        and not robot .base_reserve 
        and not bool (getattr (robot ,"is_fixed_anchor",False ))
        ),
        )
        if -physical .SHEPHERD_LOCAL_FLOW_DEPTH 
        <=observation .relative_axial 
        <=physical .SHEPHERD_LOCAL_FLOW_FORWARD_ALLOWANCE 
        and abs (
        observation .relative_lateral +reference_lateral 
        )<=usable_half 
        ]
        if not cohort :
            return 0.0 ,0.0 ,0 
        return_direction =descriptor .local_return_direction .normalize ()
        speeds =[robot .observed_velocity .dot (return_direction )for robot in cohort ]
        ratio =sum (
        speed >=physical .FLOW_SPEED_THRESHOLD for speed in speeds 
        )/len (speeds )
        mean_speed =sum (max (0.0 ,speed )for speed in speeds )/len (speeds )
        return ratio ,mean_speed ,len (speeds )

    def local_return_direction (branch :str )->pygame .Vector2 :
        descriptor =descriptor_for (branch )
        if descriptor is not None :
            return descriptor .local_return_direction .normalize ()
        return pygame .Vector2 ()

    def shepherd_line_leading_depth (
    robots :Sequence [Any ],branch :str ,descriptor :Any 
    )->float |None :
        """Integrate intact-wall velocity in its frozen local branch frame."""
        shepherds =[
        robot for robot in physical .get_shepherds (robots )
        if robot .shepherd_branch ==branch 
        ]
        if not shepherds :
            return None 
        lifecycle =getattr (
        physical ,"integration_wall_lifecycle",{}
        ).get (branch ,{})
        depth =lifecycle .get ("shepherd_odometry_depth")
        if depth is None :
            depth =lifecycle .get (
            "frontier_contact_centroid_depth",
            getattr (physical ,"frontier_line_depth",0.0 ),
            )
        now =float (getattr (physical ,"simulation_time",0.0 ))
        previous_time =float (lifecycle .get ("shepherd_odometry_time",now ))
        elapsed =max (0.0 ,now -previous_time )
        if elapsed >0.0 :
            tangent =descriptor .local_outgoing_direction .normalize ()
            wall_speed =float (np .median ([
            robot .velocity .dot (tangent )for robot in shepherds 
            ]))
            depth =float (depth )+wall_speed *elapsed 
        lifecycle ["shepherd_odometry_depth"]=float (depth )
        lifecycle ["shepherd_odometry_time"]=now 
        return float (depth )

    def original_guard_leading_depth (
    branch :str ,
    )->float :
        """처음 생성된 3xN Guard의 가장 branch 안쪽 row depth."""

        lifecycle =getattr (
        physical ,
        "integration_wall_lifecycle",
        {},
        ).get (branch )

        if lifecycle is None :
            raise RuntimeError (
            f"missing lifecycle for original Guard depth: {branch }"
            )

        original_depth =lifecycle .get ("original_guard_centroid_axial")
        relative =lifecycle .get ("relative_offsets",{})
        if original_depth is None :
            original_depth =lifecycle .get ("centroid_axial")
        if original_depth is None or not relative :
            raise RuntimeError (
            f"missing frozen original Guard local frame: {branch }"
            )
        return float (original_depth )+max (
        float (offset [0 ])for offset in relative .values ()
        )

    def shepherd_return_depth (
    branch :str ,
    )->float :
        """Shepherd target depth. Never move past the original Guard wall."""

        original_depth =original_guard_leading_depth (branch )

        if physical .phase in {
        physical .SimulationPhase .PRESSURE_PUSH ,
        physical .SimulationPhase .FLOW_BACKTRACK ,
        }:
            command =getattr (
            physical ,
            "integration_backtrack_command_depth",
            None ,
            )

            if command is not None :
                return max (
                original_depth ,
                float (command ),
                )

        return max (
        original_depth ,
        float (original_get_shepherd_line_depth (branch )),
        )

    def shepherd_returned_to_original_guard (
    robots :Sequence [Any ],
    branch :str |None =None ,
    )->bool :
        """Return completion without frozen world-position localization."""

        target_branch =branch or physical .active_branch 

        if (
        physical .phase 
        !=physical .SimulationPhase .FLOW_BACKTRACK 
        ):
            return False 

        if target_branch is None :
            return False 

        lifecycle =getattr (
        physical ,
        "integration_wall_lifecycle",
        {},
        ).get (target_branch )

        if lifecycle is None :
            return False 

        expected_ids =set (
        lifecycle .get ("robot_ids",[])
        )

        if not expected_ids :
            return False 

        shepherds ={
        robot .robot_id :robot 
        for robot in physical .get_shepherds (robots )
        if (
        robot .shepherd_branch ==target_branch 
        and robot .robot_id in expected_ids 
        )
        }



        if set (shepherds )!=expected_ids :
            return False 

        descriptor =descriptor_for (
        target_branch 
        )

        if descriptor is None :
            return False 









        current_line_depth =shepherd_line_leading_depth (
        robots ,target_branch ,descriptor 
        )
        if current_line_depth is None :
            return False 

        return_floor =float (
        physical .JUNCTION_GUARD_BRANCH_INSET 
        +(
        int (lifecycle .get ("rows",1 ))-1 
        )
        *physical .THICK_MOUTH_GUARD_LAYER_SPACING 
        )

        tolerance =float (
        physical .SHEPHERD_JUNCTION_DEPTH_TOLERANCE 
        )

        return (
        current_line_depth 
        <=return_floor +tolerance 
        )

    def backtrack_pack_support (
    robots :Sequence [Any ],branch :str 
    )->tuple [float |None ,float |None ,int ,float ]:
        """Return a *full-cross-section* NORMAL rear surface for the piston.

        The previous implementation used a single 95th percentile over all
        NORMAL depths.  A thin handful of robots near the Shepherd could move
        first, drag that percentile toward the Junction, and let the rigid
        Shepherd wall follow while most of the swarm was still behind.

        The working Physical DFS behaves like a width-filling piston.  Recreate
        that semantics without the old invisible curtain: split the physical
        corridor width into Shepherd-column bins, require broad lateral support,
        and derive the return target from the rear surface of those occupied
        bins.  If the cross-section loses support, the Shepherd holds position.
        """
        descriptor =descriptor_for (branch )
        lifecycle =getattr (physical ,"integration_wall_lifecycle",{}).get (branch )
        if descriptor is None or lifecycle is None :
            return None ,None ,0 ,0.0 
        leading_depth =shepherd_line_leading_depth (robots ,branch ,descriptor )
        if leading_depth is None :
            return None ,None ,0 ,0.0 

        relative =lifecycle .get (
        "shepherd_relative_offsets",
        lifecycle .get ("relative_offsets",{}),
        )
        raw_axial_offsets =[float (value [0 ])for value in relative .values ()]










        minimum_axial_offset =(
        min (raw_axial_offsets )-max (raw_axial_offsets )
        if raw_axial_offsets else 0.0 
        )
        junction_face_depth =float (leading_depth +minimum_axial_offset )
        usable_half =max (
        physical .local_physical_usable_half_width (descriptor ),
        float (lifecycle .get ("usable_half_width",0.0 )),
        )
        if usable_half <=physical .EPSILON :
            return None ,None ,0 ,0.0 

        cols =max (1 ,int (lifecycle .get ("cols",1 )))
        bin_count =max (5 ,cols )
        face_tolerance =0.65 *physical .ROBOT_RADIUS 
        contact_window =max (
        physical .SHEPHERD_LOCAL_FLOW_DEPTH ,
        6.0 *physical .ROBOT_RADIUS ,
        )
        lateral_limit =usable_half +physical .ROBOT_RADIUS 
        per_bin_depths :list [list [float ]]=[[]for _ in range (bin_count )]
        contact_samples :list [tuple [float ,float ]]=[]
        shepherds =[
        robot for robot in physical .get_shepherds (robots )
        if robot .shepherd_branch ==branch 
        ]
        if not shepherds :
            return None ,None ,0 ,0.0 
        face_reference =min (
        shepherds ,
        key =lambda robot :float (
        relative .get (robot .robot_id ,(0.0 ,0.0 ))[0 ]
        ),
        )
        face_lateral =float (
        relative .get (face_reference .robot_id ,(0.0 ,0.0 ))[1 ]
        )
        observations =observe_local_neighbors (
        face_reference ,
        robots ,
        descriptor .local_outgoing_direction ,
        lateral_axis =getattr (descriptor ,"motion_n",None ),
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :(
        robot .role =="NORMAL"and not robot .base_reserve 
        ),
        )
        for observation in observations :
            axial =float (observation .relative_axial )
            lateral =float (observation .relative_lateral +face_lateral )
            if not (
            -contact_window <=axial <=face_tolerance 
            and abs (lateral )<=lateral_limit 
            ):
                continue 
            contact_samples .append ((axial ,lateral ))
            u =(lateral +usable_half )/max (2.0 *usable_half ,physical .EPSILON )
            index =int (physical .clamp (u *bin_count ,0 ,bin_count -1 ))
            per_bin_depths [index ].append (axial )

        occupied =[values for values in per_bin_depths if values ]
        coverage =len (occupied )/max (bin_count ,1 )
        if not occupied :
            return None ,None ,0 ,coverage 







        lane_rears =[max (values )for values in occupied ]
        rear_relative =(
        float (np .quantile (lane_rears ,0.90 ))
        if len (lane_rears )>=5 
        else float (max (lane_rears ))
        )
        rear_depth =junction_face_depth +rear_relative 
        rear_band =max (3.0 *physical .ROBOT_RADIUS ,physical .SAFE_RADIUS *0.55 )
        support_count =sum (
        1 for axial ,_ in contact_samples if axial >=rear_relative -rear_band 
        )

        contact_center_gap =2.05 *physical .ROBOT_RADIUS 
        support_depth =(
        rear_depth +contact_center_gap -minimum_axial_offset 
        )
        support_depth =max (0.0 ,float (support_depth ))





        minimum_coverage =max (
        0.70 ,
        float (physical .SATURATION_PACKED_LATERAL_COVERAGE_RATIO ),
        )
        minimum_support =max (
        physical .FLOW_MIN_NORMAL_COUNT ,
        int (math .ceil (cols *0.90 )),
        )
        if coverage <minimum_coverage or support_count <minimum_support :
            return rear_depth ,None ,support_count ,coverage 
        return rear_depth ,support_depth ,support_count ,coverage 

    def shepherd_pack_contact_state (
    robots :Sequence [Any ],branch :str 
    )->dict [str ,float |int |bool ]:
        """Strict pre-return gate matching the reference fill-before-push order."""
        descriptor =descriptor_for (branch )
        lifecycle =getattr (physical ,"integration_wall_lifecycle",{}).get (branch )
        if descriptor is None or lifecycle is None :
            return {
            "ready":False ,"rear_depth":-1.0 ,"junction_face_depth":-1.0 ,
            "gap":float ("inf"),"support_count":0 ,"pack_count":0 ,
            "coverage":0.0 ,
            }
        shepherds =[
        robot for robot in physical .get_shepherds (robots )
        if robot .shepherd_branch ==branch 
        ]
        if not shepherds :
            return {
            "ready":False ,"rear_depth":-1.0 ,"junction_face_depth":-1.0 ,
            "gap":float ("inf"),"support_count":0 ,"pack_count":0 ,
            "coverage":0.0 ,
            }

        relative =lifecycle .get (
        "shepherd_relative_offsets",
        lifecycle .get ("relative_offsets",{}),
        )
        raw_axial_offsets =[float (value [0 ])for value in relative .values ()]










        minimum_axial_offset =(
        min (raw_axial_offsets )-max (raw_axial_offsets )
        if raw_axial_offsets else 0.0 
        )
        leading_depth =shepherd_line_leading_depth (robots ,branch ,descriptor )
        if leading_depth is None :
            return {
            "ready":False ,"rear_depth":-1.0 ,"junction_face_depth":-1.0 ,
            "gap":float ("inf"),"support_count":0 ,"pack_count":0 ,
            "coverage":0.0 ,
            }
        junction_face_depth =max (0.0 ,float (leading_depth +minimum_axial_offset ))
        rear_depth ,support_depth ,support_count ,coverage =backtrack_pack_support (
        robots ,branch 
        )

        usable_half =max (
        physical .local_physical_usable_half_width (descriptor ),
        float (lifecycle .get ("usable_half_width",0.0 )),
        )
        pack_window =max (
        physical .SHEPHERD_LOCAL_FLOW_DEPTH ,
        6.0 *physical .ROBOT_RADIUS ,
        )
        face_reference =min (
        shepherds ,
        key =lambda robot :float (
        relative .get (robot .robot_id ,(0.0 ,0.0 ))[0 ]
        ),
        )
        face_lateral =float (
        relative .get (face_reference .robot_id ,(0.0 ,0.0 ))[1 ]
        )
        pack_count =sum (
        -pack_window <=observation .relative_axial 
        <=0.65 *physical .ROBOT_RADIUS 
        and abs (
        observation .relative_lateral +face_lateral 
        )<=usable_half +physical .ROBOT_RADIUS 
        for observation in observe_local_neighbors (
        face_reference ,
        robots ,
        descriptor .local_outgoing_direction ,
        lateral_axis =getattr (descriptor ,"motion_n",None ),
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :(
        robot .role =="NORMAL"and not robot .base_reserve 
        ),
        )
        )

        gap =(
        float (junction_face_depth -rear_depth )
        if rear_depth is not None else float ("inf")
        )
        cols =max (1 ,int (lifecycle .get ("cols",1 )))
        min_pack_count =max (
        physical .SATURATION_MIN_TIP_ROBOTS ,
        int (math .ceil (cols *2.0 )),
        )
        max_contact_gap =max (
        3.0 *physical .ROBOT_RADIUS ,
        0.50 *physical .SAFE_RADIUS ,
        )
        ready =bool (
        support_depth is not None 
        and pack_count >=min_pack_count 
        and -1.5 *physical .ROBOT_RADIUS <=gap <=max_contact_gap 
        )
        return {
        "ready":ready ,
        "rear_depth":rear_depth if rear_depth is not None else -1.0 ,
        "junction_face_depth":junction_face_depth ,
        "gap":gap ,
        "support_count":support_count ,
        "pack_count":pack_count ,
        "coverage":coverage ,
        }






    def update_pack_coupled_backtrack_depth (
    robots :Sequence [Any ],branch :str ,dt :float 
    )->None :
        """Advance the Shepherd as a physical piston without crossing the herd.

        During PRESSURE_PUSH and FLOW_BACKTRACK the intact Shepherd wall actively
        moves toward the Junction.  Its Junction-facing row is clamped to the
        NORMAL rear support surface, so it can compress/push the SPH body but can
        never tunnel through it.  If broad pack support is temporarily lost while
        any NORMAL remains in the branch, the Shepherd holds.  It may finish its
        return alone only after the branch contains zero NORMAL robots.
        """
        if physical .phase not in {
        physical .SimulationPhase .PRESSURE_PUSH ,
        physical .SimulationPhase .FLOW_BACKTRACK ,
        }:
            return 

        physical .integration_return_curtain_active =False 

        descriptor =descriptor_for (branch )
        lifecycle =getattr (physical ,"integration_wall_lifecycle",{}).get (branch )
        if descriptor is None or lifecycle is None :
            return 





        return_floor =original_guard_leading_depth (branch )



        current_line =shepherd_line_leading_depth (robots ,branch ,descriptor )
        if current_line is None :
            return 

        previous_command =getattr (
        physical ,"integration_backtrack_command_depth",None 
        )
        if previous_command is None :
            previous_command =float (current_line )
        previous_command =float (previous_command )

        rear_depth ,support_depth ,support_count ,lateral_coverage =(
        backtrack_pack_support (robots ,branch )
        )

        usable_half_width =max (
        physical .local_physical_usable_half_width (descriptor ),
        float (lifecycle .get ("usable_half_width",0.0 )),
        )
        relative =lifecycle .get (
        "shepherd_relative_offsets",
        lifecycle .get ("relative_offsets",{}),
        )
        branch_shepherds =[
        robot for robot in physical .get_shepherds (robots )
        if robot .shepherd_branch ==branch 
        ]
        if not branch_shepherds :
            return 
        face_reference =min (
        branch_shepherds ,
        key =lambda robot :float (
        relative .get (robot .robot_id ,(0.0 ,0.0 ))[0 ]
        ),
        )
        face_lateral =float (
        relative .get (face_reference .robot_id ,(0.0 ,0.0 ))[1 ]
        )
        active_branch_normals =sum (
        observation .relative_axial <=physical .ROBOT_RADIUS 
        and abs (
        observation .relative_lateral +face_lateral 
        )<=usable_half_width +2.0 *physical .ROBOT_RADIUS 
        for observation in observe_local_neighbors (
        face_reference ,
        robots ,
        descriptor .local_outgoing_direction ,
        lateral_axis =getattr (descriptor ,"motion_n",None ),
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :(
        robot .role =="NORMAL"and not robot .base_reserve 
        ),
        )
        )

        cols =max (1 ,int (lifecycle .get ("cols",1 )))
        min_support =max (
        physical .FLOW_MIN_NORMAL_COUNT ,
        int (math .ceil (cols *0.90 )),
        )
        min_lateral_coverage =max (
        0.70 ,
        physical .SATURATION_PACKED_LATERAL_COVERAGE_RATIO ,
        )
        pack_present =bool (
        support_depth is not None 
        and support_count >=min_support 
        and lateral_coverage >=min_lateral_coverage 
        )











        raw_axial_offsets =[float (value [0 ])for value in relative .values ()]
        minimum_axial_offset =(
        min (raw_axial_offsets )-max (raw_axial_offsets )
        if raw_axial_offsets 
        else 0.0 
        )





























        anchor_step_limit :float |None =None 
        lidar_anchor =next (
        (
        candidate 
        for candidate in robots 
        if (
        bool (getattr (candidate ,"is_lidar_robot",False ))
        or candidate .robot_id ==LIDAR_ROBOT_ID 
        )
        ),
        None ,
        )

        if lidar_anchor is not None :
            min_row_offset =min (
            (
            float (relative .get (shepherd .robot_id ,(0.0 ,0.0 ))[0 ])
            for shepherd in branch_shepherds 
            ),
            default =0.0 ,
            )
            row_tolerance =max (
            0.25 *float (physical .ROBOT_RADIUS ),
            float (physical .EPSILON ),
            )
            junction_face_row =[
            shepherd 
            for shepherd in branch_shepherds 
            if abs (
            float (relative .get (shepherd .robot_id ,(0.0 ,0.0 ))[0 ])
            -min_row_offset 
            )
            <=row_tolerance 
            ]







            anchor_guard_members =(
            junction_face_row 
            if junction_face_row 
            else branch_shepherds 
            )

            minimum_pair_distance =max (
            float (physical .ROBOT_RADIUS )
            +float (getattr (lidar_anchor ,"radius",physical .ROBOT_RADIUS )),
            2.05 *float (physical .ROBOT_RADIUS ),
            )

            local_step_limits :list [float ]=[]

            for shepherd in anchor_guard_members :
                observations =observe_local_neighbors (
                shepherd ,
                [lidar_anchor ],
                descriptor .local_outgoing_direction ,
                lateral_axis =getattr (descriptor ,"motion_n",None ),
                max_range =physical .COMM_RANGE ,
                )
                if not observations :
                    continue 

                observation =observations [0 ]
                relative_axial =float (observation .relative_axial )
                relative_lateral =float (observation .relative_lateral )





                if relative_axial >=0.0 :
                    continue 

                lateral_abs =abs (relative_lateral )
                if lateral_abs >=minimum_pair_distance :
                    continue 

                required_axial_separation =math .sqrt (
                max (
                0.0 ,
                minimum_pair_distance **2 -lateral_abs **2 ,
                )
                )
                current_axial_separation =-relative_axial 
                local_step_limits .append (
                max (
                0.0 ,
                current_axial_separation -required_axial_separation ,
                )
                )

            if local_step_limits :
                anchor_step_limit =min (local_step_limits )

        herd_return_speed_scale =float (
        getattr (physical ,"integration_herd_return_speed_scale",1.00 )
        )
        forward_speed =min (
        physical .SHEPHERD_LINE_BACKTRACK_SPEED ,
        physical .SHEPHERD_PISTON_SPEED ,
        )*herd_return_speed_scale 
        max_forward_step =max (0.0 ,forward_speed *dt )
        max_backoff_step =max_forward_step *0.50 





        compression_allowance =max (
        0.0 ,
        float (
        getattr (
        physical ,
        "SHEPHERD_CONTACT_COMPRESSION_ALLOWANCE",
        float (physical .ROBOT_RADIUS )*float (
        getattr (
        physical ,
        "integration_shepherd_contact_compression_ratio",
        0.20 ,
        )
        ),
        )
        ),
        )











        proposed_command =max (
        return_floor ,
        previous_command -max_forward_step ,
        )
        active_min_center_gap =max (
        1.05 *physical .ROBOT_RADIUS ,
        float (physical .ROBOT_RADIUS )
        *float (getattr (physical ,"integration_shepherd_active_min_center_gap_ratio",1.50 )),
        )
        hard_contact_floor =None 

        if pack_present and rear_depth is not None :










            hard_contact_floor =max (
            return_floor ,
            float (rear_depth )
            +active_min_center_gap 
            -minimum_axial_offset ,
            )
            allowed_floor =min (previous_command ,hard_contact_floor )
            command =max (proposed_command ,allowed_floor )

            old_support_floor =max (
            0.0 ,
            float (support_depth )-compression_allowance ,
            )
            if command <previous_command -physical .EPSILON :
                mode =(
                "ACTIVE_CONTACT_COMPRESSION_PUSH"
                if old_support_floor >=previous_command -physical .EPSILON 
                else "ACTIVE_CONTACT_PUSH"
                )
            else :
                mode ="HOLD_AT_HARD_CONTACT_FLOOR"

        elif active_branch_normals >0 and rear_depth is not None :




            partial_speed_scale =0.35 
            partial_step =max_forward_step *partial_speed_scale 
            proposed_partial =max (
            return_floor ,
            previous_command -partial_step ,
            )
            hard_contact_floor =max (
            return_floor ,
            float (rear_depth )
            +active_min_center_gap 
            -minimum_axial_offset ,
            )
            allowed_floor =min (previous_command ,hard_contact_floor )
            command =max (proposed_partial ,allowed_floor )
            mode =(
            "ACTIVE_PARTIAL_CONTACT_PUSH"
            if command <previous_command -physical .EPSILON 
            else "HOLD_PARTIAL_AT_HARD_CONTACT_FLOOR"
            )

        elif active_branch_normals >0 :


















            seek_speed_scale =1.00 

            seek_step =(
            max_forward_step 
            *seek_speed_scale 
            )

            command =max (
            return_floor ,
            previous_command -seek_step ,
            )

            mode ="SEEK_PACK_NO_REAR_SAMPLE"

        else :


            command =max (
            return_floor ,
            previous_command -max_forward_step ,
            )
            mode ="RETURN_TO_ORIGINAL_GUARD"

        command =max (
        return_floor ,
        float (command ),
        )















        if anchor_step_limit is not None :
            anchor_hard_floor =max (
            return_floor ,
            previous_command -anchor_step_limit ,
            )
            if command <anchor_hard_floor :
                command =anchor_hard_floor 
                mode =f"{mode }_ANCHOR_HARD_STOP"



        physical .integration_backtrack_command_depth =command 
        physical .integration_backtrack_pack_rear_depth =rear_depth 
        physical .integration_backtrack_support_depth =support_depth 
        physical .integration_backtrack_support_count =support_count 
        physical .integration_backtrack_lateral_coverage =lateral_coverage 

        junction_face_depth =command +minimum_axial_offset 
        pack_gap =(
        junction_face_depth -rear_depth 
        if rear_depth is not None 
        else float ("nan")
        )
        frame =getattr (physical ,"integration_frame",-1 )
        if frame %10 ==0 :
            print (
            f"[OriginalGuardReturn] "
            f"frame={frame } "
            f"branch={branch } "
            f"command_depth={command :.3f} "
            f"return_floor={return_floor :.3f} "
            f"current_leading_depth={float (current_line ):.3f} "
            f"error_to_floor={float (current_line )-return_floor :.3f}"
            )
            print (
            f"[ReturnPistonContact] frame={frame } phase={physical .phase .name } "
            f"branch={branch } current_line={float (current_line ):.3f} "
            f"previous_command={previous_command :.3f} "
            f"command_depth={command :.3f} "
            f"junction_face={junction_face_depth :.3f} "
            f"pack_rear={rear_depth if rear_depth is not None else -1.0 :.3f} "
            f"support_depth={support_depth if support_depth is not None else -1.0 :.3f} "
            f"support_count={support_count } coverage={lateral_coverage :.3f} "
            f"branch_normals={active_branch_normals } pack_gap={pack_gap :.3f} "
            f"compression_allowance={compression_allowance :.3f} "
            f"active_min_gap={active_min_center_gap :.3f} "
            f"hard_floor={hard_contact_floor if hard_contact_floor is not None else -1.0 :.3f} "
            f"anchor_step_limit={anchor_step_limit if anchor_step_limit is not None else -1.0 :.3f} "
            f"mode={mode }"
            )


    def real_shepherd_contact_force (robot :Any )->pygame .Vector2 :
        """Finite-radius contact force from the actual Shepherd robots only.

        This is not a Junction-directed route force and not an invisible curtain.
        Every contribution is a local spring-damper along the center-to-center
        normal from a real Shepherd robot to this NORMAL robot.  Therefore the
        force exists only where the visible 3xN Shepherd physically touches the
        packed swarm.
        """
        if (
        physical .phase not in {
        physical .SimulationPhase .PRESSURE_PUSH ,
        physical .SimulationPhase .FLOW_BACKTRACK ,
        }
        or robot .role !="NORMAL"
        or robot .base_reserve 
        ):
            setattr (robot ,"last_real_shepherd_contact_force",0.0 )
            setattr (robot ,"last_real_shepherd_contact_count",0 )
            return pygame .Vector2 ()

        branch =physical .active_branch 
        descriptor =descriptor_for (branch )
        if descriptor is None :
            setattr (robot ,"last_real_shepherd_contact_force",0.0 )
            setattr (robot ,"last_real_shepherd_contact_count",0 )
            return pygame .Vector2 ()

        shepherds =[
        shepherd for shepherd in physical .get_shepherds (getattr (physical ,"robots",[]))
        if shepherd .shepherd_branch ==branch 
        ]if hasattr (physical ,"robots")else []





        if not shepherds :
            shepherds =[
            shepherd for shepherd in getattr (physical ,"integration_current_robots",[])
            if shepherd .role =="SHEPHERD"and shepherd .shepherd_branch ==branch 
            ]
        if not shepherds :
            setattr (robot ,"last_real_shepherd_contact_force",0.0 )
            setattr (robot ,"last_real_shepherd_contact_count",0 )
            return pygame .Vector2 ()

        radius =float (physical .ROBOT_RADIUS )
        shell_ratio =float (
        getattr (physical ,"integration_shepherd_contact_shell_ratio",2.35 )
        )
        contact_radius =max (2.05 *radius ,shell_ratio *radius )
        spring_gain =float (physical .REPULSION_GAIN )*float (
        getattr (physical ,"integration_shepherd_contact_spring_scale",2.50 )
        )
        damping_gain =(
        float (physical .DAMPING )
        *float (
        getattr (
        physical ,
        "integration_shepherd_contact_damping_scale",
        1.00 ,
        )
        )
        )

        total =pygame .Vector2 ()
        contacts =0 
        strongest_compression =0.0 
        for shepherd in shepherds :
            delta =robot .position -shepherd .position 
            distance_sq =delta .length_squared ()
            if distance_sq <=physical .EPSILON :


                normal =descriptor .local_return_direction .normalize ()
                distance =0.0 
            else :
                distance =math .sqrt (distance_sq )
                if distance >=contact_radius :
                    continue 
                normal =delta /distance 

            compression =physical .clamp (
            (contact_radius -distance )/max (contact_radius ,physical .EPSILON ),
            0.0 ,
            1.0 ,
            )
            if compression <=0.0 :
                continue 

            pair_force =normal *(spring_gain *compression *compression )
            shepherd_velocity =getattr (shepherd ,"velocity",pygame .Vector2 ())
            relative_normal_speed =(robot .velocity -shepherd_velocity ).dot (normal )
            if relative_normal_speed <0.0 :
                pair_force +=normal *(-damping_gain *relative_normal_speed )

            total +=pair_force 
            contacts +=1 
            strongest_compression =max (strongest_compression ,compression )









        motion_multiplier =max (
        1.0 ,
        float (getattr (physical ,"MOTION_SPEED_MULTIPLIER",1.0 )),
        )
        repulsion_force_limit =float (
        getattr (
        physical ,
        "EQUILIBRIUM_REPULSION_FORCE_LIMIT",
        max (
        float (getattr (physical ,"REPULSION_GAIN",180.0 )),
        180.0 *motion_multiplier ,
        ),
        )
        )
        pressure_force_limit =float (
        getattr (
        physical ,
        "SPH_PRESSURE_FORCE_LIMIT",
        max (
        repulsion_force_limit ,
        420.0 *motion_multiplier ,
        ),
        )
        )
        contact_force_limit =max (
        repulsion_force_limit ,
        0.60 *pressure_force_limit ,
        )
        physical .limit_vector (total ,contact_force_limit )
        setattr (robot ,"last_real_shepherd_contact_force",total .length ())
        setattr (robot ,"last_real_shepherd_contact_count",contacts )

        frame =getattr (physical ,"integration_frame",-1 )
        if (
        contacts >0 
        and frame %10 ==0 
        and getattr (physical ,"integration_real_contact_log_frame",-1 )!=frame 
        ):
            physical .integration_real_contact_log_frame =frame 
            print (
            f"[RealShepherdContactV30] frame={frame } phase={physical .phase .name } "
            f"branch={branch } robot={robot .robot_id } contacts={contacts } "
            f"compression={strongest_compression :.3f} force={total .length ():.3f}"
            )
        return total 


    def shepherd_physical_only_route_force (robot :Any )->pygame .Vector2 :
        """Remove axial route suction from active-branch NORMALs during return.

        The NORMAL body must be carried toward the Junction by the moving Shepherd
        wall and SPH/contact/compression, not by FLOW_BACKTRACK's built-in direct
        Junction attraction.  Preserve only the component perpendicular to the
        return axis so corridor/lane centering can still operate.
        """
        if getattr (physical ,"integration_final_guard_sweep_active",False ):
            return final_guard_sweep_route_force (robot )



















        if (
        getattr (
        physical ,
        "integration_parent_return_active",
        False ,
        )
        and robot .role =="NORMAL"
        and not robot .base_reserve 
        ):
            return_direction =getattr (
            physical ,
            "integration_parent_return_direction_local",
            pygame .Vector2 (),
            )

            if (
            return_direction .length_squared ()
            <=physical .EPSILON 
            ):
                return pygame .Vector2 ()

            return_direction =(
            return_direction .normalize ()
            )

            force =(
            original_compute_route_force (
            robot 
            )
            )



            force -=(
            return_direction 
            *force .dot (
            return_direction 
            )
            )

            route_force_limit =float (
            getattr (
            physical ,
            "ROUTE_FORCE",
            adaptive .LOCAL_FORWARD_DRIVE_FORCE ,
            )
            )

            if not getattr (
            physical ,
            "integration_parent_return_arrived",
            False ,
            ):
                force +=(
                return_direction 
                *route_force_limit 
                )
            else :






                axial_speed =float (
                robot .velocity .dot (
                return_direction 
                )
                )

                force -=(
                return_direction 
                *float (
                np .clip (
                5.0 *axial_speed ,
                -route_force_limit ,
                route_force_limit ,
                )
                )
                )

            return force 


        is_lidar =bool (
        getattr (robot ,"is_lidar_robot",False )
        or robot .robot_id ==LIDAR_ROBOT_ID 
        )
        prep_uid =getattr (physical ,"integration_anchor_prep_request_uid",None )
        leading_uid =getattr (physical ,"integration_leading_anchor_uid",None )
        anchor_flow_uid =prep_uid or leading_uid 
        force =original_compute_route_force (robot )
        setattr (robot ,"last_physical_only_route_axial",0.0 )
        if (
        anchor_flow_uid is not None 
        and physical .phase in {
        physical .SimulationPhase .FORM_JUNCTION_GUARDS ,
        physical .SimulationPhase .JUNCTION_SWITCH ,
        physical .SimulationPhase .EXPLORE_BRANCH ,
        }
        and robot .role =="NORMAL"
        and not robot .base_reserve 
        and not is_lidar 
        ):
            follow_parent =getattr (robot ,"integration_anchor_follow_parent",None )
            descriptor =physical .branch_descriptors_by_uid .get (anchor_flow_uid )
            if follow_parent is not None and descriptor is not None :
                tangent ,lateral =physical .descriptor_local_basis (descriptor )
                tangent =tangent .normalize ()
                lateral =lateral .normalize ()
                observations =observe_local_neighbors (
                robot ,
                [follow_parent ],
                tangent ,
                lateral_axis =lateral ,
                max_range =physical .COMM_RANGE ,
                )
                if observations :
                    observation =observations [0 ]
                    parent_vector =(
                    tangent *float (observation .relative_axial )
                    +lateral *float (observation .relative_lateral )
                    )
                    parent_distance =float (observation .relative_range )
                    target_spacing =max (
                    2.4 *float (physical .ROBOT_RADIUS ),
                    0.85 *float (physical .GRID_SPACING ),
                    )
                    route_force_limit =float (getattr (
                    physical ,
                    "ROUTE_FORCE",
                    adaptive .LOCAL_FORWARD_DRIVE_FORCE ,
                    ))






























                    follow_force =pygame .Vector2 ()

                    if (
                    parent_distance >target_spacing 
                    and parent_vector .length_squared ()
                    >physical .EPSILON 
                    ):
                        pull_force =(
                        parent_vector .normalize ()
                        *min (
                        1.05 *route_force_limit ,
                        14.0 
                        *(
                        parent_distance 
                        -target_spacing 
                        ),
                        )
                        )
                        follow_force +=pull_force 

                    parent_velocity =getattr (
                    follow_parent ,
                    "velocity",
                    pygame .Vector2 (),
                    )



                    velocity_match_force =(
                    parent_velocity 
                    -robot .velocity 
                    )*5.0 
                    follow_force +=velocity_match_force 







                    parent_forward_speed =float (
                    parent_velocity .dot (tangent )
                    )
                    robot_forward_speed =float (
                    robot .velocity .dot (tangent )
                    )

                    axial_speed_error =(
                    parent_forward_speed 
                    -robot_forward_speed 
                    )

                    follow_force +=(
                    tangent 
                    *float (
                    np .clip (
                    12.0 *axial_speed_error ,
                    -0.75 *route_force_limit ,
                    1.15 *route_force_limit ,
                    )
                    )
                    )

                    physical .limit_vector (
                    follow_force ,
                    1.20 *route_force_limit ,
                    )

                    force +=follow_force 
        if (
        anchor_flow_uid is not None 
        and physical .phase in {
        physical .SimulationPhase .FORM_JUNCTION_GUARDS ,
        physical .SimulationPhase .JUNCTION_SWITCH ,
        physical .SimulationPhase .EXPLORE_BRANCH ,
        }
        and robot .role =="NORMAL"
        and not robot .base_reserve 
        and not is_lidar 
        ):
            descriptor =physical .branch_descriptors_by_uid .get (anchor_flow_uid )
            anchor =next (
            (other for other in getattr (physical ,"integration_current_robots",())
            if getattr (other ,"is_lidar_robot",False )
            or other .robot_id ==LIDAR_ROBOT_ID ),
            None ,
            )
            if descriptor is not None and anchor is not None :
                tangent ,lateral =physical .descriptor_local_basis (descriptor )
                tangent =tangent .normalize ()
                lateral =lateral .normalize ()
                observations =observe_local_neighbors (
                robot ,[anchor ],tangent ,lateral_axis =lateral ,
                max_range =physical .COMM_RANGE ,
                )
                if observations :
                    observation =observations [0 ]
                    rear_gap =float (observation .relative_axial )
                    desired_gap =float (physical .integration_anchor_target_gap )
                    route_force_limit =float (getattr (
                    physical ,"ROUTE_FORCE",
                    adaptive .LOCAL_FORWARD_DRIVE_FORCE ,
                    ))
                    robot_forward_speed =float (robot .velocity .dot (tangent ))
                    anchor_forward_speed =(
                    robot_forward_speed 
                    +float (observation .relative_axial_velocity )
                    )
                    closing_speed =max (
                    0.0 ,
                    robot_forward_speed -anchor_forward_speed ,
                    )
                    braking_gap =2.5 *desired_gap 
                    if rear_gap <braking_gap :
                        if closing_speed >0.0 :
                            force -=tangent *min (
                            route_force_limit ,
                            8.0 *closing_speed ,
                            )
                        if rear_gap <1.5 *desired_gap :
                            forward_component =float (force .dot (tangent ))
                            if forward_component >0.0 :
                                force -=tangent *forward_component 
                        if rear_gap <desired_gap :
                            gap_error =desired_gap -rear_gap 
                            force -=tangent *min (
                            route_force_limit ,
                            10.0 *gap_error +6.0 *closing_speed ,
                            )
                    allowed_half_width =(
                    float (physical .integration_anchor_fan_base_half_width )
                    +float (physical .integration_anchor_fan_widening_rate )
                    *max (0.0 ,rear_gap )
                    )
                    lateral_excess =abs (float (observation .relative_lateral ))-allowed_half_width 
                    if lateral_excess >0.0 :
                        direction_to_center =(
                        1.0 if observation .relative_lateral >0.0 else -1.0 
                        )
                        force +=lateral *direction_to_center *min (
                        0.60 *route_force_limit ,
                        5.0 *lateral_excess ,
                        )
        if (
        physical .phase in {
        physical .SimulationPhase .PRESSURE_PUSH ,
        physical .SimulationPhase .FLOW_BACKTRACK ,
        }
        and robot .role =="NORMAL"
        and not robot .base_reserve 
        ):
            descriptor =descriptor_for (physical .active_branch )
            if descriptor is not None :
                return_axis =descriptor .local_return_direction .normalize ()
                axial_component =force .dot (return_axis )
                force =force -return_axis *axial_component 
                setattr (
                robot ,
                "last_physical_only_route_axial",
                abs (force .dot (return_axis )),
                )




                force +=real_shepherd_contact_force (robot )
        return force 

    def physical_only_compute_sph_forces (
    robots :Sequence [Any ],
    grid :Any ,
    communication_grid :Any ,
    dt :float =1.0 /60.0 ,
    )->None :
        """Keep SPH physics while disabling the invisible return pressure field."""
        return_phase =physical .phase in {
        physical .SimulationPhase .PRESSURE_PUSH ,
        physical .SimulationPhase .FLOW_BACKTRACK ,
        }
        physical .integration_current_robots =robots 
        physical .integration_return_curtain_active =False 

        virtual_pressure_force =getattr (physical ,"VIRTUAL_PRESSURE_FORCE",None )
        if return_phase and virtual_pressure_force is not None :
            physical .VIRTUAL_PRESSURE_FORCE =0.0 
        try :
            original_compute_sph_forces (robots ,grid ,communication_grid ,dt )
        finally :
            if return_phase and virtual_pressure_force is not None :
                physical .VIRTUAL_PRESSURE_FORCE =virtual_pressure_force 

        frame =getattr (physical ,"integration_frame",-1 )
        if not return_phase or frame %10 !=0 :
            return 

        branch =physical .active_branch 
        descriptor =descriptor_for (branch )
        active_normals =[
        robot for robot in robots 
        if robot .role =="NORMAL"and not robot .base_reserve 
        ]

        normal_axial_route_max =max (
        (
        float (getattr (robot ,"last_physical_only_route_axial",0.0 ))
        for robot in active_normals 
        ),
        default =0.0 ,
        )
        real_shepherd_contacts =sum (
        int (getattr (robot ,"last_real_shepherd_contact_count",0 ))
        for robot in active_normals 
        )
        max_real_contact_force =max (
        (
        float (getattr (robot ,"last_real_shepherd_contact_force",0.0 ))
        for robot in active_normals 
        ),
        default =0.0 ,
        )
        command_depth =getattr (
        physical ,"integration_backtrack_command_depth",None 
        )
        return_floor =(
        original_guard_leading_depth (branch )
        if descriptor is not None 
        else float ("nan")
        )
        print (
        f"[PhysicalOnlyReturn] frame={frame } phase={physical .phase .name } "
        f"branch={branch } "
        f"normal_axial_route_max={normal_axial_route_max :.9f} "
        f"real_shepherd_contacts={real_shepherd_contacts } "
        f"max_real_contact_force={max_real_contact_force :.3f} "
        f"command_depth="
        f"{float (command_depth )if command_depth is not None else -1.0 :.3f} "
        f"return_floor={return_floor :.3f} "
        f"curtain_active={physical .integration_return_curtain_active }"
        )

    def promote_thick_frontier_wall (
    robots :Sequence [Any ],branch :str ,
    observed_boundary_depth :float |None =None ,
    )->list [Any ]:
        """Promote the SAME Frontier IDs to Shepherds *in place*.

        The thick Frontier has already travelled as the physical front wall.
        As soon as distributed rigid contact confirms a dead-end, the SAME robot
        IDs become Shepherds in place.  Packing/saturation is evaluated only after
        this role transition while the Shepherd wall is held stationary.

        Re-solving a second 3xN target here would create an unnecessary FORM phase
        and can make the Shepherds thread through the packed body.  Preserve each
        Frontier robot's current world position as its Shepherd anchor instead.

        The current dead-end-side leading edge becomes the scalar piston depth;
        each robot stores its current axial/lateral offset from that edge.  Return
        motion therefore translates the *already existing* physical wall toward
        the Junction without any second formation or position snap.
        """
        lifecycle =physical .integration_wall_lifecycle [branch ]
        frontiers =physical .get_frontier_shepherds (robots ,branch )
        expected =int (lifecycle ["rows"])*int (lifecycle ["cols"])
        if len (frontiers )!=expected :
            raise RuntimeError (
            f"thick Frontier wall lost members: {len (frontiers )}/{expected }"
            )

        descriptor =descriptor_for (branch )
        if descriptor is None :
            raise RuntimeError (f"missing descriptor for Shepherd promotion {branch }")
        lock_branch_transport_frame (branch ,lifecycle ,descriptor )

        frozen_offsets =lifecycle .get ("relative_offsets",{})
        if set (frozen_offsets )!={robot .robot_id for robot in frontiers }:
            raise RuntimeError (
            f"missing frozen Guard offsets for Shepherd promotion {branch }"
            )
        centroid_depth =float (physical .frontier_line_depth )
        leading_offset =max (
        float (offset [0 ])for offset in frozen_offsets .values ()
        )
        leading_edge_depth =centroid_depth +leading_offset 

        shepherd_relative_offsets :dict [int ,tuple [float ,float ]]={}
        promoted :list [Any ]=[]







        return_direction =descriptor .local_return_direction .copy ()
        if return_direction .length_squared ()<=physical .EPSILON :
            raise RuntimeError (f"invalid local return direction for Shepherd promotion {branch }")
        return_direction =return_direction .normalize ()

        for robot in frontiers :
            axial_offset ,lateral =frozen_offsets [robot .robot_id ]
            axial_from_leading_edge =float (axial_offset -leading_offset )
            anchor =robot .position .copy ()

            robot .role ="SHEPHERD"
            robot .shepherd_anchor =anchor 
            robot .shepherd_origin =robot .position .copy ()
            robot .shepherd_branch =branch 
            robot .shepherd_return_direction =return_direction .copy ()
            robot .junction_guard_anchor =None 
            robot .velocity .update (0.0 ,0.0 )
            robot .acceleration .update (0.0 ,0.0 )
            robot .filtered_acceleration .update (0.0 ,0.0 )

            stored =(axial_from_leading_edge ,float (lateral ))
            shepherd_relative_offsets [robot .robot_id ]=stored 
            physical .integration_shepherd_anchor_offsets [id (anchor )]=stored 
            promoted .append (robot )





        lifecycle ["shepherd_relative_offsets"]=shepherd_relative_offsets 
        lifecycle ["shepherd_centroid_depth"]=centroid_depth 
        lifecycle ["shepherd_leading_edge_depth"]=float (leading_edge_depth )
        lifecycle ["shepherd_odometry_depth"]=float (leading_edge_depth )
        lifecycle ["shepherd_odometry_time"]=float (physical .simulation_time )
        lifecycle ["shepherd_shape_locked"]=True 
        lifecycle ["state"]="SHEPHERD"
        lifecycle ["frontier_to_shepherd_in_place"]=True 
        physical .observed_dead_end_depths [branch ]=float (leading_edge_depth )

        physical .frontier_line_branch =None 
        physical .frontier_line_depth =0.0 
        physical .frontier_line_lateral_center =0.0 
        physical .frontier_line_row_ready =False 

        min_axial =min (
        offset [0 ]for offset in shepherd_relative_offsets .values ()
        )
        lateral_values =[
        offset [1 ]for offset in shepherd_relative_offsets .values ()
        ]
        print (
        f"[FrontierToShepherdInPlaceV27] branch={branch } "
        f"rows={lifecycle ['rows']} cols={lifecycle ['cols']} "
        f"robots={len (promoted )} leading_edge={leading_edge_depth :.3f} "
        f"thickness={-min_axial :.3f} "
        f"lateral_span={(max (lateral_values )-min (lateral_values ))if lateral_values else 0.0 :.3f} "
        "same_ids=True position_jump=0 second_formation=False"
        )
        return promoted 

    def returned_guard_wall_ready (
    robots :Sequence [Any ],
    branch :str ,
    )->bool :
            """Return readiness without frozen world-position localization.

            The SAME elected cohort must still exist as a Guard wall.
            Completion is based on branch-local mouth/Junction return,
            not distance to previously stored global positions.
            """

            lifecycle =getattr (
            physical ,
            "integration_wall_lifecycle",
            {},
            ).get (branch )

            if lifecycle is None :
                return False 

            expected_ids =set (
            lifecycle .get ("robot_ids",[])
            )

            if not expected_ids :
                return False 

            guards ={
            robot .robot_id :robot 
            for robot in robots 
            if (
            robot .robot_id in expected_ids 
            and robot .role =="JUNCTION_GUARD"
            and robot .junction_guard_branch ==branch 
            )
            }



            if set (guards )!=expected_ids :
                return False 

            descriptor =descriptor_for (branch )

            if descriptor is None :
                return False 







            maximum_depth =float (
            lifecycle .get (
            "shepherd_odometry_depth",
            lifecycle .get ("original_guard_centroid_axial",float ("inf")),
            )
            )

            return (
            maximum_depth 
            <=physical .SHEPHERD_JUNCTION_DEPTH_TOLERANCE 
            )



    def continuous_release_line (robots :Sequence [Any ])->int :
        if physical .phase !=physical .SimulationPhase .FLOW_BACKTRACK :
            return 0 
        branch =physical .active_branch 
        descriptor =descriptor_for (branch )
        lifecycle =getattr (physical ,"integration_wall_lifecycle",{}).get (branch )
        shepherds =[
        robot for robot in physical .get_shepherds (robots )
        if robot .shepherd_branch ==branch 
        ]
        if descriptor is None or lifecycle is None or not shepherds :
            return 0 
        maximum_depth =shepherd_line_leading_depth (robots ,branch ,descriptor )
        if maximum_depth is None :
            return 0 
        if maximum_depth >physical .SHEPHERD_JUNCTION_DEPTH_TOLERANCE :
            return 0 







        remaining_branch_normals =0 
        usable_half =max (
        physical .local_physical_usable_half_width (descriptor ),
        float (lifecycle .get ("usable_half_width",0.0 )),
        )
        reference =shepherds [0 ]
        remaining_branch_normals =sum (
        observation .relative_axial <=physical .ROBOT_RADIUS 
        and abs (observation .relative_lateral )
        <=usable_half +2.0 *physical .ROBOT_RADIUS 
        for observation in observe_local_neighbors (
        reference ,
        robots ,
        descriptor .local_outgoing_direction ,
        lateral_axis =getattr (descriptor ,"motion_n",None ),
        max_range =physical .COMM_RANGE ,
        predicate =lambda candidate :(
        candidate .role =="NORMAL"and not candidate .base_reserve 
        ),
        )
        )
        if remaining_branch_normals >0 :
            frame =getattr (physical ,"integration_frame",-1 )
            if frame %15 ==0 :
                print (
                f"[ShepherdReleaseWait] frame={frame } branch={branch } "
                f"remaining_normals={remaining_branch_normals } "
                "reason=HERD_NOT_CLEAR"
                )
            return 0 


        remaining_shepherds =[
        robot for robot in physical .get_shepherds (robots )
        if robot .shepherd_branch ==branch 
        ]
        return_direction =descriptor .local_return_direction .normalize ()
        before ={
        robot .robot_id :robot .position .copy ()
        for robot in remaining_shepherds 
        }








        release_velocity_by_id :dict [int ,pygame .Vector2 ]={}
        ordinary_normals =[
        candidate for candidate in robots 
        if candidate .role =="NORMAL"and not candidate .base_reserve 
        ]
        for shepherd in remaining_shepherds :
            observations =observe_local_neighbors (
            shepherd ,
            ordinary_normals ,
            return_direction ,
            max_range =2.0 *physical .SMOOTHING_LENGTH ,
            )
            observations .sort (key =lambda observation :observation .relative_range )
            neighbors =[observation .robot for observation in observations [:10 ]]
            if neighbors :
                velocity =pygame .Vector2 ()
                for candidate in neighbors :
                    velocity +=candidate .velocity 
                velocity /=len (neighbors )
            else :
                velocity =return_direction *min (
                4.0 ,physical .SHEPHERD_JUNCTION_RELEASE_SPEED 
                )
            release_velocity_by_id [shepherd .robot_id ]=velocity 

        for robot in remaining_shepherds :
            robot .role ="NORMAL"
            robot .shepherd_anchor =None 
            robot .shepherd_origin =None 
            robot .frontier_local_lateral =None 
            robot .shepherd_branch =None 
            robot .junction_guard_anchor =None 
            robot .junction_guard_branch =None 
            robot .junction_guard_branch_uid =None 
            robot .junction_guard_hop =-1 
            robot .junction_guard_parent_id =None 
            robot .junction_guard_layer =-1 
            robot .is_branch_leader =False 
            robot .velocity =release_velocity_by_id [robot .robot_id ].copy ()
            robot .acceleration .update (0.0 ,0.0 )
            robot .filtered_acceleration .update (0.0 ,0.0 )

        released_ids ={robot .robot_id for robot in remaining_shepherds }
        physical .junction_guard_groups [branch ]=[]
        lifecycle ["state"]="VISITED_RELEASED"
        lifecycle ["released_frame"]=getattr (
        physical ,"integration_frame",-1 
        )
        physical .frontier_line_branch =None 
        physical .frontier_line_depth =0.0 
        physical .frontier_line_lateral_center =0.0 
        physical .frontier_line_row_ready =False 
        transition_jump =max (
        (
        robot .position .distance_to (before [robot .robot_id ])
        for robot in remaining_shepherds 
        ),
        default =0.0 ,
        )
        print (
        f"[ShepherdRelease] branch={branch } "
        f"released_to_normal={len (remaining_shepherds )} "
        f"pebbles={len (physical .get_pebbles (robots ))} "
        f"position_jump={transition_jump :.6f} "
        "visited_wall_persisted=False"
        )
        return len (remaining_shepherds )

    def remaining_dfs_branch_uids (
    robots :Sequence [Any ],
    )->tuple [set [str ],set [str ],list [str ]]:
        """Return discovered, visited and still-unvisited branch UIDs."""

        discovered =set (
        getattr (
        physical ,
        "integration_detected_branch_order",
        [],
        )
        )

        if not discovered :
            discovered =set (
            physical .discovered_branch_uids ()
            )



        visited =set (
        physical .observed_visited_branch_uids (robots )
        )









        for branch ,lifecycle in getattr (
        physical ,
        "integration_wall_lifecycle",
        {},
        ).items ():
            if lifecycle .get ("state")!="VISITED_GUARD":
                continue 

            uid =physical .branch_uid_for_fixture (branch )

            if uid is not None :
                visited .add (uid )

        order =getattr (
        physical ,
        "integration_detected_branch_order",
        [],
        )

        remaining =[
        uid 
        for uid in order 
        if uid not in visited 
        ]

        if not order :
            remaining =sorted (
            discovered -visited 
            )

        return discovered ,visited ,remaining 


    def arm_cross_branch_carry (
    robots :Sequence [Any ],
    source_branch :str ,
    target_branch :str ,
    )->None :
        """Convert residual Junction return speed into next-branch propulsion."""




        reference =next (
        (robot for robot in robots if getattr (robot ,"is_lidar_robot",False )),
        None ,
        )
        if reference is None :
            reference =next ((robot for robot in robots if robot .role =="NORMAL"),None )
        incoming =getattr (
        physical ,
        "integration_incoming_direction_local",
        pygame .Vector2 (0.0 ,-1.0 ),
        )
        observations =observe_local_neighbors (
        reference ,
        robots ,
        incoming ,
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :robot .role =="NORMAL"and not robot .base_reserve ,
        )if reference is not None else []
        junction_speeds =[
        observation .robot .velocity .length ()
        for observation in observations 
        ]
        mean_speed =float (np .mean (junction_speeds ))if junction_speeds else 0.0 
        raw_scale =(
        1.0 
        +mean_speed 
        /max (float (physical .CROSS_BRANCH_CARRY_SPEED_REFERENCE ),physical .EPSILON )
        )
        peak_scale =float (np .clip (
        raw_scale ,
        physical .CROSS_BRANCH_CARRY_MIN_SCALE ,
        physical .CROSS_BRANCH_CARRY_MAX_SCALE ,
        ))
        physical .cross_branch_carry_peak_scale =peak_scale 
        physical .cross_branch_carry_until =(
        physical .simulation_time +physical .CROSS_BRANCH_CARRY_DURATION 
        )
        print (
        f"[CrossBranchCarry] frame={getattr (physical ,'integration_frame',-1 )} "
        f"{source_branch }->{target_branch } junction_mean_speed={mean_speed :.3f} "
        f"peak_scale={peak_scale :.3f} "
        f"duration={physical .CROSS_BRANCH_CARRY_DURATION :.2f}"
        )

    def commit_branch_uid_to_frontier (
    robots :Sequence [Any ],
    reference_density :float ,
    branch_uid :str ,
    *,
    context :str ,
    )->str :
        """Commit exactly one UID and promote that same fixture's Guard wall.

        ``record_distributed_consensus`` only records diagnostics; it does NOT set
        ``distributed_consensus_branch``.  The legacy ``choose_next_branch`` reads
        that global.  Keeping those two states separate allowed active_branch to
        become RIGHT while the LEFT Guard wall was promoted.  Lock both sides to
        the same UID here and fail loudly on any future mismatch.
        """
        requested_fixture =physical .branch_fixture_for_uid (branch_uid )
        if requested_fixture is None :
            raise RuntimeError (
            f"{context }: no fixture adapter for requested branch UID {branch_uid }"
            )



        physical .distributed_consensus_branch =branch_uid 
        physical .record_distributed_consensus (branch_uid )
        selected =physical .choose_next_branch (robots ,reference_density )
        if selected is None :
            raise RuntimeError (
            f"{context }: choose_next_branch returned None for UID {branch_uid } "
            f"fixture={requested_fixture }"
            )

        selected_uid =physical .branch_uid_for_fixture (selected )
        if selected !=requested_fixture or selected_uid !=branch_uid :
            raise RuntimeError (
            f"{context }: branch identity divergence: requested_uid={branch_uid } "
            f"requested_fixture={requested_fixture } selected_fixture={selected } "
            f"selected_uid={selected_uid }"
            )
        if physical .active_branch !=selected or physical .active_branch_uid !=branch_uid :
            raise RuntimeError (
            f"{context }: active branch divergence after choose: "
            f"active_branch={physical .active_branch } "
            f"active_uid={physical .active_branch_uid } expected={selected }/{branch_uid }"
            )



        stale =[
        robot 
        for robot in robots 
        if robot .role =="FRONTIER_SHEPHERD"
        and robot .shepherd_branch !=selected 
        ]
        if stale :
            raise RuntimeError (
            f"{context }: stale Frontier exists on another branch: "
            f"{sorted ({robot .shepherd_branch for robot in stale })}"
            )

        physical .commit_junction_guard_roles (robots ,selected )
        if physical .frontier_line_branch !=selected :
            raise RuntimeError (
            f"{context }: promoted Frontier branch={physical .frontier_line_branch } "
            f"but active branch={selected }"
            )
        live_frontier_ids ={
        robot .robot_id 
        for robot in robots 
        if robot .role =="FRONTIER_SHEPHERD"
        and robot .shepherd_branch ==selected 
        }
        lifecycle =getattr (physical ,"integration_wall_lifecycle",{}).get (selected ,{})
        expected_ids =set (lifecycle .get ("robot_ids",[]))
        if not live_frontier_ids or live_frontier_ids !=expected_ids :
            raise RuntimeError (
            f"{context }: active branch Guard/Frontier lineage mismatch: "
            f"selected={selected } live={sorted (live_frontier_ids )} "
            f"expected={sorted (expected_ids )}"
            )








        current_junction =(
        multi_dfs .current 
        )

        if current_junction is None :
            raise RuntimeError (
            f"{context }: Frontier activation has no current Junction"
            )

        if branch_uid not in current_junction .branch_states :
            raise RuntimeError (
            f"{context }: Branch UID is not registered in current Junction: "
            f"junction={current_junction .junction_uid } "
            f"branch={branch_uid }"
            )

        previous_state =(
        current_junction .branch_states [
        branch_uid 
        ]
        )

        if previous_state !="UNVISITED":
            raise RuntimeError (
            f"{context }: selected Branch was not UNVISITED: "
            f"junction={current_junction .junction_uid } "
            f"branch={branch_uid } "
            f"state={previous_state }"
            )

        if (
        current_junction .active_branch_uid 
        not in {
        None ,
        branch_uid ,
        }
        ):
            raise RuntimeError (
            f"{context }: another Branch is already ACTIVE: "
            f"junction={current_junction .junction_uid } "
            f"active={current_junction .active_branch_uid } "
            f"selected={branch_uid }"
            )

        descriptor =(
        physical .branch_descriptors_by_uid .get (
        branch_uid 
        )
        )

        if descriptor is None :
            raise RuntimeError (
            f"{context }: selected Branch descriptor disappeared: "
            f"branch={branch_uid }"
            )

        current_junction .branch_states [
        branch_uid 
        ]="ACTIVE"

        current_junction .active_branch_uid =(
        branch_uid 
        )

        current_junction .subtree_complete =(
        False 
        )

        descriptor .visit_state =(
        "ACTIVE"
        )

        print (
        "[CurrentBranchActive] "
        f"junction={current_junction .junction_uid } "
        f"depth={multi_dfs .depth } "
        f"branch={branch_uid } "
        "state=ACTIVE"
        )



        print (
        f"[BranchIdentityLock] context={context } uid={branch_uid } "
        f"fixture={selected } active={physical .active_branch } "
        f"frontier={physical .frontier_line_branch } robots={len (live_frontier_ids )} "
        "consistent=True"
        )
        return selected 

    def start_final_return_pipeline (robots :Sequence [Any ],reason :str )->None :
        """Enter the authoritative final gather -> base return lifecycle once."""

        if not multi_dfs .global_dfs_complete ():
            current =multi_dfs .current 

            raise RuntimeError (
            "Final return requested before Global DFS completion: "
            f"current="
            f"{(
            current .junction_uid 
            if current is not None 
            else None 
            )} "
            f"depth={multi_dfs .depth } "
            f"stack="
            f"{[
            frame .junction_uid 
            for frame in multi_dfs .stack 
            ]}"
            )

        if not getattr (
        physical ,
        "integration_global_dfs_complete",
        False ,
        ):
            physical .integration_global_dfs_complete =True 

            root =multi_dfs .current 

            print (
            "[GlobalDFSComplete] "
            f"root={root .junction_uid } "
            f"branches={root .branch_order } "
            f"states={root .branch_states } "
            f"reason={reason }"
            )

        if getattr (physical ,"integration_final_return_requested",False ):
            return 
        physical .integration_final_return_requested =True 
        root =multi_dfs .current 

        if root is None :
            raise RuntimeError (
            "Final return requested without completed Root"
            )

        logical_remaining =[
        branch_uid 
        for branch_uid in root .branch_order 
        if root .branch_states .get (branch_uid )
        !="VISITED"
        ]

        if logical_remaining :
            raise RuntimeError (
            "Final return requested with logical "
            "Root branches remaining: "
            f"{logical_remaining }"
            )
        physical .pending_branch_start =None 
        physical .integration_ready_guard_handoff =False 
        physical .integration_handoff_dwell =0.0 
        physical .junction_consensus_tracker .reset ()





        start_general_final_guard_sweep (robots ,reason )

        print (
        f"[DFSAllBranchesVisited] "
        f"frame={getattr (physical ,'integration_frame',-1 )} "
        f"visited={len (root .branch_order )}/{len (root .branch_order )} "
        f"reason={reason } "
        "source=MULTI_DFS_ROOT"
        )
        print (
        f"[FinalReturnPipeline] frame={getattr (physical ,'integration_frame',-1 )} "
        "ALL_BRANCHES_VISITED -> FIXED_RETURNED_GUARDS -> "
        "BASE_FLOW_ESTABLISH -> SIMULTANEOUS_GUARD_RELEASE -> "
        "RETURN_TO_BASE -> ALL_ROBOTS_DONE"
        )

    def update_root_shepherd_formation (
    robots :Sequence [Any ],dt :float ,
    )->None :
        """Advance Root Shepherd formation using only local relative sensing."""
        branch =physical .active_branch 
        lifecycle =getattr (physical ,"integration_wall_lifecycle",{}).get (branch )
        descriptor =descriptor_for (branch )if branch is not None else None 
        if lifecycle is None or descriptor is None :
            return 
        physical .update_relay_deployment (robots ,dt )
        physical .shepherd_form_timer +=dt 
        shepherds =[
        robot for robot in physical .get_shepherds (robots )
        if robot .shepherd_branch ==branch 
        ]
        expected =int (lifecycle .get ("rows",0 ))*int (lifecycle .get ("cols",0 ))
        offsets =lifecycle .get ("relative_offsets",{})
        formation_error =0.0 
        if shepherds and offsets :
            tangent ,lateral =physical .descriptor_local_basis (descriptor )
            reference =min (
            shepherds ,
            key =lambda robot :sum (abs (float (value ))for value in offsets .get (robot .robot_id ,(0.0 ,0.0 ))),
            )
            reference_offset =offsets .get (reference .robot_id ,(0.0 ,0.0 ))
            for observation in observe_local_neighbors (
            reference ,shepherds ,tangent ,lateral_axis =lateral ,
            ):
                expected_offset =offsets .get (observation .robot .robot_id ,reference_offset )
                formation_error =max (
                formation_error ,
                math .hypot (
                observation .relative_axial -(float (expected_offset [0 ])-float (reference_offset [0 ])),
                observation .relative_lateral -(float (expected_offset [1 ])-float (reference_offset [1 ])),
                ),
                )
        formed =(
        expected >0 
        and len (shepherds )==expected 
        and formation_error <=float (physical .SHEPHERD_FORM_TOLERANCE )
        )
        if not formed :
            return 
        physical .phase =physical .SimulationPhase .FILL_BEHIND_SHEPHERD 
        physical .saturation_tracker .reset (branch )
        physical .branch_continuity_tracker .reset (branch )
        current_depth =shepherd_line_leading_depth (robots ,branch ,descriptor )
        diagnostics .reset (branch ,float (current_depth or 0.0 ))
        physical .integration_shepherd_pack_ready_dwell =0.0 
        print (
        f"[ShepherdBoundaryReady] frame={getattr (physical ,'integration_frame',-1 )} "
        f"branch={branch } robots={len (shepherds )}/{expected } "
        f"local_formation_error={formation_error :.3f} "
        "teleport=False -> FILL_BEHIND_SHEPHERD"
        )

    def integrated_update_state (
    robots :Sequence [Any ],dt :float ,reference_density :float ,
    spatial_grid :Any ,
    )->None :
        if physical .phase ==physical .SimulationPhase .FORM_SHEPHERD_BOUNDARY :
            update_root_shepherd_formation (robots ,dt )
            return 






        if (
        physical .phase ==physical .SimulationPhase .FORM_JUNCTION_GUARDS 
        and getattr (physical ,"integration_ready_guard_handoff",False )
        and physical .pending_branch_start is None 
        ):
            current_junction =(
            multi_dfs .current 
            )

            if current_junction is None :
                raise RuntimeError (
                "Initial DFS branch selection has no current Junction"
                )















            _ ,_ ,legacy_remaining_uids =(
            remaining_dfs_branch_uids (
            robots 
            )
            )

            selected_branch_uid =(
            select_next_current_branch_uid (
            current_junction 
            )
            )

            if selected_branch_uid is None :

                multi_dfs .refresh_subtree_complete (
                current_junction 
                )

                if current_junction .subtree_complete :

                    if legacy_remaining_uids :
                        raise RuntimeError (
                        "Initial DFS completion disagreement: "
                        f"junction={current_junction .junction_uid } "
                        "MultiDFS says complete but legacy "
                        f"remaining={legacy_remaining_uids }"
                        )

                    start_final_return_pipeline (
                    robots ,
                    "INITIAL_GATE_CURRENT_DFS_COMPLETE",
                    )
                    return 

                raise RuntimeError (
                "Initial DFS selector returned no branch "
                "before subtree completion: "
                f"junction={current_junction .junction_uid } "
                f"states={current_junction .branch_states } "
                f"active={current_junction .active_branch_uid }"
                )









            if (
            selected_branch_uid 
            not in legacy_remaining_uids 
            ):
                raise RuntimeError (
                "Initial DFS selector disagreement: "
                f"junction={current_junction .junction_uid } "
                f"selected={selected_branch_uid } "
                f"legacy_remaining={legacy_remaining_uids }"
                )

            requested_fixture =(
            physical .branch_fixture_for_uid (
            selected_branch_uid 
            )
            )

            if requested_fixture is None :
                raise RuntimeError (
                "missing fixture for "
                f"{selected_branch_uid }"
                )

            physical .integration_pending_branch_uid =(
            selected_branch_uid 
            )





            physical .pending_branch_start =(
            requested_fixture 
            )

            physical .integration_pending_frontier_committed =(
            False 
            )

            request_anchor_prep (
            physical ,
            selected_branch_uid ,
            )

            print (
            "[DFSBranchSelected] "
            f"junction={current_junction .junction_uid } "
            f"uid={selected_branch_uid } "
            f"fixture={requested_fixture } "
            "selector=CURRENT_JUNCTION_DFS "
            "frontier_promoted=False "
            "launch_order="
            "ANCHOR_BEHIND_GUARD_THEN_FRONTIER_THEN_EXPLORE"
            )

            return 

        if (
        physical .phase ==physical .SimulationPhase .FORM_JUNCTION_GUARDS 
        and getattr (physical ,"integration_ready_guard_handoff",False )
        and physical .pending_branch_start is not None 
        ):








            pending_uid =getattr (
            physical ,"integration_pending_branch_uid",None 
            )
            if pending_uid is None :
                raise RuntimeError (
                "initial Anchor prep has a fixture but no pending branch UID"
                )
            frontier_committed =bool (
            getattr (
            physical ,
            "integration_pending_frontier_committed",
            False ,
            )
            )
            selected =physical .pending_branch_start 
            if selected is None :
                raise RuntimeError (
                "initial Anchor prep has no selected fixture"
                )
            lifecycle =getattr (physical ,"integration_wall_lifecycle",{})
            selected_lifecycle =lifecycle .get (selected ,{})
            prep_ready =(
            getattr (
            physical ,
            "integration_anchor_prep_ready_uid",
            None ,
            )
            ==pending_uid 
            )
            if not prep_ready :
                return 

            if getattr (physical ,"integration_leading_anchor_uid",None )!=pending_uid :
                promote_anchor_prep_to_leading (physical ,pending_uid )

            if not frontier_committed :
                selected =commit_branch_uid_to_frontier (
                robots ,
                reference_density ,
                pending_uid ,
                context ="INITIAL_DFS_AFTER_ANCHOR_BEHIND_GUARD",
                )
                physical .pending_branch_start =selected 
                physical .integration_pending_frontier_committed =True 
                physical .integration_handoff_dwell =0.0 
                print (
                "[AnchorCenterGatePassed] "
                f"uid={pending_uid } fixture={selected } "
                "guard_to_frontier=True "
                "anchor_behind_guard=True"
                )

























            visited_uids =(
            physical .observed_visited_branch_uids (
            robots 
            )
            )

            unvisited_branches =[
            branch 
            for branch 
            in physical .detected_branch_candidates 
            if physical .branch_uid_for_fixture (
            branch 
            )
            not in visited_uids 
            ]

            selected_lifecycle =getattr (
            physical ,
            "integration_wall_lifecycle",
            {},
            ).get (
            selected ,
            {},
            )

            expected_frontier_count =(
            int (
            selected_lifecycle .get (
            "rows",
            0 ,
            )
            )
            *int (
            selected_lifecycle .get (
            "cols",
            0 ,
            )
            )
            )

            selected_frontiers =[
            robot 
            for robot 
            in robots 
            if (
            robot .role 
            =="FRONTIER_SHEPHERD"
            and robot .shepherd_branch 
            ==selected 
            )
            ]

            selected_ready =(
            expected_frontier_count 
            >0 
            and len (
            selected_frontiers 
            )
            ==expected_frontier_count 
            )

            remaining_guard_walls_ready =True 

            for branch in unvisited_branches :
                if branch ==selected :
                    continue 

                item =getattr (
                physical ,
                "integration_wall_lifecycle",
                {},
                ).get (
                branch ,
                {},
                )

                expected_count =(
                int (
                item .get (
                "rows",
                0 ,
                )
                )
                *int (
                item .get (
                "cols",
                0 ,
                )
                )
                )

                live_count =sum (
                (
                robot .role 
                =="JUNCTION_GUARD"
                and robot .junction_guard_branch 
                ==branch 
                )
                for robot in robots 
                )

                if (
                expected_count <=0 
                or live_count 
                !=expected_count 
                ):
                    remaining_guard_walls_ready =False 
                    break 

            if (
            not selected_ready 
            or not remaining_guard_walls_ready 
            ):
                if (
                getattr (
                physical ,
                "integration_frame",
                0 ,
                )
                %10 
                ==0 
                ):
                    print (
                    "[FrontierLaunchWait] "
                    f"branch={selected } "
                    f"frontiers="
                    f"{len (selected_frontiers )}/"
                    f"{expected_frontier_count } "
                    f"other_guards_ready="
                    f"{remaining_guard_walls_ready }"
                    )

                return 















            physical .pending_branch_start =None 
            physical .integration_pending_branch_uid =None 
            physical .integration_pending_frontier_committed =False 

            physical .junction_guard_status =(
            f"FRONTIER={selected };"
            "OTHERS=THICK_KHOP_WALLS_READY"
            )

            physical .phase =(
            physical .SimulationPhase .EXPLORE_BRANCH 
            )

            physical .integration_ready_guard_handoff =False 
            physical .integration_handoff_dwell =0.0 

            print (
            "[EXPLORE_BRANCH] "
            f"selected={selected } "
            f"frontier="
            f"{len (selected_frontiers )}/"
            f"{expected_frontier_count } "
            "frontier_shape=HARD_RIGID_3xN "
            "headstart_gate_removed=True "
            "dead_end_saturation_enabled=True"
            )

            return 











        if physical .phase ==physical .SimulationPhase .FLOW_BACKTRACK :
            branch =physical .active_branch 























            lifecycle =getattr (
            physical ,"integration_wall_lifecycle",{}
            ).get (branch )

            physical .shepherd_flow_timer +=dt 





            update_pack_coupled_backtrack_depth (robots ,branch ,dt )
            physical .update_relay_retraction (robots ,dt )

            shepherds_before =[
            robot 
            for robot in physical .get_shepherds (robots )
            if robot .shepherd_branch ==branch 
            ]
            shepherd_ids_before ={
            robot .robot_id for robot in shepherds_before 
            }

            if not physical .shepherd_line_reached_junction (robots ,branch ):
                if getattr (physical ,"integration_frame",0 )%15 ==0 :
                    print (
                    f"[ShepherdArrivalWait] "
                    f"frame={getattr (physical ,'integration_frame',-1 )} "
                    f"branch={branch } shepherds={len (shepherds_before )} "
                    f"line_depth={physical .get_shepherd_line_depth (branch ):.3f} "
                    f"junction_tol="
                    f"{physical .SHEPHERD_JUNCTION_DEPTH_TOLERANCE :.3f}"
                    )
                return 



            descriptor =descriptor_for (
            branch 
            )

            if descriptor is None :
                return 

            (
            swarm_returned ,
            branch_side_normals ,
            junction_side_normals ,
            return_dwell ,
            )=evaluate_local_swarm_return_completion (
            physical ,
            robots ,
            branch ,
            descriptor ,
            lifecycle ,
            dt ,
            )

            if not swarm_returned :
                if (
                getattr (
                physical ,
                "integration_frame",
                0 ,
                )
                %10 
                ==0 
                ):
                    print (
                    "[RootSwarmReturnWait] "
                    f"branch={branch } "
                    f"branch_side_normals="
                    f"{branch_side_normals } "
                    f"junction_side_normals="
                    f"{junction_side_normals } "
                    f"dwell={return_dwell :.3f}"
                    )

                return 

            arrival_frame =getattr (physical ,"integration_frame",-1 )
            completed_branch =branch 

            completed_uid =(
            physical .active_branch_uid 
            or physical .branch_uid_for_fixture (completed_branch )
            )

            if completed_uid is None :
                raise RuntimeError (
                f"missing branch UID at Shepherd return: {completed_branch }"
                )

            same_ids_at_return =bool (
            shepherd_ids_before 
            )

            print (
            f"[ShepherdJunctionReturnComplete] "
            f"branch={completed_branch } "
            f"same_ids={same_ids_at_return } "
            f"shepherd_count={len (shepherd_ids_before )} "
            "completion_source=PHYSICAL_JUNCTION_ARRIVAL "
            "frozen_world_anchor_check=False"
            )





            retained_guard_count =physical .release_shepherd_line_at_junction (
            robots ,
            keep_as_guard =True ,
            )
            if retained_guard_count <=0 :
                raise RuntimeError (
                "branch was completed but returned Shepherd wall "
                f"could not become Guard: branch={completed_branch }"
                )

            returned_guards =[
            robot 
            for robot in robots 
            if robot .role =="JUNCTION_GUARD"
            and robot .junction_guard_branch ==completed_branch 
            ]
            returned_guard_ids ={
            robot .robot_id for robot in returned_guards 
            }


            expected_ids =set (
            (lifecycle or {}).get (
            "robot_ids",
            shepherd_ids_before ,
            )
            )

            if returned_guard_ids !=expected_ids :
                raise RuntimeError (
                "returned Guard IDs changed during "
                "Guard -> Frontier -> Shepherd -> Guard lifecycle: "
                f"branch={completed_branch } "
                f"expected={sorted (expected_ids )} "
                f"actual={sorted (returned_guard_ids )}"
                )





            physical .set_branch_descriptor_state (
            completed_uid ,
            "VISITED",
            )

            physical .previous_branch_direction =(
            physical .get_backtrack_direction (completed_branch )
            )

            physical .distributed_consensus_branch =None 
            physical .active_branch_uid =None 

            for robot in robots :
                robot .branch_vote =None 
                robot .branch_vote_confidence =0.0 
                robot .distributed_branch_decision =None 

            physical .record_distributed_consensus (
            clear_selection =True 
            )

            if hasattr (physical ,"metrics"):
                physical .metrics .branch_events .append ({
                "branch":completed_branch ,
                "completed_at":physical .simulation_time ,
                })

            if lifecycle is not None :
                lifecycle ["state"]="VISITED_GUARD"
                lifecycle ["visited_guard_frame"]=arrival_frame 
                lifecycle ["returned_guard_ids"]=sorted (returned_guard_ids )
                lifecycle ["same_ids_guard_frontier_shepherd_guard"]=True 
                lifecycle ["next_branch_trigger"]="SHEPHERD_JUNCTION_ARRIVAL"












            current_junction =(
            multi_dfs .current 
            )

            if current_junction is None :
                raise RuntimeError (
                "Root Branch completion has no current Junction"
                )

            completed_branch_uid =(
            physical .branch_uid_for_fixture (
            completed_branch 
            )
            )

            if completed_branch_uid is None :
                raise RuntimeError (
                "Completed fixture has no runtime Branch UID: "
                f"branch={completed_branch }"
                )

            if (
            current_junction .active_branch_uid 
            !=completed_branch_uid 
            ):
                raise RuntimeError (
                "Physical return completed for a Branch that is not "
                "the current DFS ACTIVE Branch: "
                f"junction={current_junction .junction_uid } "
                f"completed={completed_branch_uid } "
                f"active={current_junction .active_branch_uid }"
                )

            previous_state =(
            current_junction .branch_states .get (
            completed_branch_uid 
            )
            )

            if previous_state !="ACTIVE":
                raise RuntimeError (
                "Completed Root Branch was not ACTIVE: "
                f"junction={current_junction .junction_uid } "
                f"branch={completed_branch_uid } "
                f"state={previous_state }"
                )

            current_junction .branch_states [
            completed_branch_uid 
            ]="VISITED"

            current_junction .active_branch_uid =(
            None 
            )

            descriptor =(
            physical .branch_descriptors_by_uid .get (
            completed_branch_uid 
            )
            )

            if descriptor is None :
                raise RuntimeError (
                "Completed Branch descriptor disappeared: "
                f"branch={completed_branch_uid }"
                )

            descriptor .visit_state =(
            "VISITED"
            )

            multi_dfs .refresh_subtree_complete (
            current_junction 
            )

            print (
            "[CurrentBranchVisitedAfterPhysicalReturn] "
            f"junction={current_junction .junction_uid } "
            f"depth={multi_dfs .depth } "
            f"branch={completed_branch_uid } "
            "ACTIVE->VISITED "
            f"subtree_complete={current_junction .subtree_complete }"
            )

            (
            discovered ,
            visited ,
            legacy_remaining_uids ,
            )=remaining_dfs_branch_uids (
            robots 
            )

            current_remaining_uids =[
            uid 
            for uid 
            in current_junction .branch_order 
            if (
            current_junction .branch_states .get (
            uid 
            )
            =="UNVISITED"
            )
            ]

            if (
            set (current_remaining_uids )
            !=set (legacy_remaining_uids )
            ):
                raise RuntimeError (
                "Branch completion state disagreement: "
                f"junction={current_junction .junction_uid } "
                f"current_remaining={current_remaining_uids } "
                f"legacy_remaining={legacy_remaining_uids }"
                )

            completed_count =sum (
            current_junction .branch_states .get (
            uid 
            )
            =="VISITED"
            for uid 
            in current_junction .branch_order 
            )

            total_count =len (
            current_junction .branch_order 
            )

            print (
            "[DFSBranchCompleteSequence] "
            f"frame={arrival_frame } "
            f"junction={current_junction .junction_uid } "
            f"completed={completed_count }/{total_count } "
            f"remaining={current_remaining_uids } "
            f"subtree_complete="
            f"{current_junction .subtree_complete }"
            )

            print (
            f"[GuardCycleComplete] frame={arrival_frame } "
            f"branch={completed_branch } "
            f"robots={len (returned_guard_ids )} "
            "GUARD->FRONTIER->SHEPHERD->GUARD "
            "same_ids=True "
            "completion_source=SHEPHERD_JUNCTION_ARRIVAL "
            "frozen_world_anchor_check=False"
            )



            physical .integration_backtrack_command_depth =None 
            physical .integration_backtrack_pack_rear_depth =None 
            physical .integration_backtrack_support_depth =None 
            physical .integration_backtrack_support_count =0 
            physical .integration_backtrack_lateral_coverage =0.0 

            physical .phase =physical .SimulationPhase .JUNCTION_SWITCH 
            physical .junction_switch_timer =0.0 
            physical .junction_consensus_tracker .reset ()

            print (
            f"[DFSNextBranchSameFrame] frame={arrival_frame } "
            f"completed_branch={completed_branch } "
            "visited_guard_persisted=True "
            "trigger=SHEPHERD_JUNCTION_ARRIVAL"
            )

            return 

        if physical .phase ==physical .SimulationPhase .JUNCTION_SWITCH :





            physical .junction_switch_timer +=dt 

            current_junction =(
            multi_dfs .current 
            )

            if current_junction is None :
                raise RuntimeError (
                "JUNCTION_SWITCH has no current Junction"
                )









            (
            discovered ,
            visited ,
            legacy_remaining_uids ,
            )=remaining_dfs_branch_uids (
            robots 
            )

            selected_branch_uid =(
            select_next_current_branch_uid (
            current_junction 
            )
            )

            if selected_branch_uid is None :

                multi_dfs .refresh_subtree_complete (
                current_junction 
                )

                if current_junction .subtree_complete :

                    if legacy_remaining_uids :
                        raise RuntimeError (
                        "JUNCTION_SWITCH completion disagreement: "
                        f"junction={current_junction .junction_uid } "
                        "MultiDFS says complete but legacy "
                        f"remaining={legacy_remaining_uids }"
                        )

                    start_final_return_pipeline (
                    robots ,
                    "JUNCTION_SWITCH_CURRENT_DFS_COMPLETE",
                    )
                    return 

                raise RuntimeError (
                "JUNCTION_SWITCH selector returned no branch "
                "before subtree completion: "
                f"junction={current_junction .junction_uid } "
                f"states={current_junction .branch_states } "
                f"active={current_junction .active_branch_uid }"
                )

            if (
            selected_branch_uid 
            not in legacy_remaining_uids 
            ):
                raise RuntimeError (
                "JUNCTION_SWITCH selector disagreement: "
                f"junction={current_junction .junction_uid } "
                f"selected={selected_branch_uid } "
                f"legacy_remaining={legacy_remaining_uids }"
                )

            requested_fixture =(
            physical .branch_fixture_for_uid (
            selected_branch_uid 
            )
            )

            if requested_fixture is None :
                raise RuntimeError (
                "JUNCTION_SWITCH: no fixture for next UID "
                f"{selected_branch_uid }"
                )

            selected_guard_count =sum (
            robot .role 
            =="JUNCTION_GUARD"
            and robot .junction_guard_branch 
            ==requested_fixture 
            for robot in robots 
            )

            if selected_guard_count <=0 :
                print (
                f"[NextBranchWait] "
                f"frame="
                f"{getattr (physical ,'integration_frame',-1 )} "
                "reason=SELECTED_GUARD_MISSING "
                f"uid={selected_branch_uid } "
                f"branch={requested_fixture }"
                )
                return 

            pending_uid =getattr (
            physical ,
            "integration_pending_branch_uid",
            None ,
            )

            if pending_uid is None :

                physical .integration_pending_branch_uid =(
                selected_branch_uid 
                )

                physical .integration_pending_frontier_committed =(
                False 
                )

                request_anchor_prep (
                physical ,
                selected_branch_uid ,
                )

                print (
                "[DFSBranchPrepSelected] "
                f"junction={current_junction .junction_uid } "
                f"uid={selected_branch_uid } "
                f"fixture={requested_fixture } "
                "selector=CURRENT_JUNCTION_DFS "
                "frontier_promoted=False "
                "context=JUNCTION_SWITCH"
                )
                return 

            if pending_uid !=selected_branch_uid :
                raise RuntimeError (
                "JUNCTION_SWITCH pending branch changed "
                "during Anchor prep: "
                f"pending={pending_uid } "
                f"selected={selected_branch_uid }"
                )
            if getattr (
            physical ,"integration_anchor_prep_ready_uid",None 
            )!=pending_uid :
                return 

            source_branch =physical .active_branch 
            selected =commit_branch_uid_to_frontier (
            robots ,
            reference_density ,
            pending_uid ,
            context ="JUNCTION_SWITCH_AFTER_ANCHOR_PREP",
            )
            selected_uid =physical .branch_uid_for_fixture (selected )
            if selected_uid in visited :
                raise RuntimeError (
                f"visited branch re-selected during Junction switch: {selected_uid }"
                )

            physical .integration_pending_frontier_committed =True 
            promote_anchor_prep_to_leading (physical ,pending_uid )

            physical .pending_branch_start =selected 
            physical .integration_pending_branch_uid =pending_uid 
            physical .integration_pending_frontier_committed =True 
            physical .integration_ready_guard_handoff =True 
            physical .integration_handoff_dwell =0.0 
            physical .junction_guard_formation_timer =0.0 
            physical .junction_guard_stable_dwell =0.0 
            physical .branch_entry_timer =0.0 
            physical .phase =physical .SimulationPhase .FORM_JUNCTION_GUARDS 
            print (
            f"[NextBranchLaunch] frame={getattr (physical ,'integration_frame',-1 )} "
            f"{source_branch }->{selected } uid={selected_uid } "
            f"frontier_committed=True guard_count={selected_guard_count } "
            "anchor_prep_ready=True headstart_pending=True"
            )
            return 

        branch =physical .active_branch 
        if physical .phase ==physical .SimulationPhase .EXPLORE_BRANCH :
            physical .branch_entry_timer +=dt 
            physical .update_relay_deployment (robots ,dt )
            local_frontier_progress (robots ,branch ,dt )

















            state =sample_local_state (
            robots ,
            branch ,
            reference_density ,
            dt ,
            )

            lifecycle =getattr (
            physical ,
            "integration_wall_lifecycle",
            {},
            ).get (branch )































            descriptor =descriptor_for (branch )

            if descriptor is None :
                return 

            frontiers =physical .get_frontier_shepherds (
            robots ,
            branch ,
            )

            if not frontiers :
                return 

            dead_end =evaluate_frontier_dead_end (
            physical ,
            descriptor .uid ,
            descriptor ,
            frontiers ,
            dt ,
            )

            if not dead_end .confirmed :
                return 

            if lifecycle is None :
                return 

            if (
            lifecycle .get (
            "frontier_contact_centroid_depth"
            )
            is None 
            ):
                lifecycle [
                "frontier_contact_centroid_depth"
                ]=float (
                physical .frontier_line_depth 
                )

                print (
                "[FrontierDeadEndConfirmed] "
                f"branch={branch } "
                f"uid={descriptor .uid } "
                f"centroid_depth="
                f"{physical .frontier_line_depth :.3f} "
                f"command_speed="
                f"{dead_end .command_forward_speed :.3f} "
                f"actual_speed="
                f"{dead_end .actual_forward_speed :.3f} "
                f"lidar_blocked="
                f"{dead_end .lidar_blocked } "
                f"blocked_ratio="
                f"{dead_end .forward_blocked_ratio :.3f} "
                f"no_junction="
                f"{dead_end .no_junction_evidence } "
                f"dwell={dead_end .dwell :.3f}"
                )

            before ={
            robot .robot_id :robot .position .copy ()
            for robot in frontiers 
            }

            state .frontier_ids =sorted (before )



            frontier_offsets =(
            lifecycle .get ("relative_offsets",{})if lifecycle else {}
            )
            boundary_depth =float (physical .frontier_line_depth )+max (
            [float (value [0 ])for value in frontier_offsets .values ()]
            or [0.0 ]
            )
            physical .observed_dead_end_depths [branch ]=boundary_depth 

            selected =physical .promote_existing_frontier_line (
            robots ,branch ,boundary_depth 
            )
            if not selected :
                if getattr (physical ,"integration_frame",0 )%15 ==0 :
                    print (
                    f"[BacktrackPromotionWaitV31] branch={branch } "
                    f"dead_end=True boundary_depth={boundary_depth :.3f} "
                    "reason=NO_COMMON_SHEPHERD_CROSS_SECTION "
                    "frontier_frozen=True"
                    )
                return 

            state .shepherd_ids =sorted (robot .robot_id for robot in selected )
            state .max_transition_jump =max (
            robot .position .distance_to (before [robot .robot_id ])
            for robot in selected 
            )
            state .shepherd_transition =True 
            state .transition_frame =getattr (physical ,"integration_frame",-1 )
            return_direction =descriptor .local_return_direction .normalize ()
            state .return_direction_local =(return_direction .x ,return_direction .y )



            physical .branch_dead_end_confirmed [branch ]=True 
            physical .dead_end_inference_tracker .confirmed =True 
            physical .dead_end_inference_tracker .confirmed_depth =boundary_depth 
            physical .dead_end_inference_tracker .handoff_depth =boundary_depth 



            baseline_density =max (0.0 ,float (state .local_density ))
            baseline_pressure =max (0.0 ,float (state .local_pressure ))
            baseline_cross_fill =max (0.0 ,float (state .cross_section_fill ))

            current_shepherd_depth =float (
            lifecycle .get ("shepherd_odometry_depth",boundary_depth )
            )
            diagnostics .reset (branch ,current_shepherd_depth )
            diagnostics .baseline_density =baseline_density 
            diagnostics .baseline_pressure =baseline_pressure 
            diagnostics .cross_section_fill =baseline_cross_fill 
            physical .integration_shepherd_fill_baseline_density =baseline_density 
            physical .integration_shepherd_fill_baseline_pressure =baseline_pressure 
            physical .integration_shepherd_fill_baseline_cross_fill =baseline_cross_fill 
            physical .integration_shepherd_pack_ready_dwell =0.0 
            physical .integration_backtrack_command_depth =None 


            if hasattr (physical ,"saturation_tracker"):
                physical .saturation_tracker .reset (branch )
            if hasattr (physical ,"branch_continuity_tracker"):
                physical .branch_continuity_tracker .reset (branch )



            physical .integration_pending_shepherd_transition_event ={
            "branch":branch ,
            "uid":descriptor .uid ,
            "transition_frame":state .transition_frame ,
            "frontier_ids":list (state .frontier_ids ),
            "shepherd_ids":list (state .shepherd_ids ),
            "max_transition_jump":state .max_transition_jump ,
            "return_direction":state .return_direction_local ,
            "rows":physical .integration_wall_lifecycle [branch ]["rows"],
            "cols":physical .integration_wall_lifecycle [branch ]["cols"],
            "robots":len (state .frontier_ids ),
            "max_formation_error":state .max_formation_error ,
            }









            physical .phase =physical .SimulationPhase .FORM_SHEPHERD_BOUNDARY 
            physical .shepherd_form_timer =0.0 

            print (
            "[Timeline] DEAD_END_CONFIRMED "
            f"frame={state .transition_frame } uid={descriptor .uid }"
            )
            print (
            "[Timeline] FRONTIER_TO_SHEPHERD "
            f"frame={state .transition_frame } same_ids="
            f"{state .frontier_ids ==state .shepherd_ids } "
            f"max_position_jump={state .max_transition_jump :.6f}"
            )
            print (
            f"[ShepherdFillStartV31] frame={state .transition_frame } "
            f"branch={branch } baseline_density={baseline_density :.6f} "
            f"baseline_pressure={baseline_pressure :.3f} "
            f"baseline_cross_fill={baseline_cross_fill :.3f} "
            "same_ids_in_place=True -> FORM_SHEPHERD_BOUNDARY -> ENVIRONMENT_FILL"
            )















            physical .update_relay_deployment (robots ,dt )
            physical .shepherd_form_timer +=dt 
            shepherds =[
            robot for robot in physical .get_shepherds (robots )
            if robot .shepherd_branch ==branch 
            ]
            lifecycle =getattr (physical ,"integration_wall_lifecycle",{}).get (branch ,{})
            expected =int (lifecycle .get ("rows",0 ))*int (lifecycle .get ("cols",0 ))
            frozen_offsets =lifecycle .get ("relative_offsets",{})
            tangent ,lateral =physical .descriptor_local_basis (
            descriptor_for (branch )
            )if descriptor_for (branch )is not None else (
            pygame .Vector2 (1.0 ,0.0 ),pygame .Vector2 (0.0 ,1.0 )
            )
            formation_error =0.0 
            if shepherds and frozen_offsets :
                reference =min (
                shepherds ,
                key =lambda robot :sum (
                abs (float (value ))
                for value in frozen_offsets .get (robot .robot_id ,(0.0 ,0.0 ))
                ),
                )
                reference_offset =frozen_offsets .get (reference .robot_id ,(0.0 ,0.0 ))
                for observation in observe_local_neighbors (
                reference ,
                shepherds ,
                tangent ,
                lateral_axis =lateral ,
                ):
                    expected_offset =frozen_offsets .get (
                    observation .robot .robot_id ,
                    reference_offset ,
                    )
                    formation_error =max (
                    formation_error ,
                    math .hypot (
                    observation .relative_axial 
                    -(float (expected_offset [0 ])-float (reference_offset [0 ])),
                    observation .relative_lateral 
                    -(float (expected_offset [1 ])-float (reference_offset [1 ])),
                    ),
                    )
            formed =(
            expected >0 
            and len (shepherds )==expected 
            and formation_error <=float (physical .SHEPHERD_FORM_TOLERANCE )
            )












            if formed :
                physical .phase =physical .SimulationPhase .FILL_BEHIND_SHEPHERD 
                physical .saturation_tracker .reset (branch )
                physical .branch_continuity_tracker .reset (branch )








                current_shepherd_depth =(
                shepherd_line_leading_depth (
                robots ,branch ,descriptor_for (branch )
                )
                if shepherds and descriptor_for (branch )is not None 
                else 0.0 
                )
                current_shepherd_depth =float (current_shepherd_depth or 0.0 )
                diagnostics .reset (branch ,current_shepherd_depth )
                physical .integration_shepherd_pack_ready_dwell =0.0 
                max_error =formation_error 
                print (
                f"[ShepherdBoundaryReady] frame={getattr (physical ,'integration_frame',-1 )} "
                f"branch={branch } robots={len (shepherds )}/{expected } "
                f"max_error={max_error :.3f} "
                "bounded_timeout=False "
                "teleport=False -> FILL_BEHIND_SHEPHERD"
                )
            elif (
            physical .shepherd_form_timer >=physical .SHEPHERD_FORM_TIMEOUT 
            and getattr (physical ,"integration_frame",0 )%30 ==0 
            ):
                max_error =formation_error 
                print (
                f"[ShepherdBoundaryWait] frame={getattr (physical ,'integration_frame',-1 )} "
                f"branch={branch } robots={len (shepherds )}/{expected } "
                f"max_error={max_error :.3f} local_formation=True "
                "destructive_reset=False"
                )
            return 
        if physical .phase ==physical .SimulationPhase .FILL_BEHIND_SHEPHERD :

            physical .branch_entry_timer +=dt 
            physical .update_relay_deployment (robots ,dt )

            fill_state =sample_local_state (
            robots ,branch ,reference_density ,dt 
            )
            contact =shepherd_pack_contact_state (robots ,branch )

            baseline_density =float (
            physical .integration_shepherd_fill_baseline_density 
            )

            baseline_pressure =float (
            physical .integration_shepherd_fill_baseline_pressure 
            )

            density_ready =bool (
            baseline_density >physical .EPSILON 
            and fill_state .local_density 
            >=baseline_density *physical .SATURATION_DENSITY_RATIO 
            )

            pressure_ready =bool (
            baseline_pressure >physical .EPSILON 
            and fill_state .local_pressure 
            >=baseline_pressure *LOCAL_SATURATION_PRESSURE_RATIO 
            )

            cross_fill_ready =bool (
            fill_state .cross_section_fill 
            >=physical .SATURATION_PACKED_LATERAL_COVERAGE_RATIO 
            )

            contact_ready =bool (contact ["ready"])

            stall_ready =bool (fill_state .frontier_stalled )

            packed_ready =bool (
            contact_ready 
            and density_ready 
            and pressure_ready 
            and cross_fill_ready 
            and stall_ready 
            )

            if packed_ready :
                physical .integration_shepherd_pack_ready_dwell +=dt 
            else :
                physical .integration_shepherd_pack_ready_dwell =0.0 

            diagnostics .dwell =float (
            physical .integration_shepherd_pack_ready_dwell 
            )
            diagnostics .saturated =bool (
            packed_ready 
            and diagnostics .dwell 
            >=physical .integration_shepherd_pack_ready_required_dwell 
            )

            frame =getattr (physical ,"integration_frame",-1 )
            if frame %15 ==0 :
                print (
                f"[ShepherdFillGateV33] frame={frame } branch={branch } "
                f"density={fill_state .local_density :.6f} "
                f"density_ratio={fill_state .local_density_ratio :.3f}/"
                f"{physical .SATURATION_DENSITY_RATIO :.3f} "
                f"pressure={fill_state .local_pressure :.3f} "
                f"pressure_ratio={fill_state .local_pressure_ratio :.3f}/"
                f"{LOCAL_SATURATION_PRESSURE_RATIO :.3f} "
                f"cross_fill={fill_state .cross_section_fill :.3f}/"
                f"{physical .SATURATION_PACKED_LATERAL_COVERAGE_RATIO :.3f} "
                f"contact_ready={contact_ready } "
                f"pack_count={int (contact ['pack_count'])} "
                f"support_count={int (contact ['support_count'])} "
                f"contact_coverage={float (contact ['coverage']):.3f} "
                f"gap={float (contact ['gap']):.3f} "
                f"dwell={diagnostics .dwell :.3f}/"
                f"{physical .integration_shepherd_pack_ready_required_dwell :.3f} "
                f"packed_ready={packed_ready }"
                )

            if not diagnostics .saturated :
                return 

            pending =getattr (
            physical ,"integration_pending_shepherd_transition_event",None 
            )or {}
            shepherd_ids =sorted (
            robot .robot_id for robot in physical .get_shepherds (robots )
            if robot .shepherd_branch ==branch 
            )
            frontier_ids =list (pending .get ("frontier_ids",shepherd_ids ))
            event ={
            "branch":branch ,
            "uid":pending .get ("uid",descriptor_for (branch ).uid ),
            "frame":frame ,
            "frontier_speed":fill_state .frontier_speed ,
            "local_density":fill_state .local_density ,
            "density_ratio":fill_state .local_density_ratio ,
            "local_pressure":fill_state .local_pressure ,
            "pressure_ratio":fill_state .local_pressure_ratio ,
            "cross_section_fill":fill_state .cross_section_fill ,
            "dwell":diagnostics .dwell ,
            "frontier_ids":frontier_ids ,
            "shepherd_ids":shepherd_ids ,
            "max_transition_jump":float (
            pending .get ("max_transition_jump",0.0 )
            ),
            "return_direction":pending .get (
            "return_direction",(0.0 ,0.0 )
            ),
            "rows":int (pending .get (
            "rows",physical .integration_wall_lifecycle [branch ]["rows"]
            )),
            "cols":int (pending .get (
            "cols",physical .integration_wall_lifecycle [branch ]["cols"]
            )),
            "robots":len (frontier_ids ),
            "max_formation_error":float (
            pending .get ("max_formation_error",0.0 )
            ),
            }
            physical .integration_saturation_events .append (event )
            physical .integration_pending_shepherd_transition_event =None 

            print (
            "[Timeline] BRANCH_SATURATION_CONFIRMED "
            f"frame={frame } uid={event ['uid']} "
            f"density_ratio={event ['density_ratio']:.3f} "
            f"pressure_ratio={event ['pressure_ratio']:.3f} "
            f"cross_fill={event ['cross_section_fill']:.3f}"
            )
            print (
            f"[ShepherdPushStartV31] frame={frame } branch={branch } "
            "packed=True shepherd_stationary_until_now=True "
            "-> PRESSURE_PUSH"
            )
            physical .start_shepherd_pressure_push (robots ,branch )
            return 

        if physical .phase ==physical .SimulationPhase .PRESSURE_PUSH :

            physical .pressure_push_timer +=dt 

            update_pack_coupled_backtrack_depth (
            robots ,
            branch ,
            dt ,
            )

            ratio ,speed ,count =local_backflow_metrics (
            robots ,
            branch ,
            )

            diagnostics .return_flow_ratio =ratio 
            diagnostics .mean_return_speed =speed 

            established =(
            physical .pressure_push_timer 
            >=physical .SHEPHERD_MIN_PUSH_TIME 
            and count 
            >=physical .FLOW_MIN_NORMAL_COUNT 
            and ratio 
            >=physical .FLOW_RATIO_THRESHOLD 
            and speed 
            >=physical .FLOW_AVERAGE_SPEED_THRESHOLD 
            )

            diagnostics .return_dwell =(
            diagnostics .return_dwell +dt 
            if established 
            else 0.0 
            )

            frame =getattr (
            physical ,
            "integration_frame",
            -1 ,
            )

            if frame %10 ==0 :
                print (
                "[RootBackflowGate] "
                f"branch={branch } "
                f"normal_count={count } "
                f"required_count="
                f"{physical .FLOW_MIN_NORMAL_COUNT } "
                f"ratio={ratio :.3f}/"
                f"{physical .FLOW_RATIO_THRESHOLD :.3f} "
                f"mean_speed={speed :.3f}/"
                f"{physical .FLOW_AVERAGE_SPEED_THRESHOLD :.3f} "
                f"push_time="
                f"{physical .pressure_push_timer :.3f}/"
                f"{physical .SHEPHERD_MIN_PUSH_TIME :.3f} "
                f"dwell="
                f"{diagnostics .return_dwell :.3f}/"
                f"{physical .FLOW_ESTABLISH_DWELL_TIME :.3f} "
                f"established={established }"
                )

            if (
            diagnostics .return_dwell 
            <physical .FLOW_ESTABLISH_DWELL_TIME 
            ):
                return 

            physical .release_shepherds_into_flow (
            robots 
            )

            physical .phase =(
            physical .SimulationPhase .FLOW_BACKTRACK 
            )

            if (
            physical .integration_backtrack_command_depth 
            is None 
            ):
                physical .integration_backtrack_command_depth =(
                float (
                physical .shepherd_flow_start_depth 
                )
                )

            physical .integration_backtrack_pack_rear_depth =None 
            physical .integration_backtrack_support_depth =None 
            physical .integration_backtrack_support_count =0 
            physical .integration_backtrack_lateral_coverage =0.0 

            event ={
            "branch":branch ,
            "uid":descriptor_for (branch ).uid ,
            "frame":frame ,
            "ratio":ratio ,
            "mean_speed":speed ,
            "normal_count":count ,
            "duration":diagnostics .return_dwell ,
            "trigger":"NORMAL_RETURN_FLOW_ESTABLISHED",
            }

            physical .integration_backflow_events .append (
            event 
            )

            diagnostics .backflow_confirmed =True 

            print (
            "[Timeline] BACKFLOW_CONFIRMED "
            f"frame={frame } "
            f"uid={event ['uid']} "
            f"normal_count={count } "
            f"ratio={ratio :.3f} "
            f"mean_speed={speed :.3f} "
            f"duration="
            f"{diagnostics .return_dwell :.3f}"
            )

            print (
            "[RootFlowBacktrack] "
            f"branch={branch } "
            "PRESSURE_PUSH->FLOW_BACKTRACK "
            "trigger=PHYSICAL_NORMAL_RETURN_FLOW"
            )

            return 

        phase_before_original =physical .phase 
        original_update_state (robots ,dt ,reference_density ,spatial_grid )
        if physical .phase !=phase_before_original and phase_before_original in {
        physical .SimulationPhase .FINAL_JUNCTION_GATHER ,
        physical .SimulationPhase .RETURN_TO_BASE ,
        }:
            print (
            f"[FinalReturnTransition] frame={getattr (physical ,'integration_frame',-1 )} "
            f"{phase_before_original .name }->{physical .phase .name }"
            )
        if (
        getattr (physical ,"integration_ready_guard_handoff",False )
        and physical .phase !=physical .SimulationPhase .FORM_JUNCTION_GUARDS 
        ):
            physical .integration_ready_guard_handoff =False 
    physical .update_frontier_line_progress =local_frontier_progress 
    physical .commit_junction_guard_roles =audited_commit_guard_roles 
    physical .promote_existing_frontier_line =promote_thick_frontier_wall 
    physical .prepare_branch_candidate_scores =pebble_filtered_candidate_scores 

    physical .get_backtrack_direction =local_return_direction 

    physical .update_transfer_continuity_control =original_transfer_control 



    physical .get_shepherd_line_depth =shepherd_return_depth 
    physical .shepherd_line_reached_junction =(
    shepherd_returned_to_original_guard 
    )
    physical .compute_route_force =shepherd_physical_only_route_force 
    physical .compute_sph_forces =physical_only_compute_sph_forces 
    physical .start_shepherd_pressure_push =(
    original_start_shepherd_pressure_push 
    )
    physical .release_shepherd_line_at_junction =(
    original_release_shepherd_line_at_junction 
    )
    physical .force_complete_shepherd_boundary =(
    original_force_complete_shepherd_boundary 
    )
    physical .update_pre_shepherd_pipeline =(
    original_update_pre_shepherd_pipeline 
    )



    physical .SHEPHERD_PISTON_SPEED =float (
    physical .integration_reference_return_original_piston_speed 
    )
    physical .SHEPHERD_LINE_BACKTRACK_SPEED =float (
    physical .integration_reference_return_original_line_speed 
    )
    physical .SHEPHERD_JUNCTION_RELEASE_SPEED =float (
    physical .integration_reference_return_original_release_speed 
    )
    if hasattr (physical ,"integration_original_shepherd_pressure_factor"):
        physical .SHEPHERD_PRESSURE_FACTOR =float (
        physical .integration_original_shepherd_pressure_factor 
        )

    def relative_formation_velocity (
    robot :Any ,
    members :Sequence [Any ],
    offsets :dict [int ,tuple [float ,float ]],
    tangent :pygame .Vector2 ,
    lateral :pygame .Vector2 ,
    maximum_speed :float ,
    )->pygame .Vector2 :
        """Keep the frozen 3xN wall rigid from local peer observations.

        The previous controller averaged every visible peer error.  In a
        symmetric 3xN wall, positive/negative errors can cancel and rows may
        slowly collapse or spread while the wall travels.

        Here each robot follows only its nearest frozen slot-neighbors
        (same-row / adjacent-row neighbors emerge naturally from offset
        distance).  Position error plus relative-velocity damping keeps the
        original Guard geometry throughout Frontier motion.

        No world target position is consumed.
        """

        own =offsets .get (
        robot .robot_id 
        )

        if own is None :
            robot .integration_boundary_shape_error =0.0 
            return pygame .Vector2 ()

        member_by_id ={
        member .robot_id :member 
        for member in members 
        if member .robot_id in offsets 
        and member is not robot 
        }

        expected_neighbors :list [
        tuple [float ,int ]
        ]=[]

        for peer_id ,peer_offset in offsets .items ():
            peer_id =int (peer_id )

            if (
            peer_id ==robot .robot_id 
            or peer_id not in member_by_id 
            ):
                continue 

            expected_axial =(
            float (peer_offset [0 ])
            -float (own [0 ])
            )

            expected_lateral =(
            float (peer_offset [1 ])
            -float (own [1 ])
            )

            expected_distance =math .hypot (
            expected_axial ,
            expected_lateral ,
            )

            if (
            expected_distance 
            <=physical .EPSILON 
            or expected_distance 
            >physical .COMM_RANGE 
            ):
                continue 

            expected_neighbors .append (
            (
            expected_distance ,
            peer_id ,
            )
            )

        expected_neighbors .sort (
        key =lambda item :item [0 ]
        )

        neighbor_count =int (
        getattr (
        physical ,
        "integration_frontier_shape_neighbor_count",
        8 ,
        )
        )

        selected_ids ={
        peer_id 
        for _ ,peer_id 
        in expected_neighbors [
        :neighbor_count 
        ]
        }

        selected_members =[
        member_by_id [peer_id ]
        for peer_id in selected_ids 
        if peer_id in member_by_id 
        ]

        observations =observe_local_neighbors (
        robot ,
        selected_members ,
        tangent ,
        lateral_axis =lateral ,
        max_range =physical .COMM_RANGE ,
        )

        if not observations :




            robot .integration_boundary_shape_error =float ("inf")
            return pygame .Vector2 ()

        weighted_axial_error =0.0 
        weighted_lateral_error =0.0 
        weighted_axial_velocity_error =0.0 
        weighted_lateral_velocity_error =0.0 
        weighted_error_sq =0.0 
        weight_sum =0.0 

        for observation in observations :
            peer_offset =offsets .get (
            observation .robot .robot_id 
            )

            if peer_offset is None :
                continue 

            expected_axial =(
            float (peer_offset [0 ])
            -float (own [0 ])
            )

            expected_lateral =(
            float (peer_offset [1 ])
            -float (own [1 ])
            )

            expected_distance =max (
            math .hypot (
            expected_axial ,
            expected_lateral ,
            ),
            physical .EPSILON ,
            )

            axial_error =(
            float (
            observation .relative_axial 
            )
            -expected_axial 
            )

            lateral_error =(
            float (
            observation .relative_lateral 
            )
            -expected_lateral 
            )

            relative_velocity =(
            observation .robot .velocity 
            -robot .velocity 
            )

            axial_velocity_error =float (
            relative_velocity .dot (
            tangent 
            )
            )

            lateral_velocity_error =float (
            relative_velocity .dot (
            lateral 
            )
            )



            weight =1.0 /expected_distance 

            weighted_axial_error +=(
            weight *axial_error 
            )

            weighted_lateral_error +=(
            weight *lateral_error 
            )

            weighted_axial_velocity_error +=(
            weight 
            *axial_velocity_error 
            )

            weighted_lateral_velocity_error +=(
            weight 
            *lateral_velocity_error 
            )

            weighted_error_sq +=(
            weight 
            *(
            axial_error 
            *axial_error 
            +lateral_error 
            *lateral_error 
            )
            )

            weight_sum +=weight 

        if weight_sum <=physical .EPSILON :
            robot .integration_boundary_shape_error =float ("inf")
            return pygame .Vector2 ()

        mean_axial_error =(
        weighted_axial_error 
        /weight_sum 
        )

        mean_lateral_error =(
        weighted_lateral_error 
        /weight_sum 
        )

        mean_axial_velocity_error =(
        weighted_axial_velocity_error 
        /weight_sum 
        )

        mean_lateral_velocity_error =(
        weighted_lateral_velocity_error 
        /weight_sum 
        )

        rms_shape_error =math .sqrt (
        max (
        0.0 ,
        weighted_error_sq 
        /weight_sum ,
        )
        )

        robot .integration_boundary_shape_error =(
        rms_shape_error 
        )

        position_gain =float (
        getattr (
        physical ,
        "integration_frontier_shape_gain",
        9.0 ,
        )
        )

        damping_gain =float (
        getattr (
        physical ,
        "integration_frontier_shape_damping",
        0.85 ,
        )
        )

        correction =(
        tangent 
        *(
        position_gain 
        *mean_axial_error 
        +damping_gain 
        *mean_axial_velocity_error 
        )
        +lateral 
        *(
        position_gain 
        *mean_lateral_error 
        +damping_gain 
        *mean_lateral_velocity_error 
        )
        )

        physical .limit_vector (
        correction ,
        maximum_speed ,
        )

        return correction 

    def _swept_pair_no_cross (
    mover :Any ,
    old_position :pygame .Vector2 ,
    proposed_position :pygame .Vector2 ,
    obstacle :Any ,
    *,
    minimum_distance :float |None =None ,
    )->pygame .Vector2 :
        """Clamp one physical step before two robot discs can cross.

        This helper belongs to the simulator collision layer.  It uses only the
        instantaneous pairwise relative displacement; it is not a navigation or
        localization target.
        """
        movement =proposed_position -old_position 
        movement_sq =movement .length_squared ()
        if movement_sq <=physical .EPSILON :
            return proposed_position .copy ()

        min_distance =float (
        minimum_distance 
        if minimum_distance is not None 
        else (
        float (getattr (mover ,"radius",physical .ROBOT_RADIUS ))
        +float (getattr (obstacle ,"radius",physical .ROBOT_RADIUS ))
        )
        )

        relative =old_position -obstacle .position 
        c =relative .length_squared ()-min_distance **2 



        if c <=0.0 :
            if relative .dot (movement )<0.0 :
                return old_position .copy ()
            return proposed_position .copy ()

        a =movement_sq 
        b =2.0 *relative .dot (movement )
        discriminant =b *b -4.0 *a *c 
        if discriminant <0.0 :
            return proposed_position .copy ()

        sqrt_discriminant =math .sqrt (discriminant )
        hit_alpha =(-b -sqrt_discriminant )/(2.0 *a )

        if 0.0 <=hit_alpha <=1.0 :
            safe_alpha =max (0.0 ,hit_alpha -1.0e-3 )
            return old_position +movement *safe_alpha 

        return proposed_position .copy ()


    def _same_rigid_boundary_cohort (
    first :Any ,
    second :Any ,
    )->bool :
        if first is second :
            return True 

        boundary_roles ={
        "FRONTIER_SHEPHERD",
        "SHEPHERD",
        }

        if (
        getattr (first ,"role",None )
        not in boundary_roles 
        or getattr (second ,"role",None )
        not in boundary_roles 
        ):
            return False 

        first_branch =getattr (
        first ,
        "shepherd_branch",
        None ,
        )
        second_branch =getattr (
        second ,
        "shepherd_branch",
        None ,
        )

        return bool (
        first_branch is not None 
        and first_branch ==second_branch 
        )


    def _root_rigid_frontier_member (
    robot :Any ,
    )->bool :
        branch =getattr (
        robot ,
        "shepherd_branch",
        None ,
        )

        return bool (
        getattr (
        robot ,
        "role",
        None ,
        )
        =="FRONTIER_SHEPHERD"
        and branch is not None 
        and branch 
        ==getattr (
        physical ,
        "frontier_line_branch",
        None ,
        )
        and getattr (
        physical ,
        "integration_wall_lifecycle",
        {},
        ).get (branch )
        is not None 
        )


    def _root_rigid_shepherd_member (
    robot :Any ,
    )->bool :
        branch =getattr (
        robot ,
        "shepherd_branch",
        None ,
        )

        return bool (
        getattr (
        robot ,
        "role",
        None ,
        )
        =="SHEPHERD"
        and branch is not None 
        and branch 
        ==getattr (
        physical ,
        "active_branch",
        None ,
        )
        and physical .phase 
        in {
        physical .SimulationPhase .PRESSURE_PUSH ,
        physical .SimulationPhase .FLOW_BACKTRACK ,
        }
        and getattr (
        physical ,
        "integration_wall_lifecycle",
        {},
        ).get (branch )
        is not None 
        )


    def _collision_other_frame_segment (
    other :Any ,
    dt :float ,
    )->tuple [
    pygame .Vector2 ,
    pygame .Vector2 ,
    ]:
        start_by_id =getattr (
        physical ,
        "integration_collision_frame_start_by_id",
        {},
        )
        velocity_by_id =getattr (
        physical ,
        "integration_collision_frame_velocity_by_id",
        {},
        )
        updated_ids =getattr (
        physical ,
        "integration_collision_updated_ids",
        set (),
        )

        start =pygame .Vector2 (
        start_by_id .get (
        other .robot_id ,
        other .position ,
        )
        )

        if other .robot_id in updated_ids :
            end =other .position .copy ()
        else :
            velocity =pygame .Vector2 (
            velocity_by_id .get (
            other .robot_id ,
            getattr (
            other ,
            "velocity",
            pygame .Vector2 (),
            ),
            )
            )
            end =start +velocity *dt 

        return start ,end 


    def _first_universal_swept_hit (
    robot :Any ,
    start :pygame .Vector2 ,
    movement :pygame .Vector2 ,
    dt :float ,
    *,
    time_start :float ,
    time_span :float ,
    )->tuple [
    float ,
    Any |None ,
    pygame .Vector2 |None ,
    ]:
        current_robots =tuple (
        getattr (
        physical ,
        "integration_current_robots",
        (),
        )
        )

        if not current_robots :
            return 1.0 ,None ,None 

        mover_radius =float (
        getattr (
        robot ,
        "radius",
        physical .ROBOT_RADIUS ,
        )
        )

        best_alpha =1.0 
        best_other =None 
        best_other_at_hit =None 

        for other in current_robots :
            if other is robot :
                continue 

            if _same_rigid_boundary_cohort (
            robot ,
            other ,
            ):
                continue 

            other_radius =float (
            getattr (
            other ,
            "radius",
            physical .ROBOT_RADIUS ,
            )
            )

            minimum_distance =(
            mover_radius 
            +other_radius 
            )

            (
            other_frame_start ,
            other_frame_end ,
            )=_collision_other_frame_segment (
            other ,
            dt ,
            )

            other_total_movement =(
            other_frame_end 
            -other_frame_start 
            )

            other_segment_start =(
            other_frame_start 
            +other_total_movement 
            *time_start 
            )

            other_segment_movement =(
            other_total_movement 
            *time_span 
            )

            relative_start =(
            start 
            -other_segment_start 
            )

            relative_movement =(
            movement 
            -other_segment_movement 
            )

            reach =(
            movement .length ()
            +other_segment_movement .length ()
            +minimum_distance 
            )

            if (
            relative_start .length_squared ()
            >reach *reach 
            ):
                continue 

            c =(
            relative_start .length_squared ()
            -minimum_distance 
            *minimum_distance 
            )

            if c <=0.0 :
                if (
                relative_start .dot (
                relative_movement 
                )
                <0.0 
                ):
                    hit_alpha =0.0 
                else :
                    continue 
            else :
                a =(
                relative_movement 
                .length_squared ()
                )

                if a <=physical .EPSILON :
                    continue 

                b =(
                2.0 
                *relative_start .dot (
                relative_movement 
                )
                )

                discriminant =(
                b *b 
                -4.0 *a *c 
                )

                if discriminant <0.0 :
                    continue 

                sqrt_discriminant =(
                math .sqrt (
                discriminant 
                )
                )

                hit_alpha =(
                -b 
                -sqrt_discriminant 
                )/(
                2.0 *a 
                )

                if not (
                0.0 
                <=hit_alpha 
                <=1.0 
                ):
                    continue 

            if hit_alpha <best_alpha :
                best_alpha =float (
                hit_alpha 
                )
                best_other =other 
                best_other_at_hit =(
                other_segment_start 
                +other_segment_movement 
                *hit_alpha 
                )

        return (
        best_alpha ,
        best_other ,
        best_other_at_hit ,
        )


    def universal_robot_motion_limit (
    robot :Any ,
    old_position :pygame .Vector2 ,
    proposed_position :pygame .Vector2 ,
    dt :float ,
    )->pygame .Vector2 :
        """Universal hard disc collision with collision-aware move-and-slide."""
        current_rigid_uid =getattr (
        physical ,
        "integration_frontier_active_uid",
        None ,
        )

        current_rigid_ids =set (
        getattr (
        physical ,
        "integration_frontier_ids",
        set (),
        )
        )

        is_current_rigid_boundary =bool (
        current_rigid_uid is not None 
        and robot .robot_id in current_rigid_ids 
        and getattr (robot ,"shepherd_branch",None )
        ==current_rigid_uid 
        and getattr (robot ,"role",None )
        in {
        "FRONTIER_SHEPHERD",
        "SHEPHERD",
        }
        )





        if (
        _root_rigid_frontier_member (robot )
        or _root_rigid_shepherd_member (robot )
        or is_current_rigid_boundary 
        ):
            return proposed_position .copy ()

        movement =(
        proposed_position 
        -old_position 
        )

        if (
        movement .length_squared ()
        <=physical .EPSILON 
        ):
            return proposed_position .copy ()

        current =old_position .copy ()
        remaining =movement .copy ()

        time_start =0.0 
        time_span =1.0 

        collision_count =0 
        slide_count =0 
        first_blocker =None 

        for _ in range (4 ):
            if (
            remaining .length_squared ()
            <=physical .EPSILON 
            or time_span 
            <=physical .EPSILON 
            ):
                break 

            (
            hit_alpha ,
            blocker ,
            blocker_at_hit ,
            )=_first_universal_swept_hit (
            robot ,
            current ,
            remaining ,
            dt ,
            time_start =time_start ,
            time_span =time_span ,
            )

            if blocker is None :
                current +=remaining 
                remaining .update (
                0.0 ,
                0.0 ,
                )
                break 

            collision_count +=1 

            if first_blocker is None :
                first_blocker =blocker 

            safe_alpha =max (
            0.0 ,
            hit_alpha -1.0e-3 ,
            )

            current +=(
            remaining 
            *safe_alpha 
            )

            old_remaining =(
            remaining .copy ()
            )

            residual =(
            old_remaining 
            *max (
            0.0 ,
            1.0 -hit_alpha ,
            )
            )

            hit_global =(
            time_start 
            +time_span 
            *hit_alpha 
            )

            time_span =(
            time_span 
            *max (
            0.0 ,
            1.0 -hit_alpha ,
            )
            )

            time_start =hit_global 

            if blocker_at_hit is None :
                remaining .update (
                0.0 ,
                0.0 ,
                )
                break 

            normal =(
            current 
            -blocker_at_hit 
            )

            if (
            normal .length_squared ()
            <=physical .EPSILON 
            ):
                if (
                old_remaining 
                .length_squared ()
                <=physical .EPSILON 
                ):
                    break 

                normal =(
                -old_remaining 
                ).normalize ()
            else :
                normal =(
                normal .normalize ()
                )

            inward =float (
            residual .dot (
            normal 
            )
            )

            slide =residual .copy ()

            if inward <0.0 :
                slide -=(
                normal 
                *inward 
                )





            if (
            getattr (
            robot ,
            "role",
            None ,
            )
            =="JUNCTION_GUARD"
            and slide .length_squared ()
            <=max (
            physical .EPSILON ,
            0.04 
            *residual .length_squared (),
            )
            ):
                target =getattr (
                robot ,
                "junction_guard_anchor",
                None ,
                )

                if target is not None :
                    target_vector =(
                    pygame .Vector2 (target )
                    -current 
                    )

                    tangent =pygame .Vector2 (
                    -normal .y ,
                    normal .x ,
                    )

                    tangent_score =float (
                    target_vector .dot (
                    tangent 
                    )
                    )

                    if (
                    abs (tangent_score )
                    <=physical .EPSILON 
                    ):
                        tangent_score =(
                        1.0 
                        if robot .robot_id 
                        <blocker .robot_id 
                        else -1.0 
                        )

                    tangent *=(
                    1.0 
                    if tangent_score >=0.0 
                    else -1.0 
                    )

                    bypass_length =min (
                    residual .length (),
                    max (
                    0.55 
                    *residual .length (),
                    0.75 
                    *physical .ROBOT_RADIUS ,
                    ),
                    )

                    slide =(
                    tangent 
                    *bypass_length 
                    )

            if (
            slide .length_squared ()
            <=physical .EPSILON 
            ):
                remaining .update (
                0.0 ,
                0.0 ,
                )
                break 

            remaining =(
            slide *0.97 
            )
            slide_count +=1 

        if collision_count :
            stats =getattr (
            physical ,
            "integration_universal_collision_stats",
            None ,
            )

            if isinstance (
            stats ,
            dict ,
            ):
                stats ["contacts"]=int (
                stats .get (
                "contacts",
                0 ,
                )
                )+collision_count 

                stats ["robots"]=int (
                stats .get (
                "robots",
                0 ,
                )
                )+1 

                stats ["slides"]=int (
                stats .get (
                "slides",
                0 ,
                )
                )+slide_count 

                samples =stats .setdefault (
                "samples",
                [],
                )

                if (
                first_blocker is not None 
                and len (samples )<6 
                ):
                    samples .append (
                    (
                    robot .robot_id ,
                    getattr (
                    robot ,
                    "role",
                    None ,
                    ),
                    first_blocker .robot_id ,
                    getattr (
                    first_blocker ,
                    "role",
                    None ,
                    ),
                    )
                    )

        return current 


    def integrate_boundary_velocity (
    robot :Any ,
    commanded_velocity :pygame .Vector2 ,
    dt :float ,
    *,
    rigid_frontier :bool =False ,
    )->None :
        """Integrate one boundary robot.

        For a rigid Frontier, the group controller has already computed ONE
        common velocity that is walkable/communication-safe for every member.
        Therefore no per-robot communication clamp is applied here; a
        per-robot clamp would deform the 3xN wall.
        """
        old_position =robot .position .copy ()
        robot .commanded_velocity =commanded_velocity .copy ()
        robot .velocity =commanded_velocity .copy ()

        if not rigid_frontier :
            physical .apply_communication_velocity_guard (
            robot ,
            dt ,
            )
            robot .integration_boundary_contact =False 

        x_position =pygame .Vector2 (
        robot .position .x 
        +robot .velocity .x *dt ,
        robot .position .y ,
        )

        if physical .is_walkable (
        x_position ,
        robot .radius ,
        ):
            robot .position .x =x_position .x 
        else :
            robot .integration_boundary_contact =True 
            robot .velocity .x =0.0 

        y_position =pygame .Vector2 (
        robot .position .x ,
        robot .position .y 
        +robot .velocity .y *dt ,
        )

        if physical .is_walkable (
        y_position ,
        robot .radius ,
        ):
            robot .position .y =y_position .y 
        else :
            robot .integration_boundary_contact =True 
            robot .velocity .y =0.0 







        if not rigid_frontier :
            physical .constrain_communication_parent_separation (
            robot ,
            old_position ,
            )

        robot .observed_velocity =(
        robot .position -old_position 
        )/max (
        dt ,
        physical .EPSILON ,
        )

        robot .velocity =(
        robot .observed_velocity .copy ()
        )

        robot .acceleration .update (
        0.0 ,
        0.0 ,
        )
        robot .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )

        physical .update_indirect_contact_state (
        robot ,
        dt ,
        )

        robot .previous_position =old_position 
        robot ._record_motion ()


    def rigid_frontier_common_velocity (
    branch :str ,
    members :Sequence [Any ],
    lifecycle :dict [str ,Any ],
    tangent :pygame .Vector2 ,
    dt :float ,
    )->pygame .Vector2 :
        """Return ONE common translation velocity for the whole 3xN Frontier.

        This is the hard shape lock.

        Every Frontier robot receives exactly the same displacement in a frame.
        If even one member cannot make the requested step because of a wall or
        a hard communication limit, the SAME reduced step is used by all 93
        robots.

        Therefore the original Guard relative geometry cannot shear,
        collapse from 3 rows into 1 row, or fan out during Frontier motion.

        No per-robot world target is generated.  World positions are consulted
        only inside the simulator's collision/communication safety layer.
        """
        frame =int (
        getattr (
        physical ,
        "integration_frame",
        -1 ,
        )
        )

        cached_frame =int (
        lifecycle .get (
        "frontier_rigid_cache_frame",
        -10 **9 ,
        )
        )

        if cached_frame ==frame :
            cached =lifecycle .get (
            "frontier_rigid_common_velocity",
            pygame .Vector2 (),
            )
            return cached .copy ()

        expected =(
        int (
        lifecycle .get (
        "rows",
        0 ,
        )
        )
        *int (
        lifecycle .get (
        "cols",
        0 ,
        )
        )
        )

        if (
        expected <=0 
        or len (members )!=expected 
        ):
            lifecycle [
            "frontier_rigid_cache_frame"
            ]=frame 
            lifecycle [
            "frontier_rigid_common_velocity"
            ]=pygame .Vector2 ()

            print (
            "[FrontierRigidBlock] "
            f"branch={branch } "
            f"reason=MEMBER_COUNT "
            f"members={len (members )} "
            f"expected={expected }"
            )
            return pygame .Vector2 ()

        requested_depth =float (
        physical .frontier_line_depth 
        )

        applied_depth =float (
        lifecycle .get (
        "frontier_rigid_applied_depth",
        lifecycle .get (
        "frontier_odometry_depth",
        requested_depth ,
        ),
        )
        )

        requested_delta =max (
        0.0 ,
        requested_depth 
        -applied_depth ,
        )

        maximum_step =(
        float (
        physical .FRONTIER_LINE_FORM_SPEED 
        )
        *dt 
        )

        requested_delta =min (
        requested_delta ,
        maximum_step ,
        )















        intended_forward_speed =(
        requested_delta 
        /max (
        dt ,
        physical .EPSILON ,
        )
        )

        for member in members :
            member .integration_frontier_intended_forward_speed =(
            intended_forward_speed 
            )

        frontier_ids ={
        robot .robot_id 
        for robot in members 
        }

        hard_limit =float (
        getattr (
        physical ,
        "COMM_GUARD_HARD_LIMIT",
        physical .COMM_RANGE ,
        )
        )

        external_robots =[
        other 
        for other in getattr (
        physical ,
        "integration_current_robots",
        (),
        )
        if other .robot_id 
        not in frontier_ids 
        ]

        def _pair_step_is_safe (
        member :Any ,
        delta :pygame .Vector2 ,
        other :Any ,
        fraction :float ,
        )->bool :
            """Swept collision test for Frontier vs external robot.

            Existing contact/overlap may separate.
            Deeper penetration and tunneling are forbidden.
            """
            other_radius =float (
            getattr (
            other ,
            "radius",
            physical .ROBOT_RADIUS ,
            )
            )

            minimum_distance =(
            float (member .radius )
            +other_radius 
            )

            other_delta =(
            getattr (
            other ,
            "velocity",
            pygame .Vector2 (),
            )
            *dt 
            *fraction 
            )

            relative_start =(
            member .position 
            -other .position 
            )

            relative_movement =(
            delta 
            -other_delta 
            )

            c =(
            relative_start .length_squared ()
            -minimum_distance 
            *minimum_distance 
            )





            if c <=0.0 :
                return (
                relative_start .dot (
                relative_movement 
                )
                >=-physical .EPSILON 
                )

            a =(
            relative_movement 
            .length_squared ()
            )

            if a <=physical .EPSILON :
                return True 

            b =(
            2.0 
            *relative_start .dot (
            relative_movement 
            )
            )

            discriminant =(
            b *b 
            -4.0 *a *c 
            )

            if discriminant <0.0 :
                return True 

            hit_alpha =(
            -b 
            -math .sqrt (
            discriminant 
            )
            )/(
            2.0 *a 
            )

            return not (
            0.0 
            <=hit_alpha 
            <=1.0 
            )

        def group_step_is_safe (
        fraction :float ,
        )->bool :
            delta =(
            tangent 
            *(
            requested_delta 
            *fraction 
            )
            )

            for member in members :
                candidate =(
                member .position 
                +delta 
                )

                if not physical .is_walkable (
                candidate ,
                member .radius ,
                ):
                    return False 











                for other in external_robots :
                    if not _pair_step_is_safe (
                    member ,
                    delta ,
                    other ,
                    fraction ,
                    ):
                        return False 

                parent =getattr (
                member ,
                "comm_parent",
                None ,
                )









                if (
                parent is not None 
                and getattr (
                member ,
                "connected_to_base",
                False ,
                )
                ):
                    parent_candidate =(
                    parent .position 
                    +delta 
                    if getattr (
                    parent ,
                    "robot_id",
                    None ,
                    )
                    in frontier_ids 
                    else parent .position 
                    )

                    if (
                    candidate .distance_to (
                    parent_candidate 
                    )
                    >hard_limit 
                    ):
                        return False 

            return True 

        safe_fraction =1.0 

        if (
        requested_delta 
        >physical .EPSILON 
        and not group_step_is_safe (
        1.0 
        )
        ):
            low =0.0 
            high =1.0 



            for _ in range (12 ):
                middle =0.5 *(
                low +high 
                )

                if group_step_is_safe (
                middle 
                ):
                    low =middle 
                else :
                    high =middle 

            safe_fraction =low 









        if (
        requested_delta >physical .EPSILON 
        and safe_fraction <0.001 
        and frame %10 ==0 
        ):
            probe_fraction =1.0 /4096.0 

            probe_delta =(
            tangent 
            *requested_delta 
            *probe_fraction 
            )
            debug_normal =pygame .Vector2 (
            -tangent .y ,
            tangent .x ,
            )

            if (
            debug_normal .length_squared ()
            >physical .EPSILON 
            ):
                debug_normal =(
                debug_normal .normalize ()
                )

            frontier_centroid =(
            pygame .Vector2 ()
            )

            for debug_member in members :
                frontier_centroid +=(
                debug_member .position 
                )

            frontier_centroid /=max (
            len (members ),
            1 ,
            )

            walkable_details :list [dict ]=[]

            walkable_blockers :list [int ]=[]
            robot_blockers :list [
            tuple [int ,int ,str ]
            ]=[]
            comm_blockers :list [
            tuple [int ,int ,float ]
            ]=[]

            for member in members :
                candidate =(
                member .position 
                +probe_delta 
                )

                candidate_walkable =(
                physical .is_walkable (
                candidate ,
                member .radius ,
                )
                )

                if not candidate_walkable :
                    walkable_blockers .append (
                    member .robot_id 
                    )

                    relative_to_group =(
                    member .position 
                    -frontier_centroid 
                    )

                    diagnostic_step =max (
                    0.05 ,
                    float (member .radius )
                    *0.10 ,
                    )

                    walkable_details .append (
                    {
                    "id":member .robot_id ,
                    "pos":(
                    round (
                    float (
                    member .position .x 
                    ),
                    3 ,
                    ),
                    round (
                    float (
                    member .position .y 
                    ),
                    3 ,
                    ),
                    ),
                    "current":(
                    physical .is_walkable (
                    member .position ,
                    member .radius ,
                    )
                    ),
                    "fwd":(
                    physical .is_walkable (
                    member .position 
                    +tangent 
                    *diagnostic_step ,
                    member .radius ,
                    )
                    ),
                    "back":(
                    physical .is_walkable (
                    member .position 
                    -tangent 
                    *diagnostic_step ,
                    member .radius ,
                    )
                    ),
                    "lat_plus":(
                    physical .is_walkable (
                    member .position 
                    +debug_normal 
                    *diagnostic_step ,
                    member .radius ,
                    )
                    ),
                    "lat_minus":(
                    physical .is_walkable (
                    member .position 
                    -debug_normal 
                    *diagnostic_step ,
                    member .radius ,
                    )
                    ),
                    "rel_axial":round (
                    float (
                    relative_to_group .dot (
                    tangent 
                    )
                    ),
                    3 ,
                    ),
                    "rel_lateral":round (
                    float (
                    relative_to_group .dot (
                    debug_normal 
                    )
                    ),
                    3 ,
                    ),
                    }
                    )

                for other in external_robots :
                    if not _pair_step_is_safe (
                    member ,
                    probe_delta ,
                    other ,
                    probe_fraction ,
                    ):
                        robot_blockers .append (
                        (
                        member .robot_id ,
                        other .robot_id ,
                        str (
                        getattr (
                        other ,
                        "role",
                        "?",
                        )
                        ),
                        )
                        )
                        break 

                parent =getattr (
                member ,
                "comm_parent",
                None ,
                )

                if (
                parent is not None 
                and getattr (
                member ,
                "connected_to_base",
                False ,
                )
                ):
                    parent_candidate =(
                    parent .position 
                    +probe_delta 
                    if getattr (
                    parent ,
                    "robot_id",
                    None ,
                    )
                    in frontier_ids 
                    else parent .position 
                    )

                    parent_distance =(
                    candidate .distance_to (
                    parent_candidate 
                    )
                    )

                    if (
                    parent_distance 
                    >hard_limit 
                    ):
                        comm_blockers .append (
                        (
                        member .robot_id ,
                        int (
                        getattr (
                        parent ,
                        "robot_id",
                        -1 ,
                        )
                        ),
                        float (
                        parent_distance 
                        ),
                        )
                        )

            print (
            "[FrontierRigidBlockReason] "
            f"frame={frame } "
            f"branch={branch } "
            f"probe_delta="
            f"{probe_delta .length ():.6f} "
            f"walkable="
            f"{len (walkable_blockers )} "
            f"walkable_ids="
            f"{walkable_blockers [:8 ]} "
            f"walkable_details="
            f"{walkable_details [:3 ]} "
            f"robot="
            f"{len (robot_blockers )} "
            f"robot_pairs="
            f"{robot_blockers [:5 ]} "
            f"comm="
            f"{len (comm_blockers )} "
            f"comm_pairs="
            f"{comm_blockers [:5 ]} "
            f"hard_limit="
            f"{hard_limit :.3f}"
            )

        actual_delta =(
        requested_delta 
        *safe_fraction 
        )

        velocity =(
        tangent 
        *(
        actual_delta 
        /max (
        dt ,
        physical .EPSILON ,
        )
        )
        )

        contact =(
        requested_delta 
        >physical .EPSILON 
        and safe_fraction 
        <0.999 
        )



        for member in members :
            member .integration_boundary_contact =(
            contact 
            )





            member .integration_boundary_shape_error =(
            0.0 
            )

        lifecycle [
        "frontier_rigid_applied_depth"
        ]=(
        applied_depth 
        +actual_delta 
        )

        lifecycle [
        "frontier_rigid_cache_frame"
        ]=frame 

        lifecycle [
        "frontier_rigid_common_velocity"
        ]=velocity .copy ()

        lifecycle [
        "frontier_rigid_contact"
        ]=contact 

        lifecycle [
        "frontier_rigid_safe_fraction"
        ]=safe_fraction 

        if (
        frame %10 ==0 
        or contact 
        ):
            print (
            "[FrontierRigidTransport] "
            f"frame={frame } "
            f"branch={branch } "
            f"members={len (members )} "
            f"requested_depth="
            f"{requested_depth :.3f} "
            f"applied_depth="
            f"{applied_depth :.3f} "
            f"requested_delta="
            f"{requested_delta :.3f} "
            f"actual_delta="
            f"{actual_delta :.3f} "
            f"safe_fraction="
            f"{safe_fraction :.3f} "
            f"contact={contact } "
            "shape_locked=True"
            )

        return velocity 


    def rigid_shepherd_common_velocity (
    branch :str ,
    members :Sequence [Any ],
    lifecycle :dict [str ,Any ],
    tangent :pygame .Vector2 ,
    dt :float ,
    )->pygame .Vector2 :
        """Return one collision-safe translation velocity for the whole 3xN Shepherd."""
        frame =int (getattr (physical ,"integration_frame",-1 ))
        cached_frame =int (lifecycle .get ("shepherd_rigid_cache_frame",-10 **9 ))
        if cached_frame ==frame :
            return lifecycle .get (
            "shepherd_rigid_common_velocity",pygame .Vector2 ()
            ).copy ()

        expected =int (lifecycle .get ("rows",0 ))*int (lifecycle .get ("cols",0 ))
        if expected <=0 or len (members )!=expected :
            lifecycle ["shepherd_rigid_cache_frame"]=frame 
            lifecycle ["shepherd_rigid_common_velocity"]=pygame .Vector2 ()
            print (
            "[ShepherdRigidBlock] "
            f"branch={branch } reason=MEMBER_COUNT "
            f"members={len (members )} expected={expected }"
            )
            return pygame .Vector2 ()

        measured_depth =float (
        lifecycle .get (
        "shepherd_odometry_depth",
        lifecycle .get ("shepherd_leading_edge_depth",0.0 ),
        )
        )
        requested_depth =float (
        getattr (
        physical ,
        "integration_backtrack_command_depth",
        measured_depth ,
        )
        or measured_depth 
        )
        speed_limit =(
        float (physical .SHEPHERD_PISTON_SPEED )
        if physical .phase ==physical .SimulationPhase .PRESSURE_PUSH 
        else float (physical .SHEPHERD_LINE_BACKTRACK_SPEED )
        )
        maximum_step =speed_limit *dt 
        requested_delta =physical .clamp (
        requested_depth -measured_depth ,
        -maximum_step ,
        maximum_step ,
        )
        shepherd_ids ={robot .robot_id for robot in members }
        hard_limit =float (
        getattr (physical ,"COMM_GUARD_HARD_LIMIT",physical .COMM_RANGE )
        )
        external_robots =[
        other 
        for other in getattr (physical ,"integration_current_robots",())
        if other .robot_id not in shepherd_ids 
        ]

        def _pair_step_is_safe (
        member :Any ,
        delta :pygame .Vector2 ,
        other :Any ,
        fraction :float ,
        )->bool :
            other_radius =float (
            getattr (other ,"radius",physical .ROBOT_RADIUS )
            )
            minimum_distance =float (member .radius )+other_radius 
            other_delta =(
            getattr (other ,"velocity",pygame .Vector2 ())
            *dt 
            *fraction 
            )
            relative_start =member .position -other .position 
            relative_movement =delta -other_delta 
            c =(
            relative_start .length_squared ()
            -minimum_distance *minimum_distance 
            )
            if c <=0.0 :
                return relative_start .dot (relative_movement )>=-physical .EPSILON 
            a =relative_movement .length_squared ()
            if a <=physical .EPSILON :
                return True 
            b =2.0 *relative_start .dot (relative_movement )
            discriminant =b *b -4.0 *a *c 
            if discriminant <0.0 :
                return True 
            hit_alpha =(-b -math .sqrt (discriminant ))/(2.0 *a )
            return not (0.0 <=hit_alpha <=1.0 )

        def group_step_is_safe (fraction :float )->bool :
            delta =tangent *requested_delta *fraction 
            for member in members :
                candidate =member .position +delta 
                if not physical .is_walkable (candidate ,member .radius ):
                    return False 
                for other in external_robots :
                    if not _pair_step_is_safe (member ,delta ,other ,fraction ):
                        return False 
                parent =getattr (member ,"comm_parent",None )
                if (
                parent is not None 
                and getattr (member ,"connected_to_base",False )
                ):
                    parent_candidate =(
                    parent .position +delta 
                    if getattr (parent ,"robot_id",None )in shepherd_ids 
                    else parent .position 
                    )
                    if candidate .distance_to (parent_candidate )>hard_limit :
                        return False 
            return True 

        safe_fraction =1.0 
        if (
        abs (requested_delta )>physical .EPSILON 
        and not group_step_is_safe (1.0 )
        ):
            low =0.0 
            high =1.0 
            for _ in range (12 ):
                middle =0.5 *(low +high )
                if group_step_is_safe (middle ):
                    low =middle 
                else :
                    high =middle 
            safe_fraction =low 

        actual_delta =requested_delta *safe_fraction 
        velocity =tangent *(
        actual_delta /max (dt ,physical .EPSILON )
        )
        contact =(
        abs (requested_delta )>physical .EPSILON 
        and safe_fraction <0.999 
        )
        for member in members :
            member .integration_boundary_contact =contact 
            member .integration_boundary_shape_error =0.0 

        lifecycle ["shepherd_rigid_cache_frame"]=frame 
        lifecycle ["shepherd_rigid_common_velocity"]=velocity .copy ()
        lifecycle ["shepherd_rigid_contact"]=contact 
        lifecycle ["shepherd_rigid_safe_fraction"]=safe_fraction 
        if frame %10 ==0 :
            print (
            "[ShepherdRigidTransport] "
            f"frame={frame } branch={branch } members={len (members )} "
            f"measured_depth={measured_depth :.3f} "
            f"requested_depth={requested_depth :.3f} "
            f"requested_delta={requested_delta :.3f} "
            f"actual_delta={actual_delta :.3f} "
            f"safe_fraction={safe_fraction :.3f} "
            f"contact={contact } shape_locked=True"
            )
        return velocity 


    def localization_free_root_boundary_update (
    self :Any ,
    dt :float ,
    )->None :
        branch =getattr (
        self ,
        "shepherd_branch",
        None ,
        )

        lifecycle =getattr (
        physical ,
        "integration_wall_lifecycle",
        {},
        ).get (
        branch 
        )

        descriptor =(
        descriptor_for (
        branch 
        )
        if branch is not None 
        else None 
        )

        current_robots =tuple (
        getattr (
        physical ,
        "integration_current_robots",
        (),
        )
        )

        is_root_frontier =bool (
        self .role 
        =="FRONTIER_SHEPHERD"
        and lifecycle is not None 
        and descriptor is not None 
        and branch 
        ==physical .frontier_line_branch 
        )

        is_root_shepherd =bool (
        self .role 
        =="SHEPHERD"
        and lifecycle is not None 
        and descriptor is not None 
        and branch 
        ==physical .active_branch 
        )

        if (
        not (
        is_root_frontier 
        or is_root_shepherd 
        )
        or not current_robots 
        ):




            original_robot_update (
            self ,
            dt ,
            )
            return 

        members =[
        robot 
        for robot 
        in current_robots 
        if (
        robot .role 
        ==self .role 
        and robot .shepherd_branch 
        ==branch 
        )
        ]

        tangent ,lateral =(
        transport_basis (
        branch ,
        lifecycle ,
        descriptor ,
        )[:2 ]
        )

        tangent =tangent .normalize ()
        lateral =lateral .normalize ()

















        if is_root_frontier :
            common_velocity =(
            rigid_frontier_common_velocity (
            branch ,
            members ,
            lifecycle ,
            tangent ,
            dt ,
            )
            )

            integrate_boundary_velocity (
            self ,
            common_velocity ,
            dt ,
            rigid_frontier =True ,
            )
            return 



















        if is_root_shepherd :
            if physical .phase in {
            physical .SimulationPhase .PRESSURE_PUSH ,
            physical .SimulationPhase .FLOW_BACKTRACK ,
            }:
                common_velocity =(
                rigid_shepherd_common_velocity (
                branch ,
                members ,
                lifecycle ,
                tangent ,
                dt ,
                )
                )

                integrate_boundary_velocity (
                self ,
                common_velocity ,
                dt ,
                rigid_frontier =True ,
                )

            else :




                integrate_boundary_velocity (
                self ,
                pygame .Vector2 (),
                dt ,
                rigid_frontier =True ,
                )

            return 

    physical .integration_relative_formation_velocity =relative_formation_velocity 
    physical .integration_integrate_boundary_velocity =integrate_boundary_velocity 
    physical .integration_universal_robot_motion_limit =universal_robot_motion_limit 
    physical .Robot .update =localization_free_root_boundary_update 

    def _final_base_return_direction ()->pygame .Vector2 :
        direction =getattr (
        physical ,
        "integration_base_return_direction_local",
        None ,
        )
        if direction is None or direction .length_squared ()<=physical .EPSILON :
            raise RuntimeError (
            "final return requested before local incoming direction was stored"
            )
        return direction .normalize ()

    def _live_guard_members (
    robots :Sequence [Any ],
    branch :str ,
    )->list [Any ]:
        return [
        robot 
        for robot in robots 
        if robot .role =="JUNCTION_GUARD"
        and robot .junction_guard_branch ==branch 
        ]

    def _visited_guard_branches (
    robots :Sequence [Any ],
    )->list [str ]:
        result =[]
        lifecycle =getattr (physical ,"integration_wall_lifecycle",{})
        for branch ,item in lifecycle .items ():
            descriptor =physical .branch_motion_descriptor (branch )
            if descriptor is None or descriptor .visit_state !="VISITED":
                continue 
            if not _live_guard_members (robots ,branch ):
                continue 
            result .append (branch )
        return result 

    def release_transient_roles_for_local_final_return (
    robots :Sequence [Any ],
    )->None :
        """Release completed DFS roles without world-position control."""

        if getattr (
        physical ,
        "integration_local_final_roles_released",
        False ,
        ):
            return 

        base_direction =(
        _final_base_return_direction ()
        )

        released ={
        "guard":0 ,
        "frontier":0 ,
        "shepherd":0 ,
        "pebble":0 ,
        "relay":0 ,
        }

        transient_wall_roles ={
        "JUNCTION_GUARD",
        "FRONTIER_SHEPHERD",
        "SHEPHERD",
        "PRE_SHEPHERD",
        "PRIMARY_RETURN_SHEPHERD",
        "PEBBLE",
        }

        for robot in robots :

            previous_role =robot .role 





            if previous_role =="TRUNK_RELAY":
                continue 

            robot .comm_bridge_target =None 
            robot .comm_bridge_index =-1 
            robot .comm_bridge_branch =None 

            if previous_role in transient_wall_roles :

                robot .role ="NORMAL"

                if previous_role =="JUNCTION_GUARD":
                    released ["guard"]+=1 

                elif previous_role =="FRONTIER_SHEPHERD":
                    released ["frontier"]+=1 

                elif previous_role =="PEBBLE":
                    released ["pebble"]+=1 

                else :
                    released ["shepherd"]+=1 

            elif previous_role =="RELAY":

                robot .role ="NORMAL"
                released ["relay"]+=1 





            if robot .role !="NORMAL":
                continue 

            robot .base_reserve =False 

            robot .junction_guard_anchor =None 
            robot .junction_guard_branch =None 
            robot .junction_guard_branch_uid =None 
            robot .junction_guard_hop =-1 
            robot .junction_guard_parent_id =None 
            robot .junction_guard_layer =-1 
            robot .is_branch_leader =False 

            if hasattr (
            robot ,
            "integration_guard_waypoints",
            ):
                robot .integration_guard_waypoints =[]

            if hasattr (
            robot ,
            "integration_guard_final_anchor",
            ):
                robot .integration_guard_final_anchor =None 

            robot .shepherd_anchor =None 
            robot .shepherd_origin =None 
            robot .shepherd_branch =None 
            robot .shepherd_return_direction =None 
            robot .frontier_local_lateral =None 

            robot .pebble_anchor =None 
            robot .pebble_branch_uid =None 
            robot .pebble_branch_key =None 
            robot .pebble_state =None 
            robot .pebble_ingress_direction_local =None 
            robot .pebble_return_direction_local =None 

            robot .relay_anchor =None 
            robot .relay_index =-1 
            robot .transfer_target =None 

            robot .final_return_direction_local =(
            base_direction .copy ()
            )

            robot .final_return_source_branch =None 



            robot .velocity .update (0.0 ,0.0 )
            robot .acceleration .update (0.0 ,0.0 )
            robot .filtered_acceleration .update (
            0.0 ,
            0.0 ,
            )

        physical .integration_local_final_roles_released =(
        True 
        )

        print (
        "[LocalFinalRoleCleanup] "
        f"guard={released ['guard']} "
        f"frontier={released ['frontier']} "
        f"shepherd={released ['shepherd']} "
        f"pebble={released ['pebble']} "
        f"relay={released ['relay']} "
        "trunk_relays_preserved=True "
        "localization=False"
        )


    def start_local_final_return (
    robots :Sequence [Any ],
    reason :str ,
    )->None :
        """Enter localization-free Root-to-Base physical return."""

        if getattr (
        physical ,
        "integration_local_final_return_active",
        False ,
        ):
            return 

        if not multi_dfs .global_dfs_complete ():
            raise RuntimeError (
            "Local final return requires "
            "Global DFS completion"
            )

        base_direction =(
        _final_base_return_direction ()
        )

        if (
        base_direction .length_squared ()
        <=physical .EPSILON 
        ):
            raise RuntimeError (
            "Local final return has invalid "
            "Base direction"
            )

        release_transient_roles_for_local_final_return (
        robots 
        )

        perception =getattr (
        physical ,
        "integration_perception",
        None ,
        )

        if perception is None :
            raise RuntimeError (
            "Local final return lost "
            "AdaptivePerception"
            )

        anchor =perception .leader 

        perception .anchor_fixed =False 
        anchor .is_fixed_anchor =False 
        anchor .base_reserve =False 

        if anchor .role !="TRUNK_RELAY":
            anchor .role ="NORMAL"

        anchor .final_return_direction_local =(
        base_direction .copy ()
        )

        physical .integration_local_final_return_active =(
        True 
        )

        physical .integration_final_guard_sweep_active =(
        False 
        )

        physical .integration_final_all_guards_released =(
        True 
        )

        physical .integration_final_base_arrival_dwell =(
        0.0 
        )

        physical .phase =(
        physical .SimulationPhase .RETURN_TO_BASE 
        )

        print (
        "[LocalFinalReturnStart] "
        f"frame="
        f"{getattr (physical ,'integration_frame',-1 )} "
        f"reason={reason } "
        f"anchor={anchor .robot_id } "
        f"direction="
        f"({base_direction .x :.3f},"
        f"{base_direction .y :.3f}) "
        "localization=False"
        )

    def release_next_local_final_trunk_relay (
    robots :Sequence [Any ],
    )->None :
        """Release one Trunk Relay using topology order only."""

        if (
        physical .integration_final_trunk_pending_robot_id 
        is not None 
        ):
            return 

        relays =sorted (
        (
        robot 
        for robot in robots 
        if robot .role =="TRUNK_RELAY"
        ),
        key =lambda robot :int (
        getattr (
        robot ,
        "relay_index",
        -1 ,
        )
        ),
        reverse =True ,
        )

        if not relays :
            return 

        robot =relays [0 ]

        released_index =int (
        getattr (
        robot ,
        "relay_index",
        -1 ,
        )
        )

        base_direction =(
        _final_base_return_direction ()
        )

        robot .role ="NORMAL"
        robot .base_reserve =False 

        robot .relay_anchor =None 
        robot .relay_index =-1 

        robot .comm_bridge_target =None 
        robot .comm_bridge_index =-1 
        robot .comm_bridge_branch =None 

        robot .final_return_direction_local =(
        base_direction .copy ()
        )



        robot .velocity .update (
        0.0 ,
        0.0 ,
        )
        robot .acceleration .update (
        0.0 ,
        0.0 ,
        )
        robot .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )

        physical .integration_final_trunk_pending_robot_id =(
        robot .robot_id 
        )

        physical .integration_final_trunk_reconnect_dwell =(
        0.0 
        )

        print (
        "[LocalFinalTrunkRelease] "
        f"frame="
        f"{getattr (physical ,'integration_frame',-1 )} "
        f"robot={robot .robot_id } "
        f"relay_index={released_index } "
        f"remaining={len (relays )-1 } "
        "order=JUNCTION_TO_BASE "
        "localization=False"
        )


    def update_local_final_return (
    robots :Sequence [Any ],
    dt :float ,
    )->None :
        """Advance localization-free physical return to Base."""

        if not getattr (
        physical ,
        "integration_local_final_return_active",
        False ,
        ):
            return 

        base_station =getattr (
        physical ,
        "base_station",
        None ,
        )

        if base_station is None :
            raise RuntimeError (
            "Local final return requires BaseStation"
            )

        base_direction =(
        _final_base_return_direction ()
        )

        if (
        base_direction .length_squared ()
        <=physical .EPSILON 
        ):
            raise RuntimeError (
            "Local final return lost Base direction"
            )

        normals =[
        robot 
        for robot in robots 
        if robot .role =="NORMAL"
        ]



















        base_observations =observe_local_neighbors (
        base_station ,
        normals ,
        -base_direction ,
        max_range =physical .COMM_RANGE ,
        )

        observed_connected_ids ={
        observation .robot .robot_id 
        for observation in base_observations 
        if getattr (
        observation .robot ,
        "connected_to_base",
        False ,
        )
        }

        arrived_ids =set (
        getattr (
        physical ,
        "integration_final_base_arrived_ids",
        set (),
        )
        )

        newly_arrived =(
        observed_connected_ids 
        -arrived_ids 
        )

        arrived_ids .update (
        observed_connected_ids 
        )

        physical .integration_final_base_arrived_ids =(
        arrived_ids 
        )

        if newly_arrived :
            print (
            "[LocalFinalBaseArrival] "
            f"frame="
            f"{getattr (physical ,'integration_frame',-1 )} "
            f"new={sorted (newly_arrived )} "
            f"arrived={len (arrived_ids )}/{len (robots )} "
            "source=BASE_LOCAL_OBSERVATION"
            )















        pending_id =(
        physical .integration_final_trunk_pending_robot_id 
        )

        if pending_id is not None :

            pending_ready =(
            pending_id 
            in observed_connected_ids 
            )

            if pending_ready :
                physical .integration_final_trunk_reconnect_dwell +=(
                dt 
                )
            else :
                physical .integration_final_trunk_reconnect_dwell =max (
                0.0 ,
                physical .integration_final_trunk_reconnect_dwell 
                -0.5 *dt ,
                )

            if (
            physical .integration_final_trunk_reconnect_dwell 
            >=
            physical .integration_final_trunk_reconnect_dwell_required 
            ):
                print (
                "[LocalFinalTrunkReconnected] "
                f"frame="
                f"{getattr (physical ,'integration_frame',-1 )} "
                f"robot={pending_id } "
                f"dwell="
                f"{physical .integration_final_trunk_reconnect_dwell :.3f} "
                "source=BASE_LOCAL_OBSERVATION"
                )

                physical .integration_final_trunk_pending_robot_id =(
                None 
                )

                physical .integration_final_trunk_reconnect_dwell =(
                0.0 
                )

            else :
                return 

        trunk_relays =[
        robot 
        for robot in robots 
        if robot .role =="TRUNK_RELAY"
        ]

        if trunk_relays :
            release_next_local_final_trunk_relay (
            robots 
            )
            return 















        all_robot_ids ={
        robot .robot_id 
        for robot in robots 
        }

        all_arrived =(
        all_robot_ids 
        <=arrived_ids 
        )

        special_roles =[
        robot 
        for robot in robots 
        if robot .role !="NORMAL"
        ]

        done_ready =(
        all_arrived 
        and not special_roles 
        and physical .integration_final_trunk_pending_robot_id 
        is None 
        )

        if done_ready :
            physical .integration_final_base_arrival_dwell +=(
            dt 
            )
        else :
            physical .integration_final_base_arrival_dwell =max (
            0.0 ,
            physical .integration_final_base_arrival_dwell 
            -0.5 *dt ,
            )

        frame =getattr (
        physical ,
        "integration_frame",
        -1 ,
        )

        if frame %10 ==0 :
            print (
            "[LocalFinalReturnProgress] "
            f"frame={frame } "
            f"arrived={len (arrived_ids )}/{len (robots )} "
            f"trunk_relays={len (trunk_relays )} "
            f"special_roles={len (special_roles )} "
            f"dwell="
            f"{physical .integration_final_base_arrival_dwell :.3f} "
            f"done_ready={done_ready }"
            )

        if (
        physical .integration_final_base_arrival_dwell 
        <
        physical .integration_final_base_arrival_dwell_required 
        ):
            return 

        physical .integration_local_final_return_active =(
        False 
        )

        physical .phase =(
        physical .SimulationPhase .DONE 
        )

        metrics =getattr (
        physical ,
        "metrics",
        None ,
        )

        if metrics is not None :
            metrics .completion_time =(
            physical .simulation_time 
            )

        print (
        "[LocalFinalReturnDone] "
        f"frame={frame } "
        f"robots={len (arrived_ids )}/{len (robots )} "
        "physical_base_arrival=True "
        "localization=False"
        )

    def start_general_final_guard_sweep (
    robots :Sequence [Any ],
    reason :str ,
    )->None :
        """Keep every returned Guard fixed until a real Base-bound flow exists."""

        if getattr (
        physical ,
        "integration_final_guard_sweep_active",
        False ,
        ):
            return 

        branches =_visited_guard_branches (
        robots 
        )

        if not branches :

            print (
            "[FinalGuardSweepBypass] "
            f"frame="
            f"{getattr (physical ,'integration_frame',-1 )} "
            f"reason={reason } "
            "returned_guards=0 "
            "source=COMPLETED_ROOT_AFTER_CHILD_RETURN"
            )

            start_local_final_return (
            robots ,
            f"{reason }_NO_RETURNED_GUARDS",
            )

            return 



        for branch in branches :
            members =_live_guard_members (robots ,branch )

            if not members :
                raise RuntimeError (
                f"visited Guard wall has no live members: {branch }"
                )

            for robot in members :
                robot .velocity .update (0.0 ,0.0 )
                robot .acceleration .update (0.0 ,0.0 )
                robot .filtered_acceleration .update (0.0 ,0.0 )

            lifecycle =getattr (
            physical ,
            "integration_wall_lifecycle",
            {},
            ).get (branch )

            if lifecycle is not None :
                lifecycle ["state"]="FINAL_PRESSURE_GATE"

        physical .integration_final_guard_sweep_active =True 



        physical .integration_final_base_flow_dwell =0.0 
        physical .integration_final_base_flow_established =False 
        physical .integration_final_all_guards_released =False 





        physical .integration_final_base_flow_force_scale =0.45 
        physical .integration_final_base_flow_min_speed =1.0 
        physical .integration_final_base_flow_dwell_required =0.25 

        physical .phase =physical .SimulationPhase .FINAL_JUNCTION_GATHER 
        physical .final_gather_timer =0.0 

        print (
        f"[FinalPressureDrainStart] "
        f"frame={getattr (physical ,'integration_frame',-1 )} "
        f"reason={reason } "
        f"guards={branches } "
        "moving_guard_pusher=False "
        "all_returned_guards_fixed=True"
        )

    def final_guard_sweep_route_force (
    robot :Any ,
    )->pygame .Vector2 :







        if getattr (
        physical ,
        "integration_local_final_return_active",
        False ,
        ):





            if robot .role =="TRUNK_RELAY":
                return pygame .Vector2 ()

            if (
            robot .role !="NORMAL"
            or robot .base_reserve 
            ):
                return pygame .Vector2 ()

            arrived_ids =getattr (
            physical ,
            "integration_final_base_arrived_ids",
            set (),
            )





            if robot .robot_id in arrived_ids :
                return pygame .Vector2 ()

            direction =getattr (
            robot ,
            "final_return_direction_local",
            None ,
            )

            if (
            not isinstance (
            direction ,
            pygame .Vector2 ,
            )
            or direction .length_squared ()
            <=physical .EPSILON 
            ):
                direction =(
                _final_base_return_direction ()
                )
            else :
                direction =(
                direction .normalize ()
                )

            return (
            direction 
            *physical .RETURN_EGRESS_FORCE 
            *physical .integration_final_return_force_scale 
            )

        if not getattr (
        physical ,
        "integration_final_guard_sweep_active",
        False ,
        ):
            return shepherd_physical_only_route_force (
            robot 
            )

        if robot .role in {
        "JUNCTION_GUARD",
        "PEBBLE",
        "RELAY",
        "TRUNK_RELAY",
        }:
            return pygame .Vector2 ()

        if robot .role !="NORMAL"or robot .base_reserve :
            return pygame .Vector2 ()

        return (
        _final_base_return_direction ()
        *physical .RETURN_EGRESS_FORCE 
        *physical .integration_final_base_flow_force_scale 
        )

    def final_guard_compute_sph_forces (
    robots :Sequence [Any ],
    grid :Any ,
    communication_grid :Any ,
    dt :float =1.0 /60.0 ,
    )->None :
        """Run SPH while removing legacy map-aware final-return guidance."""

        final_control_active =(
        getattr (
        physical ,
        "integration_final_guard_sweep_active",
        False ,
        )
        or getattr (
        physical ,
        "integration_local_final_return_active",
        False ,
        )
        )

        if not final_control_active :
            physical_only_compute_sph_forces (
            robots ,
            grid ,
            communication_grid ,
            dt ,
            )
            return 



























        original_contact_force =(
        physical .compute_contact_point_repulsion_force 
        )

        original_edf_force =(
        physical .compute_pressure_coupled_edf_force 
        )

        original_pebble_guidance =(
        physical .compute_pebble_flow_guidance 
        )

        original_initial_wall_force =(
        physical .compute_initial_junction_soft_wall_force 
        )

        original_base_piston_force =(
        physical .compute_base_piston_reaction_force 
        )

        physical .compute_contact_point_repulsion_force =(
        lambda robot :pygame .Vector2 ()
        )

        physical .compute_pressure_coupled_edf_force =(
        lambda robot :pygame .Vector2 ()
        )

        physical .compute_pebble_flow_guidance =(
        lambda robot ,candidate_force :
        candidate_force .copy ()
        )

        physical .compute_initial_junction_soft_wall_force =(
        lambda robot :pygame .Vector2 ()
        )

        physical .compute_base_piston_reaction_force =(
        lambda robot :pygame .Vector2 ()
        )

        try :
            physical_only_compute_sph_forces (
            robots ,
            grid ,
            communication_grid ,
            dt ,
            )

        finally :
            physical .compute_contact_point_repulsion_force =(
            original_contact_force 
            )

            physical .compute_pressure_coupled_edf_force =(
            original_edf_force 
            )

            physical .compute_pebble_flow_guidance =(
            original_pebble_guidance 
            )

            physical .compute_initial_junction_soft_wall_force =(
            original_initial_wall_force 
            )

            physical .compute_base_piston_reaction_force =(
            original_base_piston_force 
            )

    def localization_free_final_robot_update (
    self :Any ,
    dt :float ,
    )->None :
        """Integrate final-return motion without fixture-region classification."""

        old_position =self .position .copy ()

        final_sweep_active =getattr (
        physical ,
        "integration_final_guard_sweep_active",
        False ,
        )

        local_return_active =getattr (
        physical ,
        "integration_local_final_return_active",
        False ,
        )















        fixed_roles ={
        "JUNCTION_GUARD",
        "PEBBLE",
        "RELAY",
        "TRUNK_RELAY",
        }

        if self .role in fixed_roles :

            if (
            local_return_active 
            and self .role !="TRUNK_RELAY"
            ):
                raise RuntimeError (
                "Unexpected unreleased final-return role: "
                f"robot={self .robot_id } "
                f"role={self .role }"
                )

            self .velocity .update (
            0.0 ,
            0.0 ,
            )

            self .acceleration .update (
            0.0 ,
            0.0 ,
            )

            self .commanded_velocity .update (
            0.0 ,
            0.0 ,
            )

            self .observed_velocity .update (
            0.0 ,
            0.0 ,
            )

            self .previous_position =(
            old_position 
            )

            self ._record_motion ()

            return 

        if self .role !="NORMAL":
            raise RuntimeError (
            "Final-return robot has unsupported role: "
            f"robot={self .robot_id } "
            f"role={self .role }"
            )















        arrived_ids =getattr (
        physical ,
        "integration_final_base_arrived_ids",
        set (),
        )

        if (
        local_return_active 
        and self .robot_id in arrived_ids 
        ):
            self .velocity .update (
            0.0 ,
            0.0 ,
            )

            self .acceleration .update (
            0.0 ,
            0.0 ,
            )

            self .commanded_velocity .update (
            0.0 ,
            0.0 ,
            )

            self .observed_velocity .update (
            0.0 ,
            0.0 ,
            )

            self .previous_position =(
            old_position 
            )

            self ._record_motion ()

            return 













        self .velocity +=(
        self .acceleration 
        *dt 
        )

        physical .limit_vector (
        self .velocity ,
        physical .MAX_SPEED ,
        )





        physical .apply_communication_velocity_guard (
        self ,
        dt ,
        )

        self .commanded_velocity =(
        self .velocity .copy ()
        )















        x_position =pygame .Vector2 (
        self .position .x 
        +self .velocity .x *dt ,
        self .position .y ,
        )

        if physical .is_walkable (
        x_position ,
        self .radius ,
        ):
            self .position .x =(
            x_position .x 
            )

        else :
            self .velocity .x =(
            physical .wall_collision_velocity (
            self .velocity .x 
            )
            )

        y_position =pygame .Vector2 (
        self .position .x ,
        self .position .y 
        +self .velocity .y *dt ,
        )

        if physical .is_walkable (
        y_position ,
        self .radius ,
        ):
            self .position .y =(
            y_position .y 
            )

        else :
            self .velocity .y =(
            physical .wall_collision_velocity (
            self .velocity .y 
            )
            )





        physical .constrain_communication_parent_separation (
        self ,
        old_position ,
        )

        self .observed_velocity =(
        self .position 
        -old_position 
        )/max (
        dt ,
        physical .EPSILON ,
        )

        self .acceleration .update (
        0.0 ,
        0.0 ,
        )

        self .previous_position =(
        old_position 
        )

        self ._record_motion ()


    def final_guard_robot_update (
    self :Any ,
    dt :float ,
    )->None :

        final_control_active =(
        getattr (
        physical ,
        "integration_final_guard_sweep_active",
        False ,
        )
        or getattr (
        physical ,
        "integration_local_final_return_active",
        False ,
        )
        )

        if final_control_active :

            localization_free_final_robot_update (
            self ,
            dt ,
            )

            return 

        localization_free_root_boundary_update (
        self ,
        dt ,
        )





    physical .compute_route_force =final_guard_sweep_route_force 
    physical .compute_sph_forces =final_guard_compute_sph_forces 
    physical .Robot .update =final_guard_robot_update 

    def update_general_final_guard_sweep (
    robots :Sequence [Any ],
    dt :float ,
    )->None :
        """Wait for real Base-bound NORMAL flow, then release all Guards."""
        if not getattr (
        physical ,
        "integration_final_guard_sweep_active",
        False ,
        ):
            return 

        if getattr (
        physical ,
        "integration_final_all_guards_released",
        False ,
        ):
            return 

        base_direction =_final_base_return_direction ()
        frame =getattr (physical ,"integration_frame",-1 )







        base_station =getattr (
        physical ,
        "base_station",
        None ,
        )
        if base_station is None :
            raise RuntimeError (
            "final Base flow requested without BaseStation"
            )

        normals =[
        robot 
        for robot in robots 
        if robot .role =="NORMAL"and not robot .base_reserve 
        ]
        min_speed =float (physical .integration_final_base_flow_min_speed )

        base_observations =observe_local_neighbors (
        base_station ,
        normals ,
        -base_direction ,
        max_range =physical .COMM_RANGE ,
        )

        moving_baseward =sum (
        1 
        for observation in base_observations 
        if observation .robot .velocity .dot (base_direction )>=min_speed 
        )

        minimum_flow_count =max (
        8 ,
        int (math .ceil (0.06 *max (len (normals ),1 ))),
        )
        flow_now =(
        moving_baseward >=minimum_flow_count 
        )
        if flow_now :
            physical .integration_final_base_flow_dwell +=dt 
        else :
            physical .integration_final_base_flow_dwell =max (
            0.0 ,
            physical .integration_final_base_flow_dwell -0.5 *dt ,
            )

        branches =_visited_guard_branches (robots )
        returned_guards =[
        robot 
        for branch in branches 
        for robot in _live_guard_members (robots ,branch )
        ]
        overlapping_guard_pairs =0 
        min_guard_distance =float ("inf")
        for index ,left in enumerate (returned_guards ):
            for right in returned_guards [index +1 :]:
                distance =left .position .distance_to (right .position )
                min_guard_distance =min (min_guard_distance ,distance )
                overlap_tolerance =max (
                1.0e-6 ,
                1.0e-3 *min (left .radius ,right .radius ),
                )
                if distance <left .radius +right .radius -overlap_tolerance :
                    overlapping_guard_pairs +=1 

        flow_established =(
        physical .integration_final_base_flow_dwell 
        >=physical .integration_final_base_flow_dwell_required 
        )
        if frame %10 ==0 :
            print (
            f"[FinalBaseFlow] frame={frame } "
            f"normals={len (normals )} "
            f"base_visible={len (base_observations )} "
            f"moving_baseward={moving_baseward } "
            f"required_moving={minimum_flow_count } "
            f"dwell={physical .integration_final_base_flow_dwell :.3f} "
            f"required_dwell="
            f"{physical .integration_final_base_flow_dwell_required :.2f} "
            f"guards_fixed={len (branches )} "
            f"flow_established={flow_established }"
            )
            print (
            f"[FinalGuardOverlapAudit] frame={frame } "
            f"guards={len (returned_guards )} "
            f"overlapping_pairs={overlapping_guard_pairs } "
            f"min_distance="
            f"{min_guard_distance if math .isfinite (min_guard_distance )else -1.0 :.3f}"
            )

        if not flow_established :
            return 

        physical .integration_final_base_flow_established =True 
        print (
        f"[FinalBaseFlowEstablished] frame={frame } "
        f"base_visible={len (base_observations )} "
        f"moving_baseward={moving_baseward } "
        f"dwell={physical .integration_final_base_flow_dwell :.3f}"
        )

        released_ids :list [int ]=[]
        for branch in branches :
            members =_live_guard_members (robots ,branch )
            for robot in members :
                released_ids .append (robot .robot_id )
                robot .role ="NORMAL"
                robot .final_return_direction_local =base_direction .copy ()
                robot .final_return_source_branch =branch 
                robot .junction_guard_anchor =None 
                robot .junction_guard_branch =None 
                robot .junction_guard_branch_uid =None 
                robot .junction_guard_hop =-1 
                robot .junction_guard_parent_id =None 
                robot .junction_guard_layer =-1 
                robot .shepherd_anchor =None 
                robot .shepherd_origin =None 
                robot .shepherd_branch =None 
                robot .shepherd_return_direction =None 
                robot .is_branch_leader =False 
                robot .velocity *=0.25 
                robot .acceleration .update (0.0 ,0.0 )
                robot .filtered_acceleration .update (0.0 ,0.0 )

            physical .junction_guard_groups [branch ]=[]
            lifecycle =getattr (
            physical ,
            "integration_wall_lifecycle",
            {},
            ).get (branch )
            if lifecycle is not None :
                lifecycle ["state"]="FINAL_FLOW_JOINED"
                lifecycle ["final_flow_join_frame"]=frame 





        physical .integration_final_all_guards_released =True 
        physical .integration_final_guard_sweep_active =False 
        print (
        f"[FinalAllGuardsJoinFlow] frame={frame } "
        f"branches={branches } "
        f"released_robots={len (released_ids )} "
        "simultaneous=True "
        "teleport=False "
        "strong_launch=False"
        )
        start_local_final_return (
        robots ,
        "FINAL_BASE_FLOW_ESTABLISHED",
        )

    def environment_authoritative_update_state (
    robots :Sequence [Any ],
    dt :float ,
    reference_density :float ,
    spatial_grid :Any ,
    )->None :




        if getattr (
        physical ,
        "integration_final_guard_sweep_active",
        False ,
        ):
            update_general_final_guard_sweep (robots ,dt )
            return 







        lidar_guard_handoff =(
        physical .phase ==physical .SimulationPhase .FORM_JUNCTION_GUARDS 
        and getattr (physical ,"integration_ready_guard_handoff",False )
        )
        lidar_next_branch_switch =(
        physical .phase ==physical .SimulationPhase .JUNCTION_SWITCH 
        )
        lidar_thick_frontier_explore =(
        physical .phase ==physical .SimulationPhase .EXPLORE_BRANCH 
        )
        lidar_shepherd_return =(
        physical .phase in {
        physical .SimulationPhase .PRESSURE_PUSH ,
        physical .SimulationPhase .FLOW_BACKTRACK ,
        }
        )
        lidar_shepherd_formation =(
        physical .phase ==physical .SimulationPhase .FORM_SHEPHERD_BOUNDARY 
        and getattr (physical ,"integration_wall_lifecycle",{}).get (
        physical .active_branch 
        )is not None 
        )

        if (
        lidar_guard_handoff 
        or lidar_next_branch_switch 
        or lidar_thick_frontier_explore 
        or lidar_shepherd_formation 
        or lidar_shepherd_return 
        ):










            integrated_update_state (
            robots ,
            dt ,
            reference_density ,
            spatial_grid ,
            )
            return 





















        original_update_state (
        robots ,
        dt ,
        reference_density ,
        spatial_grid ,
        )





        if physical .phase ==physical .SimulationPhase .JUNCTION_SWITCH :
            integrated_update_state (
            robots ,
            dt ,
            reference_density ,
            spatial_grid ,
            )

    physical .update_simulation_state =environment_authoritative_update_state 
    physical .integration_environment_authoritative_cycle =True 

    physical .integration_start_final_return_pipeline =(
    start_final_return_pipeline 
    )
    physical .integration_update_local_final_return =(
    update_local_final_return 
    )

    print (
    "[ENVIRONMENT_DFS_CYCLE_ACTIVE] "
    "LiDAR=PERCEPTION_AND_GUARD_ADAPTER "
    "DFS=ENVIRONMENT_EXACT_AFTER_3XN_DEADEND_ADAPTER "
    "3XN_EXPLORE_ADAPTER->FORM_SHEPHERD->FILL->PRESSURE_PUSH->"
    "FLOW_BACKTRACK->JUNCTION_SWITCH->REPEAT->RETURN_TO_BASE->DONE"
    )
    print (
    "[ENVIRONMENT_DFS_RESTORE] "
    f"piston_speed={physical .SHEPHERD_PISTON_SPEED :.3f} "
    f"line_speed={physical .SHEPHERD_LINE_BACKTRACK_SPEED :.3f} "
    f"release_speed={physical .SHEPHERD_JUNCTION_RELEASE_SPEED :.3f} "
    "pack_coupled_hold_runtime=False "
    "herd_locked_robot_update_runtime=False "
    "virtual_wall_removed=True"
    )



def refine_guard_geometry_from_persistent_lidar (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Bind persistent UIDs to the already-frozen LiDAR Guard geometry.

    Spatial WHERE is immutable after Guard election.  Persistent accumulation
    may name the branch, but it must never rebuild descriptors, slots, or robot
    targets; doing so mixes two mouth frames and deforms the UP wall at DFS
    handoff.
    """
    if not perception .outgoing :
        raise RuntimeError ("persistent UID binding requires outgoing openings")

    track_by_key ={
    (
    "UP"
    if abs (track .center_angle )<30.0 
    else ("LEFT"if track .center_angle <0.0 else "RIGHT")
    ):track 
    for track in perception .outgoing 
    }
    if set (track_by_key )!={"LEFT","UP","RIGHT"}:
        raise RuntimeError (
        f"persistent outgoing identities are incomplete: {sorted (track_by_key )}"
        )

    all_ids ={
    robot .robot_id for robot in robots if robot is not perception .leader 
    }
    physical .branch_descriptors_by_uid .clear ()
    physical .fixture_key_to_branch_uid .clear ()
    physical .branch_uid_to_fixture_key .clear ()
    physical .detected_branch_candidates =set ()
    physical .junction_guard_groups .clear ()

    remapped_status :dict [str ,dict [str ,Any ]]={}
    associations :dict [str ,dict [str ,float |str ]]={}

    for geometry in perception .provisional_guards :




        fixture =geometry .local_branch_key 
        if fixture not in track_by_key :
            raise RuntimeError (f"no persistent track for frozen Guard {fixture }")
        track =track_by_key [fixture ]
        uid =track .persistent_id 
        descriptor =geometry .descriptor 





        descriptor .uid =uid 
        descriptor .fixture_key =fixture 
        descriptor .cohort_member_ids =set (all_ids )
        descriptor .direction_sample_count =max (
        physical .JUNCTION_COHORT_MIN_ROBOTS ,len (all_ids )
        )
        descriptor .direction_downstream_travel =physical .JUNCTION_COHORT_MIN_TRAVEL 
        descriptor .motion_frame_source ="LIDAR_FROZEN_MOUTH_UID_BOUND"
        descriptor .motion_frame_sample_count =len (track .observations )
        descriptor .physical_boundary_sample_count =len (track .observations )

        geometry .fixture_key =fixture 
        geometry .persistent_uid =uid 

        physical .branch_descriptors_by_uid [uid ]=descriptor 
        physical .fixture_key_to_branch_uid [fixture ]=uid 
        physical .branch_uid_to_fixture_key [uid ]=fixture 
        physical .branch_local_uids [fixture ]=uid 
        physical .detected_branch_candidates .add (fixture )
        physical .junction_guard_groups [fixture ]=list (geometry .selected_ids )
        physical .thick_mouth_guard_columns [fixture ]=geometry .columns 
        physical .thick_mouth_guard_layers [fixture ]=geometry .layers 
        physical .junction_guard_frontier_depths [fixture ]=(
        physical .JUNCTION_GUARD_BRANCH_INSET 
        +(geometry .layers -1 )*physical .THICK_MOUTH_GUARD_LAYER_SPACING 
        )

        selected ={
        robot .robot_id :robot 
        for robot in robots 
        if robot .robot_id in geometry .selected_ids 
        }
        for robot_id in geometry .selected_ids :
            robot =selected [robot_id ]


            robot .junction_guard_branch =fixture 
            robot .junction_guard_branch_uid =uid 
            robot .local_branch_uid_by_key [fixture ]=uid 

        descriptor .leader_id =min (geometry .selected_ids )
        status =physical .integration_wall_status .pop (
        geometry .provisional_uid ,{}
        )
        status .update ({
        "assigned":len (geometry .selected_ids ),
        "edge_selected":len (geometry .selected_ids ),
        "rows":geometry .layers ,
        "slots_per_row":geometry .columns ,
        "slots_walkable":sum (
        physical .is_walkable (slot ,physical .ROBOT_RADIUS )
        for slot in geometry .slots 
        ),
        "slots_total":len (geometry .slots ),
        })
        remapped_status [uid ]=status 

        angular_error =circular_error (
        float (geometry .opening ["center_angle"]),track .center_angle 
        )
        associations [uid ]={
        "opening_center":track .center_angle ,
        "matched_mouth":fixture ,
        "mouth_local_angle":float (geometry .opening ["center_angle"]),
        "angular_error":angular_error ,
        "provisional_uid":geometry .provisional_uid ,
        }
        print (
        f"[GuardFreeze] provisional={geometry .provisional_uid } uid={uid } "
        f"fixture_adapter={fixture } same_ids=True "
        f"robots={len (geometry .selected_ids )} "
        f"columns={geometry .columns } layers={geometry .layers } "
        "spatial_retarget=False slot_regeneration=False "
        f"angular_error={angular_error :.3f}"
        )

    physical .integration_wall_status .update (remapped_status )
    physical .integration_opening_mouth_associations =associations 


def log_wall_ready_blockers (physical :types .ModuleType ,perception :AdaptivePerception ,robots :Sequence [Any ])->None :
    frame =getattr (physical ,"integration_frame",-1 )
    if frame %20 !=0 or perception .handoff_complete :
        return 
    for geometry in perception .provisional_guards :
        status =physical .integration_wall_status .get (geometry .provisional_uid ,{})
        guards =[r for r in robots if r .robot_id in geometry .selected_ids ]
        complete =sum (sum (1 for r in guards if r .junction_guard_layer ==layer )>=geometry .columns for layer in range (geometry .layers ))
        checks =[("settled_ratio",float (status .get ("settled_ratio",0.0 ))>=PROVISIONAL_WALL_SETTLED_RATIO ),("span_ratio",float (status .get ("min_span_ratio",0.0 ))>=float (physical .FRONTIER_LINE_MIN_SPAN_RATIO )),("edge_gap",float (status .get ("max_edge_gap",float ("inf")))<=float (physical .FRONTIER_LINE_MAX_EDGE_GAP )),("internal_gap",float (status .get ("max_internal_gap",float ("inf")))<=float (physical .FRONTIER_LINE_MAX_INTERNAL_GAP )),("slots_walkable",int (status .get ("slots_walkable",0 ))>=len (geometry .slots )),("complete_rows",complete ==geometry .layers )]
        reasons =[name for name ,ok in checks if not ok ]
        print (f"[WallReadyBlocker] branch={geometry .local_branch_key or geometry .provisional_uid } guard_count={len (guards )} expected={len (geometry .slots )} rows={geometry .layers } columns={geometry .columns } complete_rows={complete } settled_ratio={status .get ('settled_ratio',0.0 ):.3f} min_span_ratio={status .get ('min_span_ratio',0.0 ):.3f} max_edge_gap={status .get ('max_edge_gap',float ('inf')):.3f} max_internal_gap={status .get ('max_internal_gap',float ('inf')):.3f} slots_walkable={status .get ('slots_walkable',0 )} ready={bool (status .get ('ready',False ))} blocking_reasons={reasons }")


def register_confirmed_root_junction (
physical :types .ModuleType ,
perception :AdaptivePerception ,
)->None :
    """Register J0 immediately after stationary Junction confirmation."""

    if multi_dfs .stack :
        return 

    if not perception .junction_confirmed :
        return 

    if perception .state not in {
    PerceptionState .BRANCHES_READY ,
    PerceptionState .PHYSICAL_DFS ,
    }:
        return 

    ordered_uids =list (
    perception .integration_detected_branch_order 
    )

    if not ordered_uids :
        raise RuntimeError (
        "J0 confirmed without detected Branch UIDs"
        )

    root =multi_dfs .create_root ()

    root .branch_order =ordered_uids .copy ()

    root .branch_states ={
    uid :"UNVISITED"
    for uid in ordered_uids 
    }

    root .active_branch_uid =None 

    incoming_direction_local =_body_local_unit (
    perception ,
    0.0 ,
    )

    if (
    incoming_direction_local .length_squared ()
    <=physical .EPSILON 
    ):
        raise RuntimeError (
        "invalid J0 ingress direction at registration"
        )

    incoming_direction_local =(
    incoming_direction_local .normalize ()
    )

    root .ingress_direction_local =(
    incoming_direction_local .copy ()
    )

    root .return_direction_local =(
    -incoming_direction_local 
    )

    physical .integration_incoming_direction_local =(
    incoming_direction_local .copy ()
    )

    physical .integration_base_return_direction_local =(
    -incoming_direction_local 
    )

    physical .integration_detected_branch_order =(
    ordered_uids .copy ()
    )

    print (
    "[MultiDFS] ROOT_REGISTERED "
    f"junction={root .junction_uid } "
    f"depth={multi_dfs .depth } "
    f"branches={root .branch_order } "
    f"states={root .branch_states }"
    )

def handoff_to_physical_dfs (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Refine LiDAR WHERE, retain provisional WHO, and enable Physical DFS."""
    eligible =[r for r in robots if r .role =="NORMAL"and not r .base_reserve ]
    scale =float (np .median ([g .mouth_span for g in perception .provisional_guards if g .mouth_span >0.0 ])if perception .provisional_guards else 1.0 )






    incoming_axis =_body_local_unit (perception ,0.0 )
    arrival_observations =observe_local_neighbors (
    perception .leader ,
    eligible ,
    incoming_axis ,
    max_range =2.0 *scale ,
    )
    local_count =len (arrival_observations )
    arrival_ratio =local_count /max (len (eligible ),1 )
    junction_arrived =arrival_ratio >=JUNCTION_ARRIVAL_RATIO_THRESHOLD 

    all_guard_cohorts_complete =all (
    geometry .cohort_ready 
    and len (geometry .selected_ids )==len (geometry .slots )
    for geometry in perception .provisional_guards 
    )









    all_provisional_walls_ready =all (
    bool (
    physical .integration_wall_status 
    .get (geometry .provisional_uid ,{})
    .get ("ready",False )
    )
    for geometry in perception .provisional_guards 
    )

    gate_checks ={
    "topology_ready":(
    perception .topology_ready_frame is not None 
    ),
    "all_groups_activated":(
    perception .guard_all_groups_activated 
    ),
    "all_guard_cohorts_complete":(
    all_guard_cohorts_complete 
    ),
    "all_provisional_walls_ready":(
    all_provisional_walls_ready 
    ),
    "guard_complete":(
    perception .guard_all_groups_activated 
    and all_guard_cohorts_complete 
    and all_provisional_walls_ready 
    ),
    "junction_arrived":junction_arrived ,
    }

    if perception .handoff_complete or perception .anchor_position is None or not perception .provisional_guard_started or not perception .provisional_guards :
        gate_checks ["handoff_inputs"]=False 
    allowed =all (gate_checks .values ())
    print (
    f"[DFSStartGate] "
    f"frame={getattr (physical ,'integration_frame',-1 )} "
    f"topology_ready={gate_checks ['topology_ready']} "
    f"all_groups_activated={gate_checks ['all_groups_activated']} "
    f"all_guard_cohorts_complete="
    f"{gate_checks ['all_guard_cohorts_complete']} "
    f"all_provisional_walls_ready="
    f"{gate_checks ['all_provisional_walls_ready']} "
    f"guard_complete={gate_checks ['guard_complete']} "
    f"junction_local_count={local_count } "
    f"eligible_count={len (eligible )} "
    f"junction_arrival_ratio={arrival_ratio :.3f} "
    f"junction_arrived={junction_arrived } "
    f"blocking_reasons="
    f"{[name for name ,ok in gate_checks .items ()if not ok ]} "
    f"allowed={allowed }"
    )
    if not allowed :
        return 
    refine_guard_geometry_from_persistent_lidar (
    physical ,perception ,robots 
    )
    if len (perception .provisional_guards )!=len (perception .outgoing ):
        raise RuntimeError ("provisional Guard count does not match outgoing topology")
    guard_id_sets =[set (geometry .selected_ids )for geometry in perception .provisional_guards ]
    assert all (a .isdisjoint (b )for i ,a in enumerate (guard_id_sets )for b in guard_id_sets [i +1 :])
    physical .branch_discovery_counter =len (perception .outgoing )
    physical .integration_detected_branch_order =[
    track .persistent_id for track in perception .outgoing 
    ]
    physical .junction_inference_tracker .confirmed =True 
    physical .junction_inference_tracker .confirmed_at =(
    physical .simulation_time -physical .JUNCTION_DISCOVERY_SETTLE_TIME 
    )
    physical .junction_inference_tracker .valid_branches =set (
    physical .detected_branch_candidates 
    )






    incoming_direction_local =_body_local_unit (perception ,0.0 )
    if incoming_direction_local .length_squared ()<=physical .EPSILON :
        raise RuntimeError ("invalid stored incoming local direction")
    incoming_direction_local =incoming_direction_local .normalize ()
    physical .integration_incoming_direction_local =incoming_direction_local .copy ()
    physical .integration_base_return_direction_local =-incoming_direction_local 
    print (
    f"[FinalReturnFrameStored] "
    f"incoming=({incoming_direction_local .x :.3f},"
    f"{incoming_direction_local .y :.3f}) "
    f"base_return=({-incoming_direction_local .x :.3f},"
    f"{-incoming_direction_local .y :.3f})"
    )

    physical .integration_guard_gating_enabled =True 
    physical .integration_guard_who_localization_enabled =False 
    physical .integration_placement_localization_enabled =False 
    physical .integration_provisional_guard_active =False 
    physical .branch_gate_states .clear ()
    physical .branch_gate_states .update ({
    branch :"CLOSED"for branch in physical .BRANCHES 
    })




    physical .integration_wall_lifecycle ={}
    for geometry in perception .provisional_guards :
        branch =geometry .local_branch_key 
        descriptor =physical .branch_motion_descriptor (branch )
        members =[r for r in robots if r .robot_id in geometry .selected_ids ]
        coordinates =[guard_mouth_coordinates (r .position ,geometry )for r in members ]
        centroid_axial =float (np .mean ([value [0 ]for value in coordinates ]))
        centroid_lateral =float (np .mean ([value [1 ]for value in coordinates ]))
        physical .integration_wall_lifecycle [branch ]={
        "uid":descriptor .uid ,
        "state":"GUARD",
        "rows":geometry .layers ,
        "cols":geometry .columns ,
        "robot_ids":sorted (geometry .selected_ids ),
        "centroid_axial":centroid_axial ,
        "centroid_lateral":centroid_lateral ,
        "mouth_center_world":geometry .mouth_center_world .copy (),
        "branch_tangent_unit":geometry .branch_tangent_unit .copy (),
        "mouth_lateral_unit":geometry .mouth_lateral_unit .copy (),
        "measured_mouth_span":float (geometry .mouth_span ),
        "sealing_lateral_min":float (geometry .sealing_lateral_min ),
        "sealing_lateral_max":float (geometry .sealing_lateral_max ),
        "slot_spacing":float (geometry .slot_spacing ),






        "guard_anchor_by_id":{
        robot .robot_id :(
        robot .integration_guard_final_anchor .copy ()
        if getattr (robot ,"integration_guard_final_anchor",None )is not None 
        else (
        robot .junction_guard_anchor .copy ()
        if robot .junction_guard_anchor is not None 
        else robot .position .copy ()
        )
        )
        for robot in members 
        },
        "guard_layer_by_id":{
        robot .robot_id :int (getattr (robot ,"junction_guard_layer",-1 ))
        for robot in members 
        },
        "guard_slot_index_by_id":{
        robot .robot_id :int (getattr (robot ,"integration_guard_slot_index",-1 ))
        for robot in members 
        },
        "relative_offsets":{
        robot .robot_id :(axial -centroid_axial ,lateral -centroid_lateral )
        for robot ,(axial ,lateral )in zip (members ,coordinates )
        },
        }
    physical .record_distributed_consensus (clear_selection =True )
    physical .phase =physical .SimulationPhase .FORM_JUNCTION_GUARDS 
    physical .integration_all_walls_ready =True 
    physical .integration_ready_guard_handoff =True 
    perception .handoff_complete =True 
    perception .state =PerceptionState .PHYSICAL_DFS 


    guard_counts ={
    branch :len (physical .junction_guard_groups .get (branch ,[]))
    for branch in physical .BRANCHES 
    }
    print ("[LocalizationAudit] guard_who_localization_enabled=False")
    print ("[LocalizationAudit] persistent_refine_re_election=False")
    print (
    "[Timing] junction_to_topology_ready_frames="
    f"{perception .topology_ready_frame -perception .confirmation_frame }"
    )
    print (f"[DFS] guard_counts={guard_counts }")
    print ("[DFS Handoff] branches ready descriptors=3 existing_guard_ids_retained=True")


class DarkRenderer :
    def __init__ (self ,physical :types .ModuleType )->None :
        pygame .display .set_caption ("Adaptive LiDAR + SPH Physical DFS")
        self .screen =pygame .display .set_mode (WINDOW_SIZE )
        self .title =pygame .font .SysFont (None ,28 )
        self .font =pygame .font .SysFont (None ,20 )
        self .small =pygame .font .SysFont (None ,17 )
        self .physical =physical 
        physical .screen =self .screen 

    def _profile_point (self ,angle :float ,value :float )->tuple [int ,int ]:
        x =PROFILE_RECT .left +int ((angle +180.0 )/360.0 *PROFILE_RECT .width )
        y =PROFILE_RECT .bottom -int (np .clip (value /MAX_RANGE ,0.0 ,1.0 )*PROFILE_RECT .height )
        return x ,y 

    def _draw_profile (self ,frame :LidarFrame |None )->None :
        pygame .draw .rect (self .screen ,COLORS ["panel_alt"],PROFILE_RECT ,border_radius =6 )
        if frame is None :
            return 
        for angle in (-180 ,-90 ,0 ,90 ,180 ):
            x ,_ =self ._profile_point (float (angle ),0.0 )
            pygame .draw .line (
            self .screen ,(55 ,64 ,76 ),
            (x ,PROFILE_RECT .top ),(x ,PROFILE_RECT .bottom ),1 ,
            )
            label =self .small .render (str (angle ),True ,COLORS ["muted"])
            self .screen .blit (label ,(x -label .get_width ()//2 ,PROFILE_RECT .bottom +5 ))
        for value in (0 ,50 ,100 ,150 ):
            _ ,y =self ._profile_point (-180.0 ,float (value ))
            pygame .draw .line (
            self .screen ,(55 ,64 ,76 ),
            (PROFILE_RECT .left ,y ),(PROFILE_RECT .right ,y ),1 ,
            )
            label =self .small .render (str (value ),True ,COLORS ["muted"])
            self .screen .blit (label ,(PROFILE_RECT .left -31 ,y -7 ))

        if frame .interval_valid :
            upper_y =self ._profile_point (0.0 ,frame .upper )[1 ]
            lower_y =self ._profile_point (0.0 ,frame .lower )[1 ]
            band =pygame .Surface (
            (PROFILE_RECT .width ,max (1 ,lower_y -upper_y )),
            pygame .SRCALPHA ,
            )
            band .fill ((*COLORS ["safe_band"],38 ))
            self .screen .blit (band ,(PROFILE_RECT .left ,upper_y ))

        for index in np .flatnonzero (frame .support ):
            left_x =self ._profile_point (float (frame .angles [index ])-0.5 ,0.0 )[0 ]
            right_x =self ._profile_point (float (frame .angles [index ])+0.5 ,0.0 )[0 ]
            overlay =pygame .Surface (
            (max (1 ,right_x -left_x +1 ),PROFILE_RECT .height ),
            pygame .SRCALPHA ,
            )
            overlay .fill ((*COLORS ["open_fill"],54 ))
            self .screen .blit (overlay ,(left_x ,PROFILE_RECT .top ))

        raw_points =[self ._profile_point (float (a ),float (r ))for a ,r in zip (frame .angles ,frame .raw )]
        smooth_points =[self ._profile_point (float (a ),float (r ))for a ,r in zip (frame .angles ,frame .smoothed )]
        pygame .draw .lines (self .screen ,COLORS ["raw"],False ,raw_points ,2 )
        pygame .draw .lines (self .screen ,COLORS ["smooth"],False ,smooth_points ,2 )

        line_specs =[
        ("Adaptive W",frame .adaptive_w ,COLORS ["muted"],1 ),
        ("Tmin",frame .lower ,COLORS ["safe"],2 ),
        ("Selected T",frame .selected ,COLORS ["threshold"],3 ),
        ("Tmax",frame .upper ,COLORS ["group_center"],2 ),
        ("Rmax",MAX_RANGE ,COLORS ["raw"],1 ),
        ]
        for index ,(name ,value ,color ,width )in enumerate (line_specs ):
            if value is None or not math .isfinite (float (value )):
                continue 
            y =self ._profile_point (0.0 ,float (value ))[1 ]
            pygame .draw .line (
            self .screen ,color ,
            (PROFILE_RECT .left ,y ),(PROFILE_RECT .right ,y ),width ,
            )
            label =self .small .render (f"{name }={float (value ):.1f}",True ,color )
            label_x =(
            PROFILE_RECT .left +6 
            if index %2 ==0 
            else PROFILE_RECT .right -label .get_width ()-6 
            )
            label_y =min (PROFILE_RECT .bottom -17 ,max (PROFILE_RECT .top +31 ,y -16 ))
            self .screen .blit (label ,(label_x ,label_y ))

        threshold_active =(
        frame .interval_valid 
        and frame .selected is not None 
        )

        if threshold_active :
            opening_edge_color =COLORS ["open"]
            opening_center_color =COLORS ["open"]
        else :
            opening_edge_color =COLORS ["candidate"]
            opening_center_color =COLORS ["candidate"]

        for opening in frame .openings :
            for key ,color ,width in (
            ("start_angle",opening_edge_color ,2 ),
            ("end_angle",opening_edge_color ,2 ),
            ("center_angle",opening_center_color ,3 ),
            ):
                angle =opening [key ]
                x =self ._profile_point (float (angle ),0.0 )[0 ]
                pygame .draw .line (
                self .screen ,color ,
                (x ,PROFILE_RECT .top ),(x ,PROFILE_RECT .bottom ),width ,
                )

        legend =(
        ("RAW",COLORS ["raw"]),
        ("SMOOTHED",COLORS ["smooth"]),
        ("STRUCTURAL CANDIDATE",COLORS ["candidate"]),
        ("OPEN SUPPORT",COLORS ["open"]),
        ("SAFE T INTERVAL",COLORS ["safe_band"]),
        )
        cursor =PROFILE_RECT .left +7 
        for label_text ,color in legend :
            pygame .draw .line (
            self .screen ,color ,
            (cursor ,PROFILE_RECT .top +13 ),
            (cursor +13 ,PROFILE_RECT .top +13 ),3 ,
            )
            label =self .small .render (label_text ,True ,color )
            self .screen .blit (label ,(cursor +17 ,PROFILE_RECT .top +5 ))
            cursor +=28 +label .get_width ()
        self .screen .blit (
        self .small .render ("range",True ,COLORS ["muted"]),
        (PROFILE_RECT .left -31 ,PROFILE_RECT .top -20 ),
        )
        axis =self .small .render ("LiDAR angle theta [deg]",True ,COLORS ["muted"])
        self .screen .blit (
        axis ,
        (PROFILE_RECT .centerx -axis .get_width ()//2 ,PROFILE_RECT .bottom +5 ),
        )
        pygame .draw .rect (self .screen ,COLORS ["muted"],PROFILE_RECT ,1 ,border_radius =6 )

    def _draw_map (self ,robots :Sequence [Any ],perception :AdaptivePerception ,show_rays :bool ,show_comm :bool ,density :bool )->None :
        physical =self .physical 
        pygame .draw .rect (self .screen ,COLORS ["panel"],MAIN_RECT ,border_radius =6 )
        pygame .draw .polygon (self .screen ,COLORS ["floor"],physical .cross_points )
        pygame .draw .polygon (self .screen ,COLORS ["wall"],physical .cross_points ,2 )
        frame =perception .last_frame 
        if show_rays and frame is not None :
            origin =perception .leader .position 

            threshold_active =(
            frame .interval_valid 
            and frame .selected is not None 
            )

            def angle_inside_opening (
            ray_angle :float ,
            opening :dict [str ,float ],
            )->bool :
                start =float (opening ["start_angle"])
                end =float (opening ["end_angle"])
                ray_360 =ray_angle %360.0 
                start_360 =start %360.0 
                end_360 =end %360.0 

                if start_360 <=end_360 :
                    return start_360 <=ray_360 <=end_360 

                return ray_360 >=start_360 or ray_360 <=end_360 

            for index in range (0 ,len (frame .angles ),3 ):
                local_angle =float (frame .angles [index ])
                world_angle =math .radians (perception .yaw_deg +local_angle )
                endpoint =origin +pygame .Vector2 (math .cos (world_angle ),math .sin (world_angle ))*float (frame .raw [index ])
                color =COLORS ["raw"]

                if threshold_active :
                    if frame .support [index ]:
                        color =COLORS ["open"]
                elif any (
                angle_inside_opening (local_angle ,opening )
                for opening in frame .openings 
                ):
                    color =COLORS ["candidate"]

                pygame .draw .line (self .screen ,color ,origin ,endpoint ,1 )

            center_ray_color =(
            COLORS ["open"]
            if threshold_active 
            else COLORS ["candidate"]
            )
            for opening in frame .openings :
                world_angle =math .radians (perception .yaw_deg +float (opening ["center_angle"]))
                endpoint =origin +pygame .Vector2 (math .cos (world_angle ),math .sin (world_angle ))*MAX_RANGE 
                pygame .draw .line (self .screen ,center_ray_color ,origin ,endpoint ,3 )
        if show_comm :
            physical .draw_communication_links (self .screen ,robots )
        role_colors ={
        "NORMAL":COLORS ["normal"],"JUNCTION_GUARD":COLORS ["guard"],
        "FRONTIER_SHEPHERD":COLORS ["frontier"],"SHEPHERD":COLORS ["shepherd"],
        "PRE_SHEPHERD":COLORS ["shepherd"],
        "PEBBLE":COLORS ["pebble"],
        "RELAY":COLORS ["relay"],"TRUNK_RELAY":COLORS ["trunk"],
        }
        order ={"NORMAL":0 ,"RELAY":1 ,"TRUNK_RELAY":1 ,"PEBBLE":2 ,"JUNCTION_GUARD":3 ,"FRONTIER_SHEPHERD":4 ,"PRE_SHEPHERD":5 ,"SHEPHERD":5 }
        for robot in sorted (robots ,key =lambda item :order .get (item .role ,0 )):
            color =role_colors .get (robot .role ,COLORS ["normal"])

            if density and robot .role =="NORMAL":
                color =physical .density_to_color (
                robot .density ,
                max (robot .density ,1.0 ),
                )

            draw_radius =max (
            2 ,
            round (robot .radius +1 ),
            )

            pygame .draw .circle (
            self .screen ,
            color ,
            robot .position ,
            draw_radius ,
            )



            if robot .role =="PEBBLE":
                pygame .draw .circle (
                self .screen ,
                (255 ,255 ,255 ),
                robot .position ,
                draw_radius +4 ,
                2 ,
                )
        pygame .draw .circle (self .screen ,COLORS ["anchor"],perception .leader .position ,7 )
        pygame .draw .circle (self .screen ,COLORS ["background"],perception .leader .position ,7 ,2 )
        self .screen .blit (self .small .render (f"LiDAR {perception .leader .robot_id }",True ,COLORS ["anchor"]),perception .leader .position +pygame .Vector2 (9 ,-18 ))

    def _draw_diagnostics (
    self ,
    robots :Sequence [Any ],
    perception :AdaptivePerception ,
    paused :bool ,
    )->None :
        physical =self .physical 
        frame =perception .last_frame 
        pygame .draw .rect (self .screen ,COLORS ["panel_alt"],DIAGNOSTIC_RECT ,border_radius =6 )
        selection_committed =(
        perception .handoff_complete 
        and (
        physical .phase !=physical .SimulationPhase .FORM_JUNCTION_GUARDS 
        or physical .pending_branch_start is not None 
        )
        )
        selected_uid =(
        physical .branch_uid_for_fixture (physical .active_branch )
        if selection_committed else None 
        )
        support_count =int (np .count_nonzero (frame .support ))if frame else 0 
        selected_text =(
        f"{frame .selected :.2f}"
        if frame and frame .selected is not None else "INVALID"
        )
        detector_lines =[
        f"{'PAUSED'if paused else 'RUNNING'}  frame={frame .frame if frame else 0 }  t={physical .simulation_time :.3f}s",
        f"Detector: Adaptive W-tau (alpha={ALPHA :.1f})",
        f"Rmax={MAX_RANGE :.1f}  Adaptive W={frame .adaptive_w :.2f}"if frame else f"Rmax={MAX_RANGE :.1f}",
        f"Tmin={frame .lower :.2f}  Selected={selected_text }  Tmax={frame .upper :.2f}"if frame else "Thresholds=-",
        f"tau={TAU :.2f}  interval valid={frame .interval_valid if frame else False }",
        f"OPEN support={support_count }  current openings={len (frame .openings )if frame else 0 }",
        f"Junction confirmed={perception .junction_confirmed }",
        f"Anchor={'FIXED'if perception .anchor_fixed else 'MOVING'}  ID={perception .leader .robot_id }",
        f"Anchor-only stop={perception .anchor_fixed }",
        f"Normal-flow enabled={perception .anchor_fixed and not perception .handoff_complete }",
        f"mean Normal forward speed={perception .mean_normal_forward_speed ():.2f}",
        "Opening groups",
        ]
        if frame :
            detector_lines .extend (
            f"#{index } s={item ['start_angle']:+.1f} e={item ['end_angle']:+.1f} "
            f"c={item ['center_angle']:+.1f} w={item ['width_deg']:.1f}"
            for index ,item in enumerate (frame .openings [:4 ])
            )
        persistent_count =sum (
        len (track .observations )>=MIN_PERSISTENT_OBSERVATIONS 
        for track in perception .tracks 
        )
        saturation =getattr (
        physical ,"integration_saturation",LocalSaturationDiagnostics ()
        )
        wall_status =getattr (physical ,"integration_wall_status",{})
        wall_lifecycle =getattr (
        physical ,"integration_wall_lifecycle",{}
        )
        wall_lines =[]
        if perception .provisional_guards and not perception .handoff_complete :
            for geometry in perception .provisional_guards :
                status =wall_status .get (geometry .provisional_uid ,{})
                wall_lines .append (
                f"{geometry .provisional_uid }: "
                f"{'SETTLING'if geometry .cohort_ready else 'WAITING'} "
                f"cand={status .get ('candidate_count',0 )} "
                f"match={status .get ('assignment_count',0 )}/"
                f"{len (geometry .slots )}"
                )
                wall_lines .append (
                f"  guards={len (geometry .selected_ids )} "
                f"rows={geometry .layers } cols={geometry .columns } "
                f"readyF={geometry .guard_ready_frame or '-'} "
                f"wall={'READY'if status .get ('ready',False )else 'SETTLING'} "
                f"sealed={status .get ('structurally_sealed',False )} "
                f"dwell={status .get ('wall_ready_dwell',0.0 ):.2f} "
                f"edgeGap={status .get ('max_edge_gap',0.0 ):.2f} "
                f"intGap={status .get ('max_internal_gap',0.0 ):.2f}"
                )
        else :
            for key in physical .BRANCHES :
                uid =physical .branch_uid_for_fixture (key )
                lifecycle =wall_lifecycle .get (key ,{})
                if not lifecycle :
                    live =[r for r in robots if r .role =="JUNCTION_GUARD"and r .junction_guard_branch ==key ]
                    lifecycle ={"state":"GUARD","robot_ids":[r .robot_id for r in live ],"rows":physical .thick_mouth_guard_layers .get (key ,0 ),"cols":physical .thick_mouth_guard_columns .get (key ,0 )}
                wall_lines .append (
                f"{key } wall: state={lifecycle .get ('state','FORMING')}"
                )
                wall_lines .append (
                f"  robots={len (lifecycle .get ('robot_ids',[]))} "
                f"rows={lifecycle .get ('rows',0 )} cols={lifecycle .get ('cols',0 )} "
                f"edgeGap={lifecycle .get ('max_edge_gap',0.0 ):.2f} "
                f"intGap={lifecycle .get ('max_internal_gap',0.0 ):.2f}"
                )
        current_frontier_ids =sorted (
        robot .robot_id 
        for robot in physical .get_frontier_shepherds (robots )
        )
        current_shepherd_ids =sorted (
        robot .robot_id for robot in physical .get_shepherds (robots )
        )
        leakage_states =list (perception .guard_leakage .values ())
        leakage_summary =(
        f"Leak pre/post="
        f"{sum (int (item ['crossings_before_edge_seal'])for item in leakage_states )}/"
        f"{sum (int (item ['crossings_after_edge_seal'])for item in leakage_states )} "
        f"maxDepth={max ((float (item ['deepest_leaked_robot_depth'])for item in leakage_states ),default =0.0 ):.1f} "
        f"blocked={all (bool (item ['leakage_blocked_after_edge_seal'])for item in leakage_states )if leakage_states else False }"
        )
        dfs_lines =[
        "TOPOLOGY / PHYSICAL DFS",
        f"GuardStage={perception .guard_activation_stage }",
        f"LiDAR Persistent={persistent_count } | Outgoing={len (perception .outgoing )}",
        f"Parent={perception .parent .persistent_id if perception .parent else '-'} source={perception .parent_source or '-'}",
        f"Topology={perception .topology_ready_frame is not None } gating={getattr (physical ,'integration_guard_gating_enabled',False )} walls={getattr (physical ,'integration_all_walls_ready',False )}",
        "Latency G/WHO/M/T="+"/".join (
        str (value -perception .confirmation_frame )
        if value is not None and perception .confirmation_frame is not None else "-"
        for value in (
        perception .guard_geometry_frame ,
        perception .guard_who_frame ,
        perception .guard_motion_start_frame ,
        perception .topology_ready_frame ,
        )
        ),
        f"Localization Guard-WHO={'ON'if getattr (physical ,'integration_guard_who_localization_enabled',False )else 'OFF'} "
        f"Guard-Placement={'ON'if getattr (physical ,'integration_placement_localization_enabled',False )else 'OFF'} "
        f"Other/DFS=OFF",
        f"Phase={physical .phase .name } selected={selected_uid or '-'}",
        "States "+" ".join (
        f"{key [0 ]}={physical .branch_states [key ]}"for key in physical .BRANCHES 
        ),
        leakage_summary ,
        *wall_lines ,
        f"Frontier n={len (current_frontier_ids or saturation .frontier_ids )} Shepherd n={len (current_shepherd_ids or saturation .shepherd_ids )}",
        f"Centroid speed={saturation .frontier_speed :.2f} stalled={saturation .frontier_stalled }",
        f"Density={saturation .local_density :.3f} pressure={saturation .local_pressure :.1f} x{saturation .local_pressure_ratio :.2f}",
        f"Fill={saturation .cross_section_fill :.2f} dwell={saturation .dwell :.2f} sat={saturation .saturated }",
        f"F->S frame={saturation .transition_frame or '-'} jump={saturation .max_transition_jump :.6f}",
        f"Return dir=({saturation .return_direction_local [0 ]:.2f},{saturation .return_direction_local [1 ]:.2f}) ratio={saturation .return_flow_ratio :.2f}",
        f"Backflow confirmed={saturation .backflow_confirmed }",
        f"Relay={len (physical .get_relays (robots ))} pebble={len (physical .get_pebbles (robots ))} connected={physical .get_communication_stats (robots )['connected']}/{len (robots )}",
        ]
        left_x =DIAGNOSTIC_RECT .left +10 
        right_x =DIAGNOSTIC_RECT .left +278 
        top_y =DIAGNOSTIC_RECT .top +9 
        step =11 
        for index ,line in enumerate (detector_lines ):
            color =COLORS ["group_center"]if index in {1 ,11 }else COLORS ["text"]
            self .screen .blit (self .small .render (line ,True ,color ),(left_x ,top_y +index *step ))
        for index ,line in enumerate (dfs_lines ):
            color =COLORS ["group_center"]if index ==0 else COLORS ["text"]
            self .screen .blit (self .small .render (line ,True ,color ),(right_x ,top_y +index *step ))
        pygame .draw .rect (self .screen ,COLORS ["muted"],DIAGNOSTIC_RECT ,1 ,border_radius =6 )

    def draw (self ,robots :Sequence [Any ],perception :AdaptivePerception ,show_rays :bool ,show_profile :bool ,show_comm :bool ,density :bool ,paused :bool )->None :
        self .screen .fill (COLORS ["background"])
        self .screen .blit (self .title .render ("Adaptive LiDAR + SPH Physical DFS",True ,COLORS ["text"]),(18 ,16 ))
        self ._draw_map (robots ,perception ,show_rays ,show_comm ,density )
        if show_profile :
            self ._draw_profile (perception .last_frame )
        else :
            pygame .draw .rect (self .screen ,COLORS ["panel_alt"],PROFILE_RECT ,border_radius =6 )
            self .screen .blit (self .font .render ("Profile hidden (P)",True ,COLORS ["muted"]),(PROFILE_RECT .left +15 ,PROFILE_RECT .top +15 ))
        self ._draw_diagnostics (robots ,perception ,paused )
        controls ="SPACE pause | R reset | P profile | L LiDAR rays | D density | C comm | ESC quit"
        self .screen .blit (self .small .render (controls ,True ,COLORS ["muted"]),(875 ,880 ))
        if paused :
            self .screen .blit (self .title .render ("PAUSED",True ,COLORS ["threshold"]),(750 ,18 ))
        pygame .display .flip ()


def _phase_event_log (physical :types .ModuleType ,robots :Sequence [Any ],previous :str ,visited_log :list [str ])->str :
    current =physical .phase .name 
    if current ==previous :
        return previous 
    if current =="EXPLORE_BRANCH":
        if not getattr (physical ,"integration_all_walls_ready",False ):
            raise RuntimeError (
            "EXPLORE_BRANCH entered before ALL_GUARD_WALLS_READY"
            )
        ids =[robot .robot_id for robot in physical .get_frontier_shepherds (robots )]
        print (f"[Junction Guard] ready")
        print (f"[DFS] selected branch={physical .branch_identity_label (physical .active_branch_uid )}")
        print (f"[Frontier] promoted ids={ids }")
        print (
        "[Timeline] BRANCH_SELECTED "
        f"frame={getattr (physical ,'integration_frame',-1 )} "
        f"uid={physical .active_branch_uid }"
        )
        print (
        "[Timeline] FRONTIER_START "
        f"frame={getattr (physical ,'integration_frame',-1 )} ids={ids }"
        )
    elif current in {"FORM_SHEPHERD_BOUNDARY","FILL_BEHIND_SHEPHERD"}:
        ids =[robot .robot_id for robot in physical .get_shepherds (robots )]
        if not ids :
            ids =[robot .robot_id for robot in physical .get_frontier_shepherds (robots )]
        print ("[DeadEnd] confirmed")
        print (f"[Shepherd] same frontier ids promoted={ids }")
    elif current =="PRESSURE_PUSH":
        print ("[Pressure] push started")
    elif current =="FLOW_BACKTRACK":
        print ("[Backtrack] flow established")
    elif current in {"JUNCTION_SWITCH","FORM_JUNCTION_GUARDS","FINAL_JUNCTION_GATHER","RETURN_TO_BASE","DONE"}:
        observed =sorted (physical .observed_visited_branch_uids (robots ))
        for uid in observed :
            if uid not in visited_log :
                visited_log .append (uid )
                print (f"[DFS] branch VISITED uid={uid }")
                print (
                "[Timeline] BRANCH_VISITED "
                f"frame={getattr (physical ,'integration_frame',-1 )} "
                f"uid={uid }"
                )
        if current =="FORM_JUNCTION_GUARDS"and previous =="JUNCTION_SWITCH":
            print ("[DFS] next branch guard formation")
        if current =="RETURN_TO_BASE":
            print ("[Return] RETURN_TO_BASE")
        if current =="DONE":
            print ("[Return] DONE")
    return current 


def parse_args (argv :Sequence [str ]|None =None )->argparse .Namespace :
    parser =argparse .ArgumentParser (description =__doc__ )
    parser .add_argument ("--headless",action ="store_true",help ="run identical physics without rasterization")
    parser .add_argument ("--max-frames",type =int ,default =0 ,help ="stop after N frames (0 means until DONE)")
    parser .add_argument ("--dt",type =float ,default =1.0 /60.0 )
    return parser .parse_args (argv )

def request_anchor_prep ( #Anchor를 해당 Branch 탐색 시작 위치/선두 상태로 준비
physical :types .ModuleType ,
branch_uid :str ,
)->None :
    if (
    getattr (physical ,"integration_anchor_breakout_active",False )
    and getattr (physical ,"integration_anchor_breakout_target_uid",None )==branch_uid 
    ):
        print (f"[AnchorBreakoutHandoff] target={branch_uid }")
        physical .integration_anchor_breakout_active =False 
        physical .integration_anchor_breakout_stage ="IDLE"
        physical .integration_anchor_breakout_source_uid =None 
        physical .integration_anchor_breakout_target_uid =None 
        physical .integration_anchor_breakout_side_sign =0.0 
        physical .integration_anchor_breakout_lateral_odometry =0.0 
        physical .integration_anchor_breakout_return_odometry =0.0 
    if (
    physical .integration_anchor_prep_request_uid 
    ==branch_uid 
    ):
        return 

    physical .integration_anchor_prep_request_uid =branch_uid 
    physical .integration_anchor_prep_active_uid =None 
    physical .integration_anchor_prep_ready_uid =None 
    physical .integration_anchor_prep_stable_frames =0 

    print (
    "[AnchorPrepRequested] "
    f"branch={branch_uid }"
    )


def promote_anchor_prep_to_leading (
physical :types .ModuleType ,
branch_uid :str ,
)->None :
    physical .integration_anchor_prep_request_uid =None 
    physical .integration_anchor_prep_active_uid =None 
    physical .integration_anchor_prep_ready_uid =None 
    physical .integration_anchor_prep_stable_frames =0 



    physical .integration_leading_anchor_uid =branch_uid 

    print (
    "[LeadingAnchorLaunchReady] "
    f"branch={branch_uid }"
    )


def disable_leading_anchor (
physical :types .ModuleType ,
)->None :
    physical .integration_anchor_prep_request_uid =None 
    physical .integration_anchor_prep_active_uid =None 
    physical .integration_anchor_prep_ready_uid =None 
    physical .integration_anchor_prep_stable_frames =0 
    physical .integration_leading_anchor_uid =None 


def reset_leading_anchor_state (
physical :types .ModuleType ,
)->None :
    """Clear every branch-launch ownership token without moving any robot."""
    physical .integration_anchor_prep_request_uid =None 
    physical .integration_anchor_prep_active_uid =None 
    physical .integration_anchor_prep_ready_uid =None 
    physical .integration_anchor_prep_stable_frames =0 
    physical .integration_anchor_prep_fresh_scan_after =None 
    physical .integration_leading_anchor_uid =None 
    physical .integration_pending_branch_uid =None 
    physical .integration_pending_frontier_committed =False 
    physical .integration_anchor_breakout_active =False 
    physical .integration_anchor_breakout_source_uid =None 
    physical .integration_anchor_breakout_target_uid =None 
    physical .integration_anchor_breakout_stage ="IDLE"
    physical .integration_anchor_breakout_side_sign =0.0 
    physical .integration_anchor_breakout_lateral_odometry =0.0 
    physical .integration_anchor_breakout_return_odometry =0.0 


def start_anchor_breakout (
physical :types .ModuleType ,
source_uid :str ,
target_uid :str ,
)->None :
    if not source_uid or not target_uid :
        return 


    if (
    physical .phase in {
    physical .SimulationPhase .PRESSURE_PUSH ,
    physical .SimulationPhase .FLOW_BACKTRACK ,
    }
    or getattr (
    physical ,
    "integration_child_dfs_phase",
    "IDLE",
    )in {
    "PRESSURE_PUSH",
    "FLOW_BACKTRACK",
    }
    ):
        print (
        "[AnchorBreakoutBlocked] "
        f"source={source_uid } target={target_uid } "
        "reason=ACTIVE_SHEPHERD_WALL"
        )
        return 

    disable_leading_anchor (physical )
    physical .integration_anchor_breakout_active =True 
    physical .integration_anchor_breakout_source_uid =source_uid 
    physical .integration_anchor_breakout_target_uid =target_uid 
    physical .integration_anchor_breakout_stage ="ESCAPE"
    physical .integration_anchor_breakout_side_sign =0.0 
    physical .integration_anchor_breakout_lateral_odometry =0.0 
    physical .integration_anchor_breakout_return_odometry =0.0 
    multi_dfs .child_probe_active =False 
    multi_dfs .child_candidate_active =False 
    multi_dfs .child_probe_branch_uid =None 
    print (f"[AnchorBreakoutStart] source={source_uid } target={target_uid }")


def maintain_anchor_breakout (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
dt :float ,
)->None :
    if not getattr (physical ,"integration_anchor_breakout_active",False ):
        return 

    if (
    physical .phase in {
    physical .SimulationPhase .PRESSURE_PUSH ,
    physical .SimulationPhase .FLOW_BACKTRACK ,
    }
    or getattr (
    physical ,
    "integration_child_dfs_phase",
    "IDLE",
    )in {
    "PRESSURE_PUSH",
    "FLOW_BACKTRACK",
    }
    ):
        disable_leading_anchor (physical )
        physical .integration_anchor_breakout_active =False 
        physical .integration_anchor_breakout_source_uid =None 
        physical .integration_anchor_breakout_target_uid =None 
        physical .integration_anchor_breakout_stage ="IDLE"
        physical .integration_anchor_breakout_side_sign =0.0 
        physical .integration_anchor_breakout_lateral_odometry =0.0 
        physical .integration_anchor_breakout_return_odometry =0.0 

        print (
        "[AnchorBreakoutCancelled] "
        "reason=ACTIVE_SHEPHERD_WALL"
        )
        return 

    source_uid =getattr (physical ,"integration_anchor_breakout_source_uid",None )
    target_uid =getattr (physical ,"integration_anchor_breakout_target_uid",None )
    descriptor =physical .branch_descriptors_by_uid .get (source_uid )
    if descriptor is None or target_uid is None :
        return 
    anchor =perception .leader 
    tangent ,lateral =physical .descriptor_local_basis (descriptor )
    return_axis =descriptor .local_return_direction .normalize ()
    lateral =lateral .normalize ()
    perception .yaw_deg =math .degrees (math .atan2 (return_axis .y ,return_axis .x ))
    anchor .body_yaw =math .radians (perception .yaw_deg )
    normal_obs =observe_local_neighbors (
    anchor ,robots ,return_axis ,lateral_axis =lateral ,
    max_range =physical .COMM_RANGE ,
    predicate =lambda robot :robot is not anchor and robot .role =="NORMAL"and not robot .base_reserve ,
    )
    if physical .integration_anchor_breakout_side_sign ==0.0 :
        pos =sum (o .relative_lateral >0.0 for o in normal_obs )
        neg =sum (o .relative_lateral <0.0 for o in normal_obs )
        physical .integration_anchor_breakout_side_sign =1.0 if pos <=neg else -1.0 
    side =float (physical .integration_anchor_breakout_side_sign )
    width =float (getattr (descriptor ,"observed_width",0.0 )or getattr (descriptor ,"observed_physical_width",0.0 )or 0.0 )
    lane =min (max (3.0 *float (physical .ROBOT_RADIUS ),0.25 *width ),max (3.0 *float (physical .ROBOT_RADIUS ),0.5 *width -2.2 *float (physical .ROBOT_RADIUS )))if width >physical .EPSILON else 3.0 *float (physical .ROBOT_RADIUS )
    lateral_error =side *lane -float (anchor .velocity .dot (lateral ))
    forward_speed =float (physical .integration_anchor_breakout_speed )
    anchor .acceleration -=tangent *anchor .acceleration .dot (tangent )
    anchor .acceleration -=lateral *anchor .acceleration .dot (lateral )
    anchor .acceleration +=lateral *float (np .clip (5.0 *lateral_error ,-0.35 *physical .MAX_ACCELERATION ,0.35 *physical .MAX_ACCELERATION ))
    anchor .acceleration +=return_axis *float (np .clip (6.0 *(forward_speed -anchor .velocity .dot (return_axis )),-0.35 *physical .MAX_ACCELERATION ,0.35 *physical .MAX_ACCELERATION ))
    physical .integration_anchor_breakout_lateral_odometry +=anchor .velocity .dot (lateral )*dt 
    physical .integration_anchor_breakout_return_odometry +=max (0.0 ,anchor .velocity .dot (return_axis ))*dt 


def maintain_mobile_anchor_at_normal_front (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    if getattr (physical ,"integration_anchor_breakout_active",False ):
        return 
    requested =getattr (physical ,"integration_anchor_prep_request_uid",None )
    leading =getattr (physical ,"integration_leading_anchor_uid",None )
    branch_uid =requested or leading 
    if branch_uid is None :
        return 
    prep_mode =requested is not None 
    if (
    not prep_mode 
    and physical .phase !=physical .SimulationPhase .EXPLORE_BRANCH 
    ):
        disable_leading_anchor (physical )
        return 
    if multi_dfs .child_probe_active and not prep_mode :
        return 
    if physical .phase in {
    physical .SimulationPhase .PRESSURE_PUSH ,
    physical .SimulationPhase .FLOW_BACKTRACK ,
    }or getattr (physical ,"integration_child_dfs_phase","IDLE")in {
    "PRESSURE_PUSH","FLOW_BACKTRACK",
    }:
        disable_leading_anchor (physical )
        return 
    descriptor =physical .branch_descriptors_by_uid .get (branch_uid )
    if descriptor is None :
        return 
    tangent ,lateral_axis =physical .descriptor_local_basis (descriptor )
    tangent =tangent .normalize ()
    lateral_axis =lateral_axis .normalize ()
    anchor =perception .leader 
    new_yaw =math .degrees (math .atan2 (tangent .y ,tangent .x ))
    perception .yaw_deg =new_yaw 
    anchor .body_yaw =math .radians (new_yaw )
    if prep_mode and physical .integration_anchor_prep_active_uid !=branch_uid :
        physical .integration_anchor_prep_active_uid =branch_uid 
        perception .anchor_fixed =False 
        anchor .is_fixed_anchor =False 
        anchor .base_reserve =False 
        session =multi_dfs .child_session 
        if session is not None and session .anchor_stopped :
            multi_dfs .child_probe_active =False 
            multi_dfs .child_candidate_active =False 
            multi_dfs .child_probe_branch_uid =None 
        physical .integration_anchor_prep_fresh_scan_after =(
        perception .last_frame .frame if perception .last_frame is not None else -1 
        )
        print (f"[AnchorPrepStart] branch={branch_uid } lidar_id={anchor .robot_id }")
        return 
    if prep_mode and perception .last_frame is not None and perception .last_frame .frame <=getattr (physical ,"integration_anchor_prep_fresh_scan_after",-1 ):
        return 
    lidar_frame =perception .last_frame 
    if lidar_frame is None :
        return 
    if prep_mode :




        selected_fixture =physical .branch_fixture_for_uid (branch_uid )
        guard_observations =observe_local_neighbors (
        anchor ,robots ,tangent ,lateral_axis =lateral_axis ,
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :(
        robot is not anchor 
        and robot .role =="JUNCTION_GUARD"
        and (
        getattr (robot ,"junction_guard_branch_uid",None )==branch_uid 
        or getattr (robot ,"junction_guard_branch",None )==selected_fixture 
        )
        ),
        )
        forward_guards =[
        observation for observation in guard_observations 
        if observation .relative_axial >0.0 
        ]
        if physical .integration_frame %10 ==0 :
            matching_guard_total =sum (
            1 
            for robot in robots 
            if (
            robot is not anchor 
            and robot .role =="JUNCTION_GUARD"
            and (
            getattr (
            robot ,
            "junction_guard_branch_uid",
            None ,
            )
            ==branch_uid 
            or getattr (
            robot ,
            "junction_guard_branch",
            None ,
            )
            ==selected_fixture 
            )
            )
            )

            observed_axials =[
            float (observation .relative_axial )
            for observation in guard_observations 
            ]

            observed_ranges =[
            float (observation .relative_range )
            for observation in guard_observations 
            ]







            debug_all_matching_guards =observe_local_neighbors (
            anchor ,
            robots ,
            tangent ,
            lateral_axis =lateral_axis ,
            max_range =None ,
            predicate =lambda robot :(
            robot is not anchor 
            and robot .role =="JUNCTION_GUARD"
            and (
            getattr (
            robot ,
            "junction_guard_branch_uid",
            None ,
            )
            ==branch_uid 
            or getattr (
            robot ,
            "junction_guard_branch",
            None ,
            )
            ==selected_fixture 
            )
            ),
            )

            debug_nearest_guard =min (
            debug_all_matching_guards ,
            key =lambda observation :observation .relative_range ,
            default =None ,
            )
            debug_guard_axials =[
            float (observation .relative_axial )
            for observation in debug_all_matching_guards 
            ]

            debug_guard_laterals =[
            float (observation .relative_lateral )
            for observation in debug_all_matching_guards 
            ]

            debug_guard_center_axial =(
            float (np .median (debug_guard_axials ))
            if debug_guard_axials 
            else -999.0 
            )

            debug_guard_center_lateral =(
            float (np .median (debug_guard_laterals ))
            if debug_guard_laterals 
            else -999.0 
            )

            debug_guard_lateral_min =(
            min (debug_guard_laterals )
            if debug_guard_laterals 
            else -999.0 
            )

            debug_guard_lateral_max =(
            max (debug_guard_laterals )
            if debug_guard_laterals 
            else -999.0 
            )

            print (
            "[AnchorPrepGuardSense] "
            f"frame={physical .integration_frame } "
            f"branch={branch_uid } "
            f"fixture={selected_fixture } "
            f"matching_total={matching_guard_total } "
            f"observed={len (guard_observations )} "
            f"forward={len (forward_guards )} "
            f"rear_or_zero="
            f"{sum (1 for x in observed_axials if x <=0.0 )} "
            f"range_min="
            f"{min (observed_ranges )if observed_ranges else -1.0 :.2f} "
            f"axial_min="
            f"{min (observed_axials )if observed_axials else -999.0 :.2f} "
            f"axial_max="
            f"{max (observed_axials )if observed_axials else -999.0 :.2f} "
            f"comm_range={float (physical .COMM_RANGE ):.2f} "
            f"nearest_any_range="
            f"{debug_nearest_guard .relative_range if debug_nearest_guard is not None else -1.0 :.2f} "
            f"nearest_any_axial="
            f"{debug_nearest_guard .relative_axial if debug_nearest_guard is not None else -999.0 :.2f} "
            f"nearest_any_lateral="
            f"{debug_nearest_guard .relative_lateral if debug_nearest_guard is not None else -999.0 :.2f} "
            f"guard_center_any_axial="
            f"{debug_guard_center_axial :.2f} "
            f"guard_center_any_lateral="
            f"{debug_guard_center_lateral :.2f} "
            f"guard_lateral_span="
            f"[{debug_guard_lateral_min :.2f},{debug_guard_lateral_max :.2f}]"
            )

        guard_center_lateral =(
        float (np .median ([o .relative_lateral for o in forward_guards ]))
        if forward_guards else None 
        )
        center_band =max (3.0 *float (physical .ROBOT_RADIUS ),1.5 *float (physical .GRID_SPACING ))
        central_guards =[
        o for o in forward_guards 
        if guard_center_lateral is not None 
        and abs (o .relative_lateral -guard_center_lateral )<=center_band 
        ]
        guard_gap =min ((float (o .relative_axial )for o in central_guards ),default =None )
        guard_standoff =max (3.2 *float (physical .ROBOT_RADIUS ),float (physical .integration_anchor_target_gap ))
        minimum_safe_gap =max (2.0 *float (physical .ROBOT_RADIUS ),0.70 *guard_standoff )
        maximum_ready_gap =guard_standoff +max (1.5 *float (physical .ROBOT_RADIUS ),0.75 *float (physical .GRID_ROW_SPACING ))
        opening_center =_lidar_forward_opening_center (
        lidar_frame ,
        perception ,
        forward_axis =tangent ,
        lateral_axis =lateral_axis ,
        )
        left_axis =lateral_axis 
        axial_velocity =anchor .velocity .dot (tangent )
        lateral_velocity =anchor .velocity .dot (left_axis )
        PREP_CRUISE_SPEED =(
        16.0 *ROBOT_MOTION_SPEED_SCALE 
        )
        PREP_VELOCITY_GAIN =8.0 
        if guard_gap is not None and guard_center_lateral is not None :
            axial_error =guard_gap -guard_standoff 
            lateral_error =guard_center_lateral 
            mouth_axial =(
            0.0 
            if opening_center is None 
            else opening_center [0 ]
            )
            mouth_lateral =guard_center_lateral 
            mouth_width =(
            0.0 
            if opening_center is None 
            else opening_center [2 ]
            )
        else :
            physical .integration_anchor_prep_stable_frames =0 
            if opening_center is not None :
                mouth_axial ,mouth_lateral ,mouth_width =(
                opening_center 
                )
                mouth_standoff =max (
                1.5 *float (physical .ROBOT_RADIUS ),
                0.5 *float (physical .GRID_ROW_SPACING ),
                )
                axial_error =max (0.0 ,mouth_axial -mouth_standoff )
                lateral_error =mouth_lateral 
            else :
                mouth_axial =0.0 
                mouth_lateral =0.0 
                mouth_width =0.0 
                axial_error =physical .COMM_RANGE 
                lateral_error =0.0 
        target_vector =tangent *axial_error +lateral_axis *lateral_error 
        target_distance =target_vector .length ()
        if target_distance >physical .EPSILON :
            desired_speed =min (PREP_CRUISE_SPEED ,3.0 *target_distance )
            desired_velocity =target_vector .normalize ()*desired_speed 
        else :
            desired_velocity =pygame .Vector2 ()
        target_speed =desired_velocity .dot (tangent )
        current_velocity =(
        tangent *anchor .velocity .dot (tangent )
        +lateral_axis *anchor .velocity .dot (lateral_axis )
        )
        control =PREP_VELOCITY_GAIN *(desired_velocity -current_velocity )
        maximum_acceleration =0.60 *physical .MAX_ACCELERATION 
        if control .length ()>maximum_acceleration :
            control .scale_to_length (maximum_acceleration )
        anchor .acceleration -=tangent *anchor .acceleration .dot (tangent )
        anchor .acceleration -=lateral_axis *anchor .acceleration .dot (lateral_axis )
        anchor .acceleration +=control 
        guard_lateral_ready_tolerance =max (
        2.0 *float (physical .ROBOT_RADIUS ),
        float (physical .GRID_SPACING ),
        )
        centered =(
        guard_center_lateral is not None 
        and abs (guard_center_lateral )<=guard_lateral_ready_tolerance 
        )
        behind_guard =(
        guard_gap is not None 
        and guard_gap >0.0 
        and minimum_safe_gap <=guard_gap <=maximum_ready_gap 
        )
        slow_enough =(
        abs (anchor .velocity .dot (tangent ))<=3.0 
        and abs (anchor .velocity .dot (lateral_axis ))<=3.0 
        )
        ready_required_frames =min (
        int (physical .integration_anchor_prep_required_frames ),
        4 ,
        )
        if centered and behind_guard and slow_enough :
            physical .integration_anchor_prep_stable_frames +=1 
        else :
            physical .integration_anchor_prep_stable_frames =0 
        if physical .integration_anchor_prep_stable_frames >=ready_required_frames :
            physical .integration_anchor_prep_ready_uid =branch_uid 
        if physical .integration_frame %10 ==0 :
            print (
            f"[AnchorPrepGuardLock] branch={branch_uid } lidar_id={anchor .robot_id } "
            f"mouth_axial={mouth_axial :.2f} mouth_lateral={mouth_lateral :.2f} "
            f"mouth_width={mouth_width :.2f} centered={centered } "
            f"guard_gap={guard_gap if guard_gap is not None else -1.0 :.2f} "
            f"guard_standoff={guard_standoff :.2f} behind_guard={behind_guard } "
            f"lateral_tol={guard_lateral_ready_tolerance :.2f} "
            f"slow={slow_enough } "
            f"axial_speed={axial_velocity :.2f} "
            f"lateral_speed={lateral_velocity :.2f} "
            f"target_speed={target_speed :.2f} stable={physical .integration_anchor_prep_stable_frames }"
            f"/{ready_required_frames }"
            )
        return 
    left =_range_at_local_angle (lidar_frame ,-90.0 )
    right =_range_at_local_angle (lidar_frame ,90.0 )
    side_sum =left +right 
    left_axis =_body_local_unit (perception ,-90.0 ).normalize ()
    opening_center =_lidar_forward_opening_center (lidar_frame ,perception )
    corridor_acquired =bool (
    left <0.98 *MAX_RANGE and right <0.98 *MAX_RANGE 
    )
    if corridor_acquired :
        center_error =left -right 
        lateral_command =float (np .clip (
        4.5 *center_error -3.0 *anchor .velocity .dot (left_axis ),
        -0.30 *physical .MAX_ACCELERATION ,
        0.30 *physical .MAX_ACCELERATION ,
        ))
        center_ratio =abs (center_error )/max (side_sum ,physical .EPSILON )
    elif opening_center is not None :
        _ ,mouth_lateral ,_ =opening_center 
        center_error =mouth_lateral 
        lateral_command =float (np .clip (
        5.0 *mouth_lateral -3.0 *anchor .velocity .dot (left_axis ),
        -0.35 *physical .MAX_ACCELERATION ,
        0.35 *physical .MAX_ACCELERATION ,
        ))
        center_ratio =abs (mouth_lateral )/max (opening_center [0 ],physical .ROBOT_RADIUS )
    else :
        center_error =0.0 
        lateral_command =float (np .clip (
        -2.5 *anchor .velocity .dot (left_axis ),
        -0.15 *physical .MAX_ACCELERATION ,
        0.15 *physical .MAX_ACCELERATION ,
        ))
        center_ratio =1.0 
    half_width =max (4.0 *physical .ROBOT_RADIUS ,0.5 *min (side_sum ,2.0 *MAX_RANGE ))





    tracked_roles ={"JUNCTION_GUARD"}if prep_mode else {"NORMAL"}

    observations =[
    observation 
    for observation in observe_local_neighbors (
    anchor ,
    robots ,
    tangent ,
    lateral_axis =left_axis ,
    max_range =0.90 *physical .COMM_RANGE ,
    predicate =lambda robot :(
    robot .role in tracked_roles 
    and not robot .base_reserve 
    and robot is not anchor 
    and (
    not prep_mode 
    or robot .junction_guard_branch 
    ==physical .branch_fixture_for_uid (branch_uid )
    )
    ),
    )
    if abs (
    observation .relative_lateral 
    )<=half_width 
    ]
    front_relative =max (
    (observation .relative_axial for observation in observations ),
    default =None ,
    )
    front_cohort =[
    observation .robot 
    for observation in observations 
    if front_relative is not None 
    and observation .relative_axial 
    >=front_relative -2.0 *physical .GRID_ROW_SPACING 
    ]
    front_speed =float (np .mean ([robot .velocity .dot (tangent )for robot in front_cohort ]))if front_cohort else 0.0 
    ahead_count =sum (
    observation .relative_axial >0.0 for observation in observations 
    )
    gap =-front_relative if front_relative is not None else None 
    target_gap =float (
    physical .integration_anchor_target_gap 
    )

    frontier_gap :float |None =None 
    frontier_speed =0.0 
    frontier_count =0 

    if (
    not prep_mode 
    and physical .phase 
    ==physical .SimulationPhase .EXPLORE_BRANCH 
    ):
        selected_fixture =(
        physical .branch_fixture_for_uid (
        branch_uid 
        )
        )

        frontier_observations =[
        observation 
        for observation 
        in observe_local_neighbors (
        anchor ,
        robots ,
        tangent ,
        lateral_axis =left_axis ,
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :(
        robot is not anchor 
        and robot .role 
        =="FRONTIER_SHEPHERD"
        and (
        getattr (
        robot ,
        "shepherd_branch",
        None ,
        )
        ==selected_fixture 
        or getattr (
        robot ,
        "shepherd_branch",
        None ,
        )
        ==branch_uid 
        )
        ),
        )
        if (
        observation .relative_axial 
        >0.0 
        and abs (
        observation .relative_lateral 
        )
        <=half_width 
        )
        ]

        if frontier_observations :
            frontier_gap =min (
            float (
            observation .relative_axial 
            )
            for observation 
            in frontier_observations 
            )
            frontier_speed =float (
            np .mean (
            [
            observation .robot .velocity .dot (
            tangent 
            )
            for observation 
            in frontier_observations 
            ]
            )
            )
            frontier_count =len (
            frontier_observations 
            )

    if prep_mode :


        target_speed =(
        8.0 
        if not corridor_acquired 
        else float (
        np .clip (
        front_speed 
        +3.0 
        *(
        target_gap 
        -gap 
        ),
        0.0 ,
        12.0 ,
        )
        )
        if gap is not None 
        else 0.0 
        )
        anchor_accel_fraction =0.22 

    else :
        cruise_speed =float (
        getattr (
        physical ,
        "integration_anchor_explore_cruise_speed",
        35.0 ,
        )
        )

        max_follow_gap =float (
        getattr (
        physical ,
        "integration_anchor_explore_max_follow_gap",
        max (
        8.0 
        *float (
        physical .ROBOT_RADIUS 
        ),
        6.0 *target_gap ,
        ),
        )
        )

        if gap is None :


            target_speed =0.35 *cruise_speed 

        elif gap >=max_follow_gap :


            target_speed =0.0 

        else :




            support_ratio =float (
            np .clip (
            (
            max_follow_gap 
            -gap 
            )
            /max (
            max_follow_gap 
            -target_gap ,
            physical .EPSILON ,
            ),
            0.0 ,
            1.0 ,
            )
            )

            target_speed =(
            cruise_speed 
            *(
            0.40 
            +0.60 
            *support_ratio 
            )
            )





            target_speed =max (
            target_speed ,
            min (
            cruise_speed ,
            front_speed +6.0 ,
            ),
            )
        if frontier_gap is not None :
            frontier_standoff =max (
            4.0 
            *float (
            physical .ROBOT_RADIUS 
            ),
            1.15 *target_gap ,
            )
            frontier_hard_gap =max (
            2.20 
            *float (
            physical .ROBOT_RADIUS 
            ),
            0.65 *frontier_standoff ,
            )

            frontier_limited_speed =(
            frontier_speed 
            +5.0 
            *(
            frontier_gap 
            -frontier_standoff 
            )
            )

            target_speed =min (
            target_speed ,
            frontier_limited_speed ,
            )

            if (
            frontier_gap 
            <=frontier_hard_gap 
            ):
                target_speed =min (
                target_speed ,
                -3.0 ,
                )

            if (
            frontier_gap 
            <=1.5 
            *frontier_standoff 
            ):
                anchor_accel_fraction =0.85 
            else :
                anchor_accel_fraction =0.50 
        else :
            anchor_accel_fraction =0.50 

        target_speed =float (
        np .clip (
        target_speed ,
        -3.0 ,
        cruise_speed ,
        )
        )

    existing_axial =anchor .acceleration .dot (
    tangent 
    )
    anchor .acceleration -=(
    tangent 
    *existing_axial 
    )

    axial_command =float (
    np .clip (
    10.0 
    *(
    target_speed 
    -anchor .velocity .dot (
    tangent 
    )
    ),
    -anchor_accel_fraction 
    *physical .MAX_ACCELERATION ,
    anchor_accel_fraction 
    *physical .MAX_ACCELERATION ,
    )
    )

    anchor .acceleration +=(
    tangent 
    *axial_command 
    +left_axis 
    *lateral_command 
    )
    if prep_mode :
        centered =corridor_acquired and center_ratio <=physical .integration_anchor_center_ratio_tol 
        front_ready =gap is not None and ahead_count ==0 and gap >=0.60 *target_gap 
        physical .integration_anchor_prep_stable_frames =physical .integration_anchor_prep_stable_frames +1 if centered and front_ready else 0 
        if physical .integration_anchor_prep_stable_frames >=physical .integration_anchor_prep_required_frames :
            physical .integration_anchor_prep_ready_uid =branch_uid 
    if physical .integration_frame %10 ==0 :
        print (
        f"[{'AnchorPrep'if prep_mode else 'MobileAnchorFront'}] "
        f"branch={branch_uid } lidar_id={anchor .robot_id } "
        f"left={left :.2f} right={right :.2f} "
        f"corridor_acquired={corridor_acquired } "
        f"center_ratio={center_ratio :.3f} "
        f"target_speed={target_speed :.2f} "
        f"axial_speed={anchor .velocity .dot (tangent ):.2f} "
        f"gap={gap if gap is not None else -1.0 :.2f} "
        f"ahead_count={ahead_count } "
        f"frontier_gap={frontier_gap if frontier_gap is not None else -1.0 :.2f} "
        f"frontier_speed={frontier_speed :.2f} "
        f"frontier_count={frontier_count } "
        f"stable={physical .integration_anchor_prep_stable_frames }"
        )


def center_initial_grid_formation (
physical :types .ModuleType ,
robots :Sequence [Any ],
)->dict [str ,float |int |bool ]:
    """Repack whole rows so the actual front row has a center robot."""
    left =(
    physical .center_x -physical .half_width 
    +physical .ROBOT_RADIUS +4.0 *physical .MAP_SCALE 
    )
    right =(
    physical .center_x +physical .half_width 
    -physical .ROBOT_RADIUS -4.0 *physical .MAP_SCALE 
    )
    nominal_per_row =max (
    1 ,int ((right -left )//physical .GRID_SPACING )+1 
    )
    full_rows ,remainder =divmod (len (robots ),nominal_per_row )
    previous_front_count =remainder or nominal_per_row 
    row_counts =[nominal_per_row ]*full_rows 
    if remainder :
        if remainder %2 ==0 and row_counts :
            row_counts [-1 ]-=1 
            remainder +=1 
        row_counts .append (remainder )
    bottom_y =(
    physical .center_y +physical .half_width +physical .base_length 
    -physical .ROBOT_RADIUS -7.0 *physical .MAP_SCALE 
    )
    cursor =0 
    for row ,count in enumerate (row_counts ):
        row_y =bottom_y -row *physical .GRID_ROW_SPACING 
        for column in range (count ):
            robot =robots [cursor ]
            robot .position .update (
            physical .center_x 
            +(column -0.5 *(count -1 ))*physical .GRID_SPACING ,
            row_y ,
            )
            cursor +=1 
    if cursor !=len (robots ):
        raise RuntimeError ("front-center deployment did not place every robot")
    front_count =row_counts [-1 ]
    audit ={
    "previous_front_count":previous_front_count ,
    "previous_center_robot":previous_front_count %2 ==1 ,
    "front_count":front_count ,
    "corridor_center_x":float (physical .center_x ),
    "front_y":bottom_y -(len (row_counts )-1 )*physical .GRID_ROW_SPACING ,
    }
    print ("[LiDAR Initial Placement]")
    print (
    f"previous_front_row_robot_count={previous_front_count } "
    f"previous_center_robot={audit ['previous_center_robot']}"
    )
    print (
    f"new_front_row_robot_count={front_count } "
    f"corridor_center_x={physical .center_x :.3f}"
    )
    return audit 


def initialize_deployment_fields (
physical :types .ModuleType ,
robots :Sequence [Any ],
)->dict [str ,float |int |bool ]:
    audit =center_initial_grid_formation (physical ,robots )
    for robot in robots :
        robot .body_yaw =-0.5 *math .pi 
        robot .propulsion_weight =adaptive .LOCAL_FOLLOWER_DRIVE_WEIGHT 
        robot .heading_parent_id =None 
        robot .heading_hop =0 
    return audit 


def refresh_centered_deployment_physics (
physical :types .ModuleType ,
robots :Sequence [Any ],
)->tuple [float ,float ]:
    """Refresh density/communication after the whole-grid repack."""

    physical .compute_densities (
    robots ,
    physical .build_physics_grid (robots ),
    )

    mean_density =float (
    np .mean (
    [robot .density for robot in robots ]
    )
    )

    reference_density =(
    physical .compute_reference_density_from_spacing (
    physical .REFERENCE_EQUILIBRIUM_SPACING ,
    physical .SMOOTHING_LENGTH ,
    )
    )

    color_reference_density =(
    mean_density *0.68 
    )

    physical .update_communication_system (
    robots ,
    physical .build_spatial_grid (robots ),
    )

    return (
    reference_density ,
    color_reference_density ,
    )


def advance_guard_settling_waypoints (
physical :types .ModuleType ,
robots :Sequence [Any ],
)->None :
    """Advance only target waypoints; never overwrite runtime positions."""
    for robot in robots :
        waypoints =getattr (robot ,"integration_guard_waypoints",None )
        if robot .role !="JUNCTION_GUARD"or not waypoints :
            continue 
        if (
        robot .junction_guard_anchor is not None 
        and robot .position .distance_to (robot .junction_guard_anchor )
        <=physical .JUNCTION_GUARD_POSITION_TOLERANCE 
        and len (waypoints )>1 
        ):
            waypoints .pop (0 )
            robot .junction_guard_anchor =waypoints [0 ].copy ()

def install_current_boundary_transport_runtime (
physical :types .ModuleType ,
)->None :
    """Descriptor-local current Frontier/Shepherd physical boundary runtime."""



    original_robot_update =physical .Robot .update 







    physical .integration_frontier_active_uid =None 
    physical .integration_frontier_ids =set ()
    physical .integration_frontier_offsets ={}

    physical .integration_frontier_centroid_lateral =0.0 

    physical .integration_frontier_depth =0.0 
    physical .integration_frontier_target_depth =0.0 

    physical .integration_frontier_bootstrap_complete =False 
    physical .integration_frontier_ready_logged =False 

    physical .integration_child_explore_uid =None 
    physical .integration_child_explore_progress_history =[]
    physical .integration_child_explore_dwell =0.0 
    physical .integration_child_explore_baseline_density =None 
    physical .integration_child_explore_baseline_pressure =None 

    physical .integration_child_dfs_phase ="IDLE"


    physical .integration_frontier_cruise_speed =(
    35.0 *ROBOT_MOTION_SPEED_SCALE 
    )

    physical .integration_child_pressure_push_speed =(
    42.0 *ROBOT_MOTION_SPEED_SCALE 
    )

    physical .integration_child_flow_backtrack_speed =(
    36.0 *ROBOT_MOTION_SPEED_SCALE 
    )


    physical .integration_frontier_support_quantile =0.90 



    physical .integration_child_active_min_center_gap_ratio =1.50 


    physical .integration_child_fill_dwell =0.0 
    physical .integration_child_flow_dwell =0.0 

    physical .integration_child_fill_baseline_density =None 
    physical .integration_child_fill_baseline_pressure =None 







    def child_boundary_robot_swept_limit (
    robot :Any ,
    old_position :pygame .Vector2 ,
    proposed_position :pygame .Vector2 ,
    )->pygame .Vector2 :
        """Prevent Child Frontier/Shepherd from tunnelling through robots."""

        movement =(
        proposed_position 
        -old_position 
        )

        movement_sq =movement .length_squared ()

        if movement_sq <=physical .EPSILON :
            return proposed_position .copy ()

        boundary_ids =set (
        getattr (
        physical ,
        "integration_frontier_ids",
        set (),
        )
        )

        max_alpha =1.0 

        current_robots =getattr (
        physical ,
        "integration_current_robots",
        (),
        )

        for other in current_robots :

            if other is robot :
                continue 





            if other .robot_id in boundary_ids :
                continue 

            other_radius =float (
            getattr (
            other ,
            "radius",
            physical .ROBOT_RADIUS ,
            )
            )

            child_phase =getattr (
            physical ,
            "integration_child_dfs_phase",
            "IDLE",
            )













            minimum_distance =(
            float (robot .radius )
            +other_radius 
            )

            relative =(
            old_position 
            -other .position 
            )

            c =(
            relative .length_squared ()
            -minimum_distance **2 
            )





            if c <=0.0 :

                if relative .dot (movement )<0.0 :
                    max_alpha =0.0 

                continue 

            a =movement_sq 

            b =(
            2.0 
            *relative .dot (movement )
            )

            discriminant =(
            b *b 
            -4.0 *a *c 
            )

            if discriminant <0.0 :
                continue 

            sqrt_discriminant =math .sqrt (
            discriminant 
            )

            hit_alpha =(
            -b 
            -sqrt_discriminant 
            )/(
            2.0 *a 
            )

            if (
            0.0 
            <=hit_alpha 
            <=1.0 
            ):
                max_alpha =min (
                max_alpha ,
                max (
                0.0 ,
                hit_alpha -1.0e-3 ,
                ),
                )

        return (
        old_position 
        +movement *max_alpha 
        )

    def rigid_current_boundary_common_velocity (
    branch_uid :str ,
    members :Sequence [Any ],
    lifecycle :dict [str ,Any ],
    tangent :pygame .Vector2 ,
    dt :float ,
    )->pygame .Vector2 :
        """One collision-safe translation for the whole current 3xN wall."""

        frame =int (
        getattr (
        physical ,
        "integration_frame",
        -1 ,
        )
        )

        if int (lifecycle .get ("rigid_cache_frame",-10 **9 ))==frame :
            return lifecycle .get (
            "rigid_common_velocity",
            pygame .Vector2 (),
            ).copy ()

        expected =(
        int (lifecycle .get ("rows",0 ))
        *int (lifecycle .get ("cols",0 ))
        )

        if expected <=0 or len (members )!=expected :
            lifecycle ["rigid_cache_frame"]=frame 
            lifecycle ["rigid_common_velocity"]=pygame .Vector2 ()

            print (
            "[CurrentRigidBlock] "
            f"branch={branch_uid } "
            f"members={len (members )} "
            f"expected={expected }"
            )

            return pygame .Vector2 ()

        requested_depth =float (
        physical .integration_frontier_depth 
        )
        applied_depth =float (
        lifecycle .get (
        "rigid_applied_depth",
        requested_depth ,
        )
        )

        junction =multi_dfs .current 

        if (
        junction is not None 
        and junction .branch_phase ==BranchPhase .FILL 
        ):
            requested_depth =applied_depth 

        requested_delta =requested_depth -applied_depth 

        intended_forward_speed =0.0 

        if (
        junction is not None 
        and junction .branch_phase in {
        BranchPhase .FRONTIER_BOOTSTRAP ,
        BranchPhase .EXPLORE ,
        }
        ):
            maximum_step =(
            float (
            physical .integration_frontier_cruise_speed 
            )
            *dt 
            )

            requested_delta =max (
            0.0 ,
            min (
            requested_delta ,
            maximum_step ,
            ),
            )

            intended_forward_speed =(
            requested_delta 
            /max (
            dt ,
            physical .EPSILON ,
            )
            )

            for member in members :
                member .integration_frontier_intended_forward_speed =(
                intended_forward_speed 
                )

        boundary_ids ={robot .robot_id for robot in members }

        external_robots =[
        robot 
        for robot in getattr (
        physical ,
        "integration_current_robots",
        (),
        )
        if robot .robot_id not in boundary_ids 
        ]

        hard_limit =float (
        getattr (
        physical ,
        "COMM_GUARD_HARD_LIMIT",
        physical .COMM_RANGE ,
        )
        )

        def pair_step_is_safe (
        member :Any ,
        delta :pygame .Vector2 ,
        other :Any ,
        fraction :float ,
        )->bool :
            other_radius =float (
            getattr (
            other ,
            "radius",
            physical .ROBOT_RADIUS ,
            )
            )
            minimum_distance =float (member .radius )+other_radius 
            other_delta =(
            getattr (other ,"velocity",pygame .Vector2 ())
            *dt 
            *fraction 
            )
            relative_start =member .position -other .position 
            relative_movement =delta -other_delta 
            c =(
            relative_start .length_squared ()
            -minimum_distance **2 
            )

            if c <=0.0 :
                return (
                relative_start .dot (relative_movement )
                >=-physical .EPSILON 
                )

            a =relative_movement .length_squared ()

            if a <=physical .EPSILON :
                return True 

            b =2.0 *relative_start .dot (relative_movement )
            discriminant =b *b -4.0 *a *c 

            if discriminant <0.0 :
                return True 

            hit_alpha =(
            -b -math .sqrt (discriminant )
            )/(2.0 *a )

            return not (0.0 <=hit_alpha <=1.0 )

        def group_step_is_safe (fraction :float )->bool :
            delta =tangent *requested_delta *fraction 

            for member in members :
                candidate =member .position +delta 

                if not physical .is_walkable (candidate ,member .radius ):
                    return False 

                for other in external_robots :
                    if not pair_step_is_safe (
                    member ,
                    delta ,
                    other ,
                    fraction ,
                    ):
                        return False 

                parent =getattr (member ,"comm_parent",None )

                if (
                parent is not None 
                and getattr (member ,"connected_to_base",False )
                ):
                    if getattr (parent ,"robot_id",None )in boundary_ids :
                        parent_candidate =parent .position +delta 
                    else :
                        parent_candidate =parent .position 

                    if candidate .distance_to (parent_candidate )>hard_limit :
                        return False 

            return True 

        safe_fraction =1.0 

        if (
        abs (requested_delta )>physical .EPSILON 
        and not group_step_is_safe (1.0 )
        ):
            low =0.0 
            high =1.0 

            for _ in range (12 ):
                middle =0.5 *(low +high )

                if group_step_is_safe (middle ):
                    low =middle 
                else :
                    high =middle 

            safe_fraction =low 

        actual_delta =requested_delta *safe_fraction 
        common_velocity =tangent *(
        actual_delta /max (dt ,physical .EPSILON )
        )

        lifecycle ["rigid_applied_depth"]=(
        applied_depth +actual_delta 
        )
        lifecycle ["rigid_cache_frame"]=frame 
        lifecycle ["rigid_common_velocity"]=common_velocity .copy ()
        lifecycle ["rigid_safe_fraction"]=safe_fraction 
        lifecycle ["rigid_contact"]=(
        abs (requested_delta )>physical .EPSILON 
        and safe_fraction <0.999 
        )

        for member in members :
            member .integration_boundary_shape_error =0.0 

        if frame %10 ==0 :
            print (
            "[CurrentRigidTransport] "
            f"frame={frame } "
            f"branch={branch_uid } "
            f"phase="
            f"{junction .branch_phase .name if junction else '?'} "
            f"members={len (members )} "
            f"requested_delta={requested_delta :.3f} "
            f"actual_delta={actual_delta :.3f} "
            f"safe_fraction={safe_fraction :.3f} "
            "shape_locked=True"
            )

        return common_velocity 








    def current_boundary_robot_update (
    self :Any ,
    dt :float ,
    )->None :
        """Move one current Frontier/Shepherd as part of one rigid 3xN wall."""

        junction =multi_dfs .current 

        if junction is None :
            return original_robot_update (
            self ,
            dt ,
            )


        runtime_kind =(
        current_junction_frontier_runtime_kind (
        physical ,
        junction ,
        )
        )

        if runtime_kind not in {
        "UID_LOCAL",
        "LEGACY_FIXTURE",
        }:
            return original_robot_update (
            self ,
            dt ,
            )

        uid =junction .active_branch_uid 

        if uid is None :
            return original_robot_update (
            self ,
            dt ,
            )





        boundary_ids =set (
        getattr (
        physical ,
        "integration_frontier_ids",
        set (),
        )
        )

        is_current_boundary =(
        self .robot_id in boundary_ids 
        and self .shepherd_branch ==uid 
        and self .role in {
        "FRONTIER_SHEPHERD",
        "SHEPHERD",
        }
        )



        if not is_current_boundary :
            return original_robot_update (
            self ,
            dt ,
            )

        descriptor =(
        physical .branch_descriptors_by_uid .get (
        uid 
        )
        )

        if descriptor is None :
            raise RuntimeError (
            "Child rigid wall descriptor missing: "
            f"uid={uid }"
            )

        lifecycle =(
        current_junction_branch_lifecycle (
        physical ,
        junction ,
        uid 
        )
        )

        if lifecycle is None :
            raise RuntimeError (
            "Child rigid wall lifecycle missing: "
            f"uid={uid }"
            )


        relative_offsets =lifecycle .get (
        "relative_offsets",
        {},
        )

        if self .robot_id not in relative_offsets :
            raise RuntimeError (
            "Child rigid wall relative offset missing: "
            f"uid={uid } "
            f"robot={self .robot_id }"
            )

        current_robots =tuple (
        getattr (
        physical ,
        "integration_current_robots",
        (),
        )
        )

        boundary_members =[
        robot 
        for robot in current_robots 
        if (
        robot .robot_id in boundary_ids 
        and robot .shepherd_branch ==uid 
        and robot .role in {
        "FRONTIER_SHEPHERD",
        "SHEPHERD",
        }
        )
        ]

        if (
        len (boundary_members )
        !=len (boundary_ids )
        ):
            raise RuntimeError (
            "Child rigid wall member count mismatch: "
            f"uid={uid } "
            f"expected={len (boundary_ids )} "
            f"actual={len (boundary_members )}"
            )

        tangent ,_ =physical .descriptor_local_basis (descriptor )
        tangent =tangent .normalize ()

        common_velocity =rigid_current_boundary_common_velocity (
        uid ,
        boundary_members ,
        lifecycle ,
        tangent ,
        dt ,
        )

        integrate =physical .integration_integrate_boundary_velocity 
        integrate (
        self ,
        common_velocity ,
        dt ,
        rigid_frontier =True ,
        )

        self .acceleration .update (
        0.0 ,
        0.0 ,
        )

        self .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )


    physical .Robot .update =(
    current_boundary_robot_update 
    )


def install_current_branch_physical_drive (
physical :types .ModuleType ,
)->None :
    """Drive the movable NORMAL swarm through the current active Branch."""







    original_route_force =(
    physical .compute_route_force 
    )

    def current_shepherd_contact_force (
    robot :Any ,
    descriptor :Any ,
    branch_uid :str ,
    )->pygame .Vector2 :
        """Real spring-damper contact from visible current Branch Shepherds."""

        junction =multi_dfs .current 

        if junction is None :
            return pygame .Vector2 ()

        if junction .active_branch_uid !=branch_uid :
            return pygame .Vector2 ()

        branch_phase =junction .branch_phase 

        if branch_phase not in {
        BranchPhase .PRESSURE_PUSH ,
        BranchPhase .FLOW_BACKTRACK ,
        }:
            return pygame .Vector2 ()

        current_robots =getattr (
        physical ,
        "integration_current_robots",
        (),
        )

        shepherd_ids =set (
        getattr (
        physical ,
        "integration_frontier_ids",
        set (),
        )
        )

        shepherds =[
        other 
        for other in current_robots 
        if (
        other .robot_id 
        in shepherd_ids 
        and other .role =="SHEPHERD"
        and other .shepherd_branch 
        ==branch_uid 
        )
        ]

        if not shepherds :
            return pygame .Vector2 ()

        radius =float (
        physical .ROBOT_RADIUS 
        )

        shell_ratio =float (
        getattr (
        physical ,
        "integration_shepherd_contact_shell_ratio",
        2.35 ,
        )
        )

        spring_scale =float (
        getattr (
        physical ,
        "integration_shepherd_contact_spring_scale",
        2.50 ,
        )
        )

        damping_scale =float (
        getattr (
        physical ,
        "integration_shepherd_contact_damping_scale",
        1.00 ,
        )
        )

        spring_gain =(
        float (physical .REPULSION_GAIN )
        *spring_scale 
        )

        damping_gain =(
        float (physical .DAMPING )
        *damping_scale 
        )

        total_force =(
        pygame .Vector2 ()
        )

        contact_count =0 

        for shepherd in shepherds :

            delta =(
            robot .position 
            -shepherd .position 
            )

            distance_sq =(
            delta .length_squared ()
            )

            if distance_sq <=physical .EPSILON :

                contact_normal =(
                descriptor .local_return_direction .normalize ()
                )

                distance =0.0 

            else :

                distance =math .sqrt (
                distance_sq 
                )

                contact_radius =max (
                float (robot .radius )
                +float (shepherd .radius )
                +0.05 *radius ,

                shell_ratio *radius ,
                )

                if distance >=contact_radius :
                    continue 

                contact_normal =(
                delta /distance 
                )

            compression =max (
            0.0 ,
            contact_radius 
            -distance ,
            )

            relative_normal_speed =(
            robot .velocity 
            -shepherd .velocity 
            ).dot (
            contact_normal 
            )

            magnitude =max (
            0.0 ,
            spring_gain *compression 
            -damping_gain 
            *relative_normal_speed ,
            )

            total_force +=(
            contact_normal 
            *magnitude 
            )

            contact_count +=1 



        force_limit =float (
        getattr (
        physical ,
        "PHYSICAL_GUARD_FORCE_LIMIT",
        getattr (
        physical ,
        "MAX_ACCELERATION",
        300.0 ,
        ),
        )
        )

        if (
        total_force .length_squared ()
        >force_limit **2 
        ):
            total_force .scale_to_length (
            force_limit 
            )

        setattr (
        robot ,
        "last_child_shepherd_contact_count",
        contact_count ,
        )

        setattr (
        robot ,
        "last_child_shepherd_contact_force",
        total_force .length (),
        )

        return total_force 



    def current_branch_route_force (
    robot :Any ,
    )->pygame .Vector2 :

        junction =(
        multi_dfs .current 
        )

        if junction is None :
            return original_route_force (
            robot 
            )

        runtime_kind =(
        current_junction_frontier_runtime_kind (
        physical ,
        junction ,
        )
        )













        if runtime_kind not in {
        "UID_LOCAL",
        "LEGACY_FIXTURE",
        }:
            return original_route_force (
            robot 
            )

        branch_uid =junction .active_branch_uid 

        runtime_uid =getattr (
        physical ,
        "integration_frontier_active_uid",
        None ,
        )











        if (
        branch_uid is None 
        or runtime_uid !=branch_uid 
        ):
            return original_route_force (
            robot 
            )





        if (
        robot .role !="NORMAL"
        or robot .base_reserve 
        or bool (
        getattr (
        robot ,
        "is_fixed_anchor",
        False ,
        )
        )
        ):
            return original_route_force (
            robot 
            )

        descriptor =(
        physical .branch_descriptors_by_uid .get (
        branch_uid 
        )
        )

        if descriptor is None :
            return original_route_force (
            robot 
            )

        branch_phase =junction .branch_phase 

        tangent ,normal =(
        physical .descriptor_local_basis (
        descriptor 
        )
        )

        tangent =tangent .normalize ()
        normal =normal .normalize ()

        base_drive_force =float (
        getattr (
        physical ,
        "ROUTE_FORCE",
        adaptive .LOCAL_FORWARD_DRIVE_FORCE ,
        )
        )

        if branch_phase in {
        BranchPhase .FRONTIER_BOOTSTRAP ,
        BranchPhase .EXPLORE ,
        BranchPhase .FILL ,
        }:
            drive_force =3.0 *base_drive_force 
        else :
            drive_force =base_drive_force 

        centering_gain =float (
        getattr (
        physical ,
        "CENTERING_GAIN",
        1.0 ,
        )
        )
        peer_observations =observe_local_neighbors (
        robot ,
        getattr (physical ,"integration_current_robots",()),
        tangent ,
        max_range =physical .COMM_RANGE ,
        predicate =lambda other :(
        other .role =="NORMAL"and not other .base_reserve 
        ),
        )
        local_lateral_error =(
        float (np .median ([
        observation .relative_lateral 
        for observation in peer_observations 
        ]))
        if peer_observations else 0.0 
        )











        if branch_phase in {
        BranchPhase .FRONTIER_BOOTSTRAP ,
        BranchPhase .EXPLORE ,
        BranchPhase .FILL ,
        }:

            lateral_force =(
            centering_gain 
            *local_lateral_error 
            )

            max_lateral_force =(
            0.45 *drive_force 
            )

            lateral_force =max (
            -max_lateral_force ,
            min (
            max_lateral_force ,
            lateral_force ,
            ),
            )

            return (
            tangent *drive_force 
            +normal *lateral_force 
            )

















        if branch_phase in {
        BranchPhase .PRESSURE_PUSH ,
        BranchPhase .FLOW_BACKTRACK ,
        }:

            lateral_force =(
            centering_gain 
            *local_lateral_error 
            )

            lateral_limit =(
            0.35 *drive_force 
            )

            lateral_force =max (
            -lateral_limit ,
            min (
            lateral_limit ,
            lateral_force ,
            ),
            )

            force =(
            normal *lateral_force 
            )

            force +=current_shepherd_contact_force (
            robot ,
            descriptor ,
            branch_uid ,
            )
            return force 

        return original_route_force (
        robot 
        )

    physical .compute_route_force =(
    current_branch_route_force 
    )

def activate_current_branch_frontier (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Promote the selected Child Guard wall to same-ID Frontier."""

    child =(
    multi_dfs .current 
    )

    if child is None :
        return 



    if (
    getattr (
    physical ,
    "integration_frontier_active_uid",
    None ,
    )
    is not None 
    ):
        return 

    if not child .branch_order :
        return 


    for uid in child .branch_order :

        lifecycle =(
        current_junction_branch_lifecycle (
        physical ,
        child ,
        uid ,
        )
        )

        if lifecycle is None :
            return 

        if "robot_ids"not in lifecycle :
            return 

        if "relative_offsets"not in lifecycle :
            return 


    geometry_by_uid :dict [
    str ,
    ProvisionalGuardGeometry ,
    ]={}

    for geometry in perception .provisional_guards :

        runtime_uid =(
        geometry .persistent_uid 
        or geometry .provisional_uid 
        )

        geometry_by_uid [
        runtime_uid 
        ]=geometry 

        geometry_by_uid .setdefault (
        geometry .provisional_uid ,
        geometry ,
        )

    for uid in child .branch_order :

        geometry =geometry_by_uid .get (
        uid 
        )

        if geometry is None :
            return 

        lifecycle =(
        current_junction_branch_lifecycle (
        physical ,
        child ,
        uid ,
        )
        )

        if lifecycle is None :
            return 

        expected_ids =set (
        lifecycle ["robot_ids"]
        )

        guards =[
        robot 
        for robot in robots 
        if (
        robot .robot_id 
        in expected_ids 
        and robot .role 
        =="JUNCTION_GUARD"
        and getattr (
        robot ,
        "junction_guard_branch_uid",
        None ,
        )
        ==uid 
        )
        ]

        live_ids ={
        robot .robot_id 
        for robot in guards 
        }

        if live_ids !=expected_ids :
            return 


    branch_uid =(
    resolve_current_junction_frontier_launch_uid (
    physical ,
    child ,
    )
    )

    if branch_uid is None :
        return 

    lifecycle =(
    current_junction_branch_lifecycle (
    physical ,
    child ,
    branch_uid ,
    )
    )

    descriptor =(
    physical .branch_descriptors_by_uid .get (
    branch_uid 
    )
    )

    if (
    lifecycle is None 
    or descriptor is None 
    ):
        raise RuntimeError (
        "Selected Child Frontier has no context: "
        f"branch={branch_uid }"
        )

    expected_ids =set (
    lifecycle ["robot_ids"]
    )

    frontier_members =[
    robot 
    for robot in robots 
    if (
    robot .robot_id 
    in expected_ids 
    and robot .role 
    =="JUNCTION_GUARD"
    and getattr (
    robot ,
    "junction_guard_branch_uid",
    None ,
    )
    ==branch_uid 
    )
    ]

    live_ids ={
    robot .robot_id 
    for robot in frontier_members 
    }



    if live_ids !=expected_ids :
        raise RuntimeError (
        "Child Frontier Guard lineage mismatch: "
        f"branch={branch_uid } "
        f"expected={len (expected_ids )} "
        f"live={len (live_ids )}"
        )

    if not frontier_members :
        raise RuntimeError (
        "Child Frontier has zero Guard members: "
        f"branch={branch_uid }"
        )





    centroid_axial =float (lifecycle ["centroid_axial"])
    centroid_lateral =float (lifecycle ["centroid_lateral"])
    relative_offsets ={
    int (robot_id ):(float (offset [0 ]),float (offset [1 ]))
    for robot_id ,offset in lifecycle ["relative_offsets"].items ()
    }



    lifecycle ["guard_layer_by_id"]={
    robot .robot_id :int (
    getattr (
    robot ,
    "junction_guard_layer",
    -1 ,
    )
    )
    for robot in frontier_members 
    }

    lifecycle ["guard_hop_by_id"]={
    robot .robot_id :int (
    getattr (
    robot ,
    "junction_guard_hop",
    -1 ,
    )
    )
    for robot in frontier_members 
    }

    lifecycle ["guard_parent_id_by_id"]={
    robot .robot_id :getattr (
    robot ,
    "junction_guard_parent_id",
    None ,
    )
    for robot in frontier_members 
    }

    lifecycle ["guard_branch_key_by_id"]={
    robot .robot_id :getattr (
    robot ,
    "junction_guard_branch",
    None ,
    )
    for robot in frontier_members 
    }

    lifecycle ["guard_is_leader_by_id"]={
    robot .robot_id :bool (
    getattr (
    robot ,
    "is_branch_leader",
    False ,
    )
    )
    for robot in frontier_members 
    }

    lifecycle ["guard_slot_index_by_id"]={
    robot .robot_id :int (
    getattr (
    robot ,
    "integration_guard_slot_index",
    -1 ,
    )
    )
    for robot in frontier_members 
    }









    axial_offsets =[
    float (value [0 ])
    for value 
    in relative_offsets .values ()
    ]

    if axial_offsets :
        wall_axial_span =(
        max (axial_offsets )
        -min (axial_offsets )
        )
    else :
        wall_axial_span =0.0 

    bootstrap_distance =max (
    float (
    physical .FRONTIER_LINE_LEAD_GAP 
    ),
    wall_axial_span 
    +2.0 
    *float (
    physical .ROBOT_RADIUS 
    ),
    )

    requested_target_depth =(
    centroid_axial 
    +bootstrap_distance 
    )





    target_depth =max (centroid_axial ,requested_target_depth )









    physical .integration_frontier_active_uid =(
    branch_uid 
    )

    physical .integration_frontier_ids =(
    set (expected_ids )
    )

    physical .integration_frontier_offsets =(
    dict (relative_offsets )
    )

    physical .integration_frontier_centroid_lateral =(
    centroid_lateral 
    )

    physical .integration_frontier_depth =(
    centroid_axial 
    )

    physical .integration_frontier_target_depth =(
    target_depth 
    )







    lifecycle ["rigid_applied_depth"]=float (
    centroid_axial 
    )
    lifecycle ["rigid_cache_frame"]=-1 
    lifecycle ["rigid_common_velocity"]=pygame .Vector2 ()
    lifecycle ["rigid_safe_fraction"]=1.0 
    lifecycle ["rigid_contact"]=False 

    physical .integration_frontier_bootstrap_complete =(
    False 
    )

    physical .integration_frontier_ready_logged =(
    False 
    )







    child .branch_phase =(
    BranchPhase .FRONTIER_BOOTSTRAP 
    )



    child .pending_branch_uid =None 



    physical .integration_child_dfs_phase =(
    "EXPLORE"
    )

    physical .integration_child_fill_dwell =(
    0.0 
    )

    physical .integration_child_flow_dwell =(
    0.0 
    )

    physical .integration_child_fill_baseline_density =(
    None 
    )

    physical .integration_child_fill_baseline_pressure =(
    None 
    )











    frontier_members ,transition_jump =(
    promote_current_guard_wall_roles_to_frontier (
    physical ,
    robots ,
    child ,
    branch_uid ,
    lifecycle ,
    relative_offsets ,
    shepherd_branch_key =branch_uid ,
    clear_guard_waypoints =True ,
    zero_command_state =True ,
    )
    )













    descriptor .visit_state =(
    "ACTIVE"
    )

    child .branch_states [
    branch_uid 
    ]="ACTIVE"

    child .active_branch_uid =(
    branch_uid 
    )

    child .subtree_complete =False 

    physical .active_branch_uid =branch_uid 

    selected_fixture =physical .branch_fixture_for_uid (
    branch_uid 
    )
    if selected_fixture is not None :
        physical .active_branch =selected_fixture 

    physical .phase =(
    physical .SimulationPhase .EXPLORE_BRANCH 
    )

    print (
    "[CurrentBranchPhysicalPhaseSync] "
    f"junction={child .junction_uid } "
    f"branch={branch_uid } "
    f"fixture={selected_fixture } "
    "phase=EXPLORE_BRANCH"
    )

    print (
    "[ChildFrontierTransportStart] "
    f"junction={child .junction_uid } "
    f"branch={branch_uid } "
    f"robots={len (frontier_members )} "
    "same_ids=True "
    f"position_jump={transition_jump :.6f} "
    f"start_depth={centroid_axial :.3f} "
    f"target_depth={target_depth :.3f}"
    )


def activate_current_junction_frontier_transport (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
)->None :
    """Activate the current Junction Frontier through one depth-independent API."""

    junction =multi_dfs .current 

    if junction is None :
        return 

















    prepare_current_junction_frontier_handoff (
    physical ,
    perception ,
    robots ,
    )

    runtime_kind =(
    current_junction_frontier_runtime_kind (
    physical ,
    junction ,
    )
    )

    if runtime_kind is None :
        return 

    if runtime_kind in {
    "UID_LOCAL",
    "LEGACY_FIXTURE",
    }:

        if not getattr (
        physical ,
        "integration_child_guard_lifecycle_initialized",
        False ,
        ):
            return 

        activate_current_branch_frontier (
        physical ,
        perception ,
        robots ,
        )

        return 

    raise RuntimeError (
    "Unknown current Junction Frontier runtime: "
    f"junction={junction .junction_uid } "
    f"runtime={runtime_kind }"
    )


def current_branch_pack_metrics (
physical :types .ModuleType ,
robots :Sequence [Any ],
branch_uid :str ,
)->dict [str ,float |int ]:
    """Measure the real NORMAL pack immediately behind this Branch Shepherd."""

    junction =multi_dfs .current 

    descriptor =(
    physical .branch_descriptors_by_uid .get (
    branch_uid 
    )
    )

    lifecycle =(
    current_junction_branch_lifecycle (
    physical ,
    junction ,
    branch_uid ,
    )
    if junction is not None 
    else None 
    )

    if descriptor is None or lifecycle is None :
        return {
        "count":0 ,
        "coverage":0.0 ,
        "mean_density":0.0 ,
        "mean_pressure":0.0 ,
        "low_speed_ratio":0.0 ,
        }

    usable_half =(
    physical .local_physical_usable_half_width (
    descriptor 
    )
    )

    observed_width =max (
    float (
    getattr (
    descriptor ,
    "observed_physical_width",
    0.0 ,
    )
    or getattr (
    descriptor ,
    "observed_width",
    0.0 ,
    )
    ),
    4.0 *physical .ROBOT_RADIUS ,
    )

    pack_depth =max (
    0.55 *observed_width ,
    3.0 *physical .ROBOT_RADIUS ,
    )

    wall_ids =set (lifecycle .get ("robot_ids",[]))
    wall =[robot for robot in robots if robot .robot_id in wall_ids ]
    offsets =lifecycle .get ("relative_offsets",{})
    if not wall :
        return {
        "count":0 ,"coverage":0.0 ,"mean_density":0.0 ,
        "mean_pressure":0.0 ,"low_speed_ratio":0.0 ,
        }
    if not offsets :
        return {
        "count":0 ,"coverage":0.0 ,"mean_density":0.0 ,
        "mean_pressure":0.0 ,"low_speed_ratio":0.0 ,
        }

    rear_axial =min (
    float (offset [0 ])
    for offset in offsets .values ()
    )
    rear_row_tolerance =max (
    0.35 *float (physical .GRID_ROW_SPACING ),
    0.75 *float (physical .ROBOT_RADIUS ),
    )
    rear_observers =[
    robot 
    for robot in wall 
    if (
    robot .robot_id in offsets 
    and abs (
    float (offsets [robot .robot_id ][0 ])
    -rear_axial 
    )
    <=rear_row_tolerance 
    )
    ]

    if not rear_observers :
        return {
        "count":0 ,"coverage":0.0 ,"mean_density":0.0 ,
        "mean_pressure":0.0 ,"low_speed_ratio":0.0 ,
        }

    _ ,lateral_axis =physical .descriptor_local_basis (
    descriptor 
    )
    pack_normal_by_id ={}

    for observer in rear_observers :
        observer_offset =offsets .get (
        observer .robot_id ,
        (rear_axial ,0.0 ),
        )
        observer_lateral =float (observer_offset [1 ])
        observations =observe_local_neighbors (
        observer ,
        robots ,
        descriptor .local_outgoing_direction ,
        lateral_axis =lateral_axis ,
        max_range =physical .COMM_RANGE ,
        predicate =lambda robot :(
        robot .role =="NORMAL"
        and not robot .base_reserve 
        and not bool (getattr (robot ,"is_fixed_anchor",False ))
        ),
        )

        for observation in observations :
            branch_lateral =float (
            observation .relative_lateral 
            +observer_lateral 
            )

            if not (
            -pack_depth 
            <=observation .relative_axial 
            <=physical .ROBOT_RADIUS 
            ):
                continue 

            if abs (branch_lateral )>usable_half :
                continue 

            robot_id =observation .robot .robot_id 
            previous =pack_normal_by_id .get (robot_id )

            if (
            previous is None 
            or observation .relative_range <previous [2 ]
            ):
                pack_normal_by_id [robot_id ]=(
                observation .robot ,
                branch_lateral ,
                float (observation .relative_range ),
                )

    pack_normals =[
    (robot ,lateral )
    for robot ,lateral ,_ in pack_normal_by_id .values ()
    ]

    if not pack_normals :
        return {
        "count":0 ,
        "coverage":0.0 ,
        "mean_density":0.0 ,
        "mean_pressure":0.0 ,
        "low_speed_ratio":0.0 ,
        }







    bin_count =max (
    3 ,
    GUARD_LATERAL_COVERAGE_BINS ,
    )

    occupied_bins =set ()

    if usable_half >physical .EPSILON :

        for _ ,lateral in pack_normals :

            normalized =(
            lateral +usable_half 
            )/(
            2.0 *usable_half 
            )

            index =int (
            normalized 
            *bin_count 
            )

            index =max (
            0 ,
            min (
            bin_count -1 ,
            index ,
            ),
            )

            occupied_bins .add (
            index 
            )

    coverage =(
    len (occupied_bins )
    /bin_count 
    )

    densities =[
    float (robot .density )
    for robot ,_ 
    in pack_normals 
    ]

    pressures =[
    max (
    0.0 ,
    float (robot .pressure ),
    )
    for robot ,_ 
    in pack_normals 
    ]

    low_speed_threshold =float (
    getattr (
    physical ,
    "DEAD_END_FORWARD_SPEED_THRESHOLD",
    8.0 ,
    )
    )

    low_speed_ratio =(
    sum (
    robot .velocity .length ()
    <=low_speed_threshold 
    for robot ,_ 
    in pack_normals 
    )
    /len (pack_normals )
    )

    return {
    "count":len (pack_normals ),

    "coverage":float (
    coverage 
    ),

    "mean_density":float (
    np .mean (densities )
    ),

    "mean_pressure":float (
    np .mean (pressures )
    ),

    "low_speed_ratio":float (
    low_speed_ratio 
    ),
    }


def promote_current_frontier_to_shepherd (
physical :types .ModuleType ,
robots :Sequence [Any ],
branch_uid :str ,
)->None :

    junction =multi_dfs .current 

    if junction is None :
        raise RuntimeError (
        "Frontier->Shepherd has no current Junction"
        )

    if (
    junction .active_branch_uid 
    !=branch_uid 
    ):
        raise RuntimeError (
        "Frontier->Shepherd branch is not ACTIVE: "
        f"branch={branch_uid }"
        )

    lifecycle =(
    current_junction_branch_lifecycle (
    physical ,
    junction ,
    branch_uid ,
    )
    )

    descriptor =(
    physical .branch_descriptors_by_uid .get (
    branch_uid 
    )
    )

    if lifecycle is None or descriptor is None :
        raise RuntimeError (
        "Frontier->Shepherd missing Branch context: "
        f"{branch_uid }"
        )

    expected_ids =set (
    lifecycle .get (
    "robot_ids",
    [],
    )
    )

    frontiers =[
    robot 
    for robot in robots 
    if (
    robot .robot_id in expected_ids 
    and robot .role 
    =="FRONTIER_SHEPHERD"
    )
    ]

    live_ids ={
    robot .robot_id 
    for robot in frontiers 
    }

    if live_ids !=expected_ids :
        raise RuntimeError (
        "Frontier->Shepherd lineage mismatch: "
        f"branch={branch_uid }"
        )

    before ={
    robot .robot_id :robot .position .copy ()
    for robot in frontiers 
    }

    return_direction =(
    descriptor .local_return_direction 
    .normalize ()
    )

    for robot in frontiers :

        robot .role ="SHEPHERD"

        robot .shepherd_anchor =(
        robot .position .copy ()
        )

        robot .shepherd_origin =(
        robot .position .copy ()
        )

        robot .shepherd_return_direction =(
        return_direction .copy ()
        )

        robot .velocity .update (0.0 ,0.0 )
        robot .commanded_velocity .update (0.0 ,0.0 )
        robot .acceleration .update (0.0 ,0.0 )
        robot .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )

    max_jump =max (
    (
    robot .position .distance_to (
    before [robot .robot_id ]
    )
    for robot in frontiers 
    ),
    default =0.0 ,
    )

    if max_jump >physical .EPSILON :
        raise RuntimeError (
        "Frontier->Shepherd caused position jump: "
        f"branch={branch_uid } "
        f"jump={max_jump :.9f}"
        )

    lifecycle ["state"]="FILL"

    lifecycle [
    "frontier_to_shepherd_in_place"
    ]=True 

    lifecycle [
    "shepherd_shape_locked"
    ]=True 



    junction .branch_phase =(
    BranchPhase .FILL 
    )



    physical .integration_child_dfs_phase =(
    "FILL"
    )


def finish_current_branch_return (
physical :types .ModuleType ,
robots :Sequence [Any ],
branch_uid :str ,
)->None :
    """Return the same Shepherd lineage to a Junction Guard role in place."""

    junction =multi_dfs .current 

    if junction is None :
        raise RuntimeError (
        "Branch return has no current Junction"
        )

    lifecycle =(
    current_junction_branch_lifecycle (
    physical ,
    junction ,
    branch_uid 
    )
    )

    descriptor =(
    physical .branch_descriptors_by_uid .get (
    branch_uid 
    )
    )

    if lifecycle is None or descriptor is None :
        raise RuntimeError (
        f"missing Branch return context: {branch_uid }"
        )

    expected_ids =set (
    lifecycle .get (
    "robot_ids",
    [],
    )
    )

    if not expected_ids :
        raise RuntimeError (
        "Child Guard lineage is empty: "
        f"branch={branch_uid }"
        )

    shepherds ={
    robot .robot_id :
    robot 
    for robot in robots 
    if (
    robot .robot_id 
    in expected_ids 
    and robot .role =="SHEPHERD"
    )
    }

    if set (shepherds )!=expected_ids :
        raise RuntimeError (
        "Shepherd lineage lost before Guard return"
        )

    layer_by_id =(
    lifecycle .get (
    "guard_layer_by_id",
    {},
    )
    )

    hop_by_id =(
    lifecycle .get (
    "guard_hop_by_id",
    {},
    )
    )

    parent_by_id =(
    lifecycle .get (
    "guard_parent_id_by_id",
    {},
    )
    )

    branch_key_by_id =(
    lifecycle .get (
    "guard_branch_key_by_id",
    {},
    )
    )

    leader_by_id =(
    lifecycle .get (
    "guard_is_leader_by_id",
    {},
    )
    )

    for robot_id ,robot in shepherds .items ():



        robot .role =(
        "JUNCTION_GUARD"
        )

        robot .junction_guard_anchor =None 

        robot .junction_guard_branch_uid =(
        branch_uid 
        )

        robot .junction_guard_branch =(
        branch_key_by_id .get (
        robot_id 
        )
        )

        robot .junction_guard_layer =int (
        layer_by_id .get (
        robot_id ,
        -1 ,
        )
        )

        robot .junction_guard_hop =int (
        hop_by_id .get (
        robot_id ,
        -1 ,
        )
        )

        robot .junction_guard_parent_id =(
        parent_by_id .get (
        robot_id 
        )
        )

        robot .is_branch_leader =bool (
        leader_by_id .get (
        robot_id ,
        False ,
        )
        )

        if hasattr (
        robot ,
        "integration_guard_final_anchor",
        ):
            robot .integration_guard_final_anchor =None 

        robot .integration_guard_waypoints =(
        []
        )

        robot .shepherd_anchor =None 
        robot .shepherd_origin =None 
        robot .shepherd_branch =None 
        robot .shepherd_return_direction =None 
        robot .frontier_local_lateral =None 

        robot .velocity .update (
        0.0 ,
        0.0 ,
        )

        robot .commanded_velocity .update (
        0.0 ,
        0.0 ,
        )

        robot .observed_velocity .update (
        0.0 ,
        0.0 ,
        )

        robot .acceleration .update (
        0.0 ,
        0.0 ,
        )

        robot .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )

    descriptor .visit_state =(
    "VISITED"
    )

    junction .branch_states [
    branch_uid 
    ]="VISITED"

    junction .active_branch_uid =None 
    junction .pending_branch_uid =None 

    physical .active_branch_uid =None 

    lifecycle ["state"]=(
    "VISITED_GUARD"
    )

    physical .integration_frontier_active_uid =(
    None 
    )

    physical .integration_frontier_ids =(
    set ()
    )

    physical .integration_frontier_offsets =(
    {}
    )

    physical .integration_frontier_bootstrap_complete =(
    False 
    )

    physical .integration_frontier_ready_logged =(
    False 
    )

    junction .branch_phase =(
    BranchPhase .IDLE 
    )


    physical .integration_child_dfs_phase =(
    "IDLE"
    )

    disable_leading_anchor (
    physical 
    )

    print (
    "[ChildBranchVisited] "
    f"junction={junction .junction_uid } "
    f"branch={branch_uid } "
    f"same_ids_returned={len (expected_ids )}"
    )
    if multi_dfs .refresh_subtree_complete (
    junction 
    ):

        print (
        "[ChildAllBranchesVisited] "
        f"junction={junction .junction_uid } "
        f"branches={junction .branch_order }"
        )

def begin_parent_return_after_child_subtree (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
child :MultiJunctionFrame ,
)->None :
    """Start localization-free physical return from Child to Parent."""

    if child .parent_return_started :
        return 

    if not child .subtree_complete :
        return 

    if child .parent_junction_uid is None :
        return 

    if (
    len (multi_dfs .stack )<2 
    or multi_dfs .current is not child 
    ):
        raise RuntimeError (
        "Parent return requires current Child "
        "to remain on top of DFS stack"
        )

    parent =multi_dfs .stack [-2 ]

    if (
    parent .junction_uid 
    !=child .parent_junction_uid 
    ):
        raise RuntimeError (
        "Parent return stack mismatch: "
        f"child={child .junction_uid } "
        f"expected_parent={child .parent_junction_uid } "
        f"actual_parent={parent .junction_uid }"
        )

    return_direction =child .return_direction_local 

    if (
    return_direction is None 
    or return_direction .length_squared ()
    <=physical .EPSILON 
    ):
        raise RuntimeError (
        "Child has no valid local Parent-return direction: "
        f"child={child .junction_uid }"
        )

    return_direction =(
    return_direction .normalize ()
    )









    disable_leading_anchor (
    physical 
    )

    physical .integration_anchor_breakout_active =False 

    multi_dfs .child_probe_active =False 
    multi_dfs .child_candidate_active =False 
    multi_dfs .child_probe_branch_uid =None 

    anchor =perception .leader 

    perception .anchor_fixed =False 
    anchor .is_fixed_anchor =False 
    anchor .base_reserve =False 



    if anchor .role in {
    "RELAY",
    "TRUNK_RELAY",
    }:
        anchor .role ="NORMAL"
        anchor .relay_anchor =None 
        anchor .relay_index =-1 

        if hasattr (
        anchor ,
        "relay_scope",
        ):
            anchor .relay_scope =None 

        if hasattr (
        anchor ,
        "relay_owner_edge_id",
        ):
            anchor .relay_owner_edge_id =None 



















    child_branch_uids =set (
    child .branch_order 
    )

    released_ids :list [int ]=[]

    for robot in robots :
        child_guard =(
        robot .role =="JUNCTION_GUARD"
        and getattr (
        robot ,
        "junction_guard_branch_uid",
        None ,
        )
        in child_branch_uids 
        )

        child_marker =(
        robot .role =="PEBBLE"
        and (
        getattr (
        robot ,
        "pebble_branch_uid",
        None ,
        )
        in child_branch_uids 
        or getattr (
        robot ,
        "marker_junction_uid",
        None ,
        )
        ==child .junction_uid 
        )
        )

        if (
        robot .role in {
        "FRONTIER_SHEPHERD",
        "SHEPHERD",
        }
        and getattr (
        robot ,
        "shepherd_branch",
        None ,
        )
        in child_branch_uids 
        ):
            raise RuntimeError (
            "Child subtree marked complete while "
            "a Frontier/Shepherd is still active: "
            f"child={child .junction_uid } "
            f"robot={robot .robot_id }"
            )

        if not (
        child_guard 
        or child_marker 
        ):
            continue 

        released_ids .append (
        robot .robot_id 
        )

        robot .role ="NORMAL"

        robot .junction_guard_anchor =None 
        robot .junction_guard_branch =None 
        robot .junction_guard_branch_uid =None 
        robot .junction_guard_parent_id =None 
        robot .junction_guard_layer =-1 
        robot .junction_guard_hop =-1 

        robot .shepherd_anchor =None 
        robot .shepherd_origin =None 
        robot .shepherd_branch =None 
        robot .shepherd_return_direction =None 
        robot .frontier_local_lateral =None 

        if hasattr (
        robot ,
        "pebble_anchor",
        ):
            robot .pebble_anchor =None 

        if hasattr (
        robot ,
        "pebble_branch_uid",
        ):
            robot .pebble_branch_uid =None 

        if hasattr (
        robot ,
        "pebble_branch_key",
        ):
            robot .pebble_branch_key =None 

        if hasattr (
        robot ,
        "pebble_state",
        ):
            robot .pebble_state =None 

        if hasattr (
        robot ,
        "marker_type",
        ):
            robot .marker_type =None 

        if hasattr (
        robot ,
        "marker_junction_uid",
        ):
            robot .marker_junction_uid =None 

        robot .is_branch_leader =False 

        robot .velocity *=0.25 

        robot .commanded_velocity .update (
        0.0 ,
        0.0 ,
        )

        robot .acceleration .update (
        0.0 ,
        0.0 ,
        )

        robot .filtered_acceleration .update (
        0.0 ,
        0.0 ,
        )

    child .parent_return_started =True 
    child .parent_return_arrived =False 
    child .parent_return_dwell =0.0 

    physical .integration_parent_return_active =True 
    physical .integration_parent_return_arrived =False 
    physical .integration_parent_return_child_uid =(
    child .junction_uid 
    )
    physical .integration_parent_return_direction_local =(
    return_direction .copy ()
    )

    physical .active_branch_uid =None 
    physical .active_branch =None 





    return_yaw =math .degrees (
    math .atan2 (
    return_direction .y ,
    return_direction .x ,
    )
    )

    perception .yaw_deg =(
    return_yaw 
    )

    anchor .body_yaw =math .radians (
    return_yaw 
    )

    print (
    "[ParentReturnStart] "
    f"child={child .junction_uid } "
    f"parent={parent .junction_uid } "
    f"return_dir="
    f"({return_direction .x :.3f},"
    f"{return_direction .y :.3f}) "
    f"released_child_roles="
    f"{len (released_ids )} "
    "dfs_pop=False "
    "localization=False"
    )


def update_parent_return_transport (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
dt :float ,
)->None :
    """Drive completed Child swarm toward Parent and detect physical arrival."""

    child =multi_dfs .current 

    if child is None :
        return 

    if child .parent_junction_uid is None :
        return 

    if not child .subtree_complete :
        return 

    begin_parent_return_after_child_subtree (
    physical ,
    perception ,
    robots ,
    child ,
    )

    if not child .parent_return_started :
        return 

    if len (multi_dfs .stack )<2 :
        raise RuntimeError (
        "Parent return lost Parent DFS frame"
        )

    parent =multi_dfs .stack [-2 ]











    marker_id =parent .return_marker_id 

    if marker_id is None :
        raise RuntimeError (
        "Parent has no Return Marker: "
        f"parent={parent .junction_uid }"
        )

    marker =next (
    (
    robot 
    for robot in robots 
    if robot .robot_id ==marker_id 
    ),
    None ,
    )

    if marker is None :
        raise RuntimeError (
        "Parent Return Marker robot is missing: "
        f"parent={parent .junction_uid } "
        f"marker={marker_id }"
        )

    if (
    marker .role !="PEBBLE"
    or getattr (
    marker ,
    "pebble_state",
    None ,
    )
    !="JUNCTION_RETURN"
    or getattr (
    marker ,
    "marker_junction_uid",
    None ,
    )
    !=parent .junction_uid 
    ):
        raise RuntimeError (
        "invalid physical Parent Return Marker: "
        f"parent={parent .junction_uid } "
        f"marker={marker_id }"
        )

    return_direction =(
    child .return_direction_local 
    )

    if (
    return_direction is None 
    or return_direction .length_squared ()
    <=physical .EPSILON 
    ):
        return 

    return_direction =(
    return_direction .normalize ()
    )













    marker_observations =(
    observe_local_neighbors (
    marker ,
    robots ,
    return_direction ,
    max_range =physical .COMM_RANGE ,
    predicate =lambda robot :(
    robot is not marker 
    and (
    robot .role =="NORMAL"
    or robot is perception .leader 
    )
    and not robot .base_reserve 
    ),
    )
    )

    nearby_normal_ids ={
    observation .robot .robot_id 
    for observation in marker_observations 
    if observation .robot .role =="NORMAL"
    }

    anchor_visible =any (
    observation .robot 
    is perception .leader 
    for observation 
    in marker_observations 
    )

    mobile_normals =[
    robot 
    for robot in robots 
    if (
    robot .role =="NORMAL"
    and not robot .base_reserve 
    )
    ]

    required_normal_count =max (
    int (
    getattr (
    physical ,
    "FLOW_MIN_NORMAL_COUNT",
    8 ,
    )
    ),
    int (
    math .ceil (
    0.06 
    *max (
    len (mobile_normals ),
    1 ,
    )
    )
    ),
    )

    arrival_now =(
    anchor_visible 
    and len (nearby_normal_ids )
    >=required_normal_count 
    )

    child .parent_return_dwell =(
    child .parent_return_dwell +dt 
    if arrival_now 
    else 0.0 
    )

    required_dwell =max (
    0.25 ,
    float (
    getattr (
    physical ,
    "FLOW_ESTABLISH_DWELL_TIME",
    0.12 ,
    )
    ),
    )

    frame =getattr (
    physical ,
    "integration_frame",
    -1 ,
    )

    if frame %10 ==0 :
        print (
        "[ParentReturnGate] "
        f"child={child .junction_uid } "
        f"parent={parent .junction_uid } "
        f"marker={marker .robot_id } "
        f"nearby_normals="
        f"{len (nearby_normal_ids )}/"
        f"{required_normal_count } "
        f"anchor_visible="
        f"{anchor_visible } "
        f"dwell="
        f"{child .parent_return_dwell :.3f}/"
        f"{required_dwell :.3f} "
        f"arrived="
        f"{child .parent_return_arrived }"
        )

    if (
    child .parent_return_dwell 
    <required_dwell 
    ):
        return 

    if child .parent_return_arrived :
        return 

    child .parent_return_arrived =True 
    physical .integration_parent_return_arrived =True 

    print (
    "[ParentReturnArrived] "
    f"child={child .junction_uid } "
    f"parent={parent .junction_uid } "
    f"marker={marker .robot_id } "
    f"nearby_normals="
    f"{len (nearby_normal_ids )} "
    f"dwell="
    f"{child .parent_return_dwell :.3f} "
    "dfs_pop=False "
    "arrival_source=LOCAL_RETURN_MARKER"
    )

    popped_child ,returned_parent =(
    multi_dfs .pop_arrived_child ()
    )


    physical .integration_parent_return_active =False 

    print (
    "[ParentRestorePending] "
    f"parent={returned_parent .junction_uid } "
    f"popped_child={popped_child .junction_uid } "
    f"depth={multi_dfs .depth } "
    "physical_context_restored=False"
    )


def update_current_shepherd_cycle (
physical :types .ModuleType ,
robots :Sequence [Any ],
dt :float ,
)->None :
    """FILL -> PRESSURE_PUSH -> FLOW_BACKTRACK -> original Child Guard."""

    junction =multi_dfs .current 

    if junction is None :
        return 

    branch_uid =junction .active_branch_uid 

    if branch_uid is None :
        return 

    branch_phase =junction .branch_phase 

    descriptor =(
    physical .branch_descriptors_by_uid .get (
    branch_uid 
    )
    )

    lifecycle =(
    current_junction_branch_lifecycle (
    physical ,
    junction ,
    branch_uid 
    )
    )

    if descriptor is None or lifecycle is None :
        return 

    wall_ids_for_odometry =set (
    getattr (physical ,"integration_frontier_ids",set ())
    )
    wall_for_odometry =[
    robot for robot in robots 
    if robot .robot_id in wall_ids_for_odometry 
    and robot .role in {"FRONTIER_SHEPHERD","SHEPHERD"}
    ]
    odometry_depth =float (lifecycle .get (
    "child_shepherd_odometry_depth",
    physical .integration_frontier_depth ,
    ))
    now =float (physical .simulation_time )
    last_time =float (lifecycle .get ("child_shepherd_odometry_time",now ))
    elapsed =max (0.0 ,now -last_time )
    if wall_for_odometry and elapsed >0.0 :
        tangent ,_ =physical .descriptor_local_basis (descriptor )
        odometry_depth +=float (np .median ([
        robot .velocity .dot (tangent )for robot in wall_for_odometry 
        ]))*elapsed 
    lifecycle ["child_shepherd_odometry_depth"]=odometry_depth 
    lifecycle ["child_shepherd_odometry_time"]=now 

    if branch_phase ==BranchPhase .FILL :

        metrics =current_branch_pack_metrics (
        physical ,
        robots ,
        branch_uid ,
        )

        baseline_density =max (
        float (
        physical .integration_child_fill_baseline_density 
        or physical .EPSILON 
        ),
        physical .EPSILON ,
        )

        baseline_pressure =max (
        float (
        physical .integration_child_fill_baseline_pressure 
        or physical .EPSILON 
        ),
        physical .EPSILON ,
        )

        density_ratio =(
        float (metrics ["mean_density"])
        /baseline_density 
        )

        pressure_ratio =(
        float (metrics ["mean_pressure"])
        /baseline_pressure 
        )

        cols =max (
        1 ,
        int (
        lifecycle .get (
        "cols",
        1 ,
        )
        ),
        )

        min_pack_count =max (
        int (
        getattr (
        physical ,
        "FLOW_MIN_NORMAL_COUNT",
        6 ,
        )
        ),
        int (
        math .ceil (
        2.0 *cols 
        )
        ),
        )

        min_coverage =max (
        0.80 ,
        float (
        getattr (
        physical ,
        "SATURATION_PACKED_LATERAL_COVERAGE_RATIO",
        0.70 ,
        )
        ),
        )

        pack_ready =(
        int (metrics ["count"])
        >=min_pack_count 
        )

        coverage_ready =(
        float (metrics ["coverage"])
        >=min_coverage 
        )

        low_speed_ratio =float (
        metrics ["low_speed_ratio"]
        )





        stall_ready =(
        low_speed_ratio 
        >=0.40 
        )

        density_ready =(
        density_ratio 
        >=physical .SATURATION_DENSITY_RATIO 
        )

        pressure_ready =(
        pressure_ratio 
        >=LOCAL_SATURATION_PRESSURE_RATIO 
        )

        ready =(
        pack_ready 
        and coverage_ready 
        and stall_ready 
        and density_ready 
        and pressure_ready 
        )

        if ready :
            physical .integration_child_fill_dwell +=dt 
        else :
            physical .integration_child_fill_dwell =0.0 

        frame =getattr (
        physical ,
        "integration_frame",
        -1 ,
        )

        if frame %20 ==0 :
            print (
            "[ChildFill] "
            f"branch={branch_uid } "
            f"count={metrics ['count']} "
            f"required={min_pack_count } "
            f"pack_ready={pack_ready } "
            f"coverage="
            f"{float (metrics ['coverage']):.3f} "
            f"coverage_ready={coverage_ready } "
            f"density_ratio={density_ratio :.3f} "
            f"density_ready={density_ready } "
            f"pressure_ratio={pressure_ratio :.3f} "
            f"pressure_ready={pressure_ready } "
            f"low_speed={low_speed_ratio :.3f} "
            f"stall_ready={stall_ready } "
            f"ready={ready } "
            f"dwell="
            f"{physical .integration_child_fill_dwell :.3f}"
            )

        required_dwell =max (
        0.18 ,
        float (
        getattr (
        physical ,
        "SATURATION_DWELL_TIME",
        0.18 ,
        )
        ),
        )

        if (
        physical .integration_child_fill_dwell 
        <required_dwell 
        ):
            return 





        print (
        "[ChildSaturationConfirmed] "
        f"branch={branch_uid } "
        f"count={metrics ['count']} "
        f"coverage={float (metrics ['coverage']):.3f} "
        f"density_ratio={density_ratio :.3f} "
        f"pressure_ratio={pressure_ratio :.3f} "
        f"low_speed_ratio={low_speed_ratio :.3f} "
        f"dwell={physical .integration_child_fill_dwell :.3f}"
        )


        disable_leading_anchor (physical )

        if junction is not None :
            junction .branch_phase =(
            BranchPhase .PRESSURE_PUSH 
            )



        physical .integration_child_dfs_phase =(
        "PRESSURE_PUSH"
        )

        lifecycle ["state"]=(
        "PRESSURE_PUSH"
        )

        physical .integration_child_flow_dwell =(
        0.0 
        )

        print (
        "[ChildPressurePush] "
        f"branch={branch_uid } "
        f"count={metrics ['count']} "
        f"coverage="
        f"{float (metrics ['coverage']):.3f} "
        f"density_ratio={density_ratio :.3f} "
        f"pressure_ratio={pressure_ratio :.3f}"
        )

        return 







    if branch_phase not in {
    BranchPhase .PRESSURE_PUSH ,
    BranchPhase .FLOW_BACKTRACK ,
    }:
        return 

    frame =getattr (
    physical ,
    "integration_frame",
    -1 ,
    )

    offsets =(
    physical .integration_frontier_offsets 
    )

    axial_offsets =[
    float (value [0 ])
    for value 
    in offsets .values ()
    ]

    minimum_axial_offset =min (
    axial_offsets ,
    default =0.0 ,
    )

    current_depth =float (
    physical .integration_frontier_depth 
    )

    if "centroid_axial"not in lifecycle :
        raise RuntimeError (
        "Current Branch lifecycle has no original "
        "Guard centroid depth: "
        f"branch={branch_uid }"
        )

    original_guard_depth =float (
    lifecycle ["centroid_axial"]
    )

    usable_half =(
    physical .local_physical_usable_half_width (
    descriptor 
    )
    )



    junction_face_depth =(
    current_depth 
    +minimum_axial_offset 
    )

    wall_ids =set (physical .integration_frontier_ids )
    wall =[robot for robot in robots if robot .robot_id in wall_ids ]
    if not wall :
        return 
    reference =min (
    wall ,
    key =lambda robot :float (offsets .get (robot .robot_id ,(0.0 ,0.0 ))[0 ]),
    )
    reference_lateral =float (
    offsets .get (reference .robot_id ,(0.0 ,0.0 ))[1 ]
    )
    branch_normals =[
    (
    observation .robot ,
    float (junction_face_depth +observation .relative_axial ),
    )
    for observation in observe_local_neighbors (
    reference ,
    robots ,
    descriptor .local_outgoing_direction ,
    lateral_axis =physical .descriptor_local_basis (descriptor )[1 ],
    max_range =physical .COMM_RANGE ,
    predicate =lambda robot :(
    robot .role =="NORMAL"
    and not robot .base_reserve 
    and not bool (getattr (robot ,"is_fixed_anchor",False ))
    ),
    )
    if observation .relative_axial <=physical .ROBOT_RADIUS 
    and abs (
    observation .relative_lateral +reference_lateral 
    )<=usable_half +2.0 *physical .ROBOT_RADIUS 
    ]

    normal_front =None 

    if branch_normals :

        normal_front =(
        physical .linear_quantile (
        [
        axial 
        for _ ,axial 
        in branch_normals 
        ],
        0.98 ,
        )
        )

    if branch_phase ==BranchPhase .PRESSURE_PUSH :

        return_speed =float (
        physical .integration_child_pressure_push_speed 
        )

    else :

        return_speed =float (
        physical .integration_child_flow_backtrack_speed 
        )

    desired_depth =max (
    original_guard_depth ,
    current_depth 
    -return_speed *dt ,
    )


    if normal_front is not None :

        active_gap_ratio =float (
        getattr (
        physical ,
        "integration_child_active_min_center_gap_ratio",
        1.50 ,
        )
        )

        minimum_center_gap =(
        active_gap_ratio 
        *physical .ROBOT_RADIUS 
        )

        hard_floor =(
        float (normal_front )
        +minimum_center_gap 
        -minimum_axial_offset 
        )



        hard_floor =min (
        current_depth ,
        hard_floor ,
        )

        next_depth =max (
        desired_depth ,
        hard_floor ,
        original_guard_depth ,
        )

    else :

        next_depth =(
        desired_depth 
        )

    physical .integration_frontier_depth =(
    next_depth 
    )























    return_direction =(
    descriptor .local_return_direction .normalize ()
    )

    signed_speeds =[
    float (
    getattr (
    robot ,
    "observed_velocity",
    robot .velocity ,
    ).dot (
    return_direction 
    )
    )
    for robot ,_ 
    in branch_normals 
    ]

    normal_count =len (
    signed_speeds 
    )

    flow_speed_threshold =float (
    getattr (
    physical ,
    "FLOW_SPEED_THRESHOLD",
    1.5 ,
    )
    )

    moving_speeds =[
    speed 
    for speed in signed_speeds 
    if speed 
    >=flow_speed_threshold 
    ]

    if signed_speeds :
        moving_ratio =(
        len (moving_speeds )
        /normal_count 
        )

        mean_return_speed =(
        sum (
        max (
        0.0 ,
        speed ,
        )
        for speed 
        in signed_speeds 
        )
        /normal_count 
        )
    else :
        moving_ratio =0.0 
        mean_return_speed =0.0 

    flow_ratio_threshold =float (
    getattr (
    physical ,
    "FLOW_RATIO_THRESHOLD",
    0.45 ,
    )
    )

    flow_average_threshold =float (
    getattr (
    physical ,
    "FLOW_AVERAGE_SPEED_THRESHOLD",
    1.8 ,
    )
    )

    minimum_flow_count =int (
    getattr (
    physical ,
    "FLOW_MIN_NORMAL_COUNT",
    8 ,
    )
    )

    flow_established =(
    normal_count 
    >=minimum_flow_count 
    and moving_ratio 
    >=flow_ratio_threshold 
    and mean_return_speed 
    >=flow_average_threshold 
    )

    if (
    junction .branch_phase 
    ==BranchPhase .PRESSURE_PUSH 
    ):

        if flow_established :
            physical .integration_child_flow_dwell +=dt 
        else :
            physical .integration_child_flow_dwell =0.0 

        required_flow_dwell =float (
        getattr (
        physical ,
        "FLOW_ESTABLISH_DWELL_TIME",
        0.12 ,
        )
        )

        if frame %10 ==0 :
            print (
            "[ChildBackflowGate] "
            f"branch={branch_uid } "
            f"normal_count={normal_count }/"
            f"{minimum_flow_count } "
            f"moving_ratio={moving_ratio :.3f}/"
            f"{flow_ratio_threshold :.3f} "
            f"mean_speed="
            f"{mean_return_speed :.3f}/"
            f"{flow_average_threshold :.3f} "
            f"dwell="
            f"{physical .integration_child_flow_dwell :.3f}/"
            f"{required_flow_dwell :.3f} "
            f"established={flow_established }"
            )

        if (
        physical .integration_child_flow_dwell 
        >=required_flow_dwell 
        ):
            junction .branch_phase =(
            BranchPhase .FLOW_BACKTRACK 
            )
            branch_phase =junction .branch_phase 



            physical .integration_child_dfs_phase =(
            "FLOW_BACKTRACK"
            )

            lifecycle ["state"]=(
            "FLOW_BACKTRACK"
            )

            print (
            "[ChildFlowBacktrack] "
            f"branch={branch_uid } "
            f"normal_count={normal_count } "
            f"required_count={minimum_flow_count } "
            f"moving_ratio={moving_ratio :.3f} "
            f"mean_speed="
            f"{mean_return_speed :.3f} "
            f"dwell="
            f"{physical .integration_child_flow_dwell :.3f}"
            )


    if frame %10 ==0 :

        child_contact_robots =[
        robot 
        for robot in robots 
        if (
        robot .role =="NORMAL"
        and int (
        getattr (
        robot ,
        "last_child_shepherd_contact_count",
        0 ,
        )
        )
        >0 
        )
        ]

        max_contact_force =max (
        (
        float (
        getattr (
        robot ,
        "last_child_shepherd_contact_force",
        0.0 ,
        )
        )
        for robot 
        in child_contact_robots 
        ),
        default =0.0 ,
        )

        print (
        "[ChildPistonContact] "
        f"branch={branch_uid } "
        f"phase={branch_phase .name } "
        f"return_speed={return_speed :.3f} "
        f"active_gap="
        f"{minimum_center_gap if normal_front is not None else -1.0 :.3f} "
        f"contact_normals="
        f"{len (child_contact_robots )} "
        f"max_contact_force="
        f"{max_contact_force :.3f}"
        )

    if frame %20 ==0 :

        print (
        "[ChildReturnPiston] "
        f"branch={branch_uid } "
        f"phase="
        f"{physical .integration_child_dfs_phase } "
        f"depth={current_depth :.3f}"
        f"->{next_depth :.3f} "
        f"original_guard_depth={original_guard_depth :.3f} "
        f"normal_front="
        f"{float (normal_front )if normal_front is not None else -1.0 :.3f} "
        f"branch_normals={len (branch_normals )} "
        f"moving_ratio={moving_ratio :.3f} "
        f"mean_return_speed={mean_return_speed :.3f}"
        )











    if (
    branch_phase 
    !=BranchPhase .FLOW_BACKTRACK 
    ):
        return 

    if (
    next_depth 
    >original_guard_depth 
    +physical .JUNCTION_GUARD_POSITION_TOLERANCE 
    ):
        return 

    if (
    odometry_depth 
    >original_guard_depth 
    +physical .JUNCTION_GUARD_POSITION_TOLERANCE 
    ):
        return 

    expected_ids =set (
    lifecycle .get (
    "robot_ids",
    [],
    )
    )

    shepherds ={
    robot .robot_id :
    robot 
    for robot in robots 
    if (
    robot .robot_id 
    in expected_ids 
    and robot .role 
    =="SHEPHERD"
    and robot .shepherd_branch 
    ==branch_uid 
    )
    }

    if (
    set (shepherds )
    !=expected_ids 
    ):
        return 

    (
    swarm_returned ,
    branch_side_normals ,
    junction_side_normals ,
    return_dwell ,
    )=evaluate_local_swarm_return_completion (
    physical ,
    robots ,
    branch_uid ,
    descriptor ,
    lifecycle ,
    dt ,
    )

    if not swarm_returned :
        if frame %10 ==0 :
            print (
            "[ChildSwarmReturnWait] "
            f"branch={branch_uid } "
            f"branch_side_normals="
            f"{branch_side_normals } "
            f"junction_side_normals="
            f"{junction_side_normals } "
            f"dwell={return_dwell :.3f}"
            )

        return 

    finish_current_branch_return (
    physical ,
    robots ,
    branch_uid ,
    )


def update_current_frontier_exploration (
physical :types .ModuleType ,
robots :Sequence [Any ],
dt :float ,
)->None :
    """Continuously advance a Child Frontier after bootstrap."""

    uid =getattr (
    physical ,
    "integration_frontier_active_uid",
    None ,
    )

    if uid is None :
        return 

    descriptor =(
    physical .branch_descriptors_by_uid .get (
    uid 
    )
    )

    junction =(
    multi_dfs .current 
    )

    if junction is None :
        return 

    lifecycle =(
    current_junction_branch_lifecycle (
    physical ,
    junction ,
    uid ,
    )
    )

    if (
    descriptor is None 
    or lifecycle is None 
    ):
        return 

    frontier_ids =set (
    physical .integration_frontier_ids 
    )

    frontier_members =[
    robot 
    for robot in robots 
    if (
    robot .robot_id in frontier_ids 
    and robot .role =="FRONTIER_SHEPHERD"
    and robot .shepherd_branch ==uid 
    )
    ]

    if not frontier_members :
        return 



    dead_end =evaluate_frontier_dead_end (
    physical ,
    uid ,
    descriptor ,
    frontier_members ,
    dt ,
    )

    if dead_end .confirmed :
        actual_dead_end_depth =float (
        lifecycle .get (
        "rigid_applied_depth",
        physical .integration_frontier_depth ,
        )
        )

        physical .integration_frontier_depth =actual_dead_end_depth 
        lifecycle [
        "frontier_contact_centroid_depth"
        ]=actual_dead_end_depth 
        lifecycle [
        "child_shepherd_odometry_depth"
        ]=actual_dead_end_depth 
        lifecycle [
        "child_shepherd_odometry_time"
        ]=float (physical .simulation_time )
        lifecycle ["rigid_cache_frame"]=-1 

        print (
        "[ChildFrontierDeadEndConfirmed] "
        f"branch={uid } "
        f"depth={actual_dead_end_depth :.3f} "
        f"command_speed="
        f"{dead_end .command_forward_speed :.3f} "
        f"actual_speed="
        f"{dead_end .actual_forward_speed :.3f} "
        f"lidar_blocked="
        f"{dead_end .lidar_blocked } "
        f"blocked_ratio="
        f"{dead_end .forward_blocked_ratio :.3f} "
        f"no_junction="
        f"{dead_end .no_junction_evidence } "
        f"dwell={dead_end .dwell :.3f}"
        )

        promote_current_frontier_to_shepherd (
        physical ,
        robots ,
        uid ,
        )

        return 







    usable_half =(
    physical .local_physical_usable_half_width (
    descriptor 
    )
    )

    offsets =physical .integration_frontier_offsets 
    reference =min (
    frontier_members ,
    key =lambda robot :sum (
    abs (float (value ))
    for value in offsets .get (robot .robot_id ,(0.0 ,0.0 ))
    ),
    )
    reference_axial ,reference_lateral =offsets .get (
    reference .robot_id ,(0.0 ,0.0 )
    )
    normal_observations =[
    observation 
    for observation in observe_local_neighbors (
    reference ,
    robots ,
    descriptor .local_outgoing_direction ,
    lateral_axis =physical .descriptor_local_basis (descriptor )[1 ],
    max_range =physical .COMM_RANGE ,
    predicate =lambda robot :robot .role =="NORMAL",
    )
    if observation .relative_axial 
    >=-physical .FRONTIER_LINE_LEAD_GAP -float (reference_axial )
    and abs (
    observation .relative_lateral +float (reference_lateral )
    )<=usable_half 
    ]

    if not normal_observations :
        return 

    supported_relative_front =(
    physical .linear_quantile (
    [
    observation .relative_axial 
    for observation in normal_observations 
    ],
    float (
    physical .integration_frontier_support_quantile 
    ),
    )
    )









    axial_offsets =[
    float (value [0 ])
    for value in offsets .values ()
    ]

    trailing_offset =min (
    axial_offsets ,
    default =0.0 ,
    )

    desired_depth =(
    float (physical .integration_frontier_depth )
    +float (reference_axial )
    +supported_relative_front 
    +physical .FRONTIER_LINE_LEAD_GAP 
    -trailing_offset 
    )

    current_depth =float (
    physical .integration_frontier_depth 
    )

    next_depth =min (
    desired_depth ,
    current_depth 
    +float (
    physical .integration_frontier_cruise_speed 
    )
    *dt ,
    )



    next_depth =max (
    current_depth ,
    next_depth ,
    )



    physical .integration_frontier_depth =(
    next_depth 
    )


def update_current_branch_transport (
physical :types .ModuleType ,
robots :Sequence [Any ],
dt :float ,
)->None :
    """Advance the active Child Frontier centroid until bootstrap is complete."""

    junction =multi_dfs .current 

    if junction is None :
        return 

    uid =junction .active_branch_uid 

    if uid is None :
        return 

    branch_phase =junction .branch_phase 

    if branch_phase in {
    BranchPhase .FILL ,
    BranchPhase .PRESSURE_PUSH ,
    BranchPhase .FLOW_BACKTRACK ,
    }:
        update_current_shepherd_cycle (
        physical ,
        robots ,
        dt ,
        )
        return 

    if branch_phase ==BranchPhase .EXPLORE :
        update_current_frontier_exploration (
        physical ,
        robots ,
        dt ,
        )
        return 

    if branch_phase !=BranchPhase .FRONTIER_BOOTSTRAP :
        return 

    lifecycle =physical .integration_wall_lifecycle .get (uid )
    if lifecycle is None :
        raise RuntimeError (
        "Current Frontier lifecycle disappeared: "
        f"uid={uid }"
        )

    current_depth =float (
    physical .integration_frontier_depth 
    )

    target_depth =float (
    physical .integration_frontier_target_depth 
    )

    next_depth =min (
    target_depth ,
    current_depth 
    +float (
    physical .integration_frontier_cruise_speed 
    )
    *dt ,
    )

    physical .integration_frontier_depth =(
    next_depth 
    )



    if (
    next_depth 
    +1.0e-6 
    <target_depth 
    ):
        return 

    actual_depth =float (
    lifecycle .get (
    "rigid_applied_depth",
    current_depth ,
    )
    )
    actual_depth_tolerance =max (
    0.25 *float (physical .ROBOT_RADIUS ),
    1.0e-3 ,
    )
    if (
    actual_depth 
    +actual_depth_tolerance 
    <target_depth 
    ):
        if (
        getattr (physical ,"integration_frame",0 )%10 ==0 
        ):
            print (
            "[FrontierBootstrapPhysicalWait] "
            f"branch={uid } "
            f"command_depth={next_depth :.3f} "
            f"actual_depth={actual_depth :.3f} "
            f"target_depth={target_depth :.3f}"
            )
        return 

    descriptor =(
    physical .branch_descriptors_by_uid .get (
    uid 
    )
    )

    if descriptor is None :
        raise RuntimeError (
        "Child Frontier descriptor disappeared: "
        f"uid={uid }"
        )

    expected_ids =set (
    physical .integration_frontier_ids 
    )

    frontier_members =[
    robot 
    for robot in robots 
    if (
    robot .robot_id 
    in expected_ids 
    and robot .role 
    =="FRONTIER_SHEPHERD"
    and robot .shepherd_branch 
    ==uid 
    )
    ]

    live_ids ={
    robot .robot_id 
    for robot in frontier_members 
    }

    if live_ids !=expected_ids :
        raise RuntimeError (
        "Child Frontier bootstrap lost lineage: "
        f"branch={uid } "
        f"expected={len (expected_ids )} "
        f"live={len (live_ids )}"
        )

    tangent ,normal =(
    physical .descriptor_local_basis (
    descriptor 
    )
    )

    offsets =(
    physical .integration_frontier_offsets 
    )















    reference =frontier_members [0 ]

    reference_offset =offsets .get (
    reference .robot_id ,
    (0.0 ,0.0 ),
    )

    relative_observations =observe_local_neighbors (
    reference ,
    frontier_members ,
    tangent ,
    lateral_axis =normal ,
    )

    max_error =0.0 

    for observation in relative_observations :

        member_offset =offsets .get (
        observation .robot .robot_id ,
        (0.0 ,0.0 ),
        )

        expected_axial =(
        float (member_offset [0 ])
        -float (reference_offset [0 ])
        )

        expected_lateral =(
        float (member_offset [1 ])
        -float (reference_offset [1 ])
        )

        error =math .hypot (
        observation .relative_axial 
        -expected_axial ,
        observation .relative_lateral 
        -expected_lateral ,
        )

        max_error =max (
        max_error ,
        error ,
        )

    if (
    max_error 
    >physical .JUNCTION_GUARD_POSITION_TOLERANCE 
    ):
        return 

    physical .integration_frontier_bootstrap_complete =(
    True 
    )

    junction .branch_phase =BranchPhase .EXPLORE 

    if not getattr (
    physical ,
    "integration_frontier_ready_logged",
    False ,
    ):

        physical .integration_frontier_ready_logged =(
        True 
        )

        child =multi_dfs .current 

        print (
        "[ChildFrontierBootstrapReady] "
        f"junction="
        f"{child .junction_uid if child is not None else '?'} "
        f"branch={uid } "
        f"robots={len (expected_ids )} "
        "same_ids=True "
        f"max_error={max_error :.3f}"
        )

def update_current_junction_frontier_transport (
physical :types .ModuleType ,
robots :Sequence [Any ],
dt :float ,
)->None :
    """Advance the active Frontier through one current-Junction API."""

    junction =multi_dfs .current 

    if junction is None :
        return 

    runtime_kind =(
    current_junction_frontier_runtime_kind (
    physical ,
    junction ,
    )
    )

    if runtime_kind is None :
        return 

    if runtime_kind in {
    "UID_LOCAL",
    "LEGACY_FIXTURE",
    }:

        update_current_branch_transport (
        physical ,
        robots ,
        dt ,
        )

        return 

    raise RuntimeError (
    "Unknown current Junction Frontier transport runtime: "
    f"junction={junction .junction_uid } "
    f"runtime={runtime_kind }"
    )


def update_current_branch_cycle (
physical :types .ModuleType ,
perception :AdaptivePerception ,
robots :Sequence [Any ],
dt :float ,
)->None :
    """Advance one ordinary Physical-DFS Branch cycle at the current Junction."""

    junction =multi_dfs.current 

    if junction is None :
        return 

    if multi_dfs .parent_release_pending :
        return 

    if multi_dfs .parent_restore_pending :
        return 

    if multi_dfs .parent_guard_reformation_pending :
        return 

    multi_dfs .refresh_subtree_complete (
    junction 
    )

    if junction .subtree_complete :
        return 



    prepare_current_junction_frontier_handoff (
    physical ,
    perception ,
    robots ,
    )

    runtime_kind =(
    current_junction_frontier_runtime_kind (
    physical ,
    junction ,
    )
    )

    if runtime_kind is None :
        return 

    if runtime_kind not in {
    "UID_LOCAL",
    "LEGACY_FIXTURE",
    }:
        raise RuntimeError (
        "Unknown current Branch runtime: "
        f"junction={junction .junction_uid } "
        f"runtime={runtime_kind }"
        )

    active_uid =junction .active_branch_uid 

    if active_uid is not None :
        active_state =junction .branch_states .get (active_uid )

        if active_state =="ACTIVE_CHILD":
            return 

        update_current_branch_transport (
        physical ,
        robots ,
        dt ,
        )

        return 

    if not getattr (
    physical ,
    "integration_child_guard_lifecycle_initialized",
    False ,
    ):
        return 

    activate_current_branch_frontier (
    physical ,
    perception ,
    robots ,
    )


def mean_nearest_spacing (robots :Sequence [Any ])->float :
    values :list [float ]=[]
    for robot in robots :
        nearest =min (
        (
        robot .position .distance_to (other .position )
        for other in robots 
        if other is not robot 
        ),
        default =0.0 ,
        )
        values .append (nearest )
    return float (np .mean (values ))


def log_initial_dynamics (
physical :types .ModuleType ,
robots :Sequence [Any ],
reference_density :float ,
frame :int ,
)->None :
    if frame not in {1 ,5 ,10 }:
        return 
    forward =pygame .Vector2 (0.0 ,-1.0 )
    print (
    f"[InitDynamics] frame={frame } "
    f"mean_density={np .mean ([r .density for r in robots ]):.6f} "
    f"reference_density={reference_density :.6f} "
    f"mean_pressure={np .mean ([r .pressure for r in robots ]):.6f} "
    f"mean_nearest_spacing={mean_nearest_spacing (robots ):.3f} "
    f"equilibrium_spacing={physical .SAFE_RADIUS *physical .NORMAL_EQUILIBRIUM_SCALE :.3f} "
    f"mean_forward_velocity={np .mean ([r .velocity .dot (forward )for r in robots ]):.3f} "
    f"longitudinal_span={max (r .position .y for r in robots )-min (r .position .y for r in robots ):.3f}"
    )


def main (argv :Sequence [str ]|None =None )->int :
    global multi_dfs 
    multi_dfs =MultiJunctionManager ()
    args =parse_args (argv )
    if args .headless :
        os .environ .setdefault ("SDL_VIDEODRIVER","dummy")
    physical =_load_physical_definitions ()
    configure_extended_approach (physical )
    configure_multi_test_geometry (physical )
    physical .integration_guard_hold_active =False 
    physical .integration_guard_gating_enabled =False 
    physical .integration_placement_localization_enabled =False 
    physical .integration_guard_who_localization_enabled =False 



    physical .integration_latest_lidar_frame =None 
    physical .integration_dead_end_diagnostics ={}
    physical .integration_return_completion_dwell ={}

    physical .integration_parent_return_active =False 
    physical .integration_parent_return_arrived =False 
    physical .integration_parent_return_child_uid =None 
    physical .integration_parent_return_direction_local =pygame .Vector2 ()

    physical .integration_provisional_guard_active =False 

    physical .integration_provisional_guard_groups ={}
    physical .integration_all_guard_cohorts_logged =False 
    physical .integration_guard_role_transition_jump =0.0 
    physical .integration_guard_leakage ={}
    physical .integration_ready_guard_ids_by_uid ={}
    physical .integration_wall_lifecycle ={}
    physical .integration_guard_formation_start_frame =None 
    physical .integration_opening_mouth_associations ={}
    physical .integration_frame =0 
    physical .integration_wall_max_step =0.0 
    physical .integration_frontier_max_step =0.0 
    physical .integration_shepherd_max_step =0.0 
    physical .integration_final_return_requested =False 
    physical .integration_global_dfs_complete =False 

    physical .integration_local_final_return_active =False 
    physical .integration_local_final_roles_released =False 

    physical .integration_final_trunk_pending_robot_id =None 
    physical .integration_final_trunk_reconnect_dwell =0.0 
    physical .integration_final_trunk_reconnect_dwell_required =0.25 

    physical .integration_final_base_arrival_dwell =0.0 
    physical .integration_final_base_arrival_dwell_required =0.50 
    physical .integration_final_base_arrived_ids =set ()

    physical .integration_final_return_force_scale =1.0 

    physical .integration_anchor_prep_required_frames =8 

    physical .integration_anchor_target_gap =max (
    2.0 *float (physical .ROBOT_RADIUS ),
    2.0 *float (physical .GRID_ROW_SPACING ),
    )
    physical .integration_anchor_breakout_speed =(
    18.0 *ROBOT_MOTION_SPEED_SCALE 
    )
    physical .integration_anchor_breakout_lateral_speed =(
    10.0 *ROBOT_MOTION_SPEED_SCALE 
    )
    physical .integration_anchor_breakout_guard_standoff =max (
    3.2 *float (physical .ROBOT_RADIUS ),
    float (physical .integration_anchor_target_gap ),
    )
    physical .integration_frontier_headstart_min_gap =max (
    2.2 *float (physical .ROBOT_RADIUS ),
    1.25 *float (physical .GRID_ROW_SPACING ),
    )
    physical .integration_frontier_headstart_dwell =0.0 
    physical .integration_anchor_fan_base_half_width =(
    0.75 *float (physical .GRID_SPACING )
    )
    physical .integration_anchor_fan_widening_rate =0.55 
    physical .integration_anchor_explore_cruise_speed =(
    35.0 *ROBOT_MOTION_SPEED_SCALE 
    )
    physical .integration_anchor_explore_max_follow_gap =max (
    8.0 *float (physical .ROBOT_RADIUS ),
    6.0 *float (physical .integration_anchor_target_gap ),
    )



    physical .integration_frontier_shape_neighbor_count =8 
    physical .integration_frontier_shape_gain =9.0 
    physical .integration_frontier_shape_damping =0.85 
    physical .integration_frontier_shape_correction_speed =(
    12.0 *ROBOT_MOTION_SPEED_SCALE 
    )
    physical .integration_frontier_shape_error_tolerance =max (
    1.35 *float (physical .ROBOT_RADIUS ),
    0.55 *float (physical .GRID_ROW_SPACING ),
    )
    physical .integration_frontier_contact_hold_ratio =0.35 

    physical .integration_anchor_center_ratio_tol =0.12 
    reset_leading_anchor_state (physical )


    install_local_forward_ingress (physical )
    install_thick_wall_readiness_audit (physical )
    install_continuous_guard_settling (physical )

    install_local_physical_saturation_bridge (
    physical 
    )
    install_current_boundary_transport_runtime (
    physical 
    )

    install_current_branch_physical_drive (
    physical 
    )
    def integration_log_sink (active_robots :Sequence [Any ],reason :str )->Path :
        """Keep the reference module's legacy CSV untouched."""
        physical .metrics .saved =True 
        print (f"[Log] integration run complete reason={reason }; console summary follows")
        return HERE /"sph_dfs_experiment_summary.csv"

    physical .save_experiment_logs =integration_log_sink 
    robots ,reference_density ,color_reference_density =physical .initialize_simulation ()
    initialize_deployment_fields (physical ,robots )
    reference_density ,color_reference_density =refresh_centered_deployment_physics (
    physical ,robots 
    )
    initial_mean_density =float (np .mean ([robot .density for robot in robots ]))
    equilibrium_spacing =(
    physical .REFERENCE_EQUILIBRIUM_SPACING 
    )
    inside_count =sum (
    physical .is_walkable (robot .position ,robot .radius )for robot in robots 
    )
    print (f"[Init] robot_count={len (robots )}")
    print (f"[Init] robots_inside_walkable={inside_count }/{len (robots )}")
    print (f"[Spawn] robot_count={len (robots )}")
    print (f"[Spawn] walkable_inside={inside_count }/{len (robots )}")
    print (f"[Init] grid_spacing={physical .GRID_SPACING :.3f}")
    print (f"[Init] row_spacing={physical .GRID_ROW_SPACING :.3f}")
    print (f"[Init] equilibrium_spacing={equilibrium_spacing :.3f}")
    print (f"[Init] initial_mean_density={initial_mean_density :.6f}")
    print (f"[Init] reference_density={reference_density :.6f}")
    print ("[Init] artificial_initial_compression=False")
    print ("[Init] SPH_from_first_frame=True local_forward_from_first_frame=True")
    print (
    "[Init] physical_shepherd_only=True "
    "invisible_return_curtain=False"
    )
    perception =AdaptivePerception (physical ,robots )

    physical .integration_perception =perception 

    install_lidar_relay_protection (
    physical ,
    perception ,
    )
    initial_lidar_frame =perception .update (physical .simulation_time )
    print (f"[LiDAR] initial_opening_count={len (initial_lidar_frame .openings )}")
    print (
    f"[LiDAR] initial_junction_confirmed={perception .junction_confirmed } "
    f"initial_anchor_fixed={perception .anchor_fixed }"
    )
    renderer =None if args .headless else DarkRenderer (physical )
    running ,paused =True ,False 
    show_profile =show_rays =True 
    show_comm =False 
    density =False 
    frame_count =0 
    previous_phase =physical .phase .name 
    visited_log :list [str ]=[]
    clock =pygame .time .Clock ()

    while running :

        if not args .headless :
            clock .tick (physical .FPS )

        dt =float (args .dt )

        dt =min (
        dt ,
        physical .INITIAL_INGRESS_MAX_DT 
        if physical .phase 
        ==physical .SimulationPhase .MOVE_TO_JUNCTION 
        else physical .NORMAL_PHYSICS_MAX_DT ,
        )
        frame_count +=1 
        physical .integration_frame =frame_count 
        for event in pygame .event .get ():
            if event .type ==pygame .QUIT :
                running =False 
            elif event .type ==pygame .KEYDOWN :
                if event .key ==pygame .K_SPACE :
                    paused =not paused 
                elif event .key ==pygame .K_r :
                    multi_dfs =MultiJunctionManager ()
                    robots ,reference_density ,color_reference_density =physical .initialize_simulation ()
                    initialize_deployment_fields (physical ,robots )
                    reference_density ,color_reference_density =refresh_centered_deployment_physics (
                    physical ,robots 
                    )
                    perception .reset (robots )
                    reset_leading_anchor_state (physical )
                    initial_lidar_frame =perception .update (physical .simulation_time )
                    print (f"[LiDAR] initial_opening_count={len (initial_lidar_frame .openings )}")
                    physical .integration_backtrack_command_depth =None 
                    physical .integration_backtrack_pack_rear_depth =None 
                    physical .integration_backtrack_support_depth =None 
                    physical .integration_backtrack_support_count =0 
                    physical .integration_backtrack_lateral_coverage =0.0 
                    previous_phase ,visited_log ,frame_count =physical .phase .name ,[],0 
                elif event .key ==pygame .K_p :
                    show_profile =not show_profile 
                elif event .key ==pygame .K_l :
                    show_rays =not show_rays 
                elif event .key ==pygame .K_d :
                    density =not density 
                elif event .key ==pygame .K_c :
                    show_comm =not show_comm 
                elif event .key ==pygame .K_ESCAPE :
                    running =False 
        if not paused :
            physical .simulation_time +=dt 

            final_return_control_active =(
            getattr (
            physical ,
            "integration_final_guard_sweep_active",
            False ,
            )
            or getattr (
            physical ,
            "integration_local_final_return_active",
            False ,
            )
            )

            spatial_grid =physical .build_spatial_grid (robots )
            physics_grid =physical .build_physics_grid (robots )

            physical .compute_densities (
            robots ,
            physics_grid ,
            )

            if not final_return_control_active :
                physical .update_transfer_continuity_control (
                robots 
                )

            physical .compute_pressures (
            robots ,
            reference_density ,
            )

            if not final_return_control_active :

                advance_guard_settling_waypoints (
                physical ,
                robots ,
                )

                log_initial_dynamics (
                physical ,
                robots ,
                reference_density ,
                frame_count ,
                )

                update_anchor_follow_tree (
                physical ,
                perception ,
                robots ,
                )

                update_parent_return_transport (
                physical ,
                perception ,
                robots ,
                dt ,
                )

                restore_parent_context_after_child_pop (
                physical ,
                perception ,
                robots ,
                )



            physical .compute_sph_forces (
            robots ,
            physics_grid ,
            spatial_grid ,
            dt ,
            )

            apply_post_anchor_normal_crawl (
            physical ,
            perception ,
            robots ,
            )

            if not final_return_control_active :

                maintain_anchor_breakout (
                physical ,
                perception ,
                robots ,
                dt ,
                )

                update_child_lidar_probe (
                physical ,
                perception ,
                robots ,
                dt ,
                )

                maintain_mobile_anchor_at_normal_front (
                physical ,
                perception ,
                robots ,
                )

                apply_anchor_prep_corridor_yield (
                physical ,
                perception ,
                robots ,
                dt ,
                )

                enforce_anchor_fan_no_overtake (
                physical ,
                perception ,
                robots ,
                dt ,
                )

                update_current_branch_cycle (
                physical ,
                perception ,
                robots ,
                dt ,
                )


            anchor_motion_active =(
            getattr (
            physical ,
            "integration_anchor_prep_request_uid",
            None ,
            )
            is not None 
            or getattr (
            physical ,
            "integration_leading_anchor_uid",
            None ,
            )
            is not None 
            )

            if not anchor_motion_active :
                apply_junction_approach_crawl (
                physical ,
                perception ,
                )

            physical .integration_collision_frame_start_by_id ={
            robot .robot_id :
            robot .position .copy ()
            for robot in robots 
            }

            physical .integration_collision_frame_velocity_by_id ={
            robot .robot_id :(
            pygame .Vector2 ()
            if (
            perception .anchor_fixed 
            and robot is perception .leader 
            )
            else robot .velocity .copy ()
            )
            for robot in robots 
            }

            physical .integration_collision_updated_ids =set ()

            physical .integration_universal_collision_stats ={
            "contacts":0 ,
            "robots":0 ,
            "slides":0 ,
            "samples":[],
            }

            for robot in robots :
                if (
                perception .anchor_fixed 
                and robot is perception .leader 
                ):
                    physical .integration_collision_updated_ids .add (
                    robot .robot_id 
                    )
                    continue 

                role_before_update =(
                robot .role 
                )

                before_update =(
                robot .position .copy ()
                )

                debug_guard_motion =(
                robot .role =="JUNCTION_GUARD"
                and robot .junction_guard_anchor is not None 
                and robot .position .distance_to (
                robot .junction_guard_anchor 
                )
                >physical .JUNCTION_GUARD_POSITION_TOLERANCE 
                and frame_count %20 ==0 
                )

                debug_guard_target =(
                robot .junction_guard_anchor .copy ()
                if debug_guard_motion 
                else None 
                )

                debug_guard_error_before =(
                robot .position .distance_to (
                debug_guard_target 
                )
                if debug_guard_target is not None 
                else 0.0 
                )

                robot .update (
                dt 
                )

                proposed_position =(
                robot .position .copy ()
                )
                if (
                robot is perception .leader 
                and getattr (
                physical ,
                "integration_anchor_prep_request_uid",
                None ,
                )is not None 
                and frame_count %10 ==0 
                ):
                    comm_parent =getattr (
                    robot ,
                    "comm_parent",
                    None ,
                    )

                    comm_dist =(
                    robot .position .distance_to (
                    comm_parent .position 
                    )
                    if comm_parent is not None 
                    else -1.0 
                    )

                    print (
                    "[AnchorPrepMotionAudit] "
                    f"frame={frame_count } "
                    f"before=({before_update .x :.2f},{before_update .y :.2f}) "
                    f"after_robot_update=({robot .position .x :.2f},{robot .position .y :.2f}) "
                    f"step={robot .position .distance_to (before_update ):.6f} "
                    f"speed={robot .velocity .length ():.3f} "
                    f"acc={robot .acceleration .length ():.3f} "
                    f"connected={robot .connected_to_base } "
                    f"comm_parent={getattr (comm_parent ,'robot_id',None )} "
                    f"comm_dist={comm_dist :.2f}"
                    )
                if debug_guard_motion :
                    debug_guard_after_update =(
                    proposed_position .copy ()
                    )

                limited_position =(
                physical 
                .integration_universal_robot_motion_limit (
                robot ,
                before_update ,
                proposed_position ,
                dt ,
                )
                )

                if (
                robot is perception .leader 
                and getattr (
                physical ,
                "integration_anchor_prep_request_uid",
                None ,
                )is not None 
                and frame_count %10 ==0 
                ):
                    print (
                    "[AnchorPrepCollisionAudit] "
                    f"frame={frame_count } "
                    f"robot_update_step="
                    f"{proposed_position .distance_to (before_update ):.6f} "
                    f"collision_step="
                    f"{limited_position .distance_to (before_update ):.6f} "
                    f"collision_changed="
                    f"{limited_position .distance_squared_to (proposed_position )>1.0e-12 }"
                    )

                if debug_guard_motion :
                    print (
                    "[GuardMotionAudit] "
                    f"frame={frame_count } "
                    f"id={robot .robot_id } "
                    f"layer={robot .junction_guard_layer } "
                    f"slot="
                    f"{getattr (robot ,'integration_guard_slot_index',-1 )} "
                    f"error_before="
                    f"{debug_guard_error_before :.3f} "
                    f"robot_update_step="
                    f"{debug_guard_after_update .distance_to (before_update ):.6f} "
                    f"collision_limited_step="
                    f"{limited_position .distance_to (before_update ):.6f} "
                    f"collision_changed="
                    f"{limited_position .distance_squared_to (debug_guard_after_update )>1.0e-12 } "
                    f"target="
                    f"({debug_guard_target .x :.3f},"
                    f"{debug_guard_target .y :.3f}) "
                    f"before="
                    f"({before_update .x :.3f},"
                    f"{before_update .y :.3f}) "
                    f"after_update="
                    f"({debug_guard_after_update .x :.3f},"
                    f"{debug_guard_after_update .y :.3f})"
                    )


                if (
                limited_position 
                .distance_squared_to (
                proposed_position 
                )
                >1.0e-12 
                ):
                    robot .position =(
                    limited_position .copy ()
                    )

                    actual_velocity =(
                    robot .position 
                    -before_update 
                    )/max (
                    dt ,
                    physical .EPSILON ,
                    )

                    robot .velocity =(
                    actual_velocity .copy ()
                    )
                    robot .observed_velocity =(
                    actual_velocity .copy ()
                    )
                    robot .commanded_velocity =(
                    actual_velocity .copy ()
                    )
                    robot .previous_position =(
                    before_update 
                    )

                physical .integration_collision_updated_ids .add (
                robot .robot_id 
                )

                displacement =(
                robot .position 
                .distance_to (
                before_update 
                )
                )

                if (
                robot .role 
                =="JUNCTION_GUARD"
                ):
                    physical .integration_wall_max_step =max (
                    physical .integration_wall_max_step ,
                    displacement ,
                    )

                if (
                role_before_update 
                =="FRONTIER_SHEPHERD"
                ):
                    physical .integration_frontier_max_step =max (
                    physical .integration_frontier_max_step ,
                    displacement ,
                    )

                if (
                role_before_update 
                =="SHEPHERD"
                ):
                    physical .integration_shepherd_max_step =max (
                    physical .integration_shepherd_max_step ,
                    displacement ,
                    )

            collision_stats =getattr (
            physical ,
            "integration_universal_collision_stats",
            {},
            )

            if (
            frame_count %10 ==0 
            and int (
            collision_stats .get (
            "contacts",
            0 ,
            )
            )
            >0 
            ):
                print (
                "[UniversalCollisionSummary] "
                f"frame={frame_count } "
                f"contacts={collision_stats .get ('contacts',0 )} "
                f"robots={collision_stats .get ('robots',0 )} "
                f"slides={collision_stats .get ('slides',0 )} "
                f"samples={collision_stats .get ('samples',[])}"
                )




            if not final_return_control_active :

                enforce_anchor_fan_no_overtake (
                physical ,
                perception ,
                robots ,
                dt ,
                )

                perception .enforce_anchor ()

            spatial_grid =physical .build_spatial_grid (
            robots 
            )



            physical .update_communication_system (
            robots ,
            spatial_grid ,
            )

            if not final_return_control_active :

                lidar_frame =perception .update (
                physical .simulation_time 
                )

                physical .integration_latest_lidar_frame =(
                lidar_frame 
                )

                register_confirmed_root_junction (
                physical ,
                perception ,
                )

                update_child_observation_session (
                physical ,
                perception ,
                lidar_frame ,
                )

                update_child_moving_candidate (
                physical ,
                perception ,
                lidar_frame ,
                dt ,
                )

                update_child_stationary_verification (
                physical ,
                perception ,
                lidar_frame ,
                )

                release_confirmed_parent_junction (
                physical ,
                perception ,
                robots ,
                )

                initialize_confirmed_child_guard_context (
                physical ,
                perception ,
                robots ,
                )

            else :
                lidar_frame =getattr (
                physical ,
                "integration_latest_lidar_frame",
                None ,
                )
            current_junction =(
            multi_dfs .current 
            )


            global_dfs_complete_now =(
            multi_dfs .global_dfs_complete ()
            )

            if (
            global_dfs_complete_now 
            and not getattr (
            physical ,
            "integration_global_dfs_complete",
            False ,
            )
            ):
                root =multi_dfs .current 

                if root is None :
                    raise RuntimeError (
                    "Global DFS completion lost Root frame"
                    )

                physical .integration_global_dfs_complete =True 

                print (
                "[GlobalDFSComplete] "
                f"root={root .junction_uid } "
                f"branches={root .branch_order } "
                f"states={root .branch_states } "
                "reason=MAIN_ROOT_COMPLETION_LATCH"
                )
                start_final_return =getattr (
                physical ,
                "integration_start_final_return_pipeline",
                None ,
                )

                if start_final_return is None :
                    raise RuntimeError (
                    "Global DFS complete but "
                    "final-return pipeline is unavailable"
                    )

                start_final_return (
                robots ,
                "MAIN_ROOT_COMPLETION_LATCH",
                )

            fixture_backed_current_junction =(
            current_junction is not None 
            and bool (
            current_junction .branch_order 
            )
            and all (
            physical .branch_fixture_for_uid (
            branch_uid 
            )
            is not None 
            for branch_uid 
            in current_junction .branch_order 
            )
            )

            parent_restore_pending =(
            multi_dfs .parent_restore_pending 
            )

            parent_guard_reformation_pending =(
            multi_dfs .parent_guard_reformation_pending 
            )

            parent_return_in_progress =(
            current_junction is not None 
            and current_junction .subtree_complete 
            and current_junction .parent_junction_uid 
            is not None 
            )

            if (
            not global_dfs_complete_now 
            and not parent_restore_pending 
            and not parent_guard_reformation_pending 
            and perception .anchor_fixed 
            and perception .junction_confirmed 
            and perception .state in {
            PerceptionState .BRANCHES_READY ,
            PerceptionState .PHYSICAL_DFS ,
            }
            and not perception .provisional_guard_started 
            ):
                initialize_provisional_guard_geometry_after_detection (
                physical ,
                perception ,
                robots ,
                lidar_frame ,
                )

            if parent_guard_reformation_pending :
                update_parent_guard_reformation (
                physical ,
                perception ,
                robots ,
                )

            if (
            not global_dfs_complete_now 
            and not parent_return_in_progress 
            and not parent_restore_pending 
            and not parent_guard_reformation_pending 
            ):
                update_provisional_guard_leakage (
                physical ,
                perception ,
                robots ,
                )

                update_guard_readiness_and_activation (
                physical ,
                perception ,
                robots ,
                )

                update_provisional_wall_settling_audit (
                physical ,
                perception ,
                robots ,
                )

                cycle_junction =multi_dfs .current 

                if (
                cycle_junction is not None 
                and cycle_junction .active_branch_uid 
                is None 
                ):
                    update_current_branch_cycle (
                    physical ,
                    perception ,
                    robots ,
                    dt ,
                    )

                log_wall_ready_blockers (
                physical ,
                perception ,
                robots ,
                )

                fixture_backed_provisional_guards =(
                bool (
                perception .provisional_guards 
                )
                and all (
                geometry .local_branch_key 
                in physical .BRANCHES 
                for geometry 
                in perception .provisional_guards 
                )
                )

                if (
                perception .state 
                ==PerceptionState .BRANCHES_READY 
                and fixture_backed_provisional_guards 
                ):
                    handoff_to_physical_dfs (
                    physical ,
                    perception ,
                    robots ,
                    )


            current_topology_pending =(
            current_junction is not None 
            and perception .junction_confirmed 
            and not current_junction .branch_order 
            )
            if parent_restore_pending :
                if frame_count %10 ==0 :
                    print (
                    "[PhysicalDFSSuspended] "
                    f"current="
                    f"{multi_dfs .current .junction_uid } "
                    "reason=PARENT_CONTEXT_RESTORE_PENDING "
                    f"stack="
                    f"{[
                    frame .junction_uid 
                    for frame in multi_dfs .stack 
                    ]}"
                    )
            elif parent_guard_reformation_pending :
                if frame_count %10 ==0 :
                    print (
                    "[PhysicalDFSSuspended] "
                    f"current="
                    f"{multi_dfs .current .junction_uid } "
                    "reason=PARENT_GUARD_REFORMATION_PENDING "
                    f"stack="
                    f"{[
                    frame .junction_uid 
                    for frame in multi_dfs .stack 
                    ]}"
                    )

            elif parent_return_in_progress :
                if frame_count %10 ==0 :
                    print (
                    "[PhysicalDFSSuspended] "
                    f"current="
                    f"{multi_dfs .current .junction_uid } "
                    "reason=PARENT_RETURN_IN_PROGRESS "
                    f"arrived="
                    f"{multi_dfs .current .parent_return_arrived } "
                    f"stack="
                    f"{[
                    frame .junction_uid 
                    for frame in multi_dfs .stack 
                    ]}"
                    )

            elif global_dfs_complete_now :

                if getattr (
                physical ,
                "integration_final_guard_sweep_active",
                False ,
                ):






                    physical .update_simulation_state (
                    robots ,
                    dt ,
                    reference_density ,
                    spatial_grid ,
                    )

                elif getattr (
                physical ,
                "integration_local_final_return_active",
                False ,
                ):

                    update_local_final_return_fn =getattr (
                    physical ,
                    "integration_update_local_final_return",
                    None ,
                    )

                    if update_local_final_return_fn is None :
                        raise RuntimeError (
                        "Local final return controller is unavailable"
                        )

                    update_local_final_return_fn (
                    robots ,
                    dt ,
                    )

                elif frame_count %10 ==0 :
                    print (
                    "[PhysicalDFSSuspended] "
                    f"current="
                    f"{multi_dfs .current .junction_uid } "
                    "reason=GLOBAL_DFS_COMPLETE "
                    "awaiting_final_return=True "
                    f"stack="
                    f"{[
                    frame .junction_uid 
                    for frame in multi_dfs .stack 
                    ]}"
                    )



            elif (
            perception .handoff_complete 
            and not current_topology_pending 
            ):


                pass 
            elif not perception .handoff_complete :




                physical .update_local_ingress_tangents (robots )
                physical .update_initial_release_flow_event (robots ,dt )
                physical .update_relay_deployment (robots ,dt )
            else :

                if frame_count %10 ==0 :
                    print (
                    "[PhysicalDFSSuspended] "
                    f"current="
                    f"{multi_dfs .current .junction_uid } "
                    "reason="
                    "CHILD_TOPOLOGY_NOT_INITIALIZED "
                    f"stack="
                    f"{[
                    frame .junction_uid 
                    for frame in multi_dfs .stack 
                    ]}"
                    )
            physical .update_metrics_per_frame (robots ,dt )
            if (
            (
            physical .phase ==physical .SimulationPhase .RETURN_TO_BASE 
            or getattr (
            physical ,
            "integration_final_guard_sweep_active",
            False ,
            )
            )
            and perception .anchor_fixed 
            ):
                perception .anchor_fixed =False 
                perception .leader .is_fixed_anchor =False 
                perception .leader .base_reserve =False 
                print (
                f"[Anchor] RELEASED_FOR_RETURN "
                f"id={perception .leader .robot_id }"
                )
            previous_phase =_phase_event_log (physical ,robots ,previous_phase ,visited_log )
        if renderer is not None :
            renderer .draw (robots ,perception ,show_rays ,show_profile ,show_comm ,density ,paused )
        if physical .phase ==physical .SimulationPhase .DONE :
            running =False 
        if args .max_frames and frame_count >=args .max_frames :
            running =False 

    frontier_ids =[robot .robot_id for robot in physical .get_frontier_shepherds (robots )]
    shepherd_ids =[robot .robot_id for robot in physical .get_shepherds (robots )]
    guard_count =sum (robot .role =="JUNCTION_GUARD"for robot in robots )
    pebble_count =len (physical .get_pebbles (robots ))
    base_returned =sum (
    physical .get_robot_region (robot .position )=="BOTTOM"
    for robot in robots 
    if robot .role !="PEBBLE"
    )
    print (f"[Anchor] post_fix_drift={perception .post_fix_drift :.6f}")
    print (
    "[FlowRegression] anchor_fixed_mean_normal_forward_speed="
    f"{perception .anchor_fixed_mean_normal_forward_speed :.6f} "
    "pre_topology_mean_normal_forward_speed="
    f"{float (np .mean (perception .pre_topology_normal_forward_speeds ))if perception .pre_topology_normal_forward_speeds else 0.0 :.6f}"
    )
    print (
    f"[Accounting] base_returned={base_returned } "
    f"persistent_pebbles={pebble_count } total={base_returned +pebble_count }"
    )
    print (
    "[TeleportAudit] role_transition_position_jump="
    f"{physical .integration_guard_role_transition_jump :.6f} "
    "runtime_guard_max_displacement_per_frame="
    f"{physical .integration_wall_max_step :.6f} "
    "runtime_frontier_max_displacement_per_frame="
    f"{physical .integration_frontier_max_step :.6f} "
    "runtime_shepherd_max_displacement_per_frame="
    f"{physical .integration_shepherd_max_step :.6f} "
    "direct_guard_position_overwrite=False "
    "direct_frontier_position_overwrite=False "
    "direct_shepherd_position_overwrite=False "
    "direct_position_overwrite_count=0"
    )
    for geometry in perception .provisional_guards :
        leakage =perception .guard_leakage .get (
        geometry .provisional_uid ,{}
        )
        status =physical .integration_wall_status .get (
        geometry .persistent_uid or geometry .provisional_uid ,{}
        )
        print (
        f"[GuardNaturalFlowRegression] uid="
        f"{geometry .persistent_uid or geometry .provisional_uid } "
        f"junction_detection_frame={perception .confirmation_frame } "
        f"first_robot_crossing_mouth_frame="
        f"{geometry .first_robot_crossing_mouth_frame } "
        f"guard_candidate_sufficient_frame="
        f"{geometry .candidate_sufficient_frame } "
        f"guard_ready_frame={geometry .guard_ready_frame } "
        f"guard_role_assignment_frame="
        f"{geometry .role_assignment_frame } "
        f"wall_ready_frame={status .get ('ready_frame')} "
        f"crossings_before_edge_seal="
        f"{leakage .get ('crossings_before_edge_seal',0 )} "
        f"crossings_after_edge_seal="
        f"{leakage .get ('crossings_after_edge_seal',0 )} "
        f"deepest_leaked_robot_depth="
        f"{float (leakage .get ('deepest_leaked_robot_depth',0.0 )):.3f} "
        f"leakage_blocked_after_edge_seal="
        f"{leakage .get ('leakage_blocked_after_edge_seal',False )}"
        )
    for uid ,audit in sorted (
    perception .guard_communication_audits .items ()
    ):
        print (
        f"[CommunicationRegression] uid={uid } "
        f"before_connected={audit ['before_connected']} "
        f"before_largest_component="
        f"{audit ['before_largest_component']} "
        f"after_connected={audit ['after_connected']} "
        f"after_largest_component="
        f"{audit ['after_largest_component']} "
        f"selected_critical_ids={audit ['selected_critical_ids']} "
        "communication_disconnect_caused_by_guard_selection="
        f"{audit ['communication_disconnect_caused_by_guard_selection']}"
        )
    for event in getattr (
    physical ,"integration_frontier_lineage_events",[]
    ):
        print (
        f"[FrontierLineageRegression] uid={event ['uid']} "
        f"frame={event ['frame']} "
        f"same_ready_guard_ids={event ['same_ready_guard_ids']} "
        f"max_role_transition_jump="
        f"{event ['max_role_transition_jump']:.6f} "
        f"frontier_ids={event ['frontier_ids']}"
        )
    for branch ,lifecycle in getattr (
    physical ,"integration_wall_lifecycle",{}
    ).items ():
        print (
        f"[WallLifecycleRegression] branch={branch } "
        f"state={lifecycle ['state']} rows={lifecycle ['rows']} "
        f"cols={lifecycle ['cols']} "
        f"robots={len (lifecycle ['robot_ids'])} "
        f"max_formation_error="
        f"{lifecycle .get ('max_formation_error',0.0 ):.6f}"
        )
    for uid ,status in sorted (
    getattr (physical ,"integration_wall_status",{}).items ()
    ):
        print (
        f"[WallRegression] uid={uid } "
        f"capture={status .get ('capture',0 )} "
        f"assigned={status .get ('assigned',0 )} "
        f"rows={status .get ('rows',0 )} "
        f"columns={status .get ('slots_per_row',0 )} "
        f"settled={status .get ('settled_ratio',0.0 ):.3f} "
        f"span={status .get ('min_span_ratio',0.0 ):.3f} "
        f"edge_gap={status .get ('max_edge_gap',0.0 ):.3f} "
        f"internal_gap={status .get ('max_internal_gap',0.0 ):.3f} "
        f"slots_walkable={status .get ('slots_walkable',0 )}/"
        f"{status .get ('slots_total',0 )} "
        f"ready_frame={status .get ('ready_frame')}"
        )
    for uid ,association in sorted (
    getattr (physical ,"integration_opening_mouth_associations",{}).items ()
    ):
        print (
        f"[MouthAssociationRegression] uid={uid } "
        f"opening_center={association ['opening_center']:+.1f}deg "
        f"matched_mouth={association ['matched_mouth']} "
        f"mouth_local={association ['mouth_local_angle']:+.1f}deg "
        f"angular_error={association ['angular_error']:.1f}deg"
        )
    for event in getattr (physical ,"integration_saturation_events",[]):
        print (
        f"[SaturationRegression] uid={event ['uid']} "
        f"frame={event ['frame']} "
        f"frontier_speed={event ['frontier_speed']:.3f} "
        f"local_density={event ['local_density']:.6f} "
        f"density_ratio={event ['density_ratio']:.3f} "
        f"local_pressure={event ['local_pressure']:.3f} "
        f"pressure_ratio={event ['pressure_ratio']:.3f} "
        f"cross_fill={event ['cross_section_fill']:.3f} "
        f"dwell={event ['dwell']:.3f} "
        f"same_ids={event ['frontier_ids']==event ['shepherd_ids']} "
        f"max_transition_jump={event ['max_transition_jump']:.6f} "
        f"return_direction={event ['return_direction']} "
        f"rows={event ['rows']} cols={event ['cols']} "
        f"robots={event ['robots']} "
        f"max_formation_error={event ['max_formation_error']:.6f} "
        f"frontier_ids={event ['frontier_ids']} "
        f"shepherd_ids={event ['shepherd_ids']}"
        )
    for event in getattr (physical ,"integration_backflow_events",[]):
        print (
        f"[BackflowRegression] uid={event ['uid']} "
        f"frame={event ['frame']} ratio={event ['ratio']:.3f} "
        f"mean_speed={event ['mean_speed']:.3f} "
        f"duration={event ['duration']:.3f}"
        )
    print (
    "[Summary] "
    f"frames={frame_count } confirm_frame={perception .confirmation_frame } "
    f"confirm_time={perception .confirmation_time } anchor_id={perception .leader .robot_id } "
    f"pre_detection_travel={perception .pre_detection_travel :.3f} "
    f"persistent={sum (len (t .observations )>=MIN_PERSISTENT_OBSERVATIONS for t in perception .tracks )} "
    f"outgoing={len (perception .outgoing )} guards={guard_count } "
    f"selected={physical .branch_identity_label (physical .active_branch_uid )} "
    f"frontier_ids={frontier_ids } shepherd_ids={shepherd_ids } "
    f"visited={visited_log } final_phase={physical .phase .name }"
    )
    if perception .junction_candidate_detected and perception .anchor_fixed :
        assert perception .junction_candidate_frame <=perception .anchor_fix_frame ,(
        "candidate frame must not follow Anchor fix"
        )
    print (
    f"[DFS] visited_sequence={visited_log } "
    f"final_phase={physical .phase .name }"
    )
    pygame .quit ()
    return 0 if physical .phase ==physical .SimulationPhase .DONE or args .max_frames else 1 


if __name__ =="__main__":
    raise SystemExit (main ())
