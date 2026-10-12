from __future__ import annotations

import builtins
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import pytest

from press.t102_dynamic_precontact_core import (
    DynamicPrepareBlocked, FKState, FrozenModel, FrozenSession, IKAudit, PreparedPlan,
    STOPPED_MODE, SUCCESS_CODE, canonical_hash, orient_outward_normal,
    prepare_dynamic_plan, verify_frozen_plan,
)
from press.t102_dynamic_precontact_core import _segment_distance
from press.t102_dynamic_precontact_core import _validate_standard_standby_acceptance


class FakeModel:
    def __init__(self):
        self.boot_epoch = "executor-boot"
        self.model_loaded = True
        self.model_failure_reason = ""
        self.execution_admission_ready = True
        self.admission_failure_reason = ""
        self.hashes = {"kinematic_model_hash":"a"*64,"collision_proxy_hash":"b"*64,"tool_model_hash":"c"*64,"template_hash":"d"*64,"calibration_hash":"9"*64,"accepted_camera_record_hash":"4"*64,"fn_artifact_hash":"8"*64,"fn_acceptance_hash":"6"*64,"standby_acceptance_hash":"5"*64,"source_registry_hash":"7"*64}
        self.source_bindings = {
            "calibration_source_path_relative":"accepted/calibration.json",
            "calibration_source_path_absolute":"/fixture/accepted/calibration.json",
            "calibration_validation_identity":"fake-calibration-validation",
            "fn_pointer_path_relative":"candidate/pointer.json",
            "fn_pointer_path_absolute":"/fixture/candidate/pointer.json",
            "fn_artifact_path_relative":"candidate/fn.json",
            "fn_artifact_path_absolute":"/fixture/candidate/fn.json",
            "fn_acceptance_path_relative":"accepted/fn.json",
            "fn_acceptance_path_absolute":"/fixture/accepted/fn.json",
            "fn_candidate_sidecar_path_relative":"candidate/fn.SHA256SUMS",
            "fn_candidate_sidecar_path_absolute":"/fixture/candidate/fn.SHA256SUMS",
            "fn_validation_identity":"fake-fn-validation",
            "collision_proxy_path_relative":"prototype/collision.json",
            "collision_proxy_path_absolute":"/fixture/prototype/collision.json",
            "collision_proxy_validation_identity":"fake-collision-validation",
            "standby_acceptance_path_relative":"prototype/standby.json",
            "standby_acceptance_path_absolute":"/fixture/prototype/standby.json",
            "standby_validation_identity":"fake-standby-validation",
        }
        self.policy = {
            "start_state_admission_mode":"FRESH_CONVERGED_DISARMED_LEGAL_Q0",
            "d_pre_m": .07, "d_stage_m": .10, "normal_reference_min_dot": .35,
            "normal_unit_tolerance": .02, "orientation_window_deg": 35.,
            "axis_residual_limit_deg": 8., "position_residual_limit_m": .0025,
            "cartesian_sample_count": 3, "segment_sample_count": 3,
            "maximum_joint_jump_rad": .45, "joint_limit_buffer_rad": .01,
            "maximum_joint_reversal_rad":.012,"jacobian_condition_limit":2500.,"manipulability_floor":1e-8,
            "joint_hard_limits_rad": [[-1.,1.]]*6, "encoder_step_rad": .001,
            "per_command_caps_rad": [.05]*6, "minimum_useful_delta_rad": .002,
            "minimum_waypoints": 3, "maximum_waypoints": 6, "ticket_ttl_s": 10.,
            "q0_drift_limit_rad": .003, "wall_margin_m": .001,
            "maximum_button_plane_offset_m": .003,
            "minimum_target_confidence":.7,"maximum_cart_confirmation_age_s":30.,"maximum_q0_age_s":1.,
            "maximum_camera_info_stamp_skew_s":.5,
            "standard_standby_rad":[.12,0,0,0,0,0],"standard_standby_tolerance_rad":.001,
            "calibration_identity":"fake-calibration",
            "capsules": [{"name":"upper","joint_a":1,"joint_b":2,"radius_m":.005},{"name":"tool","joint_a":3,"joint_b":4,"radius_m":.005}],
            "self_pairs": [["upper","tool"]], "base_check_capsules":["tool"], "bracket_check_capsules":["tool"],
            "base_proxy":{"center_m":[-.5,.5,.5],"radius_m":.01},
            "camera_bracket_proxy":{"center_m":[-.5,-.5,.5],"radius_m":.01},
        }
        self.seed_hashes=[]

    def fk(self,q):
        x=float(q[0]); points=((-.2,0,0),(x+.03,.04,0),(x+.04,.04,0),(x+.05,-.04,0),(x+.06,-.04,0),(x+.07,0,0),(x,0,0))
        return FKState((x,0.,0.),(-1.,0.,0.),points)

    def solve(self,seed,target,desired):
        self.seed_hashes.append(canonical_hash(list(seed)))
        q=list(seed); q[0]=float(target[0])
        return IKAudit(tuple(q),0.,0.,10.,.1,2,canonical_hash(list(seed)))

    def jacobian_metrics(self,q):
        return 10.,.1

    def _validate_limits(self,q):
        for i,x in enumerate(q):
            if not -.99 <= float(x) <= .99: raise DynamicPrepareBlocked("JOINT_LIMIT_REJECTED",str(i))


def session(model, **changes):
    now=1_000_000_000_000
    q=(.12,0.,0.,0.,0.,0.)
    value={"session_id":"","live_boot_epoch":"live","qc_boot_epoch":"qc","executor_boot_epoch":model.boot_epoch,
      "cart_pose_epoch":"cart","cart_confirmed_stamp_ns":now-1_000_000,"stopped_proof_mode":STOPPED_MODE,
      "target_label":"Button 1","detection_id":"det","confidence":.99,"target_stamp_ns":now-1_000_000,
      "source_frame":"gemini_depth_optical_frame","canonical_frame":"roarm_base_link",
      "p_button_base_m":(.10,0.,0.),"wall_surface_point_base_m":(.10,0.,0.),
      "candidate_normal_base":(1.,0.,0.),"camera_center_base_m":(.8,0.,0.),
      "robot_reference_base_m":(.6,0.,0.),"rgbd_identity":"rgbd","rgbd_stamp_ns":now-1_000_000,
      "camera_info_identity":"camera-info","camera_info_hash":"e"*64,"camera_info_stamp_ns":now-1_000_000,
      "d2c_identity":"d2c-calibration","d2c_hash":"f"*64,"d2c_stamp_ns":now-1_000_000,
      "tf_hash":"1"*64,"tf_stamp_ns":now-1_000_000,"calibration_identity":"fake-calibration","calibration_hash":model.hashes["calibration_hash"],
      "calibration_observation_path":"/accepted/calibration.json",
      "calibration_observation_status":"b_section_accepted_relative_unpublished",
      "calibration_observation_validation_identity":"fake-calibration-validation",
      "calibration_acceptance_artifact":"/accepted/calibration_acceptance.json",
      "calibration_acceptance_sha256":"9"*64,
      "accepted_camera_record_hash":model.hashes["accepted_camera_record_hash"],
      **model.source_bindings,"fn_artifact_hash":model.hashes["fn_artifact_hash"],"fn_acceptance_hash":model.hashes["fn_acceptance_hash"],"standby_acceptance_hash":model.hashes["standby_acceptance_hash"],"source_registry_hash":model.hashes["source_registry_hash"],
      "kinematic_model_hash":model.hashes["kinematic_model_hash"],"collision_proxy_hash":model.hashes["collision_proxy_hash"],"tool_model_hash":model.hashes["tool_model_hash"],"template_hash":model.hashes["template_hash"],
      "q0_rad":q,"q0_stamp_ns":now-1_000_000,
      "q0_hash":canonical_hash(list(q))}
    value.update(changes)
    unsigned=dict(value); unsigned.pop("session_id",None); value["session_id"]=canonical_hash(unsigned)
    return FrozenSession.from_mapping(value),now


def prepared(model=None):
    model=model or FakeModel(); s,now=session(model)
    return model,s,prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=5.)


@pytest.mark.parametrize("label", ["2", "open", "close", "up", "down", "b1", "g"])
def test_dynamic_session_binds_selected_identity_and_freshness(label):
    selected = {
        "selection_id": "operator-click",
        "label": label,
        "detection_id": "initial-frame-id",
        "semantic_track_id": "chosen-track",
        "after_stamp_ns": 999_998_000_000,
        "pressable": True,
    }
    value, _ = session(FakeModel(), target_label=label, target_selection=selected)
    assert value.target_label == label
    assert value.target_selection == selected
    assert FrozenSession.from_mapping(asdict(value)).session_id == value.session_id
    edited = asdict(value)
    edited["target_selection"]["semantic_track_id"] = "other-physical-button"
    with pytest.raises(DynamicPrepareBlocked, match="session_id hash"):
        FrozenSession.from_mapping(edited)


def test_legacy_session_round_trip_preserves_original_hash_without_selection():
    value, _ = session(FakeModel())
    assert value.target_selection is None
    assert FrozenSession.from_mapping(asdict(value)).session_id == value.session_id


def test_non_button1_dynamic_session_requires_explicit_selection():
    with pytest.raises(DynamicPrepareBlocked, match="selected target binding is required"):
        session(FakeModel(), target_label="2")


def representative_real_session(model, d_current):
    now = 2_000_000_000
    q0 = tuple(float(value) for value in model.policy["standard_standby_rad"])
    state = model.fk(q0)
    n_out = tuple(-value for value in state.axis)
    p_button = tuple(state.tcp[index] - float(d_current) * n_out[index] for index in range(3))
    value = {
        "live_boot_epoch":"live-real-fixture", "qc_boot_epoch":"qc-real-fixture",
        "executor_boot_epoch":model.boot_epoch, "cart_pose_epoch":"cart-real-fixture",
        "cart_confirmed_stamp_ns":now-100_000_000, "stopped_proof_mode":STOPPED_MODE,
        "target_label":"Button 1", "detection_id":f"real-{d_current}", "confidence":.99,
        "target_stamp_ns":now-50_000_000, "source_frame":"camera",
        "canonical_frame":"roarm_base_link", "p_button_base_m":p_button,
        "wall_surface_point_base_m":p_button, "candidate_normal_base":n_out,
        "camera_center_base_m":tuple(p_button[index]+.5*n_out[index] for index in range(3)),
        "robot_reference_base_m":tuple(p_button[index]+.2*n_out[index] for index in range(3)),
        "rgbd_identity":"representative-rgbd", "rgbd_stamp_ns":now-50_000_000,
        "camera_info_identity":"representative-camera-info", "camera_info_hash":"e"*64,
        "camera_info_stamp_ns":now-50_000_000, "d2c_identity":"representative-d2c",
        "d2c_hash":"f"*64, "d2c_stamp_ns":now-50_000_000, "tf_hash":"1"*64,
        "tf_stamp_ns":now-50_000_000, "calibration_identity":model.policy["calibration_identity"],
        "calibration_hash":model.hashes["calibration_hash"], **model.source_bindings,
        "calibration_observation_path":model.source_bindings["calibration_source_path_absolute"],
        "calibration_observation_status":"b_section_accepted_relative_unpublished",
        "calibration_observation_validation_identity":model.source_bindings["calibration_validation_identity"],
        "calibration_acceptance_artifact":"config/calibration/accepted/fake_acceptance.json",
        "calibration_acceptance_sha256":"9"*64,
        "accepted_camera_record_hash":model.hashes["accepted_camera_record_hash"],
        "fn_artifact_hash":model.hashes["fn_artifact_hash"],
        "fn_acceptance_hash":model.hashes["fn_acceptance_hash"],
        "standby_acceptance_hash":model.hashes["standby_acceptance_hash"],
        "source_registry_hash":model.hashes["source_registry_hash"],
        "kinematic_model_hash":model.hashes["kinematic_model_hash"],
        "collision_proxy_hash":model.hashes["collision_proxy_hash"],
        "tool_model_hash":model.hashes["tool_model_hash"],
        "template_hash":model.hashes["template_hash"], "q0_rad":q0,
        "q0_stamp_ns":now-10_000_000, "q0_hash":canonical_hash(list(q0)),
    }
    return FrozenSession.from_mapping({"session_id":canonical_hash(value), **value}), now


def exact_agx_button1_session(model):
    """Freeze the exact zero-motion Button1 geometry captured on AGX."""

    frozen, now = representative_real_session(model, .13)
    value = asdict(frozen)
    p_button = (0.3261430422, 0.0716131394, 0.3102506570)
    n_out = (-0.9943872928, -0.04288355, -0.0967207996)
    q0 = (
        0.035281558, -1.194971034, 2.331650798,
        0.464796179, -1.510971076, 0.001533980589793149,
    )
    value.update({
        "detection_id": "agx-button1-exact-regression",
        "p_button_base_m": p_button,
        "wall_surface_point_base_m": p_button,
        "candidate_normal_base": n_out,
        "camera_center_base_m": tuple(p_button[i] + .5*n_out[i] for i in range(3)),
        "robot_reference_base_m": tuple(p_button[i] + .2*n_out[i] for i in range(3)),
        "q0_rad": q0,
        "q0_hash": canonical_hash(list(q0)),
    })
    value.pop("session_id", None)
    value["session_id"] = canonical_hash(value)
    return FrozenSession.from_mapping(value), now


def latest_agx_button1_session(model):
    """Regression for the 2026-08-17 19:23 frozen target that needed 7
    slots with midpoint insertion despite having a valid six-slot route."""

    frozen, now = exact_agx_button1_session(model)
    value = asdict(frozen)
    value.update({
        "detection_id": "agx-button1-20260817-1923",
        "p_button_base_m": (
            0.32615742218747634,
            0.07175871655631161,
            0.3101126144540815,
        ),
        "wall_surface_point_base_m": (
            0.32615742218747634,
            0.07175871655631161,
            0.3101126144540815,
        ),
        "candidate_normal_base": (
            -0.994657347379591,
            -0.04217884017157991,
            -0.09422158322579957,
        ),
        "camera_center_base_m": (
            -0.09561468064577146,
            0.15764998499287494,
            0.1881160026534767,
        ),
        "robot_reference_base_m": (
            0.09038005447853208,
            0.0031897405005963365,
            0.20727511065770687,
        ),
    })
    value.pop("session_id", None)
    value["session_id"] = canonical_hash(value)
    return FrozenSession.from_mapping(value), now


def resumed_agx_button1_session(model):
    """Regression for the 2026-08-18 resumed Live capture.

    This target is farther from the camera than the August-17 capture and its
    bounded current-to-stage solve turns joint 3 immediately after the final
    initial-settlement sample.  Six larger Cartesian samples retain every
    command cap and the exact 3-6 waypoint contract.
    """

    frozen, now = exact_agx_button1_session(model)
    value = asdict(frozen)
    p_button = (
        0.36434571562507073,
        0.09693868792719831,
        0.3176931785280892,
    )
    n_out = (
        -0.9900433456282739,
        -0.09081762405059192,
        -0.10754688716545764,
    )
    q0 = (
        0.036815539, -1.188835111, 2.345456625,
        0.464796179, -1.512505057, 0.001533980589793149,
    )
    value.update({
        "detection_id": "agx-button1-20260818-resume",
        "p_button_base_m": p_button,
        "wall_surface_point_base_m": p_button,
        "candidate_normal_base": n_out,
        "camera_center_base_m": (
            -0.09561468064577146,
            0.15764998499287494,
            0.1881160026534767,
        ),
        "robot_reference_base_m": (0.092, 0.003, 0.203),
        "q0_rad": q0,
        "q0_hash": canonical_hash(list(q0)),
    })
    value.pop("session_id", None)
    value["session_id"] = canonical_hash(value)
    return FrozenSession.from_mapping(value), now


def physical_attempt4_agx_button1_session(model):
    """Exact target/q0 from the first physical Dynamic Move on 2026-08-18."""

    frozen, now = exact_agx_button1_session(model)
    value = asdict(frozen)
    p_button = (
        0.34652703286036185,
        0.07705991685815322,
        0.31437995977441263,
    )
    n_out = (
        -0.9928229599280098,
        -0.05198808873610371,
        -0.10770240883727891,
    )
    q0 = (
        0.036815539, -1.188835111, 2.345456625,
        0.464796179, -1.512505057, 0.001533980589793149,
    )
    value.update({
        "detection_id": "agx-button1-20260818-physical-attempt4",
        "p_button_base_m": p_button,
        "wall_surface_point_base_m": p_button,
        "candidate_normal_base": n_out,
        "camera_center_base_m": tuple(
            p_button[index] + .5 * n_out[index] for index in range(3)
        ),
        "robot_reference_base_m": tuple(
            p_button[index] + .2 * n_out[index] for index in range(3)
        ),
        "q0_rad": q0,
        "q0_hash": canonical_hash(list(q0)),
    })
    value.pop("session_id", None)
    value["session_id"] = canonical_hash(value)
    return FrozenSession.from_mapping(value), now


def physical_attempt6_agx_button1_session(model):
    """Exact Live geometry/q0 reconstructed from the sealed attempt-6 bag."""

    frozen, now = exact_agx_button1_session(model)
    value = asdict(frozen)
    value.update({
        "detection_id": "agx-button1-20260818-physical-attempt6",
        "p_button_base_m": (
            0.346534594297844,
            0.077028331079861,
            0.3143399781670778,
        ),
        "wall_surface_point_base_m": (
            0.346534594297844,
            0.077028331079861,
            0.3143399781670778,
        ),
        "candidate_normal_base": (
            -0.9923537516094603,
            -0.05385407275292763,
            -0.11105750994215675,
        ),
        "camera_center_base_m": (
            -0.09561468064577146,
            0.15764998499287494,
            0.1881160026534767,
        ),
        "robot_reference_base_m": (
            0.09173653906367048,
            0.005353166133988121,
            0.20257261411747793,
        ),
        "q0_rad": (
            0.05829127,
            -1.188835111,
            2.345456625,
            0.464796179,
            -1.510971076,
            0.001533980589793149,
        ),
    })
    value["q0_hash"] = canonical_hash(list(value["q0_rad"]))
    value.pop("session_id", None)
    value["session_id"] = canonical_hash(value)
    return FrozenSession.from_mapping(value), now


def current_approximate_agx_button1_session(model):
    """Fresh 2026-08-19 Button-1 geometry transformed by the accepted camera calibration."""

    frozen, now = physical_attempt6_agx_button1_session(model)
    value = asdict(frozen)
    p_button = (
        0.31375619002453936,
        0.09563437835486456,
        0.3111767823376975,
    )
    value.update({
        "detection_id": "agx-button1-20260819-current-approximate",
        "p_button_base_m": p_button,
        "wall_surface_point_base_m": p_button,
        "candidate_normal_base": (
            -0.9907548239586185,
            -0.0637973377947024,
            -0.11972793530766088,
        ),
        "camera_center_base_m": (
            -0.09561468064577146,
            0.15764998499287494,
            0.1881160026534767,
        ),
        "robot_reference_base_m": (0.092, 0.004, 0.206),
        "q0_rad": (
            0.046019424,
            -1.188835111,
            2.331650798,
            0.470932102,
            -1.517106999,
            0.001533980589793149,
        ),
    })
    value["q0_hash"] = canonical_hash(list(value["q0_rad"]))
    value.pop("session_id", None)
    value["session_id"] = canonical_hash(value)
    return FrozenSession.from_mapping(value), now


def current_20260820_lateral_button1_session(model):
    """Fresh stabilized Button-1 geometry after the cart position changed."""

    frozen, now = physical_attempt6_agx_button1_session(model)
    value = asdict(frozen)
    q0 = (
        0.042951462,
        -1.191903072,
        2.345456625,
        0.464796179,
        -1.517106999,
        0.001533980589793149,
    )
    p_button = (
        0.3446915897629776,
        0.14781081452764108,
        0.28517400954703265,
    )
    value.update({
        "detection_id": "agx-button1-20260820-current-lateral",
        "confidence": 0.7358420491,
        "p_button_base_m": p_button,
        "wall_surface_point_base_m": p_button,
        "candidate_normal_base": (
            -0.9925189048594556,
            -0.11432293533679419,
            -0.042854287446719035,
        ),
        "camera_center_base_m": (
            -0.09561468064577146,
            0.15764998499287494,
            0.1881160026534767,
        ),
        "robot_reference_base_m": model.fk(q0).tcp,
        "q0_rad": q0,
        "q0_hash": canonical_hash(list(q0)),
    })
    value.pop("session_id", None)
    value["session_id"] = canonical_hash(value)
    return FrozenSession.from_mapping(value), now


def code(exc): return exc.value.code


def test_normal_sign_property_and_positive_precontact():
    a=orient_outward_normal((1,0,0),(0,0,0),(1,0,0),(2,0,0))
    b=orient_outward_normal((-1,0,0),(0,0,0),(1,0,0),(2,0,0))
    assert a.n_out==b.n_out==(1.,0.,0.)
    assert sum(a.n_out[i]*((.2+.07*a.n_out[i] if i==0 else 0)-(.2 if i==0 else 0)) for i in range(3))==pytest.approx(.07)


@pytest.mark.parametrize("normal,camera,robot", [((0,1,0),(1,0,0),(2,0,0)),((1,0,0),(1,0,0),(-1,0,0)),((float('nan'),0,0),(1,0,0),(2,0,0)),((.5,0,0),(1,0,0),(2,0,0))])
def test_normal_ambiguity_conflict_nonfinite_nonunit(normal,camera,robot):
    with pytest.raises(DynamicPrepareBlocked) as exc: orient_outward_normal(normal,(0,0,0),camera,robot)
    assert code(exc) in {"NORMAL_DIRECTION_AMBIGUOUS","NONFINITE_INPUT"}


def test_session_mismatch_restart_and_stale():
    model=FakeModel()
    with pytest.raises(DynamicPrepareBlocked) as exc: session(model,tf_stamp_ns=1)
    assert code(exc)=="RGBD_TF_SESSION_MISMATCH"
    s,now=session(model,executor_boot_epoch="old")
    with pytest.raises(DynamicPrepareBlocked) as exc: prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1)
    assert code(exc)=="PLAN_INVALIDATED"
    s,_=session(model)
    with pytest.raises(DynamicPrepareBlocked) as exc: prepare_dynamic_plan(model,s,now_ns=s.target_stamp_ns+3_000_000_000,now_monotonic=1)
    assert code(exc)=="TARGET_STALE"


@pytest.mark.parametrize(
    "field,value",
    [
        ("live_boot_epoch","new-live"),
        ("qc_boot_epoch","new-qc"),
        ("tf_hash","2"*64),
        ("calibration_hash","3"*64),
        ("accepted_camera_record_hash","2"*64),
        ("fn_artifact_hash","4"*64),
            ("fn_acceptance_hash","0"*64),
            ("standby_acceptance_hash","1"*64),
        ("source_registry_hash","5"*64),
        ("fn_validation_identity","changed-fn-validation"),
        ("calibration_validation_identity","changed-calibration-validation"),
        ("fn_artifact_path_absolute","/changed/fn.json"),
        ("fn_acceptance_path_absolute","/changed/fn-acceptance.json"),
            ("collision_proxy_path_absolute","/changed/collision.json"),
            ("standby_acceptance_path_absolute","/changed/standby.json"),
        ("calibration_source_path_relative","changed/calibration.json"),
        ("detection_id","new-detection"),
    ],
)
def test_verify_rejects_any_exact_frozen_session_identity_drift(field,value):
    model,s,ticket=prepared()
    drifted=replace(s,**{field:value})
    with pytest.raises(DynamicPrepareBlocked) as exc:
        verify_frozen_plan(ticket,model=model,session=drifted,current_q0=s.q0_rad,now_monotonic=6.)
    assert code(exc)=="PLAN_INVALIDATED"


@pytest.mark.parametrize(
    "changes",
    [
        {"canonical_frame":"camera"},
        {"rgbd_stamp_ns":1},
        {"camera_info_stamp_ns":1},
        {"d2c_stamp_ns":1},
        {"camera_info_identity":""},
        {"d2c_identity":""},
    ],
)
def test_session_explicit_rgbd_camera_info_d2c_frame_binding(changes):
    model=FakeModel()
    with pytest.raises(DynamicPrepareBlocked) as exc:
        session(model,**changes)
    assert code(exc)=="RGBD_TF_SESSION_MISMATCH"


def test_surface_bound_exact_camera_info_233602ms_skew_is_preserved_and_accepted():
    model=FakeModel()
    target=1_786_961_575_258_944_000
    camera_info=1_786_961_575_492_546_000
    frozen,_=session(
        model,
        cart_confirmed_stamp_ns=target-1_000_000,
        target_stamp_ns=target,
        rgbd_stamp_ns=target,
        camera_info_stamp_ns=camera_info,
        d2c_stamp_ns=target,
        tf_stamp_ns=target,
        q0_stamp_ns=target,
    )
    ticket=prepare_dynamic_plan(model,frozen,now_ns=target+50_000_000,now_monotonic=1.)
    assert frozen.camera_info_stamp_ns == camera_info
    assert frozen.camera_info_stamp_ns-frozen.target_stamp_ns == 233_602_000
    assert ticket.plan["session"]["camera_info_stamp_ns"] == camera_info


def test_camera_info_structural_hard_cap_and_model_policy_are_both_fail_closed():
    model=FakeModel(); target=999_999_000_000
    with pytest.raises(DynamicPrepareBlocked) as exc:
        session(model,camera_info_stamp_ns=target+500_000_001)
    assert code(exc)=="RGBD_TF_SESSION_MISMATCH" and "CameraInfo stamp" in str(exc.value)

    frozen,now=session(model,camera_info_stamp_ns=target+233_602_000)
    model.policy["maximum_camera_info_stamp_skew_s"] = .2
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    assert code(exc)=="RGBD_TF_SESSION_MISMATCH" and "CameraInfo stamp policy" in str(exc.value)


@pytest.mark.parametrize("field", ["rgbd_stamp_ns", "d2c_stamp_ns", "tf_stamp_ns"])
def test_dynamic_source_stamps_retain_strict_150ms_gate(field):
    model=FakeModel(); target=999_999_000_000
    with pytest.raises(DynamicPrepareBlocked) as exc:
        session(model,**{field:target+150_000_001})
    assert code(exc)=="RGBD_TF_SESSION_MISMATCH" and "RGB-D/D2C/TF stamps" in str(exc.value)


def test_button_point_must_lie_on_bound_wall_plane():
    model=FakeModel(); s,now=session(model,wall_surface_point_base_m=(.2,0.,0.))
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1.)
    assert code(exc)=="RGBD_TF_SESSION_MISMATCH"


def test_sequential_seed_waypoints_quantization_caps_and_no_fallback():
    model,s,ticket=prepared()
    assert len(model.seed_hashes)==6 and len(set(model.seed_hashes))>1
    assert 3<=len(ticket.plan["waypoints_rad"])<=6
    assert ticket.plan["fixed_v2_fallback"] is False
    indices=[item["cartesian_sample_index"] for item in ticket.plan["post_quantization"]["waypoint_audit"]]
    assert model.policy["cartesian_sample_count"]-1 in indices and indices[-1]==2*model.policy["cartesian_sample_count"]-1
    assert all(item["condition"]==10. and item["manipulability"]==.1 for item in ticket.plan["post_quantization"]["waypoint_audit"])
    assert all(abs(x/0.001-round(x/0.001))<1e-9 for q in ticket.plan["waypoints_rad"] for x in q)
    assert ticket.plan["sampled_constraints"]["code"]==SUCCESS_CODE


def test_final_segment_enforces_minimum_useful_delta_independently_of_plan_delta():
    model=FakeModel()
    model.policy["d_stage_m"]=.071
    s,now=session(model)
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1.)
    assert code(exc)=="DEADBAND_SHAPE_REJECTED"


def test_bad_solver_seed_binding_rejects():
    model=FakeModel(); s,now=session(model)
    def bad(seed,target,axis): return IKAudit(tuple(seed),0,0,1,.1,1,"bad")
    with pytest.raises(DynamicPrepareBlocked) as exc: prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1,solver=bad)
    assert code(exc)=="IK_BRANCH_DISCONTINUITY"


@pytest.mark.parametrize("field,value,expected",[("position_residual_m",.1,"IK_RESIDUAL_EXCEEDED"),("axis_residual_deg",20.,"ORIENTATION_WINDOW_MISS"),("condition",math.inf,"IK_NEAR_SINGULAR"),("manipulability",0.,"IK_NEAR_SINGULAR")])
def test_injected_solver_audit_is_independently_enforced(field,value,expected):
    model=FakeModel(); s,now=session(model)
    def bad(seed,target,axis):
        values=dict(q=(float(target[0]),0,0,0,0,0),position_residual_m=0.,axis_residual_deg=0.,condition=10.,manipulability=.1,iterations=1,seed_hash=canonical_hash(list(seed))); values[field]=value
        return IKAudit(**values)
    with pytest.raises(DynamicPrepareBlocked) as exc: prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1,solver=bad)
    assert code(exc)==expected


def test_quantized_waypoint_recomputes_and_rejects_singular_jacobian():
    model=FakeModel(); s,now=session(model)
    model.jacobian_metrics=lambda _q:(math.inf,0.)
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1.)
    assert code(exc)=="IK_NEAR_SINGULAR" and "post quantization waypoint" in str(exc.value)


def test_session_confidence_freshness_hashes_and_arbitrary_legal_q0():
    model=FakeModel()
    for changes,expected in (({"confidence":.2},"TARGET_STALE"),({"cart_confirmed_stamp_ns":1},"CART_STOP_PROOF_MISSING"),({"tool_model_hash":"0"*64},"RGBD_TF_SESSION_MISMATCH")):
        s,now=session(model,**changes)
        with pytest.raises(DynamicPrepareBlocked) as exc: prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1)
        assert code(exc)==expected
    with pytest.raises(DynamicPrepareBlocked) as exc: session(model,q0_stamp_ns=now-200_000_000)
    assert code(exc)=="RGBD_TF_SESSION_MISMATCH"
    q=(.14,0.,0.,0.,0.,0.); s,now=session(model,q0_rad=q,q0_hash=canonical_hash(list(q)))
    ticket=prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1)
    assert ticket.plan["start_state_admission"] == {
        "mode":"FRESH_CONVERGED_DISARMED_LEGAL_Q0",
        "q0_hash":s.q0_hash,
        "q0_stamp_ns":s.q0_stamp_ns,
    }
    assert ticket.plan["automatic_retry"] is False


@pytest.mark.parametrize("joint", range(6))
def test_q0_joint_limits_reject_before_solver(joint):
    model=FakeModel(); q=[.12,0.,0.,0.,0.,0.]; q[joint]=1.0
    s,now=session(model,q0_rad=q,q0_hash=canonical_hash(q)); calls=[]
    def solver(*args):
        calls.append(args)
        raise AssertionError("solver must not run")
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1,solver=solver)
    assert code(exc)=="JOINT_LIMIT_REJECTED"
    assert calls == []


def test_q0_collision_is_included_in_sampled_path():
    model=FakeModel()
    model.policy["base_proxy"]["center_m"]=[.18,-.04,0.]
    s,now=session(model)
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1)
    assert code(exc)=="BASE_BRACKET_PROXY_REJECTED"


def test_q0_orientation_is_not_a_pre_solver_shortcut_but_path_audits_remain():
    model=FakeModel(); s,now=session(model); original_fk=model.fk
    def fk(q):
        state=original_fk(q)
        if tuple(float(value) for value in q) == s.q0_rad:
            return FKState(state.tcp,(0.,1.,0.),state.points)
        return state
    model.fk=fk
    ticket=prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1)
    assert ticket.plan["post_quantization"]["axis_residual_deg"] == pytest.approx(0.)


def test_reversal_rejected_within_cartesian_leg():
    model=FakeModel(); s,now=session(model); calls=0
    def reversing(seed,target,axis):
        nonlocal calls; calls+=1; q=[float(target[0]),0.,0.,0.,0.,0.]; q[1]=.02 if calls%2 else -.02
        return IKAudit(tuple(q),0.,0.,10.,.1,1,canonical_hash(list(seed)))
    with pytest.raises(DynamicPrepareBlocked) as exc: prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1,solver=reversing)
    assert code(exc)=="IK_BRANCH_DISCONTINUITY" and "reversal" in str(exc.value)


def test_true_segment_distance_catches_interior_crossing_and_parallel_gap():
    assert _segment_distance((-1,0,0),(1,0,0),(0,-1,0),(0,1,0))==pytest.approx(0.)
    assert _segment_distance((0,0,0),(1,0,0),(0,2,0),(1,2,0))==pytest.approx(2.)


def test_symmetric_eigen_metric_and_tool_capsule_reaches_tcp():
    assert FrozenModel._eigenvalues_symmetric([[4.,0.],[0.,1.]])==pytest.approx([1.,4.])
    policy=json.loads(open("config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json").read())
    assert policy["ik_axis_weight"] == pytest.approx(.05)
    tool=next(item for item in policy["capsules"] if item["name"]=="tool")
    assert (tool["joint_a"],tool["joint_b"])==(6,7)


@pytest.mark.parametrize("weight", [None, float("nan"), 0., -.01, 1.01])
def test_boot_loader_rejects_missing_nonfinite_or_out_of_range_ik_axis_weight(tmp_path, weight):
    policy=json.loads(Path("config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json").read_text())
    if weight is None:
        policy.pop("ik_axis_weight")
    else:
        policy["ik_axis_weight"] = weight
    candidate=tmp_path / "policy.json"
    candidate.write_text(json.dumps(policy))
    with pytest.raises(DynamicPrepareBlocked) as exc:
        FrozenModel.load_once(".", candidate)
    assert code(exc) == "MODEL_LOAD_FAILED"


@pytest.mark.parametrize("skew", [None, float("nan"), 0., -.01, .500001])
def test_boot_loader_rejects_missing_nonfinite_or_out_of_range_camera_info_skew(tmp_path, skew):
    policy=json.loads(Path("config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json").read_text())
    if skew is None:
        policy.pop("maximum_camera_info_stamp_skew_s")
    else:
        policy["maximum_camera_info_stamp_skew_s"] = skew
    candidate=tmp_path / "policy.json"
    candidate.write_text(json.dumps(policy))
    with pytest.raises(DynamicPrepareBlocked) as exc:
        FrozenModel.load_once(".", candidate)
    assert code(exc) == "MODEL_LOAD_FAILED"


@pytest.mark.parametrize("mode", [None, "STANDARD_STANDBY_ONLY", ""])
def test_release_boot_requires_exact_start_state_admission_mode(tmp_path, mode):
    release=Path("config/press/releases/t102_dynamic_precontact_policy_release_20260818.json")
    policy=json.loads(release.read_text())
    if mode is None:
        policy.pop("start_state_admission_mode")
    else:
        policy["start_state_admission_mode"]=mode
    candidate=tmp_path/"policy.json"
    candidate.write_text(json.dumps(policy))
    with pytest.raises(DynamicPrepareBlocked) as exc:
        FrozenModel.load_once(".",candidate)
    assert code(exc)=="MODEL_LOAD_FAILED"


def test_release_policy_mirror_and_start_state_mode_are_exact():
    persistent=Path("config/press/releases/t102_dynamic_precontact_policy_release_20260818.json")
    mirror=Path("temp/t102_dynamic_release_20260818/policy_release.json")
    assert persistent.read_bytes()==mirror.read_bytes()
    model=FrozenModel.load_once(".",persistent)
    assert model.policy["start_state_admission_mode"]=="FRESH_CONVERGED_DISARMED_LEGAL_Q0"


def test_authoritative_jacobian_uses_boot_frozen_axis_weight():
    model=FrozenModel.load_once(".","config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json")
    q=tuple(float(value) for value in model.policy["standard_standby_rad"])
    columns_005,*_=model._jacobian_audit(q)
    model.policy["ik_axis_weight"] = .10
    columns_010,*_=model._jacobian_audit(q)
    for low,high in zip(columns_005,columns_010):
        assert high[:3] == pytest.approx(low[:3])
        assert high[3:] == pytest.approx([2.*value for value in low[3:]])


def test_solver_limit_singularity_orientation_residual_codes():
    model=FakeModel(); s,now=session(model)
    for expected in ("JOINT_LIMIT_REJECTED","IK_NEAR_SINGULAR","ORIENTATION_WINDOW_MISS","IK_RESIDUAL_EXCEEDED"):
        def fail(seed,target,axis,e=expected): raise DynamicPrepareBlocked(e)
        with pytest.raises(DynamicPrepareBlocked) as exc: prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1,solver=fail)
        assert code(exc)==expected


def test_wall_self_base_bracket_independent_rejects():
    cases=[("wall_margin_m",.2,"WALL_HALFSPACE_REJECTED"),
           ("self_pairs",[["upper","upper"]],"SELF_PROXY_REJECTED")]
    for key,value,expected in cases:
        model=FakeModel(); model.policy[key]=value; s,now=session(model)
        with pytest.raises(DynamicPrepareBlocked) as exc: prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1)
        assert code(exc)==expected
    for proxy_key,center in (("base_proxy",[.18,-.04,0]),("camera_bracket_proxy",[.18,-.04,0])):
        model=FakeModel(); model.policy[proxy_key]["center_m"]=center; s,now=session(model)
        with pytest.raises(DynamicPrepareBlocked) as exc: prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1)
        assert code(exc)=="BASE_BRACKET_PROXY_REJECTED"


def test_plan_hash_ttl_replay_q0_drift_and_identity():
    model,s,ticket=prepared()
    assert verify_frozen_plan(ticket,model=model,session=s,current_q0=s.q0_rad,now_monotonic=6).plan_hash==ticket.plan_hash
    consumed=verify_frozen_plan(ticket,model=model,session=s,current_q0=s.q0_rad,now_monotonic=6,consume=True)
    with pytest.raises(DynamicPrepareBlocked) as exc: verify_frozen_plan(consumed,model=model,session=s,current_q0=s.q0_rad,now_monotonic=7)
    assert code(exc)=="PLAN_INVALIDATED"
    with pytest.raises(DynamicPrepareBlocked) as exc: verify_frozen_plan(ticket,model=model,session=s,current_q0=(.13,0,0,0,0,0),now_monotonic=6)
    assert code(exc)=="PLAN_INVALIDATED"
    with pytest.raises(DynamicPrepareBlocked) as exc: verify_frozen_plan(ticket,model=model,session=s,current_q0=s.q0_rad,now_monotonic=16)
    assert code(exc)=="PLAN_EXPIRED"
    plan=copy.deepcopy(ticket.plan)
    plan["start_state_admission"]["q0_hash"]="0"*64
    with pytest.raises(DynamicPrepareBlocked) as exc:
        verify_frozen_plan(PreparedPlan(plan,ticket.plan_hash,ticket.expires_monotonic),model=model,session=s,current_q0=s.q0_rad,now_monotonic=6)
    assert code(exc)=="PLAN_INVALIDATED"


def test_zero_resource_and_warm_prepare_zero_open_or_subprocess(monkeypatch):
    model=FakeModel(); s,now=session(model); opens=[]; calls=[]
    monkeypatch.setattr(builtins,"open",lambda *a,**k: opens.append(a) or (_ for _ in ()).throw(AssertionError("open")))
    monkeypatch.setattr(subprocess,"Popen",lambda *a,**k: calls.append(a) or (_ for _ in ()).throw(AssertionError("Popen")))
    ticket=prepare_dynamic_plan(model,s,now_ns=now,now_monotonic=1)
    assert opens==calls==[]
    assert ticket.plan["core_zero_resource_invariant"]["runtime_instrumentation"] is False


def test_boot_loader_once_and_warm_performance():
    model=FrozenModel.load_once(".","config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json")
    assert model.startup_load_ms>=0 and all(len(v)==64 for v in model.hashes.values())
    assert model.model_loaded is True and model.execution_admission_ready is True
    assert model.admission_failure_reason == ""
    assert model.hashes["tool_model_hash"]=="5babcb0d9b627fcfe789cb7bff3e1e04db987fe251e9c272b2585156b31d73f4"
    assert model.hashes["fn_artifact_hash"]=="580e21d842c831787bc138dcabe5453b34d571fd2b1e852c4be319aaa4762e44"
    assert model.hashes["calibration_hash"]=="25de0900b64c94dac4fb0dbf30ec5cf4360d34be52199d8eed7e45e7b8c169f4"
    assert model.hashes["source_registry_hash"]=="72de077e595b6ea079f83545e0f6715801c74fdc6f9b23498d62f2c4a22351e1"
    standby=json.loads(Path(model.source_bindings["standby_acceptance_path_absolute"]).read_text())
    assert model.hashes["standby_acceptance_hash"]==hashlib.sha256(Path(model.source_bindings["standby_acceptance_path_absolute"]).read_bytes()).hexdigest()
    assert model.policy["standard_standby_rad"]==standby["command_quantized_target_rad"]
    assert model.policy["standard_standby_tolerance_rad"]==standby["admission_tolerance_rad"]
    observed=standby["operator_evidence"]["observed_q_rad"]
    assert max(abs(a-b) for a,b in zip(observed,model.policy["standard_standby_rad"])) <= model.policy["standard_standby_tolerance_rad"]
    assert model.source_bindings["fn_validation_identity"]=="section_c_fingertip_fn_precontact_outbound_20260817_v1|BUTTON1_DYNAMIC_IK_PRECONTACT_NO_CONTACT_ONLY|accepted_for_no_contact_dynamic_prepare_geometry"
    assert model.source_bindings["calibration_validation_identity"]=="gemini335_roarm_20260803_r6__scipy_separable_absolute__b5_tag_gap_250ms_v3|gemini335_roarm_20260803_r6|scipy_separable_absolute|held_out_passed"
    assert model.source_bindings["fn_pointer_path_relative"]=="config/calibration/candidates/section_c_fingertip_c0_active_overlay_v1.json"
    assert model.source_bindings["fn_artifact_path_relative"]=="config/calibration/candidates/section_c_fingertip_c0_operator_click_best_20260806_v1.json"
    assert model.source_bindings["fn_acceptance_path_relative"]=="config/calibration/accepted/section_c_fingertip_fn_precontact_outbound_20260817_v1.json"
    assert model.source_bindings["collision_proxy_path_relative"]=="config/press/prototype/t102_dynamic_collision_proxy_20260817_v1.json"
    assert model.source_bindings["standby_acceptance_path_relative"]=="config/press/prototype/t102_dynamic_standard_standby_acceptance_20260817_v1.json"
    assert model.source_bindings["calibration_source_path_relative"]=="config/calibration/accepted/gemini335_roarm_20260803_r6_eye_on_base_relative_bundle.json"
    assert all(value.startswith("/") for key,value in model.source_bindings.items() if key.endswith("_path_absolute"))
    fake=FakeModel(); s,now=session(fake); samples=[]
    for i in range(120):
        start=time.perf_counter_ns(); prepare_dynamic_plan(fake,s,now_ns=now,now_monotonic=1+i*.001); samples.append((time.perf_counter_ns()-start)/1e6)
    samples.sort(); assert samples[math.ceil(.95*len(samples))-1] < 100.


@pytest.mark.parametrize(
    "mutation",
    [
        "fault_state", "armed", "dry_run", "fault_text", "publisher", "lease",
        "observed_q", "declared_error", "window_error", "source_role",
        "agx_runtime_path", "press_authorized", "retract_authorized",
        "return_authorized", "active_tolerance", "active_target", "q0_age",
        "voltage_low", "voltage_high", "q0_source",
    ],
)
def test_standard_standby_boot_evidence_tamper_is_rejected(mutation):
    policy=json.loads(Path("config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json").read_text())
    standby=json.loads(Path(policy["standby_acceptance_source"]).read_text())
    source_bytes={item["role"]:Path(item["path"]).read_bytes() for item in standby["source_bindings"]}
    evidence=standby["operator_evidence"]
    approval=standby["approval"]
    if mutation=="fault_state": evidence["observed_terminal_state"]="FAULT"
    elif mutation=="armed": evidence["observed_armed"]=True
    elif mutation=="dry_run": evidence["observed_dry_run"]=True
    elif mutation=="fault_text": evidence["observed_fault"]="fault"
    elif mutation=="publisher": evidence["observed_joint_target_publisher_count"]=1
    elif mutation=="lease": evidence["observed_precontact_lease_active"]=True
    elif mutation=="observed_q": evidence["observed_q_rad"]=[0.]*6
    elif mutation=="declared_error": evidence["observed_max_command_error_rad"]=99.
    elif mutation=="window_error": evidence["terminal_window_max_command_error_rad"]=99.
    elif mutation=="source_role": standby["source_bindings"][0]["role"]="wrong"
    elif mutation=="agx_runtime_path": standby["source_bindings"][1]["agx_runtime_path"]="/tmp/wrong.yaml"
    elif mutation=="press_authorized": approval["press_authorized"]=True
    elif mutation=="retract_authorized": approval["retract_authorized"]=True
    elif mutation=="return_authorized": approval["return_authorized"]=True
    elif mutation=="active_tolerance": source_bytes["active_agx_safe_driver_config"]=source_bytes["active_agx_safe_driver_config"].replace(b"tolerance_deg: 1.0",b"tolerance_deg: 2.0")
    elif mutation=="active_target": source_bytes["active_agx_safe_driver_config"]=source_bytes["active_agx_safe_driver_config"].replace(b"startup_restore_target_rad: [0.046,",b"startup_restore_target_rad: [0.146,")
    elif mutation=="q0_age": policy["maximum_q0_age_s"]=2.
    elif mutation=="voltage_low": evidence["observed_voltage_v"]=10.
    elif mutation=="voltage_high": evidence["observed_voltage_v"]=100.
    elif mutation=="q0_source": standby["q0_contract"]["source"]="not measured feedback"
    with pytest.raises(DynamicPrepareBlocked) as exc:
        _validate_standard_standby_acceptance(policy,standby,source_bytes)
    assert code(exc)=="MODEL_LOAD_FAILED"


def test_representative_frozen_model_corpus_is_real_and_precisely_fail_closed():
    model=FrozenModel.load_once(".","config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json")
    assert model.execution_admission_ready is True
    # Test-only gate override exercises the real boot-frozen chain, solver,
    # encoder caps and production prototype proxies; it never changes policy.
    model.execution_admission_ready=True
    for distance in (.13,.18):
        frozen,now=representative_real_session(model,distance)
        ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
        assert ticket.plan["sampled_constraints"]["code"] == SUCCESS_CODE
        assert 3 <= len(ticket.plan["waypoints_rad"]) <= 6
    frozen,now=representative_real_session(model,.25)
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    assert code(exc)=="WAYPOINT_CAP_REJECTED"
    frozen,now=representative_real_session(model,.13); timings=[]
    for index in range(40):
        started=time.perf_counter_ns()
        ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.+index*.001)
        assert ticket.plan["sampled_constraints"]["code"] == SUCCESS_CODE
        timings.append((time.perf_counter_ns()-started)/1e6)
    timings.sort(); p95=timings[math.ceil(.95*len(timings))-1]
    print(f"representative_real_frozen_model_p95_ms={p95:.6f}; result={SUCCESS_CODE}")
    assert p95 < 100.


def test_exact_agx_button1_geometry_produces_bounded_audited_plan():
    model=FrozenModel.load_once(".","config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json")
    frozen,now=exact_agx_button1_session(model)
    started=time.perf_counter_ns()
    ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    elapsed_ms=(time.perf_counter_ns()-started)/1e6

    assert model.policy["ik_axis_weight"] == pytest.approx(.05)
    assert 3 <= len(ticket.plan["waypoints_rad"]) <= 6
    assert ticket.plan["sampled_constraints"]["code"] == SUCCESS_CODE
    assert all(value > 0 for value in ticket.plan["sampled_constraints"]["counts"].values())
    assert ticket.plan["p_button_base_m"] == pytest.approx(frozen.p_button_base_m)
    n_out=tuple(ticket.plan["normal_proof"]["n_out"])
    assert ticket.plan["p_stage_base_m"] == pytest.approx(tuple(frozen.p_button_base_m[i]+model.policy["d_stage_m"]*n_out[i] for i in range(3)))
    assert ticket.plan["p_pre_base_m"] == pytest.approx(tuple(frozen.p_button_base_m[i]+model.policy["d_pre_m"]*n_out[i] for i in range(3)))

    for audit in ticket.plan["ik_audit"]:
        assert audit["position_residual_m"] <= model.policy["position_residual_limit_m"]
        assert audit["axis_residual_deg"] <= model.policy["axis_residual_limit_deg"]
        assert audit["condition"] <= model.policy["jacobian_condition_limit"]
        assert audit["manipulability"] >= model.policy["manipulability_floor"]
    for audit in ticket.plan["post_quantization"]["waypoint_audit"]:
        assert audit["position_residual_m"] <= model.policy["position_residual_limit_m"]
        assert audit["axis_residual_deg"] <= model.policy["axis_residual_limit_deg"]
        assert audit["condition"] <= model.policy["jacobian_condition_limit"]
        assert audit["manipulability"] >= model.policy["manipulability_floor"]

    previous=frozen.q0_rad
    for waypoint in ticket.plan["waypoints_rad"]:
        assert all(abs(waypoint[i]-previous[i]) <= model.policy["per_command_caps_rad"][i]+1e-12 for i in range(6))
        assert waypoint[5] == pytest.approx(frozen.q0_rad[5],abs=model.policy["encoder_step_rad"])
        previous=waypoint

    # The exact capture contains one bounded direction change during the
    # documented initial current->stage settling region.  It remains subject
    # to jump/limit/residual/Jacobian gates; later oscillation is prohibited.
    audits=ticket.plan["ik_audit"]
    deltas=[]; previous=frozen.q0_rad
    for audit in audits:
        current=tuple(audit["q"])
        deltas.append(tuple(current[i]-previous[i] for i in range(6)))
        previous=current
    reversal_limit=model.policy["maximum_joint_reversal_rad"]
    settlement_end=max(2,model.policy["cartesian_sample_count"]//2)
    assert any(
        deltas[index-1][joint]*deltas[index][joint] < 0
        and min(abs(deltas[index-1][joint]),abs(deltas[index][joint])) > reversal_limit
        for index in range(1,settlement_end) for joint in range(5)
    )
    print(
        "exact_agx_button1_prepare_ms={:.6f}; final_residual_mm={:.6f}; "
        "final_axis_deg={:.6f}; waypoints={}".format(
            elapsed_ms,
            1000.*ticket.plan["post_quantization"]["position_residual_m"],
            ticket.plan["post_quantization"]["axis_residual_deg"],
            len(ticket.plan["waypoints_rad"]),
        )
    )


def test_latest_agx_target_uses_valid_six_waypoint_subsequence():
    model=FrozenModel.load_once(".","config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json")
    frozen,now=latest_agx_button1_session(model)
    ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    indices=[item["cartesian_sample_index"] for item in ticket.plan["post_quantization"]["waypoint_audit"]]
    assert len(ticket.plan["waypoints_rad"]) == 6
    assert indices == [1,2,3,4,5,11]
    assert ticket.plan["sampled_constraints"] == {
        "code": SUCCESS_CODE,
        "counts": {"wall":216,"self":108,"base":108,"bracket":162},
    }


def test_resumed_agx_target_repartitions_to_six_bounded_waypoints():
    model=FrozenModel.load_once(".","config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json")
    frozen,now=resumed_agx_button1_session(model)
    ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    indices=[item["cartesian_sample_index"] for item in ticket.plan["post_quantization"]["waypoint_audit"]]
    assert len(ticket.plan["waypoints_rad"]) == 6
    assert indices == [0,2,3,4,5,11]
    assert indices[-2:] == [model.policy["cartesian_sample_count"]-1,2*model.policy["cartesian_sample_count"]-1]
    audits=ticket.plan["ik_audit"]
    deltas=[]; previous=frozen.q0_rad
    for audit in audits:
        current=tuple(audit["q"])
        deltas.append(tuple(current[i]-previous[i] for i in range(6)))
        previous=current
    reversal_limit=model.policy["maximum_joint_reversal_rad"]
    settlement_end=max(2,model.policy["cartesian_sample_count"]//2)
    assert deltas[settlement_end-1][3]*deltas[settlement_end][3] < 0
    assert min(abs(deltas[settlement_end-1][3]),abs(deltas[settlement_end][3])) > reversal_limit
    assert not any(
        deltas[index-1][joint]*deltas[index][joint] < 0
        and min(abs(deltas[index-1][joint]),abs(deltas[index][joint])) > reversal_limit
        for index in range(settlement_end+1,len(deltas)) for joint in range(5)
        if index != model.policy["cartesian_sample_count"]
    )


def test_physical_attempt4_prefers_stronger_equal_length_first_waypoint():
    model=FrozenModel.load_once(".","config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json")
    frozen,now=physical_attempt4_agx_button1_session(model)
    ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    indices=[item["cartesian_sample_index"] for item in ticket.plan["post_quantization"]["waypoint_audit"]]
    assert indices == [1,2,3,4,5,11]
    first=ticket.plan["waypoints_rad"][0]
    active=[abs(first[index]-frozen.q0_rad[index]) for index in range(6)
            if abs(first[index]-frozen.q0_rad[index]) >= .5*model.policy["encoder_step_rad"]]
    assert min(active) > math.radians(2.0)


def test_physical_attempt6_effective_release_is_exactly_four_bounded_waypoints():
    model=FrozenModel.load_once(".","config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json")
    model.policy["effective_waypoint_count"]=4
    frozen,now=physical_attempt6_agx_button1_session(model)
    ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    assert len(ticket.plan["waypoints_rad"]) == 4
    assert ticket.plan["effective_waypoint_partition"] == {
        "enabled": True,
        "mode": "JOINT_LINEAR_EFFECTIVE_TO_EXACT_DYNAMIC_ENDPOINT",
        "count": 4,
        "exact_endpoint_retained": True,
        "stage_anchor_retained_in_ik_audit": True,
        "stage_commanded_exactly": False,
    }
    audit=ticket.plan["post_quantization"]["waypoint_audit"]
    assert [item["effective_partition_ratio"] for item in audit] == pytest.approx([.25,.5,.75,1.])
    assert [item["cartesian_sample_index"] for item in audit] == [None,None,None,11]
    assert all(item["audit_reference"] == "effective_joint_partition_unquantized_fk" for item in audit)
    assert ticket.plan["sampled_constraints"] == {
        "code": SUCCESS_CODE,
        "counts": {"wall":144,"self":72,"base":72,"bracket":108},
    }
    previous=frozen.q0_rad
    for waypoint in ticket.plan["waypoints_rad"]:
        assert all(
            abs(waypoint[index]-previous[index]) <= model.policy["per_command_caps_rad"][index]+1e-12
            for index in range(6)
        )
        previous=waypoint
    assert ticket.plan["waypoints_rad"][-1] == pytest.approx((
        0.1825437137583913,
        -0.6902913545485385,
        1.8377089838869982,
        0.3758252930319821,
        -0.29299033048615747,
        0.0015339807878856412,
    ))


def test_physical_attempt6_wrist_locked_release_keeps_camera_roll_and_four_bounded_steps():
    model=FrozenModel.load_once(".","temp/t102_dynamic_release_20260818/policy_release.json")
    frozen,now=physical_attempt6_agx_button1_session(model)
    ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    waypoints=ticket.plan["waypoints_rad"]
    assert len(waypoints) == 4
    assert ticket.plan["adaptive_plan_selection"]["selected_stage_m"] == pytest.approx(0.10)
    assert ticket.plan["adaptive_plan_selection"]["selected_effective_waypoint_count"] == 4
    lock=ticket.plan["wrist_roll_lock"]
    assert lock["enabled"] is True
    assert lock["joint_index"] == 4
    assert all(waypoint[4] == lock["commanded_value_rad"] for waypoint in waypoints)
    assert abs(lock["commanded_value_rad"]-frozen.q0_rad[4]) <= .5*model.policy["encoder_step_rad"]+1e-12
    previous=frozen.q0_rad
    for waypoint in waypoints:
        assert all(
            abs(waypoint[index]-previous[index]) <= model.policy["per_command_caps_rad"][index]+1e-12
            for index in range(6)
        )
        previous=waypoint
    assert max(item["axis_residual_deg"] for item in ticket.plan["post_quantization"]["waypoint_audit"]) <= 15.0


def test_current_approximate_target_uses_reviewed_no_contact_orientation_policy():
    old=FrozenModel.load_once(".","temp/t102_dynamic_release_20260818/policy_release.json")
    old.policy["ik_axis_weight"]=0.05
    old.policy["axis_residual_limit_deg"]=15.0
    old.policy.pop("adaptive_stage_candidates_m")
    old.policy.pop("adaptive_effective_waypoint_count_enabled")
    frozen,now=current_approximate_agx_button1_session(old)
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(old,frozen,now_ns=now,now_monotonic=1.)
    assert code(exc)=="IK_RESIDUAL_EXCEEDED"

    model=FrozenModel.load_once(".","temp/t102_dynamic_release_20260818/policy_release.json")
    frozen,now=current_approximate_agx_button1_session(model)
    ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    assert model.policy["ik_axis_weight"] == pytest.approx(0.04)
    assert model.policy["axis_residual_limit_deg"] == pytest.approx(22.5)
    assert model.policy["position_residual_limit_m"] == pytest.approx(0.0025)
    assert model.policy["wrist_roll_lock_enabled"] is True
    assert model.policy["d_pre_m"] == pytest.approx(0.07)
    assert len(ticket.plan["waypoints_rad"]) == 4
    assert max(item["position_residual_m"] for item in ticket.plan["ik_audit"]) <= 0.0025
    assert max(item["axis_residual_deg"] for item in ticket.plan["ik_audit"]) <= 22.5
    lock=ticket.plan["wrist_roll_lock"]
    assert lock["enabled"] is True
    assert all(waypoint[4] == lock["commanded_value_rad"] for waypoint in ticket.plan["waypoints_rad"])
    assert ticket.plan["sampled_constraints"] == {
        "code": SUCCESS_CODE,
        "counts": {"wall":144,"self":72,"base":72,"bracket":108},
    }
    previous=frozen.q0_rad
    for waypoint in ticket.plan["waypoints_rad"]:
        assert all(
            abs(waypoint[index]-previous[index]) <= model.policy["per_command_caps_rad"][index]+1e-12
            for index in range(6)
        )
        previous=waypoint


def test_current_lateral_target_selects_farthest_valid_stage_and_five_bounded_waypoints():
    model=FrozenModel.load_once(".","config/press/releases/t102_dynamic_precontact_policy_release_20260818.json")
    frozen,now=current_20260820_lateral_button1_session(model)
    ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    selection=ticket.plan["adaptive_plan_selection"]
    assert selection == {
        "enabled": True,
        "selection_order": "FARTHEST_STAGE_THEN_MINIMUM_WAYPOINT_COUNT",
        "declared_stage_candidates_m": [0.10,0.095,0.09,0.085,0.08,0.075],
        "declared_waypoint_count_range": [3,6],
        "selected_stage_candidate_index": 5,
        "selected_stage_m": 0.075,
        "selected_effective_waypoint_count": 5,
        "partial_candidate_reuse": True,
        "runtime_retry": False,
    }
    assert ticket.plan["d_pre_m"] == pytest.approx(0.07)
    assert len(ticket.plan["waypoints_rad"]) == 5
    assert max(item["position_residual_m"] for item in ticket.plan["ik_audit"]) <= 0.0025
    assert max(item["axis_residual_deg"] for item in ticket.plan["ik_audit"]) <= 22.5
    assert ticket.plan["post_quantization"]["position_residual_m"] <= 0.0025
    assert ticket.plan["post_quantization"]["axis_residual_deg"] <= 22.5
    assert ticket.plan["sampled_constraints"] == {
        "code": SUCCESS_CODE,
        "counts": {"wall":180,"self":90,"base":90,"bracket":135},
    }
    repeated=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    assert repeated.plan_hash == ticket.plan_hash
    assert repeated.plan == ticket.plan
    tampered=dict(ticket.plan)
    tampered_selection=dict(tampered["adaptive_plan_selection"])
    tampered_selection["selected_effective_waypoint_count"]=4
    tampered["adaptive_plan_selection"]=tampered_selection
    with pytest.raises(DynamicPrepareBlocked) as exc:
        verify_frozen_plan(
            replace(ticket,plan=tampered),model=model,session=frozen,
            current_q0=frozen.q0_rad,now_monotonic=1.,
        )
    assert code(exc) == "PLAN_INVALIDATED"


def test_nearby_farther_lateral_target_fails_closed_when_all_candidates_miss_orientation():
    model=FrozenModel.load_once(".","config/press/releases/t102_dynamic_precontact_policy_release_20260818.json")
    frozen,now=current_20260820_lateral_button1_session(model)
    value=asdict(frozen)
    p_button=(0.34447401164232333,0.15220668675588692,0.2852730504309356)
    value["p_button_base_m"]=p_button
    value["wall_surface_point_base_m"]=p_button
    value.pop("session_id",None)
    value["session_id"]=canonical_hash(value)
    farther=FrozenSession.from_mapping(value)
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,farther,now_ns=now,now_monotonic=1.)
    assert code(exc) == "WAYPOINT_CAP_REJECTED"
    assert "0.075000m/6:ORIENTATION_WINDOW_MISS" in str(exc.value)


@pytest.mark.parametrize("delta,expected", [(0.14,3),(0.19,4),(0.24,5),(0.29,6)])
def test_adaptive_partition_chooses_minimum_valid_count(delta,expected):
    model=FakeModel()
    model.policy.update({
        "adaptive_effective_waypoint_count_enabled":True,
        "adaptive_stage_candidates_m":[0.10,0.075],
        "effective_waypoint_count":4,
        "capsules":[],"self_pairs":[],
        "base_check_capsules":[],"bracket_check_capsules":[],
    })
    p_button=0.05+delta
    frozen,now=session(
        model,p_button_base_m=(p_button,0.,0.),
        wall_surface_point_base_m=(p_button,0.,0.),
    )
    ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    assert ticket.plan["adaptive_plan_selection"]["selected_stage_m"] == pytest.approx(0.10)
    assert ticket.plan["adaptive_plan_selection"]["selected_effective_waypoint_count"] == expected
    assert len(ticket.plan["waypoints_rad"]) == expected


def test_adaptive_waypoint_candidates_reuse_one_completed_stage_ik_audit():
    model=FakeModel()
    model.policy.update({
        "adaptive_effective_waypoint_count_enabled":True,
        "adaptive_stage_candidates_m":[0.10,0.075],
        "effective_waypoint_count":4,
        "capsules":[],"self_pairs":[],
        "base_check_capsules":[],"bracket_check_capsules":[],
    })
    frozen,now=session(
        model,p_button_base_m=(0.05+0.24,0.,0.),
        wall_surface_point_base_m=(0.05+0.24,0.,0.),
    )
    production_calls=0
    original_solver=model.solve

    def counted_production_solver(seed,target,desired):
        nonlocal production_calls
        production_calls+=1
        return original_solver(seed,target,desired)

    model.solve=counted_production_solver
    ticket=prepare_dynamic_plan(
        model,frozen,now_ns=now,now_monotonic=1.,
    )
    assert ticket.plan["adaptive_plan_selection"]["selected_effective_waypoint_count"] == 5
    assert ticket.plan["adaptive_plan_selection"]["partial_candidate_reuse"] is True
    assert production_calls == 2*model.policy["cartesian_sample_count"]

    injected_calls=0

    def counted_injected_solver(seed,target,desired):
        nonlocal injected_calls
        injected_calls+=1
        return original_solver(seed,target,desired)

    injected=prepare_dynamic_plan(
        model,frozen,now_ns=now,now_monotonic=1.,solver=counted_injected_solver,
    )
    assert injected.plan["waypoints_rad"] == ticket.plan["waypoints_rad"]
    assert injected.plan["adaptive_plan_selection"]["partial_candidate_reuse"] is False
    assert injected_calls == 3*2*model.policy["cartesian_sample_count"]


def test_adaptive_partition_beyond_six_fails_closed_with_candidate_diagnostics():
    model=FakeModel()
    model.policy.update({
        "adaptive_effective_waypoint_count_enabled":True,
        "adaptive_stage_candidates_m":[0.10,0.075],
        "effective_waypoint_count":4,
        "capsules":[],"self_pairs":[],
        "base_check_capsules":[],"bracket_check_capsules":[],
    })
    p_button=0.05+0.31
    frozen,now=session(
        model,p_button_base_m=(p_button,0.,0.),
        wall_surface_point_base_m=(p_button,0.,0.),
    )
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    assert code(exc) == "WAYPOINT_CAP_REJECTED"
    assert "0.100000m/6:WAYPOINT_CAP_REJECTED" in str(exc.value)
    assert "0.075000m/6:WAYPOINT_CAP_REJECTED" in str(exc.value)


def test_adaptive_selection_does_not_mask_session_freshness_failure():
    model=FakeModel()
    model.policy.update({
        "adaptive_effective_waypoint_count_enabled":True,
        "adaptive_stage_candidates_m":[0.10,0.075],
    })
    frozen,now=session(model,confidence=0.2)
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    assert code(exc) == "TARGET_STALE"


def test_effective_three_waypoint_partition_fails_closed_when_caps_need_more():
    model=FrozenModel.load_once(".","config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json")
    model.policy["effective_waypoint_count"]=3
    frozen,now=resumed_agx_button1_session(model)
    with pytest.raises(DynamicPrepareBlocked) as exc:
        prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    assert code(exc) == "WAYPOINT_CAP_REJECTED"


def test_real_frozen_model_warm_prepare_has_zero_file_or_subprocess_io(monkeypatch):
    model=FrozenModel.load_once(".","config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json")
    model.execution_admission_ready=True
    frozen,now=representative_real_session(model,.13)
    def blocked(*_args,**_kwargs):
        raise AssertionError("warm Prepare attempted file or subprocess I/O")
    monkeypatch.setattr(builtins,"open",blocked)
    monkeypatch.setattr(Path,"open",blocked)
    monkeypatch.setattr(Path,"read_bytes",blocked)
    monkeypatch.setattr(Path,"read_text",blocked)
    monkeypatch.setattr(subprocess,"Popen",blocked)
    monkeypatch.setattr(subprocess,"run",blocked)
    ticket=prepare_dynamic_plan(model,frozen,now_ns=now,now_monotonic=1.)
    assert ticket.plan["sampled_constraints"]["code"] == SUCCESS_CODE


def test_fake_core_pipeline_cart_freeze_prepare_await_move_confirm():
    model,s,ticket=prepared()
    states=["IDLE","CART_PARKED","TARGET_FROZEN","PREPARE_IN_FLIGHT","AWAIT_MOVE_CONFIRM"]
    assert states[-1]=="AWAIT_MOVE_CONFIRM" and ticket.plan_hash and ticket.plan["session"]["cart_pose_epoch"]=="cart"
