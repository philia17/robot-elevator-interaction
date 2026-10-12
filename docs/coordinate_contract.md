# Lift Robot Coordinate Contract v1.0.0

**Status:** frozen on 2026-07-16
**Machine-readable authority:** [`config/coordinate_contract_v1.yaml`](../config/coordinate_contract_v1.yaml)
**Validation:** `python3 scripts/coordinate_contract/validate_coordinate_contract.py`

This contract is the single coordinate boundary shared by Orbbec Gemini 335 perception,
TF2, the RoArm-M3 press bridge, RoboDK adapters, and future iOS status APIs.
Changing frame names, axes, units, transform direction, or target fields requires
a new contract version and new calibration ID. Unknown calibration values are
represented by `null` plus `status: pending`; zero is never a placeholder.

## 1. Frozen frame tree

```text
map                           (REP-103 body: X forward, Y left, Z up)
└── odom                      (dynamic; localization/odometry)
    └── cart_base             (dynamic globally; rigid local operation root)
        ├── camera_link       (static measured mount transform)
        │   └── camera_optical_frame
        │       └── panel_frame          (runtime observation)
        │           └── button_target    (runtime observation/debug TF)
        └── roarm_base_link   (static measured mount transform)
            └── base_link     (identity adapter to vendor URDF name)
                └── ...       (vendor RoArm-M3 kinematic chain)
                    └── hand_tcp
                        └── press_contact (static measured stock contact TCP)
```

`camera_link`, `cart_base`, `roarm_base_link`, `panel_frame`, `button_target`,
and `press_contact` use the ordinary REP-103 body convention: X forward, Y
left, Z up. `camera_optical_frame` is the only optical frame: X right, Y down,
Z forward. `panel_frame` defines +X as the panel outward direction. When the
tool is aligned to press, `press_contact` +X points into the panel and therefore
aligns with the negative panel outward normal.

The RoArm-M3 vendor URDF names its root `base_link` and its TCP `hand_tcp`.
Project code must not silently rename those vendor facts. It publishes a
project frame `roarm_base_link`, co-located with vendor `base_link` through an
identity adapter, and removes/does not publish the vendor demo-only
`world -> base_link` root joint. This is traceable to:

- `RoArm-M3/ros2/roarm_ws/src/roarm_main/roarm_description/urdf/roarm_m3/roarm_m3.xacro`
- `RoArm-M3/ros2/roarm_ws/src/roarm_main/roarm_moveit_cmd/src/getposecmd_tf.cpp`

Do not add another parent to `base_link`. If a prefixed vendor robot
description is introduced later, that is a contract-versioned migration.

## 2. Transform notation and ROS direction

`T_parent_child` (also written `^parent T_child`) maps coordinates expressed in
the child into the parent:

```text
p_parent = T_parent_child * p_child
```

For `geometry_msgs/TransformStamped`:

```text
header.frame_id = parent
child_frame_id  = child
transform       = T_parent_child
```

For `geometry_msgs/PoseStamped`, `header.frame_id` is the frame in which all
pose components are expressed. It is not the target's child-frame name. A
button pose in camera optical coordinates therefore has
`header.frame_id=camera_optical_frame`, never `camera_link`.

The conventional body-to-optical edge is fixed and tested:

```text
T_camera_link_camera_optical_frame
translation_m    = [0, 0, 0]
rotation_xyzw    = [-0.5, 0.5, -0.5, 0.5]
```

Its rotation maps optical basis vectors into the body frame as follows:

```text
optical +X (right)   -> body -Y
optical +Y (down)    -> body -Z
optical +Z (forward) -> body +X
```

All quaternions are ordered XYZW, must be finite and unit length, and must form
a proper rotation (`det(R)=+1`). Euler angles are not part of the cross-node
interface.

## 3. Transform sources and timestamps

| Edge | Source | Mode | Press dependency |
|---|---|---|---|
| `map -> odom` | SLAM/localization | dynamic | no |
| `odom -> cart_base` | odometry/localization | dynamic | no |
| `cart_base -> camera_link` | measured camera mount | `/tf_static` | required |
| `camera_link -> camera_optical_frame` | REP-103 convention | `/tf_static` | required |
| `cart_base -> roarm_base_link` | measured arm mount | `/tf_static` | required |
| `roarm_base_link -> base_link` | project/vendor name adapter | `/tf_static` | required |
| `base_link -> ... -> hand_tcp` | vendor URDF + measured joints | `/tf` | required |
| `hand_tcp -> press_contact` | measured stock-contact TCP | `/tf_static` | required |
| `camera_optical_frame -> panel_frame` | synchronized fiducial/depth estimate | runtime `/tf` or stamped observation | required |
| `panel_frame -> button_target` | target resolver | runtime/debug `/tf` | required |

Static calibration is published on `/tf_static` with transient-local durability.
Runtime panel and button observations carry the synchronized RGB-D observation
timestamp. The target `PoseStamped.header.stamp`, normal, and source timestamp
refer to the same observation. The press bridge performs TF lookup at that
timestamp; it must not substitute `latest` after the cart or panel moves.

The cart must be stationary during local operation. Any cart movement,
calibration reload, camera restart that changes calibration, or expired target
invalidates `panel_frame` and `button_target`.

## 4. SLAM boundary

SLAM is only for navigating the cart to the panel. Gemini 335 and RoArm-M3 are rigid
on `cart_base`. The canonical parent-child branches are:

```text
cart_base -> camera_link -> camera_optical_frame
cart_base -> roarm_base_link
```

The local coordinate conversion uses inverse TF edges as required:

```text
T_roarm_base_link_camera_optical_frame
  = inverse(T_cart_base_roarm_base_link)
  * T_cart_base_camera_link
  * T_camera_link_camera_optical_frame
```

It does not go through `map` or `odom`. Map correction and SLAM drift must not
change the camera-to-arm transform or a local press target. The bridge rejects
any design that computes eye-to-hand from two drifting global poses at runtime.

## 5. Historical HP60C normalization boundary

The live AGX check on 2026-07-16 found:

- all RGB, RGB CameraInfo, registered depth, depth CameraInfo, and PointCloud2
  messages use raw `frame_id=ascamera_hp60c_color_0`;
- the driver conditionally advertises `/tf` only while a stream has a
  subscriber;
- it publishes two conflicting transforms with the same
  `ascamera_hp60c_camera_link_0 -> ascamera_hp60c_color_0` parent/child pair:
  one near-identity color extrinsic and one exact identity;
- it publishes no `/tf_static` edge;
- the near-identity rotation is not the REP-103 body-to-optical rotation.

Therefore vendor HP60C TF is explicitly forbidden in the execution-capable TF
tree. The integrated launch must set `pub_tfTree=false`. A project camera
adapter must:

1. validate that the raw arrays use optical axes;
2. validate RGB-depth registration independently of shared `frame_id`;
3. convert raw `16UC1` depth using a measured `depth_scale_m_per_unit`;
4. publish normalized data with `frame_id=camera_optical_frame`;
5. publish the canonical project TF edges only.

The raw optical-axis validation and depth scale remain `pending`. Until they
are measured, RGB-D geometry may run only as non-executing diagnostics.

### 5.1 Gemini 335 active normalization boundary

The Gemini 335 Phase 1 audit observed RGB, RGB CameraInfo, registered depth,
and registered-depth CameraInfo in `gemini335_color_optical_frame`. The vendor
also publishes its own body/optical TF branch. Those facts are retained as
vendor evidence, not adopted as the canonical execution tree.

The active Gemini adapter validates the audited raw optical frame, preserves
message data and timestamps, and republishes the four audited streams with
`header.frame_id=camera_optical_frame`. Its integrated launch disables Orbbec
`publish_tf` and publishes only the frozen conventional edge
`camera_link -> camera_optical_frame`. Because the camera mount has not been
measured, `cart_base -> camera_link` remains null/pending and is not published.

The Gemini physical optical-axis test and depth scale remain `pending`. The
mathematically correct body-to-optical quaternion does not substitute for the
physical X-right/Y-down/Z-forward test, and raw depth device units are not
metres until Phase 12 measures their scale.

## 6. Units

Cross-node and persisted calibrated geometry uses:

| Quantity | Unit/representation |
|---|---|
| translation, distance, TCP offset | metre (`m`) |
| joint/Cartesian angle | radian (`rad`) |
| orientation | quaternion XYZW |
| image coordinates | pixel (`px`) |
| timestamps | integer nanoseconds (`ns`) |
| confidence | `[0, 1]` |

Raw depth units are not geometry. They become metres exactly once at the camera
adapter using an explicit, calibrated scale. Existing files with explicit
legacy suffixes such as `_mm` or `_deg` are local configuration formats, not
the target interface, and must convert once at their adapter boundary.
RoboDK alone uses millimetres internally: multiply ROS metres by 1000 on entry
and divide by 1000 on exit, in one named RoboDK adapter.

## 7. Button target interface

The v1 target is published on `/elevator/button_target`. It is logically a
`PoseStamped` plus the outward panel normal and provenance. A custom ROS message
may encode it later, but it must preserve these exact fields and meanings:

```yaml
schema_version: 1
coordinate_contract_version: "1.0.0"
target_id: "unique-target-id"
detection_id: "source-detection-id"
pose:
  header:
    frame_id: camera_optical_frame   # or panel_frame
    stamp_ns: 1784171850588308649
  position_m: [0.10, -0.03, 0.60]
  orientation_xyzw: [0.0, 0.7071067812, 0.0, 0.7071067812]
panel_outward_normal:
  frame_id: camera_optical_frame
  xyz: [0.0, 0.0, -1.0]
source_timestamp_ns: 1784171850588308649
confidence: 0.95
calibration_id: "measured-bundle-id"
ttl_ms: 1000
```

The numbers above illustrate field shape only and are not project calibration.
Allowed pose frames are `camera_optical_frame` and `panel_frame`. The normal is
a finite unit vector expressed in the same frame as the pose and points out of
the panel toward the camera/cart. The pose orientation's local +X axis must
align with that normal; roll about +X comes from the panel pose. `pose.header.stamp_ns` must equal
`source_timestamp_ns`. The current thresholds are confidence at least 0.8,
target TTL at most 1000 ms, and synchronized RGB-depth skew at most 50 ms.

The surface trajectory convention is:

```text
p_pre     = p_surface + d_pre * n_outward
p_contact = p_surface
p_press   = p_surface - d_press * n_outward
p_retract = p_pre
```

`press_contact` represents the stationary contact patch on the original factory
end effector; no fingertip is installed. `hand_tcp -> press_contact` translation
and rotation must be measured from that exact patch. Until then it stays null
and press execution stays disabled.

## 8. Calibration bundle and fail-closed execution

One `calibration_id` identifies the complete compatible bundle:

- Gemini 335 intrinsics at the active 1920×1080 resolution;
- RGB-depth registration and depth scale;
- raw-to-canonical optical-axis validation;
- `cart_base -> camera_link` mount calibration;
- `cart_base -> roarm_base_link` mount calibration / eye-to-hand consistency;
- `hand_tcp -> press_contact` stock contact TCP;
- panel fixture/model data used for the current observation.

Execution readiness is false unless the contract's `calibration.execution_enabled`
is true, `calibration_id` is non-empty, all required transforms are calibrated
with that exact ID, and every required component is `calibrated`. A target with
an empty, unknown, stale, or mismatched ID is rejected. Pending transforms keep
their numeric fields null; filling them with identity is a validation error.

## 9. Required rejection rules

The press bridge or any future geometry-to-motion adapter must reject:

- `camera_link` carrying optical coordinates;
- unknown frame names or unexpected TF parent direction;
- missing, zero, future, expired, or inconsistent timestamps;
- non-finite vectors, non-unit normals, malformed quaternions, or mixed units;
- a shared raw RGB/depth frame ID used as proof of registration;
- pending or mismatched calibration;
- targets retained after cart motion;
- local press geometry routed through `map`/`odom`;
- vendor HP60C or Gemini vendor TF in the canonical execution tree.

Run the validator after any related change:

```bash
python3 scripts/coordinate_contract/validate_coordinate_contract.py
python3 -m pytest -q scripts/coordinate_contract/test
```

The first command should report a valid contract and `execution_ready=false`
until real calibrations are deliberately installed. Adding
`--require-execution-ready` must fail during the pending state.
