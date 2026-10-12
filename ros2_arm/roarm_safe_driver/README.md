# RoArm safe driver

This project-owned package is the Phase 2C hardware boundary. It does not use
the vendor driver's unsafe `/joint_states` command subscription.

For the earlier Web UI-only calibration path, `webui_monitor.launch.py` provides
a read-only `feedback_monitor`. Opening or switching the serial owner can still
reset the controller, however, so the 2026-07-24 supervised manual hand-teach
workflow does **not** switch to that monitor. It keeps the already-running safe
driver as the sole feedback owner throughout Torque Off → operator pose →
Torque On → settle → capture. See
the private deployment calibration plan (not distributed; see [public release scope](../../docs/PUBLIC_RELEASE.md)).

- Measured firmware `T:1051` state is published on `/joint_states`.
- Commands are accepted only on `/roarm_safe/joint_target`.
- `/roarm_safe/set_armed` (`std_srvs/SetBool`) explicitly arms/disarms.
- Arming is single-use and has a hard one-second command deadline. If no valid
  first T101 arrives by that deadline, the driver returns
  `ARM_COMMAND_TIMEOUT_DISARMED` without sending firmware motion. Repeating an
  arm request cannot extend that deadline.
- `/roarm_safe/clear_fault` clears a latched fault but leaves the node disarmed.
- `/roarm_safe/set_torque` (`std_srvs/SetBool`) sends firmware torque-lock
  `T:210` for supervised hand-teach (`data=false` → Torque Off / `cmd:0`,
  `data=true` → Torque On / `cmd:1`). It never arms motion, refuses while a
  command is active or any `/roarm_safe/joint_target` publisher exists, and
  disarms first if the host was armed. This is not an E-stop; physical 12 V
  cut remains the emergency stop.
- `/roarm_safe/status` publishes transient-local JSON status, timestamped
  command/feedback telemetry, rejection, drift, timeout, and convergence data.
- `/roarm_safe/feedback_status` publishes one compact JSON record for every
  accepted `T:1051` frame. It uses the exact same `feedback_stamp_ns` and six
  positions as the corresponding `/joint_states` message, and includes the six
  direct torque flags, voltage, armed/dry-run/fault state. Calibration recorder
  safe-driver mode pairs only exact matching stamps and positions; the older
  retained `/roarm_safe/status` is not sample-aligned evidence.
- Supervised hand-teach capture deliberately retains the normal torque fault:
  Torque Off latches a fault and disarms the host. After Torque On, all six
  measured torque flags must be true before a separately authorized fault clear.
  The host remains `armed=false`; no teach mode suppresses this fault.
- The calibration recorder requires healthy sample-aligned safe status for
  **every** joint frame in the complete settle window, not only its final frame,
  and refuses capture while any `/roarm_safe/joint_target` publisher exists.
- Defaults are dry-run, DISARMED, and transport `none`; no real port is opened.

The aligned feedback topic is available only after a driver process starts from
the updated code. Installing or building the package does not change an already
running process. On the current AGX hardware, reopening the CP2102N can reset the
controller and move the arm to firmware startup pose, so any restart requires a
fresh operator/physical safety gate.

For hardware, pass `transport:=serial`, `dry_run:=false`, and the tested stable
`/dev/serial/by-id/...` path. The node refuses a `/dev/ttyUSB*` hardware path.
Never run the vendor `roarm_driver` in the same test.

For a hardware-isolated runtime smoke test, use `transport:=fake`,
`fake_feedback:=true`, and keep `dry_run:=true`. This generates recorded-shape
feedback in memory and cannot open a device node.

The only supported physical command family is single-logical-joint `T:101`:

```json
{"T":101,"joint":2,"rad":0.017453292519943295,"spd":17,"acc":1}
```

The six-name command contract is retained, but exactly one logical joint must
differ from fresh measured feedback by more than one encoder step. The other
five must match it within one encoder step. Firmware joint IDs 1–6 map to base,
shoulder, elbow, wrist, roll, and gripper. Shoulder remains one logical
`joint:2` command; its two physical servos are coupled inside the firmware and
must never be split into host-side commands. Gripper uses the inverse of the
feedback convention (`firmware_rad = pi - ROS_rad`). No all-joint command is
generated.

In URDF order, speed ceilings/defaults are `[34,17,17,23,34,17]` encoder
steps/s and production maximum one-shot deltas are `[5,2,2,3,5,2]` degrees.
Values may be lowered in configuration. The dedicated inert
`diagnostic_base90.yaml` profile is the only exception: code permits base alone
to be configured up to 95 degrees for the recorded supervised Phase 2C sweep;
all other hard ceilings remain unchanged and normal defaults remain 5 degrees.
The separate inert `diagnostic_joint_characterization.yaml` keeps base at 5
degrees and permits only the recorded free-space characterization caps:
shoulder/elbow 10, wrist/gripper 15, and roll 30 degrees. Both diagnostic files
start with no transport, no serial path, and dry-run enabled; neither is a
production launch profile. Normal defaults remain `[5,2,2,3,5,2]` degrees.
The inert `phase5_base90_calibration.yaml` profile is separately scoped to the
operator-authorized eye-to-hand base sweep: base is exactly 90 degrees while
the other five limits remain those of the joint-characterization profile. It
also starts with no transport and dry-run enabled and is not a production
profile.
Every acceleration register is locked to 1, which means 100 encoder steps/s²;
zero selects maximum acceleration and is rejected. No automatic retry or
endpoint compensation is implemented.

## AGX unified-host profile

`config/agx_unified_safe.yaml` records the AGX Orin's stable CP2102N by-id path
and is an explicitly hardware-active profile: `transport=serial` and
`dry_run=false`. Loading it opens the real serial device and therefore requires
fresh operator and physical-runtime gates; cloning or launching this repository
never authorizes motion. The matching `agx_unified_safe.launch.py` starts only
this project safe driver; it does not include the vendor driver, MoveIt, RViz,
or a joint-state publisher. Use `config/safe_defaults.yaml` for the inert
`transport=none`, `dry_run=true` repository default.

If the firmware does not emit unsolicited T:1051 feedback, the node also has a
diagnostic-only `request_feedback_once_on_start` parameter. It defaults false.
When explicitly enabled it is accepted only for non-dry-run serial transport
and writes exactly one read-only `{"T":105}` line after open; it does not arm
the controller or enable any motion command.

Unified-compute ROS interfaces retain the existing camera/perception graph and
reserve `/roarm_safe/*` for arm control, `/joint_states` for measured arm state,
`/roarm_press/*` for press sequencing, and
`/elevator/button_target` for the frozen perception-to-arm coordinate contract.
Canonical TF names are `cart_base`, `camera_link`, `camera_optical_frame`,
`roarm_base_link`, vendor `base_link`/`hand_tcp`, `press_contact`, `panel_frame`,
and `button_target`. See
[`docs/coordinate_contract.md`](../../docs/coordinate_contract.md). This profile does not
publish or calibrate those transforms and cannot become execution-ready while
the versioned calibration bundle is pending.
