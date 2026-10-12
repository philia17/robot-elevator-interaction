# RoArm Phase 3 press sequencer

This project-owned ROS 2 package implements the **offline-first**, fail-closed
sequencing layer designed in `temp/cursor_agnet_update/roarm_phase3_press_primitive_design.md`.
It never opens a serial device and never constructs firmware JSON. The only
motion output is `/roarm_safe/joint_target`; `roarm_safe_driver` remains the
sole layer allowed to produce one T:101 write.

The shared camera/arm frame, unit, TCP, and target rules are frozen in
[`docs/coordinate_contract.md`](../../docs/coordinate_contract.md). The current stock end-effector contact patch is
named `press_contact` under vendor `hand_tcp`; no fingertip is assumed. The
sequencer artifact's explicit `_mm`/`_deg` fields are a package-local legacy
configuration boundary and must not appear on `/elevator/button_target`.

## Safety defaults

- `dry_run=true` and `execution_enabled=false` by default.
- There is no serial device parameter or default.
- The included YAML is a schema/template, not runnable configuration. Its
  null/TODO measurements are deliberate blockers.
- All six taught measured poses, exact URDF names, fixture/contact/button
  metadata, explicit ordered steps, and qualified driver profiles are required.
- Vendor home, T:102, implicit interpolation, multi-joint deltas, automatic
  retry, and compensation are rejected or absent.
- `PRESS` mode is blocked unless all physical measurement/contact/activation
  qualification flags are true.
- Each microstep needs a fresh confirmation bound to its transition, measured
  start snapshot, joint, target, step number, and expiry.
- Driver fault or timeout latches abort and asks the safe driver to disarm. It
  does not command an automatic retract or torque release.

## Interfaces

Subscriptions:

- `/joint_states` (`sensor_msgs/JointState`): measured pose and teach source.
- `/roarm_safe/status` (`std_msgs/String`): convergence or terminal fault.
- `/roarm_press/operator_command` (`std_msgs/String`): explicit JSON operator
  commands such as `set_mode`, `request_transition`, `confirm_microstep`,
  `complete_inspection`, `capture_pose`, `record_cycle`, and `abort`.

Outputs:

- `/roarm_safe/joint_target` (`sensor_msgs/JointState`): exact six-name measured
  snapshot with exactly one qualified joint changed.
- `/roarm_press/status` (`std_msgs/String`): high-level/control state and audit
  identifiers.
- `/roarm_safe/set_armed` (`std_srvs/SetBool` client): armed only immediately
  before an approved single target; false on abort.

Teach capture requires ten stable measured `/joint_states` samples and emits an
audit record. It never disables torque, moves the arm, or writes YAML. The
operator must review and intentionally transfer captured values into a new
artifact.

## Offline checks

```bash
cd ros2_arm/roarm_press_sequencer
python3 -m pytest -q
```

For an isolated ROS build, build only this package in a clean temporary
workspace. Do not source or launch the hardware driver. A fake ROS smoke test
must use an isolated domain and synthetic `/joint_states` and safe-driver
status only.
