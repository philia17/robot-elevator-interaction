# Public source release scope

Snapshot date: 12 October 2026.
Source revision: `84266bee557d558779f2800d9d7e7335985134df`.

This snapshot includes selected source, example configuration, contracts and existing tests. It excludes the private development history, personnel/context documents, device snapshots, raw captures, model weights, internal experiment logs, calibrated deployment bundles and historical vendor materials.

## Source adaptations

- Developer home and storage paths were replaced with generic example locations such as `/home/robot`, `/opt/robot-elevator-interaction` and `/data`.
- Real USB serial identifiers were replaced with an explicit placeholder.
- Included arm YAML profiles use `transport: none` and `dry_run: true` where these keys are present. Runtime guard logic remains unchanged.
- The public README, this scope note, licensing note and offline test entry point are new.
- A stale voice status-document link was replaced by the module's included boundary section.

## Validated checks

The supported entry point is `bash scripts/run_offline_tests.sh` from the repository root. Its exact test list is visible in that script. It runs synthetic/offline tests for coordinate and detector contracts, target tracking, geometry, camera axis, safe-driver core, torque transaction logic, selected press-readiness logic and the optional voice module.

The public snapshot also received Python syntax checks, Markdown local-link checks, a redacted secret scan and exact comparisons against sensitive strings found in the excluded development history. Source algorithms were not changed to make checks pass.

## What was not validated

- ROS 2 or C++ compilation, GPU inference, model accuracy, camera/audio capture and physical arm operation.
- Original tests that require excluded private deployment manifests, accepted calibration or captured data.
- Deployment equivalence of the inert public profiles to any particular robot.

A missing calibration or deployment artifact must remain a failed readiness condition. Do not fabricate a replacement acceptance record or disable a guard to run hardware.
