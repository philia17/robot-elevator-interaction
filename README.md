# Robot–Elevator Interaction

Perception, geometric reasoning and guarded robot-arm software for recognising and activating physical elevator buttons. This research implementation combines RGB-D perception on a Jetson AGX Orin with a RoArm-M3 arm and explicit operator control.

[Project overview and complete platform figure](https://philia17.github.io/#lift-title) · [Siheng Ye](https://github.com/philia17)

## Research and contribution

This is team research at the University of Nottingham Ningbo China. **Siheng Ye's recorded contribution** covers the staged perception pipeline, RGB-D localisation, calibration and guarded robot-arm integration. The source includes work across the integrated team system; it is not a claim of sole authorship.

The related paper, **“Robot–Elevator Interaction: From Physical Panel Recognition to Activation,” IHCI 2026 Paper 105**, was accepted for oral presentation. Proceedings publication and indexing are not claimed.

The physical evidence consists of **two qualitative staged demonstrations**. They do not establish a repeated-trial success rate or reliability across elevators. Detector benchmarks, semantic interpretation, selected targets and physical activation are separate evidence layers.

## System outline

```text
RGB-D camera → button detector → semantic interpretation → stable target
    → operator selection → guarded approach → wrist refinement → press → return
```

The integrated research system used Gemini 335 RGB-D, YOLO11n localisation, an HOG+SVM Open/Close/Other classifier and Qwen3-VL-2B semantic interpretation. This public source snapshot contains the perception interfaces, target tracking, geometric/calibration logic, guarded control components and an optional voice module. Model weights, the machine-specific HOG+SVM deployment, captured data and accepted calibration bundles are supplied separately; this is not a ready-to-run physical robot image.

The position-controlled motion path has no force control or continuous visual servo. The voice module remains dry-run and is separate from the paper's physical-motion acceptance path.

## Code map

| Directory | Contents |
| --- | --- |
| `ros2_perception/ros2_ws/src/` | ROS 2 messages, camera adapters, vision, semantic and pipeline packages |
| `ros2_arm/` | Safe driver, guarded executors, adapters and sequencer |
| `tools/press/` | Target geometry, planning, readiness and supervised execution logic |
| `tools/calibration/` | Calibration solvers and validation logic |
| `scripts/sprint3/` | Detector contract, detector adapters and offline benchmark tooling |
| `scripts/coordinate_contract/` | Coordinate contract validation |
| `scripts/annotation_web/` | Local annotation and export tools |
| `YOLO/scripts/` | Dataset preparation, training, validation and export tools |
| `voice_control/` | Optional local voice parser, confirmation state machine and tests |

Start with the [detector contract](docs/detector_interface_contract.md) and [coordinate contract](docs/coordinate_contract.md). Model and data paths in the public configuration are examples and must be configured locally.

## Run the offline checks

The supported public check does not launch ROS, access a serial device, capture camera/audio data, move hardware or download models.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
bash scripts/run_offline_tests.sh
```

Validated on macOS with Python 3.14: **318 core tests and 48 voice tests passed**. The core checks cover coordinate/detector contracts, RGB-D geometry, target tracking, camera-axis validation, driver admission/torque transactions and selected press-readiness logic. See [validation scope](docs/PUBLIC_RELEASE.md) for the precise boundary.

Additional original tests are retained beside their source. Some require ROS 2, hardware-specific deployment files or calibration/data fixtures that are intentionally excluded. A blanket `pytest` run over the entire tree is therefore not the supported offline acceptance command.

## Dependencies and external assets

For a ROS deployment, prepare ROS 2 and the relevant vendor camera/arm interfaces in a separate environment, resolve package dependencies from `package.xml`, and review the configuration before building. ROS compilation and live deployment were not validated for this public snapshot.

Neural detector/VLM and speech dependencies are optional. Obtain models from their respective publishers, check their terms, record the exact model version and configure local paths. Training images, private captures, generated audio, checkpoints and accepted hardware calibration are not bundled. Offline tests use synthetic fixtures and require no neural model.

## Physical operation boundary

The public arm profiles ship with transport disabled and dry-run enabled where those fields apply; real device identifiers are placeholders. Preserved admission, lease, feedback, collision-clearance and operator-confirmation checks must not be bypassed. A successful offline test is not authorization for motion. Any physical integration requires a new on-site hardware review, current calibration and an operator-controlled test area. Voice STOP does not replace a physical emergency stop.

## Source and licensing

This is a selective source snapshot from the research implementation, without its internal history, personnel records, machine logs or private data. See [release scope](docs/PUBLIC_RELEASE.md).

Existing package declarations are preserved: ROS perception packages declare Apache-2.0, arm packages declare MIT, and the voice package declares Proprietary. No new repository-wide license is granted by this export. See [LICENSING.md](LICENSING.md) before reusing or redistributing code or externally obtained assets.
