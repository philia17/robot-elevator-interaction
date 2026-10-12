cmake_minimum_required(VERSION 3.8)
project(elevator_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FrameStatus.msg"
  "msg/TaskContext.msg"
  "msg/ButtonDetection.msg"
  "msg/ButtonDetectionArray.msg"
  "msg/ButtonSemantic.msg"
  "msg/ButtonSemanticArray.msg"
  "msg/ButtonObservation.msg"
  "msg/ButtonObservationArray.msg"
  "msg/ButtonSurfaceGeometry.msg"
  "msg/ButtonObservation3D.msg"
  "msg/ButtonObservation3DArray.msg"
  "msg/M0Target3D.msg"
  "msg/TargetCandidate2D.msg"
  "msg/TargetButton2D.msg"
  "msg/PerceptionHealth.msg"
  "action/RecognizeTargetButton.action"
  "action/PrecontactMove.action"
  "action/PrecontactXY.action"
  "action/StageBRehearsal.action"
  "action/StandbyRoute.action"
  "srv/PreparePrecontact.srv"
  "srv/DiscardPreparedPrecontact.srv"
  "srv/PreparePrecontactXY.srv"
  "srv/PrepareStageBRehearsal.srv"
  "srv/PrecontactSpeedLease.srv"
  "srv/PrepareStandbyRoute.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
