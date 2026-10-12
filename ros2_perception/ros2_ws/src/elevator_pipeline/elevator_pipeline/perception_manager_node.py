"""ROS 2 Action server coordinating the existing Phase B pipeline."""

from threading import RLock
from time import monotonic, sleep

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from elevator_interfaces.action import RecognizeTargetButton
from elevator_interfaces.msg import (
    ButtonDetectionArray,
    ButtonObservationArray,
    ButtonSemanticArray,
    FrameStatus,
    TargetButton2D,
    TargetCandidate2D,
    TaskContext,
)

from .floor_map import FloorMap
from .manager_core import (
    CANCELLED,
    FAILED,
    SUCCESS,
    TIMEOUT,
    GoalValidationError,
    ManagerStateMachine,
)


class PerceptionManagerNode(Node):
    def __init__(self):
        super().__init__('perception_manager_node')
        self.declare_parameter('floor_map_path', '')
        self.declare_parameter('action_name', '/elevator/recognize_target_button')
        self.declare_parameter('maximum_action_timeout_sec', 120.0)
        self.declare_parameter('feedback_period_sec', 0.1)
        self.declare_parameter('task_context_topic', '/elevator/task_context')
        self.declare_parameter('task_state_topic', '/elevator/task_state')
        self.declare_parameter('frame_status_topic', '/elevator/frame_status')
        self.declare_parameter('detections_topic', '/elevator/button_detections')
        self.declare_parameter('semantics_topic', '/elevator/button_semantics')
        self.declare_parameter('observations_topic', '/elevator/button_observations')
        self.declare_parameter('candidate_topic', '/elevator/target_candidate')
        self.declare_parameter('target_topic', '/elevator/target_button')

        floor_map = FloorMap.load(self.get_parameter('floor_map_path').value)
        self._core = ManagerStateMachine(
            floor_map,
            maximum_timeout_sec=self.get_parameter(
                'maximum_action_timeout_sec'
            ).value,
        )
        self._lock = RLock()
        self._goal_reserved = False
        self._callback_group = ReentrantCallbackGroup()
        self._task_publisher = self.create_publisher(
            TaskContext,
            self.get_parameter('task_context_topic').value,
            10,
        )
        self._state_publisher = self.create_publisher(
            String,
            self.get_parameter('task_state_topic').value,
            10,
        )
        self._subscriptions = [
            self.create_subscription(
                FrameStatus,
                self.get_parameter('frame_status_topic').value,
                self._frame_callback,
                10,
                callback_group=self._callback_group,
            ),
            self.create_subscription(
                ButtonDetectionArray,
                self.get_parameter('detections_topic').value,
                self._detections_callback,
                10,
                callback_group=self._callback_group,
            ),
            self.create_subscription(
                ButtonSemanticArray,
                self.get_parameter('semantics_topic').value,
                self._semantics_callback,
                10,
                callback_group=self._callback_group,
            ),
            self.create_subscription(
                ButtonObservationArray,
                self.get_parameter('observations_topic').value,
                self._observations_callback,
                10,
                callback_group=self._callback_group,
            ),
            self.create_subscription(
                TargetCandidate2D,
                self.get_parameter('candidate_topic').value,
                self._candidate_callback,
                10,
                callback_group=self._callback_group,
            ),
            self.create_subscription(
                TargetButton2D,
                self.get_parameter('target_topic').value,
                self._target_callback,
                10,
                callback_group=self._callback_group,
            ),
        ]
        self._action_server = ActionServer(
            self,
            RecognizeTargetButton,
            self.get_parameter('action_name').value,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self._publish_state()
        self.get_logger().info('Phase C perception Action manager is active')

    def _goal_callback(self, request):
        with self._lock:
            if self._goal_reserved or self._core.is_active:
                self.get_logger().warning('rejecting goal: another task is active')
                return GoalResponse.REJECT
            try:
                self._core.validate_goal(
                    request.building_id,
                    request.requested_floor,
                    request.timeout_sec,
                )
            except GoalValidationError as exc:
                self.get_logger().warning('rejecting invalid goal: {}'.format(exc))
                return GoalResponse.REJECT
            self._goal_reserved = True
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        with self._lock:
            if not self._goal_reserved:
                return CancelResponse.REJECT
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        request = goal_handle.request
        try:
            with self._lock:
                request_id = self._core.start_goal(
                    request.building_id,
                    request.requested_floor,
                    request.timeout_sec,
                    monotonic(),
                )
                snapshot = self._core.snapshot()
            self._publish_task(snapshot, active=True)
            self._publish_state(snapshot)

            feedback_period = max(
                0.02,
                float(self.get_parameter('feedback_period_sec').value),
            )
            while rclpy.ok():
                with self._lock:
                    if goal_handle.is_cancel_requested:
                        self._core.cancel()
                    else:
                        self._core.tick(monotonic())
                    snapshot = self._core.snapshot()
                self._publish_state(snapshot)
                feedback = RecognizeTargetButton.Feedback()
                feedback.state = snapshot.state
                feedback.detected_count = snapshot.detected_count
                feedback.labeled_count = snapshot.labeled_count
                feedback.current_candidate = snapshot.current_candidate
                feedback.progress = snapshot.progress
                goal_handle.publish_feedback(feedback)
                if snapshot.terminal:
                    break
                sleep(feedback_period)

            result = RecognizeTargetButton.Result()
            if snapshot.state == SUCCESS and snapshot.target is not None:
                result.success = True
                result.target = snapshot.target
                result.failure_reason = ''
                goal_handle.succeed()
            else:
                result.success = False
                result.target = self._invalid_target(snapshot)
                result.failure_reason = snapshot.failure_reason or snapshot.state
                if snapshot.state == CANCELLED:
                    goal_handle.canceled()
                else:
                    goal_handle.abort()
            self._publish_task(snapshot, active=False)
            return result
        except Exception as exc:
            self.get_logger().error('Action execution failed: {}'.format(exc))
            with self._lock:
                self._core.fail(str(exc))
                snapshot = self._core.snapshot()
            result = RecognizeTargetButton.Result()
            result.success = False
            result.target = self._invalid_target(snapshot)
            result.failure_reason = str(exc)
            goal_handle.abort()
            self._publish_task(snapshot, active=False)
            return result
        finally:
            with self._lock:
                self._core.reset()
                self._goal_reserved = False
                idle = self._core.snapshot()
            self._publish_state(idle)

    def _invalid_target(self, snapshot):
        target = TargetButton2D()
        target.header.stamp = self.get_clock().now().to_msg()
        target.valid = False
        target.request_id = snapshot.request_id
        target.building_id = snapshot.building_id
        target.requested_floor = snapshot.requested_floor
        target.status = snapshot.state
        target.failure_reason = snapshot.failure_reason or snapshot.state
        return target

    def _publish_task(self, snapshot, active):
        message = TaskContext()
        message.header.stamp = self.get_clock().now().to_msg()
        message.active = bool(active)
        message.request_id = snapshot.request_id
        message.building_id = snapshot.building_id
        message.requested_floor = snapshot.requested_floor
        message.timeout_sec = snapshot.timeout_sec
        self._task_publisher.publish(message)

    def _publish_state(self, snapshot=None):
        if snapshot is None:
            with self._lock:
                snapshot = self._core.snapshot()
        message = String()
        message.data = snapshot.state
        self._state_publisher.publish(message)

    def _frame_callback(self, message):
        with self._lock:
            self._core.record_frame(message.accepted)

    def _detections_callback(self, message):
        with self._lock:
            self._core.record_detections(len(message.detections))

    def _semantics_callback(self, message):
        with self._lock:
            self._core.record_semantics(len(message.semantics))

    def _observations_callback(self, message):
        with self._lock:
            self._core.record_observations()

    def _candidate_callback(self, message):
        with self._lock:
            self._core.record_candidate(message)

    def _target_callback(self, message):
        with self._lock:
            self._core.record_target(message)

    def destroy_node(self):
        self._action_server.destroy()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionManagerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
