"""ROS 2 wrapper for Phase B semantic labeling backends."""

from time import monotonic

import rclpy
from rclpy.node import Node

from elevator_interfaces.msg import ButtonDetectionArray, ButtonSemantic, ButtonSemanticArray

from .vlm import LlamaCppVlm, MockVlm


class VlmLabelerNode(Node):
    def __init__(self):
        super().__init__('vlm_labeler_node')
        self.declare_parameter('backend', 'mock')
        self.declare_parameter('model_path', '')
        self.declare_parameter('mmproj_path', '')
        self.declare_parameter('detections_topic', '/elevator/button_detections')
        self.declare_parameter('semantics_topic', '/elevator/button_semantics')

        backend = self.get_parameter('backend').value
        if backend == 'mock':
            self._vlm = MockVlm()
        elif backend == 'llama_cpp':
            self._vlm = LlamaCppVlm(
                self.get_parameter('model_path').value,
                self.get_parameter('mmproj_path').value,
            )
        else:
            raise ValueError("VLM backend must be 'mock' or 'llama_cpp'")
        self._backend_name = backend
        self._publisher = self.create_publisher(
            ButtonSemanticArray,
            self.get_parameter('semantics_topic').value,
            10,
        )
        self._subscription = self.create_subscription(
            ButtonDetectionArray,
            self.get_parameter('detections_topic').value,
            self._detections_callback,
            10,
        )
        self.get_logger().info(
            'Phase B deterministic VLM labeler is active with backend={}'.format(backend)
        )

    def _detections_callback(self, message):
        detection_ids = [item.detection_id for item in message.detections]
        start = monotonic()
        results = self._vlm.label(detection_ids)
        latency_ms = (monotonic() - start) * 1000.0

        output = ButtonSemanticArray()
        output.header = message.header
        output.vlm_latency_ms = latency_ms
        output.backend = self._backend_name
        for result in results:
            item = ButtonSemantic()
            item.detection_id = result.detection_id
            item.label = result.label
            item.button_type = result.button_type
            item.pressable = result.pressable
            item.semantic_confidence = result.semantic_confidence
            item.raw_response = result.raw_response
            output.semantics.append(item)
        self._publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = VlmLabelerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
