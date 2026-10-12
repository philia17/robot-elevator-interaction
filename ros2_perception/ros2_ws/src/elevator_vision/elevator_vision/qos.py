"""ROS-independent validation for configurable image QoS."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageQosSettings:
    reliability: str
    durability: str
    depth: int


def _normalize_policy(value, allowed, policy_name):
    normalized = str(value or '').strip().lower()
    if normalized not in allowed:
        allowed_text = "', '".join(sorted(allowed))
        raise ValueError(
            "QoS {} must be one of '{}'".format(policy_name, allowed_text)
        )
    return normalized


def parse_image_qos(reliability='best_effort', durability='volatile', depth=10):
    reliability_value = _normalize_policy(
        reliability,
        {'best_effort', 'reliable'},
        'reliability',
    )
    durability_value = _normalize_policy(
        durability,
        {'volatile', 'transient_local'},
        'durability',
    )
    queue_depth = int(depth)
    if queue_depth <= 0:
        raise ValueError('QoS queue depth must be positive')
    return ImageQosSettings(
        reliability=reliability_value,
        durability=durability_value,
        depth=queue_depth,
    )


def is_compatible_image_qos(publisher, subscriber):
    """Return whether offered publisher QoS satisfies requested subscriber QoS.

    This mirrors the ROS 2 requested/offered model for the reliability and
    durability policies used by the RGB image subscription. A reliable
    subscriber cannot receive from a best-effort publisher, and a
    transient-local subscriber cannot receive from a volatile publisher.
    Queue depth is intentionally not part of compatibility.
    """

    offered = parse_image_qos(
        publisher.reliability,
        publisher.durability,
        max(1, int(publisher.depth)),
    )
    requested = parse_image_qos(
        subscriber.reliability,
        subscriber.durability,
        max(1, int(subscriber.depth)),
    )
    if (
        requested.reliability == 'reliable'
        and offered.reliability == 'best_effort'
    ):
        return False
    if (
        requested.durability == 'transient_local'
        and offered.durability == 'volatile'
    ):
        return False
    return True


def compatible_subscriber_qos_for_publisher(
    publisher_reliability,
    publisher_durability,
    publisher_depth=10,
):
    """Select a compatible subscriber QoS for an observed image publisher."""

    offered = parse_image_qos(
        publisher_reliability,
        publisher_durability,
        publisher_depth,
    )
    selected = ImageQosSettings(
        reliability=offered.reliability,
        durability='volatile',
        depth=offered.depth,
    )
    if not is_compatible_image_qos(offered, selected):
        raise ValueError('selected subscriber QoS is not compatible')
    return selected
