"""Installed entry points for elevator pipeline nodes."""


def observation_fusion_main():
    from .observation_fusion_node import main
    main()


def target_resolver_main():
    from .target_resolver_node import main
    main()


def temporal_tracker_main():
    from .temporal_tracker_node import main
    main()


def perception_manager_main():
    from .perception_manager_node import main
    main()


def visualization_main():
    from .visualization_node import main
    main()


def diagnostics_main():
    from .diagnostics_node import main
    main()
