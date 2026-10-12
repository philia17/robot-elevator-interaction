"""Installed entry points for Phase B vision nodes."""


def frame_gate_main():
    from .frame_gate_node import main
    main()


def button_detector_main():
    from .button_detector_node import main
    main()


def rgbd_button_localizer_main():
    from .rgbd_button_localizer_node import main
    main()


def m0_target_observer_main():
    from .m0_target_observer_node import main
    main()
