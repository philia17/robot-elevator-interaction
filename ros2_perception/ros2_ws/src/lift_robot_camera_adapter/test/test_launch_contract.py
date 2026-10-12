from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_integrated_launch_disables_vendor_tf_and_does_not_publish_pending_cart_edge():
    cpp_root = ROOT.parent / 'lift_robot_camera_adapter_cpp'
    launch_source = (cpp_root / 'launch' / 'gemini335_canonical_components.launch.py').read_text()
    node_source = (cpp_root / 'src' / 'gemini335_canonical_adapter.cpp').read_text()
    assert "'publish_tf': False" in launch_source
    assert 'cart_base' not in launch_source
    assert 'StaticTransformBroadcaster' in node_source
    assert 'pending cart edge is not published' in node_source


def test_adapter_has_no_hp60c_runtime_dependency():
    cpp_root = ROOT.parent / 'lift_robot_camera_adapter_cpp'
    runtime_files = [
        cpp_root / 'config' / 'gemini335_canonical_cpp.yaml',
        cpp_root / 'launch' / 'gemini335_canonical_components.launch.py',
        cpp_root / 'src' / 'gemini335_canonical_adapter.cpp',
    ]
    for path in runtime_files:
        assert 'hp60c' not in path.read_text().lower()


def test_native_adapter_preserves_resolution_and_uses_unique_ownership():
    cpp_root = ROOT.parent / 'lift_robot_camera_adapter_cpp'
    config = (cpp_root / 'config' / 'gemini335_canonical_cpp.yaml').read_text()
    node_source = (cpp_root / 'src' / 'gemini335_canonical_adapter.cpp').read_text()
    assert 'max_image_publish_hz: 30.0' in config
    assert 'phase2_preview_publish_hz: 5.0' in config
    assert 'canonical_width: 1920' in config
    assert 'canonical_height: 1080' in config
    assert 'Image::UniquePtr' in node_source
    assert 'publish(std::move(pending_color_))' in node_source
    assert 'kMaxPairSkewNs' in node_source
    assert 'make_phase2_rgbd_preview' in node_source
    assert '8UC5' in node_source
    assert 'SensorDataQoS().keep_last(1)' in node_source


def test_python_package_has_no_high_bandwidth_adapter_entry_point():
    setup_source = (ROOT / 'setup.py').read_text()
    assert 'gemini335_canonical_adapter =' not in setup_source
    assert not (ROOT / 'lift_robot_camera_adapter' / 'node.py').exists()
