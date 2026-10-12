"""CPU-only tests for RGB-D sampling and camera projection."""

import math

import numpy as np
import pytest

from elevator_vision.rgbd_geometry import (
    depth_to_meters,
    estimate_surface_geometry_from_bbox,
    project_pixel_to_camera,
    robust_depth_from_bbox,
    scale_pixel_between_images,
    surface_provenance_complete,
)


HEIGHT = 100
WIDTH = 100
FX = 200.0
FY = 205.0
CX = 49.5
CY = 49.5
BBOX = (40.0, 40.0, 60.0, 60.0)
CENTER = (50.0, 50.0)
SHA256 = 'a' * 64


def synthetic_plane_depth(
    outward_normal=(0.0, 0.0, -1.0),
    point_on_plane=(0.0, 0.0, 1.0),
    encoding='32FC1',
):
    normal = np.asarray(outward_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    point = np.asarray(point_on_plane, dtype=np.float64)
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    rays = np.stack(
        (
            (columns - CX) / FX,
            (rows - CY) / FY,
            np.ones_like(columns, dtype=np.float64),
        ),
        axis=-1,
    )
    numerator = float(np.dot(normal, point))
    denominator = np.einsum('ijk,k->ij', rays, normal)
    depth = numerator / denominator
    if encoding == '16UC1':
        return np.rint(depth * 1000.0).astype(np.uint16)
    return depth.astype(np.float32)


def add_button_surface(depth, depth_m=0.98, encoding='32FC1'):
    result = np.array(depth, copy=True)
    value = int(round(depth_m * 1000.0)) if encoding == '16UC1' else depth_m
    result[44:57, 44:57] = value
    return result


def estimate(depth, encoding='32FC1', **kwargs):
    kwargs.setdefault('minimum_plane_points', 40)
    kwargs.setdefault('minimum_plane_span_m', 0.005)
    return estimate_surface_geometry_from_bbox(
        depth,
        BBOX,
        CENTER,
        encoding,
        FX,
        FY,
        CX,
        CY,
        **kwargs,
    )


def provenance(**overrides):
    values = {
        'geometry_valid': True,
        'camera_info_sha256': SHA256,
        'canonical_frame_id': 'camera_optical_frame',
        'raw_to_canonical_alias_id': (
            'gemini335_color_optical_frame_to_camera_optical_frame_identity_v1'
        ),
        'raw_to_canonical_alias_status': 'verified',
        'raw_to_canonical_adapter_sha256': SHA256,
        'section_b_calibration_id': 'section_b_id',
        'camera_optical_axis_status': 'verified',
        'rgb_depth_registration_status': 'verified',
        'depth_scale_status': 'verified',
    }
    values.update(overrides)
    return surface_provenance_complete(**values)


def test_surface_provenance_requires_all_exact_verified_statuses():
    assert provenance()
    for field, pending in (
        (
            'raw_to_canonical_alias_status',
            'software_identity_verified_runtime_bound_physical_axes_pending',
        ),
        (
            'camera_optical_axis_status',
            'pending_physical_xyz_validation',
        ),
        (
            'rgb_depth_registration_status',
            'verified_config_and_observed_accuracy_pending',
        ),
        (
            'depth_scale_status',
            'verified_driver_sdk_metric_accuracy_pending',
        ),
    ):
        assert not provenance(**{field: pending})


@pytest.mark.parametrize(
    'field',
    (
        'camera_info_sha256',
        'canonical_frame_id',
        'raw_to_canonical_alias_id',
        'raw_to_canonical_alias_status',
        'raw_to_canonical_adapter_sha256',
        'section_b_calibration_id',
    ),
)
def test_surface_provenance_requires_all_incremental_bindings(field):
    assert not provenance(**{field: ''})


def test_16uc1_millimetres_convert_to_metres():
    result = depth_to_meters(
        np.array([[0, 500, 1250]], dtype=np.uint16),
        '16UC1',
    )
    assert math.isnan(result[0, 0])
    assert result[0, 1] == pytest.approx(0.5)
    assert result[0, 2] == pytest.approx(1.25)


def test_32fc1_metres_preserve_valid_values_and_reject_invalid_values():
    image = np.array([[0.0, np.nan, np.inf, 1.25]], dtype=np.float32)
    result = depth_to_meters(image, '32FC1')
    assert all(math.isnan(value) for value in result[0, :3])
    assert result[0, 3] == pytest.approx(1.25)


def test_all_zero_depth_is_invalid():
    sample = robust_depth_from_bbox(
        np.zeros((20, 20), dtype=np.uint16),
        (0, 0, 20, 20),
        (10, 10),
        '16UC1',
    )
    assert not sample.depth_valid
    assert math.isnan(sample.depth_m)
    assert sample.valid_ratio == 0.0


def test_bbox_is_clipped_at_image_boundary():
    image = np.full((20, 20), 800, dtype=np.uint16)
    sample = robust_depth_from_bbox(
        image,
        (-100, -50, 12, 12),
        (4, 4),
        '16UC1',
    )
    assert sample.depth_valid
    assert sample.depth_m == pytest.approx(0.8)


def test_center_roi_uses_median_and_rejects_outlier():
    image = np.full((30, 30), 2000, dtype=np.uint16)
    image[10:20, 10:20] = 700
    image[15, 15] = 4500
    sample = robust_depth_from_bbox(
        image,
        (0, 0, 30, 30),
        (15, 15),
        '16UC1',
        center_roi_ratio=0.30,
    )
    assert sample.depth_valid
    assert sample.source == 'bbox_center_roi'
    assert sample.depth_m == pytest.approx(0.7)


def test_center_window_is_used_when_bbox_center_roi_is_sparse():
    image = np.zeros((21, 21), dtype=np.uint16)
    image[6:15, 6:15] = 1200
    image[7:14, 7:14] = 0
    sample = robust_depth_from_bbox(
        image,
        (0, 0, 20, 20),
        (10, 10),
        '16UC1',
        minimum_valid_pixels=9,
        center_roi_ratio=0.30,
        fallback_window_size=9,
    )
    assert sample.depth_valid
    assert sample.source == 'center_window'
    assert sample.depth_m == pytest.approx(1.2)


def test_valid_ratio_is_reported_for_selected_region():
    image = np.zeros((10, 10), dtype=np.uint16)
    image[3:5, 3:7] = 1000
    sample = robust_depth_from_bbox(
        image,
        (0, 0, 10, 10),
        (5, 5),
        '16UC1',
        minimum_valid_pixels=4,
        center_roi_ratio=0.40,
    )
    assert sample.depth_valid
    assert sample.source == 'bbox_center_roi'
    assert sample.valid_ratio == pytest.approx(0.5)


def test_known_pinhole_projection():
    x_m, y_m, z_m = project_pixel_to_camera(
        u=420.0,
        v=190.0,
        depth_m=2.0,
        fx=400.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )
    assert x_m == pytest.approx(0.5)
    assert y_m == pytest.approx(-0.2)
    assert z_m == pytest.approx(2.0)


def test_frontoparallel_panel_uses_button_surface_and_outward_normal():
    depth = add_button_surface(synthetic_plane_depth())
    result = estimate(depth)

    assert result.geometry_valid
    assert result.failure_reason == ''
    assert result.surface_point_m[2] == pytest.approx(0.98)
    assert result.panel_outward_normal == pytest.approx(
        (0.0, 0.0, -1.0),
        abs=1e-8,
    )
    assert np.dot(
        result.panel_outward_normal,
        result.surface_point_m,
    ) < 0.0
    assert np.linalg.norm(result.panel_outward_normal) == pytest.approx(1.0)
    assert result.plane_rms_residual_m < 1e-9
    assert result.plane_inlier_ratio == pytest.approx(1.0)
    assert result.plane_inlier_count == result.plane_sample_count


def test_slanted_panel_normal_matches_known_plane():
    expected = np.asarray((0.18, -0.10, -0.9786), dtype=np.float64)
    expected /= np.linalg.norm(expected)
    depth = add_button_surface(
        synthetic_plane_depth(
            outward_normal=expected,
            point_on_plane=(0.0, 0.0, 1.0),
        )
    )
    result = estimate(depth)

    assert result.geometry_valid
    assert np.dot(result.panel_outward_normal, expected) > 0.99999
    assert np.dot(result.panel_outward_normal, result.surface_point_m) < 0.0
    assert result.normal_angular_uncertainty_rad > 0.0
    assert result.normal_angular_uncertainty_rad < math.radians(10.0)


def test_deterministic_mad_refit_rejects_plane_outliers():
    depth = add_button_surface(synthetic_plane_depth())
    rows, columns = np.mgrid[30:70:4, 30:70:4]
    depth[rows, columns] = 1.045

    result = estimate(depth)

    assert result.geometry_valid
    assert result.plane_inlier_ratio < 1.0
    assert result.plane_inlier_ratio > 0.65
    assert result.panel_outward_normal == pytest.approx(
        (0.0, 0.0, -1.0),
        abs=1e-7,
    )
    assert result.plane_rms_residual_m < 1e-7


def test_sparse_and_degenerate_annuli_fail_closed():
    sparse = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    sparse[44:57, 44:57] = 0.98
    sparse[32, 32:68] = 1.0

    result = estimate(sparse, minimum_plane_points=20)

    assert not result.geometry_valid
    assert math.isnan(result.panel_outward_normal[0])
    assert (
        'degenerate' in result.failure_reason
        or 'span' in result.failure_reason
    )


def test_nonfinite_geometry_inputs_fail_closed():
    result = estimate_surface_geometry_from_bbox(
        synthetic_plane_depth(),
        (40.0, 40.0, math.nan, 60.0),
        CENTER,
        '32FC1',
        FX,
        FY,
        CX,
        CY,
    )

    assert not result.geometry_valid
    assert 'finite' in result.failure_reason


def test_16uc1_and_32fc1_plane_estimates_agree():
    expected = np.asarray((0.12, 0.08, -0.9895), dtype=np.float64)
    expected /= np.linalg.norm(expected)
    depth_32 = add_button_surface(
        synthetic_plane_depth(expected, encoding='32FC1'),
        encoding='32FC1',
    )
    depth_16 = add_button_surface(
        synthetic_plane_depth(expected, encoding='16UC1'),
        encoding='16UC1',
    )

    result_32 = estimate(depth_32, '32FC1')
    result_16 = estimate(depth_16, '16UC1')

    assert result_32.geometry_valid
    assert result_16.geometry_valid
    assert result_16.surface_point_m == pytest.approx(
        result_32.surface_point_m,
        abs=1e-3,
    )
    assert np.dot(
        result_16.panel_outward_normal,
        result_32.panel_outward_normal,
    ) > 0.999


def test_surface_point_covariance_is_finite_symmetric_and_psd():
    depth = add_button_surface(synthetic_plane_depth())
    depth[48:53, 48:53] += np.linspace(
        -0.002,
        0.002,
        25,
        dtype=np.float32,
    ).reshape(5, 5)

    result = estimate(
        depth,
        pixel_sigma_px=1.25,
        depth_noise_floor_m=0.001,
    )
    covariance = np.asarray(
        result.surface_point_covariance_m2,
        dtype=np.float64,
    ).reshape(3, 3)

    assert result.geometry_valid
    assert np.all(np.isfinite(covariance))
    assert covariance == pytest.approx(covariance.T, abs=1e-15)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-15
    assert np.all(np.diag(covariance) > 0.0)
    assert result.depth_sigma_m >= 0.001


@pytest.mark.parametrize('fx,fy', [(0.0, 500.0), (400.0, 0.0), (-1.0, 500.0)])
def test_invalid_focal_length_raises(fx, fy):
    with pytest.raises(ValueError):
        project_pixel_to_camera(1, 1, 1, fx, fy, 0, 0)


def test_frame_gate_coordinate_scaling_preserves_endpoints():
    assert scale_pixel_between_images(0, 0, 640, 480, 1920, 1080) == (0.0, 0.0)
    assert scale_pixel_between_images(639, 479, 640, 480, 1920, 1080) == (
        1919.0,
        1079.0,
    )


def test_gemini_16_by_9_detector_coordinates_scale_to_native_rgb():
    assert scale_pixel_between_images(0, 0, 640, 360, 1920, 1080) == (0.0, 0.0)
    assert scale_pixel_between_images(639, 359, 640, 360, 1920, 1080) == (
        1919.0,
        1079.0,
    )
