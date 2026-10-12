"""ROS-independent RGB-D sampling and pinhole projection helpers."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DepthSample:
    depth_valid: bool
    depth_m: float
    valid_ratio: float
    valid_pixels: int
    sample_pixels: int
    source: str
    depth_mad_m: float = math.nan
    depth_sigma_m: float = math.nan


@dataclass(frozen=True)
class SurfaceGeometryEstimate:
    """Camera-frame button surface and surrounding panel estimate."""

    geometry_valid: bool
    surface_point_m: tuple
    surface_point_covariance_m2: tuple
    panel_outward_normal: tuple
    normal_angular_uncertainty_rad: float
    plane_rms_residual_m: float
    plane_inlier_ratio: float
    plane_sample_count: int
    plane_inlier_count: int
    plane_span_m: float
    depth_mad_m: float
    depth_sigma_m: float
    depth_source: str
    failure_reason: str


def surface_provenance_complete(
    *,
    geometry_valid,
    camera_info_sha256,
    canonical_frame_id,
    raw_to_canonical_alias_id,
    raw_to_canonical_alias_status,
    raw_to_canonical_adapter_sha256,
    section_b_calibration_id,
    camera_optical_axis_status,
    rgb_depth_registration_status,
    depth_scale_status,
):
    """Return true only for fully bound geometry with all verified gates."""

    camera_info_digest = str(camera_info_sha256 or '').strip().lower()
    adapter_digest = str(
        raw_to_canonical_adapter_sha256 or ''
    ).strip().lower()
    bindings_complete = (
        len(camera_info_digest) == 64
        and all(character in '0123456789abcdef' for character in camera_info_digest)
        and str(canonical_frame_id or '').strip() == 'camera_optical_frame'
        and bool(str(raw_to_canonical_alias_id or '').strip())
        and len(adapter_digest) == 64
        and all(character in '0123456789abcdef' for character in adapter_digest)
        and bool(str(section_b_calibration_id or '').strip())
    )
    execution_provenance_gates_verified = all(
        str(status).strip() == 'verified'
        for status in (
            raw_to_canonical_alias_status,
            camera_optical_axis_status,
            rgb_depth_registration_status,
            depth_scale_status,
        )
    )
    return bool(
        geometry_valid
        and bindings_complete
        and execution_provenance_gates_verified
    )


def depth_to_meters(depth_image, encoding):
    """Convert a 2-D 16UC1 millimetre or 32FC1 metre image to metres.

    Non-finite and non-positive values are represented as NaN so callers
    cannot accidentally treat missing depth as a physical measurement.
    """

    array = np.asarray(depth_image)
    if array.ndim != 2:
        raise ValueError('depth image must be two-dimensional')
    normalized = str(encoding or '').strip().upper()
    if normalized == '16UC1':
        converted = array.astype(np.float64) * 0.001
    elif normalized == '32FC1':
        converted = array.astype(np.float64, copy=True)
    else:
        raise ValueError('unsupported depth encoding: {}'.format(encoding))
    invalid = ~np.isfinite(converted) | (converted <= 0.0)
    converted[invalid] = np.nan
    return converted


def _valid_values(region, min_depth_m, max_depth_m):
    if region.size == 0:
        return np.empty((0,), dtype=np.float64), 0.0
    mask = (
        np.isfinite(region)
        & (region >= min_depth_m)
        & (region <= max_depth_m)
    )
    values = region[mask]
    return values, float(values.size) / float(region.size)


def _sample_result(
    region,
    min_depth_m,
    max_depth_m,
    minimum_valid_pixels,
    source,
):
    values, ratio = _valid_values(region, min_depth_m, max_depth_m)
    depth_mad_m = math.nan
    depth_sigma_m = math.nan
    if values.size:
        median = float(np.median(values))
        depth_mad_m = float(np.median(np.abs(values - median)))
        depth_sigma_m = 1.4826 * depth_mad_m
    if values.size < minimum_valid_pixels:
        return DepthSample(
            depth_valid=False,
            depth_m=math.nan,
            valid_ratio=ratio,
            valid_pixels=int(values.size),
            sample_pixels=int(region.size),
            source=source,
            depth_mad_m=depth_mad_m,
            depth_sigma_m=depth_sigma_m,
        )
    return DepthSample(
        depth_valid=True,
        depth_m=float(np.median(values)),
        valid_ratio=ratio,
        valid_pixels=int(values.size),
        sample_pixels=int(region.size),
        source=source,
        depth_mad_m=depth_mad_m,
        depth_sigma_m=depth_sigma_m,
    )


def robust_depth_from_bbox(
    depth_image,
    bbox,
    center_pixel,
    encoding,
    min_depth_m=0.15,
    max_depth_m=5.0,
    minimum_valid_pixels=9,
    center_roi_ratio=0.30,
    fallback_window_size=9,
):
    """Estimate object depth from a central bbox ROI with a 9x9 fallback."""

    if not 0.0 < float(min_depth_m) < float(max_depth_m):
        raise ValueError(
            'depth range must satisfy 0 < min_depth_m < max_depth_m'
        )
    if int(minimum_valid_pixels) <= 0:
        raise ValueError('minimum_valid_pixels must be positive')
    if not 0.0 < float(center_roi_ratio) <= 1.0:
        raise ValueError('center_roi_ratio must be in (0, 1]')
    window_size = int(fallback_window_size)
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError('fallback_window_size must be a positive odd integer')

    depth_m = depth_to_meters(depth_image, encoding)
    height, width = depth_m.shape
    x_min, y_min, x_max, y_max = (float(value) for value in bbox)
    clipped_x_min = max(0.0, min(float(width), x_min))
    clipped_y_min = max(0.0, min(float(height), y_min))
    clipped_x_max = max(0.0, min(float(width), x_max))
    clipped_y_max = max(0.0, min(float(height), y_max))

    roi = np.empty((0, 0), dtype=np.float64)
    if clipped_x_max > clipped_x_min and clipped_y_max > clipped_y_min:
        bbox_width = clipped_x_max - clipped_x_min
        bbox_height = clipped_y_max - clipped_y_min
        bbox_center_x = (clipped_x_min + clipped_x_max) / 2.0
        bbox_center_y = (clipped_y_min + clipped_y_max) / 2.0
        roi_width = max(1.0, bbox_width * float(center_roi_ratio))
        roi_height = max(1.0, bbox_height * float(center_roi_ratio))
        left = max(
            0,
            min(width, int(math.floor(bbox_center_x - roi_width / 2.0))),
        )
        right = max(
            0,
            min(width, int(math.ceil(bbox_center_x + roi_width / 2.0))),
        )
        top = max(
            0,
            min(height, int(math.floor(bbox_center_y - roi_height / 2.0))),
        )
        bottom = max(
            0,
            min(height, int(math.ceil(bbox_center_y + roi_height / 2.0))),
        )
        roi = depth_m[top:bottom, left:right]

    primary = _sample_result(
        roi,
        float(min_depth_m),
        float(max_depth_m),
        int(minimum_valid_pixels),
        'bbox_center_roi',
    )
    if primary.depth_valid:
        return primary

    center_x, center_y = (float(value) for value in center_pixel)
    pixel_x = int(round(center_x))
    pixel_y = int(round(center_y))
    half = window_size // 2
    left = max(0, pixel_x - half)
    right = min(width, pixel_x + half + 1)
    top = max(0, pixel_y - half)
    bottom = min(height, pixel_y + half + 1)
    fallback = depth_m[top:bottom, left:right]
    return _sample_result(
        fallback,
        float(min_depth_m),
        float(max_depth_m),
        int(minimum_valid_pixels),
        'center_window',
    )


def project_pixel_to_camera(u, v, depth_m, fx, fy, cx, cy):
    """Project a color pixel and metric depth into the camera optical frame."""

    values = [u, v, depth_m, fx, fy, cx, cy]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('projection inputs must be finite')
    z_m = float(depth_m)
    fx_value = float(fx)
    fy_value = float(fy)
    if fx_value <= 0.0 or fy_value <= 0.0:
        raise ValueError('fx and fy must be positive')
    if z_m <= 0.0:
        raise ValueError('depth_m must be positive')
    x_m = (float(u) - float(cx)) * z_m / fx_value
    y_m = (float(v) - float(cy)) * z_m / fy_value
    result = (x_m, y_m, z_m)
    if not all(math.isfinite(value) for value in result):
        raise ValueError('projection produced non-finite coordinates')
    return result


def _invalid_surface_estimate(
    reason,
    depth_sample=None,
    plane_sample_count=0,
    plane_inlier_count=0,
):
    nan3 = (math.nan, math.nan, math.nan)
    nan9 = tuple(math.nan for _ in range(9))
    ratio = (
        float(plane_inlier_count) / float(plane_sample_count)
        if plane_sample_count
        else 0.0
    )
    return SurfaceGeometryEstimate(
        geometry_valid=False,
        surface_point_m=nan3,
        surface_point_covariance_m2=nan9,
        panel_outward_normal=nan3,
        normal_angular_uncertainty_rad=math.nan,
        plane_rms_residual_m=math.nan,
        plane_inlier_ratio=ratio,
        plane_sample_count=int(plane_sample_count),
        plane_inlier_count=int(plane_inlier_count),
        plane_span_m=math.nan,
        depth_mad_m=(
            float(depth_sample.depth_mad_m)
            if depth_sample is not None
            else math.nan
        ),
        depth_sigma_m=(
            float(depth_sample.depth_sigma_m)
            if depth_sample is not None
            else math.nan
        ),
        depth_source=(
            str(depth_sample.source) if depth_sample is not None else ''
        ),
        failure_reason=str(reason),
    )


def _fit_plane_svd(points):
    """Return centroid, unit normal, and second-axis RMS span."""

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 3:
        raise ValueError('plane fit requires at least three 3-D points')
    if not np.all(np.isfinite(array)):
        raise ValueError('plane points must be finite')
    centroid = np.mean(array, axis=0)
    centered = array - centroid
    _u, singular_values, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    if singular_values.size < 3 or not np.all(np.isfinite(singular_values)):
        raise ValueError('plane SVD returned invalid singular values')
    normal = np.asarray(vh[-1], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError('plane normal is degenerate')
    normal /= norm
    span = float(singular_values[1] / math.sqrt(array.shape[0]))
    if not math.isfinite(span):
        raise ValueError('plane span is invalid')
    return centroid, normal, span


def _surface_point_covariance(
    u,
    v,
    depth_m,
    fx,
    fy,
    cx,
    cy,
    pixel_sigma_px,
    depth_sigma_m,
):
    """Propagate independent pixel/depth uncertainty through pinhole XYZ."""

    jacobian = np.array(
        [
            [
                float(depth_m) / float(fx),
                0.0,
                (float(u) - float(cx)) / float(fx),
            ],
            [
                0.0,
                float(depth_m) / float(fy),
                (float(v) - float(cy)) / float(fy),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    input_covariance = np.diag(
        [
            float(pixel_sigma_px) ** 2,
            float(pixel_sigma_px) ** 2,
            float(depth_sigma_m) ** 2,
        ]
    )
    covariance = jacobian @ input_covariance @ jacobian.T
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError('point covariance is non-finite')
    covariance = (
        eigenvectors
        @ np.diag(np.maximum(eigenvalues, 0.0))
        @ eigenvectors.T
    )
    return (covariance + covariance.T) * 0.5


def estimate_surface_geometry_from_bbox(
    depth_image,
    bbox,
    center_pixel,
    encoding,
    fx,
    fy,
    cx,
    cy,
    min_depth_m=0.15,
    max_depth_m=5.0,
    minimum_surface_valid_pixels=9,
    center_roi_ratio=0.30,
    fallback_window_size=9,
    annulus_expansion_ratio=1.8,
    annulus_inner_ratio=1.0,
    maximum_plane_depth_delta_m=0.08,
    minimum_plane_points=50,
    maximum_plane_points=4000,
    minimum_plane_inlier_ratio=0.65,
    maximum_plane_rms_residual_m=0.008,
    minimum_plane_span_m=0.005,
    mad_rejection_multiplier=3.5,
    pixel_sigma_px=1.0,
    depth_noise_floor_m=0.001,
    maximum_normal_uncertainty_rad=math.radians(10.0),
):
    """Estimate a button surface point and surrounding panel outward normal.

    The contact point remains the robust median depth at the button centre.
    The panel plane is fitted independently from an expanded bbox annulus that
    excludes the button centre.  The result is diagnostic geometry only; this
    helper does not assert depth-scale, registration, or robot-frame
    calibration provenance.
    """

    if not 0.0 < float(min_depth_m) < float(max_depth_m):
        raise ValueError(
            'depth range must satisfy 0 < min_depth_m < max_depth_m'
        )
    if int(minimum_surface_valid_pixels) <= 0:
        raise ValueError('minimum_surface_valid_pixels must be positive')
    if not 0.0 < float(center_roi_ratio) <= 1.0:
        raise ValueError('center_roi_ratio must be in (0, 1]')
    if int(fallback_window_size) <= 0 or int(fallback_window_size) % 2 == 0:
        raise ValueError('fallback_window_size must be a positive odd integer')
    if float(annulus_expansion_ratio) <= 1.0:
        raise ValueError('annulus_expansion_ratio must be greater than 1')
    if not 0.0 < float(annulus_inner_ratio) <= 1.0:
        raise ValueError('annulus_inner_ratio must be in (0, 1]')
    if float(maximum_plane_depth_delta_m) <= 0.0:
        raise ValueError('maximum_plane_depth_delta_m must be positive')
    if int(minimum_plane_points) < 3:
        raise ValueError('minimum_plane_points must be at least 3')
    if int(maximum_plane_points) < int(minimum_plane_points):
        raise ValueError(
            'maximum_plane_points must be at least minimum_plane_points'
        )
    if not 0.0 < float(minimum_plane_inlier_ratio) <= 1.0:
        raise ValueError('minimum_plane_inlier_ratio must be in (0, 1]')
    positive_parameters = (
        maximum_plane_rms_residual_m,
        minimum_plane_span_m,
        mad_rejection_multiplier,
        depth_noise_floor_m,
        maximum_normal_uncertainty_rad,
    )
    if not all(float(value) > 0.0 for value in positive_parameters):
        raise ValueError('plane and uncertainty thresholds must be positive')
    if float(pixel_sigma_px) < 0.0:
        raise ValueError('pixel_sigma_px must be non-negative')

    try:
        numeric_bbox = tuple(float(value) for value in bbox)
        numeric_center = tuple(float(value) for value in center_pixel)
    except (TypeError, ValueError):
        return _invalid_surface_estimate('bbox and center must be numeric')
    projection_inputs = numeric_center + (
        float(fx),
        float(fy),
        float(cx),
        float(cy),
    )
    if (
        len(numeric_bbox) != 4
        or len(numeric_center) != 2
        or not all(math.isfinite(value) for value in numeric_bbox)
        or not all(math.isfinite(value) for value in projection_inputs)
        or numeric_bbox[2] <= numeric_bbox[0]
        or numeric_bbox[3] <= numeric_bbox[1]
        or float(fx) <= 0.0
        or float(fy) <= 0.0
    ):
        return _invalid_surface_estimate(
            'bbox, center, and intrinsics must be finite and non-degenerate'
        )

    try:
        depth_m = depth_to_meters(depth_image, encoding)
        surface_sample = robust_depth_from_bbox(
            depth_image,
            numeric_bbox,
            numeric_center,
            encoding,
            min_depth_m=float(min_depth_m),
            max_depth_m=float(max_depth_m),
            minimum_valid_pixels=int(minimum_surface_valid_pixels),
            center_roi_ratio=float(center_roi_ratio),
            fallback_window_size=int(fallback_window_size),
        )
    except (TypeError, ValueError):
        return _invalid_surface_estimate('depth image or encoding is invalid')
    if not surface_sample.depth_valid:
        return _invalid_surface_estimate(
            'button surface depth is invalid',
            surface_sample,
        )

    try:
        surface_point = np.asarray(
            project_pixel_to_camera(
                numeric_center[0],
                numeric_center[1],
                surface_sample.depth_m,
                fx,
                fy,
                cx,
                cy,
            ),
            dtype=np.float64,
        )
    except ValueError:
        return _invalid_surface_estimate(
            'button surface projection failed',
            surface_sample,
        )

    height, width = depth_m.shape
    bbox_center_x = (numeric_bbox[0] + numeric_bbox[2]) * 0.5
    bbox_center_y = (numeric_bbox[1] + numeric_bbox[3]) * 0.5
    bbox_width = numeric_bbox[2] - numeric_bbox[0]
    bbox_height = numeric_bbox[3] - numeric_bbox[1]
    expanded_half_width = bbox_width * float(annulus_expansion_ratio) * 0.5
    expanded_half_height = bbox_height * float(annulus_expansion_ratio) * 0.5
    left = max(0, int(math.floor(bbox_center_x - expanded_half_width)))
    right = min(width, int(math.ceil(bbox_center_x + expanded_half_width)))
    top = max(0, int(math.floor(bbox_center_y - expanded_half_height)))
    bottom = min(height, int(math.ceil(bbox_center_y + expanded_half_height)))
    if right <= left or bottom <= top:
        return _invalid_surface_estimate(
            'expanded plane annulus is empty',
            surface_sample,
        )

    rows, columns = np.mgrid[top:bottom, left:right]
    inner_half_width = bbox_width * float(annulus_inner_ratio) * 0.5
    inner_half_height = bbox_height * float(annulus_inner_ratio) * 0.5
    inside_button = (
        (columns >= bbox_center_x - inner_half_width)
        & (columns <= bbox_center_x + inner_half_width)
        & (rows >= bbox_center_y - inner_half_height)
        & (rows <= bbox_center_y + inner_half_height)
    )
    depths = depth_m[top:bottom, left:right]
    usable = (
        ~inside_button
        & np.isfinite(depths)
        & (depths >= float(min_depth_m))
        & (depths <= float(max_depth_m))
        & (
            np.abs(depths - float(surface_sample.depth_m))
            <= float(maximum_plane_depth_delta_m)
        )
    )
    plane_depths = depths[usable]
    plane_rows = rows[usable].astype(np.float64)
    plane_columns = columns[usable].astype(np.float64)
    if plane_depths.size < int(minimum_plane_points):
        return _invalid_surface_estimate(
            'panel annulus has too few valid depth pixels',
            surface_sample,
            plane_sample_count=int(plane_depths.size),
        )

    plane_points = np.column_stack(
        (
            (plane_columns - float(cx)) * plane_depths / float(fx),
            (plane_rows - float(cy)) * plane_depths / float(fy),
            plane_depths,
        )
    )
    if not np.all(np.isfinite(plane_points)):
        return _invalid_surface_estimate(
            'panel annulus produced non-finite 3-D points',
            surface_sample,
            plane_sample_count=int(plane_points.shape[0]),
        )
    if plane_points.shape[0] > int(maximum_plane_points):
        indices = np.linspace(
            0,
            plane_points.shape[0] - 1,
            int(maximum_plane_points),
            dtype=np.int64,
        )
        plane_points = plane_points[indices]
    plane_sample_count = int(plane_points.shape[0])

    inlier_mask = np.ones(plane_sample_count, dtype=bool)
    try:
        for _iteration in range(2):
            fit_points = plane_points[inlier_mask]
            if fit_points.shape[0] < int(minimum_plane_points):
                return _invalid_surface_estimate(
                    'robust plane rejection left too few points',
                    surface_sample,
                    plane_sample_count,
                    int(fit_points.shape[0]),
                )
            centroid, normal, _span = _fit_plane_svd(fit_points)
            signed_distances = (plane_points - centroid) @ normal
            distance_median = float(np.median(signed_distances))
            distance_mad = float(
                np.median(np.abs(signed_distances - distance_median))
            )
            robust_sigma = 1.4826 * distance_mad
            rejection_threshold = max(
                float(mad_rejection_multiplier) * robust_sigma,
                float(depth_noise_floor_m),
            )
            updated_mask = (
                np.abs(signed_distances - distance_median)
                <= rejection_threshold
            )
            if np.array_equal(updated_mask, inlier_mask):
                break
            inlier_mask = updated_mask

        inlier_points = plane_points[inlier_mask]
        centroid, normal, plane_span_m = _fit_plane_svd(inlier_points)
    except (ValueError, np.linalg.LinAlgError):
        return _invalid_surface_estimate(
            'panel plane fit is degenerate',
            surface_sample,
            plane_sample_count,
            int(np.count_nonzero(inlier_mask)),
        )

    plane_inlier_count = int(inlier_points.shape[0])
    plane_inlier_ratio = plane_inlier_count / float(plane_sample_count)
    if plane_inlier_ratio < float(minimum_plane_inlier_ratio):
        return _invalid_surface_estimate(
            'panel plane inlier ratio is too low',
            surface_sample,
            plane_sample_count,
            plane_inlier_count,
        )
    if plane_span_m < float(minimum_plane_span_m):
        return _invalid_surface_estimate(
            'panel plane physical span is too small',
            surface_sample,
            plane_sample_count,
            plane_inlier_count,
        )

    signed_inlier_distances = (inlier_points - centroid) @ normal
    plane_rms_residual_m = float(
        math.sqrt(np.mean(np.square(signed_inlier_distances)))
    )
    if (
        not math.isfinite(plane_rms_residual_m)
        or plane_rms_residual_m > float(maximum_plane_rms_residual_m)
    ):
        return _invalid_surface_estimate(
            'panel plane RMS residual is too large',
            surface_sample,
            plane_sample_count,
            plane_inlier_count,
        )

    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= 0.0:
        return _invalid_surface_estimate(
            'panel normal is invalid',
            surface_sample,
            plane_sample_count,
            plane_inlier_count,
        )
    normal = normal / normal_norm
    if float(np.dot(normal, surface_point)) > 0.0:
        normal = -normal
    if not float(np.dot(normal, surface_point)) < -1e-9:
        return _invalid_surface_estimate(
            'panel normal cannot be oriented toward the camera',
            surface_sample,
            plane_sample_count,
            plane_inlier_count,
        )

    normal_angular_uncertainty_rad = float(
        math.atan2(
            max(plane_rms_residual_m, float(depth_noise_floor_m)),
            plane_span_m,
        )
    )
    if (
        not math.isfinite(normal_angular_uncertainty_rad)
        or normal_angular_uncertainty_rad
        > float(maximum_normal_uncertainty_rad)
    ):
        return _invalid_surface_estimate(
            'panel normal angular uncertainty is too large',
            surface_sample,
            plane_sample_count,
            plane_inlier_count,
        )

    depth_sigma_m = max(
        (
            float(surface_sample.depth_sigma_m)
            if math.isfinite(float(surface_sample.depth_sigma_m))
            else 0.0
        ),
        float(depth_noise_floor_m),
    )
    try:
        point_covariance = _surface_point_covariance(
            numeric_center[0],
            numeric_center[1],
            surface_sample.depth_m,
            fx,
            fy,
            cx,
            cy,
            pixel_sigma_px,
            depth_sigma_m,
        )
    except (ValueError, np.linalg.LinAlgError):
        return _invalid_surface_estimate(
            'surface point covariance is invalid',
            surface_sample,
            plane_sample_count,
            plane_inlier_count,
        )

    return SurfaceGeometryEstimate(
        geometry_valid=True,
        surface_point_m=tuple(float(value) for value in surface_point),
        surface_point_covariance_m2=tuple(
            float(value) for value in point_covariance.reshape(-1)
        ),
        panel_outward_normal=tuple(float(value) for value in normal),
        normal_angular_uncertainty_rad=normal_angular_uncertainty_rad,
        plane_rms_residual_m=plane_rms_residual_m,
        plane_inlier_ratio=plane_inlier_ratio,
        plane_sample_count=plane_sample_count,
        plane_inlier_count=plane_inlier_count,
        plane_span_m=plane_span_m,
        depth_mad_m=float(surface_sample.depth_mad_m),
        depth_sigma_m=depth_sigma_m,
        depth_source=str(surface_sample.source),
        failure_reason='',
    )


def scale_pixel_between_images(
    u,
    v,
    source_width,
    source_height,
    target_width,
    target_height,
):
    """Map coordinates across FrameGate's endpoint-preserving resize."""

    dimensions = [source_width, source_height, target_width, target_height]
    if any(int(value) <= 0 for value in dimensions):
        raise ValueError('image dimensions must be positive')
    if not math.isfinite(float(u)) or not math.isfinite(float(v)):
        raise ValueError('pixel coordinates must be finite')
    source_width = int(source_width)
    source_height = int(source_height)
    target_width = int(target_width)
    target_height = int(target_height)
    scale_x = (
        0.0
        if source_width == 1
        else (target_width - 1) / float(source_width - 1)
    )
    scale_y = (
        0.0
        if source_height == 1
        else (target_height - 1) / float(source_height - 1)
    )
    return float(u) * scale_x, float(v) * scale_y
