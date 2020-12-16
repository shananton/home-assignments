#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    TriangulationParameters,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    compute_reprojection_errors
)
from _corners import (
    FrameCorners,
    without_short_tracks
)
from collections import namedtuple

# KnownView = namedtuple('KnownView', ('frame_no', 'pose'))

CORNER_MIN_TRACK_LENGTH = 30

TRIANGULATION_MAX_REPROJECTION_ERROR = 5.0
TRIANGULATION_MIN_ANGLE_DEG = 0.1
TRIANGULATION_MIN_DEPTH = 0.0

RANSAC_INLIERS_MAX_REPROJECTION_ERROR = 1.0
RANSAC_TARGET_CONFIDENCE = 0.999

TRIANGULATION_PARAMS = TriangulationParameters(
    TRIANGULATION_MAX_REPROJECTION_ERROR,
    TRIANGULATION_MIN_ANGLE_DEG,
    TRIANGULATION_MIN_DEPTH
)


class CameraTracker:

    def __init__(self,
                 camera_parameters: CameraParameters,
                 corner_storage: CornerStorage,
                 frame_sequence_path: str,
                 known_view_1: Optional[Tuple[int, Pose]],
                 known_view_2: Optional[Tuple[int, Pose]]
                 ):
        self.rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
        self.frame_count = len(self.rgb_sequence)
        self.camera_parameters = camera_parameters
        self.corner_storage = without_short_tracks(corner_storage, CORNER_MIN_TRACK_LENGTH)
        self.intrinsic_mat = to_opencv_camera_mat3x3(camera_parameters, self.rgb_sequence[0].shape[0])
        self.point_cloud_builder = PointCloudBuilder()
        self.known_view_mats = {}
        self.unknown_view_ids = set(range(self.frame_count))
        for (frame_idx, pose) in (known_view_1, known_view_2):
            self.update_view(frame_idx, pose_to_view_mat3x4(pose))

    def update_view(self, frame_idx: int, view_mat: np.ndarray):
        self.known_view_mats[frame_idx] = view_mat
        self.unknown_view_ids.remove(frame_idx)

    def track_camera(self):
        last_idx, last_view = next(iter(self.known_view_mats.items()))
        while self.unknown_view_ids:
            for (idx, view) in self.known_view_mats.items():
                if idx != last_idx:
                    self.enrich_point_cloud(last_view, view, self.corner_storage[last_idx], self.corner_storage[idx],
                                            TRIANGULATION_PARAMS)
            # TODO: multi-threading here
            restored_views = map(lambda i: (i, self.restore_view(i)), self.unknown_view_ids)
            restored_views = list(filter(lambda val: val[1][0] is not None, restored_views))
            if len(restored_views) == 0:
                raise RuntimeError('No more views can be restored!')
            new_view_tuple = min(restored_views, key=lambda val: val[1][2])
            (new_idx, (new_view, inliers_count, _)) = new_view_tuple
            self.update_view(new_idx, new_view)
            print('Restored position at frame {} with {} inliers'.format(new_idx, inliers_count))

    def enrich_point_cloud(self, view_1: np.ndarray, view_2: np.ndarray, corners_1: FrameCorners,
                           corners_2: FrameCorners,
                           triangulation_parameters: TriangulationParameters) -> None:
        # TODO: exclude outliers here?
        correspondences = build_correspondences(corners_1, corners_2)
        # TODO: use median_cos somehow?
        points, ids, median_cos = triangulate_correspondences(
            correspondences,
            view_1,
            view_2,
            self.intrinsic_mat,
            triangulation_parameters)
        self.point_cloud_builder.add_points(ids, points)

    def restore_view(self, frame_idx: int) -> Tuple[Optional[np.ndarray], int, np.float64]:
        corners = self.corner_storage[frame_idx]
        _, (corner_ids, cloud_ids) = snp.intersect(corners.ids.flatten(),
                                                   np.sort(self.point_cloud_builder.ids.flatten()), indices=True)
        points_3d = self.point_cloud_builder.points[cloud_ids]
        points_2d = corners.points[corner_ids]
        result = self.solve_pnp(points_3d, points_2d)
        if result is None:
            return None, 0, np.inf
        view, inlier_ids = result
        inlier_ids = inlier_ids.flatten()
        reprojection_errors = compute_reprojection_errors(points_3d[inlier_ids], points_2d[inlier_ids],
                                                          self.intrinsic_mat @ view)
        return view, len(inlier_ids), reprojection_errors.mean()

    def solve_pnp(self, points_3d: np.ndarray, points_2d: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        ok, r_vec, t_vec, inliers = cv2.solvePnPRansac(
            objectPoints=points_3d,
            imagePoints=points_2d,
            cameraMatrix=self.intrinsic_mat,
            distCoeffs=None,
            confidence=RANSAC_TARGET_CONFIDENCE,
            reprojectionError=RANSAC_INLIERS_MAX_REPROJECTION_ERROR,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
        return rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec), inliers


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    camera_tracker = CameraTracker(camera_parameters, corner_storage, frame_sequence_path, known_view_1, known_view_2)
    camera_tracker.track_camera()

    view_mats = [camera_tracker.known_view_mats[i] for i in range(camera_tracker.frame_count)]

    calc_point_cloud_colors(
        camera_tracker.point_cloud_builder,
        camera_tracker.rgb_sequence,
        view_mats,
        camera_tracker.intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = camera_tracker.point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
