#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:

    h, w = frame_sequence.frame_shape[:2]

    # Corner detection parameters:
    blocks_approx_count = 1500
    block_size = round((w * h / blocks_approx_count) ** 0.5)
    corner_min_rel_quality = 0.05

    # Flow tracking parameters:
    window_size = (21, 21)
    pyramid_max_level = 4 - 1

    def mask_exclude_neighbors(positions, sizes):
        result = np.full((h, w), 255, dtype=np.uint8)
        for (y, x), s in zip(positions, sizes):
            cv2.circle(result, (int(np.rint(y)), int(np.rint(x))), int(np.rint(s)), color=0, thickness=-1)
        return result

    prev_pyramid = None
    corner_ids = np.array([], dtype=int).reshape((0, 1))
    corners = np.array([], dtype=np.float32).reshape((0, 2))
    corner_sizes = np.array([], dtype=np.float32).reshape((0, 1))
    next_id = 1

    for frame_idx, frame_img in enumerate(frame_sequence):
        img8 = np.rint(255 * frame_img).astype(np.uint8)
        last_level, pyramid = cv2.buildOpticalFlowPyramid(
            img8,
            winSize=window_size,
            maxLevel=pyramid_max_level,
            withDerivatives=False
        )
        if corners.size != 0:
            next_corners, status, err = cv2.calcOpticalFlowPyrLK(
                prevImg=prev_pyramid[0],
                nextImg=pyramid[0],
                prevPts=corners,
                winSize=window_size,
                nextPts=None,
                minEigThreshold=1e-6
            )
            err_flat = err.reshape(-1).astype(np.float64)
            err_ok = err_flat[status.flat == 1]
            err_q = np.quantile(err_ok, 0.9)
            keep = np.logical_and(status.flat == 1,
                                  err_flat < 2.0 * err_q)
            corner_ids = corner_ids[keep]
            corners = next_corners[keep]
            corner_sizes = corner_sizes[keep]

        mask = mask_exclude_neighbors(corners, corner_sizes)
        for level, level_img in enumerate(pyramid):

            new_corners = cv2.goodFeaturesToTrack(
                level_img,
                maxCorners=0,
                qualityLevel=corner_min_rel_quality,
                minDistance=block_size * 0.7,
                blockSize=block_size,
                mask=mask,
                gradientSize=3
            )
            if new_corners is not None:
                new_corners = new_corners.reshape((-1, 2)) * 2 ** level

                count = new_corners.shape[0]
                corner_ids = np.concatenate((corner_ids, np.arange(next_id, next_id + count).reshape((-1, 1))))
                next_id += count
                corners = np.concatenate((corners, new_corners))
                corner_sizes = np.concatenate((corner_sizes,
                                               np.full(count, block_size * 1.5 ** level).reshape((-1, 1))))

            mask = mask_exclude_neighbors(corners, corner_sizes)
            for _ in range(level + 1):
                mask = cv2.pyrDown(mask).astype(np.uint8)
            mask[mask <= 100] = 0

        prev_pyramid = pyramid

        builder.set_corners_at_frame(frame_idx, FrameCorners(corner_ids, corners, corner_sizes))


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
