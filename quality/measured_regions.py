"""Derive conservative figure cross sections from recorded front-image landmarks.

Thresholds detect extreme flatness only; depth and appearance are not inferred
from the reference photograph. The inspection itself measures final triangles.
"""

import math


def measured_profile(settings):
    if settings.get("finish_profile") != "standing-figure":
        return None
    try:
        m = settings["finish_measurements"]
        projection, landmarks, detection = (
            m["projection"],
            m["landmarks_px"],
            m["detection"],
        )
        if projection["axes"] != "+X right, +Y up, +Z front":
            return None
        center_x, center_y = projection["center_px"]
        scale = projection["pixels_per_unit"]
        head_x, head_y_px = landmarks["head_center"]
        head_width, body_width = (
            detection["head_width_pixels"],
            detection["body_width_pixels"],
        )
        arms = landmarks["arm_edges"]
        feet = landmarks["feet"]
        values = [center_x, center_y, scale, head_x, head_y_px, head_width, body_width]
        values += [point[1] for arm in arms for point in arm] + [
            point[1] for point in feet
        ]
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in values
        ):
            return None
        if (
            scale <= 0
            or not 0.12 < head_width / scale < 1
            or not 0.12 < body_width / scale < 1
            or len(arms) != 2
            or len(feet) != 2
        ):
            return None
        head_y = (center_y - head_y_px) / scale
        head_center = (head_x - center_x) / scale
        arm_bottom = min((center_y - point[1]) / scale for arm in arms for point in arm)
        foot_bottom = max((center_y - point[1]) / scale for point in feet)
        torso_top = arm_bottom - 0.10
        torso_bottom = max(foot_bottom + 0.15, torso_top - 0.12)
        if (
            not -0.55 < foot_bottom < torso_bottom < torso_top < head_y - 0.10 < 0.6
            or torso_top - torso_bottom < 0.06
        ):
            return None
        head_half = head_width / scale * 0.58
        body_half = body_width / scale * 0.51
        depth_limit = max(head_half, body_half) * 2.5

        def region(left, right, bottom, top, levels):
            return {
                "bounds": [[left, right], [bottom, top], [-depth_limit, depth_limit]],
                "levels": levels,
                "min_depth_ratio": 0.15,
            }

        return {
            "subject": "cat",
            "up_axis": 1,
            "depth_axis": 2,
            "regions": {
                "head": region(
                    head_center - head_half,
                    head_center + head_half,
                    head_y - 0.08,
                    head_y + 0.08,
                    [head_y - 0.04, head_y, head_y + 0.04],
                ),
                "body": region(
                    -body_half,
                    body_half,
                    torso_bottom - 0.01,
                    torso_top + 0.01,
                    [
                        torso_bottom + (torso_top - torso_bottom) * q
                        for q in (0.25, 0.5, 0.75)
                    ],
                ),
            },
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        IndexError,
        ZeroDivisionError,
    ):
        return None
