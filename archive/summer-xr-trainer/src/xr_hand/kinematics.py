"""Forward kinematics: convert per-joint local transforms to world positions.

Real-glove streams (and skeletal animation in general) express each joint's
pose as a translation + rotation relative to its parent joint. To draw the
hand we chain those transforms through the bone hierarchy.

Convention: each joint's `(x, y, z)` is a translation in its parent's local
frame, and the rotation quaternion is XYZW (qx, qy, qz, qw).
"""
import numpy as np

from .joints import BONES, HandFrame

# child_idx -> parent_idx
PARENT = {child: parent for parent, child in BONES}


def quat_to_mat3(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert an XYZW quaternion to a 3x3 rotation matrix.

    Handles non-unit quaternions by normalising the magnitude internally.
    """
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xs, ys, zs = qx * s, qy * s, qz * s
    wx, wy, wz = qw * xs, qw * ys, qw * zs
    xx, xy, xz = qx * xs, qx * ys, qx * zs
    yy, yz, zz = qy * ys, qy * zs, qz * zs
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])


def forward_kinematics(frame: HandFrame) -> list:
    """Compute world positions for all 26 joints from a HandFrame.

    Returns [(x, y, z), ...] in JOINT_NAMES order.
    """
    n = len(frame.joints)
    world_pos = [None] * n
    world_rot = [None] * n

    def compute(idx: int) -> None:
        if world_pos[idx] is not None:
            return
        j = frame.joints[idx]
        local_rot = quat_to_mat3(j.qx, j.qy, j.qz, j.qw)
        local_pos = np.array([j.x, j.y, j.z])

        if idx not in PARENT:
            world_rot[idx] = local_rot
            world_pos[idx] = local_pos
        else:
            par = PARENT[idx]
            compute(par)
            world_rot[idx] = world_rot[par] @ local_rot
            world_pos[idx] = world_pos[par] + world_rot[par] @ local_pos

    for i in range(n):
        compute(i)

    return [(float(p[0]), float(p[1]), float(p[2])) for p in world_pos]
