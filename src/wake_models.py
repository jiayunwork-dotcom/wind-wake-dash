"""
尾流模型核心算法
包含: Jensen(Park)模型, Bastankhah高斯尾流模型
多尾流叠加: 线性叠加 / 均方根叠加
"""

import numpy as np
from typing import Tuple


def jensen_wake_deficit(upstream_radius: float,
                        downstream_distance: float,
                        ct: float,
                        alpha: float = 0.075) -> float:
    """
    Jensen (Park) 模型 - 单台风机在下游点的风速亏损率
    deltaV / V0 = (1 - sqrt(1 - Ct)) / (1 + alpha * x / r0) ^ 2

    Parameters
    ----------
    upstream_radius : 上游风机叶轮半径 (m)
    downstream_distance : 下游距离 (m), x > 0
    ct : 推力系数
    alpha : 尾流扩展系数, 陆上0.075, 海上0.04

    Returns
    -------
    风速亏损率 (无量纲, 0~1)
    """
    if downstream_distance <= 0:
        return 0.0
    wake_radius = upstream_radius + alpha * downstream_distance
    numerator = 1.0 - np.sqrt(1.0 - np.clip(ct, 0.0, 0.999))
    denominator = (1.0 + alpha * downstream_distance / upstream_radius) ** 2
    return float(numerator / denominator)


def jensen_deficit_at_point(upstream_pos: np.ndarray,
                            target_pos: np.ndarray,
                            upstream_radius: float,
                            ct: float,
                            alpha: float = 0.075,
                            wind_direction_from: float = 270.0) -> float:
    """
    Jensen模型: 上游风机在目标位置产生的风速亏损
    风从 wind_direction_from 度方向吹来 (气象风向: 北=0度顺时针)

    Returns
    -------
    deltaV/V0 (若目标不在尾流区内返回0)
    """
    deficit = jensen_wake_deficit(upstream_radius, 0, ct, alpha)
    x_rel, y_rel, in_wake, x_downstream, wake_width = \
        _wake_geometry(upstream_pos, target_pos, upstream_radius, alpha, wind_direction_from,
                       wake_type="jensen")
    if not in_wake:
        return 0.0

    base_deficit = jensen_wake_deficit(upstream_radius, x_downstream, ct, alpha)
    return float(base_deficit)


def bastankhah_deficit_at_point(upstream_pos: np.ndarray,
                                target_pos: np.ndarray,
                                upstream_radius: float,
                                ct: float,
                                turbulence_intensity: float = 0.1,
                                alpha: float = 0.075,
                                wind_direction_from: float = 270.0,
                                k_star: float = 0.3837) -> float:
    """
    Bastankhah高斯尾流模型
    尾流截面速度亏损呈高斯分布:
    deltaU/U0 = (1 - sqrt(1 - Ct/(8*(sigma/D)^2))) * exp(-r^2/(2*sigma^2))
    sigma = k_star * x + epsilon * D/2  (线性尾流扩展)

    Parameters
    ----------
    turbulence_intensity : 参考湍流强度 TI
    k_star : 尾流扩展斜率 (Bastankhah原论文值)
    """
    x_rel, y_rel, in_wake, x_downstream, wake_width = \
        _wake_geometry(upstream_pos, target_pos, upstream_radius, alpha, wind_direction_from,
                       wake_type="bastankhah")
    if not in_wake or x_downstream <= 0:
        return 0.0

    D = 2.0 * upstream_radius
    epsilon = 0.25 * np.sqrt((0.5 * (1.0 + np.sqrt(1.0 - ct))) / np.sqrt(1.0 - ct))
    sigma = k_star * x_downstream + epsilon * D / 2.0

    radial_dist = abs(y_rel)
    peak_deficit = 1.0 - np.sqrt(1.0 - np.clip(ct, 0.0, 0.999) / (8.0 * (sigma / D) ** 2 + 1e-12))
    gaussian_weight = np.exp(- (radial_dist ** 2) / (2.0 * sigma ** 2 + 1e-12))

    return float(peak_deficit * gaussian_weight)


def _wake_geometry(upstream_pos: np.ndarray,
                   target_pos: np.ndarray,
                   upstream_radius: float,
                   alpha: float,
                   wind_direction_from: float,
                   wake_type: str = "jensen",
                   max_downstream: float = 10000.0) -> Tuple[float, float, bool, float, float]:
    """
    计算上游风机尾流在目标点的几何关系
    wind_direction_from: 气象风向 (北=0, 顺时针), 风来的方向

    Returns
    -------
    x_rel, y_rel, in_wake, x_downstream, wake_width_at_target
    """
    blow_to_meteo = (wind_direction_from + 180.0) % 360.0
    theta_math = np.radians(90.0 - blow_to_meteo)

    dx = target_pos[0] - upstream_pos[0]
    dy = target_pos[1] - upstream_pos[1]

    cos_t = np.cos(theta_math)
    sin_t = np.sin(theta_math)

    x_downstream = dx * cos_t + dy * sin_t
    y_cross = -dx * sin_t + dy * cos_t

    if x_downstream <= 0:
        return 0.0, 0.0, False, 0.0, 0.0

    if x_downstream > max_downstream:
        return x_downstream, y_cross, False, x_downstream, 0.0

    if wake_type == "jensen":
        wake_width = upstream_radius + alpha * x_downstream
    else:
        wake_width = (upstream_radius + alpha * x_downstream) * 2.0

    in_wake = abs(y_cross) <= wake_width
    return x_downstream, y_cross, in_wake, x_downstream, wake_width


def _meteorological_to_math(meteorological_deg: float) -> float:
    """
    气象风向角度 -> 数学坐标系角度 (弧度)
    气象风向: 北=0度, 顺时针旋转, 表示风从该方向吹来
    数学角度: X轴正方向=0度, 逆时针旋转
    转换: math_angle = (90 - meteorological_deg) * pi / 180
    """
    return np.radians(90.0 - meteorological_deg)


def rotate_coordinates(coords: np.ndarray, wind_direction_from: float) -> np.ndarray:
    """
    将坐标系旋转, 使风吹向的方向与 +X 轴对齐
    风从 wind_direction_from 方向吹来, 下游风机将有更大的 x 坐标

    coords: (N, 2) 原始坐标
    wind_direction_from: 气象风向, 风来的方向 (北0度, 顺时针)
    """
    blow_to_meteo = (wind_direction_from + 180.0) % 360.0
    blow_to_math = np.radians(90.0 - blow_to_meteo)
    theta = blow_to_math

    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    rot_matrix = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])
    rotated = coords @ rot_matrix.T
    return rotated


def combined_deficit_matrix(rotated_coords: np.ndarray,
                            rotor_radii: np.ndarray,
                            cts: np.ndarray,
                            ambient_wind_speed: float,
                            model: str = "jensen",
                            alpha: float = 0.075,
                            ti: float = 0.1,
                            superposition: str = "linear") -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化计算所有风机之间的尾流影响矩阵和各风机有效入流风速

    Parameters
    ----------
    rotated_coords : (N, 2) 旋转后的坐标 (风向对齐 +X)
    rotor_radii : (N,) 各风机叶轮半径
    cts : (N,) 各风机在来流风速下的推力系数
    model : 'jensen' 或 'bastankhah'
    superposition : 'linear' (线性叠加) 或 'rms' (均方根叠加)

    Returns
    -------
    deficit_matrix : (N, N) deficit_matrix[i,j] = 风机i受风机j影响的风速亏损率(deltaV/V0)
                     deficit_matrix[i,i] = 0
    effective_wind_speeds : (N,) 各风机有效入流风速
    """
    N = rotated_coords.shape[0]
    deficit_matrix = np.zeros((N, N), dtype=np.float64)

    xs = rotated_coords[:, 0]
    ys = rotated_coords[:, 1]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]

            if dx <= 0:
                continue

            x_downstream = dx
            if x_downstream > 10000:
                continue

            y_cross = abs(dy)

            rj = rotor_radii[j]
            ctj = cts[j]

            if model == "jensen":
                wake_width_at_i = rj + alpha * x_downstream
                if y_cross <= wake_width_at_i:
                    base_def = (1.0 - np.sqrt(1.0 - np.clip(ctj, 0.0, 0.999))) / \
                               ((1.0 + alpha * x_downstream / rj) ** 2)
                    deficit_matrix[i, j] = base_def
            else:
                D = 2.0 * rj
                k_star = 0.3837
                epsilon = 0.25 * np.sqrt((0.5 * (1.0 + np.sqrt(1.0 - ctj))) /
                                         (np.sqrt(1.0 - ctj) + 1e-12))
                sigma = k_star * x_downstream + epsilon * D / 2.0
                wake_width_at_i = 3.0 * sigma
                if y_cross <= wake_width_at_i:
                    peak = 1.0 - np.sqrt(1.0 - np.clip(ctj, 0.0, 0.999) /
                                         (8.0 * (sigma / D) ** 2 + 1e-12))
                    gauss = np.exp(-(y_cross ** 2) / (2.0 * sigma ** 2 + 1e-12))
                    deficit_matrix[i, j] = peak * gauss

    if superposition == "linear":
        total_deficits = np.sum(deficit_matrix, axis=1)
        total_deficits = np.clip(total_deficits, 0.0, 0.95)
        effective_wind_speeds = ambient_wind_speed * (1.0 - total_deficits)
    elif superposition == "rms":
        total_deficits = np.sqrt(np.sum(deficit_matrix ** 2, axis=1))
        total_deficits = np.clip(total_deficits, 0.0, 0.95)
        effective_wind_speeds = ambient_wind_speed * (1.0 - total_deficits)
    else:
        one_minus_d = 1.0 - np.clip(deficit_matrix, 0.0, 0.95)
        v_ratio = np.prod(one_minus_d, axis=1)
        effective_wind_speeds = ambient_wind_speed * v_ratio
        total_deficits = 1.0 - v_ratio

    return deficit_matrix, effective_wind_speeds


def calculate_pairwise_influences(coords: np.ndarray,
                                  rotor_diameters: np.ndarray,
                                  wind_direction_from: float) -> np.ndarray:
    """
    计算每对风机的上下游关系和尾流几何信息
    Returns (N, N, 3) array: [dx_downstream, dy_cross, is_downstream]
    """
    N = coords.shape[0]
    rotated = rotate_coordinates(coords, wind_direction_from)
    result = np.zeros((N, N, 3))

    for i in range(N):
        for j in range(N):
            dx = rotated[i, 0] - rotated[j, 0]
            dy = rotated[i, 1] - rotated[j, 1]
            result[i, j, 0] = dx
            result[i, j, 1] = dy
            result[i, j, 2] = 1.0 if dx > 0 else 0.0

    return result


def get_wake_cone_polygon(upstream_pos: np.ndarray,
                          wind_direction_from: float,
                          rotor_radius: float,
                          alpha: float,
                          length: float = 3000.0) -> np.ndarray:
    """
    获取Jensen模型尾流锥形多边形的顶点坐标 (用于可视化)
    返回 (M, 2) 坐标点
    """
    theta = _meteorological_to_math(wind_direction_from)
    dir_x = np.cos(theta)
    dir_y = np.sin(theta)
    perp_x = -dir_y
    perp_y = dir_x

    start_r = rotor_radius
    end_r = rotor_radius + alpha * length

    p1 = upstream_pos + perp_x * start_r + perp_y * 0
    p2 = upstream_pos + dir_x * length + perp_x * end_r
    p3 = upstream_pos + dir_x * length - perp_x * end_r
    p4 = upstream_pos - perp_x * start_r

    cone_rot = np.array([
        [p1[0] + perp_x * 0 + perp_y * start_r, p1[1] - perp_x * start_r + perp_y * 0],
        [upstream_pos[0] + dir_x * length + perp_x * end_r,
         upstream_pos[1] + dir_y * length + perp_y * end_r],
        [upstream_pos[0] + dir_x * length - perp_x * end_r,
         upstream_pos[1] + dir_y * length - perp_y * end_r],
        [upstream_pos[0] - perp_y * start_r, upstream_pos[1] + perp_x * start_r],
    ])

    pts = np.zeros((4, 2))
    pts[0] = upstream_pos + np.array([-perp_y, perp_x]) * start_r
    pts[1] = upstream_pos + np.array([dir_x, dir_y]) * length + np.array([-perp_y, perp_x]) * end_r
    pts[2] = upstream_pos + np.array([dir_x, dir_y]) * length - np.array([-perp_y, perp_x]) * end_r
    pts[3] = upstream_pos - np.array([-perp_y, perp_x]) * start_r

    return pts
