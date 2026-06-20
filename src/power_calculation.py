"""
功率计算流程 + 尾流损失分析
核心: 坐标旋转、上下游判断、尾流叠加、功率插值、AEP积分
"""

import numpy as np
from typing import Dict, Tuple, List, Optional

from .turbine_models import interpolate_power, interpolate_thrust
from .wake_models import rotate_coordinates, combined_deficit_matrix


def compute_turbine_params_array(turbine_library: Dict[str, Dict],
                                 model_names: List[str]) -> Dict[str, np.ndarray]:
    """
    将各风机的参数提取为numpy数组, 便于向量化计算
    """
    N = len(model_names)
    rated_power_kw = np.zeros(N)
    rotor_diameters = np.zeros(N)
    rotor_radii = np.zeros(N)
    hub_heights = np.zeros(N)
    cut_in = np.zeros(N)
    rated_ws = np.zeros(N)
    cut_out = np.zeros(N)
    power_curves_ws = []
    power_curves_kw = []
    thrust_curves_ws = []
    thrust_curves_ct = []

    for i, m in enumerate(model_names):
        p = turbine_library[m]
        rated_power_kw[i] = p["rated_power"] * 1000.0
        rotor_diameters[i] = p["rotor_diameter"]
        rotor_radii[i] = p["rotor_diameter"] / 2.0
        hub_heights[i] = p["hub_height"]
        cut_in[i] = p["cut_in_speed"]
        rated_ws[i] = p["rated_speed"]
        cut_out[i] = p["cut_out_speed"]
        power_curves_ws.append(p["power_curve_ws"])
        power_curves_kw.append(p["power_curve_kw"])
        thrust_curves_ws.append(p["thrust_curve_ws"])
        thrust_curves_ct.append(p["thrust_curve_ct"])

    return {
        "rated_power_kw": rated_power_kw,
        "rotor_diameters": rotor_diameters,
        "rotor_radii": rotor_radii,
        "hub_heights": hub_heights,
        "cut_in": cut_in,
        "rated_ws": rated_ws,
        "cut_out": cut_out,
        "power_curves_ws": power_curves_ws,
        "power_curves_kw": power_curves_kw,
        "thrust_curves_ws": thrust_curves_ws,
        "thrust_curves_ct": thrust_curves_ct,
    }


def get_effective_cts(turbine_library: Dict[str, Dict],
                      model_names: List[str],
                      wind_speeds: np.ndarray) -> np.ndarray:
    """
    获取各风机在指定入流风速下的推力系数
    考虑尾流后的有效风速迭代(一阶近似: 直接用环境风速)
    """
    N = len(model_names)
    cts = np.zeros(N)
    for i, m in enumerate(model_names):
        cts[i] = interpolate_thrust(turbine_library[m], wind_speeds[i])
    return cts


def get_turbine_powers(turbine_library: Dict[str, Dict],
                       model_names: List[str],
                       effective_wind_speeds: np.ndarray) -> np.ndarray:
    """
    各风机根据有效入流风速查询功率曲线得到功率(kW)
    """
    N = len(model_names)
    powers = np.zeros(N)
    for i, m in enumerate(model_names):
        powers[i] = interpolate_power(turbine_library[m], effective_wind_speeds[i])
    return powers


def compute_power_for_direction_windspeed(coords: np.ndarray,
                                          turbine_library: Dict[str, Dict],
                                          model_names: List[str],
                                          wind_direction_from: float,
                                          ambient_wind_speed: float,
                                          wake_model: str = "jensen",
                                          alpha: float = 0.075,
                                          ti: float = 0.1,
                                          superposition: str = "linear",
                                          n_iterations: int = 3) -> Dict:
    """
    单风向单风速下的全场功率计算

    Parameters
    ----------
    n_iterations : 尾流迭代次数(推力系数随有效风速变化)
    """
    N = coords.shape[0]
    params = compute_turbine_params_array(turbine_library, model_names)
    rotor_radii = params["rotor_radii"]

    rotated = rotate_coordinates(coords, wind_direction_from)

    effective_ws = np.full(N, ambient_wind_speed)

    for it in range(n_iterations):
        cts = get_effective_cts(turbine_library, model_names, effective_ws)
        deficit_matrix, new_effective_ws = combined_deficit_matrix(
            rotated, rotor_radii, cts, ambient_wind_speed,
            wake_model, alpha, ti, superposition
        )
        if np.max(np.abs(new_effective_ws - effective_ws)) < 1e-3:
            break
        effective_ws = new_effective_ws

    cts = get_effective_cts(turbine_library, model_names, effective_ws)
    deficit_matrix, _ = combined_deficit_matrix(
        rotated, rotor_radii, cts, ambient_wind_speed,
        wake_model, alpha, ti, superposition
    )

    turbine_powers_kw = get_turbine_powers(turbine_library, model_names, effective_ws)

    no_wake_ws = np.full(N, ambient_wind_speed)
    no_wake_powers_kw = get_turbine_powers(turbine_library, model_names, no_wake_ws)

    total_power_kw = np.sum(turbine_powers_kw)
    total_no_wake_kw = np.sum(no_wake_powers_kw)

    per_turbine_loss = np.zeros(N)
    mask = no_wake_powers_kw > 0
    per_turbine_loss[mask] = (no_wake_powers_kw[mask] - turbine_powers_kw[mask]) / no_wake_powers_kw[mask]

    overall_loss = 0.0
    if total_no_wake_kw > 0:
        overall_loss = (total_no_wake_kw - total_power_kw) / total_no_wake_kw

    return {
        "direction": wind_direction_from,
        "ambient_wind_speed": ambient_wind_speed,
        "rotated_coords": rotated,
        "effective_wind_speeds": effective_ws,
        "deficit_matrix": deficit_matrix,
        "turbine_powers_kw": turbine_powers_kw,
        "no_wake_powers_kw": no_wake_powers_kw,
        "total_power_kw": total_power_kw,
        "total_no_wake_kw": total_no_wake_kw,
        "per_turbine_loss": per_turbine_loss,
        "overall_loss": overall_loss,
    }


def scan_all_directions_windspeeds(coords: np.ndarray,
                                   turbine_library: Dict[str, Dict],
                                   model_names: List[str],
                                   directions: np.ndarray,
                                   wind_speeds: np.ndarray,
                                   frequency_matrix: np.ndarray,
                                   wake_model: str = "jensen",
                                   alpha: float = 0.075,
                                   ti: float = 0.1,
                                   superposition: str = "linear") -> Dict:
    """
    扫描所有风向风速组合, 计算AEP

    Parameters
    ----------
    directions : (D,) 风向扇区中心 (度)
    wind_speeds : (W,) 风速分箱中心 (m/s)
    frequency_matrix : (D, W) 该风向风速组合的频率 (总和=1)

    Returns
    -------
    Dict, 包含:
        direction_power_kw : (D, W) 各工况全场功率 (kW)
        direction_no_wake_kw : (D, W) 无尾流全场功率 (kW)
        per_turbine_power : (N, D, W) 各风机功率
        per_turbine_loss_avg : (N,) 各风机按频率加权的平均尾流损失
        overall_loss : 全场整体尾流损失率
        aep_kwh : 年发电量 (kWh)
        aep_no_wake_kwh : 无尾流年发电量
        capacity_factor : 容量系数
        total_rated_kw : 总额定功率
    """
    N = coords.shape[0]
    D = len(directions)
    W = len(wind_speeds)
    params = compute_turbine_params_array(turbine_library, model_names)
    total_rated_kw = np.sum(params["rated_power_kw"])

    direction_power = np.zeros((D, W))
    direction_no_wake = np.zeros((D, W))
    per_turbine_power = np.zeros((N, D, W))
    per_turbine_loss = np.zeros((N, D, W))
    deficit_matrix_sum = np.zeros((N, N))

    for di, d in enumerate(directions):
        for wi, ws in enumerate(wind_speeds):
            freq = frequency_matrix[di, wi]
            if freq <= 1e-12 or ws <= 0.5:
                continue

            res = compute_power_for_direction_windspeed(
                coords, turbine_library, model_names,
                d, ws, wake_model, alpha, ti, superposition
            )
            direction_power[di, wi] = res["total_power_kw"]
            direction_no_wake[di, wi] = res["total_no_wake_kw"]
            per_turbine_power[:, di, wi] = res["turbine_powers_kw"]
            per_turbine_loss[:, di, wi] = res["per_turbine_loss"]
            deficit_matrix_sum += res["deficit_matrix"] * freq

    freq_flat = frequency_matrix.ravel()  # (D*W,)

    aep_kwh = 8760.0 * np.sum(direction_power.ravel() * freq_flat)
    aep_no_wake_kwh = 8760.0 * np.sum(direction_no_wake.ravel() * freq_flat)

    cap_factor = aep_kwh / (total_rated_kw * 8760.0) if total_rated_kw > 0 else 0.0

    weighted_loss = np.sum(per_turbine_loss.reshape(N, -1) * freq_flat[None, :], axis=1)
    total_weight = np.sum((per_turbine_loss > 0).reshape(N, -1) * freq_flat[None, :], axis=1)
    total_weight = np.where(total_weight > 0, total_weight, 1.0)
    per_turbine_loss_avg = weighted_loss / total_weight

    overall_loss = 0.0
    if aep_no_wake_kwh > 0:
        overall_loss = (aep_no_wake_kwh - aep_kwh) / aep_no_wake_kwh

    direction_mean_power = np.sum(direction_power * frequency_matrix, axis=1) / \
                           (np.sum(frequency_matrix, axis=1) + 1e-15)
    direction_mean_no_wake = np.sum(direction_no_wake * frequency_matrix, axis=1) / \
                             (np.sum(frequency_matrix, axis=1) + 1e-15)
    direction_loss = np.zeros_like(direction_mean_power)
    mask = direction_mean_no_wake > 0
    direction_loss[mask] = (direction_mean_no_wake[mask] - direction_mean_power[mask]) / direction_mean_no_wake[mask]

    direction_aep_contrib = 8760.0 * np.sum(direction_power * frequency_matrix, axis=1)

    most_affected_idx = np.argsort(-per_turbine_loss_avg)
    top5_affected = most_affected_idx[:min(5, N)]

    loss_matrix_normalized = deficit_matrix_sum

    return {
        "directions": directions,
        "wind_speeds": wind_speeds,
        "frequency_matrix": frequency_matrix,
        "direction_power_kw": direction_power,
        "direction_no_wake_kw": direction_no_wake,
        "per_turbine_power": per_turbine_power,
        "per_turbine_loss_avg": per_turbine_loss_avg,
        "overall_loss": overall_loss,
        "aep_kwh": aep_kwh,
        "aep_no_wake_kwh": aep_no_wake_kwh,
        "capacity_factor": cap_factor,
        "total_rated_kw": total_rated_kw,
        "direction_mean_power_kw": direction_mean_power,
        "direction_mean_no_wake_kw": direction_mean_no_wake,
        "direction_loss": direction_loss,
        "direction_aep_contrib_kwh": direction_aep_contrib,
        "top5_affected_indices": top5_affected,
        "loss_matrix": loss_matrix_normalized,
    }


def scan_direction_rose(coords: np.ndarray,
                        turbine_library: Dict[str, Dict],
                        model_names: List[str],
                        wind_speed: float,
                        wake_model: str = "jensen",
                        alpha: float = 0.075,
                        ti: float = 0.1,
                        superposition: str = "linear",
                        step_deg: float = 1.0) -> Dict:
    """
    逐度扫描风向(0~359度), 用于绘制风向-功率极坐标玫瑰图

    Returns
    -------
    angles_deg, total_power_kw, total_no_wake_kw, per_turbine_loss
    """
    angles = np.arange(0, 360, step_deg)
    P = len(angles)
    N = coords.shape[0]

    powers = np.zeros(P)
    no_wake_p = np.zeros(P)
    losses = np.zeros((N, P))

    for i, a in enumerate(angles):
        res = compute_power_for_direction_windspeed(
            coords, turbine_library, model_names,
            a, wind_speed, wake_model, alpha, ti, superposition
        )
        powers[i] = res["total_power_kw"]
        no_wake_p[i] = res["total_no_wake_kw"]
        losses[:, i] = res["per_turbine_loss"]

    best_idx = np.argmax(powers)
    worst_idx = np.argmax((no_wake_p - powers) / (no_wake_p + 1e-15))

    return {
        "angles_deg": angles,
        "total_power_kw": powers,
        "total_no_wake_kw": no_wake_p,
        "per_turbine_loss": losses,
        "best_direction": angles[best_idx],
        "best_power_kw": powers[best_idx],
        "worst_direction": angles[worst_idx],
        "worst_loss": (no_wake_p[worst_idx] - powers[worst_idx]) / (no_wake_p[worst_idx] + 1e-15),
    }


def sensitivity_scan_alpha_ti(coords: np.ndarray,
                              turbine_library: Dict[str, Dict],
                              model_names: List[str],
                              directions: np.ndarray,
                              wind_speeds: np.ndarray,
                              frequency_matrix: np.ndarray,
                              alpha_values: np.ndarray,
                              ti_values: np.ndarray,
                              wake_model: str = "jensen",
                              superposition: str = "linear",
                              progress_callback=None) -> Dict:
    """
    参数敏感性分析: 扫描alpha和TI的所有组合, 对每个组合计算AEP

    Parameters
    ----------
    alpha_values : (A,) 尾流扩展系数扫描序列
    ti_values : (T,) 湍流强度扫描序列
    progress_callback : 可选回调 fn(completed, total)

    Returns
    -------
    Dict:
        alpha_values, ti_values: 输入参数
        aep_matrix: (T, A) AEP矩阵 (GWh), 行=TI, 列=alpha
        best_alpha, best_ti, best_aep: AEP最高的组合
        worst_alpha, worst_ti, worst_aep: AEP最低的组合
    """
    A = len(alpha_values)
    T = len(ti_values)
    total = A * T
    aep_matrix = np.zeros((T, A))

    completed = 0
    for ai, a_val in enumerate(alpha_values):
        for ti_i, ti_val in enumerate(ti_values):
            result = scan_all_directions_windspeeds(
                coords, turbine_library, model_names,
                directions, wind_speeds, frequency_matrix,
                wake_model, a_val, ti_val, superposition,
            )
            aep_matrix[ti_i, ai] = result["aep_kwh"] / 1e6
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)

    best_flat = np.argmax(aep_matrix)
    best_ti_idx, best_alpha_idx = np.unravel_index(best_flat, aep_matrix.shape)
    worst_flat = np.argmin(aep_matrix)
    worst_ti_idx, worst_alpha_idx = np.unravel_index(worst_flat, aep_matrix.shape)

    return {
        "alpha_values": alpha_values,
        "ti_values": ti_values,
        "aep_matrix": aep_matrix,
        "best_alpha": alpha_values[best_alpha_idx],
        "best_ti": ti_values[best_ti_idx],
        "best_aep": aep_matrix[best_ti_idx, best_alpha_idx],
        "worst_alpha": alpha_values[worst_alpha_idx],
        "worst_ti": ti_values[worst_ti_idx],
        "worst_aep": aep_matrix[worst_ti_idx, worst_alpha_idx],
    }


def check_spacing_constraint(coords: np.ndarray,
                             rotor_diameters: np.ndarray,
                             min_spacing_multiple: float = 2.0) -> Tuple[bool, np.ndarray, float]:
    """
    检查风机间距约束 (不小于min_spacing_multiple倍叶轮直径)
    使用两两最大直径作为间距要求

    Returns
    -------
    all_ok: 全部满足
    violation_matrix: (N,N) 1表示不满足约束
    min_spacing_ratio: 全场最小的 实际间距/要求间距 比值
    """
    N = coords.shape[0]
    violations = np.zeros((N, N), dtype=int)
    min_ratio = np.inf

    for i in range(N):
        for j in range(i + 1, N):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            required = min_spacing_multiple * max(rotor_diameters[i], rotor_diameters[j])
            ratio = dist / required if required > 0 else np.inf
            if ratio < min_ratio:
                min_ratio = ratio
            if dist < required:
                violations[i, j] = 1
                violations[j, i] = 1

    all_ok = np.sum(violations) == 0
    return all_ok, violations, min_ratio
