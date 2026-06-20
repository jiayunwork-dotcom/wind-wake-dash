"""
风电场机型参数模块
包含内置常见机型参数和自定义机型上传处理
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def build_power_curve(cut_in: float, rated: float, cut_out: float,
                      rated_power: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建简化功率曲线 (风速-功率对照表)
    使用三次曲线拟合切入到额定风速段
    """
    wind_speeds = np.arange(0, 31, 1.0)
    powers = np.zeros_like(wind_speeds)

    for i, v in enumerate(wind_speeds):
        if v < cut_in or v > cut_out:
            powers[i] = 0.0
        elif cut_in <= v < rated:
            ratio = (v - cut_in) / (rated - cut_in)
            powers[i] = rated_power * (ratio ** 3)
        else:
            powers[i] = rated_power

    return wind_speeds, powers


def build_thrust_curve(cut_in: float, rated: float, cut_out: float,
                       ct_rated: float = 0.8, ct_cutin: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建简化推力系数曲线
    额定风速以上推力系数逐渐下降
    """
    wind_speeds = np.arange(0, 31, 1.0)
    cts = np.zeros_like(wind_speeds)

    for i, v in enumerate(wind_speeds):
        if v < cut_in or v > cut_out:
            cts[i] = 0.0
        elif cut_in <= v <= rated:
            ratio = (v - cut_in) / (rated - cut_in)
            cts[i] = ct_cutin - (ct_cutin - ct_rated) * (ratio ** 2)
        else:
            ratio = (v - rated) / (cut_out - rated)
            cts[i] = ct_rated * (1 - 0.7 * ratio)

    return wind_speeds, cts


def get_builtin_turbines() -> Dict[str, Dict]:
    """
    返回内置的常见机型参数库
    包含2MW、3MW、5MW、8MW级别各一款
    """
    turbines = {}

    ws2, p2 = build_power_curve(3.0, 12.0, 25.0, 2000.0)
    _, ct2 = build_thrust_curve(3.0, 12.0, 25.0)
    turbines["DTU_2MW"] = {
        "rated_power": 2.0,
        "rotor_diameter": 90.0,
        "hub_height": 80.0,
        "cut_in_speed": 3.0,
        "rated_speed": 12.0,
        "cut_out_speed": 25.0,
        "power_curve_ws": ws2,
        "power_curve_kw": p2,
        "thrust_curve_ws": ws2,
        "thrust_curve_ct": ct2,
    }

    ws3, p3 = build_power_curve(3.0, 13.0, 25.0, 3000.0)
    _, ct3 = build_thrust_curve(3.0, 13.0, 25.0)
    turbines["Vestas_V112_3MW"] = {
        "rated_power": 3.0,
        "rotor_diameter": 112.0,
        "hub_height": 94.0,
        "cut_in_speed": 3.0,
        "rated_speed": 13.0,
        "cut_out_speed": 25.0,
        "power_curve_ws": ws3,
        "power_curve_kw": p3,
        "thrust_curve_ws": ws3,
        "thrust_curve_ct": ct3,
    }

    ws5, p5 = build_power_curve(3.0, 11.5, 25.0, 5000.0)
    _, ct5 = build_thrust_curve(3.0, 11.5, 25.0)
    turbines["SG_5MW_132"] = {
        "rated_power": 5.0,
        "rotor_diameter": 132.0,
        "hub_height": 110.0,
        "cut_in_speed": 3.0,
        "rated_speed": 11.5,
        "cut_out_speed": 25.0,
        "power_curve_ws": ws5,
        "power_curve_kw": p5,
        "thrust_curve_ws": ws5,
        "thrust_curve_ct": ct5,
    }

    ws8, p8 = build_power_curve(3.5, 13.0, 25.0, 8000.0)
    _, ct8 = build_thrust_curve(3.5, 13.0, 25.0)
    turbines["Haliade_X_8MW"] = {
        "rated_power": 8.0,
        "rotor_diameter": 167.0,
        "hub_height": 120.0,
        "cut_in_speed": 3.5,
        "rated_speed": 13.0,
        "cut_out_speed": 25.0,
        "power_curve_ws": ws8,
        "power_curve_kw": p8,
        "thrust_curve_ws": ws8,
        "thrust_curve_ct": ct8,
    }

    return turbines


def parse_custom_turbine_csv(file_content: str) -> Dict:
    """
    解析用户上传的自定义机型CSV
    格式: 第一行元数据, 后续风速、功率(kW)、Ct
    """
    lines = file_content.strip().split('\n')

    meta = {}
    data_start = 0
    for i, line in enumerate(lines[:10]):
        if '=' in line:
            key, val = line.split('=', 1)
            meta[key.strip()] = val.strip()
        elif line.startswith('wind_speed') or ',' in line and 'wind' in line.lower():
            data_start = i
            break
        else:
            data_start = i
            break

    data_lines = lines[data_start:]
    header = data_lines[0].split(',')
    rows = []
    for line in data_lines[1:]:
        if line.strip():
            rows.append([float(x) for x in line.split(',')])

    data = np.array(rows)
    ws = data[:, 0]
    power_kw = data[:, 1] if data.shape[1] > 1 else np.zeros_like(ws)
    ct = data[:, 2] if data.shape[1] > 2 else np.zeros_like(ws)

    rated_power_kw = float(meta.get('rated_power_kw', np.max(power_kw)))
    if rated_power_kw == 0:
        rated_power_kw = np.max(power_kw)

    return {
        "rated_power": rated_power_kw / 1000.0,
        "rotor_diameter": float(meta.get('rotor_diameter', 120.0)),
        "hub_height": float(meta.get('hub_height', 100.0)),
        "cut_in_speed": float(meta.get('cut_in_speed', 3.0)),
        "rated_speed": float(meta.get('rated_speed', 12.0)),
        "cut_out_speed": float(meta.get('cut_out_speed', 25.0)),
        "power_curve_ws": ws,
        "power_curve_kw": power_kw,
        "thrust_curve_ws": ws,
        "thrust_curve_ct": ct,
    }


def interpolate_power(turbine_params: Dict, wind_speed: float) -> float:
    """
    在功率曲线上插值得到单机功率(kW)
    超出切入切出范围返回0
    """
    ws = turbine_params["power_curve_ws"]
    pw = turbine_params["power_curve_kw"]

    if wind_speed <= turbine_params["cut_in_speed"] or wind_speed >= turbine_params["cut_out_speed"]:
        return 0.0

    return float(np.interp(wind_speed, ws, pw))


def interpolate_thrust(turbine_params: Dict, wind_speed: float) -> float:
    """
    在推力系数曲线上插值
    """
    ws = turbine_params["thrust_curve_ws"]
    ct = turbine_params["thrust_curve_ct"]

    if wind_speed <= turbine_params["cut_in_speed"] or wind_speed >= turbine_params["cut_out_speed"]:
        return 0.0

    return float(np.interp(wind_speed, ws, ct))


def parse_farm_layout_csv(file_content: str,
                          turbine_library: Dict[str, Dict]) -> Tuple[np.ndarray, list, list]:
    """
    解析风电场布局CSV
    字段: 风机编号, X坐标(米), Y坐标(米), 风机型号
    返回: 坐标矩阵(N,2), 型号列表, 编号列表
    """
    lines = file_content.strip().split('\n')
    header = [h.strip().lower() for h in lines[0].split(',')]

    idx_id = idx_x = idx_y = idx_model = -1
    for i, h in enumerate(header):
        if '编号' in h or 'id' in h or 'name' in h or 'turbine' in h:
            idx_id = i
        elif 'x' == h or 'x坐标' in h or 'easting' in h:
            idx_x = i
        elif 'y' == h or 'y坐标' in h or 'northing' in h:
            idx_y = i
        elif '型号' in h or 'model' in h or 'type' in h:
            idx_model = i

    coords = []
    turbine_ids = []
    model_names = []

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue

        tid = parts[idx_id] if idx_id >= 0 else f"T{len(turbine_ids)+1}"
        try:
            x = float(parts[idx_x])
            y = float(parts[idx_y])
        except (ValueError, IndexError):
            continue
        model = parts[idx_model] if idx_model >= 0 and idx_model < len(parts) else list(turbine_library.keys())[0]

        if model not in turbine_library:
            model = list(turbine_library.keys())[0]

        coords.append([x, y])
        turbine_ids.append(tid)
        model_names.append(model)

    return np.array(coords), turbine_ids, model_names
