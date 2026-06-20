"""
风资源数据处理模块
支持: 风向风速联合频率表导入, Weibull分布参数输入
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats


def weibull_pdf(v: np.ndarray, k: float, A: float) -> np.ndarray:
    """
    Weibull概率密度函数
    f(v) = (k/A) * (v/A)^(k-1) * exp(-(v/A)^k)
    """
    v = np.asarray(v, dtype=float)
    mask = v > 0
    result = np.zeros_like(v)
    vv = v[mask]
    result[mask] = (k / A) * (vv / A) ** (k - 1) * np.exp(-(vv / A) ** k)
    return result


def weibull_cdf(v: np.ndarray, k: float, A: float) -> np.ndarray:
    """
    Weibull累积分布函数
    F(v) = 1 - exp(-(v/A)^k)
    """
    v = np.asarray(v, dtype=float)
    return 1.0 - np.exp(-(v / A) ** k)


def generate_frequency_from_weibull(direction_sectors: int = 12,
                                    wind_speed_bins: Optional[np.ndarray] = None,
                                    k: float = 2.2,
                                    A: float = 8.5,
                                    dir_weights: Optional[np.ndarray] = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    基于Weibull参数生成风向风速联合频率表

    Parameters
    ----------
    direction_sectors : 风向扇区数 (12 或 36)
    wind_speed_bins : 风速分箱边界 (m/s), 若None则默认0~30, 步长1
    k : Weibull形状因子
    A : Weibull尺度因子 (m/s)
    dir_weights : 各扇区相对权重, 若None则均匀分布但略偏好NNE-SSW

    Returns
    -------
    direction_centers : (direction_sectors,) 扇区中心角度
    wind_speed_centers : (N_wind_bins,) 风速分箱中心
    freq_matrix : (direction_sectors, N_wind_bins) 联合频率 (总和≈1)
    """
    if wind_speed_bins is None:
        wind_speed_bins = np.arange(0, 31, 1.0)
    if len(wind_speed_bins) < 2:
        wind_speed_bins = np.arange(0, 31, 1.0)

    ws_edges = np.array(wind_speed_bins)
    ws_centers = 0.5 * (ws_edges[:-1] + ws_edges[1:]) if len(ws_edges) > 1 else ws_edges

    sector_width = 360.0 / direction_sectors
    dir_centers = np.arange(direction_sectors) * sector_width

    if dir_weights is None:
        preferred_directions = [0, 30, 210, 225, 240, 330, 345]
        dir_weights = np.ones(direction_sectors)
        for deg in preferred_directions:
            idx = int(((deg + sector_width / 2) % 360) / sector_width)
            if 0 <= idx < direction_sectors:
                dir_weights[idx] *= 1.6
        dir_weights = dir_weights / dir_weights.sum()

    dir_weights = np.asarray(dir_weights, dtype=float)
    dir_weights = dir_weights / dir_weights.sum()

    prob_ws = weibull_cdf(ws_edges[1:], k, A) - weibull_cdf(ws_edges[:-1], k, A)
    prob_ws = prob_ws / prob_ws.sum()

    freq_matrix = dir_weights[:, None] * prob_ws[None, :]
    freq_matrix = freq_matrix / freq_matrix.sum()

    return dir_centers, ws_centers, freq_matrix


def parse_frequency_csv(file_content: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解析用户上传的风向风速联合频率表CSV

    支持两种格式:
    格式A: 矩阵形式 - 第一行风速分箱, 第一列风向扇区中心, 值为频率%
    格式B: 长表形式 - 三列: 风向, 风速, 频率

    Returns
    -------
    direction_centers, wind_speed_centers, freq_matrix (总和归一化到1)
    """
    lines = file_content.strip().split('\n')
    header = [h.strip() for h in lines[0].split(',')]

    is_long_format = False
    for h in header:
        hl = h.lower()
        if '频率' in hl or 'freq' in hl or 'prob' in hl:
            is_long_format = True
            break

    if is_long_format:
        return _parse_long_format(lines, header)
    else:
        return _parse_matrix_format(lines, header)


def _parse_long_format(lines, header) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dir_idx = ws_idx = freq_idx = -1
    for i, h in enumerate(header):
        hl = h.lower()
        if 'dir' in hl or '风向' in hl:
            dir_idx = i
        elif 'speed' in hl or '风速' in hl or 'wind' in hl:
            ws_idx = i
        elif 'freq' in hl or '频率' in hl or 'prob' in hl:
            freq_idx = i

    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[dir_idx]), float(parts[ws_idx]), float(parts[freq_idx])])
        except (ValueError, IndexError):
            continue

    arr = np.array(rows)
    if arr.size == 0:
        raise ValueError("无法解析风资源CSV长表")

    directions = np.unique(arr[:, 0])
    windspeeds = np.unique(arr[:, 1])

    directions.sort()
    windspeeds.sort()

    freq = np.zeros((len(directions), len(windspeeds)))
    d_map = {d: i for i, d in enumerate(directions)}
    w_map = {w: i for i, w in enumerate(windspeeds)}

    for r in arr:
        freq[d_map[r[0]], w_map[r[1]]] = r[2]

    total = freq.sum()
    if total > 1.5:
        freq = freq / 100.0
    freq = freq / freq.sum()

    return directions, windspeeds, freq


def _parse_matrix_format(lines, header) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ws_headers = []
    for h in header[1:]:
        try:
            ws_headers.append(float(h))
        except ValueError:
            ws_headers.append(0.0)

    directions = []
    freq_rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        try:
            directions.append(float(parts[0]))
            row = []
            for p in parts[1:len(ws_headers) + 1]:
                row.append(float(p))
            while len(row) < len(ws_headers):
                row.append(0.0)
            freq_rows.append(row)
        except (ValueError, IndexError):
            continue

    dirs = np.array(directions)
    wss = np.array(ws_headers)
    freq = np.array(freq_rows, dtype=float)

    total = freq.sum()
    if total > 1.5:
        freq = freq / 100.0
    freq = freq / freq.sum()

    return dirs, wss, freq


def parse_weibull_per_sector_csv(file_content: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解析各扇区Weibull参数CSV
    列: 风向中心, 形状因子k, 尺度因子A, 扇区频率权重(可选)
    """
    lines = file_content.strip().split('\n')
    header = [h.strip().lower() for h in lines[0].split(',')]

    dir_idx = k_idx = A_idx = w_idx = -1
    for i, h in enumerate(header):
        if 'dir' in h or '风向' in h:
            dir_idx = i
        elif 'k' == h or 'shape' in h or '形状' in h:
            k_idx = i
        elif 'a' == h or 'scale' in h or '尺度' in h or 'c' == h:
            A_idx = i
        elif 'weight' in h or 'freq' in h or '权重' in h or '频率' in h:
            w_idx = i

    directions = []
    k_list = []
    A_list = []
    weights = []

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        try:
            directions.append(float(parts[dir_idx]))
            k_list.append(float(parts[k_idx]))
            A_list.append(float(parts[A_idx]))
            weights.append(float(parts[w_idx]) if w_idx >= 0 else 1.0)
        except (ValueError, IndexError):
            continue

    directions = np.array(directions)
    ks = np.array(k_list)
    As = np.array(A_list)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    return directions, ks, As


def expand_weibull_to_frequency(directions: np.ndarray,
                                k_per_sector: np.ndarray,
                                A_per_sector: np.ndarray,
                                sector_weights: Optional[np.ndarray] = None,
                                wind_speed_max: float = 30.0,
                                ws_step: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将各扇区Weibull参数展开为风向风速联合频率矩阵
    """
    if sector_weights is None:
        sector_weights = np.ones_like(directions) / len(directions)

    ws_edges = np.arange(0, wind_speed_max + ws_step, ws_step)
    ws_centers = 0.5 * (ws_edges[:-1] + ws_edges[1:])

    N_dir = len(directions)
    N_ws = len(ws_centers)

    freq = np.zeros((N_dir, N_ws))
    for i in range(N_dir):
        cdf = weibull_cdf(ws_edges, k_per_sector[i], A_per_sector[i])
        prob = cdf[1:] - cdf[:-1]
        prob = prob / prob.sum()
        freq[i, :] = sector_weights[i] * prob

    freq = freq / freq.sum()

    return directions, ws_centers, freq


def sample_rose_from_frequency(directions: np.ndarray,
                               ws_centers: np.ndarray,
                               freq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从频率矩阵计算风向频率玫瑰图(按风向积分所有风速)
    Returns
    -------
    directions, direction_frequencies (总和1)
    """
    dir_freq = freq.sum(axis=1)
    return directions, dir_freq


def mean_wind_speed_from_frequency(ws_centers: np.ndarray,
                                   freq_matrix: np.ndarray) -> float:
    """
    计算全场平均风速
    """
    dir_freq = freq_matrix.sum(axis=1, keepdims=True)
    ws_prob = (freq_matrix / (dir_freq + 1e-15)).sum(axis=0)
    ws_prob = ws_prob / ws_prob.sum()
    return float(np.sum(ws_centers * ws_prob))
