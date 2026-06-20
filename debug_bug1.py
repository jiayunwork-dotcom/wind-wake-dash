"""
详细诊断Bug1: 尾流损失率过高
"""
import sys
sys.path.insert(0, '.')

from src import (
    get_builtin_turbines, parse_farm_layout_csv,
    interpolate_power, interpolate_thrust,
)
from src.wake_models import rotate_coordinates, combined_deficit_matrix
from src.power_calculation import get_effective_cts, get_turbine_powers
import os
import numpy as np

tl = get_builtin_turbines()
data_path = os.path.join('.', 'data', 'sample_farm_layout.csv')
with open(data_path, 'r', encoding='utf-8') as f:
    content = f.read()
coords, tids, mnames = parse_farm_layout_csv(content, tl)

print("=" * 70)
print("诊断 270°风向, 10m/s:")
print("=" * 70)

rotated = rotate_coordinates(coords, 270.0)
print("\n旋转后坐标 (风吹向+X方向):")
for i in range(12):
    print(f"  {tids[i]}: x={rotated[i,0]:.1f}, y={rotated[i,1]:.1f}")

from src.power_calculation import compute_turbine_params_array
params = compute_turbine_params_array(tl, mnames)
rotor_radii = params["rotor_radii"]

N = 12
ambient_ws = 10.0
effective_ws = np.full(N, ambient_ws)

for it in range(3):
    print(f"\n--- 迭代 {it+1} ---")
    cts = get_effective_cts(tl, mnames, effective_ws)
    print(f"Ct值: {cts}")
    
    deficit_matrix, effective_ws_new = combined_deficit_matrix(
        rotated, rotor_radii, cts, ambient_ws, "jensen", 0.075, 0.1, "linear"
    )
    
    print(f"\n尾流亏损矩阵 deficit_matrix (行i受列j影响, ΔV/V0):")
    for i in range(N):
        row_str = "  "
        for j in range(N):
            if deficit_matrix[i,j] > 0.01:
                row_str += f"{deficit_matrix[i,j]*100:5.1f} "
            else:
                row_str += "   .  "
        total = np.sum(deficit_matrix[i,:])
        row_str += f"| 合计: {total*100:5.1f}% → V_eff = {ambient_ws*(1-total):.2f}"
        print(row_str)
    
    print(f"\n有效入流风速:")
    powers = get_turbine_powers(tl, mnames, effective_ws_new)
    no_wake_p = get_turbine_powers(tl, mnames, np.full(N, ambient_ws))
    for i in range(N):
        loss_pct = (no_wake_p[i] - powers[i]) / no_wake_p[i] * 100 if no_wake_p[i] > 0 else 0
        print(f"  {tids[i]}: V_eff={effective_ws_new[i]:.2f} m/s, P={powers[i]/1000:.3f} MW, 无尾流P={no_wake_p[i]/1000:.3f} MW, 损失={loss_pct:.1f}%")
    
    effective_ws = effective_ws_new.copy()

total_p = np.sum(powers) / 1000
total_nw = np.sum(no_wake_p) / 1000
overall_loss = (total_nw - total_p) / total_nw * 100

print(f"\n全场总功率: {total_p:.2f} MW, 无尾流: {total_nw:.2f} MW, 尾流损失: {overall_loss:.1f}%")

print("\n" + "=" * 70)
print("问题分析:")
print("=" * 70)
print("1. 多尾流线性叠加太激进，如T04同时受T01、T02、T03三台影响，")
print("   亏损直接相加可达70%+，导致V_eff降至3m/s以下，功率几乎为0")
print()
print("2. 物理上更合理的叠加方式: 动能亏损叠加 = (1-ΣdU/U)的乘积形式")
print("   V_eff² / V0² = Π (1 - dU_j/V0)²")
print("   即 V_eff = V0 * Π (1 - dU_j/V0)")
