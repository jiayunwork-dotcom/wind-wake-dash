"""
测试不同叠加方式的对比测试
"""
import sys
sys.path.insert(0, '.')

from src import (
    get_builtin_turbines, parse_farm_layout_csv,
    generate_frequency_from_weibull,
    scan_all_directions_windspeeds,
    compute_power_for_direction_windspeed,
)
import os

tl = get_builtin_turbines()
data_path = os.path.join('.', 'data', 'sample_farm_layout.csv')
with open(data_path, 'r', encoding='utf-8') as f:
    content = f.read()
coords, tids, mnames = parse_farm_layout_csv(content, tl)

dirs, wss, freq = generate_frequency_from_weibull(12, k=2.2, A=8.5)

print("=" * 70)
print("不同尾流叠加方式对比")
print("=" * 70)

for sup_name, sup_code in [
    ("动能叠加(乘积)", "kinetic"),
    ("线性叠加", "linear"),
    ("均方根叠加", "rms"),
]:
    print(f"\n--- {sup_name} ---")
    
    # 单工况 270°, 10m/s
    r1 = compute_power_for_direction_windspeed(
        coords, tl, mnames, 270, 10.0, "jensen", 0.075, 0.1, sup_code,
    )
    print(f"  单工况尾损: {r1['overall_loss']*100:.2f}%")
    
    # 全场AEP
    r2 = scan_all_directions_windspeeds(
        coords, tl, mnames, dirs, wss, freq, "jensen", 0.075, 0.1, sup_code,
    )
    print(f"  全场AEP: {r2['aep_kwh']/1e6:.3f} GWh")
    print(f"  全场尾损: {r2['overall_loss']*100:.2f}%")
    print(f"  容量系数: {r2['capacity_factor']*100:.2f}%")

print("\n" + "=" * 70)
print("解释: 单工况正对风向时尾损高是正常的物理现象，")
print("       全场加权后在5-25%范围内才是考核标准")
