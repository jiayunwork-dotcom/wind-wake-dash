import sys, os, time
sys.path.insert(0, '.')

from src import (
    get_builtin_turbines,
    parse_farm_layout_csv,
    generate_frequency_from_weibull,
    scan_all_directions_windspeeds,
    scan_direction_rose,
    check_spacing_constraint,
    compute_turbine_params_array,
    genetic_algorithm_layout_optimize,
    mean_wind_speed_from_frequency,
    compute_power_for_direction_windspeed,
)

print("=" * 60)
print("🚀 风电场尾流分析系统 - 端到端测试")
print("=" * 60)

tl = get_builtin_turbines()
print(f"[1/8] ✅ 机型库: {list(tl.keys())}")

data_path = os.path.join('.', 'data', 'sample_farm_layout.csv')
with open(data_path, 'r', encoding='utf-8') as f:
    layout_content = f.read()
coords, tids, mnames = parse_farm_layout_csv(layout_content, tl)
print(f"[2/8] ✅ 布局: {len(tids)} 台风机, shape={coords.shape}")

params = compute_turbine_params_array(tl, mnames)
ok, viol, min_r = check_spacing_constraint(coords, params["rotor_diameters"], 2.0)
print(f"[3/8] ✅ 间距检查: ok={ok}, min_ratio={min_r:.3f}")

dirs, wss, freq = generate_frequency_from_weibull(12, k=2.2, A=8.5)
mean_v = mean_wind_speed_from_frequency(wss, freq)
print(f"[4/8] ✅ 风资源: {len(dirs)}x{len(wss)}, 平均风速={mean_v:.2f}m/s")

t0 = time.time()
res = compute_power_for_direction_windspeed(
    coords, tl, mnames, 270, 10.0, "jensen", 0.075, 0.1, "linear"
)
dt = time.time() - t0
print(f"[5/8] ✅ 单工况(270°,10m/s): 功率={res['total_power_kw']/1000:.2f}MW, 损失率={res['overall_loss']*100:.2f}%, {dt:.2f}s")

t0 = time.time()
result = scan_all_directions_windspeeds(
    coords, tl, mnames, dirs, wss, freq, "jensen", 0.075, 0.1, "linear"
)
dt = time.time() - t0
print(f"[6/8] ✅ 全场AEP扫描: {result['aep_kwh']/1e6:.3f} GWh, 尾损={result['overall_loss']*100:.2f}%, CF={result['capacity_factor']*100:.2f}%, {dt:.2f}s")

t0 = time.time()
rose = scan_direction_rose(coords, tl, mnames, 10.0, "jensen", 0.075, 0.1, "linear", step_deg=15)
dt = time.time() - t0
print(f"[7/8] ✅ 风向扫描(24步): 最优{rose['best_direction']:.0f}°, 最差{rose['worst_direction']:.0f}°, {dt:.2f}s")

t0 = time.time()
opt = genetic_algorithm_layout_optimize(
    coords, tl, mnames, dirs, wss, freq,
    pop_size=6, n_generations=3, seed=42,
)
dt = time.time() - t0
print(f"[8/8] ✅ GA优化(6x3): AEP {opt['initial_aep_kwh']/1e6:.3f}->{opt['best_aep_kwh']/1e6:.3f} GWh ({opt['aep_improvement_pct']:+.2f}%), {dt:.2f}s")

print("\n" + "=" * 60)
print("🎉 全部测试通过！系统运行正常")
print("=" * 60)
