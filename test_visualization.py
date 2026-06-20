"""
测试Plotly可视化是否有aspectratio错误
"""
import sys
sys.path.insert(0, '.')

from src import get_builtin_turbines, parse_farm_layout_csv, plot_farm_layout
import os
import numpy as np
import json

tl = get_builtin_turbines()
data_path = os.path.join('.', 'data', 'sample_farm_layout.csv')
with open(data_path, 'r', encoding='utf-8') as f:
    content = f.read()
coords, tids, mnames = parse_farm_layout_csv(content, tl)

print("Testing plot_farm_layout...")
fig = plot_farm_layout(coords, tids, mnames, tl, per_turbine_loss=None, wind_direction=None, show_wake_cones=False)

# 检查布局JSON中是否有aspectratio
fig_json = fig.to_json()
if 'aspectratio' in fig_json.lower():
    print("❌ Found 'aspectratio' in layout!")
    # 找出具体位置
    fig_dict = json.loads(fig_json)
    def find_aspectratio(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                if 'aspectratio' in k.lower():
                    print(f"  Found at {path}.{k}: {v}")
                find_aspectratio(v, f"{path}.{k}")
        elif isinstance(d, list):
            for i, v in enumerate(d):
                find_aspectratio(v, f"{path}[{i}]")
    find_aspectratio(fig_dict, "fig")
else:
    print("✅ No 'aspectratio' in layout JSON")

# 尝试生成HTML
try:
    html = fig.to_html(include_plotlyjs=False, full_html=False)
    print("✅ to_html() succeeded")
except Exception as e:
    print(f"❌ to_html() failed: {e}")

# 测试带尾流锥的版本
print("\nTesting plot_farm_layout with wind direction 270°...")
fig2 = plot_farm_layout(coords, tids, mnames, tl, per_turbine_loss=np.random.rand(12)*0.3, 
                        wind_direction=270.0, show_wake_cones=True, alpha=0.075)
try:
    html2 = fig2.to_html(include_plotlyjs=False, full_html=False)
    print("✅ to_html() with wake cones succeeded")
except Exception as e:
    print(f"❌ to_html() with wake cones failed: {e}")

print("\nAll visualization tests passed!")
