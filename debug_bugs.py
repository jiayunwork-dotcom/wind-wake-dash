"""
Bug诊断脚本 - 验证风向转换和可视化bug
"""
import numpy as np

def _meteorological_to_math(meteorological_deg: float) -> float:
    return np.radians(90.0 - meteorological_deg)

print("=== 风向转换测试 ===")
for deg in [0, 90, 180, 270]:
    theta = _meteorological_to_math(deg)
    dir_x = np.cos(theta)
    dir_y = np.sin(theta)
    print(f"气象风向 {deg}° (风从{deg}°来 → 吹向{(deg+180)%360}°")
    print(f"  数学角度(弧度): {theta:.3f}")
    print(f"  dir_x, dir_y = ({dir_x:.3f}, {dir_y:.3f}")
    print(f"  尾流锥延伸方向向量 = ({dir_x:.2f}, {dir_y:.2f})")
    print()

print("=== 分析 ===")
print("期望：风向270°(西风) → 风吹向东(+X方向)")
print("但当前dir_x=-1, dir_y=0 → 尾流锥朝-X方向（西），这是错的！")
print("这就是尾流锥出现在风机的上游而不是下游！")
print()

from src import get_wake_cone_polygon
from src.wake_models import rotate_coordinates, _meteorological_to_math

coords = np.array([[0, 0], [600, 0], [1200, 0]])
print("原始坐标:")
print(coords)
print()

print("旋转后坐标 (风向270°):")
rotated = rotate_coordinates(coords, 270)
print(rotated)
print("下游风机X坐标更大，说明下游是+X方向（东），正确！")
print("尾流锥应该朝+X方向延伸，但当前dir_x=-1，所以画出来反了！")
