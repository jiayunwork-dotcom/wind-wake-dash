"""
风电场尾流效应分析与发电功率预测系统
核心模块包
"""

from .turbine_models import (
    get_builtin_turbines,
    parse_custom_turbine_csv,
    parse_farm_layout_csv,
    interpolate_power,
    interpolate_thrust,
)

from .wake_models import (
    jensen_wake_deficit,
    jensen_deficit_at_point,
    bastankhah_deficit_at_point,
    rotate_coordinates,
    combined_deficit_matrix,
)

from .wind_resource import (
    generate_frequency_from_weibull,
    parse_frequency_csv,
    parse_weibull_per_sector_csv,
    expand_weibull_to_frequency,
    weibull_pdf,
    weibull_cdf,
    sample_rose_from_frequency,
    mean_wind_speed_from_frequency,
)

from .power_calculation import (
    compute_power_for_direction_windspeed,
    scan_all_directions_windspeeds,
    scan_direction_rose,
    check_spacing_constraint,
    compute_turbine_params_array,
)

from .layout_optimization import (
    genetic_algorithm_layout_optimize,
    evaluate_aep,
)

from .visualization import (
    plot_farm_layout,
    plot_power_rose,
    plot_wake_loss_bar,
    plot_aep_direction_pie,
    plot_optimization_convergence,
    plot_layout_comparison,
    plot_loss_matrix_heatmap,
    plot_power_curve,
)

__all__ = [
    "get_builtin_turbines",
    "parse_custom_turbine_csv",
    "parse_farm_layout_csv",
    "interpolate_power",
    "interpolate_thrust",
    "jensen_wake_deficit",
    "jensen_deficit_at_point",
    "bastankhah_deficit_at_point",
    "rotate_coordinates",
    "combined_deficit_matrix",
    "generate_frequency_from_weibull",
    "parse_frequency_csv",
    "parse_weibull_per_sector_csv",
    "expand_weibull_to_frequency",
    "weibull_pdf",
    "weibull_cdf",
    "sample_rose_from_frequency",
    "mean_wind_speed_from_frequency",
    "compute_power_for_direction_windspeed",
    "scan_all_directions_windspeeds",
    "scan_direction_rose",
    "check_spacing_constraint",
    "compute_turbine_params_array",
    "genetic_algorithm_layout_optimize",
    "evaluate_aep",
    "plot_farm_layout",
    "plot_power_rose",
    "plot_wake_loss_bar",
    "plot_aep_direction_pie",
    "plot_optimization_convergence",
    "plot_layout_comparison",
    "plot_loss_matrix_heatmap",
    "plot_power_curve",
]
