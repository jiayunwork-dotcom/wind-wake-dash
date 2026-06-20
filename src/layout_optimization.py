"""
风电场布局优化模块
简化遗传算法, 最大化AEP为目标, 支持进度条和收敛曲线
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

from .power_calculation import (
    scan_all_directions_windspeeds,
    check_spacing_constraint,
    compute_turbine_params_array,
)


def evaluate_aep(coords: np.ndarray,
                 turbine_library: Dict[str, Dict],
                 model_names: List[str],
                 directions: np.ndarray,
                 wind_speeds: np.ndarray,
                 frequency_matrix: np.ndarray,
                 wake_model: str = "jensen",
                 alpha: float = 0.075,
                 ti: float = 0.1,
                 superposition: str = "linear",
                 min_spacing_multiple: float = 2.0) -> Tuple[float, float]:
    """
    评价函数: 返回 (AEP_kWh, 间距惩罚系数)
    间距不满足约束时惩罚为负
    """
    params = compute_turbine_params_array(turbine_library, model_names)
    rotor_diams = params["rotor_diameters"]

    all_ok, _, min_ratio = check_spacing_constraint(coords, rotor_diams, min_spacing_multiple)
    penalty = 0.0
    if min_ratio < 1.0:
        penalty = -1e12 * (1.0 - min_ratio)

    result = scan_all_directions_windspeeds(
        coords, turbine_library, model_names,
        directions, wind_speeds, frequency_matrix,
        wake_model, alpha, ti, superposition
    )
    aep = result["aep_kwh"]
    return aep + penalty, aep


def _random_feasible_individual(base_coords: np.ndarray,
                                movable_mask: np.ndarray,
                                bounds: Tuple[np.ndarray, np.ndarray],
                                rotor_diams: np.ndarray,
                                min_spacing_multiple: float = 2.0,
                                max_tries: int = 50) -> np.ndarray:
    """
    生成一个随机可行个体 (尽量满足间距约束)
    """
    for _ in range(max_tries):
        coords = base_coords.copy()
        for i in range(len(coords)):
            if movable_mask[i]:
                coords[i, 0] = np.random.uniform(bounds[0][i, 0], bounds[1][i, 0])
                coords[i, 1] = np.random.uniform(bounds[0][i, 1], bounds[1][i, 1])
        all_ok, _, _ = check_spacing_constraint(coords, rotor_diams, min_spacing_multiple)
        if all_ok:
            return coords
    return coords


def genetic_algorithm_layout_optimize(initial_coords: np.ndarray,
                                      turbine_library: Dict[str, Dict],
                                      model_names: List[str],
                                      directions: np.ndarray,
                                      wind_speeds: np.ndarray,
                                      frequency_matrix: np.ndarray,
                                      movable_mask: Optional[np.ndarray] = None,
                                      bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                      pop_size: int = 30,
                                      n_generations: int = 50,
                                      crossover_rate: float = 0.8,
                                      mutation_rate: float = 0.15,
                                      mutation_step_frac: float = 0.1,
                                      wake_model: str = "jensen",
                                      alpha: float = 0.075,
                                      ti: float = 0.1,
                                      superposition: str = "linear",
                                      min_spacing_multiple: float = 2.0,
                                      progress_callback: Optional[Callable[[int, Dict], None]] = None,
                                      seed: Optional[int] = 42) -> Dict:
    """
    简化遗传算法布局优化

    Parameters
    ----------
    movable_mask : (N,) bool 数组, True表示该风机可以移动
    bounds : (lb, ub) 各风机坐标边界, 形状均为 (N,2)
    progress_callback: 每次调用时传入 (当前代数, 统计信息字典)

    Returns
    -------
    Dict: best_coords, best_aep, initial_aep, history_best, history_avg
    """
    if seed is not None:
        np.random.seed(seed)

    N = initial_coords.shape[0]
    params = compute_turbine_params_array(turbine_library, model_names)
    rotor_diams = params["rotor_diameters"]

    if movable_mask is None:
        movable_mask = np.ones(N, dtype=bool)

    if bounds is None:
        span_x = initial_coords[:, 0].max() - initial_coords[:, 0].min() + 500
        span_y = initial_coords[:, 1].max() - initial_coords[:, 1].min() + 500
        cx = (initial_coords[:, 0].max() + initial_coords[:, 0].min()) / 2
        cy = (initial_coords[:, 1].max() + initial_coords[:, 1].min()) / 2
        lb = np.tile(np.array([cx - span_x / 2, cy - span_y / 2]), (N, 1))
        ub = np.tile(np.array([cx + span_x / 2, cy + span_y / 2]), (N, 1))
        bounds = (lb, ub)

    lb, ub = bounds

    initial_evaluated, initial_aep = evaluate_aep(
        initial_coords, turbine_library, model_names,
        directions, wind_speeds, frequency_matrix,
        wake_model, alpha, ti, superposition, min_spacing_multiple
    )

    population = []
    for _ in range(pop_size):
        individual = _random_feasible_individual(
            initial_coords, movable_mask, bounds, rotor_diams, min_spacing_multiple
        )
        population.append(individual)
    population[0] = initial_coords.copy()

    history_best = []
    history_avg = []
    best_ever_fitness = initial_evaluated
    best_ever_coords = initial_coords.copy()
    best_ever_aep = initial_aep

    def _eval_pop(pop):
        fitnesses = np.zeros(len(pop))
        aeps = np.zeros(len(pop))
        for i, ind in enumerate(pop):
            fit, aep = evaluate_aep(
                ind, turbine_library, model_names,
                directions, wind_speeds, frequency_matrix,
                wake_model, alpha, ti, superposition, min_spacing_multiple
            )
            fitnesses[i] = fit
            aeps[i] = aep
        return fitnesses, aeps

    for gen in range(n_generations):
        fitnesses, aeps = _eval_pop(population)

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]
        gen_best_aep = aeps[gen_best_idx]
        gen_avg_aep = float(np.mean(aeps))

        if gen_best_fit > best_ever_fitness:
            best_ever_fitness = gen_best_fit
            best_ever_coords = population[gen_best_idx].copy()
            best_ever_aep = gen_best_aep

        history_best.append(best_ever_aep)
        history_avg.append(gen_avg_aep)

        if progress_callback is not None:
            progress_callback(gen, {
                "generation": gen,
                "total_generations": n_generations,
                "best_aep_kwh": best_ever_aep,
                "initial_aep_kwh": initial_aep,
                "generation_best_aep_kwh": gen_best_aep,
                "generation_avg_aep_kwh": gen_avg_aep,
                "improvement_pct": 100.0 * (best_ever_aep - initial_aep) / max(initial_aep, 1e-6),
            })

        ranked_idx = np.argsort(-fitnesses)
        sorted_pop = [population[i] for i in ranked_idx]
        sorted_fit = fitnesses[ranked_idx]

        new_pop = []
        new_pop.append(sorted_pop[0].copy())
        new_pop.append(sorted_pop[1].copy())

        def _tournament_select(pop, fits, k=3):
            indices = np.random.choice(len(pop), k, replace=False)
            best_local = indices[np.argmax(fits[indices])]
            return pop[best_local].copy()

        while len(new_pop) < pop_size:
            p1 = _tournament_select(population, fitnesses, 3)
            p2 = _tournament_select(population, fitnesses, 3)

            if np.random.rand() < crossover_rate:
                child = np.zeros_like(p1)
                for i in range(N):
                    for j in range(2):
                        if movable_mask[i]:
                            child[i, j] = p1[i, j] if np.random.rand() < 0.5 else p2[i, j]
                        else:
                            child[i, j] = initial_coords[i, j]
            else:
                child = p1.copy()

            if np.random.rand() < mutation_rate:
                for i in range(N):
                    if not movable_mask[i]:
                        continue
                    for j in range(2):
                        if np.random.rand() < 0.5:
                            step = mutation_step_frac * (ub[i, j] - lb[i, j])
                            direction = np.random.choice([-1, 1])
                            child[i, j] += direction * step * np.random.rand()
                            child[i, j] = np.clip(child[i, j], lb[i, j], ub[i, j])

            new_pop.append(child)

        population = new_pop

    fitnesses, aeps = _eval_pop(population)
    final_best_idx = int(np.argmax(fitnesses))
    if fitnesses[final_best_idx] > best_ever_fitness:
        best_ever_fitness = fitnesses[final_best_idx]
        best_ever_coords = population[final_best_idx].copy()
        best_ever_aep = aeps[final_best_idx]

    history_best.append(best_ever_aep)
    history_avg.append(float(np.mean(aeps)))

    if progress_callback is not None:
        progress_callback(n_generations, {
            "generation": n_generations,
            "total_generations": n_generations,
            "best_aep_kwh": best_ever_aep,
            "initial_aep_kwh": initial_aep,
            "improvement_pct": 100.0 * (best_ever_aep - initial_aep) / max(initial_aep, 1e-6),
        })

    return {
        "initial_coords": initial_coords,
        "initial_aep_kwh": initial_aep,
        "best_coords": best_ever_coords,
        "best_aep_kwh": best_ever_aep,
        "aep_improvement_pct": 100.0 * (best_ever_aep - initial_aep) / max(initial_aep, 1e-6),
        "history_best_aep": np.array(history_best),
        "history_avg_aep": np.array(history_avg),
        "n_generations": n_generations,
        "pop_size": pop_size,
    }
