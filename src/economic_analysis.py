"""
经济性分析与投资回报评估模块
包含:
  - 贷款等额本息计算
  - 逐年发电量衰减 (年0.5%)
  - 阶梯电价收入计算
  - 运维成本计算 (固定+可变)
  - 度电成本 LCOE
  - 净现值 NPV
  - 内部收益率 IRR (二分法)
  - 投资回收期 (含小数)
  - 敏感性分析
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class EconomicParams:
    """经济性分析输入参数"""
    total_investment: float = 0.0
    annual_fixed_om: float = 0.0
    variable_om_unit: float = 0.0
    electricity_price_1: float = 0.45
    electricity_price_2: float = 0.35
    price_switch_year: int = 10
    project_lifetime: int = 20
    discount_rate: float = 6.0
    salvage_rate: float = 5.0
    loan_ratio: float = 70.0
    loan_interest_rate: float = 4.5
    loan_tenor: int = 15
    aep_kwh: float = 0.0
    power_decay_rate: float = 0.5


@dataclass
class EconomicResults:
    """经济性分析输出结果"""
    lcoe: float = 0.0
    npv: float = 0.0
    irr: float = 0.0
    payback_period: float = 0.0
    total_revenue: float = 0.0
    total_profit: float = 0.0
    annual_generation: np.ndarray = field(default_factory=lambda: np.array([]))
    annual_revenue: np.ndarray = field(default_factory=lambda: np.array([]))
    annual_fixed_om_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    annual_variable_om_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    annual_loan_payment: np.ndarray = field(default_factory=lambda: np.array([]))
    annual_total_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    annual_cash_flow: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_cash_flow: np.ndarray = field(default_factory=lambda: np.array([]))
    loan_payment_amount: float = 0.0
    salvage_value: float = 0.0
    years: np.ndarray = field(default_factory=lambda: np.array([]))


def calculate_loan_payment(
    principal: float,
    annual_interest_rate: float,
    tenor_years: int,
) -> float:
    """
    等额本息还款计算: 每年固定还款金额
    A = P * r * (1+r)^n / ((1+r)^n - 1)
    """
    r = annual_interest_rate / 100.0
    n = tenor_years
    if r == 0:
        return principal / n
    annuity_factor = r * (1 + r) ** n / ((1 + r) ** n - 1)
    return principal * annuity_factor


def calculate_electricity_price(
    year: int,
    price_1: float,
    price_2: float,
    switch_year: int,
) -> float:
    """计算第year年的电价 (阶梯电价)"""
    if year <= switch_year:
        return price_1
    else:
        return price_2


def calculate_annual_generation(
    base_aep_kwh: float,
    project_lifetime: int,
    decay_rate_pct: float = 0.5,
) -> np.ndarray:
    """
    计算逐年发电量, 考虑年功率衰减
    第1年 = base * (1 - decay)^0
    第2年 = base * (1 - decay)^1
    ...
    """
    decay = decay_rate_pct / 100.0
    years = np.arange(1, project_lifetime + 1)
    generation = base_aep_kwh * (1 - decay) ** (years - 1)
    return generation


def calculate_lcoe(
    total_investment: float,
    annual_fixed_om: float,
    variable_om_unit: float,
    annual_generation: np.ndarray,
    discount_rate: float,
    project_lifetime: int,
    salvage_rate: float,
) -> float:
    """
    计算度电成本 LCOE (Levelized Cost of Energy)
    LCOE = 生命周期成本现值 / 生命周期发电量现值
    采用无杠杆(项目整体)视角, 不考虑融资结构(贷款比例不影响LCOE)
    生命周期成本包括: 初始投资 + 运维成本 - 残值
    单位说明: 成本输入为万元, 发电量为kWh, LCOE输出为元/kWh
    """
    r = discount_rate / 100.0
    years = np.arange(1, project_lifetime + 1)
    discount_factors = 1.0 / (1 + r) ** years

    fixed_om_costs = np.full(project_lifetime, annual_fixed_om)
    variable_om_costs = variable_om_unit * annual_generation / 10000.0
    total_om_costs = fixed_om_costs + variable_om_costs

    cost_pv = total_investment + np.sum(total_om_costs * discount_factors)

    salvage_value = total_investment * salvage_rate / 100.0
    salvage_pv = salvage_value * discount_factors[-1]
    cost_pv -= salvage_pv

    generation_pv = np.sum(annual_generation * discount_factors)

    if generation_pv > 0:
        lcoe = cost_pv * 10000.0 / generation_pv
    else:
        lcoe = 0.0
    return lcoe


def calculate_npv(
    cash_flows: np.ndarray,
    discount_rate: float,
) -> float:
    """
    计算净现值 NPV
    NPV = Σ CF_t / (1+r)^t, t=0..n
    注意: cash_flows[0] 是第1年的现金流 (即t=1), t=0只有初始投资
    """
    r = discount_rate / 100.0
    n = len(cash_flows)
    years = np.arange(1, n + 1)
    discount_factors = 1.0 / (1 + r) ** years
    return np.sum(cash_flows * discount_factors)


def calculate_irr(
    initial_investment: float,
    cash_flows: np.ndarray,
    tol: float = 1e-7,
    max_iter: int = 1000,
) -> float:
    """
    二分法求解内部收益率 IRR
    使 NPV = -initial_investment + Σ CF_t/(1+IRR)^t = 0
    """
    def npv_func(rate):
        r = rate / 100.0
        n = len(cash_flows)
        years = np.arange(1, n + 1)
        discount_factors = 1.0 / (1 + r) ** years
        return -initial_investment + np.sum(cash_flows * discount_factors)

    low = -50.0
    high = 100.0

    npv_low = npv_func(low)
    npv_high = npv_func(high)

    if npv_low * npv_high > 0:
        if np.abs(npv_low) < np.abs(npv_high):
            return low
        else:
            return high

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        npv_mid = npv_func(mid)

        if np.abs(npv_mid) < tol:
            return mid

        if npv_low * npv_mid <= 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid

    return (low + high) / 2.0


def calculate_payback_period(
    cumulative_cash_flow: np.ndarray,
) -> float:
    """
    计算投资回收期 (精确值, 显示时再四舍五入)
    累计净现金流首次由负转正的年份
    返回: 回收期(年), 若项目寿命期内无法回收则返回项目寿命+1
    """
    n = len(cumulative_cash_flow)

    for i in range(n):
        if cumulative_cash_flow[i] >= 0:
            if i == 0:
                return 1.0
            negative_last = cumulative_cash_flow[i - 1]
            positive_current = cumulative_cash_flow[i]
            if positive_current - negative_last == 0:
                return float(i + 1)
            fraction = (-negative_last) / (positive_current - negative_last)
            return i + fraction

    return float(n + 1)


def run_economic_analysis(
    params: EconomicParams,
) -> EconomicResults:
    """
    执行完整的经济性分析
    """
    results = EconomicResults()

    project_lifetime = params.project_lifetime
    results.years = np.arange(1, project_lifetime + 1)

    results.annual_generation = calculate_annual_generation(
        params.aep_kwh, project_lifetime, params.power_decay_rate
    )

    prices = np.array([
        calculate_electricity_price(
            y, params.electricity_price_1,
            params.electricity_price_2, params.price_switch_year
        )
        for y in results.years
    ])
    results.annual_revenue = results.annual_generation * prices / 10000.0

    results.annual_fixed_om_cost = np.full(
        project_lifetime, params.annual_fixed_om
    )
    results.annual_variable_om_cost = (
        params.variable_om_unit * results.annual_generation / 10000.0
    )

    loan_amount = params.total_investment * params.loan_ratio / 100.0
    equity_investment = params.total_investment - loan_amount

    if loan_amount > 0 and params.loan_tenor > 0:
        results.loan_payment_amount = calculate_loan_payment(
            loan_amount, params.loan_interest_rate, params.loan_tenor
        )
        results.annual_loan_payment = np.zeros(project_lifetime)
        results.annual_loan_payment[:params.loan_tenor] = results.loan_payment_amount
    else:
        results.loan_payment_amount = 0.0
        results.annual_loan_payment = np.zeros(project_lifetime)

    results.annual_total_cost = (
        results.annual_fixed_om_cost
        + results.annual_variable_om_cost
        + results.annual_loan_payment
    )

    results.annual_cash_flow = results.annual_revenue - results.annual_total_cost

    results.salvage_value = params.total_investment * params.salvage_rate / 100.0
    results.annual_cash_flow[-1] += results.salvage_value

    results.cumulative_cash_flow = np.zeros(project_lifetime)
    results.cumulative_cash_flow[0] = -params.total_investment + results.annual_cash_flow[0]
    for i in range(1, project_lifetime):
        results.cumulative_cash_flow[i] = (
            results.cumulative_cash_flow[i - 1] + results.annual_cash_flow[i]
        )

    results.lcoe = calculate_lcoe(
        params.total_investment,
        params.annual_fixed_om,
        params.variable_om_unit,
        results.annual_generation,
        params.discount_rate,
        project_lifetime,
        params.salvage_rate,
    )

    results.npv = calculate_npv(results.annual_cash_flow, params.discount_rate)
    results.npv -= equity_investment

    results.irr = calculate_irr(equity_investment, results.annual_cash_flow)

    cum_with_initial = np.zeros(project_lifetime + 1)
    cum_with_initial[0] = -params.total_investment
    for i in range(project_lifetime):
        cum_with_initial[i + 1] = cum_with_initial[i] + results.annual_cash_flow[i]
    results.payback_period = calculate_payback_period(cum_with_initial[1:])

    results.total_revenue = np.sum(results.annual_revenue)
    total_cost_life = (
        params.total_investment
        + np.sum(results.annual_fixed_om_cost)
        + np.sum(results.annual_variable_om_cost)
        + np.sum(results.annual_loan_payment)
        - results.salvage_value
    )
    results.total_profit = results.total_revenue - total_cost_life

    return results


def run_sensitivity_analysis(
    base_params: EconomicParams,
    param_name: str,
    variation_pct: float = 30.0,
    num_points: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    单变量敏感性分析
    参数:
        base_params: 基准参数
        param_name: 变化的参数名 ('electricity_price', 'total_investment', 'discount_rate')
        variation_pct: 变化范围 (±%)
        num_points: 扫描点数
    返回:
        param_values: 变化的参数值数组
        npv_values: 对应NPV数组
        critical_value: NPV=0时的临界值 (若存在)
    """
    param_map = {
        'electricity_price': 'electricity_price_1',
        'total_investment': 'total_investment',
        'discount_rate': 'discount_rate',
    }

    if param_name not in param_map:
        raise ValueError(f"不支持的敏感性分析参数: {param_name}")

    attr_name = param_map[param_name]
    base_value = getattr(base_params, attr_name)

    min_val = base_value * (1 - variation_pct / 100.0)
    max_val = base_value * (1 + variation_pct / 100.0)

    param_values = np.linspace(min_val, max_val, num_points)
    npv_values = np.zeros(num_points)

    for i, val in enumerate(param_values):
        test_params = EconomicParams()
        for k, v in base_params.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(test_params, k, v.copy())
            else:
                setattr(test_params, k, v)

        setattr(test_params, attr_name, val)
        results = run_economic_analysis(test_params)
        npv_values[i] = results.npv

    critical_value = None
    sign_changes = np.where(np.diff(np.sign(npv_values)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        x0, x1 = param_values[idx], param_values[idx + 1]
        y0, y1 = npv_values[idx], npv_values[idx + 1]
        if y1 != y0:
            critical_value = x0 - y0 * (x1 - x0) / (y1 - y0)

    return param_values, npv_values, critical_value
