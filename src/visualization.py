"""
可视化模块
基于Plotly生成交互式图表:
  - 风场俯视图 (风机+尾流锥+着色)
  - 风向功率极坐标玫瑰图
  - 风机尾流损失柱状图
  - AEP风向贡献饼图
  - 优化收敛曲线
  - 优化前后布局对比图
  - 尾流损失矩阵热力图
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from .wake_models import _meteorological_to_math, get_wake_cone_polygon
from .power_calculation import compute_turbine_params_array


def _green_to_red(loss_fraction: float) -> str:
    """
    尾流损失 0~1 映射到 绿→黄→红
    """
    lf = np.clip(loss_fraction, 0, 1)
    if lf < 0.5:
        r = int(255 * lf * 2)
        g = 255
        b = 0
    else:
        r = 255
        g = int(255 * (1 - (lf - 0.5) * 2))
        b = 0
    return f"rgb({r},{g},{b})"


def plot_farm_layout(coords: np.ndarray,
                     turbine_ids: List[str],
                     model_names: List[str],
                     turbine_library: Dict[str, Dict],
                     per_turbine_loss: Optional[np.ndarray] = None,
                     wind_direction: Optional[float] = None,
                     show_wake_cones: bool = True,
                     alpha: float = 0.075,
                     highlight_turbines: Optional[List[int]] = None,
                     title: str = "风电场布局俯视图") -> go.Figure:
    """
    风场俯视图: 散点+叶轮圆圈+按尾流损失着色+可选尾流锥
    """
    N = coords.shape[0]
    params = compute_turbine_params_array(turbine_library, model_names)
    rotor_radii = params["rotor_radii"]

    if per_turbine_loss is None:
        per_turbine_loss = np.zeros(N)

    colors = [_green_to_red(l) for l in per_turbine_loss]

    fig = go.Figure()

    if show_wake_cones and wind_direction is not None:
        blow_to_meteo = (wind_direction + 180.0) % 360.0
        theta = _meteorological_to_math(blow_to_meteo)
        dir_x = np.cos(theta)
        dir_y = np.sin(theta)
        perp_x = -dir_y
        perp_y = dir_x

        max_extent = max(
            coords[:, 0].max() - coords[:, 0].min(),
            coords[:, 1].max() - coords[:, 1].min(),
            1000.0,
        )
        wake_length = max(max_extent * 2, 3000.0)

        for i in range(N):
            start_r = rotor_radii[i]
            end_r = start_r + alpha * wake_length
            cx, cy = coords[i]

            p1x = cx + perp_x * start_r
            p1y = cy + perp_y * start_r
            p2x = cx + dir_x * wake_length + perp_x * end_r
            p2y = cy + dir_y * wake_length + perp_y * end_r
            p3x = cx + dir_x * wake_length - perp_x * end_r
            p3y = cy + dir_y * wake_length - perp_y * end_r
            p4x = cx - perp_x * start_r
            p4y = cy - perp_y * start_r

            fig.add_trace(go.Scatter(
                x=[p1x, p2x, p3x, p4x, p1x],
                y=[p1y, p2y, p3y, p4y, p1y],
                fill="toself",
                fillcolor=f"rgba(100,149,237,0.10)",
                line=dict(width=1, color="rgba(100,149,237,0.5)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"Wake_{turbine_ids[i]}",
            ))

    for i in range(N):
        cx, cy = coords[i]
        r = rotor_radii[i]
        theta_pts = np.linspace(0, 2 * np.pi, 40)
        circ_x = cx + r * np.cos(theta_pts)
        circ_y = cy + r * np.sin(theta_pts)

        c = colors[i]
        highlight = False
        if highlight_turbines is not None and i in highlight_turbines:
            highlight = True

        fig.add_trace(go.Scatter(
            x=circ_x, y=circ_y,
            fill="toself",
            fillcolor=c.replace("rgb(", "rgba(").replace(")", ",0.25)"),
            line=dict(width=2 if highlight else 1, color="black" if highlight else c),
            hoverinfo="skip",
            showlegend=False,
        ))

        size_marker = max(8, r * 0.15)
        loss_pct = round(per_turbine_loss[i] * 100, 1)
        text = f"{turbine_ids[i]}<br>{model_names[i]}<br>尾流损失: {loss_pct}%<br>({cx:.0f}, {cy:.0f})m"

        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            marker=dict(
                size=size_marker,
                color=c,
                line=dict(width=2 if highlight else 1, color="black" if highlight else "white"),
            ),
            text=turbine_ids[i],
            textposition="top center",
            hovertemplate=text + "<extra></extra>",
            showlegend=False,
            name=turbine_ids[i],
        ))

    if wind_direction is not None:
        cx = coords[:, 0].mean()
        cy = coords[:, 1].max() + max(rotor_radii) * 3
        blow_to_meteo = (wind_direction + 180.0) % 360.0
        theta = _meteorological_to_math(blow_to_meteo)
        arrow_len = max(200, max(rotor_radii) * 2)

        fig.add_annotation(
            x=cx + np.cos(theta) * arrow_len,
            y=cy + np.sin(theta) * arrow_len,
            ax=cx,
            ay=cy,
            xref="x", yref="y",
            axref="x", ayref="y",
            text=f"风向 {wind_direction:.0f}°",
            showarrow=True,
            arrowhead=2, arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="red",
            font=dict(color="red", size=12, family="Arial"),
        )

    fig.update_layout(
        title=title,
        xaxis_title="X 坐标 (m)",
        yaxis_title="Y 坐标 (m)",
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False),
        height=600,
        hovermode="closest",
        plot_bgcolor="rgba(245,245,245,0.3)",
    )
    return fig


def plot_power_rose(angles_deg: np.ndarray,
                    power_kw: np.ndarray,
                    no_wake_power_kw: Optional[np.ndarray] = None,
                    best_angle: Optional[float] = None,
                    worst_angle: Optional[float] = None,
                    direction_freq: Optional[np.ndarray] = None,
                    title: str = "风向 - 功率极坐标玫瑰图") -> go.Figure:
    """
    风向-功率极坐标玫瑰图
    """
    fig = go.Figure()

    fig.add_trace(go.Barpolar(
        r=power_kw / 1000.0,
        theta=angles_deg,
        name="实际功率 (MW)",
        marker_color="royalblue",
        opacity=0.8,
        hovertemplate="风向: %{theta:.0f}°<br>功率: %{r:.2f} MW<extra></extra>",
    ))

    if no_wake_power_kw is not None:
        fig.add_trace(go.Scatterpolar(
            r=no_wake_power_kw / 1000.0,
            theta=angles_deg,
            mode="lines",
            name="无尾流功率 (MW)",
            line=dict(color="darkred", width=2, dash="dash"),
            hovertemplate="风向: %{theta:.0f}°<br>无尾流: %{r:.2f} MW<extra></extra>",
        ))

    if direction_freq is not None:
        freq_norm = direction_freq / direction_freq.max()
        fig.add_trace(go.Scatterpolar(
            r=freq_norm * (np.max(no_wake_power_kw if no_wake_power_kw is not None else power_kw) / 1000.0),
            theta=angles_deg,
            mode="lines",
            name="风向频率 (相对)",
            line=dict(color="green", width=2),
            opacity=0.6,
            hovertemplate="风向: %{theta:.0f}°<extra></extra>",
        ))

    annotations = []
    if best_angle is not None:
        best_idx = int(np.argmin(np.abs(angles_deg - best_angle)))
        annotations.append(dict(
            text=f"★ 最优风向 {best_angle:.0f}°",
            xref="paper", yref="paper",
            x=0.15, y=0.95, showarrow=False,
            font=dict(color="green", size=12),
        ))

    if worst_angle is not None:
        annotations.append(dict(
            text=f"⚠ 最差风向 {worst_angle:.0f}°",
            xref="paper", yref="paper",
            x=0.85, y=0.95, showarrow=False,
            font=dict(color="red", size=12),
        ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(title="功率 (MW)", showgrid=True),
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                direction="clockwise",
                rotation=90,
            ),
        ),
        annotations=annotations,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
    )
    return fig


def plot_wake_loss_bar(turbine_ids: List[str],
                       per_turbine_loss: np.ndarray,
                       top5_indices: Optional[List[int]] = None,
                       title: str = "各风机尾流损失率 (排序)") -> go.Figure:
    """
    各风机尾流损失柱状图, 按损失率降序排列
    """
    order = np.argsort(-per_turbine_loss)
    sorted_ids = [turbine_ids[i] for i in order]
    sorted_loss = per_turbine_loss[order] * 100.0

    colors = []
    if top5_indices is not None:
        top5_set = set(top5_indices)
        for idx_orig in order:
            colors.append("crimson" if idx_orig in top5_set else "cornflowerblue")
    else:
        colors = ["cornflowerblue"] * len(sorted_ids)

    fig = go.Figure(go.Bar(
        x=sorted_ids,
        y=sorted_loss,
        marker_color=colors,
        hovertemplate="风机: %{x}<br>尾流损失: %{y:.2f}%<extra></extra>",
        text=[f"{v:.1f}%" for v in sorted_loss],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="风机编号",
        yaxis_title="尾流损失率 (%)",
        height=500,
        xaxis_tickangle=-45,
    )
    return fig


def plot_aep_direction_pie(directions: np.ndarray,
                           aep_contrib_kwh: np.ndarray,
                           title: str = "AEP 各风向贡献占比") -> go.Figure:
    """
    AEP各风向贡献饼图
    """
    sector_width = 360.0 / len(directions) if len(directions) > 0 else 30
    labels = []
    for d in directions:
        s = d - sector_width / 2
        e = d + sector_width / 2
        labels.append(f"{s:.0f}°~{e:.0f}°")

    total = aep_contrib_kwh.sum()
    if total <= 0:
        total = 1.0
    percents = aep_contrib_kwh / total * 100.0

    fig = go.Figure(go.Pie(
        labels=labels,
        values=aep_contrib_kwh / 1e6,
        hovertemplate="扇区: %{label}<br>AEP贡献: %{value:.2f} GWh<br>占比: %{percent}<extra></extra>",
        textinfo="label+percent",
        textposition="inside",
    ))
    fig.update_layout(title=title, height=600)
    return fig


def plot_optimization_convergence(history_best: np.ndarray,
                                  history_avg: np.ndarray,
                                  initial_aep: float,
                                  best_aep: float,
                                  title: str = "遗传算法收敛曲线") -> go.Figure:
    """
    优化收敛曲线: 历代最佳AEP + 历代平均AEP
    """
    gens = np.arange(len(history_best))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gens,
        y=history_best / 1e6,
        mode="lines+markers",
        name="历代最佳 AEP",
        line=dict(color="royalblue", width=2),
        hovertemplate="第%{x}代<br>最佳: %{y:.3f} GWh<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=gens,
        y=history_avg / 1e6,
        mode="lines",
        name="种群平均 AEP",
        line=dict(color="orange", width=2, dash="dash"),
        hovertemplate="第%{x}代<br>平均: %{y:.3f} GWh<extra></extra>",
    ))

    improvement = (best_aep - initial_aep) / max(initial_aep, 1e-6) * 100.0
    fig.add_annotation(
        x=0.5, y=0.05, xref="paper", yref="paper",
        text=(f"初始 AEP: {initial_aep/1e6:.3f} GWh<br>"
              f"最佳 AEP: {best_aep/1e6:.3f} GWh<br>"
              f"提升: {improvement:+.2f}%"),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray", borderwidth=1,
    )

    fig.update_layout(
        title=title,
        xaxis_title="代数",
        yaxis_title="AEP (GWh)",
        height=450,
        legend=dict(x=0.01, y=0.95),
    )
    return fig


def plot_layout_comparison(before_coords: np.ndarray,
                           after_coords: np.ndarray,
                           turbine_ids: List[str],
                           model_names: List[str],
                           turbine_library: Dict[str, Dict],
                           movable_mask: Optional[np.ndarray] = None,
                           title: str = "优化前后布局对比") -> go.Figure:
    """
    优化前后布局对比图 (左右子图)
    """
    if movable_mask is None:
        movable_mask = np.ones(len(turbine_ids), dtype=bool)

    N = len(turbine_ids)
    params = compute_turbine_params_array(turbine_library, model_names)
    rotor_radii = params["rotor_radii"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("优化前布局", "优化后布局"),
                        shared_xaxes=False, shared_yaxes=False)

    def _add_turbines(fig_sub, coords, col, label_prefix, colors_override=None):
        for i in range(N):
            cx, cy = coords[i]
            r = rotor_radii[i]
            theta_pts = np.linspace(0, 2 * np.pi, 40)
            circ_x = cx + r * np.cos(theta_pts)
            circ_y = cy + r * np.sin(theta_pts)

            if colors_override is None:
                fill_color = "rgba(200,50,50,0.2)" if movable_mask[i] else "rgba(100,100,200,0.2)"
                line_color = "crimson" if movable_mask[i] else "navy"
            else:
                fill_color = colors_override[i].replace("rgb(", "rgba(").replace(")", ",0.2)")
                line_color = colors_override[i]

            fig_sub.add_trace(go.Scatter(
                x=circ_x, y=circ_y,
                fill="toself",
                fillcolor=fill_color,
                line=dict(width=1, color=line_color),
                hoverinfo="skip",
                showlegend=False,
            ), row=1, col=col)

            fig_sub.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(
                    size=max(6, r * 0.12),
                    color=("crimson" if movable_mask[i] else "navy") if colors_override is None else line_color,
                    line=dict(width=1, color="white"),
                ),
                text=turbine_ids[i],
                textposition="top center",
                textfont=dict(size=9),
                hovertemplate=(f"{label_prefix}{turbine_ids[i]}<br>{model_names[i]}<br>"
                               f"坐标: ({cx:.0f},{cy:.0f})m<extra></extra>"),
                showlegend=False,
            ), row=1, col=col)

    _add_turbines(fig, before_coords, 1, "")
    _add_turbines(fig, after_coords, 2, "")

    all_x = np.concatenate([before_coords[:, 0], after_coords[:, 0]])
    all_y = np.concatenate([before_coords[:, 1], after_coords[:, 1]])
    pad = max(rotor_radii) * 3
    xr = [all_x.min() - pad, all_x.max() + pad]
    yr = [all_y.min() - pad, all_y.max() + pad]

    for col in [1, 2]:
        fig.update_xaxes(range=xr, scaleanchor=f"y{col}", scaleratio=1, row=1, col=col)
        fig.update_yaxes(range=yr, row=1, col=col)

    fig.update_layout(title=title, height=600)
    return fig


def plot_loss_matrix_heatmap(loss_matrix: np.ndarray,
                             turbine_ids: List[str],
                             title: str = "尾流损失矩阵 (行=受影响, 列=施加影响)") -> go.Figure:
    """
    尾流损失矩阵热力图
    """
    matrix_pct = loss_matrix * 100.0
    fig = go.Figure(go.Heatmap(
        z=matrix_pct,
        x=turbine_ids,
        y=turbine_ids,
        colorscale="RdYlGn_r",
        hovertemplate="%{y} 受 %{x} 影响<br>亏损率: %{z:.3f}%<extra></extra>",
        colorbar=dict(title="风速亏损率 (%)"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="施加尾流影响的风机",
        yaxis_title="受尾流影响的风机",
        height=600,
        width=700,
    )
    return fig


def plot_sensitivity_heatmap(alpha_values: np.ndarray,
                             ti_values: np.ndarray,
                             aep_matrix: np.ndarray,
                             title: str = "参数敏感性分析 — AEP热力图") -> go.Figure:
    """
    alpha-TI参数敏感性热力图
    横轴alpha, 纵轴TI, 色块=AEP(GWh)
    """
    alpha_labels = [f"{v:.3f}" for v in alpha_values]
    ti_labels = [f"{v:.2f}" for v in ti_values]

    fig = go.Figure(go.Heatmap(
        z=aep_matrix,
        x=alpha_labels,
        y=ti_labels,
        colorscale="Viridis",
        hovertemplate="α=%{x}<br>TI=%{y}<br>AEP=%{z:.4f} GWh<extra></extra>",
        colorbar=dict(title="AEP (GWh)"),
        text=np.round(aep_matrix, 4),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="尾流扩展系数 α",
        yaxis_title="湍流强度 TI",
        height=550,
    )
    return fig


def plot_model_comparison_wind_speed(turbine_ids: List[str],
                                      jensen_ws: np.ndarray,
                                      bastankhah_ws: np.ndarray,
                                      jensen_powers_kw: np.ndarray,
                                      bastankhah_powers_kw: np.ndarray,
                                      title: str = "Jensen vs Bastankhah 模型对比") -> go.Figure:
    """
    两种尾流模型下各风机有效入流风速对比 (实线=Jensen, 虚线=Bastankhah)
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(turbine_ids))),
        y=jensen_ws,
        mode="lines+markers",
        name="Jensen (Park)",
        line=dict(color="royalblue", width=2),
        hovertemplate="风机: %{x}<br>Jensen风速: %{y:.2f} m/s<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(turbine_ids))),
        y=bastankhah_ws,
        mode="lines+markers",
        name="Bastankhah 高斯",
        line=dict(color="crimson", width=2, dash="dash"),
        hovertemplate="风机: %{x}<br>Bastankhah风速: %{y:.2f} m/s<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="风机编号",
            tickmode="array",
            tickvals=list(range(len(turbine_ids))),
            ticktext=turbine_ids,
            tickangle=-45,
        ),
        yaxis_title="有效入流风速 (m/s)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    return fig


def plot_power_curve(turbine_params: Dict, name: str) -> go.Figure:
    """
    绘制功率曲线和推力系数曲线 (双Y轴)
    """
    ws = turbine_params["power_curve_ws"]
    p_kw = turbine_params["power_curve_kw"]
    ct = turbine_params["thrust_curve_ct"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=ws, y=p_kw / 1000.0, mode="lines+markers",
        name="功率 (MW)", line=dict(color="royalblue", width=2),
        hovertemplate="风速: %{x} m/s<br>功率: %{y:.3f} MW<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=ws, y=ct, mode="lines+markers",
        name="推力系数 Ct", line=dict(color="crimson", width=2),
        hovertemplate="风速: %{x} m/s<br>Ct: %{y:.3f}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        title=f"机型 {name} 功率曲线 & 推力曲线",
        xaxis_title="风速 (m/s)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
    )
    fig.update_yaxes(title_text="功率 (MW)", secondary_y=False)
    fig.update_yaxes(title_text="推力系数 Ct", secondary_y=True)
    return fig


def plot_cash_flow_lines(
    years: np.ndarray,
    revenue: np.ndarray,
    cost: np.ndarray,
    net_flow: np.ndarray,
    title: str = "逐年现金流分析",
) -> go.Figure:
    """
    逐年现金流折线图: 收入线、支出线、净现金流线
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years,
        y=revenue,
        mode="lines+markers",
        name="收入",
        line=dict(color="#22c55e", width=3),
        marker=dict(size=8),
        hovertemplate="第%{x}年<br>收入: %{y:.2f} 万元<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=years,
        y=cost,
        mode="lines+markers",
        name="支出",
        line=dict(color="#ef4444", width=3),
        marker=dict(size=8),
        hovertemplate="第%{x}年<br>支出: %{y:.2f} 万元<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=years,
        y=net_flow,
        mode="lines+markers",
        name="净现金流",
        line=dict(color="#3b82f6", width=3, dash="solid"),
        marker=dict(size=8),
        hovertemplate="第%{x}年<br>净现金流: %{y:.2f} 万元<extra></extra>",
    ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.7,
    )

    fig.update_layout(
        title=title,
        xaxis_title="年份",
        yaxis_title="金额 (万元)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        plot_bgcolor="rgba(245,245,245,0.3)",
    )

    return fig


def plot_cumulative_cash_flow(
    years: np.ndarray,
    cumulative_flow: np.ndarray,
    payback_period: Optional[float] = None,
    title: str = "累计净现金流曲线",
) -> go.Figure:
    """
    累计净现金流面积图, 标注回收期拐点
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years,
        y=cumulative_flow,
        mode="lines",
        fill="tozeroy",
        name="累计净现金流",
        line=dict(color="#8b5cf6", width=3),
        fillcolor="rgba(139, 92, 246, 0.2)",
        hovertemplate="第%{x}年<br>累计: %{y:.2f} 万元<extra></extra>",
    ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        opacity=0.8,
        annotation_text="盈亏平衡点",
        annotation_position="bottom right",
        annotation_font=dict(color="red"),
    )

    if payback_period is not None and payback_period <= years[-1]:
        pb_year = int(np.floor(payback_period))
        if pb_year >= 1 and pb_year <= len(years):
            fig.add_annotation(
                x=payback_period,
                y=0,
                text=f"★ 投资回收期: {payback_period:.1f} 年",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="red",
                font=dict(color="red", size=12),
                ax=0,
                ay=-60,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="red",
                borderwidth=1,
            )

            fig.add_trace(go.Scatter(
                x=[payback_period],
                y=[0],
                mode="markers",
                marker=dict(size=14, color="red", symbol="circle"),
                name="投资回收点",
                hovertemplate=f"回收期: {payback_period:.1f} 年<extra></extra>",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="年份",
        yaxis_title="累计金额 (万元)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        plot_bgcolor="rgba(245,245,245,0.3)",
    )

    return fig


def plot_sensitivity_curve(
    param_values: np.ndarray,
    npv_values: np.ndarray,
    param_name: str,
    critical_value: Optional[float] = None,
    base_value: Optional[float] = None,
    title: str = "单变量敏感性分析",
) -> go.Figure:
    """
    单变量敏感性分析曲线: NPV随参数变化
    """
    param_label_map = {
        'electricity_price': '电价 (元/kWh)',
        'total_investment': '总投资成本 (万元)',
        'discount_rate': '折现率 (%)',
    }
    x_label = param_label_map.get(param_name, '参数值')

    fig = go.Figure()

    colors = np.where(npv_values >= 0, "#22c55e", "#ef4444")

    fig.add_trace(go.Scatter(
        x=param_values,
        y=npv_values,
        mode="lines+markers",
        name="NPV",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=10, color=colors, line=dict(width=2, color="#1e40af")),
        hovertemplate=f"{x_label.split(' ')[0]}: %{{x:.4f}}<br>NPV: %{{y:.2f}} 万元<extra></extra>",
    ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        opacity=0.8,
        annotation_text="NPV = 0",
        annotation_position="top right",
        annotation_font=dict(color="red"),
    )

    if base_value is not None:
        base_npv_idx = np.argmin(np.abs(param_values - base_value))
        base_npv = npv_values[base_npv_idx]
        fig.add_trace(go.Scatter(
            x=[base_value],
            y=[base_npv],
            mode="markers",
            marker=dict(size=14, color="#f59e0b", symbol="star"),
            name="基准值",
            hovertemplate=f"基准值: {base_value:.4f}<br>NPV: {base_npv:.2f} 万元<extra></extra>",
        ))

    if critical_value is not None:
        x_min, x_max = param_values.min(), param_values.max()
        if x_min <= critical_value <= x_max:
            fig.add_trace(go.Scatter(
                x=[critical_value],
                y=[0],
                mode="markers",
                marker=dict(size=14, color="red", symbol="circle"),
                name="临界值",
                hovertemplate=f"临界值: {critical_value:.4f}<br>NPV = 0<extra></extra>",
            ))

            unit = x_label.split('(')[-1].rstrip(')')
            fig.add_annotation(
                x=critical_value,
                y=0,
                text=f"🔴 临界值: {critical_value:.4f} {unit}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="red",
                font=dict(color="red", size=12),
                ax=0,
                ay=-50,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="red",
                borderwidth=1,
            )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="净现值 NPV (万元)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        plot_bgcolor="rgba(245,245,245,0.3)",
    )

    return fig
