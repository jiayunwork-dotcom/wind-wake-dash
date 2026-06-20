"""
风电场尾流效应分析与发电功率预测系统 - Streamlit主应用
"""

import os
import sys
import io
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple

from src import (
    get_builtin_turbines,
    parse_custom_turbine_csv,
    parse_farm_layout_csv,
    generate_frequency_from_weibull,
    parse_frequency_csv,
    parse_weibull_per_sector_csv,
    expand_weibull_to_frequency,
    compute_power_for_direction_windspeed,
    scan_all_directions_windspeeds,
    scan_direction_rose,
    check_spacing_constraint,
    compute_turbine_params_array,
    genetic_algorithm_layout_optimize,
    plot_farm_layout,
    plot_power_rose,
    plot_wake_loss_bar,
    plot_aep_direction_pie,
    plot_optimization_convergence,
    plot_layout_comparison,
    plot_loss_matrix_heatmap,
    plot_power_curve,
    sample_rose_from_frequency,
    mean_wind_speed_from_frequency,
)


st.set_page_config(
    page_title="风电场尾流分析与AEP预测系统",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(show_spinner=False)
def _get_builtin_turbines_cached():
    return get_builtin_turbines()


def _load_sample_layout():
    data_path = os.path.join(os.path.dirname(__file__), "data", "sample_farm_layout.csv")
    with open(data_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_sample_wind_resource():
    data_path = os.path.join(os.path.dirname(__file__), "data", "sample_wind_resource.csv")
    with open(data_path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# Session State 初始化
# ============================================================
default_state = {
    "turbine_library": _get_builtin_turbines_cached(),
    "farm_coords": None,
    "farm_ids": [],
    "farm_models": [],
    "wind_directions": None,
    "wind_speeds": None,
    "wind_freq": None,
    "turbulence_intensity": 0.10,
    "scanning_results": None,
    "rose_results": None,
    "opt_results": None,
    "single_dir_results": None,
    "spacing_checked": False,
    "spacing_ok": False,
    "spacing_min_ratio": 1.0,
}

for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = copy.deepcopy(v)


# ============================================================
# 侧边栏 - 全局参数
# ============================================================
with st.sidebar:
    st.title("🌬️ 风电尾流分析系统")
    st.caption("Wake Effect Analysis & AEP Prediction")

    st.divider()
    st.subheader("📐 尾流模型参数")
    wake_model = st.selectbox(
        "尾流模型",
        ["Jensen (Park)", "Bastankhah 高斯"],
        index=0,
        help="Jensen: 线性锥形尾流, 计算快; Bastankhah: 高斯分布截面, 精度高"
    )
    WAKE_MODEL_MAP = {"Jensen (Park)": "jensen", "Bastankhah 高斯": "bastankhah"}
    wake_model_code = WAKE_MODEL_MAP[wake_model]

    alpha_label = st.radio(
        "尾流扩展系数 α 预设",
        ["陆上 (α=0.075)", "海上 (α=0.040)", "自定义"],
        horizontal=True,
    )
    if "陆上" in alpha_label:
        alpha_default = 0.075
    elif "海上" in alpha_label:
        alpha_default = 0.040
    else:
        alpha_default = 0.075
    alpha = st.slider("α (尾流扩展系数)", 0.02, 0.15, alpha_default, 0.005,
                      format="%.3f")

    superposition = st.selectbox(
        "多尾流叠加策略",
        ["线性叠加 (Linear)", "均方根叠加 (RMS)"],
        index=0,
    )
    SUPER_MAP = {"线性叠加 (Linear)": "linear", "均方根叠加 (RMS)": "rms"}
    superposition_code = SUPER_MAP[superposition]

    st.session_state.turbulence_intensity = st.slider(
        "参考湍流强度 TI",
        0.02, 0.25, st.session_state.turbulence_intensity, 0.01,
        format="%.2f",
        help="影响Bastankhah模型尾流恢复速率",
    )

    spacing_multiple = st.slider(
        "最小间距倍数 (叶轮直径)",
        1.0, 5.0, 2.0, 0.1,
        help="布局检查约束: 风机间距 ≥ 该值 × 最大叶轮直径",
    )

    st.divider()
    st.subheader("🔧 运行控制")
    if st.button("🔄 重置全部数据", type="secondary"):
        for k, v in default_state.items():
            st.session_state[k] = copy.deepcopy(v)
        st.success("已重置")
        st.rerun()


# ============================================================
# 主区域 - Tab 布局
# ============================================================
tab_overview, tab_turbines, tab_layout, tab_wind, tab_single, tab_aep, tab_opt = \
    st.tabs([
        "📊 总览面板",
        "⚙️ 机型管理",
        "📍 风电场布局",
        "🌬️ 风资源数据",
        "🧭 单风向分析",
        "📈 AEP与损失分析",
        "🧬 布局优化",
    ])


# ============================================================
# Tab 1: 机型管理
# ============================================================
with tab_turbines:
    st.header("⚙️ 风机机型参数库")
    st.write("内置 4 款常见机型，也支持用户自定义上传机型参数。")

    builtin = st.session_state.turbine_library
    names = list(builtin.keys())

    model_col, info_col = st.columns([1, 2])
    with model_col:
        st.subheader("机型列表")
        selected_model = st.selectbox("查看机型详情", names)

        st.divider()
        st.subheader("📤 自定义机型")
        uploaded_custom = st.file_uploader(
            "上传机型CSV",
            type=["csv"],
            key="custom_turbine_csv",
            help="格式: 首行元数据(k=v), 然后列为: wind_speed,power_kw,thrust_coeff",
        )
        custom_name = st.text_input("自定义机型名称", value="Custom_Model")
        if st.button("添加自定义机型", disabled=uploaded_custom is None):
            try:
                content = uploaded_custom.getvalue().decode("utf-8")
                params = parse_custom_turbine_csv(content)
                st.session_state.turbine_library[custom_name] = params
                st.success(f"已添加机型: {custom_name}")
                st.rerun()
            except Exception as e:
                st.error(f"解析失败: {e}")

    with info_col:
        if selected_model:
            params = builtin[selected_model]
            st.subheader(f"📋 {selected_model} 参数")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("额定功率", f"{params['rated_power']:.1f} MW")
            c2.metric("叶轮直径", f"{params['rotor_diameter']:.0f} m")
            c3.metric("轮毂高度", f"{params['hub_height']:.0f} m")
            c4.metric("切入/额定/切出",
                      f"{params['cut_in_speed']:.1f}/{params['rated_speed']:.1f}/{params['cut_out_speed']:.1f} m/s")

            st.plotly_chart(
                plot_power_curve(params, selected_model),
                use_container_width=True,
            )

            with st.expander("查看功率曲线/推力曲线数据表"):
                df = pd.DataFrame({
                    "风速 (m/s)": params["power_curve_ws"],
                    "功率 (kW)": params["power_curve_kw"],
                    "推力系数 Ct": params["thrust_curve_ct"],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📚 当前机型库概览")
    summary_rows = []
    for name, p in builtin.items():
        summary_rows.append({
            "机型": name,
            "额定功率 (MW)": p["rated_power"],
            "叶轮直径 (m)": p["rotor_diameter"],
            "轮毂高度 (m)": p["hub_height"],
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ============================================================
# Tab 2: 风电场布局
# ============================================================
with tab_layout:
    st.header("📍 风电场布局导入与检查")
    st.markdown("上传CSV包含: **风机编号, X坐标, Y坐标, 风机型号**")

    l_col, r_col = st.columns([1, 1.3])
    with l_col:
        st.subheader("📤 布局数据输入")

        use_sample = st.button("📥 加载示例风场 (12台风机)", type="primary")
        if use_sample:
            content = _load_sample_layout()
            try:
                coords, tids, mnames = parse_farm_layout_csv(content, st.session_state.turbine_library)
                st.session_state.farm_coords = coords
                st.session_state.farm_ids = tids
                st.session_state.farm_models = mnames
                st.session_state.spacing_checked = False
                st.success(f"已加载 {len(tids)} 台风机的示例布局")
            except Exception as e:
                st.error(f"加载失败: {e}")

        uploaded_layout = st.file_uploader(
            "上传风场布局CSV",
            type=["csv"],
            key="layout_csv",
        )
        if uploaded_layout:
            try:
                content = uploaded_layout.getvalue().decode("utf-8")
                coords, tids, mnames = parse_farm_layout_csv(content, st.session_state.turbine_library)
                st.session_state.farm_coords = coords
                st.session_state.farm_ids = tids
                st.session_state.farm_models = mnames
                st.session_state.spacing_checked = False
                st.success(f"成功解析 {len(tids)} 台风机")
            except Exception as e:
                st.error(f"解析失败: {e}")

        if st.session_state.farm_coords is not None:
            st.divider()
            st.subheader("🔍 间距约束检查")
            if st.button("检查风机间距约束", type="secondary"):
                params = compute_turbine_params_array(
                    st.session_state.turbine_library,
                    st.session_state.farm_models,
                )
                ok, viol, min_r = check_spacing_constraint(
                    st.session_state.farm_coords,
                    params["rotor_diameters"],
                    spacing_multiple,
                )
                st.session_state.spacing_checked = True
                st.session_state.spacing_ok = ok
                st.session_state.spacing_min_ratio = min_r

            if st.session_state.spacing_checked:
                r1, r2 = st.columns(2)
                r1.metric("最小间距/要求比值", f"{st.session_state.spacing_min_ratio:.2f}")
                if st.session_state.spacing_ok:
                    r2.success("✅ 全部风机满足间距约束")
                else:
                    r2.error("❌ 存在不满足约束的风机对")

            st.divider()
            df_farm = pd.DataFrame({
                "风机编号": st.session_state.farm_ids,
                "X (m)": st.session_state.farm_coords[:, 0],
                "Y (m)": st.session_state.farm_coords[:, 1],
                "机型": st.session_state.farm_models,
            })
            st.dataframe(df_farm, use_container_width=True, hide_index=True)

    with r_col:
        st.subheader("🗺️ 布局可视化")
        if st.session_state.farm_coords is not None:
            layout_fig = plot_farm_layout(
                st.session_state.farm_coords,
                st.session_state.farm_ids,
                st.session_state.farm_models,
                st.session_state.turbine_library,
                per_turbine_loss=None,
                wind_direction=None,
                show_wake_cones=False,
                alpha=alpha,
            )
            st.plotly_chart(layout_fig, use_container_width=True)

            total_rated = np.sum([
                st.session_state.turbine_library[m]["rated_power"]
                for m in st.session_state.farm_models
            ])
            c1, c2, c3 = st.columns(3)
            c1.metric("风机总数", f"{len(st.session_state.farm_ids)}")
            c2.metric("总额定功率", f"{total_rated:.1f} MW")
            span_x = st.session_state.farm_coords[:, 0].max() - st.session_state.farm_coords[:, 0].min()
            span_y = st.session_state.farm_coords[:, 1].max() - st.session_state.farm_coords[:, 1].min()
            c3.metric("场地尺度", f"{span_x:.0f} × {span_y:.0f} m")
        else:
            st.info("💡 请先加载或上传风电场布局数据")


# ============================================================
# Tab 3: 风资源数据
# ============================================================
with tab_wind:
    st.header("🌬️ 风资源数据输入")

    mode = st.radio(
        "数据输入模式",
        [
            "🎯 生成示例数据 (Weibull参数)",
            "📤 上传风向风速联合频率表 CSV",
            "📊 上传各扇区Weibull参数 CSV",
        ],
        horizontal=True,
    )

    if "🎯 生成示例数据" in mode:
        c1, c2, c3 = st.columns(3)
        with c1:
            dir_sectors = st.selectbox("风向扇区数", [12, 36], index=0)
        with c2:
            k_weibull = st.slider("Weibull 形状因子 k", 1.5, 4.0, 2.2, 0.1)
        with c3:
            A_weibull = st.slider("Weibull 尺度因子 A (m/s)", 4.0, 15.0, 8.5, 0.5)

        if st.button("生成风资源数据", type="primary"):
            dirs, wss, freq = generate_frequency_from_weibull(
                direction_sectors=dir_sectors,
                k=k_weibull, A=A_weibull,
            )
            st.session_state.wind_directions = dirs
            st.session_state.wind_speeds = wss
            st.session_state.wind_freq = freq
            st.success("已生成风资源数据")

    elif "联合频率表" in mode:
        use_sample_wr = st.button("📥 加载示例风资源数据", type="primary")
        if use_sample_wr:
            content = _load_sample_wind_resource()
            try:
                dirs, wss, freq = parse_frequency_csv(content)
                st.session_state.wind_directions = dirs
                st.session_state.wind_speeds = wss
                st.session_state.wind_freq = freq
                st.success(f"已加载 {len(dirs)} × {len(wss)} 风资源矩阵")
            except Exception as e:
                st.error(f"解析失败: {e}")

        uploaded_wr = st.file_uploader("上传风向风速频率CSV", type=["csv"], key="wr_csv")
        if uploaded_wr:
            try:
                content = uploaded_wr.getvalue().decode("utf-8")
                dirs, wss, freq = parse_frequency_csv(content)
                st.session_state.wind_directions = dirs
                st.session_state.wind_speeds = wss
                st.session_state.wind_freq = freq
                st.success(f"已解析 {len(dirs)} 个扇区, {len(wss)} 个风速分箱")
            except Exception as e:
                st.error(f"解析失败: {e}")

    else:
        uploaded_wb = st.file_uploader("上传各扇区Weibull参数CSV", type=["csv"], key="wb_csv")
        if uploaded_wb:
            try:
                content = uploaded_wb.getvalue().decode("utf-8")
                dirs, ks, As = parse_weibull_per_sector_csv(content)
                dirs2, wss, freq = expand_weibull_to_frequency(dirs, ks, As)
                st.session_state.wind_directions = dirs2
                st.session_state.wind_speeds = wss
                st.session_state.wind_freq = freq
                st.success(f"已展开为 {len(dirs2)} × {len(wss)} 频率矩阵")
            except Exception as e:
                st.error(f"解析失败: {e}")

    # --- 风资源数据展示 ---
    st.divider()
    if st.session_state.wind_freq is not None:
        dirs = st.session_state.wind_directions
        wss = st.session_state.wind_speeds
        freq = st.session_state.wind_freq

        c1, c2, c3, c4 = st.columns(4)
        mean_v = mean_wind_speed_from_frequency(wss, freq)
        dir_freq_arr, dir_freq_vals = sample_rose_from_frequency(dirs, wss, freq)
        dominant_idx = int(np.argmax(dir_freq_vals))

        c1.metric("扇区数", f"{len(dirs)}")
        c2.metric("风速分箱数", f"{len(wss)}")
        c3.metric("平均风速", f"{mean_v:.2f} m/s")
        c4.metric("主导风向", f"{dirs[dominant_idx]:.0f}° ({dir_freq_vals[dominant_idx]*100:.1f}%)")

        st.subheader("📊 风向频率分布")
        freq_d_fig = plot_power_rose(
            dirs,
            dir_freq_vals * 100,
            None, None, None, None,
            title="风向频率玫瑰图 (%)",
        )
        # 复用功率玫瑰图会有"功率MW"标签，这里直接用Pie更简洁
        pie_data = go.Pie(
            labels=[f"{d:.0f}°" for d in dirs],
            values=dir_freq_vals * 100,
            textinfo="label+percent",
            hovertemplate="风向%{label}<br>频率: %{value:.2f}%<extra></extra>",
        )
        import plotly.graph_objects as go
        dir_pie = go.Figure(pie_data)
        dir_pie.update_layout(title="各风向扇区频率占比", height=450)
        st.plotly_chart(dir_pie, use_container_width=True)

        with st.expander("查看风向风速联合频率矩阵 (%)"):
            df_freq = pd.DataFrame(
                (freq * 100).round(4),
                index=[f"{d:.0f}°" for d in dirs],
                columns=[f"{w:.1f}m/s" for w in wss],
            )
            st.dataframe(df_freq, use_container_width=True)
    else:
        st.info("💡 请先生成或上传风资源数据")


# ============================================================
# Tab 4: 单风向分析
# ============================================================
with tab_single:
    st.header("🧭 单风向工况分析")

    data_ready = (
        st.session_state.farm_coords is not None
        and st.session_state.wind_freq is not None
    )
    if not data_ready:
        st.warning("⚠️ 请先完成『风电场布局』和『风资源数据』配置")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            analysis_direction = st.slider(
                "分析风向 (气象风向, 风来的方向)",
                0, 359, 270, 1,
                help="0°=北风, 90°=东风, 180°=南风, 270°=西风",
            )
        with c2:
            mean_ws = mean_wind_speed_from_frequency(
                st.session_state.wind_speeds, st.session_state.wind_freq
            )
            analysis_ws = st.slider(
                "环境风速 (m/s)",
                0.5, 30.0, float(round(mean_ws, 1)), 0.5,
            )
        with c3:
            st.write("")
            run_single = st.button("▶️ 计算此工况", type="primary")

        show_wake = st.checkbox("显示尾流锥形区域", value=True)

        if run_single or st.session_state.single_dir_results is not None:
            if run_single:
                with st.spinner("计算中..."):
                    res = compute_power_for_direction_windspeed(
                        st.session_state.farm_coords,
                        st.session_state.turbine_library,
                        st.session_state.farm_models,
                        analysis_direction, analysis_ws,
                        wake_model_code, alpha,
                        st.session_state.turbulence_intensity,
                        superposition_code,
                    )
                    st.session_state.single_dir_results = res
            res = st.session_state.single_dir_results

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("全场功率", f"{res['total_power_kw']/1000:.2f} MW")
            t2.metric("无尾流功率", f"{res['total_no_wake_kw']/1000:.2f} MW")
            t3.metric("全场尾流损失率", f"{res['overall_loss']*100:.2f} %",
                      delta=f"-{(res['total_no_wake_kw']-res['total_power_kw'])/1000:.2f} MW")
            t4.metric("效率", f"{(1-res['overall_loss'])*100:.1f} %")

            fig = plot_farm_layout(
                st.session_state.farm_coords,
                st.session_state.farm_ids,
                st.session_state.farm_models,
                st.session_state.turbine_library,
                per_turbine_loss=res["per_turbine_loss"],
                wind_direction=analysis_direction,
                show_wake_cones=show_wake,
                alpha=alpha,
                title=f"风场俯视图 - 风向 {analysis_direction}°, 风速 {analysis_ws:.1f} m/s",
            )
            st.plotly_chart(fig, use_container_width=True)

            c_fig1, c_fig2 = st.columns([1.2, 1])
            with c_fig1:
                bar_fig = plot_wake_loss_bar(
                    st.session_state.farm_ids,
                    res["per_turbine_loss"],
                    title="各风机尾流损失率 (单工况)",
                )
                st.plotly_chart(bar_fig, use_container_width=True)
            with c_fig2:
                df_p = pd.DataFrame({
                    "风机": st.session_state.farm_ids,
                    "机型": st.session_state.farm_models,
                    "入流风速 (m/s)": res["effective_wind_speeds"].round(2),
                    "单机功率 (kW)": res["turbine_powers_kw"].round(1),
                    "无尾流功率 (kW)": res["no_wake_powers_kw"].round(1),
                    "尾流损失 (%)": (res["per_turbine_loss"] * 100).round(2),
                })
                st.dataframe(df_p, use_container_width=True, hide_index=True)

            st.subheader("📉 尾流损失矩阵 (本工况)")
            st.plotly_chart(
                plot_loss_matrix_heatmap(res["deficit_matrix"], st.session_state.farm_ids),
                use_container_width=True,
            )


# ============================================================
# Tab 5: AEP 总分析
# ============================================================
with tab_aep:
    st.header("📈 AEP年发电量 & 尾流损失综合分析")

    data_ready = (
        st.session_state.farm_coords is not None
        and st.session_state.wind_freq is not None
    )
    if not data_ready:
        st.warning("⚠️ 请先完成『风电场布局』和『风资源数据』配置")
    else:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            do_scan = st.button("🚀 运行全场扫描计算", type="primary",
                                help="遍历所有风向×风速组合, 计算AEP")
            do_rose = st.button("🌹 逐度风向扫描 (绘制玫瑰图)",
                                help="0~359度逐度扫描, 风速固定为平均风速")
            rose_ws = st.slider("玫瑰图风速 (m/s)", 0.5, 30.0,
                                float(round(mean_wind_speed_from_frequency(
                                    st.session_state.wind_speeds, st.session_state.wind_freq
                                ), 1)), 0.5)

        with col_b:
            st.info(
                f"待扫描规模: {len(st.session_state.wind_directions)} 风向 × "
                f"{len(st.session_state.wind_speeds)} 风速 = "
                f"{len(st.session_state.wind_directions) * len(st.session_state.wind_speeds)} 工况"
            )

        if do_scan:
            with st.spinner("全场扫描计算中... (多工况向量化运算)"):
                result = scan_all_directions_windspeeds(
                    st.session_state.farm_coords,
                    st.session_state.turbine_library,
                    st.session_state.farm_models,
                    st.session_state.wind_directions,
                    st.session_state.wind_speeds,
                    st.session_state.wind_freq,
                    wake_model_code, alpha,
                    st.session_state.turbulence_intensity,
                    superposition_code,
                )
                st.session_state.scanning_results = result
            st.success("全场扫描完成!")

        if do_rose:
            with st.spinner("逐度风向扫描中... (360个方向)"):
                r_res = scan_direction_rose(
                    st.session_state.farm_coords,
                    st.session_state.turbine_library,
                    st.session_state.farm_models,
                    rose_ws, wake_model_code, alpha,
                    st.session_state.turbulence_intensity,
                    superposition_code,
                )
                st.session_state.rose_results = r_res
            st.success(f"风向扫描完成! 最优: {r_res['best_direction']:.0f}°, 最差: {r_res['worst_direction']:.0f}°")

        # --- 展示扫描结果 ---
        if st.session_state.scanning_results is not None:
            r = st.session_state.scanning_results
            st.subheader("🎯 关键指标")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("AEP (实际)", f"{r['aep_kwh']/1e6:.3f} GWh")
            k2.metric("AEP (无尾流)", f"{r['aep_no_wake_kwh']/1e6:.3f} GWh")
            aep_loss = (r['aep_no_wake_kwh'] - r['aep_kwh'])
            k3.metric("尾流损失电量", f"{aep_loss/1e6:.3f} GWh",
                      delta=f"-{r['overall_loss']*100:.2f}%")
            k4.metric("容量系数", f"{r['capacity_factor']*100:.2f} %")
            k5.metric("总额定功率", f"{r['total_rated_kw']/1000:.1f} MW")

            st.divider()
            g1, g2 = st.columns([1.2, 1])
            with g1:
                st.subheader("🌹 风向-功率玫瑰图")
                if st.session_state.rose_results is not None:
                    rr = st.session_state.rose_results
                    dir_freq_aligned = np.interp(
                        rr["angles_deg"],
                        st.session_state.wind_directions,
                        st.session_state.wind_freq.sum(axis=1),
                    )
                    rose_fig = plot_power_rose(
                        rr["angles_deg"],
                        rr["total_power_kw"],
                        rr["total_no_wake_kw"],
                        rr["best_direction"],
                        rr["worst_direction"],
                        dir_freq_aligned,
                    )
                    st.plotly_chart(rose_fig, use_container_width=True)
                else:
                    st.info("💡 点击『逐度风向扫描』可获得完整的逐度玫瑰图")
                    dir_fig = plot_power_rose(
                        r["directions"],
                        r["direction_mean_power_kw"],
                        r["direction_mean_no_wake_kw"],
                        r["directions"][int(np.argmax(r["direction_mean_power_kw"]))],
                        r["directions"][int(np.argmax(r["direction_loss"]))],
                        r["frequency_matrix"].sum(axis=1),
                        title=f"{len(r['directions'])} 扇区风向-平均功率玫瑰图",
                    )
                    st.plotly_chart(dir_fig, use_container_width=True)

            with g2:
                st.subheader("🍰 各风向AEP贡献")
                pie_fig = plot_aep_direction_pie(
                    r["directions"], r["direction_aep_contrib_kwh"]
                )
                st.plotly_chart(pie_fig, use_container_width=True)

            st.subheader("📉 风机尾流损失分析")
            b1, b2 = st.columns([1.3, 1])
            with b1:
                loss_bar_fig = plot_wake_loss_bar(
                    st.session_state.farm_ids,
                    r["per_turbine_loss_avg"],
                    list(r["top5_affected_indices"]),
                    title="各风机加权平均尾流损失率 (降序)",
                )
                st.plotly_chart(loss_bar_fig, use_container_width=True)

            with b2:
                st.markdown("#### 🔴 受影响最严重的5台风机")
                top5 = list(r["top5_affected_indices"])
                top_rows = []
                for rank, idx in enumerate(top5, 1):
                    top_rows.append({
                        "排名": rank,
                        "风机": st.session_state.farm_ids[idx],
                        "机型": st.session_state.farm_models[idx],
                        "平均损失率": f"{r['per_turbine_loss_avg'][idx]*100:.2f} %",
                    })
                st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

                st.divider()
                st.markdown("#### 📊 全场损失汇总")
                df_sum = pd.DataFrame({
                    "指标": [
                        "全场整体尾流损失率",
                        "总风机数",
                        "平均单机损失率",
                        "最高单机损失率",
                    ],
                    "数值": [
                        f"{r['overall_loss']*100:.2f} %",
                        f"{len(st.session_state.farm_ids)}",
                        f"{np.mean(r['per_turbine_loss_avg'])*100:.2f} %",
                        f"{np.max(r['per_turbine_loss_avg'])*100:.2f} %",
                    ],
                })
                st.dataframe(df_sum, use_container_width=True, hide_index=True)

            st.subheader("🔥 综合尾流影响矩阵")
            st.plotly_chart(
                plot_loss_matrix_heatmap(r["loss_matrix"], st.session_state.farm_ids),
                use_container_width=True,
            )

            with st.expander("📋 详细数据表格: 每风机功率/损失明细"):
                detail_df = pd.DataFrame({
                    "风机编号": st.session_state.farm_ids,
                    "机型": st.session_state.farm_models,
                    "额定功率 (MW)": [
                        st.session_state.turbine_library[m]["rated_power"]
                        for m in st.session_state.farm_models
                    ],
                    "加权平均尾流损失 (%)": (r["per_turbine_loss_avg"] * 100).round(2),
                })
                st.dataframe(detail_df, use_container_width=True, hide_index=True)


# ============================================================
# Tab 6: 布局优化
# ============================================================
with tab_opt:
    st.header("🧬 布局优化 (遗传算法)")
    st.write("以最大化 AEP 为目标, 可指定哪些风机允许移动及移动边界")

    data_ready = (
        st.session_state.farm_coords is not None
        and st.session_state.wind_freq is not None
    )
    if not data_ready:
        st.warning("⚠️ 请先完成『风电场布局』和『风资源数据』配置, 并建议先跑一次AEP扫描")
    else:
        N = len(st.session_state.farm_ids)
        st.subheader("🎚️ 优化参数配置")

        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            pop_size = st.slider("种群规模", 10, 80, 30, 5)
        with c_p2:
            n_gen = st.slider("进化代数", 10, 200, 50, 10)
        with c_p3:
            crossover_r = st.slider("交叉概率", 0.4, 1.0, 0.8, 0.05)

        c_p4, c_p5, c_p6 = st.columns(3)
        with c_p4:
            mutation_r = st.slider("变异概率", 0.02, 0.40, 0.15, 0.01)
        with c_p5:
            mutation_step = st.slider("变异步长 (边界比例)", 0.02, 0.30, 0.10, 0.01)
        with c_p6:
            seed = st.number_input("随机种子", 0, 999, 42)

        st.divider()
        st.subheader("📍 风机可移动性配置")
        move_all = st.checkbox("允许所有风机移动 (均匀边界)", value=True)

        coords = st.session_state.farm_coords
        span_x = coords[:, 0].max() - coords[:, 0].min() + 1000
        span_y = coords[:, 1].max() - coords[:, 1].min() + 1000
        cx = (coords[:, 0].max() + coords[:, 0].min()) / 2
        cy = (coords[:, 1].max() + coords[:, 1].min()) / 2

        if move_all:
            bx1, bx2 = st.columns(2)
            with bx1:
                margin_x = st.slider("X方向扩展余量 (m)", 0.0, 5000.0, span_x / 2, 100.0)
            with bx2:
                margin_y = st.slider("Y方向扩展余量 (m)", 0.0, 5000.0, span_y / 2, 100.0)
            lb = np.tile(np.array([cx - span_x / 2 - margin_x, cy - span_y / 2 - margin_y]), (N, 1))
            ub = np.tile(np.array([cx + span_x / 2 + margin_x, cy + span_y / 2 + margin_y]), (N, 1))
            movable = np.ones(N, dtype=bool)
        else:
            movable = np.zeros(N, dtype=bool)
            lb = np.zeros((N, 2))
            ub = np.zeros((N, 2))
            for i, tid in enumerate(st.session_state.farm_ids):
                st.markdown(f"**{tid} ({st.session_state.farm_models[i]})**")
                c_m, c_lb_x, c_ub_x, c_lb_y, c_ub_y = st.columns(5)
                with c_m:
                    movable[i] = st.checkbox(f"可移动", key=f"mov_{i}", value=True)
                with c_lb_x:
                    lb[i, 0] = st.number_input("X下限", value=float(coords[i, 0] - 1000), key=f"lbx_{i}")
                with c_ub_x:
                    ub[i, 0] = st.number_input("X上限", value=float(coords[i, 0] + 1000), key=f"ubx_{i}")
                with c_lb_y:
                    lb[i, 1] = st.number_input("Y下限", value=float(coords[i, 1] - 1000), key=f"lby_{i}")
                with c_ub_y:
                    ub[i, 1] = st.number_input("Y上限", value=float(coords[i, 1] + 1000), key=f"uby_{i}")

        bounds = (lb, ub)
        st.caption(
            f"可移动风机: {np.sum(movable)} / {N} 台 | "
            f"搜索范围 X: [{lb[0,0]:.0f},{ub[0,0]:.0f}]m | Y: [{lb[0,1]:.0f},{ub[0,1]:.0f}]m"
        )

        run_opt = st.button("🧬 开始布局优化", type="primary",
                            disabled=np.sum(movable) == 0)

        if run_opt:
            progress_bar = st.progress(0.0, text="初始化种群...")
            status_box = st.empty()

            def _progress_cb(gen, info):
                total_gen = info["total_generations"]
                pct = (gen + 1) / max(total_gen, 1)
                progress_bar.progress(min(pct, 1.0))
                imp_pct = info.get("improvement_pct", 0.0)
                status_box.markdown(
                    f"**第 {gen}/{total_gen} 代** | "
                    f"当前最佳 AEP: {info.get('best_aep_kwh',0)/1e6:.3f} GWh | "
                    f"本代平均: {info.get('generation_avg_aep_kwh',0)/1e6:.3f} GWh | "
                    f"提升: {imp_pct:+.2f}%"
                )

            with st.spinner("遗传算法运行中..."):
                opt_result = genetic_algorithm_layout_optimize(
                    initial_coords=st.session_state.farm_coords,
                    turbine_library=st.session_state.turbine_library,
                    model_names=st.session_state.farm_models,
                    directions=st.session_state.wind_directions,
                    wind_speeds=st.session_state.wind_speeds,
                    frequency_matrix=st.session_state.wind_freq,
                    movable_mask=movable,
                    bounds=bounds,
                    pop_size=pop_size,
                    n_generations=n_gen,
                    crossover_rate=crossover_r,
                    mutation_rate=mutation_r,
                    mutation_step_frac=mutation_step,
                    wake_model=wake_model_code,
                    alpha=alpha,
                    ti=st.session_state.turbulence_intensity,
                    superposition=superposition_code,
                    min_spacing_multiple=spacing_multiple,
                    progress_callback=_progress_cb,
                    seed=seed,
                )
                st.session_state.opt_results = opt_result

            progress_bar.progress(1.0)
            status_box.success("✅ 优化完成!")
            st.success(
                f"AEP 从 {opt_result['initial_aep_kwh']/1e6:.3f} GWh → "
                f"{opt_result['best_aep_kwh']/1e6:.3f} GWh, "
                f"提升 {opt_result['aep_improvement_pct']:+.2f}%"
            )

        if st.session_state.opt_results is not None:
            opt = st.session_state.opt_results
            st.divider()
            st.subheader("📊 优化结果")

            kk1, kk2, kk3 = st.columns(3)
            kk1.metric("优化前 AEP", f"{opt['initial_aep_kwh']/1e6:.3f} GWh")
            kk2.metric("优化后 AEP", f"{opt['best_aep_kwh']/1e6:.3f} GWh")
            kk3.metric("AEP 提升", f"{opt['aep_improvement_pct']:+.2f} %",
                       delta=f"{(opt['best_aep_kwh']-opt['initial_aep_kwh'])/1e6:+.3f} GWh")

            r1, r2 = st.columns([1, 1])
            with r1:
                conv_fig = plot_optimization_convergence(
                    opt["history_best_aep"],
                    opt["history_avg_aep"],
                    opt["initial_aep_kwh"],
                    opt["best_aep_kwh"],
                )
                st.plotly_chart(conv_fig, use_container_width=True)

            with r2:
                cmp_fig = plot_layout_comparison(
                    opt["initial_coords"],
                    opt["best_coords"],
                    st.session_state.farm_ids,
                    st.session_state.farm_models,
                    st.session_state.turbine_library,
                    movable,
                )
                st.plotly_chart(cmp_fig, use_container_width=True)

            with st.expander("📥 推荐布局坐标 (可下载CSV)"):
                df_rec = pd.DataFrame({
                    "风机编号": st.session_state.farm_ids,
                    "原X (m)": opt["initial_coords"][:, 0].round(1),
                    "原Y (m)": opt["initial_coords"][:, 1].round(1),
                    "推荐X (m)": opt["best_coords"][:, 0].round(1),
                    "推荐Y (m)": opt["best_coords"][:, 1].round(1),
                    "ΔX (m)": (opt["best_coords"][:, 0] - opt["initial_coords"][:, 0]).round(1),
                    "ΔY (m)": (opt["best_coords"][:, 1] - opt["initial_coords"][:, 1]).round(1),
                    "机型": st.session_state.farm_models,
                    "可移动": movable,
                })
                st.dataframe(df_rec, use_container_width=True, hide_index=True)

                csv_buf = io.StringIO()
                df_rec.to_csv(csv_buf, index=False, encoding="utf-8-sig")
                st.download_button(
                    "💾 下载推荐布局CSV",
                    data=csv_buf.getvalue(),
                    file_name="optimized_layout.csv",
                    mime="text/csv",
                )


# ============================================================
# Tab 0: 总览面板 (把最关键的汇总信息放最前面)
# ============================================================
with tab_overview:
    st.title("🌬️ 风电场尾流效应分析与发电功率预测系统")
    st.markdown(
        """
        **面向风电运营人员** — 分析风机之间的互相遮挡影响，估算全场年发电量。
        
        ### 快速开始
        1. **⚙️ 机型管理** — 选择内置机型或自定义上传
        2. **📍 风电场布局** — 上传CSV (风机编号, X, Y, 型号) 或加载示例
        3. **🌬️ 风资源数据** — 生成/上传风向风速频率表
        4. **🧭 单风向分析** — 指定风向风速查看尾流细节
        5. **📈 AEP与损失分析** — 全场扫描, 计算AEP和尾流损失
        6. **🧬 布局优化** — 遗传算法优化风机位置, 最大化AEP
        """
    )

    st.divider()

    has_layout = st.session_state.farm_coords is not None
    has_wind = st.session_state.wind_freq is not None
    has_scan = st.session_state.scanning_results is not None
    has_opt = st.session_state.opt_results is not None

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("✅ 布局已配置", "是" if has_layout else "否")
    s2.metric("✅ 风资源已配置", "是" if has_wind else "否")
    s3.metric("✅ AEP扫描完成", "是" if has_scan else "否")
    s4.metric("✅ 布局优化完成", "是" if has_opt else "否")

    if has_layout and has_wind:
        st.divider()
        st.subheader("🏃 快速行动")
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("📊 一键运行全场AEP扫描", type="primary", disabled=has_scan):
                with st.spinner("全场扫描计算中..."):
                    result = scan_all_directions_windspeeds(
                        st.session_state.farm_coords,
                        st.session_state.turbine_library,
                        st.session_state.farm_models,
                        st.session_state.wind_directions,
                        st.session_state.wind_speeds,
                        st.session_state.wind_freq,
                        wake_model_code, alpha,
                        st.session_state.turbulence_intensity,
                        superposition_code,
                    )
                    st.session_state.scanning_results = result
                st.success("扫描完成, 请前往『📈 AEP与损失分析』查看详情")
                st.rerun()

        with qc2:
            if st.button("🌹 一键生成风向玫瑰图", type="secondary"):
                with st.spinner("逐度扫描中..."):
                    rr = scan_direction_rose(
                        st.session_state.farm_coords,
                        st.session_state.turbine_library,
                        st.session_state.farm_models,
                        mean_wind_speed_from_frequency(st.session_state.wind_speeds, st.session_state.wind_freq),
                        wake_model_code, alpha,
                        st.session_state.turbulence_intensity,
                        superposition_code,
                    )
                    st.session_state.rose_results = rr
                st.success("扫描完成")

    if has_scan:
        r = st.session_state.scanning_results
        st.divider()
        st.subheader("🎯 关键指标总览")
        kk1, kk2, kk3, kk4, kk5 = st.columns(5)
        kk1.metric("AEP", f"{r['aep_kwh']/1e6:.3f} GWh")
        kk2.metric("无尾流 AEP", f"{r['aep_no_wake_kwh']/1e6:.3f} GWh")
        kk3.metric("尾流损失率", f"{r['overall_loss']*100:.2f} %")
        kk4.metric("容量系数", f"{r['capacity_factor']*100:.2f} %")
        kk5.metric("额定总功率", f"{r['total_rated_kw']/1000:.1f} MW")

        st.subheader("🗺️ 风场俯视图 (按损失着色)")
        layout_loss_fig = plot_farm_layout(
            st.session_state.farm_coords,
            st.session_state.farm_ids,
            st.session_state.farm_models,
            st.session_state.turbine_library,
            per_turbine_loss=r["per_turbine_loss_avg"],
            wind_direction=None,
            show_wake_cones=False,
            highlight_turbines=list(r["top5_affected_indices"]),
            title="风机平均尾流损失 (绿→黄→红, 红框标注前5受影响风机)",
        )
        st.plotly_chart(layout_loss_fig, use_container_width=True)
