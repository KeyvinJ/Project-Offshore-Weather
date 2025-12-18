"""
Phase 5E: Integrated Risk Dashboard
Streamlit Interactive Application

Combines all Phase 5 analyses:
- Phase 5A: Historical Delay Prediction
- Phase 5B: Seasonal & Directional Analysis
- Phase 5C: Risk-Based Scheduling
- Phase 5D: Weather Windows & Forecasting

Run with: streamlit run app/phase5e_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Phase 5E: Offshore Operations Risk Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data
def load_phase5_data():
    """Load all Phase 5 results"""
    base_path = Path(__file__).parent.parent / 'data' / 'processed'

    data = {}

    # Phase 5A
    try:
        with open(base_path / 'phase5a' / 'delay_prediction_results.pkl', 'rb') as f:
            data['phase5a'] = pickle.load(f)
    except:
        data['phase5a'] = None

    # Phase 5B1
    try:
        with open(base_path / 'phase5b1' / 'seasonal_eva_results.pkl', 'rb') as f:
            data['phase5b1'] = pickle.load(f)
    except:
        data['phase5b1'] = None

    # Phase 5B2
    try:
        with open(base_path / 'phase5b2' / 'seasonal_copulas.pkl', 'rb') as f:
            data['phase5b2'] = pickle.load(f)
    except:
        data['phase5b2'] = None

    # Phase 5D1
    try:
        with open(base_path / 'phase5d1' / 'weather_window_results.pkl', 'rb') as f:
            data['phase5d1'] = pickle.load(f)
    except:
        data['phase5d1'] = None

    # Phase 5D2
    try:
        with open(base_path / 'phase5d2' / 'extreme_risk_results.pkl', 'rb') as f:
            data['phase5d2'] = pickle.load(f)
    except:
        data['phase5d2'] = None

    # Phase 5D3
    try:
        with open(base_path / 'phase5d3' / 'operational_forecast_results.pkl', 'rb') as f:
            data['phase5d3'] = pickle.load(f)
    except:
        data['phase5d3'] = None

    return data

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">🌊 Offshore Operations Risk Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Phase 5E: Integrated Risk Assessment & Decision Support**")
    st.markdown("*Location: UK Northeast Coast | Data: 2015-2025 (10 years)*")

    st.markdown("---")

    # Load data
    with st.spinner("Loading Phase 5 analysis results..."):
        data = load_phase5_data()

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Analysis:",
        [
            "🏠 Executive Summary",
            "📅 Monthly Recommendations",
            "🌤️ Weather Windows (5D1)",
            "⚠️ Extreme Risk (5D2)",
            "🎯 Optimal Scheduling (5D3)",
            "📈 Seasonal Analysis (5B)",
            "⏱️ Project Delays (5A)",
            "📄 Generate Report"
        ]
    )

    # Location disclaimer
    st.sidebar.markdown("---")
    st.sidebar.warning("⚠️ **Location-Specific Analysis**\n\nThese results are for **UK Northeast Coast** only. Different locations require re-analysis.")

    # Page routing
    if page == "🏠 Executive Summary":
        show_executive_summary(data)
    elif page == "📅 Monthly Recommendations":
        show_monthly_recommendations(data)
    elif page == "🌤️ Weather Windows (5D1)":
        show_weather_windows(data)
    elif page == "⚠️ Extreme Risk (5D2)":
        show_extreme_risk(data)
    elif page == "🎯 Optimal Scheduling (5D3)":
        show_optimal_scheduling(data)
    elif page == "📈 Seasonal Analysis (5B)":
        show_seasonal_analysis(data)
    elif page == "⏱️ Project Delays (5A)":
        show_project_delays(data)
    elif page == "📄 Generate Report":
        show_report_generator(data)


def show_executive_summary(data):
    """Executive Summary Dashboard"""
    st.markdown('<div class="sub-header">Executive Summary</div>', unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    if data['phase5d3']:
        combined_scores = data['phase5d3']['combined_scores']
        best_month = combined_scores.loc[combined_scores['Combined_Score'].idxmax()]
        worst_month = combined_scores.loc[combined_scores['Combined_Score'].idxmin()]

        with col1:
            st.metric(
                "🏆 Best Month",
                best_month['Month'],
                f"Score: {best_month['Combined_Score']:.2f}"
            )

        with col2:
            st.metric(
                "📊 5-Day Workability",
                f"{best_month['Workability_5d']*100:.1f}%",
                "Best month"
            )

        with col3:
            st.metric(
                "🛡️ Safety Score",
                f"{best_month['Safety_30d']*100:.1f}%",
                "Low extreme risk"
            )

        with col4:
            st.metric(
                "❌ Worst Month",
                worst_month['Month'],
                f"Score: {worst_month['Combined_Score']:.2f}",
                delta_color="inverse"
            )

    st.markdown("---")

    # Quick insights
    st.markdown("### 🎯 Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("**✅ OPTIMAL SCHEDULING MONTHS**")
        if data['phase5d3']:
            optimal = combined_scores[combined_scores['Combined_Score'] > 0.6].sort_values('Combined_Score', ascending=False)
            if len(optimal) > 0:
                for idx, row in optimal.iterrows():
                    st.write(f"- **{row['Month']}**: Score {row['Combined_Score']:.2f}")
            else:
                st.write("No months with score > 0.6")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="danger-box">', unsafe_allow_html=True)
        st.markdown("**❌ MONTHS TO AVOID**")
        if data['phase5d3']:
            avoid = combined_scores[combined_scores['Combined_Score'] < 0.4].sort_values('Combined_Score')
            if len(avoid) > 0:
                for idx, row in avoid.iterrows():
                    st.write(f"- **{row['Month']}**: Score {row['Combined_Score']:.2f}")
            else:
                st.write("All months above 0.4 threshold")
        st.markdown('</div>', unsafe_allow_html=True)

    # Seasonal comparison
    st.markdown("### 📊 Seasonal Comparison")

    if data['phase5d2']:
        seasonal_extremes = data['phase5d2']['seasonal_extremes']

        fig = go.Figure()

        seasons = seasonal_extremes['Season'].tolist()
        hs6_freq = (seasonal_extremes['Hs>6.0m'] * 100).tolist()

        fig.add_trace(go.Bar(
            x=seasons,
            y=hs6_freq,
            marker_color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
            text=[f"{v:.1f}%" for v in hs6_freq],
            textposition='outside'
        ))

        fig.update_layout(
            title="Extreme Event Frequency by Season (Hs>6m)",
            xaxis_title="Season",
            yaxis_title="% of Hours",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Phase 5 modules summary
    st.markdown("### 📦 Analysis Modules")

    modules = [
        {"name": "5A: Project Delays", "status": data['phase5a'] is not None, "desc": "Historical delay prediction"},
        {"name": "5B: Seasonal Analysis", "status": data['phase5b1'] is not None, "desc": "EVA & copula by season"},
        {"name": "5D1: Weather Windows", "status": data['phase5d1'] is not None, "desc": "Copula-enhanced probabilities"},
        {"name": "5D2: Extreme Risk", "status": data['phase5d2'] is not None, "desc": "Return period analysis"},
        {"name": "5D3: Forecasting", "status": data['phase5d3'] is not None, "desc": "Optimal scheduling"}
    ]

    col1, col2 = st.columns(2)
    for i, module in enumerate(modules):
        with (col1 if i % 2 == 0 else col2):
            status_icon = "✅" if module['status'] else "❌"
            st.markdown(f"{status_icon} **{module['name']}**: {module['desc']}")


def show_monthly_recommendations(data):
    """Monthly Recommendations Page"""
    st.markdown('<div class="sub-header">📅 Monthly Recommendations</div>', unsafe_allow_html=True)

    if not data['phase5d3']:
        st.error("Phase 5D3 data not available")
        return

    combined_scores = data['phase5d3']['combined_scores']

    # Month selector
    selected_month = st.selectbox(
        "Select Month:",
        combined_scores['Month'].tolist()
    )

    month_data = combined_scores[combined_scores['Month'] == selected_month].iloc[0]

    st.markdown("---")

    # Month overview
    col1, col2, col3 = st.columns(3)

    with col1:
        score = month_data['Combined_Score']
        if score > 0.6:
            st.success(f"**Combined Score: {score:.3f}**\n\n✅ GOOD MONTH")
        elif score > 0.4:
            st.warning(f"**Combined Score: {score:.3f}**\n\n⚠️ MODERATE")
        else:
            st.error(f"**Combined Score: {score:.3f}**\n\n❌ POOR MONTH")

    with col2:
        st.metric("5-Day Workability", f"{month_data['Workability_5d']*100:.1f}%")
        st.metric("Persistence", f"{month_data['Persistence']:.3f}")

    with col3:
        st.metric("30-Day Safety", f"{month_data['Safety_30d']*100:.1f}%")
        st.metric("Historical Windows", f"{month_data['Historical_Windows']}")

    # Detailed breakdown
    st.markdown("### 📊 Detailed Breakdown")

    # Radar chart
    categories = ['Workability', 'Safety', 'Persistence']
    values = [
        month_data['Workability_5d'],
        month_data['Safety_30d'],
        month_data['Persistence']
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=selected_month
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=f"{selected_month} Score Components"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.markdown("### 💡 Recommendations")

    if score > 0.6:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **{selected_month} is EXCELLENT for offshore operations:**
        - High probability of 5-day continuous weather windows ({month_data['Workability_5d']*100:.1f}%)
        - Low extreme event risk ({(1-month_data['Safety_30d'])*100:.1f}% chance of Hs>6m in 30 days)
        - Weather conditions are stable (persistence: {month_data['Persistence']:.3f})
        - Historically, {month_data['Historical_Windows']} favorable 5-day windows occurred in this month

        **✅ Recommended for: Critical operations, tight schedules, high-value projects**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    elif score > 0.4:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **{selected_month} is MODERATE for offshore operations:**
        - Moderate weather window availability
        - Some extreme event risk
        - Plan for contingencies

        **⚠️ Recommended for: Flexible operations, with backup dates**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="danger-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **{selected_month} is POOR for offshore operations:**
        - Low probability of continuous weather windows
        - High extreme event risk
        - Rapidly changing conditions

        **❌ NOT RECOMMENDED unless absolutely necessary**
        """)
        st.markdown('</div>', unsafe_allow_html=True)


def show_weather_windows(data):
    """Weather Windows Analysis (Phase 5D1)"""
    st.markdown('<div class="sub-header">🌤️ Weather Window Probabilities (Phase 5D1)</div>', unsafe_allow_html=True)

    if not data['phase5d1']:
        st.error("Phase 5D1 data not available")
        return

    df_windows = data['phase5d1']['continuous_window_probabilities']

    # Duration selector
    duration = st.select_slider(
        "Select Window Duration:",
        options=[1, 3, 5, 7, 10],
        value=5,
        format_func=lambda x: f"{x} days"
    )

    col_name = f'{duration}d_Copula'

    # Bar chart
    fig = px.bar(
        df_windows,
        x='Month',
        y=col_name,
        title=f'{duration}-Day Continuous Weather Window Probability',
        labels={col_name: 'Probability'},
        color=col_name,
        color_continuous_scale='RdYlGn'
    )

    fig.update_yaxis(tickformat='.1%')
    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("### 📋 Detailed Probabilities")

    display_df = df_windows[['Month', '1d_Copula', '3d_Copula', '5d_Copula', '7d_Copula', '10d_Copula']].copy()

    for col in display_df.columns:
        if col != 'Month':
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")

    display_df.columns = ['Month', '1-day', '3-day', '5-day', '7-day', '10-day']

    st.dataframe(display_df, use_container_width=True)


def show_extreme_risk(data):
    """Extreme Risk Analysis (Phase 5D2)"""
    st.markdown('<div class="sub-header">⚠️ Extreme Event Risk (Phase 5D2)</div>', unsafe_allow_html=True)

    if not data['phase5d2']:
        st.error("Phase 5D2 data not available")
        return

    df_risk = data['phase5d2']['project_risk']

    # Project duration selector
    duration = st.select_slider(
        "Select Project Duration:",
        options=[7, 14, 30, 60, 90],
        value=30,
        format_func=lambda x: f"{x} days"
    )

    col_name = f'{duration}d_Hs>6m'

    # Create risk categories
    df_risk['Risk_Category'] = df_risk[col_name].apply(
        lambda x: 'Low (<10%)' if x < 0.10 else 'Medium (10-20%)' if x < 0.20 else 'High (>20%)'
    )

    # Bar chart with risk zones
    fig = go.Figure()

    colors = ['green' if x < 0.10 else 'orange' if x < 0.20 else 'red' for x in df_risk[col_name]]

    fig.add_trace(go.Bar(
        x=df_risk['Month'],
        y=df_risk[col_name] * 100,
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in df_risk[col_name]],
        textposition='outside'
    ))

    # Risk threshold lines
    fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="Low Risk Threshold (10%)")
    fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="High Risk Threshold (20%)")

    fig.update_layout(
        title=f'Probability of Encountering Hs>6m During {duration}-Day Project',
        xaxis_title='Month',
        yaxis_title='Probability (%)',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Risk summary
    st.markdown("### 📊 Risk Summary")

    low_risk = df_risk[df_risk[col_name] < 0.10]['Month'].tolist()
    med_risk = df_risk[(df_risk[col_name] >= 0.10) & (df_risk[col_name] < 0.20)]['Month'].tolist()
    high_risk = df_risk[df_risk[col_name] >= 0.20]['Month'].tolist()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success(f"**Low Risk (<10%)**\n\n{len(low_risk)} months")
        if low_risk:
            st.write(", ".join(low_risk))

    with col2:
        st.warning(f"**Medium Risk (10-20%)**\n\n{len(med_risk)} months")
        if med_risk:
            st.write(", ".join(med_risk))

    with col3:
        st.error(f"**High Risk (>20%)**\n\n{len(high_risk)} months")
        if high_risk:
            st.write(", ".join(high_risk))


def show_optimal_scheduling(data):
    """Optimal Scheduling (Phase 5D3)"""
    st.markdown('<div class="sub-header">🎯 Optimal Scheduling (Phase 5D3)</div>', unsafe_allow_html=True)

    if not data['phase5d3']:
        st.error("Phase 5D3 data not available")
        return

    combined_scores = data['phase5d3']['combined_scores']

    # Ranking table
    st.markdown("### 🏆 Monthly Rankings")

    ranked = combined_scores.sort_values('Combined_Score', ascending=False).copy()
    ranked['Rank'] = range(1, len(ranked) + 1)

    # Format for display
    display_cols = ['Rank', 'Month', 'Combined_Score', 'Workability_5d', 'Safety_30d', 'Persistence', 'Historical_Windows']
    display_df = ranked[display_cols].copy()

    # Apply styling
    def color_rank(val):
        if val <= 3:
            return 'background-color: #d4edda'
        elif val >= 10:
            return 'background-color: #f8d7da'
        return ''

    styled_df = display_df.style.applymap(color_rank, subset=['Rank'])

    st.dataframe(styled_df, use_container_width=True)

    # Comparison tool
    st.markdown("### 🔄 Compare Months")

    col1, col2 = st.columns(2)

    with col1:
        month1 = st.selectbox("Select First Month:", combined_scores['Month'].tolist(), key='month1')

    with col2:
        month2 = st.selectbox("Select Second Month:", combined_scores['Month'].tolist(), index=1, key='month2')

    if month1 != month2:
        data1 = combined_scores[combined_scores['Month'] == month1].iloc[0]
        data2 = combined_scores[combined_scores['Month'] == month2].iloc[0]

        # Comparison chart
        categories = ['Workability', 'Safety', 'Persistence', 'Combined Score']

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name=month1,
            x=categories,
            y=[data1['Workability_5d'], data1['Safety_30d'], data1['Persistence'], data1['Combined_Score']]
        ))

        fig.add_trace(go.Bar(
            name=month2,
            x=categories,
            y=[data2['Workability_5d'], data2['Safety_30d'], data2['Persistence'], data2['Combined_Score']]
        ))

        fig.update_layout(
            title=f'{month1} vs {month2} Comparison',
            barmode='group',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


def show_seasonal_analysis(data):
    """Seasonal Analysis (Phase 5B)"""
    st.markdown('<div class="sub-header">📈 Seasonal Analysis (Phase 5B)</div>', unsafe_allow_html=True)

    st.info("Phase 5B includes seasonal EVA, copulas, directional, and tidal analysis")

    if data['phase5b2']:
        st.markdown("### 🔗 Seasonal Dependence (Kendall's τ)")

        seasonal_dep = data['phase5b2']['seasonal_dependence']

        seasons = list(seasonal_dep.keys())
        tau_hw = [seasonal_dep[s]['hs_wind']['kendall_tau'] for s in seasons]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=seasons,
            y=tau_hw,
            text=[f"{v:.3f}" for v in tau_hw],
            textposition='outside',
            marker_color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        ))

        fig.update_layout(
            title="Seasonal Hs-Wind Dependence (Kendall's τ)",
            xaxis_title="Season",
            yaxis_title="Kendall's τ",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Key Finding from Phase 5B2:**
        - Winter has WEAKEST Hs-Wind coupling (τ=0.26) → Atlantic swell decoupled from local wind
        - Autumn has STRONGEST coupling (τ=0.65) → Local storm generation
        """)


def show_project_delays(data):
    """Project Delays (Phase 5A)"""
    st.markdown('<div class="sub-header">⏱️ Project Delay Prediction (Phase 5A)</div>', unsafe_allow_html=True)

    st.info("Phase 5A provides historical delay prediction using bootstrap resampling")

    st.markdown("""
    ### Methodology
    - **Bootstrap Resampling**: Preserves autocorrelation
    - **Copula-Based Monte Carlo**: Captures Hs-Wind dependence
    - **Operation-Specific**: Diving, ROV, Crane operations

    ### Typical Results
    For a 30 operational-day project:
    - **P50 (Median)**: ~44 calendar days
    - **P90 (Conservative)**: ~51 calendar days
    - **Naive Independence**: Underestimates by 10-15%
    """)


def show_report_generator(data):
    """Report Generator"""
    st.markdown('<div class="sub-header">📄 Generate Report</div>', unsafe_allow_html=True)

    st.markdown("### Report Configuration")

    report_type = st.radio(
        "Select Report Type:",
        ["Executive Summary", "Technical Report", "Monthly Analysis"]
    )

    include_sections = st.multiselect(
        "Include Sections:",
        [
            "Weather Windows (5D1)",
            "Extreme Risk (5D2)",
            "Optimal Scheduling (5D3)",
            "Seasonal Analysis (5B)",
            "Project Delays (5A)"
        ],
        default=["Weather Windows (5D1)", "Extreme Risk (5D2)", "Optimal Scheduling (5D3)"]
    )

    if st.button("Generate Report", type="primary"):
        st.success("Report generation functionality would export to PDF here")
        st.info("Implementation requires additional libraries (reportlab, weasyprint, etc.)")


if __name__ == "__main__":
    main()
