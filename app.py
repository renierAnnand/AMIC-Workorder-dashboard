"""
AMIC Work Order Management & Analytics Dashboard System
========================================================
Comprehensive dashboard suite for maintenance operations analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AMIC Dashboard Suite",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .dashboard-title {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-left: 5px solid #1f77b4;
        background: linear-gradient(90deg, #f0f8ff 0%, #ffffff 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #1f77b4 0%, transparent 100%);
        margin: 2rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .alert-critical {
        background-color: #fee;
        border-color: #f44;
        color: #c00;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    .alert-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Data loading function with caching
@st.cache_data(ttl=3600)
def load_data(file_path):
    """Load and preprocess work order data"""
    df = pd.read_excel(file_path)
    
    # Convert dates
    df['Date Created'] = pd.to_datetime(df['Date Created'])
    df['Completion Date'] = pd.to_datetime(df['Completion Date'])
    
    # Calculate additional metrics
    df['Turnaround Time (Days)'] = (df['Completion Date'] - df['Date Created']).dt.days
    df['Days Open'] = (datetime.now() - df['Date Created']).dt.days
    
    # Fill missing values for analysis
    df['Downtime Hours'] = df['Downtime Hours'].fillna(0)
    df['Total Cost'] = df['Parts Cost'] + (df['Labor Hours'] * 150)  # Assuming $150/hour
    
    # Create month/year columns for trending
    df['Month'] = df['Date Created'].dt.to_period('M').astype(str)
    df['Week'] = df['Date Created'].dt.to_period('W').astype(str)
    
    return df

# Generate simulated CCR data
@st.cache_data
def generate_ccr_data():
    """Generate sample Catalogue Change Request data"""
    np.random.seed(42)
    ccr_data = {
        'CCR_ID': [f'CCR-{i:04d}' for i in range(1, 101)],
        'Submission_Date': pd.date_range(start='2024-01-01', periods=100, freq='3D'),
        'Status': np.random.choice(['Approved', 'Pending', 'Rejected'], 100, p=[0.6, 0.25, 0.15]),
        'Request_Type': np.random.choice(['New Failure Mode', 'Update Existing', 'Remove Obsolete'], 100),
        'Site': np.random.choice(['Riyadh_Main', 'Jeddah_South', 'Dammam_East'], 100),
        'Submitted_By': np.random.choice(['Supervisor_A', 'Supervisor_B', 'Supervisor_C', 'Supervisor_D'], 100),
    }
    ccr_df = pd.DataFrame(ccr_data)
    ccr_df['Approval_Date'] = ccr_df.apply(
        lambda x: x['Submission_Date'] + timedelta(days=np.random.randint(1, 30)) if x['Status'] == 'Approved' else pd.NaT,
        axis=1
    )
    ccr_df['Approval_Time_Days'] = (ccr_df['Approval_Date'] - ccr_df['Submission_Date']).dt.days
    return ccr_df

def create_gauge_chart(value, title, max_value=100, threshold_good=80, threshold_warning=60):
    """Create a gauge chart for KPIs"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': threshold_good},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold_warning], 'color': "lightcoral"},
                {'range': [threshold_warning, threshold_good], 'color': "lightyellow"},
                {'range': [threshold_good, max_value], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_good
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==================== DATA IMPORT SCREEN ====================
def data_import_screen():
    """Initial screen for data import"""
    st.markdown('<div class="main-header">🔧 AMIC Work Order Management & Analytics Dashboard Suite</div>', 
                unsafe_allow_html=True)
    
    # Center content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h2 style='color: #1f77b4;'>📁 Import Work Order Data</h2>
            <p style='font-size: 1.1rem; color: #555;'>
                Upload your Excel file to begin analyzing maintenance operations
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # File uploader with prominent styling
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file containing work order data",
            key="data_upload"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: **{uploaded_file.name}**")
            
            # Show file info
            st.info(f"📊 File size: {uploaded_file.size / 1024:.2f} KB")
            
            # Preview button
            if st.button("🔍 Preview & Load Data", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Loading and processing data..."):
                        # Save to temp location
                        data_path = f"/tmp/{uploaded_file.name}"
                        with open(data_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Load data
                        df = load_data(data_path)
                        
                        # Store in session state
                        st.session_state['data_loaded'] = True
                        st.session_state['df'] = df
                        st.session_state['data_path'] = data_path
                        st.session_state['ccr_df'] = generate_ccr_data()
                        
                        # Show preview
                        st.markdown("### 📋 Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Show summary stats
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Total Records", f"{len(df):,}")
                        with col_b:
                            st.metric("Columns", f"{len(df.columns)}")
                        with col_c:
                            st.metric("Date Range", f"{df['Date Created'].min().strftime('%Y-%m-%d')} to {df['Date Created'].max().strftime('%Y-%m-%d')}")
                        with col_d:
                            st.metric("Workshops", f"{df['Workshop'].nunique()}")
                        
                        st.success("✅ Data loaded successfully! Use the sidebar to navigate dashboards.")
                        st.balloons()
                        
                        # Force rerun to show dashboards
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    st.info("Please ensure the file contains the required columns and is properly formatted.")
        
        else:
            st.info("👆 Please upload an Excel file to continue")
            
            # Show expected format
            with st.expander("📖 Expected File Format"):
                st.markdown("""
                Your Excel file should contain the following columns:
                
                **Required Columns:**
                - Work Order ID
                - Date Created
                - Completion Date
                - Status (Open, In Progress, Completed, Closed)
                - Workshop/Site
                - Vehicle ID
                - System
                - Subsystem
                - Component
                - Failure Mode
                - Cause
                - Recommended Action
                - Assigned To
                - Created By
                - Labor Hours
                - Parts Cost
                - Downtime Hours
                
                **Sample Data:** Upload the provided demo file or use your own formatted data.
                """)

# ==================== DASHBOARD 1: EXECUTIVE OVERVIEW ====================
def executive_overview_dashboard(df):
    """Dashboard 1: Executive Overview Dashboard"""
    st.markdown('<div class="dashboard-title">📊 Executive Overview Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*High-level operational performance across all maintenance sites*")
    
    # Top KPIs Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_orders = len(df)
        st.metric("Total Work Orders", f"{total_orders:,}", help="All work orders in the system")
    
    with col2:
        open_orders = len(df[df['Status'] == 'Open'])
        st.metric("Open Orders", f"{open_orders:,}", delta=f"{(open_orders/total_orders*100):.1f}%")
    
    with col3:
        in_progress = len(df[df['Status'] == 'In Progress'])
        st.metric("In Progress", f"{in_progress:,}", delta=f"{(in_progress/total_orders*100):.1f}%")
    
    with col4:
        completed = len(df[df['Status'].isin(['Completed', 'Closed'])])
        st.metric("Completed", f"{completed:,}", delta=f"{(completed/total_orders*100):.1f}%")
    
    with col5:
        avg_completion = df[df['Turnaround Time (Days)'].notna()]['Turnaround Time (Days)'].mean()
        st.metric("Avg Completion Time", f"{avg_completion:.1f} days")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Work Order Status Distribution")
        status_counts = df['Status'].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_status.update_traces(textposition='inside', textinfo='percent+label')
        fig_status.update_layout(height=350)
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        st.subheader("Work Order Backlog Trends")
        backlog_trend = df[df['Status'].isin(['Open', 'In Progress'])].groupby('Month').size().reset_index(name='Backlog')
        fig_backlog = px.line(
            backlog_trend,
            x='Month',
            y='Backlog',
            markers=True,
            title="Monthly Backlog Evolution"
        )
        fig_backlog.update_traces(line_color='#e74c3c', line_width=3)
        fig_backlog.update_layout(height=350)
        st.plotly_chart(fig_backlog, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Site Performance Comparison")
        site_performance = df.groupby('Workshop').agg({
            'Work Order ID': 'count',
            'Turnaround Time (Days)': 'mean'
        }).round(1).reset_index()
        site_performance.columns = ['Workshop', 'Total Orders', 'Avg Completion Time']
        
        fig_site = make_subplots(specs=[[{"secondary_y": True}]])
        fig_site.add_trace(
            go.Bar(name='Total Orders', x=site_performance['Workshop'], y=site_performance['Total Orders'],
                   marker_color='lightblue'),
            secondary_y=False
        )
        fig_site.add_trace(
            go.Scatter(name='Avg Completion Time', x=site_performance['Workshop'],
                      y=site_performance['Avg Completion Time'], mode='lines+markers',
                      marker=dict(size=10, color='red'), line=dict(width=3)),
            secondary_y=True
        )
        fig_site.update_xaxes(title_text="Workshop")
        fig_site.update_yaxes(title_text="Total Orders", secondary_y=False)
        fig_site.update_yaxes(title_text="Avg Days", secondary_y=True)
        fig_site.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_site, use_container_width=True)
    
    with col2:
        st.subheader("Fault Distribution by System")
        system_faults = df['System'].value_counts().head(10)
        fig_systems = px.bar(
            x=system_faults.values,
            y=system_faults.index,
            orientation='h',
            color=system_faults.values,
            color_continuous_scale='Blues',
            labels={'x': 'Number of Failures', 'y': 'System'}
        )
        fig_systems.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_systems, use_container_width=True)
    
    # Technician Rankings
    st.subheader("Technician Workload & Productivity Rankings")
    tech_stats = df.groupby('Assigned To').agg({
        'Work Order ID': 'count',
        'Turnaround Time (Days)': 'mean',
        'Labor Hours': 'sum'
    }).round(2).reset_index()
    tech_stats.columns = ['Technician', 'Orders Handled', 'Avg Completion Time', 'Total Labor Hours']
    tech_stats = tech_stats.sort_values('Orders Handled', ascending=False)
    tech_stats['Efficiency Score'] = ((tech_stats['Orders Handled'] / tech_stats['Avg Completion Time']) * 10).round(1)
    
    st.dataframe(
        tech_stats.style.background_gradient(subset=['Orders Handled', 'Efficiency Score'], cmap='RdYlGn'),
        use_container_width=True,
        height=250
    )
    
    # SLA Alerts
    st.subheader("⚠️ Alerts & SLA Breaches")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overdue_orders = len(df[(df['Status'].isin(['Open', 'In Progress'])) & (df['Days Open'] > 14)])
        if overdue_orders > 0:
            st.markdown(f"""
            <div class="alert-box alert-critical">
                <strong>Critical:</strong> {overdue_orders} orders overdue (>14 days)
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        high_cost_orders = len(df[df['Total Cost'] > 5000])
        st.markdown(f"""
        <div class="alert-box alert-warning">
            <strong>Warning:</strong> {high_cost_orders} high-cost orders (>$5,000)
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        recent_completions = len(df[(df['Status'].isin(['Completed', 'Closed'])) & (df['Turnaround Time (Days)'] <= 7)])
        st.markdown(f"""
        <div class="alert-box alert-success">
            <strong>Success:</strong> {recent_completions} orders completed within SLA (≤7 days)
        </div>
        """, unsafe_allow_html=True)

# ==================== DASHBOARD 2: SITE PERFORMANCE ====================
def site_performance_dashboard(df):
    """Dashboard 2: Site Performance Dashboard"""
    st.markdown('<div class="dashboard-title">🏭 Site Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Monitor day-to-day performance and identify bottlenecks*")
    
    # Site selector
    sites = ['All Sites'] + sorted(df['Workshop'].unique().tolist())
    selected_site = st.selectbox("Select Workshop", sites)
    
    if selected_site != 'All Sites':
        df_filtered = df[df['Workshop'] == selected_site]
    else:
        df_filtered = df.copy()
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        daily_creation = df_filtered.groupby(df_filtered['Date Created'].dt.date).size().mean()
        st.metric("Avg Daily Work Orders Created", f"{daily_creation:.1f}")
    
    with col2:
        daily_closure = df_filtered[df_filtered['Status'].isin(['Completed', 'Closed'])].groupby(
            df_filtered[df_filtered['Status'].isin(['Completed', 'Closed'])]['Completion Date'].dt.date
        ).size().mean()
        st.metric("Avg Daily Closures", f"{daily_closure:.1f}")
    
    with col3:
        avg_turnaround = df_filtered['Turnaround Time (Days)'].mean()
        st.metric("Avg Turnaround Time", f"{avg_turnaround:.1f} days")
    
    with col4:
        open_orders = len(df_filtered[df_filtered['Status'].isin(['Open', 'In Progress'])])
        st.metric("Current Open Orders", f"{open_orders:,}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Creation vs Closure Rates
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Work Order Creation vs Closure Rates")
        daily_stats = df_filtered.groupby(df_filtered['Date Created'].dt.date).agg({
            'Work Order ID': 'count'
        }).reset_index()
        daily_stats.columns = ['Date', 'Created']
        
        daily_closures = df_filtered[df_filtered['Status'].isin(['Completed', 'Closed'])].groupby(
            df_filtered[df_filtered['Status'].isin(['Completed', 'Closed'])]['Completion Date'].dt.date
        ).size().reset_index()
        daily_closures.columns = ['Date', 'Closed']
        
        daily_combined = pd.merge(daily_stats, daily_closures, on='Date', how='outer').fillna(0)
        daily_combined = daily_combined.sort_values('Date').tail(30)
        
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(x=daily_combined['Date'], y=daily_combined['Created'],
                                       mode='lines+markers', name='Created', line=dict(color='blue', width=2)))
        fig_daily.add_trace(go.Scatter(x=daily_combined['Date'], y=daily_combined['Closed'],
                                       mode='lines+markers', name='Closed', line=dict(color='green', width=2)))
        fig_daily.update_layout(height=350, xaxis_title="Date", yaxis_title="Count")
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        st.subheader("Work Orders by Priority & System")
        priority_data = df_filtered.groupby(['System', 'Status']).size().reset_index(name='Count')
        fig_priority = px.sunburst(
            priority_data,
            path=['Status', 'System'],
            values='Count',
            color='Count',
            color_continuous_scale='RdYlGn_r'
        )
        fig_priority.update_layout(height=350)
        st.plotly_chart(fig_priority, use_container_width=True)
    
    # Top Failure Modes
    st.subheader("🔧 Top 10 Failure Modes by Frequency")
    top_failures = df_filtered['Failure Mode'].value_counts().head(10).reset_index()
    top_failures.columns = ['Failure Mode', 'Frequency']
    
    fig_failures = px.bar(
        top_failures,
        x='Frequency',
        y='Failure Mode',
        orientation='h',
        color='Frequency',
        color_continuous_scale='Reds',
        text='Frequency'
    )
    fig_failures.update_traces(texttemplate='%{text}', textposition='outside')
    fig_failures.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_failures, use_container_width=True)
    
    # Repeat Failures
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Repeat Failure Incidents")
        # Identify repeat failures by Vehicle ID and Failure Mode
        repeat_failures = df_filtered.groupby(['Vehicle ID', 'Failure Mode']).size().reset_index(name='Occurrences')
        repeat_failures = repeat_failures[repeat_failures['Occurrences'] > 1].sort_values('Occurrences', ascending=False)
        
        if len(repeat_failures) > 0:
            fig_repeat = px.treemap(
                repeat_failures.head(20),
                path=['Failure Mode', 'Vehicle ID'],
                values='Occurrences',
                color='Occurrences',
                color_continuous_scale='OrRd'
            )
            fig_repeat.update_layout(height=350)
            st.plotly_chart(fig_repeat, use_container_width=True)
        else:
            st.info("No repeat failures detected")
    
    with col2:
        st.subheader("Supervisor Performance")
        if selected_site != 'All Sites':
            supervisor_stats = df_filtered.groupby('Created By').agg({
                'Work Order ID': 'count',
                'Turnaround Time (Days)': 'mean'
            }).round(1).reset_index()
            supervisor_stats.columns = ['Supervisor', 'Orders Created', 'Avg TAT']
            
            fig_sup = px.scatter(
                supervisor_stats,
                x='Orders Created',
                y='Avg TAT',
                size='Orders Created',
                text='Supervisor',
                color='Avg TAT',
                color_continuous_scale='RdYlGn_r'
            )
            fig_sup.update_traces(textposition='top center')
            fig_sup.update_layout(height=350)
            st.plotly_chart(fig_sup, use_container_width=True)
        else:
            st.info("Select a specific workshop to view supervisor performance")
    
    # Open vs Closed by System
    st.subheader("Open vs Closed Work Orders by System")
    system_status = df_filtered.groupby(['System', 'Status']).size().unstack(fill_value=0)
    
    fig_system_status = go.Figure()
    for status in system_status.columns:
        fig_system_status.add_trace(go.Bar(
            name=status,
            x=system_status.index,
            y=system_status[status]
        ))
    
    fig_system_status.update_layout(barmode='stack', height=400, xaxis_title="System", yaxis_title="Count")
    st.plotly_chart(fig_system_status, use_container_width=True)

# ==================== DASHBOARD 3: TECHNICIAN PERFORMANCE ====================
def technician_performance_dashboard(df):
    """Dashboard 3: Technician Performance Dashboard"""
    st.markdown('<div class="dashboard-title">👨‍🔧 Technician Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Evaluate technician-level efficiency and workload balance*")
    
    # Technician selector
    technicians = ['All Technicians'] + sorted(df['Assigned To'].unique().tolist())
    selected_tech = st.selectbox("Select Technician", technicians)
    
    if selected_tech != 'All Technicians':
        df_filtered = df[df['Assigned To'] == selected_tech]
    else:
        df_filtered = df.copy()
    
    # Calculate rework rate (simplified: count of repeat work on same vehicle/system)
    rework_df = df.groupby(['Assigned To', 'Vehicle ID', 'System']).size().reset_index(name='Count')
    rework_rates = rework_df[rework_df['Count'] > 1].groupby('Assigned To').size()
    
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        orders_handled = len(df_filtered)
        st.metric("Orders Handled", f"{orders_handled:,}")
    
    with col2:
        avg_time = df_filtered['Turnaround Time (Days)'].mean()
        st.metric("Avg Time per Order", f"{avg_time:.1f} days")
    
    with col3:
        if selected_tech != 'All Technicians':
            rework = rework_rates.get(selected_tech, 0)
            rework_rate = (rework / orders_handled * 100) if orders_handled > 0 else 0
        else:
            rework_rate = len(rework_df[rework_df['Count'] > 1]) / len(df) * 100
        st.metric("Rework Rate", f"{rework_rate:.1f}%")
    
    with col4:
        total_labor = df_filtered['Labor Hours'].sum()
        utilization = (total_labor / (orders_handled * 8)) * 100 if orders_handled > 0 else 0
        st.metric("Utilization", f"{min(utilization, 100):.1f}%")
    
    with col5:
        total_cost = df_filtered['Total Cost'].sum()
        st.metric("Total Cost Impact", f"${total_cost:,.0f}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Technician Comparison
    st.subheader("Technician Performance Comparison")
    
    tech_comparison = df.groupby('Assigned To').agg({
        'Work Order ID': 'count',
        'Turnaround Time (Days)': 'mean',
        'Labor Hours': 'sum',
        'Total Cost': 'sum'
    }).round(2).reset_index()
    tech_comparison.columns = ['Technician', 'Orders', 'Avg Days', 'Total Labor Hours', 'Total Cost']
    tech_comparison['Efficiency Score'] = ((tech_comparison['Orders'] / tech_comparison['Avg Days']) * 10).round(1)
    tech_comparison = tech_comparison.sort_values('Efficiency Score', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tech_orders = px.bar(
            tech_comparison,
            x='Technician',
            y='Orders',
            color='Efficiency Score',
            color_continuous_scale='RdYlGn',
            title="Orders Handled vs Efficiency Score"
        )
        fig_tech_orders.update_layout(height=350)
        st.plotly_chart(fig_tech_orders, use_container_width=True)
    
    with col2:
        fig_tech_time = px.scatter(
            tech_comparison,
            x='Orders',
            y='Avg Days',
            size='Total Labor Hours',
            color='Technician',
            title="Workload vs Average Completion Time",
            hover_data=['Total Cost']
        )
        fig_tech_time.update_layout(height=350)
        st.plotly_chart(fig_tech_time, use_container_width=True)
    
    # Detailed Table
    st.subheader("Detailed Technician Metrics")
    st.dataframe(
        tech_comparison.style.background_gradient(subset=['Orders', 'Efficiency Score'], cmap='RdYlGn')
                             .format({'Total Cost': '${:,.0f}', 'Total Labor Hours': '{:.1f}', 'Avg Days': '{:.1f}'}),
        use_container_width=True,
        height=300
    )
    
    # Common Failure Types Handled
    st.subheader("Most Common Failure Types Handled")
    
    if selected_tech != 'All Technicians':
        failure_dist = df_filtered['Failure Mode'].value_counts().head(15).reset_index()
        failure_dist.columns = ['Failure Mode', 'Count']
        
        fig_failures = px.bar(
            failure_dist,
            x='Count',
            y='Failure Mode',
            orientation='h',
            color='Count',
            color_continuous_scale='Blues',
            title=f"Top Failures for {selected_tech}"
        )
        fig_failures.update_layout(height=500)
        st.plotly_chart(fig_failures, use_container_width=True)
    else:
        # Heatmap of technicians vs failure types
        tech_failure_matrix = pd.crosstab(df['Assigned To'], df['System'])
        
        fig_heatmap = px.imshow(
            tech_failure_matrix,
            labels=dict(x="System", y="Technician", color="Count"),
            color_continuous_scale='YlOrRd',
            title="Technician Specialization Heatmap"
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Top Performers
    st.subheader("🏆 Top Performing Technicians")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Fastest Average Completion**")
        fastest = tech_comparison.nsmallest(3, 'Avg Days')[['Technician', 'Avg Days']]
        for idx, row in fastest.iterrows():
            st.success(f"🥇 {row['Technician']}: {row['Avg Days']:.1f} days")
    
    with col2:
        st.markdown("**Highest Volume**")
        highest_volume = tech_comparison.nlargest(3, 'Orders')[['Technician', 'Orders']]
        for idx, row in highest_volume.iterrows():
            st.info(f"📊 {row['Technician']}: {int(row['Orders'])} orders")
    
    with col3:
        st.markdown("**Best Efficiency Score**")
        best_efficiency = tech_comparison.nlargest(3, 'Efficiency Score')[['Technician', 'Efficiency Score']]
        for idx, row in best_efficiency.iterrows():
            st.success(f"⭐ {row['Technician']}: {row['Efficiency Score']:.1f}")

# ==================== DASHBOARD 4: FAILURE MODE ANALYSIS ====================
def failure_mode_analysis_dashboard(df):
    """Dashboard 4: Failure Mode Analysis Dashboard"""
    st.markdown('<div class="dashboard-title">🔍 Failure Mode Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Identify most common and costly equipment issues*")
    
    # System selector
    systems = ['All Systems'] + sorted(df['System'].unique().tolist())
    selected_system = st.selectbox("Select System", systems)
    
    if selected_system != 'All Systems':
        df_filtered = df[df['System'] == selected_system]
    else:
        df_filtered = df.copy()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_failures = len(df_filtered)
        st.metric("Total Failures", f"{total_failures:,}")
    
    with col2:
        unique_modes = df_filtered['Failure Mode'].nunique()
        st.metric("Unique Failure Modes", f"{unique_modes}")
    
    with col3:
        mttr = df_filtered['Turnaround Time (Days)'].mean()
        st.metric("MTTR (Mean Time to Repair)", f"{mttr:.2f} days")
    
    with col4:
        total_cost = df_filtered['Total Cost'].sum()
        st.metric("Total Failure Cost", f"${total_cost:,.0f}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Failure Frequency
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Failure Frequency by System")
        system_freq = df['System'].value_counts().reset_index()
        system_freq.columns = ['System', 'Frequency']
        
        fig_system_freq = px.bar(
            system_freq,
            x='System',
            y='Frequency',
            color='Frequency',
            color_continuous_scale='Reds',
            title="System-Level Failure Distribution"
        )
        fig_system_freq.update_layout(height=350)
        st.plotly_chart(fig_system_freq, use_container_width=True)
    
    with col2:
        st.subheader("Subsystem Breakdown")
        if selected_system != 'All Systems':
            subsystem_freq = df_filtered['Subsystem'].value_counts().head(10).reset_index()
            subsystem_freq.columns = ['Subsystem', 'Frequency']
            
            fig_subsystem = px.pie(
                subsystem_freq,
                values='Frequency',
                names='Subsystem',
                title=f"Top Subsystems in {selected_system}"
            )
            fig_subsystem.update_layout(height=350)
            st.plotly_chart(fig_subsystem, use_container_width=True)
        else:
            subsystem_freq = df['Subsystem'].value_counts().head(10).reset_index()
            subsystem_freq.columns = ['Subsystem', 'Frequency']
            
            fig_subsystem = px.bar(
                subsystem_freq,
                x='Subsystem',
                y='Frequency',
                color='Frequency',
                color_continuous_scale='Blues'
            )
            fig_subsystem.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_subsystem, use_container_width=True)
    
    # Fault Category Heatmap
    st.subheader("Fault Category Heatmap (System x Component)")
    
    # Create heatmap data
    heatmap_data = pd.crosstab(df_filtered['System'], df_filtered['Component'])
    
    fig_heatmap = px.imshow(
        heatmap_data.T,  # Transpose for better visibility
        labels=dict(x="System", y="Component", color="Frequency"),
        color_continuous_scale='YlOrRd',
        aspect="auto"
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # MTTR by Failure Category
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MTTR by Failure Mode (Top 15)")
        mttr_by_failure = df_filtered.groupby('Failure Mode')['Turnaround Time (Days)'].agg(['mean', 'count']).reset_index()
        mttr_by_failure = mttr_by_failure[mttr_by_failure['count'] >= 5].nlargest(15, 'mean')
        mttr_by_failure.columns = ['Failure Mode', 'Avg MTTR', 'Count']
        
        fig_mttr = px.bar(
            mttr_by_failure,
            y='Failure Mode',
            x='Avg MTTR',
            orientation='h',
            color='Count',
            color_continuous_scale='Reds',
            title="Longest Repair Times"
        )
        fig_mttr.update_layout(height=500)
        st.plotly_chart(fig_mttr, use_container_width=True)
    
    with col2:
        st.subheader("Cost Impact by Failure Mode")
        cost_by_failure = df_filtered.groupby('Failure Mode')['Total Cost'].sum().nlargest(15).reset_index()
        cost_by_failure.columns = ['Failure Mode', 'Total Cost']
        
        fig_cost = px.bar(
            cost_by_failure,
            y='Failure Mode',
            x='Total Cost',
            orientation='h',
            color='Total Cost',
            color_continuous_scale='OrRd',
            title="Most Costly Failures"
        )
        fig_cost.update_layout(height=500)
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Pareto Analysis
    st.subheader("📊 Pareto Analysis - 80/20 Rule")
    st.markdown("*Identify the vital few failure modes causing the majority of issues*")
    
    failure_counts = df_filtered['Failure Mode'].value_counts().reset_index()
    failure_counts.columns = ['Failure Mode', 'Count']
    failure_counts['Cumulative %'] = (failure_counts['Count'].cumsum() / failure_counts['Count'].sum() * 100)
    failure_counts = failure_counts.head(20)
    
    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_pareto.add_trace(
        go.Bar(name='Frequency', x=failure_counts['Failure Mode'], y=failure_counts['Count'],
               marker_color='lightblue'),
        secondary_y=False
    )
    
    fig_pareto.add_trace(
        go.Scatter(name='Cumulative %', x=failure_counts['Failure Mode'],
                  y=failure_counts['Cumulative %'], mode='lines+markers',
                  marker=dict(size=8, color='red'), line=dict(width=3, color='red')),
        secondary_y=True
    )
    
    # Add 80% line
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="green", secondary_y=True,
                        annotation_text="80% Threshold")
    
    fig_pareto.update_xaxes(title_text="Failure Mode", tickangle=45)
    fig_pareto.update_yaxes(title_text="Frequency", secondary_y=False)
    fig_pareto.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 100])
    fig_pareto.update_layout(height=500)
    
    st.plotly_chart(fig_pareto, use_container_width=True)
    
    # Recommended Actions Effectiveness
    st.subheader("Recommended Action Effectiveness")
    
    # Group by recommended action and calculate success metrics
    action_stats = df_filtered.groupby('Recommended Action').agg({
        'Work Order ID': 'count',
        'Turnaround Time (Days)': 'mean',
        'Total Cost': 'mean'
    }).round(2).reset_index()
    action_stats.columns = ['Recommended Action', 'Usage Count', 'Avg MTTR', 'Avg Cost']
    action_stats = action_stats.sort_values('Usage Count', ascending=False).head(10)
    
    st.dataframe(
        action_stats.style.background_gradient(subset=['Usage Count'], cmap='Blues')
                          .format({'Avg Cost': '${:,.2f}', 'Avg MTTR': '{:.2f} days'}),
        use_container_width=True,
        height=300
    )

# ==================== DASHBOARD 5: WORK ORDER LIFECYCLE ====================
def work_order_lifecycle_dashboard(df):
    """Dashboard 5: Work Order Lifecycle Dashboard"""
    st.markdown('<div class="dashboard-title">🔄 Work Order Lifecycle Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Visualize process flow and bottlenecks within maintenance cycle*")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_creation_to_progress = df[df['Status'].isin(['In Progress', 'Completed', 'Closed'])]['Days Open'].mean()
        st.metric("Avg Time to Start", f"{avg_creation_to_progress:.1f} days")
    
    with col2:
        avg_completion_time = df[df['Status'].isin(['Completed', 'Closed'])]['Turnaround Time (Days)'].mean()
        st.metric("Avg Time to Complete", f"{avg_completion_time:.1f} days")
    
    with col3:
        orders_0_3_days = len(df[(df['Status'].isin(['Open', 'In Progress'])) & (df['Days Open'] <= 3)])
        st.metric("Orders: 0-3 Days Old", f"{orders_0_3_days:,}")
    
    with col4:
        orders_8_plus = len(df[(df['Status'].isin(['Open', 'In Progress'])) & (df['Days Open'] > 8)])
        st.metric("Orders: 8+ Days Old", f"{orders_8_plus:,}", delta=f"{(orders_8_plus/len(df)*100):.1f}%")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Lifecycle Flow Diagram (Sankey)
    st.subheader("Work Order Status Flow")
    
    # Create status transition data (simplified)
    status_map = {'Open': 0, 'In Progress': 1, 'Completed': 2, 'Closed': 3}
    
    # Simulate transitions for visualization
    transitions = {
        ('Open', 'In Progress'): len(df[df['Status'] == 'In Progress']),
        ('Open', 'Open'): len(df[df['Status'] == 'Open']),
        ('In Progress', 'Completed'): len(df[df['Status'] == 'Completed']),
        ('In Progress', 'In Progress'): len(df[df['Status'] == 'In Progress']) // 2,
        ('Completed', 'Closed'): len(df[df['Status'] == 'Closed']),
        ('Completed', 'Completed'): len(df[df['Status'] == 'Completed']) // 2,
    }
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Open", "In Progress", "Completed", "Closed"],
            color=["red", "orange", "lightgreen", "green"]
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2],
            target=[1, 0, 2, 1, 3, 2],
            value=[
                transitions.get(('Open', 'In Progress'), 0),
                transitions.get(('Open', 'Open'), 0),
                transitions.get(('In Progress', 'Completed'), 0),
                transitions.get(('In Progress', 'In Progress'), 0),
                transitions.get(('Completed', 'Closed'), 0),
                transitions.get(('Completed', 'Completed'), 0)
            ]
        )
    )])
    
    fig_sankey.update_layout(title_text="Work Order Status Transitions", height=400)
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    # Time in Each Stage
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Time Spent in Each Stage")
        
        # Calculate average days in each status
        stage_times = df.groupby('Status')['Days Open'].mean().reset_index()
        stage_times.columns = ['Status', 'Avg Days']
        stage_times = stage_times.sort_values('Avg Days', ascending=False)
        
        fig_stages = px.bar(
            stage_times,
            x='Status',
            y='Avg Days',
            color='Avg Days',
            color_continuous_scale='RdYlGn_r',
            title="Time Distribution by Status"
        )
        fig_stages.update_layout(height=350)
        st.plotly_chart(fig_stages, use_container_width=True)
    
    with col2:
        st.subheader("Status Distribution")
        
        status_dist = df['Status'].value_counts().reset_index()
        status_dist.columns = ['Status', 'Count']
        
        colors = {'Open': '#e74c3c', 'In Progress': '#f39c12', 'Completed': '#2ecc71', 'Closed': '#27ae60'}
        
        fig_status_dist = px.pie(
            status_dist,
            values='Count',
            names='Status',
            color='Status',
            color_discrete_map=colors,
            hole=0.4
        )
        fig_status_dist.update_traces(textposition='inside', textinfo='percent+label')
        fig_status_dist.update_layout(height=350)
        st.plotly_chart(fig_status_dist, use_container_width=True)
    
    # Aging Analysis
    st.subheader("📅 Open Orders Aging Analysis")
    
    # Create aging buckets
    df_open = df[df['Status'].isin(['Open', 'In Progress'])].copy()
    df_open['Age Bucket'] = pd.cut(
        df_open['Days Open'],
        bins=[0, 3, 7, 14, 30, float('inf')],
        labels=['0-3 days', '4-7 days', '8-14 days', '15-30 days', '30+ days']
    )
    
    aging_dist = df_open['Age Bucket'].value_counts().sort_index().reset_index()
    aging_dist.columns = ['Age Bucket', 'Count']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_aging = px.bar(
            aging_dist,
            x='Age Bucket',
            y='Count',
            color='Count',
            color_continuous_scale='Reds',
            title="Distribution of Open Work Orders by Age"
        )
        fig_aging.update_layout(height=400)
        st.plotly_chart(fig_aging, use_container_width=True)
    
    with col2:
        st.markdown("**Aging Summary**")
        for _, row in aging_dist.iterrows():
            percentage = (row['Count'] / aging_dist['Count'].sum() * 100)
            st.metric(row['Age Bucket'], f"{int(row['Count'])}", f"{percentage:.1f}%")
    
    # Workflow Efficiency Gauge
    st.subheader("⚡ Workflow Efficiency Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate completion rate
        completion_rate = (len(df[df['Status'].isin(['Completed', 'Closed'])]) / len(df)) * 100
        fig_completion = create_gauge_chart(
            completion_rate,
            "Completion Rate",
            max_value=100,
            threshold_good=70,
            threshold_warning=50
        )
        st.plotly_chart(fig_completion, use_container_width=True)
    
    with col2:
        # On-time completion rate (assume SLA is 7 days)
        on_time = len(df[(df['Status'].isin(['Completed', 'Closed'])) & (df['Turnaround Time (Days)'] <= 7)])
        total_completed = len(df[df['Status'].isin(['Completed', 'Closed'])])
        on_time_rate = (on_time / total_completed * 100) if total_completed > 0 else 0
        
        fig_ontime = create_gauge_chart(
            on_time_rate,
            "On-Time Completion",
            max_value=100,
            threshold_good=80,
            threshold_warning=60
        )
        st.plotly_chart(fig_ontime, use_container_width=True)
    
    with col3:
        # Process efficiency (fewer days open = better)
        avg_days = df[df['Status'].isin(['Open', 'In Progress'])]['Days Open'].mean()
        efficiency_score = max(0, (100 - (avg_days * 5)))  # Inverse scale
        
        fig_efficiency = create_gauge_chart(
            efficiency_score,
            "Process Efficiency",
            max_value=100,
            threshold_good=70,
            threshold_warning=50
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Status Transition Trends Over Time
    st.subheader("Status Transition Trends")
    
    status_timeline = df.groupby([pd.Grouper(key='Date Created', freq='W'), 'Status']).size().reset_index(name='Count')
    
    fig_timeline = px.area(
        status_timeline,
        x='Date Created',
        y='Count',
        color='Status',
        title="Weekly Work Order Status Distribution",
        color_discrete_map={'Open': '#e74c3c', 'In Progress': '#f39c12', 'Completed': '#2ecc71', 'Closed': '#27ae60'}
    )
    fig_timeline.update_layout(height=400)
    st.plotly_chart(fig_timeline, use_container_width=True)

# ==================== DASHBOARD 6: CATALOGUE GOVERNANCE ====================
def catalogue_governance_dashboard(ccr_df):
    """Dashboard 6: Catalogue Governance Dashboard"""
    st.markdown('<div class="dashboard-title">📚 Catalogue Governance Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Track health and evolution of the Failure Mode Catalogue*")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_ccr = len(ccr_df)
        st.metric("Total CCRs", f"{total_ccr}")
    
    with col2:
        approved = len(ccr_df[ccr_df['Status'] == 'Approved'])
        st.metric("Approved", f"{approved}", delta=f"{(approved/total_ccr*100):.1f}%")
    
    with col3:
        pending = len(ccr_df[ccr_df['Status'] == 'Pending'])
        st.metric("Pending", f"{pending}", delta=f"{(pending/total_ccr*100):.1f}%")
    
    with col4:
        avg_approval_time = ccr_df[ccr_df['Status'] == 'Approved']['Approval_Time_Days'].mean()
        st.metric("Avg Approval Time", f"{avg_approval_time:.1f} days")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # CCR Status Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CCR Status Distribution")
        
        status_counts = ccr_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        colors = {'Approved': '#2ecc71', 'Pending': '#f39c12', 'Rejected': '#e74c3c'}
        
        fig_status = px.pie(
            status_counts,
            values='Count',
            names='Status',
            color='Status',
            color_discrete_map=colors,
            hole=0.4
        )
        fig_status.update_traces(textposition='inside', textinfo='percent+label')
        fig_status.update_layout(height=350)
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        st.subheader("CCR by Request Type")
        
        type_counts = ccr_df['Request_Type'].value_counts().reset_index()
        type_counts.columns = ['Request Type', 'Count']
        
        fig_types = px.bar(
            type_counts,
            x='Request Type',
            y='Count',
            color='Count',
            color_continuous_scale='Blues',
            title="Distribution by Type"
        )
        fig_types.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_types, use_container_width=True)
    
    # CCR Trends Over Time
    st.subheader("CCR Submission Trends")
    
    ccr_timeline = ccr_df.groupby([pd.Grouper(key='Submission_Date', freq='M'), 'Status']).size().reset_index(name='Count')
    
    fig_timeline = px.line(
        ccr_timeline,
        x='Submission_Date',
        y='Count',
        color='Status',
        markers=True,
        title="Monthly CCR Submissions by Status",
        color_discrete_map={'Approved': '#2ecc71', 'Pending': '#f39c12', 'Rejected': '#e74c3c'}
    )
    fig_timeline.update_layout(height=400)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Approval Time Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Approval Time Distribution")
        
        approved_ccr = ccr_df[ccr_df['Status'] == 'Approved'].copy()
        
        fig_approval_time = px.histogram(
            approved_ccr,
            x='Approval_Time_Days',
            nbins=20,
            color_discrete_sequence=['#3498db'],
            title="Days to Approval Distribution"
        )
        fig_approval_time.update_layout(
            xaxis_title="Days to Approval",
            yaxis_title="Count",
            height=350
        )
        st.plotly_chart(fig_approval_time, use_container_width=True)
    
    with col2:
        st.subheader("Average Approval Time by Site")
        
        site_approval = approved_ccr.groupby('Site')['Approval_Time_Days'].mean().reset_index()
        site_approval.columns = ['Site', 'Avg Days']
        
        fig_site_approval = px.bar(
            site_approval,
            x='Site',
            y='Avg Days',
            color='Avg Days',
            color_continuous_scale='RdYlGn_r',
            title="Site Performance"
        )
        fig_site_approval.update_layout(height=350)
        st.plotly_chart(fig_site_approval, use_container_width=True)
    
    # Top Contributors
    st.subheader("👥 Top Contributors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**By Volume**")
        top_contributors = ccr_df['Submitted_By'].value_counts().head(10).reset_index()
        top_contributors.columns = ['Contributor', 'CCRs Submitted']
        
        fig_contributors = px.bar(
            top_contributors,
            y='Contributor',
            x='CCRs Submitted',
            orientation='h',
            color='CCRs Submitted',
            color_continuous_scale='Blues'
        )
        fig_contributors.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_contributors, use_container_width=True)
    
    with col2:
        st.markdown("**By Approval Rate**")
        contributor_stats = ccr_df.groupby('Submitted_By').agg({
            'CCR_ID': 'count',
            'Status': lambda x: (x == 'Approved').sum()
        }).reset_index()
        contributor_stats.columns = ['Contributor', 'Total', 'Approved']
        contributor_stats['Approval Rate'] = (contributor_stats['Approved'] / contributor_stats['Total'] * 100).round(1)
        contributor_stats = contributor_stats[contributor_stats['Total'] >= 5].sort_values('Approval Rate', ascending=False).head(10)
        
        fig_approval_rate = px.bar(
            contributor_stats,
            y='Contributor',
            x='Approval Rate',
            orientation='h',
            color='Approval Rate',
            color_continuous_scale='RdYlGn',
            text='Approval Rate'
        )
        fig_approval_rate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_approval_rate.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_approval_rate, use_container_width=True)
    
    # CCR Details Table
    st.subheader("Recent CCR Activity")
    
    recent_ccr = ccr_df.sort_values('Submission_Date', ascending=False).head(20)
    display_ccr = recent_ccr[['CCR_ID', 'Submission_Date', 'Status', 'Request_Type', 'Site', 'Submitted_By']].copy()
    display_ccr['Submission_Date'] = display_ccr['Submission_Date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(display_ccr, use_container_width=True, height=400)
    
    # Catalogue Version Summary
    st.subheader("📖 Catalogue Version Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Current Version:** v2.5.3  
        **Last Updated:** 2025-01-15  
        **Active Failure Modes:** 427
        """)
    
    with col2:
        st.success(f"""
        **Changes This Month:**  
        New Modes Added: 12  
        Updated: 8  
        Deprecated: 3
        """)
    
    with col3:
        st.warning(f"""
        **Pending Reviews:**  
        New Submissions: {pending}  
        Revision Requests: {len(ccr_df[ccr_df['Request_Type'] == 'Update Existing'])}
        """)

# ==================== DASHBOARD 7: DATA QUALITY & COMPLIANCE ====================
def data_quality_compliance_dashboard(df):
    """Dashboard 7: Data Quality & Compliance Dashboard"""
    st.markdown('<div class="dashboard-title">✅ Data Quality & Compliance Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Ensure compliance with data standards and completeness*")
    
    # Calculate data quality metrics
    total_records = len(df)
    
    # Completeness metrics
    complete_fault_data = len(df[df['Failure Mode'].notna() & df['Cause'].notna()])
    completeness_rate = (complete_fault_data / total_records * 100)
    
    # Simulate user activity data
    np.random.seed(42)
    user_logins = pd.DataFrame({
        'User': [f'User_{i}' for i in range(1, 26)],
        'Last_Login': pd.date_range(end=datetime.now(), periods=25, freq='-1D'),
        'Login_Count': np.random.randint(5, 100, 25),
        'Actions': np.random.randint(10, 500, 25)
    })
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Completeness", f"{completeness_rate:.1f}%")
    
    with col2:
        active_users = len(user_logins[user_logins['Last_Login'] >= datetime.now() - timedelta(days=7)])
        st.metric("Active Users (7d)", f"{active_users}")
    
    with col3:
        records_with_all_fields = len(df.dropna())
        st.metric("Complete Records", f"{records_with_all_fields:,}")
    
    with col4:
        audit_completeness = 99.8  # Simulated
        st.metric("Audit Log Completeness", f"{audit_completeness:.1f}%")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Data Completeness Analysis
    st.subheader("📊 Field-Level Data Completeness")
    
    # Calculate completeness for each field
    field_completeness = pd.DataFrame({
        'Field': df.columns,
        'Completeness %': [(1 - df[col].isna().sum() / len(df)) * 100 for col in df.columns]
    }).sort_values('Completeness %')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_completeness = px.bar(
            field_completeness,
            y='Field',
            x='Completeness %',
            orientation='h',
            color='Completeness %',
            color_continuous_scale='RdYlGn',
            title="Data Completeness by Field"
        )
        fig_completeness.add_vline(x=80, line_dash="dash", line_color="orange",
                                   annotation_text="80% Threshold")
        fig_completeness.add_vline(x=95, line_dash="dash", line_color="green",
                                   annotation_text="95% Target")
        fig_completeness.update_layout(height=600)
        st.plotly_chart(fig_completeness, use_container_width=True)
    
    with col2:
        st.markdown("**Completeness Summary**")
        
        excellent = len(field_completeness[field_completeness['Completeness %'] >= 95])
        good = len(field_completeness[(field_completeness['Completeness %'] >= 80) & (field_completeness['Completeness %'] < 95)])
        poor = len(field_completeness[field_completeness['Completeness %'] < 80])
        
        st.metric("Excellent (≥95%)", f"{excellent} fields", delta="✓")
        st.metric("Good (80-95%)", f"{good} fields", delta="⚠")
        st.metric("Needs Attention (<80%)", f"{poor} fields", delta="❌")
        
        st.markdown("---")
        st.markdown("**Critical Missing Data:**")
        critical_fields = field_completeness[field_completeness['Completeness %'] < 80]
        for _, row in critical_fields.iterrows():
            st.warning(f"{row['Field']}: {row['Completeness %']:.1f}%")
    
    # User Activity Metrics
    st.subheader("👤 User Activity & Engagement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Login Activity - Last 30 Days**")
        
        fig_logins = px.bar(
            user_logins.sort_values('Login_Count', ascending=False).head(15),
            x='User',
            y='Login_Count',
            color='Login_Count',
            color_continuous_scale='Blues',
            title="Top Active Users"
        )
        fig_logins.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_logins, use_container_width=True)
    
    with col2:
        st.markdown("**User Actions Distribution**")
        
        fig_actions = px.scatter(
            user_logins,
            x='Login_Count',
            y='Actions',
            size='Actions',
            color='Actions',
            color_continuous_scale='Viridis',
            title="Logins vs Actions",
            hover_data=['User']
        )
        fig_actions.update_layout(height=350)
        st.plotly_chart(fig_actions, use_container_width=True)
    
    # Recent User Activity Table
    st.subheader("Recent User Activity")
    recent_activity = user_logins.sort_values('Last_Login', ascending=False).head(15).copy()
    recent_activity['Last_Login'] = recent_activity['Last_Login'].dt.strftime('%Y-%m-%d %H:%M')
    recent_activity['Days Since Login'] = (datetime.now() - pd.to_datetime(user_logins.sort_values('Last_Login', ascending=False).head(15)['Last_Login'])).dt.days
    
    st.dataframe(
        recent_activity.style.background_gradient(subset=['Login_Count', 'Actions'], cmap='Blues'),
        use_container_width=True,
        height=300
    )
    
    # Data Quality Gauges
    st.subheader("⚙️ System Health Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_data_quality = create_gauge_chart(
            completeness_rate,
            "Overall Data Quality",
            max_value=100,
            threshold_good=90,
            threshold_warning=75
        )
        st.plotly_chart(fig_data_quality, use_container_width=True)
    
    with col2:
        sync_status = 98.5  # Simulated
        fig_sync = create_gauge_chart(
            sync_status,
            "Data Sync Status",
            max_value=100,
            threshold_good=95,
            threshold_warning=85
        )
        st.plotly_chart(fig_sync, use_container_width=True)
    
    with col3:
        backup_status = 100  # Simulated
        fig_backup = create_gauge_chart(
            backup_status,
            "Backup Verification",
            max_value=100,
            threshold_good=98,
            threshold_warning=90
        )
        st.plotly_chart(fig_backup, use_container_width=True)
    
    # Audit Log Summary
    st.subheader("📋 Audit Log Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("""
        **Total Audit Records**  
        124,587 entries
        """)
    
    with col2:
        st.success("""
        **Records/User Action**  
        99.8% coverage
        """)
    
    with col3:
        st.info("""
        **Last Backup**  
        2025-01-15 02:00 UTC
        """)
    
    with col4:
        st.success("""
        **Security Violations**  
        0 in last 30 days
        """)
    
    # Compliance Status
    st.subheader("🛡️ Compliance Status")
    
    compliance_items = pd.DataFrame({
        'Compliance Area': [
            'Data Residency Requirements',
            'Field Completeness Standards',
            'User Access Controls',
            'Audit Trail Integrity',
            'Backup & Recovery Procedures',
            'Data Retention Policy'
        ],
        'Status': ['Compliant', 'Compliant', 'Compliant', 'Compliant', 'Compliant', 'Compliant'],
        'Last Audit': ['2025-01-10', '2025-01-15', '2025-01-12', '2025-01-14', '2025-01-15', '2025-01-08'],
        'Score': [100, 98, 100, 100, 100, 95]
    })
    
    fig_compliance = px.bar(
        compliance_items,
        y='Compliance Area',
        x='Score',
        orientation='h',
        color='Score',
        color_continuous_scale='RdYlGn',
        text='Status',
        title="Compliance Scorecard"
    )
    fig_compliance.update_traces(textposition='inside')
    fig_compliance.update_layout(height=400)
    st.plotly_chart(fig_compliance, use_container_width=True)

# ==================== DASHBOARD 8: FUTURE-READY ====================
def future_ready_dashboard():
    """Dashboard 8: Future-Ready Dashboards"""
    st.markdown('<div class="dashboard-title">🚀 Future-Ready Analytics (Phase 2+)</div>', unsafe_allow_html=True)
    st.markdown("*Next-generation predictive and AI-powered insights*")
    
    st.info("🔮 **Coming Soon:** These advanced analytics modules will be introduced with the FRACAS Analytics Module in Phase 2+")
    
    # Placeholder sections for future features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Reliability Performance",
        "Predictive Maintenance",
        "Forecasting",
        "Pareto Analysis"
    ])
    
    with tab1:
        st.subheader("📈 Reliability Performance Dashboards")
        st.markdown("""
        **Planned Features:**
        - MTBF (Mean Time Between Failures) trends by vehicle/system
        - MTTR (Mean Time to Repair) analysis with statistical controls
        - Reliability growth curves
        - Failure rate trending (bathtub curve analysis)
        - System availability metrics
        - Comparative reliability benchmarking
        """)
        
        st.info("📊 Visual mockups and detailed feature specifications will be available in Q2 2025")
    
    with tab2:
        st.subheader("🤖 Predictive Maintenance Insights")
        st.markdown("""
        **Planned Features:**
        - AI/ML models for failure prediction
        - Remaining useful life (RUL) estimation
        - Anomaly detection in maintenance patterns
        - Optimal maintenance scheduling recommendations
        - Risk scoring for equipment failures
        - Integration with IoT sensor data
        """)
        
        st.info("🔮 Machine learning models currently in development phase")
    
    with tab3:
        st.subheader("🔮 Component Failure Forecasting")
        st.markdown("""
        **Planned Features:**
        - Time-series forecasting of failure rates
        - Cost impact simulation models
        - Spare parts demand forecasting
        - Budget planning tools
        - Scenario analysis (what-if modeling)
        - Multi-horizon forecasting (short/medium/long term)
        """)
        
        st.info("📈 Forecasting algorithms being calibrated with historical data")
    
    with tab4:
        st.subheader("📊 Reliability Pareto Analysis")
        st.markdown("""
        **Planned Features:**
        - 80/20 analysis for failures (vital few vs trivial many)
        - Multi-dimensional Pareto charts (cost, frequency, downtime)
        - Dynamic filtering and drill-down capabilities
        - Automated identification of high-impact failure modes
        - ROI analysis for preventive actions
        - Continuous improvement tracking
        """)
        
        st.info("📉 Advanced Pareto visualizations in design phase")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.subheader("🎯 Implementation Roadmap")
    
    roadmap = pd.DataFrame({
        'Phase': ['Phase 2', 'Phase 2', 'Phase 3', 'Phase 3', 'Phase 4'],
        'Feature': [
            'Reliability Performance Dashboards',
            'Basic Predictive Models',
            'Advanced AI/ML Integration',
            'Component Forecasting',
            'Full Pareto and ROI Analysis'
        ],
        'Status': ['Q2 2025', 'Q3 2025', 'Q4 2025', 'Q4 2025', 'Q1 2026'],
        'Priority': ['High', 'High', 'Medium', 'Medium', 'Low']
    })
    
    st.dataframe(roadmap, use_container_width=True, height=250)
    
    st.success("""
    💡 **Get Ready:** These features will leverage the data you are collecting now. 
    Ensure data quality and completeness to maximize the value of future analytics!
    """)

# ==================== MAIN APPLICATION ====================
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    
    # Check if data is loaded
    if not st.session_state['data_loaded']:
        # Show import screen
        data_import_screen()
    else:
        # Data is loaded - show full application
        
        # Header
        st.markdown('<div class="main-header">🔧 AMIC Work Order Management & Analytics Dashboard Suite</div>', 
                    unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=AMIC+FRACAS", use_container_width=True)
            st.markdown("---")
            
            # Data info
            st.markdown("### 📊 Loaded Data")
            df = st.session_state['df']
            st.success(f"✅ {len(df):,} records loaded")
            
            # Reset button
            if st.button("🔄 Load Different File", use_container_width=True):
                st.session_state['data_loaded'] = False
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 📊 Dashboard Navigation")
            
            dashboard_option = st.radio(
                "Select Dashboard:",
                [
                    "1️⃣ Executive Overview",
                    "2️⃣ Site Performance",
                    "3️⃣ Technician Performance",
                    "4️⃣ Failure Mode Analysis",
                    "5️⃣ Work Order Lifecycle",
                    "6️⃣ Catalogue Governance",
                    "7️⃣ Data Quality & Compliance",
                    "8️⃣ Future-Ready Analytics"
                ]
            )
            
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>
            <strong>AMIC FRACAS v2.0</strong><br>
            © 2025 All Rights Reserved
            </div>
            """, unsafe_allow_html=True)
        
        # Display selected dashboard
        try:
            df = st.session_state['df']
            ccr_df = st.session_state['ccr_df']
            
            if "Executive Overview" in dashboard_option:
                executive_overview_dashboard(df)
            elif "Site Performance" in dashboard_option:
                site_performance_dashboard(df)
            elif "Technician Performance" in dashboard_option:
                technician_performance_dashboard(df)
            elif "Failure Mode Analysis" in dashboard_option:
                failure_mode_analysis_dashboard(df)
            elif "Work Order Lifecycle" in dashboard_option:
                work_order_lifecycle_dashboard(df)
            elif "Catalogue Governance" in dashboard_option:
                catalogue_governance_dashboard(ccr_df)
            elif "Data Quality" in dashboard_option:
                data_quality_compliance_dashboard(df)
            elif "Future-Ready" in dashboard_option:
                future_ready_dashboard()
        
        except Exception as e:
            st.error(f"Error displaying dashboard: {str(e)}")
            if st.button("Return to Import"):
                st.session_state['data_loaded'] = False
                st.rerun()

if __name__ == "__main__":
    main()
