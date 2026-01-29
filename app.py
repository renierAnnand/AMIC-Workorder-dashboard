"""
AMIC IMSS - Integrated Maintenance Support System Analytics Dashboard
========================================================================
Updated to work with actual IMSS data structure
Version: 2.1
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import traceback

# Page configuration
st.set_page_config(
    page_title="IMSS Analytics Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c5f2d;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #2c5f2d;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #f0f8f0 0%, #ffffff 100%);
    }
    .dashboard-title {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-left: 5px solid #2c5f2d;
        background: linear-gradient(90deg, #f0f8f0 0%, #ffffff 100%);
    }
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #2c5f2d 0%, transparent 100%);
        margin: 2rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .alert-critical { background-color: #fee; border-color: #f44; color: #c00; }
    .alert-warning { background-color: #fff3cd; border-color: #ffc107; color: #856404; }
    .alert-success { background-color: #d4edda; border-color: #28a745; color: #155724; }
    .alert-info { background-color: #d1ecf1; border-color: #17a2b8; color: #0c5460; }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_and_process_data(file_path):
    """Load IMSS data and standardize column names"""
    try:
        df = pd.read_excel(file_path)
        
        # Column mapping from IMSS to dashboard standard
        column_mapping = {
            'MngWoNumber': 'WO Number',
            'OldSystemWoId': 'Old System ID',
            'CreatedBy': 'Created By',
            'CreatedDate': 'Created Date',
            'MngWoCreationDate': 'MNG Creation Date',
            'WorkshopCode': 'Workshop',
            'VehicleOwningUnitCode': 'Unit Code',
            'MngVehicleId': 'Vehicle ID',
            'Status': 'Status',
            'MaintenanceType': 'Maintenance Type',
            'Priority': 'Priority',
            'Description': 'Description',
            'TechnicianNotes': 'Technician Notes',
            'TechnicianAmicId': 'Assigned To',
            'StartDate': 'Start Date',
            'CompletionDate': 'Completion Date',
            'MileageAtService': 'Mileage',
            'RequiresSpareParts': 'Requires Parts',
            'SparePartsReceived': 'Parts Received',
            'SparePartReceiptDate': 'Parts Receipt Date'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure date columns are datetime
        date_columns = ['Created Date', 'MNG Creation Date', 'Start Date', 'Completion Date', 'Parts Receipt Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate metrics with proper handling of NaT values
        df['Days Open'] = (datetime.now() - df['Created Date']).dt.days
        df['Turnaround Time (Days)'] = (df['Completion Date'] - df['Created Date']).dt.days
        
        # Extract priority level
        df['Priority Level'] = df['Priority'].str.extract(r'(\d+)')[0].astype(float)
        
        # Create time periods
        df['Month'] = df['Created Date'].dt.to_period('M').astype(str)
        df['Week'] = df['Created Date'].dt.to_period('W').astype(str)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the file has the correct IMSS column structure")
        raise

def create_gauge_chart(value, title, max_value=100, threshold_good=80, threshold_warning=60):
    """Create a gauge chart for KPIs"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 18}},
        delta={'reference': threshold_good},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "#2c5f2d"},
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
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def data_import_screen():
    """Initial screen for data import"""
    st.markdown('<div class="main-header">🔧 AMIC IMSS - Integrated Maintenance Support System</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h2 style='color: #2c5f2d;'>📁 Import Work Order Data</h2>
            <p style='font-size: 1.1rem; color: #555;'>
                Upload your IMSS Excel export file to begin analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx', 'xls'],
            help="Upload IMSS work order export file",
            key="data_upload"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: **{uploaded_file.name}**")
            st.info(f"📊 File size: {uploaded_file.size / 1024:.2f} KB")
            
            if st.button("🔍 Load & Analyze Data", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Processing IMSS data..."):
                        data_path = f"/tmp/{uploaded_file.name}"
                        with open(data_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Debug: Show what we're loading
                        st.info("Reading Excel file...")
                        
                        df = load_and_process_data(data_path)
                        
                        st.session_state['data_loaded'] = True
                        st.session_state['df'] = df
                        
                        st.markdown("### 📋 Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Total Work Orders", f"{len(df):,}")
                        with col_b:
                            st.metric("Workshops", f"{df['Workshop'].nunique()}")
                        with col_c:
                            st.metric("Vehicles", f"{df['Vehicle ID'].nunique()}")
                        with col_d:
                            date_range_days = (df['Created Date'].max() - df['Created Date'].min()).days
                            st.metric("Date Range", f"{date_range_days} days")
                        
                        st.success("✅ Data loaded successfully! Use sidebar to navigate.")
                        st.balloons()
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    st.info("**Troubleshooting:**")
                    st.info("1. Ensure file is a valid Excel file (.xlsx)")
                    st.info("2. Check that all required columns exist")
                    st.info("3. Verify date columns are formatted as dates")
                    
                    # Show detailed error for debugging
                    with st.expander("🔍 Technical Details"):
                        st.code(str(e))
                        st.code(traceback.format_exc())
        else:
            st.info("👆 Please upload the IMSS Excel export file")
            
            with st.expander("📖 Expected File Format"):
                st.markdown("""
                **Required Columns:**
                - MngWoNumber, OldSystemWoId, CreatedBy, CreatedDate
                - WorkshopCode, VehicleOwningUnitCode, MngVehicleId
                - Status, MaintenanceType, Priority, Description
                - TechnicianAmicId, StartDate, CompletionDate
                - MileageAtService, RequiresSpareParts, SparePartsReceived
                
                Use the **IMSS_WorkOrders_5000_Demo.xlsx** file provided for testing.
                """)

def executive_overview_dashboard(df):
    """Command-Level Executive Overview"""
    st.markdown('<div class="dashboard-title">⭐ Command-Level Executive Overview</div>', unsafe_allow_html=True)
    
    # Top KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Work Orders", f"{len(df):,}")
    
    with col2:
        active = len(df[~df['Status'].isin(['Completed', 'Closed'])])
        st.metric("Active Orders", f"{active:,}", delta=f"{active/len(df)*100:.1f}%")
    
    with col3:
        critical = len(df[df['Priority Level'] <= 2])
        st.metric("Critical/High", f"{critical:,}", delta_color="inverse")
    
    with col4:
        completed = len(df[df['Status'].isin(['Completed', 'Closed'])])
        st.metric("Completion Rate", f"{completed/len(df)*100:.1f}%")
    
    with col5:
        avg_tat = df['Turnaround Time (Days)'].mean()
        st.metric("Avg TAT", f"{avg_tat:.1f} days")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Status Distribution")
        status_counts = df['Status'].value_counts()
        colors = {
            'Open': '#dc3545', 'In Progress': '#ffc107',
            'Waiting Parts': '#fd7e14', 'Under Maintenance': '#17a2b8',
            'Completed': '#28a745', 'Closed': '#20c997'
        }
        fig = px.pie(values=status_counts.values, names=status_counts.index,
                    color=status_counts.index, color_discrete_map=colors, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Monthly Trend")
        monthly = df.groupby('Month').size().reset_index(name='Count')
        fig = px.line(monthly, x='Month', y='Count', markers=True)
        fig.update_traces(line_color='#2c5f2d', line_width=3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Workshop Performance
    st.subheader("🏢 Workshop Performance")
    workshop_stats = df.groupby('Workshop').agg({
        'WO Number': 'count',
        'Turnaround Time (Days)': 'mean'
    }).round(1).reset_index()
    workshop_stats.columns = ['Workshop', 'Work Orders', 'Avg TAT']
    workshop_stats = workshop_stats.sort_values('Work Orders', ascending=False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name='Work Orders', x=workshop_stats['Workshop'],
                        y=workshop_stats['Work Orders'], marker_color='#2c5f2d'),
                 secondary_y=False)
    fig.add_trace(go.Scatter(name='Avg TAT', x=workshop_stats['Workshop'],
                            y=workshop_stats['Avg TAT'], mode='lines+markers',
                            marker=dict(size=10, color='red'), line=dict(width=3)),
                 secondary_y=True)
    fig.update_xaxes(title_text="Workshop")
    fig.update_yaxes(title_text="Work Orders", secondary_y=False)
    fig.update_yaxes(title_text="Avg TAT (Days)", secondary_y=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Alerts
    st.subheader("⚠️ Alerts")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        waiting_parts = len(df[df['Status'] == 'Waiting Parts'])
        st.markdown(f"""
        <div class="alert-box alert-warning">
            <strong>⏳ Waiting Parts:</strong> {waiting_parts} orders
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        critical_old = len(df[(df['Priority Level'] == 1) & (df['Days Open'] > 7)])
        if critical_old > 0:
            st.markdown(f"""
            <div class="alert-box alert-critical">
                <strong>🚨 Critical Overdue:</strong> {critical_old} orders
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box alert-success">
                <strong>✅ No Critical Overdue</strong>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        parts_pending = len(df[(df['Requires Parts'] == True) & (df['Parts Received'] == False)])
        st.markdown(f"""
        <div class="alert-box alert-info">
            <strong>📦 Parts Pending:</strong> {parts_pending} orders
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        fast = len(df[(df['Status'].isin(['Completed', 'Closed'])) & (df['Turnaround Time (Days)'] <= 5)])
        st.markdown(f"""
        <div class="alert-box alert-success">
            <strong>🚀 Fast Completion:</strong> {fast} orders (≤5 days)
        </div>
        """, unsafe_allow_html=True)

def parts_analysis_dashboard(df):
    """Spare Parts Analysis"""
    st.markdown('<div class="dashboard-title">📦 Spare Parts & Supply Chain Analysis</div>', unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        requires = (df['Requires Parts'] == True).sum()
        st.metric("Requires Parts", f"{requires:,}", delta=f"{requires/len(df)*100:.1f}%")
    
    with col2:
        waiting = len(df[df['Status'] == 'Waiting Parts'])
        st.metric("Waiting for Parts", f"{waiting:,}")
    
    with col3:
        received = (df['Parts Received'] == True).sum()
        st.metric("Parts Received", f"{received:,}")
    
    with col4:
        pending = ((df['Requires Parts'] == True) & (df['Parts Received'] == False)).sum()
        st.metric("Parts Pending", f"{pending:,}")
    
    with col5:
        avg_wait = df[df['Status'] == 'Waiting Parts']['Days Open'].mean()
        st.metric("Avg Wait Time", f"{avg_wait:.1f} days")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Parts trend
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Parts Requirements Over Time")
        parts_trend = df[df['Requires Parts'] == True].groupby('Month').size().reset_index(name='Count')
        fig = px.line(parts_trend, x='Month', y='Count', markers=True)
        fig.update_traces(line_color='#2c5f2d', line_width=3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Parts Status")
        parts_status = pd.DataFrame({
            'Status': ['Requires Parts', 'Parts Received', 'Waiting for Parts', 'No Parts Needed'],
            'Count': [
                requires,
                received,
                waiting,
                len(df) - requires
            ]
        })
        fig = px.bar(parts_status, x='Status', y='Count', color='Count',
                    color_continuous_scale='Oranges')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Workshop comparison
    st.subheader("🏢 Parts Waiting by Workshop")
    workshop_parts = df[df['Status'] == 'Waiting Parts'].groupby('Workshop').agg({
        'WO Number': 'count',
        'Days Open': 'mean'
    }).reset_index()
    workshop_parts.columns = ['Workshop', 'Orders Waiting', 'Avg Wait Days']
    workshop_parts = workshop_parts.sort_values('Orders Waiting', ascending=False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name='Orders', x=workshop_parts['Workshop'],
                        y=workshop_parts['Orders Waiting'], marker_color='#ffc107'),
                 secondary_y=False)
    fig.add_trace(go.Scatter(name='Avg Wait', x=workshop_parts['Workshop'],
                            y=workshop_parts['Avg Wait Days'], mode='lines+markers',
                            marker=dict(size=10, color='red'), line=dict(width=3)),
                 secondary_y=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application"""
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    
    if not st.session_state['data_loaded']:
        data_import_screen()
    else:
        st.markdown('<div class="main-header">🔧 AMIC IMSS Analytics</div>', unsafe_allow_html=True)
        
        with st.sidebar:
            st.markdown("### 🔧 AMIC IMSS")
            st.markdown("*Integrated Maintenance Support System*")
            st.markdown("---")
            
            df = st.session_state['df']
            st.success(f"✅ {len(df):,} records loaded")
            
            if st.button("🔄 Load Different File", use_container_width=True):
                st.session_state['data_loaded'] = False
                st.rerun()
            
            st.markdown("---")
            dashboard = st.radio("Select Dashboard:", [
                "1️⃣ Executive Overview",
                "2️⃣ Parts Analysis"
            ])
            
            st.markdown("---")
            st.markdown("### 📈 Quick Stats")
            total = len(df)
            active = len(df[~df['Status'].isin(['Completed', 'Closed'])])
            completion = len(df[df['Status'].isin(['Completed', 'Closed'])]) / total * 100
            
            st.metric("Total", f"{total:,}")
            st.metric("Active", f"{active:,}")
            st.metric("Completed", f"{completion:.1f}%")
        
        try:
            df = st.session_state['df']
            
            if "Executive Overview" in dashboard:
                executive_overview_dashboard(df)
            elif "Parts Analysis" in dashboard:
                parts_analysis_dashboard(df)
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
