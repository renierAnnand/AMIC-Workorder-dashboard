"""
AMIC IMSS - Integrated Maintenance Support System Analytics Dashboard
========================================================================
Enhanced version with Role-Based Security and Comprehensive Analytics
Version: 3.0
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
    
    /* Status Color Definitions - Aligned with IMSS System */
    .status-draft { background-color: #6c757d; } /* Gray */
    .status-waiting-parts { background-color: #ffc107; } /* Yellow */
    .status-waiting-approval { background-color: #fd7e14; } /* Orange */
    .status-under-maintenance { background-color: #17a2b8; } /* Cyan */
    .status-rejected { background-color: #dc3545; } /* Red */
    .status-completed { background-color: #28a745; } /* Green */
    .role-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        color: white;
        margin: 0.5rem 0;
    }
    .role-supervisor { background-color: #17a2b8; }
    .role-fleet { background-color: #28a745; }
    .role-exec { background-color: #dc3545; }
    
    /* Quick Filter Buttons */
    .quick-filter-section {
        background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #dee2e6;
    }
    .filter-active {
        background-color: #2c5f2d !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .filter-inactive {
        background-color: #e9ecef !important;
        color: #495057 !important;
    }
    .filter-indicator {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def get_user_scope():
    """
    Return user role and allowed workshops.
    In production, this would query from database/auth system.
    For demo, using session state selection.
    """
    # TODO: Replace with actual authentication system
    
    if 'user_role' not in st.session_state:
        st.session_state['user_role'] = 'Exec'
        st.session_state['user_workshops'] = []
    
    role = st.session_state.get('user_role', 'Exec')
    workshops = st.session_state.get('user_workshops', [])
    
    return role, workshops

@st.cache_data
def build_lookup_mappings():
    """
    Create lookup mappings for cascading filters.
    TODO: Replace with actual database queries in production.
    """
    
    # Workshop -> Brigade -> City -> Province mapping
    workshop_hierarchy = pd.DataFrame({
        'Workshop': ['PMABPS', 'PMABPS', 'KSBPS', 'KSBPS', '2SRIB', '2SRIB', 
                     'MECH1', 'MECH1', 'ARM5', 'ARM5', 'MAINT1', 'MAINT2', 'MAINT3', 'DSW-01', 'DSW-02'],
        'Brigade': [
            'Prince Mohammed bin Abdulrahman Brigade',
            'Prince Mohammed bin Abdulrahman Brigade',
            'King Saud Brigade', 
            'King Saud Brigade',
            '2nd Special Rapid Intervention Brigade',
            '2nd Special Rapid Intervention Brigade',
            '1st Mechanized Brigade',
            '1st Mechanized Brigade',
            '5th Armored Brigade',
            '5th Armored Brigade',
            'Maintenance Command East',
            'Maintenance Command Central',
            'Maintenance Command West',
            'Direct Support Workshop Alpha',
            'Direct Support Workshop Beta'
        ],
        'City': [
            'Riyadh', 'Riyadh', 'Riyadh', 'Riyadh', 'Riyadh', 'Riyadh',
            'Dammam', 'Dammam', 'Tabuk', 'Tabuk', 
            'Dammam', 'Riyadh', 'Jeddah', 'Khobar', 'Medina'
        ],
        'Province': [
            'Riyadh Province', 'Riyadh Province', 'Riyadh Province', 'Riyadh Province',
            'Riyadh Province', 'Riyadh Province',
            'Eastern Province', 'Eastern Province', 'Tabuk Province', 'Tabuk Province',
            'Eastern Province', 'Riyadh Province', 'Makkah Province', 
            'Eastern Province', 'Madinah Province'
        ]
    }).drop_duplicates()
    
    # Vehicle ID -> Vehicle Type mapping
    # Since we have ~5000 vehicle IDs, create a mapping based on ID patterns
    vehicle_types = [
        'Arive (armored infantry) personnel carrier (PCA)',
        'M113 Armored Personnel Carrier',
        'LAV-25 Light Armored Vehicle',
        'HMMWV',
        'M1117 Armored Security Vehicle',
        'Oshkosh M-ATV',
        'Caiman MRAP'
    ]
    
    # This will be populated dynamically based on actual Vehicle IDs in data
    
    # Unit Code -> Workshop mapping (for additional context)
    unit_workshop = pd.DataFrame({
        'Unit Code': ['RCB-51', 'SB-42', 'IB-33', 'MIB-12', 'TB-08',
                      'AB-15', 'RB-22', 'SIB-19', 'MTB-07', 'HQ-01'],
        'Workshop': ['PMABPS', 'KSBPS', '2SRIB', 'MECH1', 'ARM5',
                     'MAINT1', 'MAINT2', 'MAINT3', 'DSW-01', 'DSW-02']
    })
    
    return {
        'workshop_hierarchy': workshop_hierarchy,
        'vehicle_types': vehicle_types,
        'unit_workshop': unit_workshop
    }

def enrich_work_orders(df, mappings):
    """
    Enrich work orders with lookup data (Province, City, Brigade, Vehicle Type).
    """
    df_enriched = df.copy()
    
    # Add hierarchy fields from workshop mapping
    df_enriched = df_enriched.merge(
        mappings['workshop_hierarchy'][['Workshop', 'Brigade', 'City', 'Province']].drop_duplicates(),
        on='Workshop',
        how='left'
    )
    
    # Add Vehicle Type based on Vehicle ID pattern (mock logic)
    # TODO: Replace with actual vehicle type lookup from database
    def assign_vehicle_type(vehicle_id):
        vid_str = str(vehicle_id)
        if not vid_str or vid_str == 'nan':
            return 'Unknown'
        
        # Simple hash-based assignment for demo
        idx = int(vid_str) % len(mappings['vehicle_types'])
        return mappings['vehicle_types'][idx]
    
    df_enriched['Vehicle Type'] = df_enriched['Vehicle ID'].apply(assign_vehicle_type)
    
    return df_enriched

def apply_role_scope(df, role, allowed_workshops):
    """
    Apply role-based row-level security.
    """
    if role == 'Exec':
        # Full access to all data
        return df.copy()
    
    elif role == 'Fleet Manager':
        # Access to assigned workshops only
        if not allowed_workshops:
            # No workshops assigned - return empty dataframe with same structure
            return df.iloc[0:0].copy()
        return df[df['Workshop'].isin(allowed_workshops)].copy()
    
    elif role == 'Supervisor':
        # Access to single workshop only
        if not allowed_workshops or len(allowed_workshops) == 0:
            return df.iloc[0:0].copy()
        
        # Supervisor should have exactly one workshop
        supervisor_workshop = allowed_workshops[0]
        return df[df['Workshop'] == supervisor_workshop].copy()
    
    else:
        # Unknown role - no access
        return df.iloc[0:0].copy()

def apply_cascading_filters(df, filters):
    """
    Apply cascading lookup filters.
    Filters applied in order: Province -> City -> Brigade -> Workshop -> Unit Code -> Vehicle Type
    """
    df_filtered = df.copy()
    
    if filters.get('Province') and filters['Province'] != 'All':
        df_filtered = df_filtered[df_filtered['Province'] == filters['Province']]
    
    if filters.get('City') and filters['City'] != 'All':
        df_filtered = df_filtered[df_filtered['City'] == filters['City']]
    
    if filters.get('Brigade') and filters['Brigade'] != 'All':
        df_filtered = df_filtered[df_filtered['Brigade'] == filters['Brigade']]
    
    if filters.get('Workshop') and filters['Workshop'] != 'All':
        df_filtered = df_filtered[df_filtered['Workshop'] == filters['Workshop']]
    
    if filters.get('Unit Code') and filters['Unit Code'] != 'All':
        df_filtered = df_filtered[df_filtered['Unit Code'] == filters['Unit Code']]
    
    if filters.get('Vehicle Type') and filters['Vehicle Type'] != 'All':
        df_filtered = df_filtered[df_filtered['Vehicle Type'] == filters['Vehicle Type']]
    
    # Date range filter
    if filters.get('date_from'):
        df_filtered = df_filtered[df_filtered['Created Date'] >= filters['date_from']]
    
    if filters.get('date_to'):
        df_filtered = df_filtered[df_filtered['Created Date'] <= filters['date_to']]
    
    return df_filtered

def compute_derived_metrics(df):
    """
    Add derived metrics columns safely.
    """
    df_computed = df.copy()
    
    # Days Open (for all orders, based on Created Date)
    df_computed['Days Open'] = (datetime.now() - df_computed['Created Date']).dt.days
    
    # Queue Time = Start Date - MNG Creation Date (or Created Date if Start Date missing)
    df_computed['Queue Time (Days)'] = (
        df_computed['Start Date'] - df_computed['MNG Creation Date']
    ).dt.days
    
    # For orders without Start Date, queue time is ongoing
    df_computed.loc[df_computed['Start Date'].isna(), 'Queue Time (Days)'] = (
        datetime.now() - df_computed.loc[df_computed['Start Date'].isna(), 'MNG Creation Date']
    ).dt.days
    
    # Repair Time = Completion Date - Start Date
    df_computed['Repair Time (Days)'] = (
        df_computed['Completion Date'] - df_computed['Start Date']
    ).dt.days
    
    # Total Cycle Time = Completion Date - MNG Creation Date
    df_computed['Total Cycle Time (Days)'] = (
        df_computed['Completion Date'] - df_computed['MNG Creation Date']
    ).dt.days
    
    # Extract numeric priority level
    df_computed['Priority Level'] = df_computed['Priority'].str.extract(r'(\d+)')[0].astype(float)
    
    # Month and Week periods
    df_computed['Month'] = df_computed['Created Date'].dt.to_period('M').astype(str)
    df_computed['Week'] = df_computed['Created Date'].dt.to_period('W').astype(str)
    
    # Is Active flag - Based on actual IMSS system statuses
    # Active: Draft, Waiting Parts, Waiting FM Approval, Under Maintenance
    # Inactive: Completed, Rejected
    active_statuses = ['Draft', 'Waiting Parts', 'Waiting FM Approval', 'Under Maintenance']
    df_computed['Is Active'] = df_computed['Status'].isin(active_statuses)
    
    return df_computed

@st.cache_data(ttl=3600)
def load_and_process_data(file_path):
    """Load IMSS data and standardize column names"""
    try:
        df = pd.read_excel(file_path)
        
        # Column mapping
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
        
        df = df.rename(columns=column_mapping)
        
        # Ensure date columns are datetime
        date_columns = ['Created Date', 'MNG Creation Date', 'Start Date', 'Completion Date', 'Parts Receipt Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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

# ==================== DATA IMPORT SCREEN ====================

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
                        
                        df = load_and_process_data(data_path)
                        
                        # Build lookup mappings
                        mappings = build_lookup_mappings()
                        
                        # Enrich with hierarchy
                        df = enrich_work_orders(df, mappings)
                        
                        # Compute derived metrics
                        df = compute_derived_metrics(df)
                        
                        st.session_state['data_loaded'] = True
                        st.session_state['df'] = df
                        st.session_state['mappings'] = mappings
                        
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
                        
                        st.success("✅ Data loaded successfully! Configure your role and start analyzing.")
                        st.balloons()
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    with st.expander("🔍 Technical Details"):
                        st.code(traceback.format_exc())
        else:
            st.info("👆 Please upload the IMSS Excel export file")

# ==================== DASHBOARD FUNCTIONS ====================

def executive_overview_dashboard(df):
    """Executive Overview Dashboard"""
    st.markdown('<div class="dashboard-title">⭐ Executive Overview</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Provide command-level visibility into overall maintenance operations and fleet health.
        
        **Key Performance Indicators:**
        - **Total Work Orders**: Complete maintenance workload volume
        - **Active Orders**: Work currently in progress (not completed or closed)
        - **Critical/High Priority**: Urgent work requiring immediate attention (Priority 1-2)
        - **Completion Rate**: Percentage of work orders successfully closed
        - **Avg Cycle Time**: Average days from work order creation to completion
        
        **Visual Analytics:**
        - **Status Distribution**: Current state of all work orders
        - **Monthly Trend**: Workload patterns over time
        - **Workshop Performance**: Volume and efficiency by location
        
        **Use This Dashboard To:** Monitor overall fleet readiness, identify systemic bottlenecks, and track operational efficiency.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # Top KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Work Orders", f"{len(df):,}")
    
    with col2:
        active = len(df[df['Is Active']])
        st.metric("Active Orders", f"{active:,}", delta=f"{active/len(df)*100:.1f}%")
    
    with col3:
        critical = len(df[df['Priority Level'] <= 2])
        st.metric("Critical/High", f"{critical:,}", delta_color="inverse")
    
    with col4:
        completed = len(df[~df['Is Active']])
        st.metric("Completion Rate", f"{completed/len(df)*100:.1f}%")
    
    with col5:
        avg_cycle = df['Total Cycle Time (Days)'].mean()
        st.metric("Avg Cycle Time", f"{avg_cycle:.1f} days" if not pd.isna(avg_cycle) else "N/A")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Status Distribution")
        status_counts = df['Status'].value_counts()
        
        # IMSS System Status Colors
        status_colors = {
            'Draft': '#6c757d',                    # Gray
            'Waiting Parts': '#ffc107',            # Yellow
            'Waiting FM Approval': '#fd7e14',      # Orange
            'Under Maintenance': '#17a2b8',        # Cyan
            'Rejected': '#dc3545',                 # Red
            'Completed': '#28a745'                 # Green
        }
        
        fig = px.pie(values=status_counts.values, names=status_counts.index,
                    color=status_counts.index, color_discrete_map=status_colors, hole=0.4)
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
        'Total Cycle Time (Days)': 'mean'
    }).round(1).reset_index()
    workshop_stats.columns = ['Workshop', 'Work Orders', 'Avg Cycle Time']
    workshop_stats = workshop_stats.sort_values('Work Orders', ascending=False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name='Work Orders', x=workshop_stats['Workshop'],
                        y=workshop_stats['Work Orders'], marker_color='#2c5f2d'),
                 secondary_y=False)
    fig.add_trace(go.Scatter(name='Avg Cycle Time', x=workshop_stats['Workshop'],
                            y=workshop_stats['Avg Cycle Time'], mode='lines+markers',
                            marker=dict(size=10, color='red'), line=dict(width=3)),
                 secondary_y=True)
    fig.update_xaxes(title_text="Workshop")
    fig.update_yaxes(title_text="Work Orders", secondary_y=False)
    fig.update_yaxes(title_text="Avg Cycle Time (Days)", secondary_y=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def parts_analysis_dashboard(df):
    """Spare Parts Analysis"""
    st.markdown('<div class="dashboard-title">📦 Parts & Supply Chain Analysis</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Monitor spare parts requirements and supply chain performance to reduce equipment downtime.
        
        **Key Metrics:**
        - **Requires Parts**: Work orders that need spare parts (vs. labor-only repairs)
        - **Waiting for Parts**: Orders currently stuck waiting for parts delivery
        - **Parts Received**: Orders where required parts have arrived
        - **Parts Pending**: Orders needing parts that haven't been received yet
        - **Avg Wait Time**: Average days work orders spend in "Waiting Parts" status
        
        **Supply Chain Health Indicators:**
        - High "Waiting Parts" count = Supply chain bottleneck
        - Long avg wait time = Procurement or logistics issues
        - High parts requirement % = Equipment aging or poor reliability
        
        **Action Items:** Focus on reducing wait times through better inventory management and supplier relationships.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
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
        st.metric("Avg Wait Time", f"{avg_wait:.1f} days" if not pd.isna(avg_wait) else "N/A")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Parts Requirements Over Time")
        parts_trend = df[df['Requires Parts'] == True].groupby('Month').size().reset_index(name='Count')
        fig = px.line(parts_trend, x='Month', y='Count', markers=True)
        fig.update_traces(line_color='#2c5f2d', line_width=3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏢 Parts Waiting by Workshop")
        workshop_parts = df[df['Status'] == 'Waiting Parts'].groupby('Workshop').size().reset_index(name='Count')
        fig = px.bar(workshop_parts, x='Workshop', y='Count', color='Count',
                    color_continuous_scale='Oranges')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

def backlog_aging_dashboard(df):
    """Backlog Aging Dashboard"""
    st.markdown('<div class="dashboard-title">⏰ Backlog Aging Analysis</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Monitor open and waiting work orders by age to identify bottlenecks and prioritize action.
        
        **Key Metrics:**
        - **Open Backlog**: Work orders not yet completed or closed
        - **Waiting Parts**: Orders stuck waiting for spare parts delivery
        - **Avg Age**: Average days since work order creation
        - **Aging Buckets**: Work orders grouped by how long they've been open (0-2, 3-7, 8-14, 15-30, 30+ days)
        
        **How to Use:** Focus on orders aging beyond 14 days and high-priority items to prevent SLA breaches.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # Filter to active orders only
    df_active = df[df['Is Active']].copy()
    
    if len(df_active) == 0:
        st.info("No active work orders in the selected scope")
        return
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        open_backlog = len(df_active)
        st.metric("Open Backlog", f"{open_backlog:,}")
    
    with col2:
        waiting_parts = len(df_active[df_active['Status'] == 'Waiting Parts'])
        st.metric("Waiting Parts", f"{waiting_parts:,}")
    
    with col3:
        avg_age = df_active['Days Open'].mean()
        st.metric("Avg Age", f"{avg_age:.1f} days")
    
    with col4:
        critical_aged = len(df_active[(df_active['Priority Level'] <= 2) & (df_active['Days Open'] > 7)])
        st.metric("Critical Aged >7d", f"{critical_aged:,}", delta_color="inverse")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Aging buckets
    df_active['Age Bucket'] = pd.cut(
        df_active['Days Open'],
        bins=[0, 2, 7, 14, 30, float('inf')],
        labels=['0-2 days', '3-7 days', '8-14 days', '15-30 days', '30+ days']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Backlog by Aging Bucket")
        aging_dist = df_active['Age Bucket'].value_counts().sort_index()
        fig = px.bar(x=aging_dist.index, y=aging_dist.values, 
                    color=aging_dist.values, color_continuous_scale='Reds',
                    labels={'x': 'Age Bucket', 'y': 'Count'})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏢 Backlog by Workshop")
        workshop_backlog = df_active.groupby('Workshop').size().reset_index(name='Count')
        workshop_backlog = workshop_backlog.sort_values('Count', ascending=False)
        fig = px.bar(workshop_backlog, x='Workshop', y='Count', color='Count',
                    color_continuous_scale='Oranges')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Oldest work orders table
    st.subheader("🔴 Oldest 20 Active Work Orders")
    oldest = df_active.nlargest(20, 'Days Open')[
        ['WO Number', 'Workshop', 'Priority', 'Status', 'Days Open', 'Description']
    ].copy()
    
    # Color code by days open instead of using background_gradient
    def color_age(row):
        if row['Days Open'] > 30:
            return ['background-color: #ffcccc'] * len(row)
        elif row['Days Open'] > 14:
            return ['background-color: #ffe6cc'] * len(row)
        else:
            return ['background-color: #ffffcc'] * len(row)
    
    st.dataframe(oldest, use_container_width=True, height=400)

def lifecycle_dashboard(df):
    """Work Order Lifecycle Dashboard (Queue vs Repair vs Total)"""
    st.markdown('<div class="dashboard-title">🔄 Work Order Lifecycle Analysis</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Understand where time is spent in the maintenance process to optimize workflow efficiency.
        
        **Key Metrics:**
        - **Queue Time**: Time from work order creation (MNG system) to when work actually starts
          - *Calculation:* Start Date - MNG Creation Date
        - **Repair Time**: Actual time spent working on the repair
          - *Calculation:* Completion Date - Start Date
        - **Total Cycle Time**: End-to-end time from creation to completion
          - *Calculation:* Completion Date - MNG Creation Date
        
        **How to Use:** High queue times indicate capacity issues. High repair times suggest complexity or resource constraints.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # Filter to completed orders with valid dates
    df_completed = df[~df['Is Active'] & df['Total Cycle Time (Days)'].notna()].copy()
    
    if len(df_completed) == 0:
        st.info("No completed work orders with lifecycle data in the selected scope")
        return
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_queue = df_completed['Queue Time (Days)'].mean()
        st.metric("Avg Queue Time", f"{avg_queue:.1f} days" if not pd.isna(avg_queue) else "N/A")
    
    with col2:
        avg_repair = df_completed['Repair Time (Days)'].mean()
        st.metric("Avg Repair Time", f"{avg_repair:.1f} days" if not pd.isna(avg_repair) else "N/A")
    
    with col3:
        avg_total = df_completed['Total Cycle Time (Days)'].mean()
        st.metric("Avg Total Cycle Time", f"{avg_total:.1f} days")
    
    with col4:
        queue_pct = (avg_queue / avg_total * 100) if not pd.isna(avg_queue) and avg_total > 0 else 0
        st.metric("Queue % of Total", f"{queue_pct:.1f}%")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Workshop comparison
    st.subheader("🏢 Workshop Comparison: Queue vs Repair Time")
    workshop_lifecycle = df_completed.groupby('Workshop').agg({
        'Queue Time (Days)': 'mean',
        'Repair Time (Days)': 'mean',
        'Total Cycle Time (Days)': 'mean'
    }).round(1).reset_index()
    workshop_lifecycle.columns = ['Workshop', 'Avg Queue', 'Avg Repair', 'Avg Total']
    workshop_lifecycle = workshop_lifecycle.sort_values('Avg Total', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Queue Time', x=workshop_lifecycle['Workshop'], 
                        y=workshop_lifecycle['Avg Queue'], marker_color='#ffc107'))
    fig.add_trace(go.Bar(name='Repair Time', x=workshop_lifecycle['Workshop'],
                        y=workshop_lifecycle['Avg Repair'], marker_color='#2c5f2d'))
    fig.update_layout(barmode='stack', height=400, xaxis_title="Workshop", 
                     yaxis_title="Days")
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend over time
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Queue Time Trend")
        queue_trend = df_completed.groupby('Month')['Queue Time (Days)'].mean().reset_index()
        fig = px.line(queue_trend, x='Month', y='Queue Time (Days)', markers=True)
        fig.update_traces(line_color='#ffc107', line_width=3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Repair Time Trend")
        repair_trend = df_completed.groupby('Month')['Repair Time (Days)'].mean().reset_index()
        fig = px.line(repair_trend, x='Month', y='Repair Time (Days)', markers=True)
        fig.update_traces(line_color='#2c5f2d', line_width=3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

def priority_risk_dashboard(df):
    """Priority & Risk Dashboard"""
    st.markdown('<div class="dashboard-title">⚠️ Priority & Risk Analysis</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Track and manage high-priority work orders to prevent critical equipment downtime.
        
        **Priority Levels:**
        - **Priority 1 - Critical**: Mission-critical, immediate attention required
        - **Priority 2 - High**: Important, may impact operations
        - **Priority 3 - Normal**: Standard maintenance work
        - **Priority 4 - Low**: Can be deferred
        - **Priority 5 - Planning**: Scheduled future work
        
        **Key Focus:** Monitor P1/P2 open work orders and their aging to ensure timely resolution.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_pri_open = len(df[(df['Priority Level'] <= 2) & df['Is Active']])
        st.metric("Critical/High Priority Open", f"{high_pri_open:,}")
    
    with col2:
        high_pri_age = df[(df['Priority Level'] <= 2) & df['Is Active']]['Days Open'].mean()
        st.metric("Avg Age (P1/P2)", f"{high_pri_age:.1f} days" if not pd.isna(high_pri_age) else "N/A")
    
    with col3:
        p1_count = len(df[(df['Priority Level'] == 1) & df['Is Active']])
        st.metric("Priority 1 Critical", f"{p1_count:,}", delta_color="inverse")
    
    with col4:
        total_active = len(df[df['Is Active']])
        high_pri_pct = (high_pri_open / total_active * 100) if total_active > 0 else 0
        st.metric("% High Priority", f"{high_pri_pct:.1f}%")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Priority distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Priority Distribution")
        priority_dist = df['Priority'].value_counts().sort_index()
        priority_colors = {
            '1 - Critical': '#dc3545',
            '2 - High': '#fd7e14',
            '3 - Normal': '#ffc107',
            '4 - Low': '#28a745',
            '5 - Planning': '#17a2b8'
        }
        fig = px.pie(values=priority_dist.values, names=priority_dist.index,
                    color=priority_dist.index, color_discrete_map=priority_colors, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Avg Cycle Time by Priority")
        priority_cycle = df.groupby('Priority')['Total Cycle Time (Days)'].mean().reset_index()
        priority_cycle = priority_cycle.sort_values('Priority')
        fig = px.bar(priority_cycle, x='Priority', y='Total Cycle Time (Days)',
                    color='Total Cycle Time (Days)', color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # High priority open work orders table
    st.subheader("🔴 Critical/High Priority Open Work Orders")
    high_pri = df[(df['Priority Level'] <= 2) & df['Is Active']].nlargest(20, 'Days Open')[
        ['WO Number', 'Priority', 'Workshop', 'Status', 'Days Open', 'Description']
    ]
    
    if len(high_pri) > 0:
        st.dataframe(high_pri, use_container_width=True, height=400)
    else:
        st.info("No high priority open work orders")

def preventive_corrective_dashboard(df):
    """Preventive vs Corrective Dashboard"""
    st.markdown('<div class="dashboard-title">🔧 Preventive vs Corrective Maintenance</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Monitor the balance between preventive and corrective maintenance to improve reliability.
        
        **Maintenance Types:**
        - **Preventive (PM)**: Scheduled maintenance to prevent failures
        - **Corrective (CM)**: Repairs after equipment failure
        - **Emergency**: Unplanned urgent repairs
        - **Modification**: Equipment upgrades or changes
        
        **Target Ratio:** Industry best practice aims for 30-40% preventive maintenance.
        **PM/CM Ratio Calculation:** PM Count / CM Count
        
        **How to Use:** Higher PM ratios indicate proactive maintenance culture and better equipment reliability.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # KPIs
    preventive_count = len(df[df['Maintenance Type'] == 'Preventive'])
    corrective_count = len(df[df['Maintenance Type'] == 'Corrective'])
    total = len(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Preventive", f"{preventive_count:,}", delta=f"{preventive_count/total*100:.1f}%")
    
    with col2:
        st.metric("Corrective", f"{corrective_count:,}", delta=f"{corrective_count/total*100:.1f}%")
    
    with col3:
        ratio = (preventive_count / corrective_count) if corrective_count > 0 else 0
        st.metric("PM/CM Ratio", f"{ratio:.2f}")
    
    with col4:
        target_ratio = 0.30  # 30% PM target
        actual_ratio = preventive_count / total if total > 0 else 0
        delta = actual_ratio - target_ratio
        st.metric("PM % (Target 30%)", f"{actual_ratio*100:.1f}%", 
                 delta=f"{delta*100:+.1f}%", 
                 delta_color="normal" if delta >= 0 else "inverse")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Maintenance Type Distribution")
        maint_dist = df['Maintenance Type'].value_counts()
        fig = px.pie(values=maint_dist.values, names=maint_dist.index, hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Avg Cycle Time by Type")
        cycle_by_type = df.groupby('Maintenance Type')['Total Cycle Time (Days)'].mean().reset_index()
        fig = px.bar(cycle_by_type, x='Maintenance Type', y='Total Cycle Time (Days)',
                    color='Total Cycle Time (Days)', color_continuous_scale='Blues')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trend over time
    st.subheader("📈 Preventive vs Corrective Trend")
    monthly_trend = df.groupby(['Month', 'Maintenance Type']).size().reset_index(name='Count')
    fig = px.line(monthly_trend, x='Month', y='Count', color='Maintenance Type', markers=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Parts dependency by maintenance type
    st.subheader("📦 Parts Dependency by Maintenance Type")
    parts_by_type = df.groupby('Maintenance Type').agg({
        'Requires Parts': lambda x: (x == True).sum() / len(x) * 100
    }).round(1).reset_index()
    parts_by_type.columns = ['Maintenance Type', '% Requires Parts']
    
    fig = px.bar(parts_by_type, x='Maintenance Type', y='% Requires Parts',
                color='% Requires Parts', color_continuous_scale='Oranges',
                text='% Requires Parts')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def repeat_issues_dashboard(df):
    """Repeat Issues Dashboard"""
    st.markdown('<div class="dashboard-title">🔁 Repeat Issues Analysis</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Identify vehicles and issues requiring repeated maintenance to address root causes.
        
        **Key Indicators:**
        - **Top Vehicles by WO Count**: Vehicles requiring the most maintenance attention
        - **Top Issue Descriptions**: Most frequently occurring maintenance problems
        - **Repeat Issues (30-day window)**: Same vehicle with same issue multiple times recently
        
        **Why It Matters:**
        - Repeat issues indicate incomplete repairs or underlying systemic problems
        - High-frequency vehicles may need major overhaul or replacement
        - Common issues may benefit from design changes or improved procedures
        
        **Action Items:** Investigate top repeat offenders for root cause analysis.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # Top vehicles by WO count
    vehicle_counts = df.groupby('Vehicle ID').size().reset_index(name='WO Count')
    vehicle_counts = vehicle_counts.sort_values('WO Count', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_vehicle = vehicle_counts.iloc[0] if len(vehicle_counts) > 0 else None
        if top_vehicle is not None:
            st.metric("Top Vehicle", f"ID: {top_vehicle['Vehicle ID']}", 
                     delta=f"{top_vehicle['WO Count']} WOs")
    
    with col2:
        avg_wo_per_vehicle = vehicle_counts['WO Count'].mean()
        st.metric("Avg WOs per Vehicle", f"{avg_wo_per_vehicle:.1f}")
    
    with col3:
        high_repeat = len(vehicle_counts[vehicle_counts['WO Count'] > 5])
        st.metric("Vehicles >5 WOs", f"{high_repeat:,}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Top 10 vehicles
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚗 Top 10 Vehicles by Work Order Count")
        top_10_vehicles = vehicle_counts.head(10)
        fig = px.bar(top_10_vehicles, y='Vehicle ID', x='WO Count', orientation='h',
                    color='WO Count', color_continuous_scale='Reds')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📝 Top 10 Issue Descriptions")
        desc_counts = df['Description'].value_counts().head(10).reset_index()
        desc_counts.columns = ['Description', 'Count']
        # Truncate long descriptions
        desc_counts['Description Short'] = desc_counts['Description'].str[:40] + '...'
        fig = px.bar(desc_counts, y='Description Short', x='Count', orientation='h',
                    color='Count', color_continuous_scale='Oranges')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Repeat issues within 30 days
    st.subheader("🔴 Vehicles with Repeated Same Description (Last 30 Days)")
    
    # Filter last 30 days
    df_recent = df[df['Created Date'] >= (datetime.now() - timedelta(days=30))].copy()
    
    # Find vehicles with same description multiple times
    repeat_issues = df_recent.groupby(['Vehicle ID', 'Description']).agg({
        'WO Number': 'count',
        'Created Date': ['min', 'max']
    }).reset_index()
    repeat_issues.columns = ['Vehicle ID', 'Description', 'Occurrences', 'First Date', 'Last Date']
    repeat_issues = repeat_issues[repeat_issues['Occurrences'] > 1].sort_values('Occurrences', ascending=False)
    
    if len(repeat_issues) > 0:
        repeat_issues['Description Short'] = repeat_issues['Description'].str[:50] + '...'
        st.dataframe(
            repeat_issues[['Vehicle ID', 'Description Short', 'Occurrences', 'First Date', 'Last Date']].head(20),
            use_container_width=True,
            height=400
        )
    else:
        st.info("No repeat issues found in the last 30 days")

def technician_productivity_dashboard(df):
    """Technician Productivity Dashboard"""
    st.markdown('<div class="dashboard-title">👨‍🔧 Technician Productivity Analysis</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Evaluate technician performance to identify training needs and recognize top performers.
        
        **Key Metrics:**
        - **Completed WOs**: Total work orders completed by each technician
        - **Avg Repair Time**: Average days to complete repairs (lower is better)
        - **Avg Queue Time**: Average wait time before work starts (workshop capacity indicator)
        - **High Priority WOs**: Count of critical/high priority work handled
        
        **Performance Balance:**
        - High completion count + Low repair time = Highly efficient technician
        - High completion count + High repair time = May be handling complex repairs
        - Low completion count = May need additional training or have capacity constraints
        
        **How to Use:** Balance workload across technicians and identify top performers for recognition.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # Filter to work orders with assigned technicians
    df_assigned = df[df['Assigned To'].notna()].copy()
    
    if len(df_assigned) == 0:
        st.info("No work orders with assigned technicians in the selected scope")
        return
    
    # Calculate metrics per technician
    tech_stats = df_assigned.groupby('Assigned To').agg({
        'WO Number': 'count',
        'Repair Time (Days)': 'mean',
        'Queue Time (Days)': 'mean',
        'Total Cycle Time (Days)': 'mean'
    }).round(1).reset_index()
    tech_stats.columns = ['Technician', 'Completed WOs', 'Avg Repair Time', 'Avg Queue Time', 'Avg Total Time']
    
    # Count high priority WOs handled
    high_pri_counts = df_assigned[df_assigned['Priority Level'] <= 2].groupby('Assigned To').size()
    tech_stats = tech_stats.merge(
        high_pri_counts.reset_index().rename(columns={0: 'High Priority WOs'}),
        left_on='Technician',
        right_on='Assigned To',
        how='left'
    )
    tech_stats['High Priority WOs'] = tech_stats['High Priority WOs'].fillna(0).astype(int)
    tech_stats = tech_stats.drop('Assigned To_y', axis=1, errors='ignore')
    tech_stats = tech_stats.rename(columns={'Assigned To_x': 'Technician'}, errors='ignore')
    
    # Remove Assigned To duplicate column if exists
    if 'Assigned To' in tech_stats.columns:
        tech_stats = tech_stats.drop('Assigned To', axis=1)
    
    tech_stats = tech_stats.sort_values('Completed WOs', ascending=False)
    
    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_techs = len(tech_stats)
        st.metric("Active Technicians", f"{total_techs:,}")
    
    with col2:
        avg_wo_per_tech = tech_stats['Completed WOs'].mean()
        st.metric("Avg WOs per Tech", f"{avg_wo_per_tech:.1f}")
    
    with col3:
        top_tech = tech_stats.iloc[0] if len(tech_stats) > 0 else None
        if top_tech is not None:
            st.metric("Top Performer", f"{top_tech['Technician']}", 
                     delta=f"{int(top_tech['Completed WOs'])} WOs")
    
    with col4:
        avg_repair = tech_stats['Avg Repair Time'].mean()
        st.metric("Fleet Avg Repair Time", f"{avg_repair:.1f} days" if not pd.isna(avg_repair) else "N/A")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Leaderboard
    st.subheader("🏆 Technician Leaderboard")
    st.dataframe(
        tech_stats.head(20),
        use_container_width=True,
        height=400
    )
    
    # Scatter plot: Completed count vs Avg repair time
    st.subheader("📊 Productivity vs Speed")
    fig = px.scatter(tech_stats, x='Completed WOs', y='Avg Repair Time',
                    size='High Priority WOs', color='Avg Total Time',
                    hover_data=['Technician'], color_continuous_scale='RdYlGn_r',
                    labels={'Avg Repair Time': 'Avg Repair Time (Days)'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def data_quality_dashboard(df):
    """Data Quality & Compliance Dashboard"""
    st.markdown('<div class="dashboard-title">✅ Data Quality & Compliance</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Monitor data completeness and integrity to ensure accurate reporting and compliance.
        
        **Critical Quality Checks:**
        - **Missing Start Date**: Work orders without a recorded start time
        - **Missing Completion Date**: Completed/Closed orders missing closure date (status mismatch)
        - **Missing Technician Notes**: Work orders without repair documentation
        - **Missing Assigned To**: Work orders without technician assignment
        
        **Data Quality Score Calculation:**
        - Average of completeness percentages across all fields
        - Target: >95% completeness for operational excellence
        
        **Compliance Impacts:**
        - Poor data quality affects analytics accuracy
        - Missing completion dates cause incorrect cycle time calculations
        - Undocumented work creates audit and accountability issues
        
        **Action Items:** Address problematic records shown in the table below.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # Calculate quality metrics
    total = len(df)
    
    missing_start = df['Start Date'].isna().sum()
    pct_missing_start = (missing_start / total * 100)
    
    completed_df = df[df['Status'].isin(['Completed', 'Closed'])]
    if len(completed_df) > 0:
        missing_completion = completed_df['Completion Date'].isna().sum()
        pct_missing_completion = (missing_completion / len(completed_df) * 100)
    else:
        missing_completion = 0
        pct_missing_completion = 0
    
    missing_notes = df['Technician Notes'].isna().sum()
    pct_missing_notes = (missing_notes / total * 100)
    
    missing_assigned = df['Assigned To'].isna().sum()
    pct_missing_assigned = (missing_assigned / total * 100)
    
    # Status mismatch
    status_mismatch = len(df[df['Status'].isin(['Completed', 'Closed']) & df['Completion Date'].isna()])
    
    # KPI Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Data Quality Score", 
                 f"{100 - ((pct_missing_start + pct_missing_completion + pct_missing_notes)/3):.1f}%")
    
    with col2:
        st.metric("Critical Issues", f"{status_mismatch:,}", delta_color="inverse")
    
    with col3:
        complete_records = total - missing_start - missing_completion - missing_notes - missing_assigned
        pct_complete = (complete_records / total * 100) if total > 0 else 0
        st.metric("Complete Records", f"{pct_complete:.1f}%")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Quality metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Completeness Metrics")
        
        quality_data = pd.DataFrame({
            'Field': ['Start Date', 'Completion Date (Completed)', 'Technician Notes', 'Assigned To'],
            'Missing': [missing_start, missing_completion, missing_notes, missing_assigned],
            'Total': [total, len(completed_df), total, total]
        })
        quality_data['% Complete'] = (1 - quality_data['Missing'] / quality_data['Total']) * 100
        
        fig = px.bar(quality_data, x='Field', y='% Complete', 
                    color='% Complete', color_continuous_scale='RdYlGn',
                    text='% Complete')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=350, yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Data Quality Issues")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Missing Start Date", f"{missing_start:,}", 
                     delta=f"{pct_missing_start:.1f}%", delta_color="inverse")
        with col_b:
            st.metric("Missing Completion", f"{missing_completion:,}",
                     delta=f"{pct_missing_completion:.1f}%", delta_color="inverse")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Missing Notes", f"{missing_notes:,}",
                     delta=f"{pct_missing_notes:.1f}%", delta_color="inverse")
        with col_b:
            st.metric("Missing Assigned To", f"{missing_assigned:,}",
                     delta=f"{pct_missing_assigned:.1f}%", delta_color="inverse")
    
    # Problematic records
    st.subheader("🔴 Problematic Records")
    
    problematic = df[
        (df['Status'].isin(['Completed', 'Closed']) & df['Completion Date'].isna()) |
        (df['Start Date'].notna() & df['Assigned To'].isna())
    ][['WO Number', 'Workshop', 'Status', 'Start Date', 'Completion Date', 'Assigned To']].head(20)
    
    if len(problematic) > 0:
        st.dataframe(problematic, use_container_width=True, height=400)
    else:
        st.success("✅ No critical data quality issues found!")

def owning_unit_dashboard(df):
    """Owning Unit Dashboard"""
    st.markdown('<div class="dashboard-title">🏢 Owning Unit Analysis</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Analyze maintenance patterns by military unit to identify high-demand units and support planning.
        
        **Key Metrics:**
        - **WO Count by Unit**: Total maintenance workload per unit
        - **Avg Cycle Time by Unit**: How long repairs take for each unit's vehicles
        - **Priority Mix**: Distribution of urgent vs. routine work by unit
        
        **Strategic Insights:**
        - High WO volume units may need additional equipment or preventive maintenance focus
        - Long cycle times may indicate parts availability issues or complex equipment
        - Priority distribution shows operational tempo and equipment reliability
        
        **Use Cases:**
        - Resource allocation planning
        - Unit readiness assessment
        - Equipment replacement prioritization
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # Unit statistics
    unit_stats = df.groupby('Unit Code').agg({
        'WO Number': 'count',
        'Total Cycle Time (Days)': 'mean',
        'Priority Level': lambda x: (x <= 2).sum()
    }).round(1).reset_index()
    unit_stats.columns = ['Unit Code', 'WO Count', 'Avg Cycle Time', 'High Priority Count']
    unit_stats = unit_stats.sort_values('WO Count', ascending=False)
    
    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_units = len(unit_stats)
        st.metric("Total Units", f"{total_units:,}")
    
    with col2:
        top_unit = unit_stats.iloc[0] if len(unit_stats) > 0 else None
        if top_unit is not None:
            st.metric("Top Unit by Volume", f"{top_unit['Unit Code']}", 
                     delta=f"{int(top_unit['WO Count'])} WOs")
    
    with col3:
        avg_wo_per_unit = unit_stats['WO Count'].mean()
        st.metric("Avg WOs per Unit", f"{avg_wo_per_unit:.1f}")
    
    with col4:
        worst_cycle = unit_stats.nlargest(1, 'Avg Cycle Time').iloc[0] if len(unit_stats) > 0 else None
        if worst_cycle is not None:
            st.metric("Longest Avg Cycle", f"{worst_cycle['Unit Code']}", 
                     delta=f"{worst_cycle['Avg Cycle Time']:.1f} days", delta_color="inverse")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Top 10 Units by WO Volume")
        top_10 = unit_stats.head(10)
        fig = px.bar(top_10, y='Unit Code', x='WO Count', orientation='h',
                    color='WO Count', color_continuous_scale='Blues')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Worst 10 Units by Cycle Time")
        worst_10 = unit_stats.nlargest(10, 'Avg Cycle Time')
        fig = px.bar(worst_10, y='Unit Code', x='Avg Cycle Time', orientation='h',
                    color='Avg Cycle Time', color_continuous_scale='Reds')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Priority mix by unit
    st.subheader("🎯 Priority Mix by Unit")
    priority_mix = df.groupby(['Unit Code', 'Priority']).size().reset_index(name='Count')
    priority_mix = priority_mix[priority_mix['Unit Code'].isin(unit_stats.head(10)['Unit Code'])]
    
    fig = px.bar(priority_mix, x='Unit Code', y='Count', color='Priority',
                color_discrete_map={
                    '1 - Critical': '#dc3545',
                    '2 - High': '#fd7e14',
                    '3 - Normal': '#ffc107',
                    '4 - Low': '#28a745',
                    '5 - Planning': '#17a2b8'
                })
    fig.update_layout(height=400, barmode='stack')
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed unit table
    st.subheader("📋 Unit Performance Summary")
    st.dataframe(unit_stats, use_container_width=True, height=400)

def vehicle_mileage_dashboard(df):
    """Vehicle Mileage Dashboard"""
    st.markdown('<div class="dashboard-title">🚗 Vehicle Mileage Analysis</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Correlate maintenance patterns with vehicle mileage to optimize fleet replacement and service intervals.
        
        **Mileage Buckets:**
        - **0-20k km**: New vehicles, break-in period
        - **20-50k km**: Early operational life
        - **50-100k km**: Mid-life, increased maintenance expected
        - **100k+ km**: High-mileage, consider replacement
        
        **Key Analyses:**
        - **WO Count by Mileage**: Maintenance frequency increases with age/mileage
        - **Avg Cycle Time by Mileage**: Complex repairs increase with vehicle age
        - **Mileage-Cycle Correlation**: Statistical relationship between mileage and repair time
        
        **Strategic Decisions:**
        - High-mileage vehicles with frequent repairs are replacement candidates
        - Unusual patterns in low-mileage vehicles indicate quality or operational issues
        - Plan preventive maintenance schedules based on mileage thresholds
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # Create mileage buckets
    df_mileage = df[df['Mileage'].notna()].copy()
    
    if len(df_mileage) == 0:
        st.info("No mileage data available in the selected scope")
        return
    
    df_mileage['Mileage Bucket'] = pd.cut(
        df_mileage['Mileage'],
        bins=[0, 20000, 50000, 100000, float('inf')],
        labels=['0-20k', '20-50k', '50-100k', '100k+']
    )
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_mileage = df_mileage['Mileage'].mean()
        st.metric("Avg Fleet Mileage", f"{avg_mileage:,.0f} km")
    
    with col2:
        high_mileage = len(df_mileage[df_mileage['Mileage'] > 100000])
        st.metric("High Mileage (>100k)", f"{high_mileage:,}")
    
    with col3:
        low_mileage_wo = len(df_mileage[df_mileage['Mileage'] < 20000])
        pct_low = (low_mileage_wo / len(df_mileage) * 100)
        st.metric("Low Mileage WOs (<20k)", f"{low_mileage_wo:,}", delta=f"{pct_low:.1f}%")
    
    with col4:
        correlation = df_mileage[['Mileage', 'Total Cycle Time (Days)']].corr().iloc[0, 1]
        st.metric("Mileage-Cycle Correlation", f"{correlation:.2f}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 WO Count by Mileage Bucket")
        bucket_counts = df_mileage['Mileage Bucket'].value_counts().sort_index()
        fig = px.bar(x=bucket_counts.index, y=bucket_counts.values,
                    color=bucket_counts.values, color_continuous_scale='Blues',
                    labels={'x': 'Mileage Bucket', 'y': 'Count'})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Avg Cycle Time by Mileage Bucket")
        bucket_cycle = df_mileage.groupby('Mileage Bucket')['Total Cycle Time (Days)'].mean().reset_index()
        fig = px.bar(bucket_cycle, x='Mileage Bucket', y='Total Cycle Time (Days)',
                    color='Total Cycle Time (Days)', color_continuous_scale='Reds')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot: Mileage vs Cycle Time
    st.subheader("📈 Mileage vs Cycle Time Correlation")
    sample_df = df_mileage.sample(min(500, len(df_mileage)))
    fig = px.scatter(sample_df, x='Mileage', y='Total Cycle Time (Days)',
                    color='Priority Level', size='Days Open',
                    hover_data=['Vehicle ID', 'Workshop'],
                    color_continuous_scale='RdYlGn_r')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def process_mining_dashboard(df):
    """Process Mining Dashboard - Track work order flow from start to finish"""
    st.markdown('<div class="dashboard-title">🔄 Process Mining & Bottleneck Analysis</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Visualize and analyze the end-to-end work order process to identify bottlenecks and optimization opportunities.
        
        **Process Mining Capabilities:**
        - **Process Flow Visualization**: See how work orders flow through different statuses
        - **Bottleneck Identification**: Identify stages where work orders get stuck
        - **Time Analysis**: Measure time spent in each stage
        - **Process Variants**: Discover different paths work orders take
        - **Transition Analysis**: Understand status change patterns
        
        **Key Stages:**
        1. **Draft** → Work order created, not yet started
        2. **Waiting Parts** → Stuck waiting for spare parts
        3. **Waiting FM Approval** → Pending Fleet Manager approval
        4. **Under Maintenance** → Active repair work in workshop
        5. **Completed** → Work finished
        6. **Rejected** → Order rejected/cancelled
        
        **How to Use:** Identify which stages have longest durations and highest volumes to focus improvement efforts.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # For process mining, we need to reconstruct the status journey
    # Since we don't have historical status changes, we'll use current status and dates to infer process
    
    # Calculate stage durations (approximations based on available dates)
    df_process = df.copy()
    
    # Define process stages based on IMSS system statuses
    status_order = ['Draft', 'Waiting Parts', 'Waiting FM Approval', 'Under Maintenance', 'Rejected', 'Completed']
    
    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_wo = len(df_process)
        st.metric("Total Work Orders", f"{total_wo:,}")
    
    with col2:
        avg_duration = df_process['Total Cycle Time (Days)'].mean()
        st.metric("Avg Process Duration", f"{avg_duration:.1f} days" if not pd.isna(avg_duration) else "N/A")
    
    with col3:
        stuck_in_parts = len(df_process[df_process['Status'] == 'Waiting Parts'])
        st.metric("Stuck Waiting Parts", f"{stuck_in_parts:,}", delta_color="inverse")
    
    with col4:
        completed_wo = len(df_process[df_process['Status'].isin(['Completed', 'Closed'])])
        completion_rate = (completed_wo / total_wo * 100)
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Process Flow Visualization (Sankey Diagram)
    st.subheader("🔀 Process Flow Diagram")
    
    # Create simplified process flow based on current status
    # We'll show transitions from Created → Current Status → Final Status
    
    status_dist = df_process['Status'].value_counts()
    
    # Create Sankey diagram data
    # Source: Start, Target: Current Status
    labels = ['Start'] + status_order
    
    source_indices = []
    target_indices = []
    values = []
    colors = []
    
    color_map = {
        'Draft': 'rgba(108, 117, 125, 0.4)',         # Gray
        'Waiting Parts': 'rgba(255, 193, 7, 0.4)',   # Yellow  
        'Waiting FM Approval': 'rgba(253, 126, 20, 0.4)',  # Orange
        'Under Maintenance': 'rgba(23, 162, 184, 0.4)',    # Cyan
        'Rejected': 'rgba(220, 53, 69, 0.4)',        # Red
        'Completed': 'rgba(40, 167, 69, 0.4)'        # Green
    }
    
    for status in status_order:
        if status in status_dist.index:
            source_indices.append(0)  # From Start
            target_indices.append(labels.index(status))
            values.append(status_dist[status])
            colors.append(color_map.get(status, 'rgba(128, 128, 128, 0.4)'))
    
    # Add transitions from intermediate statuses to Completed
    # Assume: Draft, Waiting Parts, Waiting FM Approval, Under Maintenance → Completed
    intermediate_statuses = ['Draft', 'Waiting Parts', 'Waiting FM Approval', 'Under Maintenance']
    for status in intermediate_statuses:
        if status in status_dist.index and 'Completed' in status_dist.index:
            # Add flow to Completed
            source_indices.append(labels.index(status))
            target_indices.append(labels.index('Completed'))
            values.append(int(status_dist[status] * 0.6))  # Assume 60% complete
            colors.append(color_map.get('Completed', 'rgba(40, 167, 69, 0.4)'))
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=['#2c5f2d', '#6c757d', '#ffc107', '#fd7e14', 
                   '#17a2b8', '#dc3545', '#28a745']
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=colors
        )
    )])
    
    fig.update_layout(
        title="Work Order Process Flow",
        font_size=12,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Time Analysis by Stage
    st.subheader("⏱️ Time Analysis by Process Stage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Current Status Distribution**")
        status_dist_df = status_dist.reset_index()
        status_dist_df.columns = ['Status', 'Count']
        
        fig = px.bar(status_dist_df, x='Status', y='Count',
                    color='Status',
                    color_discrete_map={
                        'Open': '#dc3545',
                        'In Progress': '#ffc107',
                        'Waiting Parts': '#fd7e14',
                        'Under Maintenance': '#17a2b8',
                        'Completed': '#28a745',
                        'Closed': '#20c997'
                    },
                    text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**⏰ Average Time in Each Status**")
        
        # Calculate avg days in current status (approximation)
        time_in_status = df_process.groupby('Status')['Days Open'].mean().reset_index()
        time_in_status.columns = ['Status', 'Avg Days']
        time_in_status = time_in_status.sort_values('Avg Days', ascending=False)
        
        fig = px.bar(time_in_status, x='Status', y='Avg Days',
                    color='Avg Days', color_continuous_scale='Reds',
                    text='Avg Days')
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(height=350, showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottleneck Analysis
    st.subheader("🚨 Bottleneck Identification")
    
    col1, col2, col3 = st.columns(3)
    
    # Identify bottlenecks (statuses with high count AND high duration)
    bottleneck_scores = df_process.groupby('Status').agg({
        'WO Number': 'count',
        'Days Open': 'mean'
    }).reset_index()
    bottleneck_scores.columns = ['Status', 'Count', 'Avg Days']
    
    # Normalize and calculate bottleneck score
    bottleneck_scores['Count Norm'] = (bottleneck_scores['Count'] - bottleneck_scores['Count'].min()) / (bottleneck_scores['Count'].max() - bottleneck_scores['Count'].min())
    bottleneck_scores['Days Norm'] = (bottleneck_scores['Avg Days'] - bottleneck_scores['Avg Days'].min()) / (bottleneck_scores['Avg Days'].max() - bottleneck_scores['Avg Days'].min())
    bottleneck_scores['Bottleneck Score'] = (bottleneck_scores['Count Norm'] * 0.5 + bottleneck_scores['Days Norm'] * 0.5) * 100
    bottleneck_scores = bottleneck_scores.sort_values('Bottleneck Score', ascending=False)
    
    with col1:
        if len(bottleneck_scores) > 0:
            top_bottleneck = bottleneck_scores.iloc[0]
            st.metric("Primary Bottleneck", 
                     top_bottleneck['Status'],
                     delta=f"Score: {top_bottleneck['Bottleneck Score']:.0f}",
                     delta_color="inverse")
    
    with col2:
        if len(bottleneck_scores) > 1:
            second_bottleneck = bottleneck_scores.iloc[1]
            st.metric("Secondary Bottleneck",
                     second_bottleneck['Status'],
                     delta=f"Score: {second_bottleneck['Bottleneck Score']:.0f}",
                     delta_color="inverse")
    
    with col3:
        # Show status with longest individual orders
        longest_status = df_process.groupby('Status')['Days Open'].max().idxmax()
        longest_days = df_process.groupby('Status')['Days Open'].max().max()
        st.metric("Longest Single Order",
                 longest_status,
                 delta=f"{longest_days:.0f} days",
                 delta_color="inverse")
    
    # Bottleneck details table
    st.markdown("**🔍 Bottleneck Analysis Details**")
    bottleneck_display = bottleneck_scores[['Status', 'Count', 'Avg Days', 'Bottleneck Score']].copy()
    bottleneck_display['Bottleneck Score'] = bottleneck_display['Bottleneck Score'].round(1)
    bottleneck_display['Avg Days'] = bottleneck_display['Avg Days'].round(1)
    
    # Add severity classification
    bottleneck_display['Severity'] = bottleneck_display['Bottleneck Score'].apply(
        lambda x: '🔴 Critical' if x > 70 else ('🟡 Moderate' if x > 40 else '🟢 Low')
    )
    
    st.dataframe(bottleneck_display, use_container_width=True, height=250)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Process Variants Analysis
    st.subheader("🔀 Process Variants & Paths")
    
    # Group by maintenance type and status to show different process paths
    variants = df_process.groupby(['Maintenance Type', 'Status']).size().reset_index(name='Count')
    variants = variants.sort_values('Count', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🛤️ Process Paths by Maintenance Type**")
        
        # Top 5 most common paths
        top_variants = variants.head(10)
        top_variants['Path'] = top_variants['Maintenance Type'] + ' → ' + top_variants['Status']
        
        fig = px.bar(top_variants, y='Path', x='Count', orientation='h',
                    color='Count', color_continuous_scale='Viridis',
                    text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**⚙️ Process Complexity by Type**")
        
        # Show cycle time distribution by maintenance type
        cycle_by_type = df_process.groupby('Maintenance Type')['Total Cycle Time (Days)'].agg([
            ('Count', 'count'),
            ('Avg', 'mean'),
            ('Median', 'median'),
            ('Max', 'max')
        ]).round(1).reset_index()
        
        st.dataframe(cycle_by_type, use_container_width=True, height=400)
    
    # Transition Matrix
    st.subheader("📊 Status Transition Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**🔄 Expected vs Actual Process Duration**")
        
        # Compare by priority level
        priority_process = df_process.groupby('Priority').agg({
            'Queue Time (Days)': 'mean',
            'Repair Time (Days)': 'mean',
            'Total Cycle Time (Days)': 'mean'
        }).round(1).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Queue Time', x=priority_process['Priority'],
                            y=priority_process['Queue Time (Days)'],
                            marker_color='#ffc107'))
        fig.add_trace(go.Bar(name='Repair Time', x=priority_process['Priority'],
                            y=priority_process['Repair Time (Days)'],
                            marker_color='#2c5f2d'))
        
        fig.update_layout(
            barmode='stack',
            height=400,
            xaxis_title="Priority",
            yaxis_title="Days",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Process Efficiency Metrics**")
        
        # Calculate process efficiency metrics
        completed_df = df_process[df_process['Status'].isin(['Completed', 'Closed'])]
        
        if len(completed_df) > 0:
            avg_queue = completed_df['Queue Time (Days)'].mean()
            avg_repair = completed_df['Repair Time (Days)'].mean()
            avg_total = completed_df['Total Cycle Time (Days)'].mean()
            
            if not pd.isna(avg_queue) and not pd.isna(avg_repair) and avg_total > 0:
                queue_pct = (avg_queue / avg_total * 100)
                repair_pct = (avg_repair / avg_total * 100)
                
                st.metric("Queue Time %", f"{queue_pct:.1f}%")
                st.metric("Repair Time %", f"{repair_pct:.1f}%")
                st.metric("Process Efficiency", 
                         f"{repair_pct:.1f}%",
                         help="Higher % means more time in productive repair vs waiting")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Top Stuck Orders
    st.subheader("🔴 Work Orders Stuck in Process (Top 20)")
    
    stuck_orders = df_process[df_process['Is Active']].nlargest(20, 'Days Open')[
        ['WO Number', 'Status', 'Priority', 'Workshop', 'Days Open', 'Maintenance Type', 'Description']
    ].copy()
    
    if len(stuck_orders) > 0:
        stuck_orders['Description'] = stuck_orders['Description'].str[:50] + '...'
        st.dataframe(stuck_orders, use_container_width=True, height=400)
    else:
        st.success("✅ No stuck work orders found!")
    
    # Recommendations
    st.subheader("💡 Process Improvement Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Quick Wins**")
        
        # Generate recommendations based on bottlenecks
        if len(bottleneck_scores) > 0:
            top_bottleneck = bottleneck_scores.iloc[0]
            
            if top_bottleneck['Status'] == 'Waiting Parts':
                st.markdown("- ⚠️ **Focus on Parts Availability**: {:.0f} orders stuck waiting for parts".format(top_bottleneck['Count']))
                st.markdown("- 📦 Improve parts inventory management")
                st.markdown("- 🤝 Strengthen supplier relationships")
            
            elif top_bottleneck['Status'] == 'In Progress':
                st.markdown("- 👨‍🔧 **Increase Technician Capacity**: {:.0f} orders in progress".format(top_bottleneck['Count']))
                st.markdown("- 📚 Provide additional training")
                st.markdown("- 🔧 Review work complexity")
            
            elif top_bottleneck['Status'] == 'Open':
                st.markdown("- ⚡ **Reduce Queue Time**: {:.0f} orders waiting to start".format(top_bottleneck['Count']))
                st.markdown("- 📋 Improve work prioritization")
                st.markdown("- 👥 Review resource allocation")
    
    with col2:
        st.markdown("**📈 Strategic Improvements**")
        
        st.markdown("- 🔄 **Implement Lean Principles**")
        st.markdown("  - Reduce handoffs between statuses")
        st.markdown("  - Eliminate non-value-added steps")
        
        st.markdown("- 📊 **Measure & Monitor**")
        st.markdown("  - Set KPIs for each process stage")
        st.markdown("  - Track improvements over time")
        
        st.markdown("- 🤖 **Automation Opportunities**")
        st.markdown("  - Auto-assign work orders")
        st.markdown("  - Automated parts ordering")

def vehicle_fleet_analysis_dashboard(df):
    """Vehicle Fleet Analysis Dashboard"""
    st.markdown('<div class="dashboard-title">🚙 Vehicle Fleet Analysis</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Comprehensive fleet-level analysis to identify problem vehicles, parts patterns, and optimize fleet composition.
        
        **Key Analyses:**
        - **Vehicle Type Distribution**: Fleet composition and workload by vehicle category
        - **Parts Requirements by Vehicle**: Which vehicles need the most spare parts
        - **Top Faults per Vehicle**: Most common issues affecting specific vehicles
        - **High-Maintenance Vehicles**: Individual vehicles requiring excessive attention
        
        **Strategic Insights:**
        - Identify vehicle types with highest maintenance burden
        - Plan parts inventory based on vehicle-specific patterns
        - Target problem vehicles for replacement or major overhaul
        - Optimize future procurement based on reliability data
        
        **Use Cases:**
        - Fleet composition planning
        - Parts inventory optimization
        - Vehicle replacement prioritization
        - Procurement decision support
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_vehicles = df['Vehicle ID'].nunique()
        st.metric("Total Unique Vehicles", f"{total_vehicles:,}")
    
    with col2:
        total_types = df['Vehicle Type'].nunique()
        st.metric("Vehicle Types", f"{total_types:,}")
    
    with col3:
        avg_wo_per_vehicle = len(df) / total_vehicles if total_vehicles > 0 else 0
        st.metric("Avg WOs per Vehicle", f"{avg_wo_per_vehicle:.1f}")
    
    with col4:
        parts_intensive = (df['Requires Parts'] == True).sum() / len(df) * 100
        st.metric("Parts-Intensive WOs", f"{parts_intensive:.1f}%")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Vehicle Type Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚗 Fleet Composition by Vehicle Type")
        type_counts = df.groupby('Vehicle Type')['Vehicle ID'].nunique().reset_index()
        type_counts.columns = ['Vehicle Type', 'Vehicle Count']
        type_counts = type_counts.sort_values('Vehicle Count', ascending=False)
        
        fig = px.bar(type_counts, y='Vehicle Type', x='Vehicle Count', orientation='h',
                    color='Vehicle Count', color_continuous_scale='Blues',
                    text='Vehicle Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Work Orders by Vehicle Type")
        type_wo_counts = df.groupby('Vehicle Type').size().reset_index(name='WO Count')
        type_wo_counts = type_wo_counts.sort_values('WO Count', ascending=False)
        
        fig = px.bar(type_wo_counts, y='Vehicle Type', x='WO Count', orientation='h',
                    color='WO Count', color_continuous_scale='Oranges',
                    text='WO Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Parts Analysis by Vehicle Type
    st.subheader("📦 Parts Requirements by Vehicle Type")
    
    parts_by_type = df.groupby('Vehicle Type').agg({
        'WO Number': 'count',
        'Requires Parts': lambda x: (x == True).sum(),
        'Total Cycle Time (Days)': 'mean'
    }).reset_index()
    parts_by_type.columns = ['Vehicle Type', 'Total WOs', 'WOs Requiring Parts', 'Avg Cycle Time']
    parts_by_type['Parts %'] = (parts_by_type['WOs Requiring Parts'] / parts_by_type['Total WOs'] * 100).round(1)
    parts_by_type = parts_by_type.sort_values('WOs Requiring Parts', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(parts_by_type, x='Vehicle Type', y='WOs Requiring Parts',
                    color='Parts %', color_continuous_scale='Reds',
                    text='WOs Requiring Parts',
                    labels={'WOs Requiring Parts': 'Work Orders Requiring Parts'})
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Parts Dependency Summary**")
        st.dataframe(
            parts_by_type[['Vehicle Type', 'Parts %', 'WOs Requiring Parts']].head(10),
            use_container_width=True,
            height=350
        )
    
    # Top Problem Vehicles
    st.subheader("🔴 Top 20 High-Maintenance Vehicles")
    
    vehicle_stats = df.groupby('Vehicle ID').agg({
        'WO Number': 'count',
        'Requires Parts': lambda x: (x == True).sum(),
        'Total Cycle Time (Days)': 'mean',
        'Vehicle Type': 'first',
        'Workshop': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    vehicle_stats.columns = ['Vehicle ID', 'WO Count', 'Parts Required', 'Avg Cycle Time', 'Vehicle Type', 'Primary Workshop']
    vehicle_stats = vehicle_stats.sort_values('WO Count', ascending=False).head(20)
    
    st.dataframe(vehicle_stats, use_container_width=True, height=400)
    
    # Top Faults/Issues per Vehicle Type
    st.subheader("⚙️ Top 5 Faults by Vehicle Type")
    
    # Select top 3 vehicle types by WO volume
    top_types = df['Vehicle Type'].value_counts().head(3).index.tolist()
    
    tabs = st.tabs([f"📋 {vtype}" for vtype in top_types])
    
    for idx, vtype in enumerate(top_types):
        with tabs[idx]:
            df_type = df[df['Vehicle Type'] == vtype]
            
            # Top 5 issues for this vehicle type
            top_issues = df_type['Description'].value_counts().head(5).reset_index()
            top_issues.columns = ['Issue Description', 'Frequency']
            top_issues['Issue Description'] = top_issues['Issue Description'].str[:60] + '...'
            
            col_a, col_b = st.columns([3, 1])
            
            with col_a:
                fig = px.bar(top_issues, y='Issue Description', x='Frequency', orientation='h',
                            color='Frequency', color_continuous_scale='Reds',
                            text='Frequency')
                fig.update_traces(textposition='outside')
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                st.markdown(f"**Total WOs:** {len(df_type):,}")
                parts_pct = (df_type['Requires Parts'] == True).sum() / len(df_type) * 100
                st.markdown(f"**Parts Required:** {parts_pct:.1f}%")
                avg_cycle = df_type['Total Cycle Time (Days)'].mean()
                st.markdown(f"**Avg Cycle Time:** {avg_cycle:.1f} days")
    
    # Vehicle Type Performance Comparison
    st.subheader("📊 Vehicle Type Performance Comparison")
    
    type_performance = df.groupby('Vehicle Type').agg({
        'WO Number': 'count',
        'Total Cycle Time (Days)': 'mean',
        'Requires Parts': lambda x: (x == True).sum() / len(x) * 100,
        'Priority Level': lambda x: (x <= 2).sum()
    }).round(1).reset_index()
    type_performance.columns = ['Vehicle Type', 'Total WOs', 'Avg Cycle Time (Days)', 'Parts Required %', 'High Priority Count']
    type_performance = type_performance.sort_values('Total WOs', ascending=False)
    
    st.dataframe(type_performance, use_container_width=True, height=300)
    
    # Parts vs Non-Parts Distribution
    st.subheader("🔧 Maintenance Profile by Vehicle Type")
    
    maintenance_profile = df.groupby(['Vehicle Type', 'Requires Parts']).size().reset_index(name='Count')
    maintenance_profile['Requires Parts'] = maintenance_profile['Requires Parts'].map({True: 'Parts Required', False: 'Labor Only'})
    
    fig = px.bar(maintenance_profile, x='Vehicle Type', y='Count', color='Requires Parts',
                barmode='group', color_discrete_map={'Parts Required': '#dc3545', 'Labor Only': '#28a745'})
    fig.update_layout(height=400, xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# ==================== MAIN APPLICATION ====================

def summary_view_dashboard(df):
    """Summary View Dashboard - Consolidated view of all key metrics"""
    st.markdown('<div class="dashboard-title">📊 Summary View - All Dashboards at a Glance</div>', unsafe_allow_html=True)
    
    # Dashboard description
    with st.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        **Purpose:** Provide a consolidated, at-a-glance view of all key performance indicators across all 13 dashboards.
        
        **What You'll See:**
        - Critical KPIs from each functional area
        - Color-coded status indicators (🟢 Good, 🟡 Warning, 🔴 Critical)
        - Quick navigation to detailed dashboards
        - Trend indicators showing improvement or decline
        
        **Best Used For:**
        - Daily morning briefings (5 minutes)
        - Executive reporting
        - Quick health checks
        - Identifying which areas need immediate attention
        
        **How to Use:** Review status indicators, then drill into specific dashboards for details.
        """)
    
    if len(df) == 0:
        st.warning("No data available for selected filters")
        return
    
    # Calculate all key metrics
    total_wo = len(df)
    active_wo = len(df[df['Is Active']])
    completed_wo = len(df[~df['Is Active']])
    completion_rate = (completed_wo / total_wo * 100) if total_wo > 0 else 0
    
    # Top-Level KPIs
    st.markdown("### 🎯 Overall Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Work Orders", f"{total_wo:,}")
    
    with col2:
        status_color = "🟢" if active_wo < total_wo * 0.3 else "🟡" if active_wo < total_wo * 0.5 else "🔴"
        st.metric("Active Orders", f"{active_wo:,}", delta=f"{active_wo/total_wo*100:.1f}%")
        st.markdown(f"{status_color} Status")
    
    with col3:
        comp_color = "🟢" if completion_rate > 70 else "🟡" if completion_rate > 50 else "🔴"
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
        st.markdown(f"{comp_color} Status")
    
    with col4:
        critical = len(df[(df['Priority Level'] <= 2) & df['Is Active']])
        crit_color = "🟢" if critical < 50 else "🟡" if critical < 100 else "🔴"
        st.metric("Critical/High Open", f"{critical:,}")
        st.markdown(f"{crit_color} Status")
    
    with col5:
        avg_cycle = df[~df['Is Active']]['Total Cycle Time (Days)'].mean()
        cycle_color = "🟢" if avg_cycle < 7 else "🟡" if avg_cycle < 10 else "🔴"
        st.metric("Avg Cycle Time", f"{avg_cycle:.1f} days" if not pd.isna(avg_cycle) else "N/A")
        st.markdown(f"{cycle_color} Status")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Dashboard Summaries - 3 columns
    st.markdown("### 📋 Dashboard Summaries")
    
    # Row 1: Executive, Parts, Backlog
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Executive Overview")
        status_dist = df['Status'].value_counts()
        in_progress = status_dist.get('In Progress', 0)
        st.metric("In Progress", f"{in_progress:,}")
        st.metric("Workshops", f"{df['Workshop'].nunique()}")
        st.markdown("---")
    
    with col2:
        st.markdown("#### 2️⃣ Parts Analysis")
        requires_parts = (df['Requires Parts'] == True).sum()
        waiting_parts = len(df[df['Status'] == 'Waiting Parts'])
        parts_pct = (requires_parts / total_wo * 100) if total_wo > 0 else 0
        parts_color = "🟢" if parts_pct < 40 else "🟡" if parts_pct < 60 else "🔴"
        st.metric("Requires Parts", f"{requires_parts:,} ({parts_pct:.0f}%)")
        waiting_color = "🟢" if waiting_parts < 50 else "🟡" if waiting_parts < 100 else "🔴"
        st.metric("Waiting Parts", f"{waiting_parts:,}")
        st.markdown(f"{waiting_color} Status")
        st.markdown("---")
    
    with col3:
        st.markdown("#### 3️⃣ Backlog Aging")
        open_backlog = len(df[df['Is Active']])
        avg_age = df[df['Is Active']]['Days Open'].mean()
        aged_critical = len(df[(df['Is Active']) & (df['Priority Level'] <= 2) & (df['Days Open'] > 7)])
        age_color = "🟢" if avg_age < 7 else "🟡" if avg_age < 14 else "🔴"
        st.metric("Open Backlog", f"{open_backlog:,}")
        st.metric("Avg Age", f"{avg_age:.1f} days")
        st.markdown(f"{age_color} Status")
        st.markdown("---")
    
    # Row 2: Lifecycle, Priority, Prev/Corr
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 4️⃣ Work Order Lifecycle")
        df_completed = df[~df['Is Active']]
        avg_queue = df_completed['Queue Time (Days)'].mean()
        avg_repair = df_completed['Repair Time (Days)'].mean()
        queue_color = "🟢" if avg_queue < 3 else "🟡" if avg_queue < 5 else "🔴"
        st.metric("Avg Queue", f"{avg_queue:.1f} days" if not pd.isna(avg_queue) else "N/A")
        st.metric("Avg Repair", f"{avg_repair:.1f} days" if not pd.isna(avg_repair) else "N/A")
        st.markdown(f"{queue_color} Queue Status")
        st.markdown("---")
    
    with col2:
        st.markdown("#### 5️⃣ Priority & Risk")
        high_pri = len(df[(df['Priority Level'] <= 2) & df['Is Active']])
        p1_count = len(df[(df['Priority Level'] == 1) & df['Is Active']])
        high_pri_age = df[(df['Priority Level'] <= 2) & df['Is Active']]['Days Open'].mean()
        pri_color = "🟢" if p1_count == 0 else "🟡" if p1_count < 5 else "🔴"
        st.metric("P1 Critical", f"{p1_count:,}")
        st.metric("P1/P2 Open", f"{high_pri:,}")
        st.markdown(f"{pri_color} Status")
        st.markdown("---")
    
    with col3:
        st.markdown("#### 6️⃣ Preventive vs Corrective")
        preventive = len(df[df['Maintenance Type'] == 'Preventive'])
        corrective = len(df[df['Maintenance Type'] == 'Corrective'])
        pm_pct = (preventive / total_wo * 100) if total_wo > 0 else 0
        pm_color = "🟢" if pm_pct > 30 else "🟡" if pm_pct > 20 else "🔴"
        st.metric("PM %", f"{pm_pct:.1f}%")
        st.metric("PM/CM Ratio", f"{preventive/corrective:.2f}" if corrective > 0 else "N/A")
        st.markdown(f"{pm_color} Status")
        st.markdown("---")
    
    # Row 3: Repeat, Technician, Quality
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 7️⃣ Repeat Issues")
        vehicle_counts = df.groupby('Vehicle ID').size()
        high_repeat = len(vehicle_counts[vehicle_counts > 5])
        top_vehicle_wo = vehicle_counts.max() if len(vehicle_counts) > 0 else 0
        repeat_color = "🟢" if high_repeat < 20 else "🟡" if high_repeat < 50 else "🔴"
        st.metric("Vehicles >5 WOs", f"{high_repeat:,}")
        st.metric("Top Vehicle WOs", f"{int(top_vehicle_wo)}")
        st.markdown(f"{repeat_color} Status")
        st.markdown("---")
    
    with col2:
        st.markdown("#### 8️⃣ Technician Productivity")
        techs = df[df['Assigned To'].notna()]['Assigned To'].nunique()
        avg_wo_tech = (df[df['Assigned To'].notna()].shape[0] / techs) if techs > 0 else 0
        tech_color = "🟢" if avg_wo_tech < 100 else "🟡" if avg_wo_tech < 150 else "🔴"
        st.metric("Active Technicians", f"{techs:,}")
        st.metric("Avg WOs/Tech", f"{avg_wo_tech:.1f}")
        st.markdown(f"{tech_color} Status")
        st.markdown("---")
    
    with col3:
        st.markdown("#### 9️⃣ Data Quality")
        missing_start = df['Start Date'].isna().sum()
        completed_df = df[df['Status'].isin(['Completed', 'Closed'])]
        missing_completion = completed_df['Completion Date'].isna().sum() if len(completed_df) > 0 else 0
        quality_score = 100 - ((missing_start + missing_completion) / total_wo * 100)
        quality_color = "🟢" if quality_score > 90 else "🟡" if quality_score > 75 else "🔴"
        st.metric("Quality Score", f"{quality_score:.1f}%")
        st.metric("Missing Data", f"{missing_start + missing_completion:,}")
        st.markdown(f"{quality_color} Status")
        st.markdown("---")
    
    # Row 4: Unit, Mileage, Fleet
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔟 Owning Unit")
        units = df['Unit Code'].nunique()
        unit_stats = df.groupby('Unit Code').size()
        top_unit_wo = unit_stats.max() if len(unit_stats) > 0 else 0
        st.metric("Total Units", f"{units:,}")
        st.metric("Top Unit WOs", f"{int(top_unit_wo)}")
        st.markdown("---")
    
    with col2:
        st.markdown("#### 1️⃣1️⃣ Vehicle Mileage")
        df_mileage = df[df['Mileage'].notna()]
        avg_mileage = df_mileage['Mileage'].mean()
        high_mileage = len(df_mileage[df_mileage['Mileage'] > 100000])
        mileage_color = "🟢" if high_mileage < 100 else "🟡" if high_mileage < 200 else "🔴"
        st.metric("Avg Mileage", f"{avg_mileage:,.0f} km" if not pd.isna(avg_mileage) else "N/A")
        st.metric("High Mileage (>100k)", f"{high_mileage:,}")
        st.markdown(f"{mileage_color} Status")
        st.markdown("---")
    
    with col3:
        st.markdown("#### 1️⃣2️⃣ Vehicle Fleet")
        vehicles = df['Vehicle ID'].nunique()
        vehicle_types = df['Vehicle Type'].nunique()
        avg_wo_vehicle = total_wo / vehicles if vehicles > 0 else 0
        st.metric("Total Vehicles", f"{vehicles:,}")
        st.metric("Avg WOs/Vehicle", f"{avg_wo_vehicle:.1f}")
        st.markdown("---")
    
    # Row 5: Process Mining
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣3️⃣ Process Mining")
        bottleneck_state = df[df['Is Active']]['Status'].value_counts().idxmax() if len(df[df['Is Active']]) > 0 else "None"
        bottleneck_count = df[df['Is Active']]['Status'].value_counts().max() if len(df[df['Is Active']]) > 0 else 0
        stuck_30 = len(df[(df['Is Active']) & (df['Days Open'] > 30)])
        stuck_color = "🟢" if stuck_30 < 10 else "🟡" if stuck_30 < 30 else "🔴"
        st.metric("Bottleneck State", bottleneck_state)
        st.metric("WOs in Bottleneck", f"{int(bottleneck_count):,}")
        st.metric("Stuck >30 Days", f"{stuck_30:,}")
        st.markdown(f"{stuck_color} Status")
        st.markdown("---")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Critical Alerts Section
    st.markdown("### 🚨 Critical Alerts & Action Items")
    
    alerts = []
    
    # Check for critical issues
    if p1_count > 0:
        alerts.append({
            'severity': '🔴',
            'dashboard': 'Priority & Risk',
            'issue': f'{p1_count} Priority 1 Critical work orders open',
            'action': 'Review and expedite immediately'
        })
    
    if waiting_parts > 100:
        alerts.append({
            'severity': '🔴',
            'dashboard': 'Parts Analysis',
            'issue': f'{waiting_parts} work orders waiting for parts',
            'action': 'Review parts inventory and supplier performance'
        })
    
    if stuck_30 > 20:
        alerts.append({
            'severity': '🔴',
            'dashboard': 'Process Mining',
            'issue': f'{stuck_30} work orders stuck >30 days',
            'action': 'Immediate investigation required'
        })
    
    if aged_critical > 10:
        alerts.append({
            'severity': '🟡',
            'dashboard': 'Backlog Aging',
            'issue': f'{aged_critical} critical orders aged >7 days',
            'action': 'Prioritize high-priority backlog'
        })
    
    if pm_pct < 20:
        alerts.append({
            'severity': '🟡',
            'dashboard': 'Preventive vs Corrective',
            'issue': f'PM percentage low at {pm_pct:.1f}% (target 30%)',
            'action': 'Increase preventive maintenance scheduling'
        })
    
    if quality_score < 80:
        alerts.append({
            'severity': '🟡',
            'dashboard': 'Data Quality',
            'issue': f'Data quality score at {quality_score:.1f}% (target >90%)',
            'action': 'Address missing data fields'
        })
    
    if len(alerts) > 0:
        alert_df = pd.DataFrame(alerts)
        st.dataframe(alert_df, use_container_width=True, height=min(len(alerts) * 50 + 50, 400))
    else:
        st.success("✅ No critical alerts - All systems operating within acceptable parameters")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Quick Stats Grid
    st.markdown("### 📈 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📅 Time Period**")
        date_range = (df['Created Date'].max() - df['Created Date'].min()).days
        st.markdown(f"Date Range: {date_range} days")
        st.markdown(f"From: {df['Created Date'].min().strftime('%Y-%m-%d')}")
        st.markdown(f"To: {df['Created Date'].max().strftime('%Y-%m-%d')}")
    
    with col2:
        st.markdown("**🏢 Coverage**")
        st.markdown(f"Provinces: {df['Province'].nunique()}")
        st.markdown(f"Cities: {df['City'].nunique()}")
        st.markdown(f"Brigades: {df['Brigade'].nunique()}")
        st.markdown(f"Workshops: {df['Workshop'].nunique()}")
    
    with col3:
        st.markdown("**🚗 Fleet**")
        st.markdown(f"Vehicles: {df['Vehicle ID'].nunique():,}")
        st.markdown(f"Vehicle Types: {df['Vehicle Type'].nunique()}")
        st.markdown(f"Units: {df['Unit Code'].nunique()}")
    
    with col4:
        st.markdown("**👥 Resources**")
        st.markdown(f"Technicians: {df[df['Assigned To'].notna()]['Assigned To'].nunique()}")
        st.markdown(f"Avg Load: {avg_wo_tech:.1f} WOs/Tech")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Top Issues Summary
    st.markdown("### 🔝 Top Issues & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 5 Most Common Issues**")
        top_issues = df['Description'].value_counts().head(5).reset_index()
        top_issues.columns = ['Issue', 'Count']
        top_issues['Issue'] = top_issues['Issue'].str[:50] + '...'
        
        fig = px.bar(top_issues, y='Issue', x='Count', orientation='h',
                    color='Count', color_continuous_scale='Reds',
                    text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Top 5 Workshops by Volume**")
        top_workshops = df.groupby('Workshop').size().reset_index(name='Count').nlargest(5, 'Count')
        
        fig = px.bar(top_workshops, y='Workshop', x='Count', orientation='h',
                    color='Count', color_continuous_scale='Blues',
                    text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Navigation Helper
    st.markdown("### 🧭 Navigate to Detailed Dashboard")
    st.info("💡 **Tip:** Use the dashboard navigation in the sidebar to drill into specific areas for detailed analysis")

def main():
    """Main application entry point"""
    
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    
    if not st.session_state['data_loaded']:
        data_import_screen()
    else:
        # Data is loaded - show full application
        st.markdown('<div class="main-header">🔧 AMIC IMSS Analytics</div>', unsafe_allow_html=True)
        
        # Initialize session state for quick filters
        if 'quick_province_filter' not in st.session_state:
            st.session_state['quick_province_filter'] = 'All'
        
        # Get base data and mappings
        df_base = st.session_state['df']
        mappings = st.session_state.get('mappings', build_lookup_mappings())
        
        # Sidebar
        with st.sidebar:
            st.markdown("### 🔧 AMIC IMSS")
            st.markdown("*Integrated Maintenance Support System*")
            st.markdown("---")
            
            # Role selection
            st.markdown("### 👤 User Role")
            
            role = st.selectbox(
                "Select Role",
                ["Exec", "Fleet Manager", "Supervisor"],
                index=0,
                key="role_selector"
            )
            
            # Workshop assignment based on role
            if role == "Supervisor":
                # Supervisor: single workshop (locked)
                available_workshops = sorted(df_base['Workshop'].unique())
                selected_workshop = st.selectbox("Assigned Workshop (Locked)", available_workshops)
                user_workshops = [selected_workshop]
                st.markdown(f'<div class="role-badge role-supervisor">Supervisor: {selected_workshop}</div>', 
                           unsafe_allow_html=True)
            
            elif role == "Fleet Manager":
                # Fleet Manager: multiple workshops
                available_workshops = sorted(df_base['Workshop'].unique())
                selected_workshops = st.multiselect(
                    "Assigned Workshops",
                    available_workshops,
                    default=available_workshops[:3]  # Demo: assign first 3
                )
                user_workshops = selected_workshops if selected_workshops else []
                st.markdown(f'<div class="role-badge role-fleet">Fleet Manager: {len(user_workshops)} workshops</div>', 
                           unsafe_allow_html=True)
            
            else:  # Exec
                user_workshops = []
                st.markdown('<div class="role-badge role-exec">Executive: Full Access</div>', 
                           unsafe_allow_html=True)
            
            # Store role and workshops
            st.session_state['user_role'] = role
            st.session_state['user_workshops'] = user_workshops
            
            st.markdown("---")
            
            # Apply role-based row-level security
            df_scoped = apply_role_scope(df_base, role, user_workshops)
            
            if len(df_scoped) == 0:
                st.error("⚠️ No data available for your role")
                st.stop()
            
            st.success(f"✅ {len(df_scoped):,} records in scope")
            
            st.markdown("---")
            st.markdown("### 🔍 Filters")
            
            # Date range filter
            min_date = df_scoped['Created Date'].min().date()
            max_date = df_scoped['Created Date'].max().date()
            default_start = max_date - timedelta(days=90)
            
            date_from = st.date_input("From Date", value=default_start, min_value=min_date, max_value=max_date)
            date_to = st.date_input("To Date", value=max_date, min_value=min_date, max_value=max_date)
            
            # Cascading filters (computed from scoped data)
            filters = {}
            filters['date_from'] = pd.Timestamp(date_from)
            filters['date_to'] = pd.Timestamp(date_to)
            
            # Province
            provinces = ['All'] + sorted(df_scoped['Province'].dropna().unique().tolist())
            
            # Use session state for default value (updated by quick filter buttons)
            province_index = 0
            if st.session_state['quick_province_filter'] in provinces:
                province_index = provinces.index(st.session_state['quick_province_filter'])
            
            selected_province = st.selectbox("Province", provinces, index=province_index, key='province_selectbox')
            
            # Update session state when selectbox changes
            st.session_state['quick_province_filter'] = selected_province
            
            filters['Province'] = selected_province
            
            # City (based on selected province)
            if filters['Province'] != 'All':
                df_province = df_scoped[df_scoped['Province'] == filters['Province']]
            else:
                df_province = df_scoped.copy()
            
            cities = ['All'] + sorted(df_province['City'].dropna().unique().tolist())
            filters['City'] = st.selectbox("City", cities)
            
            # Brigade (based on selected city)
            if filters['City'] != 'All':
                df_city = df_province[df_province['City'] == filters['City']]
            else:
                df_city = df_province.copy()
            
            brigades = ['All'] + sorted(df_city['Brigade'].dropna().unique().tolist())
            filters['Brigade'] = st.selectbox("Brigade", brigades)
            
            # Workshop (based on selected brigade and role)
            if filters['Brigade'] != 'All':
                df_brigade = df_city[df_city['Brigade'] == filters['Brigade']]
            else:
                df_brigade = df_city.copy()
            
            if role == "Supervisor":
                # Workshop locked for supervisor
                filters['Workshop'] = user_workshops[0]
                st.info(f"🔒 Workshop: {filters['Workshop']}")
            else:
                workshops = ['All'] + sorted(df_brigade['Workshop'].unique().tolist())
                filters['Workshop'] = st.selectbox("Workshop", workshops)
            
            # Unit Code
            if filters['Workshop'] != 'All':
                df_workshop = df_brigade[df_brigade['Workshop'] == filters['Workshop']]
            else:
                df_workshop = df_brigade.copy()
            
            units = ['All'] + sorted(df_workshop['Unit Code'].dropna().unique().tolist())
            filters['Unit Code'] = st.selectbox("Vehicle Owning Unit", units)
            
            # Vehicle Type
            if filters['Unit Code'] != 'All':
                df_unit = df_workshop[df_workshop['Unit Code'] == filters['Unit Code']]
            else:
                df_unit = df_workshop.copy()
            
            vehicle_types = ['All'] + sorted(df_unit['Vehicle Type'].dropna().unique().tolist())
            filters['Vehicle Type'] = st.selectbox("Vehicle Type", vehicle_types)
            
            st.markdown("---")
            
            # Apply all filters
            df_filtered = apply_cascading_filters(df_scoped, filters)
            
            st.info(f"📊 {len(df_filtered):,} records after filters")
            
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 📊 Dashboard Navigation")
            
            dashboard_options = [
                "📊 Summary View",
                "1️⃣ Executive Overview",
                "2️⃣ Parts Analysis",
                "3️⃣ Backlog Aging",
                "4️⃣ Work Order Lifecycle",
                "5️⃣ Priority & Risk",
                "6️⃣ Preventive vs Corrective",
                "7️⃣ Repeat Issues",
                "8️⃣ Technician Productivity",
                "9️⃣ Data Quality",
                "🔟 Owning Unit Analysis",
                "1️⃣1️⃣ Vehicle Mileage",
                "1️⃣2️⃣ Vehicle Fleet Analysis",
                "1️⃣3️⃣ Process Mining & Bottlenecks"
            ]
            
            dashboard = st.radio("Select Dashboard:", dashboard_options)
            
            st.markdown("---")
            st.markdown(f"**Role:** {role}")
            st.markdown(f"**Records:** {len(df_filtered):,}")
        
        # Quick Province Filter Buttons
        st.markdown("""
        <div style='background: linear-gradient(90deg, #f0f8f0 0%, #ffffff 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    margin-bottom: 2rem; 
                    border: 2px solid #2c5f2d;'>
            <h3 style='margin: 0 0 1rem 0; color: #2c5f2d;'>
                🌍 Quick Province Filter
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get unique provinces from scoped data
        available_provinces = sorted(df_scoped['Province'].dropna().unique().tolist())
        
        # Current selection from session state
        current_province = st.session_state.get('quick_province_filter', 'All')
        
        # Create filter buttons with better spacing
        button_cols = st.columns(len(available_provinces) + 1)
        
        # "All" button
        with button_cols[0]:
            all_selected = (current_province == 'All')
            if st.button("🌐 **All Provinces**", 
                        type="primary" if all_selected else "secondary",
                        use_container_width=True,
                        key="province_all"):
                st.session_state['quick_province_filter'] = 'All'
                st.rerun()
        
        # Province buttons
        for idx, province in enumerate(available_provinces):
            with button_cols[idx + 1]:
                # Extract province short name for button
                province_short = province.replace(' Province', '').replace('Province', '').strip()
                
                # Count work orders in this province
                province_count = len(df_scoped[df_scoped['Province'] == province])
                
                # Determine button type based on selection
                is_selected = (current_province == province)
                button_type = "primary" if is_selected else "secondary"
                
                # Button label with count
                button_label = f"📍 **{province_short}**\n{province_count:,} WOs"
                
                if st.button(button_label, 
                           type=button_type,
                           use_container_width=True,
                           key=f"province_{idx}_{province.replace(' ', '_').replace('(', '').replace(')', '')}",
                           help=f"Filter by {province} ({province_count:,} work orders)"):
                    st.session_state['quick_province_filter'] = province
                    st.rerun()
        
        # Show active filter indicator
        if current_province != 'All':
            province_wo_count = len(df_scoped[df_scoped['Province'] == current_province])
            st.markdown(f"""
            <div style='background-color: #d1ecf1; 
                        border-left: 4px solid #17a2b8; 
                        padding: 0.75rem; 
                        margin: 1rem 0; 
                        border-radius: 4px;'>
                <strong>🔍 Active Filter:</strong> {current_province} ({province_wo_count:,} work orders)
                <br><small>💡 Click "All Provinces" to clear filter</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Apply filters
        df_filtered = apply_cascading_filters(df_scoped, filters)
        
        # Display selected dashboard
        try:
            if len(df_filtered) == 0:
                st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")
                st.stop()
            
            # Display selected dashboard
            if "Summary View" in dashboard:
                summary_view_dashboard(df_filtered)
            elif "Executive Overview" in dashboard:
                executive_overview_dashboard(df_filtered)
            elif "Parts Analysis" in dashboard:
                parts_analysis_dashboard(df_filtered)
            elif "Backlog Aging" in dashboard:
                backlog_aging_dashboard(df_filtered)
            elif "Work Order Lifecycle" in dashboard:
                lifecycle_dashboard(df_filtered)
            elif "Priority & Risk" in dashboard:
                priority_risk_dashboard(df_filtered)
            elif "Preventive vs Corrective" in dashboard:
                preventive_corrective_dashboard(df_filtered)
            elif "Repeat Issues" in dashboard:
                repeat_issues_dashboard(df_filtered)
            elif "Technician Productivity" in dashboard:
                technician_productivity_dashboard(df_filtered)
            elif "Data Quality" in dashboard:
                data_quality_dashboard(df_filtered)
            elif "Owning Unit" in dashboard:
                owning_unit_dashboard(df_filtered)
            elif "Vehicle Mileage" in dashboard:
                vehicle_mileage_dashboard(df_filtered)
            elif "Vehicle Fleet Analysis" in dashboard:
                vehicle_fleet_analysis_dashboard(df_filtered)
            elif "Process Mining" in dashboard:
                process_mining_dashboard(df_filtered)
        
        except Exception as e:
            st.error(f"Error displaying dashboard: {str(e)}")
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
