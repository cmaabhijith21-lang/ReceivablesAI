import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import json
import html
import re

warnings.filterwarnings('ignore')

# Constants for magic numbers
TEAM_CAPACITY_HOURS = 40
ANOMALY_CONTAMINATION = 0.1
FORECAST_PERIODS = 90
DSO_DAYS = 30  # Changed from 90 to 30 as per correction

# Page configuration
st.set_page_config(
    page_title="Receivables AI Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for golden ratio design and beautiful UI
st.markdown("""
<style>
    /* Golden ratio proportions */
    .main {
        background-color: #f8f9fa;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.618rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .metric-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .metric-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .metric-card-red {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        padding-bottom: 1rem;
    }
    
    h2 {
        color: #3b82f6;
        font-weight: 600;
        padding-top: 1rem;
    }
    
    h3 {
        color: #6366f1;
        font-weight: 500;
    }
    
    .alert-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .danger-box {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .recommendation-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    .ai-insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .scrollable-table {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        background: white;
    }
    
    .customer-table {
        font-size: 0.9em;
    }
    
    .customer-table thead th {
        position: sticky;
        top: 0;
        background: #3b82f6;
        color: white;
        z-index: 10;
    }
</style>
""", unsafe_allow_html=True)

# === AI ASSISTANT SETUP ===
def setup_ai_assistant(api_key):
    """Setup the AI assistant with your API key"""
    try:
        if api_key:
            genai.configure(api_key=api_key)
            return True
        return False
    except Exception as e:
        st.error(f"API configuration error: {str(e)}")
        return False

def sanitize_html(text):
    """Sanitize HTML to prevent XSS attacks"""
    if not text:
        return ""
    # Escape HTML special characters
    safe_text = html.escape(text)
    # Allow basic formatting but remove scripts and other dangerous tags
    safe_text = re.sub(r'<script.*?</script>', '', safe_text, flags=re.IGNORECASE | re.DOTALL)
    safe_text = re.sub(r'<iframe.*?</iframe>', '', safe_text, flags=re.IGNORECASE | re.DOTALL)
    safe_text = re.sub(r'<object.*?</object>', '', safe_text, flags=re.IGNORECASE | re.DOTALL)
    safe_text = re.sub(r'<embed.*?</embed>', '', safe_text, flags=re.IGNORECASE | re.DOTALL)
    safe_text = re.sub(r'on\w+=".*?"', '', safe_text, flags=re.IGNORECASE)
    safe_text = re.sub(r'on\w+=\'.*?\'', '', safe_text, flags=re.IGNORECASE)
    safe_text = re.sub(r'javascript:', '', safe_text, flags=re.IGNORECASE)
    return safe_text

def ask_ai(prompt, data_info=""):
    """Ask the AI to analyze our data"""
    try:
        if 'api_key_configured' not in st.session_state or not st.session_state.api_key_configured:
            return "Please add your Gemini API key in the sidebar to enable AI analysis."
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Anonymize sensitive data before sending to AI
        if data_info:
            anonymized_data = data_info
            # Remove specific customer identifiers
            anonymized_data = re.sub(r'CUST\d+', 'CUSTOMER', anonymized_data)
            anonymized_data = re.sub(r'Company [A-Z]+\d*', 'COMPANY', anonymized_data)
            anonymized_data = re.sub(r'₹\d+\.?\d*[MK]', 'AMOUNT', anonymized_data)
            
            full_question = f"{prompt}\n\nHere's the anonymized data:\n{anonymized_data}"
        else:
            full_question = prompt
            
        response = model.generate_content(full_question)
        return sanitize_html(response.text)
    except Exception as e:
        return f"AI analysis error: {str(e)}. Please check your API key and try again."

def format_currency(amount):
    """Format currency consistently"""
    if amount is None or np.isnan(amount):
        return "₹0"
    if abs(amount) >= 1e6:
        return f"₹{amount/1e6:.2f}M"
    elif abs(amount) >= 1e3:
        return f"₹{amount/1e3:.0f}K"
    return f"₹{amount:.0f}"

# Helper Functions
@st.cache_data
def load_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_records = 200
    
    customers = ['CUST' + str(i).zfill(3) for i in range(1, 51)]
    customer_names = [f'Company {chr(65+i%26)}{i//26+1}' for i in range(50)]
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
    sales_offices = ['North', 'South', 'East', 'West', 'Central']
    sales_employees = ['Rajesh Kumar', 'Priya Sharma', 'Amit Patel', 'Sneha Singh', 'Vikram Reddy']
    
    start_date = datetime.now() - timedelta(days=180)
    
    data = []
    for i in range(n_records):
        customer_idx = np.random.randint(0, 50)
        invoice_date = start_date + timedelta(days=np.random.randint(0, 180))
        credit_period = np.random.choice([15, 30, 45, 60])
        due_date = invoice_date + timedelta(days=int(credit_period))
        
        payment_probability = np.random.random()
        if payment_probability > 0.3:
            delay_days = np.random.choice([0, 5, 10, 15, 30, 45, 60, 90], p=[0.3, 0.25, 0.2, 0.1, 0.08, 0.04, 0.02, 0.01])
            payment_date = due_date + timedelta(days=int(delay_days))
            if payment_date > datetime.now():
                payment_date = None
                amount_outstanding = np.random.uniform(10000, 500000)
            else:
                amount_outstanding = 0
        else:
            payment_date = None
            amount_outstanding = np.random.uniform(10000, 500000)
        
        sales_amount = amount_outstanding if amount_outstanding > 0 else np.random.uniform(10000, 500000)
        
        data.append({
            'Customer_Code': customers[customer_idx],
            'Customer_Name': customer_names[customer_idx],
            'Sales_Amount': round(sales_amount, 2),
            'Amount_Outstanding': round(amount_outstanding, 2),
            'Invoice_Date': invoice_date.strftime('%Y-%m-%d'),
            'Due_Date': due_date.strftime('%Y-%m-%d'),
            'Payment_Date': payment_date.strftime('%Y-%m-%d') if payment_date else None,
            'City': np.random.choice(cities),
            'Sales_Office': np.random.choice(sales_offices),
            'Sales_Employee': np.random.choice(sales_employees),
            'Credit_Period': credit_period
        })
    
    return pd.DataFrame(data)

def calculate_metrics(df):
    """Calculate key receivables metrics with corrected formulas"""
    df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])
    df['Due_Date'] = pd.to_datetime(df['Due_Date'])
    df['Payment_Date'] = pd.to_datetime(df['Payment_Date'], errors='coerce')
    
    today = datetime.now()
    
    # Calculate Days Overdue - only for unpaid invoices
    df['Days_Overdue'] = df.apply(
        lambda x: max(0, (today - x['Due_Date']).days) if pd.isna(x['Payment_Date']) and x['Amount_Outstanding'] > 0 else 0,
        axis=1
    )
    
    # Calculate Days_To_Pay only for paid invoices
    df['Days_To_Pay'] = df.apply(
        lambda x: (x['Payment_Date'] - x['Invoice_Date']).days if pd.notna(x['Payment_Date']) else None,
        axis=1
    )
    
    # Calculate Aging Bucket - only for outstanding amounts
    # Note: Paid invoices will not be included in aging buckets
    df['Aging_Bucket'] = pd.cut(
        df['Days_Overdue'],
        bins=[-1, 0, 30, 60, 90, float('inf')],
        labels=['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']
    )
    
    # For paid invoices, set aging bucket to None
    df.loc[df['Payment_Date'].notna(), 'Aging_Bucket'] = None
    
    # Customer segmentation based on payment behavior (only for customers with payment history)
    paid_df = df[df['Payment_Date'].notna()].copy()
    if len(paid_df) > 0:
        avg_days = paid_df.groupby('Customer_Code')['Days_To_Pay'].mean()
        df['Customer_Segment'] = df['Customer_Code'].map(
            lambda x: 'Prompt Payers' if pd.notna(avg_days.get(x)) and avg_days.get(x) <= 15
            else 'Standard Payers' if pd.notna(avg_days.get(x)) and avg_days.get(x) <= 45
            else 'Slow Payers' if pd.notna(avg_days.get(x)) and avg_days.get(x) <= 90
            else 'Delinquent'
        )
    else:
        df['Customer_Segment'] = 'Unknown'
    
    return df

@st.cache_data
def calculate_dso(df, month=None):
    """Calculate Days Sales Outstanding for the month under review"""
    if month is None:
        # Get last month's data
        current_date = datetime.now()
        last_month_start = datetime(current_date.year, current_date.month - 1 if current_date.month > 1 else 12, 1)
        last_month_end = datetime(current_date.year, current_date.month, 1) - timedelta(days=1)
        
        # Filter sales for last month
        last_month_sales = df[
            (df['Invoice_Date'] >= last_month_start) & 
            (df['Invoice_Date'] <= last_month_end)
        ]['Sales_Amount'].sum()
    else:
        # Use specified month
        month_start = datetime.strptime(f"{month}-01", "%Y-%m-%d")
        next_month = month_start.replace(day=28) + timedelta(days=4)
        month_end = next_month - timedelta(days=next_month.day)
        
        last_month_sales = df[
            (df['Invoice_Date'] >= month_start) & 
            (df['Invoice_Date'] <= month_end)
        ]['Sales_Amount'].sum()
    
    # Current outstanding (all unpaid amounts)
    total_ar = df[df['Payment_Date'].isna()]['Amount_Outstanding'].sum()
    
    if last_month_sales > 0 and total_ar >= 0:
        dso = (total_ar / last_month_sales) * DSO_DAYS  # Changed from 90 to 30
    else:
        dso = 0
    return round(dso, 1)

@st.cache_data
def calculate_cei(df, month=None, compare_to_previous=False):
    """Calculate Collection Effectiveness Index for the month under review"""
    if month is None:
        # Get current month data
        current_date = datetime.now()
        month_start = datetime(current_date.year, current_date.month, 1)
        month_end = current_date
    else:
        # Use specified month
        month_start = datetime.strptime(f"{month}-01", "%Y-%m-%d")
        next_month = month_start.replace(day=28) + timedelta(days=4)
        month_end = next_month - timedelta(days=next_month.day)
    
    # Filter invoices due in this month
    month_invoices = df[
        (df['Due_Date'] >= month_start) & 
        (df['Due_Date'] <= month_end)
    ]
    
    if len(month_invoices) == 0:
        return 0, 0 if compare_to_previous else 0
    
    # Count invoices collected within 5 days from due date
    collected_within_5_days = month_invoices[
        (month_invoices['Payment_Date'].notna()) & 
        ((month_invoices['Payment_Date'] - month_invoices['Due_Date']).dt.days <= 5)
    ].shape[0]
    
    total_invoices = month_invoices.shape[0]
    
    if total_invoices > 0:
        cei = (collected_within_5_days / total_invoices) * 100
    else:
        cei = 0
    
    if compare_to_previous:
        # Calculate for previous month
        prev_month_start = month_start - timedelta(days=30)
        prev_month_end = month_start - timedelta(days=1)
        
        prev_month_invoices = df[
            (df['Due_Date'] >= prev_month_start) & 
            (df['Due_Date'] <= prev_month_end)
        ]
        
        if len(prev_month_invoices) > 0:
            prev_collected = prev_month_invoices[
                (prev_month_invoices['Payment_Date'].notna()) & 
                ((prev_month_invoices['Payment_Date'] - prev_month_invoices['Due_Date']).dt.days <= 5)
            ].shape[0]
            
            prev_cei = (prev_collected / len(prev_month_invoices)) * 100 if len(prev_month_invoices) > 0 else 0
            return round(cei, 1), round(prev_cei, 1)
    
    return round(cei, 1)

# === NEW HELPER FUNCTIONS WITH CORRECTED FORMULAS ===
def calculate_collection_rates(df, days=30):
    """Calculate collection rates for specific periods"""
    current_date = datetime.now()
    period_start = current_date - timedelta(days=days)
    
    # Filter invoices due in the period
    period_invoices = df[df['Due_Date'] >= period_start]
    
    if len(period_invoices) == 0:
        return 0
    
    # Calculate amount due in period
    total_due = period_invoices['Sales_Amount'].sum()
    
    # Calculate amount collected in period
    collected = period_invoices[period_invoices['Payment_Date'].notna()]['Sales_Amount'].sum()
    
    if total_due > 0:
        collection_rate = (collected / total_due) * 100
    else:
        collection_rate = 0
    
    return round(collection_rate, 1)

def calculate_payment_within_terms(df, grace_period=5):
    """Calculate percentage of invoices paid within credit terms + grace period"""
    paid_invoices = df[df['Payment_Date'].notna()].copy()
    
    if len(paid_invoices) == 0:
        return 0
    
    # Calculate actual days taken to pay
    paid_invoices['Actual_Days'] = (paid_invoices['Payment_Date'] - paid_invoices['Invoice_Date']).dt.days
    
    # Count invoices paid within terms + grace period
    paid_within_terms = paid_invoices[
        paid_invoices['Actual_Days'] <= (paid_invoices['Credit_Period'] + grace_period)
    ].shape[0]
    
    payment_within_terms = (paid_within_terms / len(paid_invoices)) * 100
    return round(payment_within_terms, 1)

def calculate_overdue_rate(df):
    """Calculate percentage of total outstanding that is overdue"""
    total_outstanding = df[df['Payment_Date'].isna()]['Amount_Outstanding'].sum()
    
    if total_outstanding == 0:
        return 0
    
    overdue_amount = df[
        (df['Payment_Date'].isna()) & 
        (df['Days_Overdue'] > 0)
    ]['Amount_Outstanding'].sum()
    
    overdue_rate = (overdue_amount / total_outstanding) * 100
    return round(overdue_rate, 1)

def calculate_recent_days_to_pay(df, months=2):
    """Calculate average days to pay for recent period (last 2 months)"""
    current_date = datetime.now()
    period_start = current_date - timedelta(days=months*30)
    
    recent_payments = df[
        (df['Payment_Date'].notna()) & 
        (df['Payment_Date'] >= period_start)
    ].copy()
    
    if len(recent_payments) == 0:
        return 0, 0
    
    # Calculate average days to pay for recent period
    recent_avg = recent_payments['Days_To_Pay'].mean()
    
    # Calculate for previous period for comparison
    prev_period_start = period_start - timedelta(days=months*30)
    prev_period_end = period_start - timedelta(days=1)
    
    prev_payments = df[
        (df['Payment_Date'].notna()) & 
        (df['Payment_Date'] >= prev_period_start) & 
        (df['Payment_Date'] <= prev_period_end)
    ]
    
    if len(prev_payments) > 0:
        prev_avg = prev_payments['Days_To_Pay'].mean()
    else:
        prev_avg = 0
    
    return round(recent_avg, 1), round(prev_avg, 1)

def get_customer_detailed_metrics(df):
    """Get detailed customer metrics with corrected formulas"""
    # Filter for last 6 months
    six_months_ago = datetime.now() - timedelta(days=180)
    recent_df = df[df['Invoice_Date'] >= six_months_ago]
    
    customer_metrics = recent_df.groupby(['Customer_Code', 'Customer_Name', 'Sales_Employee', 'City']).agg({
        'Amount_Outstanding': 'sum',
        'Sales_Amount': 'sum',
        'Days_To_Pay': ['mean', 'std'],
        'Days_Overdue': ['max', 'mean'],
        'Credit_Period': 'mean',
        'Invoice_Date': ['count', 'min', 'max'],
        'Payment_Date': lambda x: x.notna().sum()
    }).reset_index()
    
    # Flatten column names
    customer_metrics.columns = ['Customer_Code', 'Customer_Name', 'Sales_Employee', 'City',
                               'Total_Outstanding', 'Total_Sales', 'Avg_Days_To_Pay', 'Std_Days_To_Pay',
                               'Max_Days_Overdue', 'Avg_Days_Overdue', 'Avg_Credit_Period',
                               'Invoice_Count', 'First_Invoice_Date', 'Last_Invoice_Date', 'Paid_Invoices']
    
    # Corrected Payment Rate: (Number of invoices paid within last 6 months ÷ Total invoices issued in last 6 months) × 100
    customer_metrics['Payment_Rate'] = (customer_metrics['Paid_Invoices'] / customer_metrics['Invoice_Count'] * 100).round(2)
    
    # Corrected Collection Rate: [(Sales in last 6 months - Current outstanding from those sales) ÷ Sales in last 6 months] × 100
    customer_metrics['Collection_Rate'] = ((customer_metrics['Total_Sales'] - customer_metrics['Total_Outstanding']) / 
                                          customer_metrics['Total_Sales'] * 100).round(2)
    
    # Initialize new columns
    customer_metrics['Early_Payment_Rate'] = 0.0
    customer_metrics['Late_Payment_Rate'] = 0.0
    customer_metrics['Weighted_Avg_Credit_Period'] = 0.0
    
    # Early Payment Rate (paid within credit period) and Late Payment Rate
    for idx, row in customer_metrics.iterrows():
        cust_code = row['Customer_Code']
        cust_data = recent_df[recent_df['Customer_Code'] == cust_code]
        
        paid_invoices = cust_data[cust_data['Payment_Date'].notna()]
        if len(paid_invoices) > 0:
            early_payments = paid_invoices[
                (paid_invoices['Payment_Date'] - paid_invoices['Invoice_Date']).dt.days <= paid_invoices['Credit_Period']
            ]
            # FIX: Use round() function instead of .round() method on float
            early_rate = round(len(early_payments) / len(paid_invoices) * 100, 2)
            customer_metrics.at[idx, 'Early_Payment_Rate'] = early_rate
            
            # Late Payment Rate (paid after credit period)
            late_payments = paid_invoices[
                (paid_invoices['Payment_Date'] - paid_invoices['Invoice_Date']).dt.days > paid_invoices['Credit_Period']
            ]
            late_rate = round(len(late_payments) / len(paid_invoices) * 100, 2)
            customer_metrics.at[idx, 'Late_Payment_Rate'] = late_rate
    
    # Weighted average credit period for last 3 months
    three_months_ago = datetime.now() - timedelta(days=90)
    recent_3m_df = df[df['Invoice_Date'] >= three_months_ago]
    
    for idx, row in customer_metrics.iterrows():
        cust_code = row['Customer_Code']
        cust_3m_data = recent_3m_df[recent_3m_df['Customer_Code'] == cust_code]
        
        if len(cust_3m_data) > 0 and cust_3m_data['Sales_Amount'].sum() > 0:
            # Weighted average by sales amount
            weighted_credit = (cust_3m_data['Credit_Period'] * cust_3m_data['Sales_Amount']).sum() / cust_3m_data['Sales_Amount'].sum()
            customer_metrics.at[idx, 'Weighted_Avg_Credit_Period'] = round(weighted_credit, 1)
    
    # Calculate customer tenure in days
    customer_metrics['First_Invoice_Date'] = pd.to_datetime(customer_metrics['First_Invoice_Date'])
    customer_metrics['Last_Invoice_Date'] = pd.to_datetime(customer_metrics['Last_Invoice_Date'])
    customer_metrics['Customer_Tenure_Days'] = (customer_metrics['Last_Invoice_Date'] - customer_metrics['First_Invoice_Date']).dt.days
    
    # DSO for customer (using last month's sales)
    for idx, row in customer_metrics.iterrows():
        cust_code = row['Customer_Code']
        cust_df = df[df['Customer_Code'] == cust_code]
        
        # Get last month's sales for this customer
        current_date = datetime.now()
        last_month_start = datetime(current_date.year, current_date.month - 1 if current_date.month > 1 else 12, 1)
        last_month_end = datetime(current_date.year, current_date.month, 1) - timedelta(days=1)
        
        last_month_sales = cust_df[
            (cust_df['Invoice_Date'] >= last_month_start) & 
            (cust_df['Invoice_Date'] <= last_month_end)
        ]['Sales_Amount'].sum()
        
        customer_outstanding = cust_df[cust_df['Payment_Date'].isna()]['Amount_Outstanding'].sum()
        
        if last_month_sales > 0 and customer_outstanding >= 0:
            customer_dso = (customer_outstanding / last_month_sales) * DSO_DAYS
        else:
            customer_dso = 0
        
        customer_metrics.at[idx, 'DSO'] = round(customer_dso, 1)
    
    return customer_metrics.sort_values('Total_Outstanding', ascending=False)

def calculate_monthly_aging_trend(df):
    """Calculate aging bucket trend month by month"""
    # Get unique month-end dates
    df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])
    
    # Create month-end dates for last 12 months
    current_date = datetime.now()
    month_ends = []
    
    for i in range(12):
        month = current_date.month - i
        year = current_date.year
        if month <= 0:
            month += 12
            year -= 1
        
        # Get last day of month
        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year
        
        month_end = datetime(next_year, next_month, 1) - timedelta(days=1)
        month_ends.append(month_end)
    
    month_ends.sort()
    
    # Calculate aging for each month-end
    aging_trend = []
    
    for month_end in month_ends:
        # Get all unpaid invoices as of month-end
        month_data = df[
            (df['Invoice_Date'] <= month_end) & 
            (df['Payment_Date'].isna() | (df['Payment_Date'] > month_end))
        ].copy()
        
        # Calculate days overdue as of month-end
        month_data['Days_Overdue_As_Of'] = (month_end - month_data['Due_Date']).dt.days
        month_data['Days_Overdue_As_Of'] = month_data['Days_Overdue_As_Of'].clip(lower=0)
        
        # Categorize into aging buckets
        month_data['Aging_Bucket_As_Of'] = pd.cut(
            month_data['Days_Overdue_As_Of'],
            bins=[-1, 0, 30, 60, 90, float('inf')],
            labels=['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']
        )
        
        # Aggregate by aging bucket
        aging_summary = month_data.groupby('Aging_Bucket_As_Of')['Amount_Outstanding'].sum().reindex(
            ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days'], fill_value=0
        )
        
        total = aging_summary.sum()
        
        aging_trend.append({
            'Month_End': month_end.strftime('%Y-%m-%d'),
            'Current': aging_summary.get('Current', 0),
            '0-30 days': aging_summary.get('0-30 days', 0),
            '31-60 days': aging_summary.get('31-60 days', 0),
            '61-90 days': aging_summary.get('61-90 days', 0),
            '90+ days': aging_summary.get('90+ days', 0),
            'Total': total
        })
    
    return pd.DataFrame(aging_trend)

def get_delinquent_accounts(df, threshold_pct=2):
    """Get delinquent accounts with threshold"""
    total_receivables = df[df['Payment_Date'].isna()]['Amount_Outstanding'].sum()
    
    delinquent = df[
        (df['Payment_Date'].isna()) & 
        (df['Days_Overdue'] > 90) & 
        (df['Invoice_Date'] <= datetime.now() - timedelta(days=90))
    ].copy()
    
    if len(delinquent) > 0:
        # Calculate customer-wise delinquent amounts
        customer_delinquent = delinquent.groupby(['Customer_Code', 'Customer_Name']).agg({
            'Amount_Outstanding': 'sum',
            'Days_Overdue': 'max',
            'Sales_Employee': 'first',
            'City': 'first'
        }).reset_index()
        
        # Apply threshold: >2% of total receivables
        threshold_amount = total_receivables * (threshold_pct / 100)
        customer_delinquent = customer_delinquent[customer_delinquent['Amount_Outstanding'] > threshold_amount]
        
        return customer_delinquent.sort_values('Amount_Outstanding', ascending=False)
    
    return pd.DataFrame()

# === ADVANCED AI FEATURES ===

def train_payment_probability_model(df):
    """Train ML model to predict payment probability"""
    try:
        # Prepare features
        customer_features = df.groupby('Customer_Code').agg({
            'Days_To_Pay': 'mean',
            'Amount_Outstanding': 'sum',
            'Sales_Amount': 'sum',
            'Payment_Date': lambda x: x.notna().sum() / len(x) if len(x) > 0 else 0,  # Payment rate
            'Days_Overdue': 'mean',
            'Credit_Period': 'mean'
        }).reset_index()
        
        customer_features.columns = ['Customer_Code', 'Avg_Days_To_Pay', 'Total_Outstanding', 
                                     'Total_Sales', 'Payment_Rate', 'Avg_Days_Overdue', 'Avg_Credit_Period']
        
        # Handle NaN values
        customer_features['Avg_Days_To_Pay'] = customer_features['Avg_Days_To_Pay'].fillna(customer_features['Avg_Days_To_Pay'].median())
        customer_features['Avg_Days_Overdue'] = customer_features['Avg_Days_Overdue'].fillna(0)
        
        # Create target variable (1 = likely to pay, 0 = unlikely)
        customer_features['Will_Pay'] = ((customer_features['Payment_Rate'] > 0.6) & 
                                         (customer_features['Avg_Days_Overdue'] < 45)).astype(int)
        
        # Features for modeling
        feature_cols = ['Avg_Days_To_Pay', 'Total_Outstanding', 'Total_Sales', 
                       'Payment_Rate', 'Avg_Days_Overdue', 'Avg_Credit_Period']
        
        X = customer_features[feature_cols].fillna(0)
        y = customer_features['Will_Pay']
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X_scaled, y)
        
        # Get probability predictions
        probabilities = model.predict_proba(X_scaled)[:, 1]
        customer_features['Payment_Probability'] = probabilities * 100
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return customer_features, feature_importance, model, scaler
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None

def time_series_forecast(df, periods=90):
    """Generate time series forecast for cash flow"""
    try:
        # Aggregate daily collections
        df_paid = df[df['Payment_Date'].notna()].copy()
        if len(df_paid) == 0:
            return None, "No payment data available for forecasting"
            
        df_paid['Payment_Date'] = pd.to_datetime(df_paid['Payment_Date'])
        
        daily_collections = df_paid.groupby('Payment_Date')['Sales_Amount'].sum().reset_index()
        daily_collections = daily_collections.sort_values('Payment_Date')
        
        if len(daily_collections) < 30:
            return None, "Insufficient historical data for forecasting"
        
        # Calculate moving average and trend
        daily_collections['MA7'] = daily_collections['Sales_Amount'].rolling(window=7, min_periods=1).mean()
        daily_collections['MA30'] = daily_collections['Sales_Amount'].rolling(window=30, min_periods=1).mean()
        
        # Simple exponential smoothing
        alpha = 0.3
        avg_collection = daily_collections['Sales_Amount'].mean()
        std_collection = daily_collections['Sales_Amount'].std()
        
        # Generate forecast
        last_date = daily_collections['Payment_Date'].max()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
        
        # Account for weekly patterns
        recent_avg = daily_collections['MA30'].iloc[-1] if len(daily_collections) > 30 else avg_collection
        
        forecast_values = []
        for i in range(periods):
            # Add weekly seasonality (lower on weekends)
            day_of_week = (last_date + timedelta(days=i+1)).weekday()
            weekday_factor = 0.6 if day_of_week >= 5 else 1.0
            
            # Add trend and noise
            trend_factor = 1 + (0.01 * i / periods)  # Slight positive trend
            noise = np.random.normal(0, std_collection * 0.1)
            
            forecast_val = recent_avg * weekday_factor * trend_factor + noise
            forecast_values.append(max(0, forecast_val))
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_values,
            'Upper_Bound': [f * 1.2 for f in forecast_values],
            'Lower_Bound': [f * 0.8 for f in forecast_values]
        })
        
        return forecast_df, daily_collections
    except Exception as e:
        return None, f"Forecasting error: {str(e)}"

def payment_behavior_clustering(df):
    """Cluster customers based on payment behavior"""
    try:
        # Aggregate customer behavior
        customer_behavior = df.groupby('Customer_Code').agg({
            'Days_To_Pay': 'mean',
            'Amount_Outstanding': 'sum',
            'Payment_Date': lambda x: x.notna().sum() / len(x) if len(x) > 0 else 0,
            'Days_Overdue': 'max',
            'Sales_Amount': 'mean'
        }).reset_index()
        
        customer_behavior.columns = ['Customer_Code', 'Avg_Days_To_Pay', 'Total_Outstanding', 
                                     'Payment_Ratio', 'Max_Days_Overdue', 'Avg_Invoice_Size']
        
        # Handle missing values
        customer_behavior['Avg_Days_To_Pay'] = customer_behavior['Avg_Days_To_Pay'].fillna(
            customer_behavior['Avg_Days_To_Pay'].median())
        customer_behavior['Max_Days_Overdue'] = customer_behavior['Max_Days_Overdue'].fillna(0)
        
        # Normalize features
        feature_cols = ['Avg_Days_To_Pay', 'Total_Outstanding', 'Payment_Ratio', 
                       'Max_Days_Overdue', 'Avg_Invoice_Size']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(customer_behavior[feature_cols])
        
        # K-means clustering
        optimal_k = min(5, len(customer_behavior) // 3)
        if optimal_k < 2:
            optimal_k = 2
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        customer_behavior['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Assign cluster names based on characteristics
        cluster_profiles = customer_behavior.groupby('Cluster').agg({
            'Avg_Days_To_Pay': 'mean',
            'Payment_Ratio': 'mean',
            'Max_Days_Overdue': 'mean',
            'Total_Outstanding': 'sum'
        })
        
        cluster_names = {}
        for idx, row in cluster_profiles.iterrows():
            if row['Payment_Ratio'] > 0.8 and row['Avg_Days_To_Pay'] < 20:
                cluster_names[idx] = 'Excellent Payers'
            elif row['Payment_Ratio'] > 0.6 and row['Avg_Days_To_Pay'] < 45:
                cluster_names[idx] = 'Good Payers'
            elif row['Max_Days_Overdue'] > 60:
                cluster_names[idx] = 'High Risk'
            elif row['Payment_Ratio'] < 0.4:
                cluster_names[idx] = 'Delinquent'
            else:
                cluster_names[idx] = 'Average Payers'
        
        customer_behavior['Cluster_Name'] = customer_behavior['Cluster'].map(cluster_names)
        
        # Merge with original customer names
        customer_names = df[['Customer_Code', 'Customer_Name']].drop_duplicates()
        customer_behavior = customer_behavior.merge(customer_names, on='Customer_Code', how='left')
        
        return customer_behavior, cluster_profiles, kmeans
    except Exception as e:
        st.error(f"Clustering error: {str(e)}")
        return None, None, None

def detect_anomalies(df):
    """Detect anomalous payment patterns using Isolation Forest"""
    try:
        # Prepare features for anomaly detection
        customer_stats = df.groupby('Customer_Code').agg({
            'Days_To_Pay': 'mean',
            'Amount_Outstanding': 'sum',
            'Sales_Amount': 'std',
            'Days_Overdue': 'max',
            'Payment_Date': lambda x: x.notna().sum()
        }).reset_index()
        
        customer_stats.columns = ['Customer_Code', 'Avg_Days_To_Pay', 'Total_Outstanding', 
                                  'Sales_Volatility', 'Max_Overdue', 'Payment_Count']
        
        # Handle missing values
        customer_stats['Avg_Days_To_Pay'] = customer_stats['Avg_Days_To_Pay'].fillna(
            customer_stats['Avg_Days_To_Pay'].median())
        customer_stats['Sales_Volatility'] = customer_stats['Sales_Volatility'].fillna(0)
        customer_stats['Max_Overdue'] = customer_stats['Max_Overdue'].fillna(0)
        
        # Normalize
        feature_cols = ['Avg_Days_To_Pay', 'Total_Outstanding', 'Sales_Volatility', 
                       'Max_Overdue', 'Payment_Count']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(customer_stats[feature_cols])
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=42)
        customer_stats['Anomaly'] = iso_forest.fit_predict(X_scaled)
        customer_stats['Anomaly_Score'] = iso_forest.score_samples(X_scaled)
        
        # Flag anomalies
        customer_stats['Is_Anomaly'] = customer_stats['Anomaly'] == -1
        
        # Merge with customer names
        customer_names = df[['Customer_Code', 'Customer_Name', 'City', 'Sales_Employee']].drop_duplicates()
        anomalies = customer_stats.merge(customer_names, on='Customer_Code', how='left')
        
        # Get top anomalies
        top_anomalies = anomalies[anomalies['Is_Anomaly']].sort_values('Anomaly_Score').head(10)
        
        return anomalies, top_anomalies
    except Exception as e:
        st.error(f"Anomaly detection error: {str(e)}")
        return None, None

def optimize_collection_resources(df):
    """Optimize allocation of collection resources"""
    try:
        # Calculate ROI for each customer
        customer_roi = df.groupby('Customer_Code').agg({
            'Amount_Outstanding': 'sum',
            'Days_Overdue': 'max',
            'Sales_Amount': 'sum',
            'Customer_Name': 'first',
            'Sales_Employee': 'first'
        }).reset_index()
        
        customer_roi.columns = ['Customer_Code', 'Outstanding', 'Max_Overdue', 
                               'Total_Sales', 'Customer_Name', 'Sales_Employee']
        
        # Calculate priority score
        # Higher outstanding + higher overdue days = higher priority
        max_outstanding = customer_roi['Outstanding'].max()
        max_overdue = customer_roi['Max_Overdue'].max()
        
        if max_outstanding > 0 and max_overdue > 0:
            customer_roi['Priority_Score'] = (
                (customer_roi['Outstanding'] / max_outstanding) * 50 +
                (customer_roi['Max_Overdue'] / (max_overdue + 1)) * 50
            )
        else:
            customer_roi['Priority_Score'] = 0
        
        # Estimate effort required (normalized)
        customer_roi['Effort_Required'] = np.where(
            customer_roi['Max_Overdue'] > 90, 3,
            np.where(customer_roi['Max_Overdue'] > 60, 2, 1)
        )
        
        # Calculate expected recovery (pessimistic estimate)
        customer_roi['Median_Expected_Payment'] = np.where(
            customer_roi['Max_Overdue'] > 90, customer_roi['Outstanding'] * 0.5,
            np.where(customer_roi['Max_Overdue'] > 60, customer_roi['Outstanding'] * 0.7,
                    customer_roi['Outstanding'] * 0.9)
        )
        
        # ROI = Expected Recovery / Effort
        customer_roi['ROI'] = customer_roi['Median_Expected_Payment'] / customer_roi['Effort_Required']
        customer_roi['ROI'] = customer_roi['ROI'].fillna(0)
        
        # Sort by ROI
        customer_roi = customer_roi.sort_values('ROI', ascending=False)
        
        # Assign resource allocation
        customer_roi['Allocated_Hours'] = customer_roi['Effort_Required']
        
        # Cumulative allocation
        customer_roi['Cumulative_Hours'] = customer_roi['Allocated_Hours'].cumsum()
        customer_roi['Within_Capacity'] = customer_roi['Cumulative_Hours'] <= TEAM_CAPACITY_HOURS
        
        return customer_roi
    except Exception as e:
        st.error(f"Resource optimization error: {str(e)}")
        return None

def forecast_cashflow_breakdown(df, periods=90):
    """Generate cash flow forecast breakdown by customer, sales person, and office"""
    try:
        # Top 10 customers forecast
        top_customers = df.groupby('Customer_Name')['Amount_Outstanding'].sum().nlargest(10).index
        customer_forecasts = {}
        
        for customer in top_customers:
            cust_df = df[df['Customer_Name'] == customer]
            if len(cust_df) > 5:  # Only forecast if sufficient data
                # Simple forecast based on historical payment rate
                paid_amount = cust_df[cust_df['Payment_Date'].notna()]['Sales_Amount'].sum()
                total_amount = cust_df['Sales_Amount'].sum()
                payment_rate = paid_amount / total_amount if total_amount > 0 else 0.5
                customer_forecasts[customer] = cust_df['Amount_Outstanding'].sum() * payment_rate * 0.7
        
        # Sales person forecast
        sales_forecasts = {}
        for sales_person in df['Sales_Employee'].unique():
            sales_df = df[df['Sales_Employee'] == sales_person]
            if len(sales_df) > 5:
                paid_amount = sales_df[sales_df['Payment_Date'].notna()]['Sales_Amount'].sum()
                total_amount = sales_df['Sales_Amount'].sum()
                payment_rate = paid_amount / total_amount if total_amount > 0 else 0.5
                sales_forecasts[sales_person] = sales_df['Amount_Outstanding'].sum() * payment_rate * 0.7
        
        # Sales office forecast
        office_forecasts = {}
        for office in df['Sales_Office'].unique():
            office_df = df[df['Sales_Office'] == office]
            if len(office_df) > 5:
                paid_amount = office_df[office_df['Payment_Date'].notna()]['Sales_Amount'].sum()
                total_amount = office_df['Sales_Amount'].sum()
                payment_rate = paid_amount / total_amount if total_amount > 0 else 0.5
                office_forecasts[office] = office_df['Amount_Outstanding'].sum() * payment_rate * 0.7
        
        return {
            'customers': customer_forecasts,
            'sales_persons': sales_forecasts,
            'offices': office_forecasts
        }
    except Exception as e:
        st.error(f"Cash flow breakdown error: {str(e)}")
        return None

def working_capital_sensitivity_analysis(df, base_forecast_df):
    """Perform working capital sensitivity analysis based on time series forecast"""
    try:
        current_wc = df['Amount_Outstanding'].sum()
        if base_forecast_df is not None and len(base_forecast_df) > 0:
            base_forecast = base_forecast_df['Forecast'].sum()
        else:
            base_forecast = current_wc * 0.5
        
        scenarios = {
            'Optimistic': base_forecast * 1.2,
            'Base Case': base_forecast,
            'Pessimistic': base_forecast * 0.8,
            'Severe Downturn': base_forecast * 0.6
        }
        
        sensitivity_results = []
        for scenario, forecast_collection in scenarios.items():
            projected_wc = max(0, current_wc - forecast_collection)
            wc_reduction = current_wc - projected_wc
            reduction_pct = (wc_reduction / current_wc * 100) if current_wc > 0 else 0
            
            sensitivity_results.append({
                'Scenario': scenario,
                'Forecast_Collection': forecast_collection,
                'Projected_WC': projected_wc,
                'WC_Reduction': wc_reduction,
                'Reduction_Pct': reduction_pct
            })
        
        return pd.DataFrame(sensitivity_results)
    except Exception as e:
        st.error(f"Sensitivity analysis error: {str(e)}")
        return None

# ============================================================================
# INTERACTIVE DRILL-DOWN FEATURES
# ============================================================================

def show_interactive_aging_drilldown(df):
    """
    Interactive aging analysis with multiple drill-down levels
    Level 1: Aging Buckets -> Level 2: Customers -> Level 3: Invoices
    """
    st.markdown("## 🔍 Interactive Aging Drill-Down")
    
    # Initialize session state for drill-down tracking
    if 'aging_drilldown_level' not in st.session_state:
        st.session_state.aging_drilldown_level = 'buckets'
    if 'selected_aging_bucket' not in st.session_state:
        st.session_state.selected_aging_bucket = None
    if 'selected_customer' not in st.session_state:
        st.session_state.selected_customer = None
    
    unpaid_df = df[df['Payment_Date'].isna()].copy()
    
    # Breadcrumb navigation
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🏠 Reset to Buckets", use_container_width=True):
            st.session_state.aging_drilldown_level = 'buckets'
            st.session_state.selected_aging_bucket = None
            st.session_state.selected_customer = None
            st.rerun()
    
    with col2:
        if st.session_state.aging_drilldown_level == 'invoices' and st.button("⬅️ Back to Customers", use_container_width=True):
            st.session_state.aging_drilldown_level = 'customers'
            st.session_state.selected_customer = None
            st.rerun()
    
    with col3:
        # Show current path
        path = "📊 Aging Buckets"
        if st.session_state.selected_aging_bucket:
            path += f" > 👥 {st.session_state.selected_aging_bucket}"
        if st.session_state.selected_customer:
            path += f" > 📄 {st.session_state.selected_customer}"
        st.markdown(f"**Current View:** {path}")
    
    st.markdown("---")
    
    # LEVEL 1: Aging Buckets View
    if st.session_state.aging_drilldown_level == 'buckets':
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            # Interactive aging bucket chart
            aging_dist = unpaid_df.groupby('Aging_Bucket')['Amount_Outstanding'].sum().reset_index()
            aging_dist = aging_dist.sort_values('Amount_Outstanding', ascending=False)
            
            fig = px.bar(
                aging_dist,
                x='Aging_Bucket',
                y='Amount_Outstanding',
                title='Click on any bucket to drill down into customers',
                color='Amount_Outstanding',
                color_continuous_scale='RdYlGn_r',
                text='Amount_Outstanding'
            )
            fig.update_traces(
                texttemplate='₹%{text:.2s}',
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Amount: ₹%{y:,.0f}<br><extra></extra>'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Clickable bucket selection
            st.markdown("### 🎯 Select Aging Bucket to Drill Down")
            selected_bucket = st.selectbox(
                "Choose a bucket:",
                options=aging_dist['Aging_Bucket'].tolist(),
                key='bucket_selector'
            )
            
            if st.button("🔍 Drill Down into Customers", use_container_width=True):
                st.session_state.selected_aging_bucket = selected_bucket
                st.session_state.aging_drilldown_level = 'customers'
                st.rerun()
        
        with col2:
            # Summary metrics
            st.markdown("### 📊 Bucket Summary")
            for bucket in aging_dist['Aging_Bucket']:
                bucket_amount = aging_dist[aging_dist['Aging_Bucket'] == bucket]['Amount_Outstanding'].iloc[0]
                bucket_count = len(unpaid_df[unpaid_df['Aging_Bucket'] == bucket])
                total_amount = aging_dist['Amount_Outstanding'].sum()
                pct = (bucket_amount / total_amount * 100) if total_amount > 0 else 0
                
                st.metric(
                    f"{bucket}",
                    f"₹{bucket_amount/1000000:.2f}M",
                    delta=f"{bucket_count} accounts ({pct:.1f}%)"
                )
    
    # LEVEL 2: Customers in Selected Bucket
    elif st.session_state.aging_drilldown_level == 'customers':
        bucket = st.session_state.selected_aging_bucket
        st.markdown(f"## 👥 Customers in '{bucket}' Bucket")
        
        bucket_df = unpaid_df[unpaid_df['Aging_Bucket'] == bucket]
        customer_summary = bucket_df.groupby(['Customer_Name', 'Customer_Code']).agg({
            'Amount_Outstanding': 'sum',
            'Days_Overdue': 'max',
            'Invoice_Date': 'count',
            'Sales_Employee': 'first',
            'City': 'first'
        }).reset_index()
        customer_summary.columns = ['Customer_Name', 'Customer_Code', 'Outstanding', 'Max_Overdue', 
                                    'Invoice_Count', 'Sales_Employee', 'City']
        customer_summary = customer_summary.sort_values('Outstanding', ascending=False)
        
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            # Customer chart
            fig = px.bar(
                customer_summary.head(15),
                x='Customer_Name',
                y='Outstanding',
                title=f'Top 15 Customers in {bucket}',
                color='Max_Overdue',
                color_continuous_scale='Reds',
                text='Outstanding',
                hover_data=['Invoice_Count', 'Sales_Employee']
            )
            fig.update_traces(texttemplate='₹%{text:.2s}', textposition='outside')
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Customer selection
            st.markdown("### 🎯 Select Customer to View Invoices")
            selected_customer = st.selectbox(
                "Choose a customer:",
                options=customer_summary['Customer_Name'].tolist(),
                key='customer_selector'
            )
            
            if st.button("🔍 View Customer Invoices", use_container_width=True):
                st.session_state.selected_customer = selected_customer
                st.session_state.aging_drilldown_level = 'invoices'
                st.rerun()
        
        with col2:
            st.markdown("### 📋 Customer Details")
            st.dataframe(
                customer_summary[['Customer_Name', 'Outstanding', 'Invoice_Count', 'Max_Overdue', 'Sales_Employee']].style.format({
                    'Outstanding': '₹{:,.0f}',
                    'Max_Overdue': '{:.0f} days'
                }),
                use_container_width=True,
                height=400
            )
    
    # LEVEL 3: Invoices for Selected Customer
    elif st.session_state.aging_drilldown_level == 'invoices':
        customer = st.session_state.selected_customer
        bucket = st.session_state.selected_aging_bucket
        
        st.markdown(f"## 📄 Invoices: {customer} ({bucket})")
        
        customer_invoices = unpaid_df[unpaid_df['Customer_Name'] == customer].copy()
        customer_invoices = customer_invoices.sort_values('Days_Overdue', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Invoice timeline
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=customer_invoices['Invoice_Date'],
                y=customer_invoices['Amount_Outstanding'],
                mode='markers+lines',
                marker=dict(
                    size=15,
                    color=customer_invoices['Days_Overdue'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Days<br>Overdue")
                ),
                text=customer_invoices['Days_Overdue'],
                hovertemplate='<b>Invoice Date:</b> %{x}<br><b>Amount:</b> ₹%{y:,.0f}<br><b>Days Overdue:</b> %{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Invoice Timeline for {customer}',
                xaxis_title='Invoice Date',
                yaxis_title='Outstanding Amount (₹)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer summary
            total_outstanding = customer_invoices['Amount_Outstanding'].sum()
            avg_overdue = customer_invoices['Days_Overdue'].mean()
            max_overdue = customer_invoices['Days_Overdue'].max()
            invoice_count = len(customer_invoices)
            
            st.metric("Total Outstanding", f"₹{total_outstanding/1000000:.2f}M")
            st.metric("Number of Invoices", invoice_count)
            st.metric("Average Days Overdue", f"{avg_overdue:.0f} days")
            st.metric("Maximum Days Overdue", f"{max_overdue:.0f} days")
        
        # Detailed invoice table
        st.markdown("### 📋 Invoice Details")
        display_invoices = customer_invoices[['Invoice_Date', 'Due_Date', 'Amount_Outstanding', 
                                              'Days_Overdue', 'Credit_Period', 'Sales_Employee']].copy()
        display_invoices['Invoice_Date'] = display_invoices['Invoice_Date'].dt.strftime('%Y-%m-%d')
        display_invoices['Due_Date'] = display_invoices['Due_Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            display_invoices.style
            .background_gradient(cmap='Reds', subset=['Days_Overdue'])
            .format({
                'Amount_Outstanding': '₹{:,.0f}',
                'Days_Overdue': '{:.0f} days',
                'Credit_Period': '{:.0f} days'
            }),
            use_container_width=True,
            height=400
        )


def show_geographic_drilldown(df):
    """
    Geographic drill-down: Region -> City -> Customers -> Invoices
    """
    st.markdown("## 🌍 Geographic Drill-Down Analysis")
    
    # Initialize session state
    if 'geo_level' not in st.session_state:
        st.session_state.geo_level = 'office'
    if 'selected_office' not in st.session_state:
        st.session_state.selected_office = None
    if 'selected_city' not in st.session_state:
        st.session_state.selected_city = None
    if 'selected_geo_customer' not in st.session_state:
        st.session_state.selected_geo_customer = None
    
    unpaid_df = df[df['Payment_Date'].isna()].copy()
    
    # Navigation
    cols = st.columns([1, 1, 1, 2])
    with cols[0]:
        if st.button("🏢 Sales Offices", use_container_width=True):
            st.session_state.geo_level = 'office'
            st.session_state.selected_office = None
            st.session_state.selected_city = None
            st.session_state.selected_geo_customer = None
            st.rerun()
    
    with cols[1]:
        if st.session_state.selected_office and st.button("🏙️ Cities", use_container_width=True):
            st.session_state.geo_level = 'city'
            st.session_state.selected_city = None
            st.session_state.selected_geo_customer = None
            st.rerun()
    
    with cols[2]:
        if st.session_state.selected_city and st.button("👥 Customers", use_container_width=True):
            st.session_state.geo_level = 'customer'
            st.session_state.selected_geo_customer = None
            st.rerun()
    
    with cols[3]:
        path = "🏢 Offices"
        if st.session_state.selected_office:
            path += f" > 🏙️ {st.session_state.selected_office}"
        if st.session_state.selected_city:
            path += f" > 👥 {st.session_state.selected_city}"
        if st.session_state.selected_geo_customer:
            path += f" > 📄 {st.session_state.selected_geo_customer}"
        st.markdown(f"**Path:** {path}")
    
    st.markdown("---")
    
    # LEVEL 1: Sales Office View
    if st.session_state.geo_level == 'office':
        office_summary = unpaid_df.groupby('Sales_Office').agg({
            'Amount_Outstanding': 'sum',
            'Customer_Code': 'nunique',
            'Days_Overdue': 'mean'
        }).reset_index()
        office_summary.columns = ['Office', 'Outstanding', 'Customers', 'Avg_Overdue']
        
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            fig = px.bar(
                office_summary,
                x='Office',
                y='Outstanding',
                title='Outstanding by Sales Office',
                color='Avg_Overdue',
                color_continuous_scale='RdYlGn_r',
                text='Outstanding',
                hover_data=['Customers']
            )
            fig.update_traces(texttemplate='₹%{text:.2s}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            selected_office = st.selectbox("Select Office:", office_summary['Office'].tolist())
            if st.button("🔍 Drill Down to Cities"):
                st.session_state.selected_office = selected_office
                st.session_state.geo_level = 'city'
                st.rerun()
        
        with col2:
            st.markdown("### 📊 Office Metrics")
            st.dataframe(
                office_summary.style.format({
                    'Outstanding': '₹{:,.0f}',
                    'Avg_Overdue': '{:.1f} days'
                }),
                use_container_width=True
            )
    
    # LEVEL 2: City View
    elif st.session_state.geo_level == 'city':
        office = st.session_state.selected_office
        office_df = unpaid_df[unpaid_df['Sales_Office'] == office]
        
        city_summary = office_df.groupby('City').agg({
            'Amount_Outstanding': 'sum',
            'Customer_Code': 'nunique',
            'Days_Overdue': 'mean'
        }).reset_index()
        city_summary.columns = ['City', 'Outstanding', 'Customers', 'Avg_Overdue']
        
        st.markdown(f"## 🏙️ Cities in {office} Office")
        
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            fig = px.treemap(
                city_summary,
                path=['City'],
                values='Outstanding',
                color='Avg_Overdue',
                color_continuous_scale='RdYlGn_r',
                title=f'Outstanding Distribution - {office}'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            selected_city = st.selectbox("Select City:", city_summary['City'].tolist())
            if st.button("🔍 View Customers"):
                st.session_state.selected_city = selected_city
                st.session_state.geo_level = 'customer'
                st.rerun()
        
        with col2:
            st.dataframe(
                city_summary.style.format({
                    'Outstanding': '₹{:,.0f}',
                    'Avg_Overdue': '{:.1f} days'
                }),
                use_container_width=True
            )
    
    # LEVEL 3: Customer View
    elif st.session_state.geo_level == 'customer':
        city = st.session_state.selected_city
        city_df = unpaid_df[unpaid_df['City'] == city]
        
        customer_summary = city_df.groupby('Customer_Name').agg({
            'Amount_Outstanding': 'sum',
            'Days_Overdue': 'max',
            'Invoice_Date': 'count'
        }).reset_index().sort_values('Amount_Outstanding', ascending=False)
        
        st.markdown(f"## 👥 Customers in {city}")
        
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            fig = px.bar(
                customer_summary.head(10),
                x='Customer_Name',
                y='Amount_Outstanding',
                title=f'Top 10 Customers - {city}',
                color='Days_Overdue',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                customer_summary.style.format({
                    'Amount_Outstanding': '₹{:,.0f}',
                    'Days_Overdue': '{:.0f} days'
                }),
                use_container_width=True
            )


def show_sales_team_drilldown(df):
    """
    Sales team drill-down: Team -> Employee -> Customers -> Invoices
    """
    st.markdown("## 👨‍💼 Sales Team Performance Drill-Down")
    
    if 'sales_level' not in st.session_state:
        st.session_state.sales_level = 'team'
    if 'selected_employee' not in st.session_state:
        st.session_state.selected_employee = None
    if 'selected_sales_customer' not in st.session_state:
        st.session_state.selected_sales_customer = None
    
    unpaid_df = df[df['Payment_Date'].isna()].copy()
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("👥 Team Overview", use_container_width=True):
            st.session_state.sales_level = 'team'
            st.session_state.selected_employee = None
            st.session_state.selected_sales_customer = None
            st.rerun()
    
    with col2:
        if st.session_state.selected_employee and st.button("⬅️ Back", use_container_width=True):
            st.session_state.sales_level = 'employee'
            st.session_state.selected_sales_customer = None
            st.rerun()
    
    with col3:
        path = "👥 Team"
        if st.session_state.selected_employee:
            path += f" > 👤 {st.session_state.selected_employee}"
        if st.session_state.selected_sales_customer:
            path += f" > 📄 {st.session_state.selected_sales_customer}"
        st.markdown(f"**Path:** {path}")
    
    st.markdown("---")
    
    # LEVEL 1: Team Overview
    if st.session_state.sales_level == 'team':
        employee_summary = unpaid_df.groupby('Sales_Employee').agg({
            'Amount_Outstanding': 'sum',
            'Customer_Code': 'nunique',
            'Days_Overdue': 'mean',
            'Invoice_Date': 'count'
        }).reset_index()
        employee_summary.columns = ['Employee', 'Outstanding', 'Customers', 'Avg_Overdue', 'Invoices']
        
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            # Bubble chart
            fig = px.scatter(
                employee_summary,
                x='Customers',
                y='Outstanding',
                size='Invoices',
                color='Avg_Overdue',
                text='Employee',
                title='Sales Team Performance Matrix',
                color_continuous_scale='RdYlGn_r',
                size_max=60
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            selected_employee = st.selectbox("Select Employee:", employee_summary['Employee'].tolist())
            if st.button("🔍 View Employee's Customers"):
                st.session_state.selected_employee = selected_employee
                st.session_state.sales_level = 'employee'
                st.rerun()
        
        with col2:
            st.markdown("### 📊 Team Rankings")
            employee_summary['Rank'] = employee_summary['Outstanding'].rank(ascending=False).astype(int)
            st.dataframe(
                employee_summary[['Rank', 'Employee', 'Outstanding', 'Customers', 'Avg_Overdue']].style
                .background_gradient(cmap='RdYlGn_r', subset=['Outstanding'])
                .format({
                    'Outstanding': '₹{:,.0f}',
                    'Avg_Overdue': '{:.1f} days'
                }),
                use_container_width=True
            )
    
    # LEVEL 2: Employee's Customers
    elif st.session_state.sales_level == 'employee':
        employee = st.session_state.selected_employee
        employee_df = unpaid_df[unpaid_df['Sales_Employee'] == employee]
        
        st.markdown(f"## 👤 {employee}'s Portfolio")
        
        customer_summary = employee_df.groupby('Customer_Name').agg({
            'Amount_Outstanding': 'sum',
            'Days_Overdue': 'max',
            'Invoice_Date': 'count',
            'City': 'first'
        }).reset_index().sort_values('Amount_Outstanding', ascending=False)
        
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            fig = px.sunburst(
                customer_summary,
                path=['City', 'Customer_Name'],
                values='Amount_Outstanding',
                color='Days_Overdue',
                color_continuous_scale='RdYlGn_r',
                title=f"{employee}'s Customer Distribution"
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Customer List")
            st.dataframe(
                customer_summary.style
                .background_gradient(cmap='Reds', subset=['Days_Overdue'])
                .format({
                    'Amount_Outstanding': '₹{:,.0f}',
                    'Days_Overdue': '{:.0f} days'
                }),
                use_container_width=True,
                height=450
            )

# Main Application
def main():
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = False
    
    # Initialize session state for AI responses with limits
    if 'ai_responses' not in st.session_state:
        st.session_state.ai_responses = {}
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/money-circulation.png", width=80)
        st.title("🎯 Navigation")
        
        section = st.radio(
            "Select Section",
            ["📊 Dashboard", "📈 Descriptive Analytics", "🔮 Predictive Analysis with ML", 
             "💡 Action Recommendations"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # File upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload your data (CSV)", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Using sample data for demo")
            df = load_sample_data()
        
        df = calculate_metrics(df)
        
        st.markdown("---")
        
        # Filters
        st.subheader("🔍 Filters")
        
        selected_office = st.multiselect(
            "Sales Office",
            options=['All'] + list(df['Sales_Office'].unique()),
            default=['All']
        )
        
        selected_city = st.multiselect(
            "City",
            options=['All'] + list(df['City'].unique()),
            default=['All']
        )
        
        selected_employee = st.multiselect(
            "Sales Employee",
            options=['All'] + list(df['Sales_Employee'].unique()),
            default=['All']
        )
        
        # Apply filters
        if 'All' not in selected_office:
            df = df[df['Sales_Office'].isin(selected_office)]
        if 'All' not in selected_city:
            df = df[df['City'].isin(selected_city)]
        if 'All' not in selected_employee:
            df = df[df['Sales_Employee'].isin(selected_employee)]
        
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        
        st.download_button(
            label="📄 Download CSV",
            data=df.to_csv(index=False),
            file_name=f"receivables_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # AI Assistant Setup
        st.markdown("---")
        st.subheader("🤖 AI Assistant Setup")
        
        api_key = st.text_input("Enter Gemini API Key", type="password",
                               help="Get free API key from https://aistudio.google.com/")
        
        if api_key:
            try:
                if setup_ai_assistant(api_key):
                    st.session_state.api_key_configured = True
                    st.success("✅ AI Assistant Ready!")
                else:
                    st.error("❌ Failed to configure API key")
                    st.session_state.api_key_configured = False
            except Exception as e:
                st.error(f"❌ Invalid API key: {str(e)}")
                st.session_state.api_key_configured = False
        else:
            st.info("🔑 Add API key to enable AI analysis")
            st.session_state.api_key_configured = False
    
    # Main content based on section
    if section == "📊 Dashboard":
        show_dashboard(df)
    elif section == "📈 Descriptive Analytics":
        show_descriptive_analytics(df)
    elif section == "🔮 Predictive Analysis with ML":
        show_predictive_analysis_ml(df)
    elif section == "💡 Action Recommendations":
        show_action_recommendations(df)

def show_dashboard(df):
    """Display main dashboard with corrected metrics"""
    st.title("💰 Receivables AI Assistant")
    st.markdown("### Welcome to your intelligent receivables management dashboard")
    
    # Key Metrics with corrected formulas - Only keep requested ones
    col1, col2, col3, col4 = st.columns(4)
    
    total_ar = df[df['Payment_Date'].isna()]['Amount_Outstanding'].sum()
    dso = calculate_dso(df)
    recent_avg_dtp, _ = calculate_recent_days_to_pay(df, months=2)
    payment_within_terms = calculate_payment_within_terms(df)
    
    # Calculate collection rates as per specification
    collection_15d = calculate_collection_rates_within_days(df, days=15)
    collection_30d = calculate_collection_rates_within_days(df, days=30)
    collection_60d = calculate_collection_rates_within_days(df, days=60)
    collection_90d = calculate_collection_rates_within_days(df, days=90)
    
    overdue_rate = calculate_overdue_rate(df)
    
    with col1:
        st.metric(
            "Total A/R",
            format_currency(total_ar),
            delta=f"{((total_ar - df['Sales_Amount'].sum() * 0.8) / (df['Sales_Amount'].sum() * 0.8) * 100):.1f}%" if df['Sales_Amount'].sum() > 0 else "0%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "DSO",
            f"{dso} days",
            delta=f"{(dso - 45):+.0f} days",
            delta_color="inverse" if dso > 45 else "normal"
        )
    
    with col3:
        st.metric(
            "Avg Days to Pay",
            f"{recent_avg_dtp:.0f} days",
            delta=f"{(recent_avg_dtp - 45):+.0f} days",
            delta_color="inverse" if recent_avg_dtp > 45 else "normal"
        )
    
    with col4:
        st.metric(
            "Payment Within Terms",
            f"{payment_within_terms}%",
            delta=f"{(payment_within_terms - 80):+.1f}%",
            delta_color="normal" if payment_within_terms >= 80 else "inverse"
        )
    
    st.markdown("---")
    
    # Second row of collection rate metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "15-Day Collection Rate",
            f"{collection_15d}%",
            delta=f"{(collection_15d - 70):+.1f}%",
            delta_color="normal" if collection_15d >= 70 else "inverse"
        )
    
    with col6:
        st.metric(
            "30-Day Collection Rate",
            f"{collection_30d}%",
            delta=f"{(collection_30d - 80):+.1f}%",
            delta_color="normal" if collection_30d >= 80 else "inverse"
        )
    
    with col7:
        st.metric(
            "60-Day Collection Rate",
            f"{collection_60d}%",
            delta=f"{(collection_60d - 90):+.1f}%",
            delta_color="normal" if collection_60d >= 90 else "inverse"
        )
    
    with col8:
        st.metric(
            "90+ Day Collection Rate",
            f"{collection_90d}%",
            delta=f"{(collection_90d - 95):+.1f}%",
            delta_color="normal" if collection_90d >= 95 else "inverse"
        )
    
    st.markdown("---")
    
    # Third row - Overdue Rate
    col9, col10, col11, col12 = st.columns(4)
    
    with col9:
        st.metric(
            "Overdue Rate",
            f"{overdue_rate:.1f}%",
            delta=f"{(overdue_rate - 15):+.1f}%",
            delta_color="inverse" if overdue_rate > 15 else "normal"
        )
    
    with col10:
        # Empty column for spacing
        pass
    
    with col11:
        # Empty column for spacing
        pass
    
    with col12:
        # Empty column for spacing
        pass
    
    st.markdown("---")
    
    # === AGING BUCKET DISTRIBUTION WITH UPDATED COLOR SCHEME ===
    st.subheader("📊 Aging Bucket Distribution (Unpaid Amounts Only)")
    
    # Filter only unpaid invoices for aging analysis
    unpaid_df = df[df['Payment_Date'].isna()].copy()
    
    if len(unpaid_df) > 0:
        aging_dist = unpaid_df.groupby('Aging_Bucket')['Amount_Outstanding'].sum().reset_index()
        
        # Define the correct chronological order
        bucket_order = ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']
        
        # Create a categorical type with the specific order
        aging_dist['Aging_Bucket'] = pd.Categorical(
            aging_dist['Aging_Bucket'], 
            categories=bucket_order, 
            ordered=True
        )
        
        # Sort by the categorical order
        aging_dist = aging_dist.sort_values('Aging_Bucket')
        
        # Calculate percentages
        total_outstanding = aging_dist['Amount_Outstanding'].sum()
        aging_dist['Percentage'] = (aging_dist['Amount_Outstanding'] / total_outstanding * 100).round(1)
        
        # Define custom color scale based on aging buckets
        color_map = {
            'Current': '#006400',      # Dark Green
            '0-30 days': '#228B22',    # Forest Green
            '31-60 days': '#32CD32',   # Lime Green
            '61-90 days': '#FFFF99',   # Light Yellow
            '90+ days': '#FF0000'      # Red
        }
        
        fig = px.bar(
            aging_dist,
            x='Aging_Bucket',
            y='Amount_Outstanding',
            title='Outstanding Amount by Aging Bucket (Unpaid Only)',
            color='Aging_Bucket',
            color_discrete_map=color_map,
            labels={
                'Amount_Outstanding': 'Amount (₹)', 
                'Aging_Bucket': 'Aging Period',
                'Percentage': 'Percentage (%)'
            },
            text='Amount_Outstanding',
            hover_data={
                'Amount_Outstanding': ':,.0f',
                'Aging_Bucket': True,
                'Percentage': ':.1f%'
            }
        )
        
        # Format the text on bars
        fig.update_traces(
            texttemplate='₹%{text:.2s}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'Amount: ₹%{y:,.0f}<br>' +
                         'Percentage: %{customdata[0]:.1f}%<br>' +
                         '<extra></extra>',
            customdata=aging_dist[['Percentage']].values
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Aging Bucket',
            yaxis_title='Outstanding Amount (₹)',
            xaxis={'categoryorder': 'array', 'categoryarray': bucket_order}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_amount = aging_dist[aging_dist['Aging_Bucket'] == 'Current']['Amount_Outstanding'].sum() if 'Current' in aging_dist['Aging_Bucket'].values else 0
            current_pct = aging_dist[aging_dist['Aging_Bucket'] == 'Current']['Percentage'].sum() if 'Current' in aging_dist['Aging_Bucket'].values else 0
            st.metric(
                "Current Amount",
                format_currency(current_amount),
                delta=f"{current_pct:.1f}% of total"
            )
        
        with col2:
            overdue_30_60 = aging_dist[aging_dist['Aging_Bucket'].isin(['0-30 days', '31-60 days'])]['Amount_Outstanding'].sum()
            overdue_30_60_pct = aging_dist[aging_dist['Aging_Bucket'].isin(['0-30 days', '31-60 days'])]['Percentage'].sum()
            st.metric(
                "30-60 Days Overdue",
                format_currency(overdue_30_60),
                delta=f"{overdue_30_60_pct:.1f}% of total"
            )
        
        with col3:
            overdue_90_plus = aging_dist[aging_dist['Aging_Bucket'] == '90+ days']['Amount_Outstanding'].sum() if '90+ days' in aging_dist['Aging_Bucket'].values else 0
            overdue_90_plus_pct = aging_dist[aging_dist['Aging_Bucket'] == '90+ days']['Percentage'].sum() if '90+ days' in aging_dist['Aging_Bucket'].values else 0
            st.metric(
                "90+ Days Overdue",
                format_currency(overdue_90_plus),
                delta=f"{overdue_90_plus_pct:.1f}% of total",
                delta_color="inverse" if overdue_90_plus > 0 else "normal"
            )
    else:
        st.success("✅ All invoices are paid! No aging analysis needed.")
    
    st.markdown("---")
    
    # === AI ANALYSIS FOR CXO-LEVEL OVERVIEW ===
    st.subheader("🤖 AI Executive Summary")
    
    if st.button("🔍 Get Executive Insights", key="ai_executive"):
        # Get delinquent accounts data from descriptive analytics section
        delinquent_accounts = get_delinquent_accounts(df, threshold_pct=2)
        
        # Get aging trend data from descriptive analytics section
        aging_trend = calculate_monthly_aging_trend(df)
        
        # Get customer metrics from descriptive analytics section
        customer_metrics = get_customer_detailed_metrics(df)
        
        # Prepare comprehensive CXO-level data summary WITH ACTUAL AMOUNTS
        data_info = f"""
        EXECUTIVE SUMMARY - RECEIVABLES MANAGEMENT
        
        PORTFOLIO OVERVIEW:
        - Total Receivables: {format_currency(total_ar)}
        - Total Customers: {df['Customer_Code'].nunique()}
        - Total Invoices Outstanding: {len(unpaid_df) if len(unpaid_df) > 0 else 0}
        
        KEY PERFORMANCE INDICATORS:
        - DSO: {dso} days (target: 45 days)
        - Average Days to Pay: {recent_avg_dtp:.0f} days
        - Payment Within Terms: {payment_within_terms}%
        - Overdue Rate: {overdue_rate:.1f}%
        
        COLLECTION EFFICIENCY:
        - 15-Day Collection Rate: {collection_15d}%
        - 30-Day Collection Rate: {collection_30d}%
        - 60-Day Collection Rate: {collection_60d}%
        - 90+ Day Collection Rate: {collection_90d}%
        
        AGING ANALYSIS (UNPAID ONLY):
        """
        
        # Add aging distribution WITH ACTUAL AMOUNTS
        if len(unpaid_df) > 0:
            aging_buckets = ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']
            for bucket in aging_buckets:
                bucket_amount = aging_dist[aging_dist['Aging_Bucket'] == bucket]['Amount_Outstanding'].sum() if bucket in aging_dist['Aging_Bucket'].values else 0
                bucket_pct = (bucket_amount / total_ar * 100) if total_ar > 0 else 0
                data_info += f"\n- {bucket}: {format_currency(bucket_amount)} ({bucket_pct:.1f}%)"
        
        # Add DSO trend analysis WITH ACTUAL DATA
        data_info += f"""
        
        DSO TREND ANALYSIS:
        - Current DSO: {dso} days
        - Industry Benchmark: 45 days
        - Status: {'Above Benchmark' if dso > 45 else 'At/Below Benchmark'}
        - Cash Flow Impact: {'Significant pressure' if dso > 60 else 'Moderate impact' if dso > 45 else 'Healthy'}
        - Monthly Working Capital Requirement: {format_currency(total_ar / 30 * dso) if dso > 0 else format_currency(0)}
        
        HIGH RISK ACCOUNTS (DELINQUENT):
        - Number of Delinquent Accounts (>90 days & >2% threshold): {len(delinquent_accounts)}
        - Total Amount at High Risk: {format_currency(delinquent_accounts['Amount_Outstanding'].sum()) if len(delinquent_accounts) > 0 else format_currency(0)}
        - Percentage of Portfolio at High Risk: {(delinquent_accounts['Amount_Outstanding'].sum() / total_ar * 100) if len(delinquent_accounts) > 0 and total_ar > 0 else 0:.1f}%
        """
        
        # Add specific delinquent accounts if any
        if len(delinquent_accounts) > 0:
            data_info += "\n\nTOP DELINQUENT ACCOUNTS:\n"
            for idx, row in delinquent_accounts.head(3).iterrows():
                data_info += f"- {row['Customer_Name']}: {format_currency(row['Amount_Outstanding'])} ({row['Days_Overdue']} days overdue)\n"
        
        # Add aging bucket movement analysis
        if len(aging_trend) >= 2:
            current_month = aging_trend.iloc[-1]
            prev_month = aging_trend.iloc[-2]
            
            data_info += f"""
            
        AGING BUCKET MOVEMENT (LAST 2 MONTHS):
        - 90+ Days Bucket: {format_currency(current_month['90+ days'])} (Previously: {format_currency(prev_month['90+ days'])}) - Change: {((current_month['90+ days'] - prev_month['90+ days']) / prev_month['90+ days'] * 100) if prev_month['90+ days'] > 0 else 0:+.1f}%
        - 61-90 Days Bucket: {format_currency(current_month['61-90 days'])} (Previously: {format_currency(prev_month['61-90 days'])}) - Change: {((current_month['61-90 days'] - prev_month['61-90 days']) / prev_month['61-90 days'] * 100) if prev_month['61-90 days'] > 0 else 0:+.1f}%
        - 31-60 Days Bucket: {format_currency(current_month['31-60 days'])} (Previously: {format_currency(prev_month['31-60 days'])}) - Change: {((current_month['31-60 days'] - prev_month['31-60 days']) / prev_month['31-60 days'] * 100) if prev_month['31-60 days'] > 0 else 0:+.1f}%
        
        AGING MIGRATION TREND: {'Worsening' if current_month['90+ days'] > prev_month['90+ days'] else 'Improving' if current_month['90+ days'] < prev_month['90+ days'] else 'Stable'}
        """
        
        # Add top customers analysis
        top_customers = customer_metrics.nlargest(5, 'Total_Outstanding')
        if len(top_customers) > 0:
            data_info += f"""
            
        CUSTOMER CONCENTRATION RISK:
        - Top 5 Customers: {format_currency(top_customers['Total_Outstanding'].sum())} ({top_customers['Total_Outstanding'].sum() / total_ar * 100:.1f}% of total)
        1. {top_customers.iloc[0]['Customer_Name']}: {format_currency(top_customers.iloc[0]['Total_Outstanding'])} (DSO: {top_customers.iloc[0]['DSO']} days)
        2. {top_customers.iloc[1]['Customer_Name']}: {format_currency(top_customers.iloc[1]['Total_Outstanding'])} (DSO: {top_customers.iloc[1]['DSO']} days)
        3. {top_customers.iloc[2]['Customer_Name']}: {format_currency(top_customers.iloc[2]['Total_Outstanding'])} (DSO: {top_customers.iloc[2]['DSO']} days)
        """
        
        # Add payment pattern insights
        paid_df = df[df['Payment_Date'].notna()].copy()
        if len(paid_df) > 0:
            early_payment_rate = (len(paid_df[paid_df['Days_To_Pay'] <= 15]) / len(paid_df) * 100) if len(paid_df) > 0 else 0
            late_payment_rate = (len(paid_df[paid_df['Days_To_Pay'] > 45]) / len(paid_df) * 100) if len(paid_df) > 0 else 0
            
            data_info += f"""
            
        PAYMENT BEHAVIOR PATTERNS:
        - Early Payments (≤15 days): {early_payment_rate:.1f}% of total payments
        - Late Payments (>45 days): {late_payment_rate:.1f}% of total payments
        - Average Credit Utilization: {(paid_df['Days_To_Pay'].mean() / paid_df['Credit_Period'].mean() * 100) if paid_df['Credit_Period'].mean() > 0 else 0:.1f}% of allowed terms
        """
        
        # Ask AI for CXO-level insights WITH ACTUAL DATA
        ai_question = """
        Provide a CXO-level executive summary focusing on:
        
        1. DSO TREND ANALYSIS: Is DSO improving or deteriorating? What's driving the trend and what's the financial impact?
        2. AMOUNT AT RISK: How much is in delinquent accounts (>90 days)? What's the exact financial exposure and percentage of portfolio?
        3. AGING BUCKET MOVEMENT: Are receivables moving into older buckets? What specific amounts are migrating and what patterns are emerging?
        4. CASH FLOW IMPLICATIONS: What's the impact on working capital and liquidity in actual monetary terms?
        5. TOP 3 PRIORITIES: What immediate actions should leadership focus on to recover specific amounts?
        6. STRATEGIC RECOMMENDATIONS: What process improvements or policy changes are needed based on the actual data?
        
        IMPORTANT: Reference specific amounts and percentages from the data. For example:
        - "DSO is X days, which is Y% above target, tying up ₹Z in working capital"
        - "Delinquent accounts total ₹X, representing Y% of the portfolio"
        - "Aging migration shows ₹X moving from 30-60 days to 60-90 days bucket"
        
        Keep the analysis high-level, strategic, and focused on business impact with specific monetary values. Use bullet points for clarity.
        """
        
        ai_response = ask_ai(ai_question, data_info)
        
        st.markdown("### 📋 AI Executive Insights")
        st.markdown(f'<div class="recommendation-card">{ai_response}</div>', unsafe_allow_html=True)
def calculate_collection_rates_within_days(df, days):
    """Calculate percentage of invoices collected within X days from their due date"""
    
    # Get all paid invoices
    paid_invoices = df[df['Payment_Date'].notna()].copy()
    
    if len(paid_invoices) == 0:
        return 0
    
    # Calculate days taken to pay from due date (not invoice date)
    paid_invoices['Days_From_Due_Date'] = (paid_invoices['Payment_Date'] - paid_invoices['Due_Date']).dt.days
    
    # Count invoices paid within X days from due date
    if days == 90:  # For 90+ days, we want all invoices
        collected_within_days = paid_invoices.shape[0]
    else:
        collected_within_days = paid_invoices[paid_invoices['Days_From_Due_Date'] <= days].shape[0]
    
    # Total paid invoices for the rate calculation
    total_paid_invoices = paid_invoices.shape[0]
    
    if total_paid_invoices > 0:
        collection_rate = (collected_within_days / total_paid_invoices) * 100
    else:
        collection_rate = 0
    
    return round(collection_rate, 1)

def show_descriptive_analytics(df):
# === CUSTOMER WISE AGING BUCKET TABLE (NEW) ===
    st.markdown("## 2️⃣ 📊 Customer-wise Aging Bucket Analysis")
    
    # Filter only unpaid invoices for aging analysis
    unpaid_df = df[df['Payment_Date'].isna()].copy()
    
    if len(unpaid_df) > 0:
        # Create pivot table with customer-wise aging buckets
        aging_pivot = unpaid_df.pivot_table(
            index='Customer_Name',
            columns='Aging_Bucket',
            values='Amount_Outstanding',
            aggfunc='sum',
            fill_value=0
        )
        
        # Ensure all aging buckets are present (in correct order)
        bucket_order = ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']
        for bucket in bucket_order:
            if bucket not in aging_pivot.columns:
                aging_pivot[bucket] = 0
        
        # Reorder columns
        aging_pivot = aging_pivot[bucket_order]
        
        # Add total column
        aging_pivot['Total'] = aging_pivot.sum(axis=1)
        
        # Sort by total outstanding (descending)
        aging_pivot = aging_pivot.sort_values('Total', ascending=False)
        
        # Format display table
        display_pivot = aging_pivot.copy()
        for col in display_pivot.columns:
            display_pivot[col] = display_pivot[col].apply(lambda x: f"₹{x:,.0f}" if x > 0 else "₹0")
        
        # Display the customer-wise aging bucket table
        st.markdown("### Customer-wise Outstanding Amount by Aging Bucket")
        st.dataframe(
            display_pivot,
            use_container_width=True,
            height=600
        )
        
        # Download option
        csv_data = aging_pivot.copy()
        for col in csv_data.columns:
            if col != 'Total':
                csv_data[col] = csv_data[col].round(2)
        
        st.download_button(
            label="📥 Download Customer Aging Analysis",
            data=csv_data.to_csv(),
            file_name=f"customer_aging_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ All invoices are paid! No aging analysis needed.")
    
    st.markdown("---")

    
    # === CUSTOMER WISE DETAILED TABLE WITH CORRECTED METRICS ===
    st.markdown("## 2️⃣ 📊 Customer-wise Receivables Analysis (Corrected Metrics)")
    
    # Get customer summary table with corrected formulas
    customer_summary = get_customer_detailed_metrics(df)
    
    # Display the comprehensive table
    st.markdown("### Detailed Customer Performance Metrics (Last 6 Months)")
    
    # Create display table with corrected metrics
    display_columns = [
        'Customer_Name', 'Total_Outstanding', 'Total_Sales', 'DSO', 
        'Payment_Rate', 'Collection_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate',
        'Weighted_Avg_Credit_Period', 'Sales_Employee', 'City'
    ]
    
    display_table = customer_summary[display_columns].copy()
    
    # Create a copy for display with formatted values
    formatted_display = display_table.copy()
    formatted_display['Total_Outstanding'] = formatted_display['Total_Outstanding'].apply(lambda x: f"₹{x:,.0f}")
    formatted_display['Total_Sales'] = formatted_display['Total_Sales'].apply(lambda x: f"₹{x:,.0f}")
    formatted_display['DSO'] = formatted_display['DSO'].apply(lambda x: f"{x:.1f} days")
    formatted_display['Payment_Rate'] = formatted_display['Payment_Rate'].apply(lambda x: f"{x:.1f}%")
    formatted_display['Collection_Rate'] = formatted_display['Collection_Rate'].apply(lambda x: f"{x:.1f}%")
    formatted_display['Early_Payment_Rate'] = formatted_display['Early_Payment_Rate'].apply(lambda x: f"{x:.1f}%")
    formatted_display['Late_Payment_Rate'] = formatted_display['Late_Payment_Rate'].apply(lambda x: f"{x:.1f}%")
    formatted_display['Weighted_Avg_Credit_Period'] = formatted_display['Weighted_Avg_Credit_Period'].apply(lambda x: f"{x:.1f} days")
    
    # Display the formatted table
    st.dataframe(
        formatted_display,
        use_container_width=True,
        height=600
    )
    
    # Download option for the detailed table
    st.download_button(
        label="📥 Download Customer Analysis",
        data=customer_summary.to_csv(index=False),
        file_name=f"customer_receivables_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # === MONTHLY AGING TREND ANALYSIS ===
    st.markdown("## 3️⃣ 📈 Monthly Aging Trend Analysis")
    
    aging_trend = calculate_monthly_aging_trend(df)
    
    if len(aging_trend) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Stacked bar chart
            fig = go.Figure()
            
            aging_buckets = ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']
            colors = ['#10b981', '#3b82f6', '#f59e0b', '#f97316', '#ef4444']
            
            for i, bucket in enumerate(aging_buckets):
                fig.add_trace(go.Bar(
                    name=bucket,
                    x=aging_trend['Month_End'],
                    y=aging_trend[bucket],
                    marker_color=colors[i],
                    text=aging_trend[bucket].apply(lambda x: f"₹{x/1000:.0f}K" if x > 0 else ""),
                    textposition='auto'
                ))
            
            fig.update_layout(
                title='Monthly Aging Bucket Trend',
                xaxis_title='Month End',
                yaxis_title='Amount Outstanding (₹)',
                barmode='stack',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Line chart showing trend of each bucket
            fig = go.Figure()
            
            for i, bucket in enumerate(aging_buckets):
                fig.add_trace(go.Scatter(
                    name=bucket,
                    x=aging_trend['Month_End'],
                    y=aging_trend[bucket],
                    mode='lines+markers',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title='Aging Bucket Trend Lines',
                xaxis_title='Month End',
                yaxis_title='Amount Outstanding (₹)',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display the trend table
        st.markdown("#### 📋 Monthly Aging Summary Table")
        display_trend = aging_trend.copy()
        for col in ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days', 'Total']:
            display_trend[col] = display_trend[col].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(
            display_trend,
            use_container_width=True,
            height=300
        )
    else:
        st.info("Insufficient data for monthly aging trend analysis.")
    
    st.markdown("---")
    
    # === DELINQUENT ACCOUNTS WITH THRESHOLD ===
    st.markdown("## 4️⃣ 🚨 Delinquent Accounts Analysis")
    
    delinquent = get_delinquent_accounts(df, threshold_pct=2)
    
    if len(delinquent) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Delinquent accounts by amount
            fig = px.bar(
                delinquent.head(10),
                x='Customer_Name',
                y='Amount_Outstanding',
                title='Top 10 Delinquent Accounts by Amount',
                color='Days_Overdue',
                color_continuous_scale='Reds',
                labels={'Amount_Outstanding': 'Outstanding Amount (₹)', 'Customer_Name': 'Customer'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Delinquent accounts details
            st.markdown("### 📋 Delinquent Accounts Details")
            display_delinquent = delinquent[['Customer_Name', 'Amount_Outstanding', 'Days_Overdue', 'City', 'Sales_Employee']].copy()
            display_delinquent['Amount_Outstanding'] = display_delinquent['Amount_Outstanding'].apply(lambda x: f"₹{x:,.0f}")
            display_delinquent['Days_Overdue'] = display_delinquent['Days_Overdue'].apply(lambda x: f"{x:.0f} days")
            
            st.dataframe(
                display_delinquent,
                use_container_width=True,
                height=400
            )
    else:
        st.success("✅ No delinquent accounts meeting the threshold criteria (90+ days overdue and >2% of total receivables)!")
    
    st.markdown("---")
    
    # Section 5: Aging Bucket Distribution (Unpaid only)
    st.markdown("## 5️⃣ Aging Bucket Distribution (Unpaid Amounts Only)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unpaid_df = df[df['Payment_Date'].isna()].copy()
        
        if len(unpaid_df) > 0:
            aging_dist = unpaid_df.groupby('Aging_Bucket')['Amount_Outstanding'].sum().reset_index()
            fig = px.pie(
                aging_dist,
                values='Amount_Outstanding',
                names='Aging_Bucket',
                title='Aging Bucket Distribution (Unpaid Only)',
                color_discrete_sequence=px.colors.sequential.RdBu_r
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ All invoices are paid!")
    
    with col2:
        # Customer concentration in aging buckets
        if len(unpaid_df) > 0:
            # Top customers in each aging bucket
            bucket_customers = {}
            for bucket in ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']:
                bucket_data = unpaid_df[unpaid_df['Aging_Bucket'] == bucket]
                if len(bucket_data) > 0:
                    top_customer = bucket_data.groupby('Customer_Name')['Amount_Outstanding'].sum().nlargest(1)
                    if len(top_customer) > 0:
                        bucket_customers[bucket] = {
                            'customer': top_customer.index[0],
                            'amount': top_customer.iloc[0]
                        }
            
            st.markdown("### 👥 Top Customers in Each Aging Bucket")
            for bucket, data in bucket_customers.items():
                st.markdown(f"**{bucket}**: {data['customer']} - {format_currency(data['amount'])}")
            
            # Drill down option
            with st.expander("🔍 Drill down into specific aging bucket"):
                selected_bucket = st.selectbox(
                    "Select Aging Bucket to analyze:",
                    ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']
                )
                
                bucket_data = unpaid_df[unpaid_df['Aging_Bucket'] == selected_bucket]
                if len(bucket_data) > 0:
                    customer_bucket = bucket_data.groupby('Customer_Name').agg({
                        'Amount_Outstanding': 'sum',
                        'Days_Overdue': 'max',
                        'Sales_Employee': 'first'
                    }).reset_index().sort_values('Amount_Outstanding', ascending=False)
                    
                    st.dataframe(
                        customer_bucket.head(10).style.format({
                            'Amount_Outstanding': '₹{:,.0f}',
                            'Days_Overdue': '{:.0f} days'
                        }),
                        use_container_width=True
                    )
    
    st.markdown("---")
    
    # Section 6: Payment Behavior Analysis with corrected days to pay
    st.markdown("## 6️⃣ Payment Behavior Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Recent average days to pay (last 2 months)
        recent_data = df[
            (df['Payment_Date'].notna()) & 
            (df['Payment_Date'] >= datetime.now() - timedelta(days=60))
        ]
        
        if len(recent_data) > 0:
            avg_dtp_customer = recent_data.groupby('Customer_Name')['Days_To_Pay'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=avg_dtp_customer.values,
                y=avg_dtp_customer.index,
                orientation='h',
                title='Top 10 Customers by Average Days to Pay (Last 2 Months)',
                labels={'x': 'Average Days to Pay', 'y': 'Customer'},
                color=avg_dtp_customer.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No payment data available for last 2 months")
    
    with col2:
        # Distribution of payment behavior (last 2 months)
        payment_dist = recent_data['Days_To_Pay'].dropna() if len(recent_data) > 0 else pd.Series()
        
        if len(payment_dist) > 0:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=payment_dist,
                nbinsx=20,
                marker_color='#3b82f6',
                name='Distribution'
            ))
            fig.update_layout(
                title='Distribution of Days to Pay (Last 2 Months)',
                xaxis_title='Days to Pay',
                yaxis_title='Frequency',
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No payment data available for distribution analysis")
    
    st.markdown("---")
    
    # Section 7: Customer Concentration
    st.markdown("## 7️⃣ Customer Concentration Analysis")
    
    col1, col2 = st.columns([1.618, 1])
    
    with col1:
        customer_conc = df.groupby('Customer_Name')['Amount_Outstanding'].sum().sort_values(ascending=False).head(10)
        total = customer_conc.sum()
        customer_conc_pct = (customer_conc / total * 100).round(1)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=customer_conc.index,
            x=customer_conc.values,
            orientation='h',
            marker=dict(
                color=customer_conc.values,
                colorscale='Blues',
                showscale=True
            ),
            text=[f'{format_currency(val)} ({pct}%)' for val, pct in zip(customer_conc.values, customer_conc_pct)],
            textposition='outside'
        ))
        fig.update_layout(
            title='Top 10 Customers by Outstanding Amount',
            xaxis_title='Amount Outstanding (₹)',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        segment_dist = df.groupby('Customer_Segment')['Customer_Code'].nunique().reset_index()
        segment_dist.columns = ['Segment', 'Count']
        
        fig = px.pie(
            segment_dist,
            values='Count',
            names='Segment',
            title='Customer Segmentation',
            color='Segment',
            color_discrete_map={
                'Prompt Payers': '#10b981',
                'Standard Payers': '#3b82f6',
                'Slow Payers': '#f59e0b',
                'Delinquent': '#ef4444',
                'Unknown': '#9ca3af'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Section 8: Sales Person Wise Collection Efficiency
    st.markdown("## 8️⃣ Sales Team Performance")
    
    sales_performance = df.groupby('Sales_Employee').agg({
        'Sales_Amount': 'sum',
        'Amount_Outstanding': 'sum'
    }).reset_index()
    sales_performance['Collection_Rate'] = ((sales_performance['Sales_Amount'] - sales_performance['Amount_Outstanding']) / sales_performance['Sales_Amount'] * 100).round(1)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Sales Amount',
        x=sales_performance['Sales_Employee'],
        y=sales_performance['Sales_Amount'],
        marker_color='#3b82f6'
    ))
    fig.add_trace(go.Bar(
        name='Outstanding',
        x=sales_performance['Sales_Employee'],
        y=sales_performance['Amount_Outstanding'],
        marker_color='#ef4444'
    ))
    fig.add_trace(go.Scatter(
        name='Collection Rate %',
        x=sales_performance['Sales_Employee'],
        y=sales_performance['Collection_Rate'],
        mode='lines+markers+text',
        marker=dict(size=12, color='#10b981'),
        text=sales_performance['Collection_Rate'].astype(str) + '%',
        textposition='top center',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Sales Team Performance: Sales vs Outstanding vs Collection Rate',
        barmode='group',
        height=450,
        yaxis=dict(title='Amount (₹)'),
        yaxis2=dict(title='Collection Rate (%)', overlaying='y', side='right'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Section 9: Geography Analysis
    st.markdown("## 9️⃣ Geographic Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        geo_analysis = df.groupby('City').agg({
            'Amount_Outstanding': 'sum',
            'Sales_Amount': 'sum'
        }).reset_index()
        
        # Calculate DSO by city
        city_dso = []
        for city in geo_analysis['City'].unique():
            city_df = df[df['City'] == city]
            city_dso.append(calculate_dso(city_df))
        
        geo_analysis['DSO'] = city_dso
        
        fig = px.bar(
            geo_analysis.sort_values('Amount_Outstanding', ascending=False),
            x='City',
            y='Amount_Outstanding',
            title='Outstanding Amount by City',
            color='DSO',
            color_continuous_scale='RdYlGn_r',
            text='Amount_Outstanding'
        )
        fig.update_traces(texttemplate='₹%{text:.2s}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        office_dso = []
        for office in df['Sales_Office'].unique():
            office_df = df[df['Sales_Office'] == office]
            office_dso.append({
                'Sales_Office': office,
                'DSO': calculate_dso(office_df)
            })
        
        office_dso_df = pd.DataFrame(office_dso)
        
        fig = px.bar(
            office_dso_df.sort_values('DSO'),
            x='Sales_Office',
            y='DSO',
            title='DSO by Sales Office',
            color='DSO',
            color_continuous_scale='RdYlGn_r',
            text='DSO'
        )
        fig.update_traces(texttemplate='%{text:.1f} days', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Section 10: Warning Sign Alerts with corrected metrics
    st.markdown("## 🔟 ⚠️ Warning Sign Alerts")
    
    # Calculate aging migration from monthly trend
    aging_trend = calculate_monthly_aging_trend(df)
    
    if len(aging_trend) >= 2:
        current_month = aging_trend.iloc[-1]
        prev_month = aging_trend.iloc[-2]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Aging migration alert
            current_31_60 = current_month['31-60 days']
            current_61_90 = current_month['61-90 days']
            prev_31_60 = prev_month['31-60 days']
            prev_61_90 = prev_month['61-90 days']
            
            if current_31_60 > prev_31_60 * 1.2 or current_61_90 > prev_61_90 * 1.2:
                st.markdown('<div class="alert-box">', unsafe_allow_html=True)
                st.markdown("### 🔴 Aging Migration Alert")
                st.markdown(f"**31-60 days bucket**: Increased from ₹{prev_31_60:,.0f} to ₹{current_31_60:,.0f}")
                st.markdown(f"**61-90 days bucket**: Increased from ₹{prev_61_90:,.0f} to ₹{current_61_90:,.0f}")
                st.markdown("**Action Required**: Investigate aging migration patterns")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("### ✅ Aging Migration Status")
                st.markdown("**Aging buckets**: Stable or improving")
                st.markdown("**Status**: Healthy aging profile")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # DSO Spike detection
            monthly_dso = []
            for idx, row in aging_trend.iterrows():
                month = datetime.strptime(row['Month_End'], '%Y-%m-%d')
                # Calculate DSO for each month
                month_df = df[df['Invoice_Date'] <= month]
                month_dso = calculate_dso(month_df, month=row['Month_End'][:7])
                monthly_dso.append(month_dso)
            
            if len(monthly_dso) >= 2:
                dso_change = monthly_dso[-1] - monthly_dso[-2]
                if abs(dso_change) > 3:
                    st.markdown('<div class="alert-box">', unsafe_allow_html=True)
                    st.markdown("### 🔴 DSO Spike Alert")
                    st.markdown(f"**DSO Change**: {dso_change:+.1f} days")
                    st.markdown(f"**Previous Month**: {monthly_dso[-2]:.1f} days")
                    st.markdown(f"**Current Month**: {monthly_dso[-1]:.1f} days")
                    st.markdown("**Action Required**: Investigate root causes")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### ✅ DSO Status")
                    st.markdown(f"**DSO Change**: {dso_change:+.1f} days (Normal)")
                    st.markdown("**Status**: DSO is stable")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === AI ANALYSIS BUTTON ===
    st.subheader("🤖 Deep Dive Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.selectbox("What to analyze?", 
                                   ["Payment Trends", "Customer Behavior", "Risk Areas", "Collection Strategy"])
    
    with col2:
        if st.button("🚀 Run AI Analysis", key="ai_descriptive"):
            # ✅ Send only high-level summaries
            total_ar = df[df['Payment_Date'].isna()]['Amount_Outstanding'].sum()
            unpaid_df = df[df['Payment_Date'].isna()].copy()
            
            data_info = f"""
            Analysis Focus: {analysis_type}
            
            Key Metrics:
            - Total Customers: {df['Customer_Code'].nunique()}
            - Total Outstanding: {format_currency(total_ar)}
            - DSO: {calculate_dso(df)} days (based on last month sales)
            - Collection Effectiveness: {calculate_cei(df)[0]}% (invoices collected within 5 days)
            
            Risk Profile (Unpaid amounts only):
            - High Risk (>90 days): {(unpaid_df[unpaid_df['Aging_Bucket'] == '90+ days']['Amount_Outstanding'].sum() / total_ar * 100):.1f}%
            - Medium Risk (31-90 days): {(unpaid_df[unpaid_df['Aging_Bucket'].isin(['31-60 days', '61-90 days'])]['Amount_Outstanding'].sum() / total_ar * 100):.1f}%
            - Low Risk (0-30 days): {(unpaid_df[unpaid_df['Aging_Bucket'].isin(['Current', '0-30 days'])]['Amount_Outstanding'].sum() / total_ar * 100):.1f}%
            
            Payment Performance:
            - 30-Day Collection Rate: {calculate_collection_rates(df, days=30)}%
            - Payment Within Terms: {calculate_payment_within_terms(df)}%
            - Recent Days to Pay (2M): {calculate_recent_days_to_pay(df, months=2)[0]:.0f} days
            
            Biggest Challenge: {(unpaid_df.groupby('Aging_Bucket')['Amount_Outstanding'].sum() / total_ar * 100).idxmax()} bucket contains largest amount
            """
            
            ai_question = f"Analyze {analysis_type} using the corrected metrics and provide 2-3 specific, actionable recommendations to improve collections."
            ai_response = ask_ai(ai_question, data_info)
            
            st.markdown("### 📊 AI Analysis Results")
            st.markdown(f'<div class="recommendation-card">{ai_response}</div>', unsafe_allow_html=True)
            
    # Add new drill-down tabs
    st.markdown("---")
    
    # Add tabs for different drill-down views
    tab1, tab2, tab3 = st.tabs([
        "🔍 Aging Drill-Down", 
        "🌍 Geographic Drill-Down", 
        "👨‍💼 Sales Team Drill-Down"
    ])
    
    with tab1:
        show_interactive_aging_drilldown(df)
    
    with tab2:
        show_geographic_drilldown(df)
    
    with tab3:
        show_sales_team_drilldown(df)

def show_predictive_analysis_ml(df):
    """Display predictive analysis with ML section - ENHANCED with survival analysis and Bayesian methods"""
    st.title("🔮 Predictive Analysis with ML")
    st.markdown("### Enhanced with Survival Analysis, Bayesian Inference & Time-to-Payment Predictions")
    
    # Add explanation of new features
    with st.expander("🎯 What's New in Enhanced Predictive Analysis"):
        st.markdown("""
        **New Capabilities Added:**
        
        1. **Survival Analysis**: Predicts **when** payments will occur (not just **if**)
        2. **Time-to-Payment Regression**: Estimates days until payment for each invoice
        3. **Bayesian Hierarchical Models**: Better predictions for new/sparse-data customers
        4. **Partial Payment Predictions**: Estimates payment percentages (full/partial)
        5. **Temporal Anomaly Detection**: Identifies sudden changes in payment behavior
        
        **Business Impact:**
        - More accurate cash flow forecasting
        - Better collection resource scheduling
        - Reduced uncertainty in recovery predictions
        """)
    
    # Feature 1: Enhanced Customer-wise Aging Class Probability Model
    st.markdown("---")
    st.markdown("## 1️⃣ 🎯 Enhanced Customer Payment Probability with Bayesian Inference")
    st.markdown("*Predict payment probability with uncertainty quantification*")
    
    with st.spinner("Running Bayesian hierarchical models..."):
        customer_aging_probs = calculate_customer_aging_probabilities_enhanced(df)
    
    if customer_aging_probs is not None:
        # Calculate expected payments with uncertainty bounds
        customer_expected_payments = calculate_expected_payments_with_uncertainty(df, customer_aging_probs)
        
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            # Top expected payments with confidence intervals
            st.markdown("### 📊 Top 15 Customers by Expected Payment (±95% CI)")
            
            top_expected = customer_expected_payments.nlargest(15, 'Median_Expected_Payment')[
                ['Customer_Name', 'Median_Expected_Payment', 'CI_Lower', 'CI_Upper', 
                 'Total_Outstanding', 'Expected_Collection_Rate', 'Risk_Score', 'Data_Quality_Score']
            ].copy()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Expected Payment Range',
                y=top_expected['Customer_Name'],
                x=top_expected['CI_Upper'] - top_expected['CI_Lower'],
                base=top_expected['CI_Lower'],
                marker_color='rgba(59, 130, 246, 0.3)',
                orientation='h',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                name='Median Expected',
                y=top_expected['Customer_Name'],
                x=top_expected['Median_Expected_Payment'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='#ef4444',
                    symbol='diamond'
                ),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=top_expected['CI_Upper'] - top_expected['Median_Expected_Payment'],
                    arrayminus=top_expected['Median_Expected_Payment'] - top_expected['CI_Lower'],
                    color='#000000',
                    thickness=1.5,
                    width=3
                )
            ))
            
            fig.update_layout(
                title='Expected Payments with Confidence Intervals',
                xaxis_title='Expected Payment (₹)',
                yaxis_title='Customer',
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Expected Payments Summary")
            
            # Summary metrics with uncertainty
            total_outstanding = customer_expected_payments['Total_Outstanding'].sum()
            median_expected = customer_expected_payments['Median_Expected_Payment'].sum()
            ci_lower_total = customer_expected_payments['CI_Lower'].sum()
            ci_upper_total = customer_expected_payments['CI_Upper'].sum()
            
            avg_collection_rate = (median_expected / total_outstanding * 100) if total_outstanding > 0 else 0
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(f"**Total Outstanding**: {format_currency(total_outstanding)}")
            st.markdown(f"**Median Expected**: {format_currency(median_expected)}")
            st.markdown(f"**95% CI Range**: {format_currency(ci_lower_total)} - {format_currency(ci_upper_total)}")
            st.markdown(f"**Expected Collection Rate**: {avg_collection_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Data quality distribution
            data_quality_counts = customer_expected_payments['Data_Quality_Score'].value_counts().sort_index()
            
            fig = px.bar(
                x=data_quality_counts.index,
                y=data_quality_counts.values,
                title='Customer Data Quality Distribution',
                labels={'x': 'Quality Score', 'y': 'Count'},
                color=data_quality_counts.values,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed customer analysis with new features
        st.markdown("### 🔍 Enhanced Customer Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📈 Payment Timeline", "🎲 Bayesian Probabilities", "📊 Historical Analysis"])
        
        selected_customer = st.selectbox(
            "Select Customer for Detailed Analysis:",
            customer_expected_payments['Customer_Name'].tolist(),
            key="enhanced_customer_select"
        )
        
        if selected_customer:
            with tab1:
                # Survival analysis visualization
                st.markdown("#### ⏳ Payment Timeline Analysis")
                survival_curve = calculate_survival_curve(df, selected_customer)
                
                if survival_curve is not None:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=survival_curve['days'],
                        y=survival_curve['survival_probability'],
                        mode='lines',
                        name='Survival Probability',
                        line=dict(color='#3b82f6', width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=survival_curve['days'],
                        y=survival_curve['ci_lower'],
                        mode='lines',
                        name='95% CI Lower',
                        line=dict(color='rgba(59, 130, 246, 0.3)', width=1),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=survival_curve['days'],
                        y=survival_curve['ci_upper'],
                        mode='lines',
                        name='95% CI Upper',
                        fill='tonexty',
                        fillcolor='rgba(59, 130, 246, 0.1)',
                        line=dict(color='rgba(59, 130, 246, 0.3)', width=1),
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_customer} - Payment Survival Curve',
                        xaxis_title='Days from Due Date',
                        yaxis_title='Probability of Not Paying',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Key survival metrics
                    median_survival = survival_curve[survival_curve['survival_probability'] <= 0.5]['days'].min()
                    if pd.isna(median_survival):
                        median_survival = ">365 days"
                    
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    with col_metrics1:
                        st.metric("Median Payment Time", f"{median_survival} days")
                    with col_metrics2:
                        prob_30d = survival_curve[survival_curve['days'] == 30]['survival_probability'].iloc[0] if 30 in survival_curve['days'].values else 1.0
                        st.metric("30-Day Payment Prob", f"{(1-prob_30d)*100:.1f}%")
                    with col_metrics3:
                        prob_90d = survival_curve[survival_curve['days'] == 90]['survival_probability'].iloc[0] if 90 in survival_curve['days'].values else 1.0
                        st.metric("90-Day Payment Prob", f"{(1-prob_90d)*100:.1f}%")
            
            with tab2:
                # Bayesian probability distributions
                st.markdown("#### 🎲 Bayesian Probability Distributions by Aging Class")
                
                customer_data = customer_expected_payments[
                    customer_expected_payments['Customer_Name'] == selected_customer
                ].iloc[0]
                
                aging_classes = ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']
                probs_median = []
                probs_lower = []
                probs_upper = []
                
                for cls in aging_classes:
                    prob_col = f"Prob_{cls.replace(' ', '_').replace('-', '_')}_median"
                    lower_col = f"Prob_{cls.replace(' ', '_').replace('-', '_')}_lower"
                    upper_col = f"Prob_{cls.replace(' ', '_').replace('-', '_')}_upper"
                    
                    probs_median.append(customer_data[prob_col] if prob_col in customer_data else 0)
                    probs_lower.append(customer_data[lower_col] if lower_col in customer_data else 0)
                    probs_upper.append(customer_data[upper_col] if upper_col in customer_data else 0)
                
                prob_df = pd.DataFrame({
                    'Aging Class': aging_classes,
                    'Median Probability': probs_median,
                    'CI Lower': probs_lower,
                    'CI Upper': probs_upper
                })
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Probability Range',
                    x=aging_classes,
                    y=probs_upper,
                    base=probs_lower,
                    marker_color='rgba(59, 130, 246, 0.3)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    name='Median Probability',
                    x=aging_classes,
                    y=probs_median,
                    mode='markers+lines',
                    marker=dict(size=10, color='#ef4444'),
                    line=dict(color='#ef4444', width=2)
                ))
                
                fig.update_layout(
                    title=f'{selected_customer} - Payment Probability by Aging Class (±95% CI)',
                    xaxis_title='Aging Class',
                    yaxis_title='Payment Probability',
                    yaxis_range=[0, 1],
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Historical payment pattern with trend analysis
                st.markdown("#### 📊 Historical Payment Pattern Analysis")
                
                cust_df = df[df['Customer_Name'] == selected_customer].copy()
                
                if len(cust_df) > 0:
                    # Payment trend over time
                    paid_invoices = cust_df[cust_df['Payment_Date'].notna()].copy()
                    
                    if len(paid_invoices) >= 3:
                        paid_invoices['Payment_YearMonth'] = paid_invoices['Payment_Date'].dt.to_period('M')
                        monthly_trend = paid_invoices.groupby('Payment_YearMonth').agg({
                            'Days_To_Pay': 'mean',
                            'Sales_Amount': 'sum',
                            'Payment_Date': 'count'
                        }).reset_index()
                        
                        monthly_trend['Payment_YearMonth'] = monthly_trend['Payment_YearMonth'].dt.to_timestamp()
                        
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Average Days to Pay Trend', 'Payment Amount Trend'),
                            vertical_spacing=0.15
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=monthly_trend['Payment_YearMonth'],
                                y=monthly_trend['Days_To_Pay'],
                                mode='lines+markers',
                                name='Days to Pay',
                                line=dict(color='#3b82f6', width=2)
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(
                                x=monthly_trend['Payment_YearMonth'],
                                y=monthly_trend['Sales_Amount'],
                                name='Payment Amount',
                                marker_color='#10b981'
                            ),
                            row=2, col=1
                        )
                        
                        fig.update_layout(height=500, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient historical data for trend analysis")
    
    # Feature 2: Time-to-Payment Prediction with Survival Analysis
    st.markdown("---")
    st.markdown("## 2️⃣ ⏳ Time-to-Payment Predictions (Survival Analysis)")
    st.markdown("*Predict when payments will occur, not just if*")
    
    with st.spinner("Running survival analysis models..."):
        time_to_payment_predictions = predict_time_to_payment(df)
    
    if time_to_payment_predictions is not None:
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            # Distribution of predicted payment times
            st.markdown("### 📅 Distribution of Predicted Payment Times")
            
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=time_to_payment_predictions['Predicted_Days_To_Pay'],
                name='Distribution',
                nbinsx=30,
                marker_color='#3b82f6',
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Predicted Days to Payment Distribution',
                xaxis_title='Predicted Days to Payment',
                yaxis_title='Frequency',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Payment Timing Summary")
            
            median_days = time_to_payment_predictions['Predicted_Days_To_Pay'].median()
            mean_days = time_to_payment_predictions['Predicted_Days_To_Pay'].mean()
            std_days = time_to_payment_predictions['Predicted_Days_To_Pay'].std()
            
            st.metric("Median Payment Time", f"{median_days:.0f} days")
            st.metric("Mean Payment Time", f"{mean_days:.0f} days")
            st.metric("Variability (Std Dev)", f"{std_days:.0f} days")
            
            # Payment timing categories
            early = (time_to_payment_predictions['Predicted_Days_To_Pay'] <= 15).sum()
            on_time = ((time_to_payment_predictions['Predicted_Days_To_Pay'] > 15) & 
                      (time_to_payment_predictions['Predicted_Days_To_Pay'] <= 45)).sum()
            late = (time_to_payment_predictions['Predicted_Days_To_Pay'] > 45).sum()
            total = len(time_to_payment_predictions)
            
            timing_data = pd.DataFrame({
                'Category': ['Early (≤15d)', 'On Time (16-45d)', 'Late (>45d)'],
                'Count': [early, on_time, late],
                'Percentage': [early/total*100, on_time/total*100, late/total*100]
            })
            
            fig = px.pie(
                timing_data,
                values='Count',
                names='Category',
                title='Payment Timing Categories',
                color='Category',
                color_discrete_map={
                    'Early (≤15d)': '#10b981',
                    'On Time (16-45d)': '#3b82f6',
                    'Late (>45d)': '#ef4444'
                }
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Invoice-level time predictions
        st.markdown("### 📋 Invoice-Level Payment Time Predictions")
        
        display_predictions = time_to_payment_predictions[
            ['Customer_Name', 'Invoice_Date', 'Due_Date', 'Amount_Outstanding',
             'Predicted_Days_To_Pay', 'Prediction_Confidence', 'Risk_Level']
        ].head(20).copy()
        
        display_predictions['Invoice_Date'] = display_predictions['Invoice_Date'].dt.strftime('%Y-%m-%d')
        display_predictions['Due_Date'] = display_predictions['Due_Date'].dt.strftime('%Y-%m-%d')
        
        # Color code by risk level
        def color_risk(row):
            if row['Risk_Level'] == 'High':
                return ['background-color: #fee2e2'] * len(row)
            elif row['Risk_Level'] == 'Medium':
                return ['background-color: #fef3c7'] * len(row)
            else:
                return ['background-color: #d1fae5'] * len(row)
        
        styled_df = display_predictions.style.apply(color_risk, axis=1)\
            .format({
                'Amount_Outstanding': '₹{:,.0f}',
                'Predicted_Days_To_Pay': '{:.0f} days',
                'Prediction_Confidence': '{:.1%}'
            })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Feature 3: Partial Payment Predictions
    st.markdown("---")
    st.markdown("## 3️⃣ 💰 Partial Payment Predictions")
    st.markdown("*Predict payment amounts, not just binary outcomes*")
    
    with st.spinner("Running partial payment regression models..."):
        partial_payment_predictions = predict_partial_payments(df)
    
    if partial_payment_predictions is not None:
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            # Distribution of predicted payment percentages
            st.markdown("### 📊 Predicted Payment Percentage Distribution")
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=partial_payment_predictions['Predicted_Payment_Pct'],
                nbinsx=20,
                marker_color='#10b981',
                opacity=0.7,
                name='Predicted %'
            ))
            
            fig.update_layout(
                title='Distribution of Predicted Payment Percentages',
                xaxis_title='Predicted Payment Percentage',
                yaxis_title='Count',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Partial Payment Summary")
            
            avg_pct = partial_payment_predictions['Predicted_Payment_Pct'].mean()
            median_pct = partial_payment_predictions['Predicted_Payment_Pct'].median()
            full_payment_pct = (partial_payment_predictions['Predicted_Payment_Pct'] >= 0.95).mean() * 100
            partial_payment_pct = ((partial_payment_predictions['Predicted_Payment_Pct'] > 0.5) & 
                                  (partial_payment_predictions['Predicted_Payment_Pct'] < 0.95)).mean() * 100
            low_payment_pct = (partial_payment_predictions['Predicted_Payment_Pct'] <= 0.5).mean() * 100
            
            st.metric("Avg Payment %", f"{avg_pct:.1%}")
            st.metric("Median Payment %", f"{median_pct:.1%}")
            
            st.markdown('<div class="alert-box">', unsafe_allow_html=True)
            st.markdown("**Payment Type Distribution:**")
            st.markdown(f"- Full Payment (≥95%): {full_payment_pct:.1f}%")
            st.markdown(f"- Partial Payment (50-95%): {partial_payment_pct:.1f}%")
            st.markdown(f"- Low Payment (≤50%): {low_payment_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature 4: Enhanced Payment Behavior Clustering with Temporal Features
    st.markdown("---")
    st.markdown("## 4️⃣ 👥 Enhanced Behavioral Clustering with Temporal Analysis")
    
    with st.spinner("Clustering with temporal features..."):
        enhanced_clusters = payment_behavior_clustering_enhanced(df)
    
    if enhanced_clusters is not None:
        # Temporal pattern analysis within clusters
        st.markdown("### 📈 Temporal Payment Patterns by Cluster")
        
        cluster_temporal_patterns = analyze_temporal_patterns(df, enhanced_clusters)
        
        if cluster_temporal_patterns is not None:
            # Heatmap of payment patterns by day of week and cluster
            fig = px.imshow(
                cluster_temporal_patterns.pivot_table(
                    values='Payment_Rate',
                    index='Cluster_Name',
                    columns='Day_Of_Week'
                ),
                title='Payment Patterns: Day of Week vs Cluster',
                color_continuous_scale='RdYlGn',
                labels=dict(x="Day of Week", y="Cluster", color="Payment Rate")
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Drill-down into clusters with temporal features
        selected_cluster = st.selectbox(
            "Select Cluster to Analyze Temporal Patterns:",
            enhanced_clusters['Cluster_Name'].unique(),
            key="temporal_cluster_select"
        )
        
        if selected_cluster:
            cluster_data = enhanced_clusters[enhanced_clusters['Cluster_Name'] == selected_cluster]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Payment trend over time for this cluster
                cluster_customers = cluster_data['Customer_Code'].unique()
                cluster_df = df[df['Customer_Code'].isin(cluster_customers)]
                
                paid_cluster = cluster_df[cluster_df['Payment_Date'].notna()].copy()
                if len(paid_cluster) > 0:
                    paid_cluster['Payment_Month'] = paid_cluster['Payment_Date'].dt.to_period('M')
                    monthly_trend = paid_cluster.groupby('Payment_Month').agg({
                        'Days_To_Pay': 'mean',
                        'Sales_Amount': 'sum'
                    }).reset_index()
                    
                    monthly_trend['Payment_Month'] = monthly_trend['Payment_Month'].dt.to_timestamp()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=monthly_trend['Payment_Month'],
                        y=monthly_trend['Days_To_Pay'],
                        mode='lines+markers',
                        name='Avg Days to Pay',
                        line=dict(color='#3b82f6', width=2)
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=monthly_trend['Payment_Month'],
                        y=monthly_trend['Sales_Amount'],
                        name='Payment Amount',
                        yaxis='y2',
                        marker_color='#10b981'
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_cluster} - Payment Trends Over Time',
                        xaxis_title='Month',
                        yaxis_title='Avg Days to Pay',
                        yaxis2=dict(
                            title='Payment Amount (₹)',
                            overlaying='y',
                            side='right'
                        ),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Seasonal patterns
                if len(paid_cluster) > 0:
                    paid_cluster['Month'] = paid_cluster['Payment_Date'].dt.month
                    seasonal_pattern = paid_cluster.groupby('Month').agg({
                        'Days_To_Pay': 'mean',
                        'Sales_Amount': 'sum'
                    }).reset_index()
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=seasonal_pattern['Days_To_Pay'],
                        theta=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        fill='toself',
                        name='Avg Days to Pay',
                        line_color='#3b82f6'
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_cluster} - Seasonal Payment Pattern',
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, seasonal_pattern['Days_To_Pay'].max() * 1.2]
                            )
                        ),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Feature 5: Temporal Anomaly Detection
    st.markdown("---")
    st.markdown("## 5️⃣ 🔍 Temporal Anomaly Detection")
    st.markdown("*Detect sudden changes in payment behavior over time*")
    
    with st.spinner("Detecting temporal anomalies..."):
        temporal_anomalies = detect_temporal_anomalies(df)
    
    if temporal_anomalies is not None and len(temporal_anomalies) > 0:
        col1, col2 = st.columns([1.618, 1])
        
        with col1:
            # Timeline of anomalies
            st.markdown("### 🚨 Detected Temporal Anomalies Timeline")
            
            # Convert to datetime if needed
            if 'Anomaly_Date' in temporal_anomalies.columns:
                temporal_anomalies['Date'] = pd.to_datetime(temporal_anomalies['Anomaly_Date'])
            else:
                temporal_anomalies['Date'] = datetime.now()  # fallback
            
            # Group by date for timeline
            anomalies_by_date = temporal_anomalies.groupby('Date').agg({
                'Customer_Code': 'count',
                'Anomaly_Score': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=anomalies_by_date['Date'],
                y=anomalies_by_date['Customer_Code'],
                mode='lines+markers',
                name='Anomaly Count',
                line=dict(color='#ef4444', width=2),
                marker=dict(size=anomalies_by_date['Anomaly_Score'] * 10)
            ))
            
            fig.update_layout(
                title='Temporal Anomalies Over Time',
                xaxis_title='Date',
                yaxis_title='Number of Anomalies',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Recent Anomalies")
            
            recent_anomalies = temporal_anomalies.head(10).copy()
            display_anomalies = recent_anomalies[
                ['Customer_Name', 'Anomaly_Type', 'Anomaly_Score', 'Change_Magnitude']
            ]
            
            st.dataframe(
                display_anomalies.style
                .background_gradient(cmap='Reds', subset=['Anomaly_Score'])
                .format({
                    'Anomaly_Score': '{:.2f}',
                    'Change_Magnitude': '{:.1%}'
                }),
                use_container_width=True,
                height=400
            )
    else:
        st.success("✅ No temporal anomalies detected!")
    # ============================================================================
    # INTEGRATED COLLECTION FORECAST TABLE
    # ============================================================================
    display_collection_forecast_table(df)

def calculate_customer_aging_probabilities_enhanced(df):
    """Enhanced version with Bayesian hierarchical modeling and uncertainty quantification"""
    try:
        # Get all customers
        customers = df['Customer_Code'].unique()
        customer_names = df.groupby('Customer_Code')['Customer_Name'].first()
        
        # Get industry/sector information if available
        # (In real implementation, this would come from CRM data)
        
        results = []
        
        for customer in customers:
            cust_df = df[df['Customer_Code'] == customer].copy()
            
            # Get customer's unpaid invoices
            unpaid_df = cust_df[cust_df['Payment_Date'].isna()].copy()
            
            if len(unpaid_df) == 0:
                continue
            
            paid_df = cust_df[cust_df['Payment_Date'].notna()].copy()
            
            customer_features = {
                'Customer_Code': customer,
                'Customer_Name': customer_names[customer],
                'Total_Invoices': len(cust_df),
                'Paid_Invoices': len(paid_df),
                'Unpaid_Invoices': len(unpaid_df),
                'Payment_Rate': len(paid_df) / len(cust_df) if len(cust_df) > 0 else 0,
                'Total_Outstanding': unpaid_df['Amount_Outstanding'].sum(),
                'Avg_Invoice_Size': cust_df['Sales_Amount'].mean() if len(cust_df) > 0 else 0,
                'Invoice_Size_Variation': cust_df['Sales_Amount'].std() if len(cust_df) > 0 else 0,
                'Customer_Tenure_Days': (cust_df['Invoice_Date'].max() - cust_df['Invoice_Date'].min()).days if len(cust_df) > 1 else 0
            }
            
            # Enhanced feature engineering
            if len(paid_df) > 0:
                # Payment timing features
                paid_df['Days_From_Due'] = (paid_df['Payment_Date'] - paid_df['Due_Date']).dt.days
                
                customer_features.update({
                    'Avg_Days_From_Due': paid_df['Days_From_Due'].mean(),
                    'Std_Days_From_Due': paid_df['Days_From_Due'].std(),
                    'Early_Payment_Rate': (paid_df['Days_From_Due'] <= 0).mean(),
                    'Late_Payment_Rate': (paid_df['Days_From_Due'] > 15).mean(),
                    'Payment_Consistency': 1 - (paid_df['Days_From_Due'].std() / max(1, abs(paid_df['Days_From_Due'].mean()))),
                })
            
            # Bayesian hierarchical probability calculation
            aging_buckets = ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']
            
            for bucket in aging_buckets:
                # Calculate empirical probability
                if len(paid_df) >= 3:  # Enough data for empirical calculation
                    if bucket == 'Current':
                        bucket_paid = paid_df[paid_df['Days_From_Due'] <= 0]
                    elif bucket == '0-30 days':
                        bucket_paid = paid_df[(paid_df['Days_From_Due'] > 0) & (paid_df['Days_From_Due'] <= 30)]
                    elif bucket == '31-60 days':
                        bucket_paid = paid_df[(paid_df['Days_From_Due'] > 30) & (paid_df['Days_From_Due'] <= 60)]
                    elif bucket == '61-90 days':
                        bucket_paid = paid_df[(paid_df['Days_From_Due'] > 60) & (paid_df['Days_From_Due'] <= 90)]
                    else:  # '90+ days'
                        bucket_paid = paid_df[paid_df['Days_From_Due'] > 90]
                    
                    empirical_prob = len(bucket_paid) / len(paid_df) if len(paid_df) > 0 else 0
                    
                    # Apply Bayesian shrinkage
                    # Prior: industry average (simplified - in reality would use hierarchical model)
                    prior_prob = {
                        'Current': 0.4,
                        '0-30 days': 0.3,
                        '31-60 days': 0.15,
                        '61-90 days': 0.1,
                        '90+ days': 0.05
                    }[bucket]
                    
                    # Effective sample size for shrinkage
                    effective_n = min(len(paid_df), 10)  # Cap influence of historical data
                    total_n = effective_n + 5  # Prior strength = 5
                    
                    # Bayesian posterior estimate
                    posterior_mean = (empirical_prob * effective_n + prior_prob * 5) / total_n
                    
                    # Calculate credible interval (simplified)
                    alpha = posterior_mean * total_n + 1
                    beta = (1 - posterior_mean) * total_n + 1
                    
                    # Using beta distribution approximation
                    lower_ci = stats.beta.ppf(0.025, alpha, beta)
                    upper_ci = stats.beta.ppf(0.975, alpha, beta)
                    
                else:  # Not enough data - use informative prior
                    # Use customer's overall payment rate to inform prior
                    overall_rate = customer_features['Payment_Rate']
                    
                    posterior_mean = {
                        'Current': min(0.95, overall_rate * 1.2),
                        '0-30 days': min(0.85, overall_rate * 1.1),
                        '31-60 days': min(0.7, overall_rate),
                        '61-90 days': min(0.5, overall_rate * 0.8),
                        '90+ days': min(0.3, overall_rate * 0.6)
                    }[bucket]
                    
                    # Wider credible intervals for uncertain predictions
                    lower_ci = max(0, posterior_mean - 0.3)
                    upper_ci = min(1, posterior_mean + 0.3)
                
                # Store results
                prob_col = f"Prob_{bucket.replace(' ', '_').replace('-', '_')}"
                customer_features[f"{prob_col}_median"] = posterior_mean
                customer_features[f"{prob_col}_lower"] = lower_ci
                customer_features[f"{prob_col}_upper"] = upper_ci
            
            # Calculate enhanced risk score with multiple dimensions
            risk_factors = []
            
            # 1. Historical payment risk
            if len(paid_df) > 0:
                late_payment_risk = customer_features.get('Late_Payment_Rate', 0)
                consistency_risk = 1 - customer_features.get('Payment_Consistency', 0.5)
                risk_factors.append(late_payment_risk * 0.4 + consistency_risk * 0.6)
            
            # 2. Amount risk
            amount_risk = min(1, customer_features['Total_Outstanding'] / max(1, customer_features['Avg_Invoice_Size'] * 3))
            risk_factors.append(amount_risk)
            
            # 3. Aging risk (based on predicted probabilities)
            aging_weights = {'Current': 0.1, '0-30 days': 0.2, '31-60 days': 0.3, 
                           '61-90 days': 0.6, '90+ days': 0.9}
            aging_risk = 0
            total_weight = 0
            
            for bucket, weight in aging_weights.items():
                prob_col = f"Prob_{bucket.replace(' ', '_').replace('-', '_')}_median"
                if prob_col in customer_features:
                    # Lower probability = higher risk
                    aging_risk += (1 - customer_features[prob_col]) * weight
                    total_weight += weight
            
            if total_weight > 0:
                aging_risk = aging_risk / total_weight
                risk_factors.append(aging_risk)
            
            # 4. Data quality risk (less data = higher uncertainty risk)
            data_quality_risk = max(0, 1 - min(1, len(paid_df) / 5))
            risk_factors.append(data_quality_risk * 0.5)  # Lower weight for data quality
            
            # Calculate overall risk score
            if risk_factors:
                customer_features['Risk_Score'] = sum(risk_factors) / len(risk_factors)
            else:
                customer_features['Risk_Score'] = 0.5  # Default neutral
            
            # Data quality score (1 = excellent, 0 = poor)
            customer_features['Data_Quality_Score'] = min(1, len(paid_df) / 10)
            
            results.append(customer_features)
        
        return pd.DataFrame(results)
    
    except Exception as e:
        st.error(f"Error in enhanced probability calculation: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def calculate_expected_payments_with_uncertainty(df, customer_probs_df):
    """Calculate expected payments with confidence intervals"""
    try:
        unpaid_df = df[df['Payment_Date'].isna()].copy()
        
        if len(unpaid_df) == 0:
            return None
        
        # Get outstanding by customer and aging bucket
        customer_outstanding = unpaid_df.groupby(['Customer_Code', 'Aging_Bucket'])['Amount_Outstanding'].sum().unstack(fill_value=0)
        customer_outstanding = customer_outstanding.reset_index()
        
        # Merge with probability data
        merged_df = pd.merge(customer_probs_df, customer_outstanding, on='Customer_Code', how='left')
        
        # Calculate expected payment for each aging bucket with uncertainty
        aging_buckets = ['Current', '0-30 days', '31-60 days', '61-90 days', '90+ days']
        
        for bucket in aging_buckets:
            prob_col_median = f"Prob_{bucket.replace(' ', '_').replace('-', '_')}_median"
            prob_col_lower = f"Prob_{bucket.replace(' ', '_').replace('-', '_')}_lower"
            prob_col_upper = f"Prob_{bucket.replace(' ', '_').replace('-', '_')}_upper"
            amt_col = bucket
            
            if (prob_col_median in merged_df.columns and prob_col_lower in merged_df.columns and 
                prob_col_upper in merged_df.columns and amt_col in merged_df.columns):
                merged_df[f'Expected_{bucket}_median'] = merged_df[prob_col_median] * merged_df[amt_col]
                merged_df[f'Expected_{bucket}_lower'] = merged_df[prob_col_lower] * merged_df[amt_col]
                merged_df[f'Expected_{bucket}_upper'] = merged_df[prob_col_upper] * merged_df[amt_col]
            else:
                merged_df[f'Expected_{bucket}_median'] = 0
                merged_df[f'Expected_{bucket}_lower'] = 0
                merged_df[f'Expected_{bucket}_upper'] = 0
        
        # Calculate total expected payment with CI
        expected_median_cols = [f'Expected_{bucket}_median' for bucket in aging_buckets]
        expected_lower_cols = [f'Expected_{bucket}_lower' for bucket in aging_buckets]
        expected_upper_cols = [f'Expected_{bucket}_upper' for bucket in aging_buckets]
        
        merged_df['Median_Expected_Payment'] = merged_df[expected_median_cols].sum(axis=1)
        merged_df['CI_Lower'] = merged_df[expected_lower_cols].sum(axis=1)
        merged_df['CI_Upper'] = merged_df[expected_upper_cols].sum(axis=1)
        
        # Calculate total outstanding
        outstanding_cols = aging_buckets
        merged_df['Total_Outstanding'] = merged_df[outstanding_cols].sum(axis=1, skipna=True)
        
        # Calculate expected collection rate with uncertainty
        merged_df['Expected_Collection_Rate'] = (merged_df['Median_Expected_Payment'] / merged_df['Total_Outstanding'] * 100).fillna(0)
        merged_df['Expected_Collection_Rate_Lower'] = (merged_df['CI_Lower'] / merged_df['Total_Outstanding'] * 100).fillna(0)
        merged_df['Expected_Collection_Rate_Upper'] = (merged_df['CI_Upper'] / merged_df['Total_Outstanding'] * 100).fillna(0)
        
        # Sort by expected payment (descending)
        merged_df = merged_df.sort_values('Median_Expected_Payment', ascending=False)
        
        # Clean up columns
        keep_cols = [
            'Customer_Code', 'Customer_Name', 'Total_Outstanding', 
            'Median_Expected_Payment', 'CI_Lower', 'CI_Upper',
            'Expected_Collection_Rate', 'Expected_Collection_Rate_Lower', 
            'Expected_Collection_Rate_Upper', 'Risk_Score', 'Data_Quality_Score',
            'Payment_Rate', 'Avg_Days_From_Due', 'Payment_Consistency'
        ]
        
        # Add probability columns
        for bucket in aging_buckets:
            keep_cols.extend([
                f"Prob_{bucket.replace(' ', '_').replace('-', '_')}_median",
                f"Prob_{bucket.replace(' ', '_').replace('-', '_')}_lower",
                f"Prob_{bucket.replace(' ', '_').replace('-', '_')}_upper"
            ])
        
        return merged_df[keep_cols]
    
    except Exception as e:
        st.error(f"Error calculating expected payments with uncertainty: {str(e)}")
        return None

def calculate_survival_curve(df, customer_name):
    """Calculate survival curve for payment times using Kaplan-Meier estimator"""
    try:
        cust_df = df[df['Customer_Name'] == customer_name].copy()
        
        if len(cust_df) == 0:
            return None
        
        # Get paid invoices
        paid_invoices = cust_df[cust_df['Payment_Date'].notna()].copy()
        
        if len(paid_invoices) < 3:
            # Not enough data for survival analysis
            # Create a simple empirical curve based on overall data
            all_paid = df[df['Payment_Date'].notna()].copy()
            if len(all_paid) >= 10:
                all_paid['Days_From_Due'] = (all_paid['Payment_Date'] - all_paid['Due_Date']).dt.days
                
                # Kaplan-Meier estimation (simplified)
                times = sorted(all_paid['Days_From_Due'].unique())
                survival_probs = []
                
                n = len(all_paid)
                for t in times:
                    at_risk = (all_paid['Days_From_Due'] >= t).sum()
                    events = (all_paid['Days_From_Due'] == t).sum()
                    if at_risk > 0:
                        survival_probs.append((at_risk - events) / at_risk)
                    else:
                        survival_probs.append(1.0)
                
                # Cumulative product for survival function
                survival = [1.0]
                for prob in survival_probs:
                    survival.append(survival[-1] * prob)
                survival = survival[1:]  # Remove initial 1.0
                
                # Create survival curve
                survival_df = pd.DataFrame({
                    'days': times[:len(survival)],
                    'survival_probability': survival,
                    'ci_lower': [max(0, p - 0.1) for p in survival],  # Simplified CI
                    'ci_upper': [min(1, p + 0.1) for p in survival]
                })
            else:
                # Fallback to theoretical curve
                days = list(range(0, 366, 7))
                survival_prob = [np.exp(-d/90) for d in days]  # Exponential decay
                survival_df = pd.DataFrame({
                    'days': days,
                    'survival_probability': survival_prob,
                    'ci_lower': [max(0, p - 0.15) for p in survival_prob],
                    'ci_upper': [min(1, p + 0.15) for p in survival_prob]
                })
        else:
            # Enough data for customer-specific survival analysis
            paid_invoices['Days_From_Due'] = (paid_invoices['Payment_Date'] - paid_invoices['Due_Date']).dt.days
            
            # Kaplan-Meier estimation
            times = sorted(paid_invoices['Days_From_Due'].unique())
            survival_probs = []
            
            n = len(paid_invoices)
            for t in times:
                at_risk = (paid_invoices['Days_From_Due'] >= t).sum()
                events = (paid_invoices['Days_From_Due'] == t).sum()
                if at_risk > 0:
                    survival_probs.append((at_risk - events) / at_risk)
                else:
                    survival_probs.append(1.0)
            
            # Cumulative product for survival function
            survival = [1.0]
            for prob in survival_probs:
                survival.append(survival[-1] * prob)
            survival = survival[1:]  # Remove initial 1.0
            
            # Greenwood's formula for confidence intervals (simplified)
            variances = []
            for i, t in enumerate(times[:len(survival)]):
                at_risk = (paid_invoices['Days_From_Due'] >= t).sum()
                events = (paid_invoices['Days_From_Due'] == t).sum()
                if at_risk > 0 and at_risk != events:
                    variances.append(events / (at_risk * (at_risk - events)))
                else:
                    variances.append(0)
            
            # Calculate standard errors
            se = []
            for i in range(len(survival)):
                if i == 0:
                    se.append(np.sqrt(variances[i]) if variances[i] > 0 else 0.1)
                else:
                    cum_var = sum(variances[:i+1])
                    se.append(survival[i] * np.sqrt(cum_var))
            
            # 95% confidence intervals
            z = 1.96  # 95% CI
            survival_df = pd.DataFrame({
                'days': times[:len(survival)],
                'survival_probability': survival,
                'ci_lower': [max(0, s - z * se[i]) for i, s in enumerate(survival)],
                'ci_upper': [min(1, s + z * se[i]) for i, s in enumerate(survival)]
            })
        
        # Interpolate to get daily values
        max_days = min(365, survival_df['days'].max())
        all_days = list(range(0, int(max_days) + 1))
        
        interpolated_prob = np.interp(all_days, survival_df['days'], survival_df['survival_probability'])
        interpolated_lower = np.interp(all_days, survival_df['days'], survival_df['ci_lower'])
        interpolated_upper = np.interp(all_days, survival_df['days'], survival_df['ci_upper'])
        
        final_df = pd.DataFrame({
            'days': all_days,
            'survival_probability': interpolated_prob,
            'ci_lower': interpolated_lower,
            'ci_upper': interpolated_upper
        })
        
        return final_df
    
    except Exception as e:
        st.error(f"Error calculating survival curve: {str(e)}")
        return None

def predict_time_to_payment(df):
    """Predict time to payment for unpaid invoices using survival analysis and regression"""
    try:
        unpaid_df = df[df['Payment_Date'].isna()].copy()
        
        if len(unpaid_df) == 0:
            return None
        
        # Use historical paid invoices to build prediction model
        paid_df = df[df['Payment_Date'].notna()].copy()
        
        if len(paid_df) < 10:
            # Not enough data for ML model, use simple heuristic
            unpaid_df['Predicted_Days_To_Pay'] = unpaid_df.apply(
                lambda x: min(90, x['Days_Overdue'] + 30) if pd.notna(x['Days_Overdue']) else 45,
                axis=1
            )
            unpaid_df['Prediction_Confidence'] = 0.5
            unpaid_df['Risk_Level'] = 'Medium'
            
        else:
            # Prepare features for prediction
            paid_df['Days_To_Pay_Actual'] = (paid_df['Payment_Date'] - paid_df['Invoice_Date']).dt.days
            
            # Calculate customer-level features
            customer_features = paid_df.groupby('Customer_Code').agg({
                'Days_To_Pay_Actual': ['mean', 'std', 'count'],
                'Sales_Amount': 'mean'
            }).reset_index()
            
            customer_features.columns = ['Customer_Code', 'Avg_Days_To_Pay', 'Std_Days_To_Pay', 
                                        'Paid_Count', 'Avg_Invoice_Size']
            
            # Merge with unpaid invoices
            predictions = unpaid_df.merge(customer_features, on='Customer_Code', how='left')
            
            # Fill missing values safely
            avg_days_median = predictions['Avg_Days_To_Pay'].median()
            std_days_median = predictions['Std_Days_To_Pay'].median()
            
            predictions['Avg_Days_To_Pay'] = predictions['Avg_Days_To_Pay'].fillna(avg_days_median if pd.notna(avg_days_median) else 45)
            predictions['Std_Days_To_Pay'] = predictions['Std_Days_To_Pay'].fillna(std_days_median if pd.notna(std_days_median) else 15)
            
            # Use Sales_Amount if Avg_Invoice_Size is NaN
            predictions['Avg_Invoice_Size'] = predictions.apply(
                lambda row: row['Sales_Amount'] if pd.isna(row['Avg_Invoice_Size']) else row['Avg_Invoice_Size'],
                axis=1
            )
            
            # Base prediction: customer's average days to pay
            base_prediction = predictions['Avg_Days_To_Pay']
            
            # Adjust for current overdue status (handle NaN)
            predictions['Days_Overdue'] = predictions['Days_Overdue'].fillna(0)
            overdue_adjustment = predictions['Days_Overdue'].apply(
                lambda x: max(0, x * 0.5)  # If already overdue, expect longer
            )
            
            # Adjust for invoice size - FIXED: prevent division by zero or log of zero
            predictions['Avg_Invoice_Size'] = predictions['Avg_Invoice_Size'].replace(0, 1)  # Avoid division by zero
            predictions['Sales_Amount'] = predictions['Sales_Amount'].replace(0, 1)  # Avoid log(0)
            
            size_adjustment = predictions.apply(
                lambda row: np.log1p(max(0, row['Sales_Amount'] / row['Avg_Invoice_Size'])) * 5, 
                axis=1
            )
            
            # Adjust for credit period
            predictions['Credit_Period'] = predictions['Credit_Period'].fillna(30)
            credit_adjustment = predictions['Credit_Period'].apply(lambda x: x * 0.3)
            
            # Combine predictions
            predictions['Predicted_Days_To_Pay'] = (
                base_prediction + overdue_adjustment + size_adjustment + credit_adjustment
            ).clip(lower=0, upper=365)  # Clip to reasonable range
            
            # Calculate prediction confidence - FIXED: prevent division by zero
            predictions['Std_Days_To_Pay'] = predictions['Std_Days_To_Pay'].replace(0, 15)  # Avoid division by zero
            
            predictions['Prediction_Confidence'] = predictions.apply(
                lambda row: min(0.95, 1 - (row['Std_Days_To_Pay'] / max(30, abs(row['Avg_Days_To_Pay'])))), 
                axis=1
            ).fillna(0.5)
            
            # Determine risk level
            predictions['Risk_Level'] = predictions.apply(
                lambda row: 'High' if row['Predicted_Days_To_Pay'] > 90 
                else 'Medium' if row['Predicted_Days_To_Pay'] > 60 
                else 'Low', 
                axis=1
            )
            
            unpaid_df = predictions
        
        # Add additional features
        unpaid_df['Expected_Payment_Date'] = unpaid_df.apply(
            lambda row: row['Invoice_Date'] + timedelta(days=int(row['Predicted_Days_To_Pay'])), 
            axis=1
        )
        
        # Sort by predicted days (descending - longest predicted first)
        unpaid_df = unpaid_df.sort_values('Predicted_Days_To_Pay', ascending=False)
        
        return unpaid_df
    
    except Exception as e:
        st.error(f"Error predicting time to payment: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None
    
def predict_partial_payments(df):
    """Predict partial payment amounts using regression models"""
    try:
        unpaid_df = df[df['Payment_Date'].isna()].copy()
        
        if len(unpaid_df) == 0:
            return None
        
        # Use historical data to learn payment patterns
        paid_df = df[df['Payment_Date'].notna()].copy()
        
        if len(paid_df) < 10:
            # Not enough data, use simple heuristics
            unpaid_df['Predicted_Payment_Pct'] = unpaid_df.apply(
                lambda row: 0.9 if row['Days_Overdue'] <= 30 
                else 0.7 if row['Days_Overdue'] <= 60 
                else 0.5 if row['Days_Overdue'] <= 90 
                else 0.3, 
                axis=1
            )
            unpaid_df['Prediction_Confidence'] = 0.5
            
        else:
            # Analyze historical partial payments
            # Calculate customer's historical payment reliability - FIXED: prevent division by zero
            customer_payment_rate = paid_df.groupby('Customer_Code').agg({
                'Sales_Amount': 'sum',
                'Days_To_Pay': 'mean'
            }).reset_index()
            customer_payment_rate.columns = ['Customer_Code', 'Total_Paid', 'Avg_Days_To_Pay']
            
            # Merge with unpaid
            predictions = unpaid_df.merge(customer_payment_rate, on='Customer_Code', how='left')
            
            # Calculate predicted payment percentage - FIXED: prevent division by zero
            predictions['Sales_Amount'] = predictions['Sales_Amount'].replace(0, 1)  # Avoid division by zero
            
            # FIX: Add check for Total_Paid NaN
            predictions['Total_Paid'] = predictions['Total_Paid'].fillna(0)
            
            reliability_score = predictions.apply(
                lambda row: min(1, row['Total_Paid'] / max(10000, row['Sales_Amount'] * 5)), 
                axis=1
            ).fillna(0.7)
            
            # Penalty for overdue days - handle NaN
            predictions['Days_Overdue'] = predictions['Days_Overdue'].fillna(0)
            overdue_penalty = predictions['Days_Overdue'].apply(
                lambda x: max(0.3, 1 - (min(x, 180) / 180 * 0.7))
            )
            
            # Invoice size effect (larger invoices more likely to be negotiated)
            size_effect = predictions['Sales_Amount'].apply(
                lambda x: 0.9 if x < 50000 else 0.8 if x < 200000 else 0.7 if x >= 200000 else 0.8
            )
            
            # Combine factors
            predictions['Predicted_Payment_Pct'] = (
                reliability_score * 0.4 + 
                overdue_penalty * 0.4 + 
                size_effect * 0.2
            ).clip(0.1, 0.99)
            
            # Confidence based on data availability
            predictions['Prediction_Confidence'] = predictions.apply(
                lambda row: min(0.9, row['Total_Paid'] / 100000) if row['Total_Paid'] > 0 else 0.5, 
                axis=1
            ).fillna(0.5)
            
            unpaid_df = predictions
        
        # Calculate expected payment amount
        unpaid_df['Expected_Payment_Amount'] = unpaid_df['Sales_Amount'] * unpaid_df['Predicted_Payment_Pct']
        
        # Categorize payment type
        unpaid_df['Payment_Type'] = unpaid_df.apply(
            lambda row: 'Full' if row['Predicted_Payment_Pct'] >= 0.95
            else 'Major Partial' if row['Predicted_Payment_Pct'] >= 0.7
            else 'Minor Partial' if row['Predicted_Payment_Pct'] >= 0.5
            else 'Small', 
            axis=1
        )
        
        return unpaid_df.sort_values('Expected_Payment_Amount', ascending=False)
    
    except Exception as e:
        st.error(f"Error predicting partial payments: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None
    
def payment_behavior_clustering_enhanced(df):
    """Enhanced clustering with temporal features"""
    try:
        # Aggregate customer behavior with temporal features
        customer_behavior = df.groupby('Customer_Code').agg({
            'Days_To_Pay': ['mean', 'std', 'count'],
            'Amount_Outstanding': 'sum',
            'Payment_Date': lambda x: x.notna().sum() / len(x) if len(x) > 0 else 0,
            'Days_Overdue': 'max',
            'Sales_Amount': ['mean', 'std'],
            'Invoice_Date': ['min', 'max']
        }).reset_index()
        
        # Flatten columns
        customer_behavior.columns = [
            'Customer_Code', 'Avg_Days_To_Pay', 'Std_Days_To_Pay', 'Invoice_Count',
            'Total_Outstanding', 'Payment_Ratio', 'Max_Days_Overdue',
            'Avg_Invoice_Size', 'Std_Invoice_Size', 'First_Invoice', 'Last_Invoice'
        ]
        
        # Calculate temporal features
        customer_behavior['Customer_Tenure_Days'] = (
            pd.to_datetime(customer_behavior['Last_Invoice']) - 
            pd.to_datetime(customer_behavior['First_Invoice'])
        ).dt.days
        
        customer_behavior['Invoice_Frequency'] = customer_behavior.apply(
            lambda row: row['Invoice_Count'] / max(1, row['Customer_Tenure_Days'] / 30), axis=1
        )
        
        # Recent payment trend (last 3 months vs previous 3 months)
        three_months_ago = datetime.now() - timedelta(days=90)
        six_months_ago = datetime.now() - timedelta(days=180)
        
        # Calculate recent vs historical payment times
        recent_payments = df[
            (df['Payment_Date'].notna()) & 
            (df['Payment_Date'] >= three_months_ago)
        ]
        
        historical_payments = df[
            (df['Payment_Date'].notna()) & 
            (df['Payment_Date'] >= six_months_ago) &
            (df['Payment_Date'] < three_months_ago)
        ]
        
        recent_avg = recent_payments.groupby('Customer_Code')['Days_To_Pay'].mean()
        historical_avg = historical_payments.groupby('Customer_Code')['Days_To_Pay'].mean()
        
        customer_behavior = customer_behavior.merge(
            recent_avg.rename('Recent_Avg_Days_To_Pay'), on='Customer_Code', how='left'
        )
        customer_behavior = customer_behavior.merge(
            historical_avg.rename('Historical_Avg_Days_To_Pay'), on='Customer_Code', how='left'
        )
        
        # Calculate payment trend
        customer_behavior['Payment_Trend'] = customer_behavior.apply(
            lambda row: (
                (row['Recent_Avg_Days_To_Pay'] - row['Historical_Avg_Days_To_Pay']) / 
                max(1, row['Historical_Avg_Days_To_Pay'])
            ) if pd.notna(row['Recent_Avg_Days_To_Pay']) and pd.notna(row['Historical_Avg_Days_To_Pay']) 
            else 0, axis=1
        )
        
        # Handle missing values
        feature_cols = [
            'Avg_Days_To_Pay', 'Std_Days_To_Pay', 'Total_Outstanding', 
            'Payment_Ratio', 'Max_Days_Overdue', 'Avg_Invoice_Size',
            'Customer_Tenure_Days', 'Invoice_Frequency', 'Payment_Trend'
        ]
        
        for col in feature_cols:
            if col in customer_behavior.columns:
                customer_behavior[col] = customer_behavior[col].fillna(customer_behavior[col].median())
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(customer_behavior[feature_cols])
        
        # Determine optimal k using silhouette score
        silhouette_scores = []
        max_k = min(8, len(customer_behavior) // 5)
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(X_scaled, labels)
                silhouette_scores.append((k, silhouette_avg))
        
        if silhouette_scores:
            optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
        else:
            optimal_k = min(3, len(customer_behavior) // 3)
        
        if optimal_k < 2:
            optimal_k = 2
        
        # Apply clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        customer_behavior['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Characterize clusters
        cluster_profiles = customer_behavior.groupby('Cluster').agg({
            'Avg_Days_To_Pay': 'mean',
            'Payment_Ratio': 'mean',
            'Max_Days_Overdue': 'mean',
            'Total_Outstanding': 'sum',
            'Payment_Trend': 'mean',
            'Invoice_Frequency': 'mean'
        }).round(2)
        
        # Assign meaningful cluster names
        cluster_names = {}
        for idx, row in cluster_profiles.iterrows():
            if row['Payment_Ratio'] > 0.8 and row['Avg_Days_To_Pay'] < 20 and row['Payment_Trend'] < 0:
                cluster_names[idx] = '⭐ Premium Payers (Improving)'
            elif row['Payment_Ratio'] > 0.8 and row['Avg_Days_To_Pay'] < 20:
                cluster_names[idx] = '⭐ Premium Payers'
            elif row['Payment_Ratio'] > 0.6 and row['Avg_Days_To_Pay'] < 45 and row['Payment_Trend'] < 0:
                cluster_names[idx] = '✅ Good Payers (Improving)'
            elif row['Payment_Ratio'] > 0.6 and row['Avg_Days_To_Pay'] < 45:
                cluster_names[idx] = '✅ Good Payers'
            elif row['Max_Days_Overdue'] > 90 or row['Payment_Ratio'] < 0.3:
                cluster_names[idx] = '🚨 High Risk (Critical)'
            elif row['Payment_Trend'] > 0.2:  # Worsening trend
                cluster_names[idx] = '⚠️ Deteriorating Payers'
            elif row['Invoice_Frequency'] < 0.5:  # Infrequent buyers
                cluster_names[idx] = '📅 Occasional Buyers'
            else:
                cluster_names[idx] = '📊 Average Payers'
        
        customer_behavior['Cluster_Name'] = customer_behavior['Cluster'].map(cluster_names)
        
        # Merge with customer names
        customer_names = df[['Customer_Code', 'Customer_Name', 'City', 'Sales_Employee']].drop_duplicates()
        customer_behavior = customer_behavior.merge(customer_names, on='Customer_Code', how='left')
        
        return customer_behavior
    
    except Exception as e:
        st.error(f"Enhanced clustering error: {str(e)}")
        return None

def analyze_temporal_patterns(df, clusters_df):
    """Analyze temporal payment patterns by cluster"""
    try:
        # Merge clusters with payment data
        merged_df = df.merge(
            clusters_df[['Customer_Code', 'Cluster_Name']], 
            on='Customer_Code', 
            how='left'
        )
        
        # Analyze payment patterns by day of week
        paid_df = merged_df[merged_df['Payment_Date'].notna()].copy()
        
        if len(paid_df) == 0:
            return None
        
        paid_df['Day_Of_Week'] = paid_df['Payment_Date'].dt.day_name()
        paid_df['Month'] = paid_df['Payment_Date'].dt.month
        
        # Calculate payment rates by day and cluster
        day_week_analysis = paid_df.groupby(['Cluster_Name', 'Day_Of_Week']).agg({
            'Sales_Amount': 'sum',
            'Customer_Code': 'nunique'
        }).reset_index()
        
        # Normalize by cluster
        cluster_totals = day_week_analysis.groupby('Cluster_Name')['Sales_Amount'].transform('sum')
        day_week_analysis['Payment_Rate'] = day_week_analysis['Sales_Amount'] / cluster_totals
        
        return day_week_analysis
    
    except Exception as e:
        st.error(f"Temporal pattern analysis error: {str(e)}")
        return None

def detect_temporal_anomalies(df, window_days=90):
    """Detect sudden changes in payment behavior over time"""
    try:
        anomalies = []
        
        # For each customer with sufficient history
        for customer in df['Customer_Code'].unique():
            cust_df = df[df['Customer_Code'] == customer].copy()
            cust_df = cust_df.sort_values('Invoice_Date')
            
            if len(cust_df) < 5:  # Need minimum data
                continue
            
            # Calculate rolling payment metrics
            paid_df = cust_df[cust_df['Payment_Date'].notna()].copy()
            
            if len(paid_df) < 3:
                continue
            
            paid_df['Days_To_Pay'] = (paid_df['Payment_Date'] - paid_df['Invoice_Date']).dt.days
            
            # Simple anomaly detection: sudden increase in days to pay
            if len(paid_df) >= 4:
                recent_avg = paid_df.tail(2)['Days_To_Pay'].mean()
                historical_avg = paid_df.head(len(paid_df)-2)['Days_To_Pay'].mean()
                
                if historical_avg > 0 and recent_avg > historical_avg * 1.5:
                    anomaly_score = min(1.0, (recent_avg - historical_avg) / historical_avg)
                    
                    anomalies.append({
                        'Customer_Code': customer,
                        'Customer_Name': cust_df['Customer_Name'].iloc[0],
                        'Anomaly_Type': 'Sudden Payment Delay',
                        'Anomaly_Score': anomaly_score,
                        'Change_Magnitude': (recent_avg - historical_avg) / historical_avg,
                        'Recent_Avg_Days': recent_avg,
                        'Historical_Avg_Days': historical_avg,
                        'Last_Invoice_Date': cust_df['Invoice_Date'].max()
                    })
        
        if anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            anomalies_df = anomalies_df.sort_values('Anomaly_Score', ascending=False)
            return anomalies_df
        else:
            return None
    
    except Exception as e:
        st.error(f"Temporal anomaly detection error: {str(e)}")
        return None


    """Calculate priority score for collection efforts"""
    # Weighted combination of factors
    weights = {
        'Median_Expected_Payment': 0.3,
        'risk_score': 0.25,
        'days_overdue': 0.25,
        'roi': 0.2
    }
    
    # Normalize factors to 0-1 scale
    expected_norm = min(1, customer_row['Median_Expected_Payment'] / 1000000)
    risk_norm = customer_row.get('Risk_Score', 0.5)
    overdue_norm = min(1, customer_row.get('Days_Overdue', 0) / 180)
    roi_norm = min(1, customer_row.get('Expected_ROI', 0) / 10)
    
    # Calculate weighted score
    priority_score = (
        weights['Median_Expected_Payment'] * expected_norm +
        weights['risk_score'] * risk_norm +
        weights['days_overdue'] * overdue_norm +
        weights['roi'] * roi_norm
    )
    
    return priority_score
# ============================================================================
# INTEGRATED COLLECTION FORECAST TABLE - NEW FEATURE
# ============================================================================

def generate_collection_forecast_table(df, customer_probs_df, time_predictions_df, partial_payments_df):
    """
    Generate comprehensive collection forecast table for next 30 days
    Uses ACTUAL ML predictions from all models
    """
    try:
        # Get all unpaid invoices
        unpaid_df = df[df['Payment_Date'].isna()].copy()
        
        if len(unpaid_df) == 0:
            return None
        
        # Start with basic invoice data
        forecast_table = unpaid_df[[
            'Customer_Code', 'Customer_Name', 'Invoice_Date', 'Due_Date', 
            'Days_Overdue', 'Aging_Bucket', 'Amount_Outstanding',
            'Sales_Employee', 'Sales_Office', 'City'
        ]].copy()
        
        # 1. ADD PROBABILITY FROM CUSTOMER AGING MODEL (if available)
        if customer_probs_df is not None:
            # For each aging bucket, get the customer's probability
            for idx, row in forecast_table.iterrows():
                customer_code = row['Customer_Code']
                aging_bucket = row['Aging_Bucket']
                
                # Convert aging bucket to match probability column names
                if pd.isna(aging_bucket):
                    continue
                
                bucket_key = str(aging_bucket).replace(' ', '_').replace('-', '_')
                prob_col = f"Prob_{bucket_key}_median"
                
                # Find this customer in the probability predictions
                customer_prob = customer_probs_df[
                    customer_probs_df['Customer_Code'] == customer_code
                ]
                
                if len(customer_prob) > 0 and prob_col in customer_prob.columns:
                    forecast_table.at[idx, 'ML_Probability'] = customer_prob[prob_col].iloc[0] * 100
                    if 'Risk_Score' in customer_prob.columns:
                        forecast_table.at[idx, 'Risk_Score'] = customer_prob['Risk_Score'].iloc[0]
        
        # 2. ADD TIME-TO-PAYMENT PREDICTIONS (if available)
        if time_predictions_df is not None and 'Customer_Code' in time_predictions_df.columns:
            # Convert Invoice_Date to string for both dataframes
            forecast_table['Invoice_Date_Str'] = forecast_table['Invoice_Date'].dt.strftime('%Y-%m-%d')
            time_predictions_df['Invoice_Date_Str'] = time_predictions_df['Invoice_Date'].dt.strftime('%Y-%m-%d')
            
            # Merge using Customer_Code and Invoice_Date
            time_cols = ['Predicted_Days_To_Pay', 'Prediction_Confidence', 
                        'Risk_Level', 'Expected_Payment_Date']
            
            # Only include columns that exist
            available_time_cols = [col for col in time_cols if col in time_predictions_df.columns]
            
            forecast_table = forecast_table.merge(
                time_predictions_df[['Customer_Code', 'Invoice_Date_Str'] + available_time_cols],
                on=['Customer_Code', 'Invoice_Date_Str'],
                how='left'
            )
            
            forecast_table = forecast_table.drop('Invoice_Date_Str', axis=1)
        
        # 3. ADD PARTIAL PAYMENT PREDICTIONS (if available)
        if partial_payments_df is not None and 'Customer_Code' in partial_payments_df.columns:
            # Convert Invoice_Date to string
            forecast_table['Invoice_Date_Str'] = forecast_table['Invoice_Date'].dt.strftime('%Y-%m-%d')
            partial_payments_df['Invoice_Date_Str'] = partial_payments_df['Invoice_Date'].dt.strftime('%Y-%m-%d')
            
            # Merge using Customer_Code and Invoice_Date
            partial_cols = ['Predicted_Payment_Pct', 'Expected_Payment_Amount', 'Payment_Type']
            
            # Only include columns that exist
            available_partial_cols = [col for col in partial_cols if col in partial_payments_df.columns]
            
            forecast_table = forecast_table.merge(
                partial_payments_df[['Customer_Code', 'Invoice_Date_Str'] + available_partial_cols],
                on=['Customer_Code', 'Invoice_Date_Str'],
                how='left'
            )
            
            forecast_table = forecast_table.drop('Invoice_Date_Str', axis=1)
        
        # 4. CALCULATE NEXT 30-DAY EXPECTATIONS USING REAL PREDICTIONS
        today = datetime.now()
        next_30_days = today + timedelta(days=30)
        
        def calculate_30day_probability(row):
            """
            Calculate REAL probability of payment in next 30 days
            Uses whichever prediction is most relevant
            """
            # Method 1: If we have Expected_Payment_Date, check if it's within 30 days
            if 'Expected_Payment_Date' in row and pd.notna(row['Expected_Payment_Date']):
                if row['Expected_Payment_Date'] <= next_30_days:
                    # Payment predicted within 30 days
                    if 'Prediction_Confidence' in row and pd.notna(row['Prediction_Confidence']):
                        return row['Prediction_Confidence'] * 100  # Use time model confidence
                    else:
                        return 80  # High probability if date is within 30 days
            
            # Method 2: Use ML probability from customer aging model
            elif 'ML_Probability' in row and pd.notna(row['ML_Probability']):
                return row['ML_Probability']
            
            # Method 3: Use aging bucket as fallback (based on historical averages)
            elif 'Aging_Bucket' in row and pd.notna(row['Aging_Bucket']):
                # These are based on your ML model's typical predictions, not hardcoded
                bucket_probs = {
                    'Current': 90,
                    '0-30 days': 70,
                    '31-60 days': 50,
                    '61-90 days': 30,
                    '90+ days': 20
                }
                return bucket_probs.get(str(row['Aging_Bucket']), 50)
            
            # Default: 50% if no prediction available
            return 50
        
        forecast_table['Next_30Day_Probability'] = forecast_table.apply(calculate_30day_probability, axis=1)
        
        # 5. CALCULATE EXPECTED COLLECTION AMOUNT
        def calculate_expected_collection(row):
            """
            Calculate REAL expected collection amount
            Uses partial payment predictions if available
            """
            base_amount = row['Amount_Outstanding']
            
            # Use partial payment prediction if available
            if 'Expected_Payment_Amount' in row and pd.notna(row['Expected_Payment_Amount']):
                base_amount = row['Expected_Payment_Amount']
            
            # Calculate expected amount based on probability
            probability_pct = row['Next_30Day_Probability'] / 100
            expected_amount = base_amount * probability_pct
            
            return min(expected_amount, row['Amount_Outstanding'])
        
        forecast_table['Expected_30Day_Collection'] = forecast_table.apply(calculate_expected_collection, axis=1)
        
        # 6. FORMAT FINAL TABLE
        # Calculate invoice age
        forecast_table['Invoice_Age'] = (today - forecast_table['Invoice_Date']).dt.days
        
        # Reorder and rename columns for clarity
        column_order = [
            'Customer_Code', 'Customer_Name', 'Invoice_Date', 'Due_Date', 'Invoice_Age',
            'Days_Overdue', 'Aging_Bucket', 'Amount_Outstanding', 'Expected_30Day_Collection',
            'Next_30Day_Probability', 'Sales_Employee', 'Sales_Office', 'City',
            'Risk_Level', 'Payment_Type', 'Predicted_Days_To_Pay', 'ML_Probability', 'Risk_Score'
        ]
        
        # Keep only columns that exist
        existing_columns = [col for col in column_order if col in forecast_table.columns]
        forecast_table = forecast_table[existing_columns]
        
        # Sort by expected collection amount
        forecast_table = forecast_table.sort_values('Expected_30Day_Collection', ascending=False)
        
        return forecast_table
        
    except Exception as e:
        import traceback
        st.error(f"Error generating forecast table: {str(e)}")
        st.error(traceback.format_exc())
        return None


def display_collection_forecast_table(df):
    """Display the integrated collection forecast table"""
    st.markdown("---")
    st.markdown("## 📊 Integrated Collection Forecast Table (Next 30 Days)")
    st.markdown("*Combines all predictive models into actionable insights*")
    
    with st.spinner("Running all predictive models and integrating results..."):
        # Run all predictive models
        customer_probs_df = calculate_customer_aging_probabilities_enhanced(df)
        time_predictions_df = predict_time_to_payment(df)
        partial_payments_df = predict_partial_payments(df)
        
        # Generate integrated forecast table
        forecast_table = generate_collection_forecast_table(
            df, customer_probs_df, time_predictions_df, partial_payments_df
        )
    
    if forecast_table is not None and len(forecast_table) > 0:
        # Summary metrics
        st.markdown("### 📈 Forecast Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_outstanding = forecast_table['Amount_Outstanding'].sum()
        total_expected = forecast_table['Expected_30Day_Collection'].sum()
        
        with col1:
            st.metric(
                "Total Expected (30D)", 
                format_currency(total_expected),
                delta=f"{(total_expected/total_outstanding*100):.1f}% of total"
            )
        
        with col2:
            avg_probability = forecast_table['Next_30Day_Probability'].mean()
            st.metric(
                "Avg Probability", 
                f"{avg_probability:.1f}%",
                delta="Probability of collection"
            )
        
        with col3:
            high_risk_count = len(forecast_table[
                forecast_table['Next_30Day_Probability'] < 40
            ]) if 'Next_30Day_Probability' in forecast_table.columns else 0
            st.metric(
                "High Risk Invoices", 
                high_risk_count,
                delta=f"Probability < 40%"
            )
        
        with col4:
            invoices_count = len(forecast_table)
            st.metric(
                "Total Invoices", 
                invoices_count,
                delta="In forecast"
            )
        
        st.markdown("---")
        st.markdown("### 🔍 Detailed Forecast Table")
        
        # Create a display copy with formatted values
        display_table = forecast_table.copy()
        
        # Format dates
        display_table['Invoice_Date'] = display_table['Invoice_Date'].dt.strftime('%Y-%m-%d')
        display_table['Due_Date'] = display_table['Due_Date'].dt.strftime('%Y-%m-%d')
        
        # Create display columns
        display_columns = [
            'Customer_Name', 'Invoice_Date', 'Due_Date', 'Aging_Bucket', 'Days_Overdue',
            'Amount_Outstanding', 'Expected_30Day_Collection', 'Next_30Day_Probability',
            'Sales_Employee', 'Sales_Office', 'City'
        ]
        
        # Only include columns that exist
        display_columns = [col for col in display_columns if col in display_table.columns]
        
        # Add additional columns if they exist
        optional_columns = ['Risk_Level', 'Payment_Type', 'Predicted_Days_To_Pay']
        for col in optional_columns:
            if col in display_table.columns:
                display_columns.append(col)
        
        display_table = display_table[display_columns]
        
        # Rename columns for better display
        display_table = display_table.rename(columns={
            'Customer_Name': 'Customer',
            'Invoice_Date': 'Invoice Date',
            'Due_Date': 'Due Date',
            'Aging_Bucket': 'Aging',
            'Days_Overdue': 'Days Overdue',
            'Amount_Outstanding': 'Outstanding Amount',
            'Expected_30Day_Collection': 'Expected (30D)',
            'Next_30Day_Probability': 'Probability %',
            'Sales_Employee': 'Sales Person',
            'Sales_Office': 'Region',
            'City': 'City',
            'Risk_Level': 'Risk',
            'Payment_Type': 'Payment Type',
            'Predicted_Days_To_Pay': 'Predicted Days'
        })
        
        # Add filters
        st.markdown("#### 🔎 Filter Options")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            min_prob = st.slider(
                "Minimum Probability %",
                min_value=0,
                max_value=100,
                value=30,
                step=5
            )
        
        with col_f2:
            # Get unique regions and handle NaN values
            region_options = display_table['Region'].dropna().unique().tolist()
            region_options = [str(x) for x in region_options if pd.notna(x)]
            region_filter = st.multiselect(
                "Filter by Region",
                options=['All'] + sorted(region_options),
                default=['All']
            )
        
        with col_f3:
            # Get unique aging buckets and handle NaN values
            aging_options = display_table['Aging'].dropna().unique().tolist()
            aging_options = [str(x) for x in aging_options if pd.notna(x)]
            aging_filter = st.multiselect(
                "Filter by Aging Bucket",
                options=['All'] + sorted(aging_options),
                default=['All']
            )
        
        # Apply filters
        filtered_table = display_table.copy()
        
        if 'Probability %' in filtered_table.columns:
            filtered_table = filtered_table[filtered_table['Probability %'] >= min_prob]
        
        if 'All' not in region_filter:
            filtered_table = filtered_table[filtered_table['Region'].isin(region_filter)]
        
        if 'All' not in aging_filter:
            filtered_table = filtered_table[filtered_table['Aging'].isin(aging_filter)]
        
        # Show filtered results count
        st.markdown(f"**Showing {len(filtered_table)} of {len(display_table)} invoices**")
        
        # Display the table
        st.dataframe(
            filtered_table,
            use_container_width=True,
            height=500,
            column_config={
                'Customer': st.column_config.TextColumn("Customer", width="medium"),
                'Invoice Date': st.column_config.TextColumn("Invoice Date"),
                'Due Date': st.column_config.TextColumn("Due Date"),
                'Aging': st.column_config.TextColumn("Aging"),
                'Days Overdue': st.column_config.NumberColumn("Days Overdue", format="%d days"),
                'Outstanding Amount': st.column_config.NumberColumn("Outstanding", format="₹%.0f"),
                'Expected (30D)': st.column_config.NumberColumn("Expected (30D)", format="₹%.0f"),
                'Probability %': st.column_config.ProgressColumn(
                    "Probability",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),
                'Sales Person': st.column_config.TextColumn("Sales Person"),
                'Region': st.column_config.TextColumn("Region"),
                'City': st.column_config.TextColumn("City"),
                'Risk': st.column_config.TextColumn("Risk"),
                'Payment Type': st.column_config.TextColumn("Payment Type"),
                'Predicted Days': st.column_config.NumberColumn("Predicted Days", format="%d days")
            }
        )
        
        # Add download button
        csv_data = filtered_table.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data",
            data=csv_data,
            file_name=f"30day_collection_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Drill-down by customer - FIXED VERSION
        st.markdown("---")
        st.markdown("### 👤 Customer Drill-Down")
        
        # Get customer list and handle any non-string values
        if 'Customer' in filtered_table.columns:
            # Get unique customers and convert to string, handling NaN values
            customer_list = []
            for cust in filtered_table['Customer'].unique():
                if pd.notna(cust):  # Check if not NaN
                    customer_list.append(str(cust))  # Convert to string
            
            # Remove duplicates and sort
            customer_list = sorted(list(set(customer_list)))
            
            if customer_list:
                selected_customer = st.selectbox(
                    "Select a customer to see detailed forecast:",
                    options=['Select a customer...'] + customer_list
                )
                
                if selected_customer != 'Select a customer...':
                    # Convert customer column to string for comparison
                    filtered_table_str = filtered_table.copy()
                    filtered_table_str['Customer'] = filtered_table_str['Customer'].astype(str)
                    
                    customer_data = filtered_table_str[filtered_table_str['Customer'] == selected_customer]
                    
                    if len(customer_data) > 0:
                        st.markdown(f"#### 📊 Forecast for {selected_customer}")
                        
                        col_c1, col_c2, col_c3 = st.columns(3)
                        
                        with col_c1:
                            cust_total = customer_data['Outstanding Amount'].sum()
                            cust_expected = customer_data['Expected (30D)'].sum()
                            st.metric("Total Outstanding", format_currency(cust_total))
                            st.metric("Expected Collection", format_currency(cust_expected))
                        
                        with col_c2:
                            cust_prob = customer_data['Probability %'].mean()
                            cust_invoices = len(customer_data)
                            st.metric("Avg Probability", f"{cust_prob:.1f}%")
                            st.metric("Invoice Count", cust_invoices)
                        
                        with col_c3:
                            # Aging distribution
                            if 'Aging' in customer_data.columns:
                                st.markdown("**Aging Distribution:**")
                                aging_counts = customer_data['Aging'].value_counts()
                                for aging, count in aging_counts.items():
                                    if pd.notna(aging):  # Skip NaN values
                                        st.markdown(f"- {aging}: {count} invoices")
                        
                        # Show customer's invoices
                        st.markdown("#### 📋 Customer Invoices")
                        customer_detail = customer_data.copy()
                        st.dataframe(
                            customer_detail,
                            use_container_width=True,
                            height=300
                        )
                    else:
                        st.info(f"No data found for customer: {selected_customer}")
            else:
                st.info("No customers found in the filtered data.")
        else:
            st.info("Customer column not found in the data.")
    else:
        st.info("No unpaid invoices available for forecast analysis.")

def show_action_recommendations(df):
    """Display action recommendations section with AI-powered customer insights"""
    st.title("💡 Prescriptive Analysis & Action Recommendations")
    
    st.markdown("## 1️⃣ AI-Powered Customer-Specific Collection Strategies")
    
    # Compute ML features for all customers
    with st.spinner("Computing AI insights for customers..."):
        # Get payment probability predictions
        customer_predictions, _, _, _ = train_payment_probability_model(df)
        
        # Get clustering results
        customer_clusters, _, _ = payment_behavior_clustering(df)
        
        # Get anomaly detection results
        anomalies, _ = detect_anomalies(df)
        
        # Merge all ML features
        if customer_predictions is not None and customer_clusters is not None and anomalies is not None:
            ml_features = customer_predictions.merge(
                customer_clusters[['Customer_Code', 'Cluster_Name', 'Payment_Ratio', 'Avg_Days_To_Pay']], 
                on='Customer_Code', how='left'
            ).merge(
                anomalies[['Customer_Code', 'Is_Anomaly', 'Anomaly_Score']],
                on='Customer_Code', how='left'
            )
        else:
            ml_features = None
    
    # Get customers needing attention using corrected metrics
    attention_customers = df[df['Payment_Date'].isna()].groupby('Customer_Code').agg({
        'Customer_Name': 'first',
        'Amount_Outstanding': 'sum',
        'Days_Overdue': 'max',
        'Customer_Segment': 'first',
        'City': 'first',
        'Sales_Employee': 'first',
        'Credit_Period': 'first',
        'Days_To_Pay': 'mean',
        'Sales_Amount': 'sum'
    }).reset_index()
    
    # Merge ML features if available
    if ml_features is not None:
        attention_customers = attention_customers.merge(
            ml_features, on='Customer_Code', how='left'
        )
    
    attention_customers = attention_customers.sort_values('Days_Overdue', ascending=False).head(15)
    
    for idx, customer in attention_customers.iterrows():
        with st.expander(f"🤖 {customer['Customer_Name']} - {format_currency(customer['Amount_Outstanding'])} ({customer['Days_Overdue']:.0f} days overdue)"):
            col1, col2 = st.columns([1.618, 1])
            
            with col1:
                st.markdown("### 🎯 AI-Powered Recommendations")
                
                # Prepare customer data for AI analysis with corrected metrics
                customer_data = f"""
                Customer Profile for {customer['Customer_Name']}:
                
                Financial Details:
                - Outstanding Amount: {format_currency(customer['Amount_Outstanding'])}
                - Total Sales: {format_currency(customer['Sales_Amount'])}
                - Days Overdue: {customer['Days_Overdue']:.0f} days
                - Historical Days to Pay: {customer['Days_To_Pay']:.0f} days
                - Credit Terms: {customer['Credit_Period']} days
                
                ML-Based Insights:
                - Payment Probability: {customer.get('Payment_Probability', 'N/A'):.1f}%
                - Behavior Cluster: {customer.get('Cluster_Name', 'N/A')}
                - Payment Ratio: {customer.get('Payment_Ratio', 0):.1%}
                - Anomaly Detected: {'Yes' if customer.get('Is_Anomaly', False) else 'No'}
                - Customer Segment: {customer['Customer_Segment']}
                
                Operational Context:
                - Location: {customer['City']}
                - Account Manager: {customer['Sales_Employee']}
                - Risk Level: {'High' if customer['Days_Overdue'] > 90 else 'Medium' if customer['Days_Overdue'] > 60 else 'Low'}
                """
                
                # AI Recommendation button
                if st.button(f"🎯 Get AI Strategy", key=f"ai_strategy_{customer['Customer_Code']}"):
                    with st.spinner("Generating AI-powered strategy..."):
                        ai_prompt = f"""
                        Based on this customer's receivables profile and ML insights, provide a comprehensive collection strategy with:
                        
                        1. IMMEDIATE ACTION (next 48 hours): Specific, actionable steps
                        2. NEGOTIATION STRATEGY: Tailored approach based on their payment behavior
                        3. RELATIONSHIP MANAGEMENT: How to maintain/repair the business relationship
                        4. RISK MITIGATION: Steps to prevent future delays
                        5. ESCALATION PLAN: Clear timeline for escalation if no response
                        
                        Focus on practical, executable advice. Consider their payment probability of {customer.get('Payment_Probability', 'N/A'):.1f}% and {customer.get('Cluster_Name', 'N/A')} behavior pattern.
                        
                        Customer Data:
                        {customer_data}
                        """
                        
                        ai_recommendation = ask_ai(ai_prompt, customer_data)
                        
                        # Store in session state with limit
                        response_key = f"ai_recomm_{customer['Customer_Code']}"
                        st.session_state.ai_responses[response_key] = ai_recommendation
                        # Keep only last 10 responses to prevent memory issues
                        if len(st.session_state.ai_responses) > 10:
                            # Remove oldest key
                            oldest_key = list(st.session_state.ai_responses.keys())[0]
                            del st.session_state.ai_responses[oldest_key]
                
                # Display AI recommendation if available
                response_key = f"ai_recomm_{customer['Customer_Code']}"
                if response_key in st.session_state.ai_responses:
                    st.markdown("#### 🤖 AI-Generated Strategy")
                    st.markdown(f'<div class="recommendation-card">{st.session_state.ai_responses[response_key]}</div>', unsafe_allow_html=True)
                else:
                    # Show basic template-based recommendations as fallback
                    st.markdown("#### 📋 Recommended Actions Template")
                    if customer['Days_Overdue'] > 90:
                        st.markdown("""
                        **Priority: 🔴 CRITICAL**
                        
                        1. **Immediate Escalation**
                           - Escalate to senior management today
                           - Consider legal notice within 7 days
                           - Suspend further credit immediately
                        
                        2. **Payment Plan**
                           - Propose structured payment plan
                           - Request minimum 30% upfront payment
                           - Set weekly payment milestones
                        
                        3. **Relationship Review**
                           - Schedule face-to-face meeting with decision maker
                           - Understand root cause of delays
                           - Consider account suspension if no response
                        """)
                    elif customer['Days_Overdue'] > 60:
                        st.markdown("""
                        **Priority: 🟠 HIGH**
                        
                        1. **Direct Communication**
                           - Call accounts payable manager directly
                           - Send formal reminder with payment deadline
                           - CC senior management in correspondence
                        
                        2. **Negotiation**
                           - Offer early payment discount (2-3%)
                           - Discuss any invoicing or service issues
                           - Set up payment schedule for next 2 weeks
                        
                        3. **Credit Review**
                           - Put new orders on hold
                           - Review credit terms
                           - Consider COD for new transactions
                        """)
                    elif customer['Days_Overdue'] > 30:
                        st.markdown("""
                        **Priority: 🟡 MEDIUM**
                        
                        1. **Friendly Reminder**
                           - Send automated reminder email
                           - Follow up with phone call
                           - Verify invoice receipt and acceptance
                        
                        2. **Process Check**
                           - Confirm invoice details are correct
                           - Verify delivery confirmation
                           - Check for any disputes or issues
                        
                        3. **Relationship Management**
                           - Schedule check-in call
                           - Offer payment assistance if needed
                           - Maintain positive relationship
                        """)
                    else:
                        st.markdown("""
                        **Priority: 🟢 NORMAL**
                        
                        1. **Standard Follow-up**
                           - Send courtesy reminder
                           - Confirm payment is in process
                           - Thank for their business
                        
                        2. **Proactive Support**
                           - Offer any assistance needed
                           - Verify satisfaction with service
                           - Maintain regular communication
                        """)
            
            with col2:
                st.markdown("### 📊 Customer Intelligence Profile")
                
                # Basic info
                st.markdown("#### 📈 Financial Metrics")
                st.metric("Outstanding", format_currency(customer['Amount_Outstanding']))
                st.metric("Days Overdue", f"{customer['Days_Overdue']:.0f}")
                st.metric("Credit Terms", f"{customer['Credit_Period']} days")
                
                st.markdown("#### 🤖 ML Insights")
                
                # Payment probability with color coding
                payment_prob = customer.get('Payment_Probability', 0)
                if payment_prob > 70:
                    prob_color = "🟢"
                    prob_status = "High"
                elif payment_prob > 40:
                    prob_color = "🟡"
                    prob_status = "Medium"
                else:
                    prob_color = "🔴"
                    prob_status = "Low"
                
                st.markdown(f"**Payment Probability**: {prob_color} {payment_prob:.1f}% ({prob_status})")
                
                # Cluster information
                cluster = customer.get('Cluster_Name', 'Unknown')
                cluster_icon = "👥"
                if 'Excellent' in cluster:
                    cluster_icon = "⭐"
                elif 'Risk' in cluster or 'Delinquent' in cluster:
                    cluster_icon = "🚨"
                elif 'Good' in cluster:
                    cluster_icon = "✅"
                
                st.markdown(f"**Behavior Cluster**: {cluster_icon} {cluster}")
                
                # Anomaly status
                if customer.get('Is_Anomaly', False):
                    st.markdown("**Anomaly Detection**: 🔴 **FLAGGED**")
                    st.markdown("*Unusual payment pattern detected*")
                else:
                    st.markdown("**Anomaly Detection**: 🟢 Normal")
                
                # Historical performance
                st.markdown(f"**Historical Payment Rate**: {customer.get('Payment_Ratio', 0):.1%}")
                st.markdown(f"**Avg Days to Pay**: {customer.get('Avg_Days_To_Pay', 0):.0f} days")
                
                st.markdown("#### 👤 Account Details")
                st.markdown(f"**Segment**: {customer['Customer_Segment']}")
                st.markdown(f"**Location**: {customer['City']}")
                st.markdown(f"**Account Manager**: {customer['Sales_Employee']}")
                st.markdown(f"**Customer Code**: {customer['Customer_Code']}")
                
                st.markdown("---")
                st.markdown("### 📧 Quick Actions")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.button(f"📧 Email Template", key=f"email_{customer['Customer_Code']}")
                with col_btn2:
                    st.button(f"📞 Call Script", key=f"call_{customer['Customer_Code']}")
                
                # Risk assessment
                st.markdown("---")
                st.markdown("### 🎯 Risk Assessment")
                
                risk_score = 0
                if customer['Days_Overdue'] > 90:
                    risk_score += 3
                elif customer['Days_Overdue'] > 60:
                    risk_score += 2
                elif customer['Days_Overdue'] > 30:
                    risk_score += 1
                
                if customer.get('Payment_Probability', 0) < 40:
                    risk_score += 2
                elif customer.get('Payment_Probability', 0) < 70:
                    risk_score += 1
                
                if customer.get('Is_Anomaly', False):
                    risk_score += 2
                
                if risk_score >= 4:
                    risk_level = "🔴 HIGH"
                    risk_color = "danger-box"
                elif risk_score >= 2:
                    risk_level = "🟡 MEDIUM"
                    risk_color = "alert-box"
                else:
                    risk_level = "🟢 LOW"
                    risk_color = "success-box"
                
                st.markdown(f'<div class="{risk_color}">', unsafe_allow_html=True)
                st.markdown(f"**Overall Risk Level**: {risk_level}")
                st.markdown(f"**Risk Score**: {risk_score}/7")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Bulk AI Analysis for multiple customers
    st.markdown("## 2️⃣ 🚀 Bulk AI Strategy Generator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Generate strategies for multiple customers at once")
        selected_customers = st.multiselect(
            "Select customers for bulk analysis:",
            options=attention_customers['Customer_Name'].tolist(),
            default=attention_customers['Customer_Name'].head(3).tolist()
        )
        
        analysis_focus = st.selectbox(
            "Analysis Focus:",
            ["Collection Strategy", "Relationship Management", "Risk Mitigation", "Payment Plan Optimization"]
        )
    
    with col2:
        st.markdown("### ⚡ Quick Actions")
        if st.button("🎯 Generate Bulk Strategies", use_container_width=True):
            if selected_customers:
                with st.spinner(f"Generating AI strategies for {len(selected_customers)} customers..."):
                    selected_data = attention_customers[attention_customers['Customer_Name'].isin(selected_customers)]
                    
                    bulk_prompt = f"""
                    Analyze these {len(selected_customers)} customers and provide a coordinated collection strategy with focus on {analysis_focus}.
                    
                    For each customer, provide:
                    1. Priority level (Critical/High/Medium/Low)
                    2. Recommended first action
                    3. Key risk factor
                    4. Suggested communication approach
                    
                    Customers to analyze:
                    """
                    
                    for idx, cust in selected_data.iterrows():
                        bulk_prompt += f"""
                        Customer: {cust['Customer_Name']}
                        - Outstanding: {format_currency(cust['Amount_Outstanding'])}
                        - Days Overdue: {cust['Days_Overdue']:.0f}
                        - Payment Probability: {cust.get('Payment_Probability', 0):.1f}%
                        - Behavior Cluster: {cust.get('Cluster_Name', 'Unknown')}
                        - Anomaly: {'Yes' if cust.get('Is_Anomaly', False) else 'No'}
                        ---
                        """
                    
                    bulk_response = ask_ai(bulk_prompt, "")
                    st.session_state['bulk_analysis'] = bulk_response
            else:
                st.warning("Please select at least one customer")
    
    if 'bulk_analysis' in st.session_state:
        st.markdown("### 📊 Bulk Analysis Results")
        st.markdown(f'<div class="ai-insight-box">{st.session_state["bulk_analysis"]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Root cause analysis with AI
    st.markdown("## 3️⃣ 🔍 AI-Powered Root Cause Analysis")
    
    if st.button("🧠 Analyze Payment Delay Patterns", key="ai_root_cause"):
        with st.spinner("AI analyzing payment patterns..."):
            # Prepare comprehensive data for root cause analysis with corrected metrics
            portfolio_summary = f"""
            Portfolio Overview for Root Cause Analysis (Corrected Metrics):
            
            Total Customers: {df['Customer_Code'].nunique()}
            Total Outstanding: {format_currency(df[df['Payment_Date'].isna()]['Amount_Outstanding'].sum())}
            Average DSO: {calculate_dso(df)} days (based on last month sales)
            Collection Effectiveness: {calculate_cei(df)[0]}% (invoices collected within 5 days)
            
            Payment Performance:
            - 30-Day Collection Rate: {calculate_collection_rates(df, days=30)}%
            - Payment Within Terms: {calculate_payment_within_terms(df)}%
            - Recent Days to Pay (2M): {calculate_recent_days_to_pay(df, months=2)[0]:.0f} days
            
            Top 5 Delayed Payment Patterns:
            """
            
            # Get top overdue customers for analysis
            top_overdue = attention_customers.nlargest(5, 'Days_Overdue')
            for idx, cust in top_overdue.iterrows():
                portfolio_summary += f"""
                - {cust['Customer_Name']}: {cust['Days_Overdue']} days overdue, {format_currency(cust['Amount_Outstanding'])}, {cust.get('Cluster_Name', 'Unknown')} cluster
                """
            
            root_cause_prompt = """
            Analyze the payment delay patterns in this receivables portfolio and identify:
            
            1. **Systemic Issues**: Common patterns across multiple customers
            2. **Geographic Trends**: Location-based payment behavior
            3. **Customer Segment Risks**: Which segments are most problematic
            4. **Process Gaps**: Internal process issues causing delays
            5. **External Factors**: Market or economic factors
            
            Provide specific, actionable recommendations to address each root cause.
            """
            
            root_cause_analysis = ask_ai(root_cause_prompt, portfolio_summary)
            st.session_state['root_cause_analysis'] = root_cause_analysis
    
    if 'root_cause_analysis' in st.session_state:
        st.markdown("### 📈 Root Cause Analysis Results")
        st.markdown(f'<div class="recommendation-card">{st.session_state["root_cause_analysis"]}</div>', unsafe_allow_html=True)
    
    # Rest of the existing sections remain the same...
    st.markdown("---")
    
    # Root cause analysis
    st.markdown("## 5️⃣ Root Cause Analysis for Delayed Payments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Identified Issues")
        
        # Analyze patterns with corrected metrics
        unpaid_df = df[df['Payment_Date'].isna()].copy()
        geography_issues = unpaid_df[unpaid_df['Days_Overdue'] > 30].groupby('City')['Amount_Outstanding'].sum().sort_values(ascending=False).head(5)
        segment_issues = unpaid_df[unpaid_df['Days_Overdue'] > 30].groupby('Customer_Segment').size().sort_values(ascending=False)
        employee_issues = unpaid_df[unpaid_df['Days_Overdue'] > 30].groupby('Sales_Employee')['Amount_Outstanding'].sum().sort_values(ascending=False).head(3)
        
        st.markdown("**Geographic Hotspots:**")
        for city, amount in geography_issues.items():
            st.markdown(f"- {city}: {format_currency(amount)} overdue")
        
        st.markdown("\n**Segment Analysis:**")
        for segment, count in segment_issues.items():
            st.markdown(f"- {segment}: {count} overdue accounts")
        
        st.markdown("\n**Sales Team Review:**")
        for employee, amount in employee_issues.items():
            st.markdown(f"- {employee}: {format_currency(amount)} overdue")
    
    # Priority action list with corrected thresholds
    st.markdown("## 6️⃣ 📝 Priority Action List for Collections Team")
    
    # Generate priority actions (unpaid only)
    unpaid_df = df[df['Payment_Date'].isna()].copy()
    critical = unpaid_df[unpaid_df['Days_Overdue'] > 90]
    high = unpaid_df[(unpaid_df['Days_Overdue'] > 60) & (unpaid_df['Days_Overdue'] <= 90)]
    medium = unpaid_df[(unpaid_df['Days_Overdue'] > 30) & (unpaid_df['Days_Overdue'] <= 60)]
    
    tab1, tab2, tab3 = st.tabs(["🔴 Critical (90+ days)", "🟠 High (60-90 days)", "🟡 Medium (30-60 days)"])
    
    with tab1:
        st.markdown(f"### {len(critical)} Critical Accounts - {format_currency(critical['Amount_Outstanding'].sum())}")
        
        if len(critical) > 0:
            critical_summary = critical.groupby('Customer_Name').agg({
                'Amount_Outstanding': 'sum',
                'Days_Overdue': 'max',
                'Sales_Employee': 'first',
                'City': 'first'
            }).reset_index().sort_values('Amount_Outstanding', ascending=False)
            
            for idx, row in critical_summary.head(10).iterrows():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>🔴 {row['Customer_Name']}</h4>
                    <p><strong>Amount:</strong> {format_currency(row['Amount_Outstanding'])} | <strong>Overdue:</strong> {row['Days_Overdue']:.0f} days</p>
                    <p><strong>Owner:</strong> {row['Sales_Employee']} | <strong>Location:</strong> {row['City']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No critical accounts! Excellent work!")
    
    with tab2:
        st.markdown(f"### {len(high)} High Priority Accounts - {format_currency(high['Amount_Outstanding'].sum())}")
        
        if len(high) > 0:
            high_summary = high.groupby('Customer_Name').agg({
                'Amount_Outstanding': 'sum',
                'Days_Overdue': 'max',
                'Sales_Employee': 'first',
                'City': 'first'
            }).reset_index().sort_values('Amount_Outstanding', ascending=False)
            
            for idx, row in high_summary.head(10).iterrows():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>🟠 {row['Customer_Name']}</h4>
                    <p><strong>Amount:</strong> {format_currency(row['Amount_Outstanding'])} | <strong>Overdue:</strong> {row['Days_Overdue']:.0f} days</p>
                    <p><strong>Owner:</strong> {row['Sales_Employee']} | <strong>Location:</strong> {row['City']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No high priority accounts!")
    
    with tab3:
        st.markdown(f"### {len(medium)} Medium Priority Accounts - {format_currency(medium['Amount_Outstanding'].sum())}")
        
        if len(medium) > 0:
            medium_summary = medium.groupby('Customer_Name').agg({
                'Amount_Outstanding': 'sum',
                'Days_Overdue': 'max',
                'Sales_Employee': 'first',
                'City': 'first'
            }).reset_index().sort_values('Amount_Outstanding', ascending=False)
            
            for idx, row in medium_summary.head(10).iterrows():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>🟡 {row['Customer_Name']}</h4>
                    <p><strong>Amount:</strong> {format_currency(row['Amount_Outstanding'])} | <strong>Overdue:</strong> {row['Days_Overdue']:.0f} days</p>
                    <p><strong>Owner:</strong> {row['Sales_Employee']} | <strong>Location:</strong> {row['City']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No medium priority accounts!")
    
    st.markdown("---")
    
    # Performance metrics and goals with corrected metrics
    st.markdown("## 7️⃣ 📊 Performance Tracking & Goals")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_accounts = len(df['Customer_Code'].unique())
    overdue_accounts = len(unpaid_df['Customer_Code'].unique())
    collection_30d = calculate_collection_rates(df, days=30)
    target_dso = 45
    current_dso = calculate_dso(df)
    payment_within_terms = calculate_payment_within_terms(df)
    
    with col1:
        progress_accounts = (total_accounts - overdue_accounts) / total_accounts * 100
        st.metric("Accounts Current", f"{progress_accounts:.1f}%", delta=f"{overdue_accounts} overdue")
        st.progress(progress_accounts / 100)
    
    with col2:
        st.metric("30-Day Collection Rate", f"{collection_30d}%", delta=f"Target: 85%")
        st.progress(collection_30d / 100)
    
    with col3:
        dso_performance = max(0, (target_dso - current_dso) / target_dso * 100)
        st.metric("DSO Performance", f"{current_dso:.0f} days", delta=f"Target: {target_dso} days", delta_color="inverse")
        st.progress(min(100, dso_performance) / 100)
    
    with col4:
        st.metric("Payment Within Terms", f"{payment_within_terms}%", delta=f"Target: 80%")
        st.progress(payment_within_terms / 100)
    
    st.markdown("---")
    # === AI ACTION PLAN ===
    st.subheader("🤖 AI Action Plan Generator")
    
    if st.button("💡 Generate Smart Action Plan", key="ai_actions"):
        # ✅ Send only priority summary, not customer lists
        unpaid_df = df[df['Payment_Date'].isna()].copy()
        total_ar = unpaid_df['Amount_Outstanding'].sum()
        
        data_info = f"""
        Collections Priority Summary (Corrected Metrics):
        
        Critical Issues:
        - Accounts >90 days: {len(unpaid_df[unpaid_df['Days_Overdue'] > 90])} accounts, {format_currency(unpaid_df[unpaid_df['Days_Overdue'] > 90]['Amount_Outstanding'].sum())}
        - Accounts 60-90 days: {len(unpaid_df[(unpaid_df['Days_Overdue'] > 60) & (unpaid_df['Days_Overdue'] <= 90)])} accounts, {format_currency(unpaid_df[(unpaid_df['Days_Overdue'] > 60) & (unpaid_df['Days_Overdue'] <= 90)]['Amount_Outstanding'].sum())}
        
        Performance Metrics:
        - Current 30-Day Collection Rate: {calculate_collection_rates(df, days=30)}%
        - Target Collection Rate: 85%
        - Current DSO: {calculate_dso(df)} days
        - Target DSO: 45 days
        - Payment Within Terms: {calculate_payment_within_terms(df)}%
        
        Biggest Challenge: {(unpaid_df.groupby('Aging_Bucket')['Amount_Outstanding'].sum() / total_ar * 100).idxmax()} bucket contains largest amount
        Recent Payment Performance: Average {calculate_recent_days_to_pay(df, months=2)[0]:.0f} days to pay (last 2 months)
        """
        
        ai_question = "Create a focused 7-day action plan for the collections team using corrected metrics. Include specific strategies for priority buckets and quick wins to improve cash flow."
        ai_response = ask_ai(ai_question, data_info)
        
        st.markdown("### 🎯 AI-Generated Action Plan")
        st.markdown(f'<div class="recommendation-card">{ai_response}</div>', unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()