import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import re # For regular expressions in text cleaning
from collections import Counter # For counting word frequencies
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Import mlxtend for Apriori
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Import the PDF generation function from pdf.py
from pdf import generate_segment_report

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Global dataframes and lists to be shared across callbacks
df = pd.DataFrame() # Stores the raw uploaded DataFrame
clustered_data = pd.DataFrame() # Stores the DataFrame after clustering (either general or RFM-based)
feature_columns_global = [] # Stores the feature columns used for clustering (e.g., 'Recency', 'Frequency', 'Monetary' for RFM)
generated_figures = [] # Stores Plotly figures for PDF generation (only clustering related for now)
clv_data = pd.DataFrame() # Stores the DataFrame after CLV calculation
rules_df = pd.DataFrame() # Stores the association rules from Apriori for the recommendation engine

# --- Helper Function to Assign RFM Segments ---
def assign_rfm_segment(row):
    """
    Assigns a descriptive customer segment based on RFM scores.
    This logic can be customized based on business needs.
    """
    R, F, M = row['R'], row['F'], row['M']

    if R >= 4 and F >= 4 and M >= 4:
        return 'Loyal Customers'
    elif R >= 4 and F >= 3:
        return 'Recent Engaged'
    elif R >= 3 and M >= 4:
        return 'Big Spenders'
    elif R >= 4:
        return 'Recent Customers'
    elif F >= 4:
        return 'Frequent Buyers'
    elif M >= 4:
        return 'High Value Customers'
    elif R >= 3 and F >= 3:
        return 'Promising'
    elif R <= 2 and F <= 2 and M <= 2:
        return 'Churned/Lost'
    elif R <= 2:
        return 'At Risk'
    else:
        return 'Others'


# --- Helper Function for RFM Calculation ---
def calculate_rfm(df_raw, customer_id_col, invoice_date_col, invoice_col, quantity_col, price_col):
    """
    Performs RFM (Recency, Frequency, Monetary) analysis on the raw transactional data.

    Parameters:
    -----------
    df_raw : pd.DataFrame
        The raw DataFrame uploaded by the user.
    customer_id_col : str
        Name of the column containing customer IDs.
    invoice_date_col : str
        Name of the column containing invoice/transaction dates.
    invoice_col : str
        Name of the column containing invoice/transaction IDs.
    quantity_col : str
        Name of the column containing product quantities.
    price_col : str
        Name of the column containing product prices.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with 'CustomerID', 'Recency', 'Frequency', 'Monetary',
        'R', 'F', 'M' scores, and 'RFM_Segment' columns.
        Returns None if essential columns are missing or data is invalid.
    """
    temp_df = df_raw.copy()

    # Ensure required columns are present and not empty
    required_cols = [customer_id_col, invoice_date_col, invoice_col, quantity_col, price_col]
    if not all(col in temp_df.columns for col in required_cols):
        raise ValueError("One or more required RFM columns are missing from the data.")
    
    temp_df = temp_df.dropna(subset=required_cols)
    if temp_df.empty:
        raise ValueError("No data left after dropping rows with missing RFM-related values.")

    # Convert date column to datetime
    try:
        temp_df[invoice_date_col] = pd.to_datetime(temp_df[invoice_date_col])
    except Exception as e:
        raise ValueError(f"Could not convert '{invoice_date_col}' to datetime. Error: {e}")

    # Ensure quantity and price are numeric and positive
    try:
        temp_df[quantity_col] = pd.to_numeric(temp_df[quantity_col], errors='coerce')
        temp_df[price_col] = pd.to_numeric(temp_df[price_col], errors='coerce')
        temp_df = temp_df.dropna(subset=[quantity_col, price_col]) # Drop NaNs introduced by coercion
        temp_df = temp_df[temp_df[quantity_col] > 0] # Filter out negative quantities (returns/cancellations)
    except Exception as e:
        raise ValueError(f"Could not convert quantity or price to numeric. Error: {e}")

    # Calculate Revenue (product of Quantity and Price)
    temp_df['Revenue'] = temp_df[quantity_col] * temp_df[price_col]

    # Determine the reference date (latest transaction date + 1 day)
    if temp_df[invoice_date_col].empty:
        raise ValueError("Invoice Date column is empty after cleaning. Cannot determine reference date.")
    ref_date = temp_df[invoice_date_col].max() + timedelta(days=1)

    # Calculate RFM values
    rfm = temp_df.groupby(customer_id_col).agg(
        Recency=(invoice_date_col, lambda date: (ref_date - date.max()).days),
        Frequency=(invoice_col, 'nunique'),
        Monetary=('Revenue', 'sum') # Use 'Revenue' for Monetary component
    ).reset_index()

    rfm.rename(columns={customer_id_col: 'CustomerID'}, inplace=True)
    
    # Filter out customers with 0 or negative Monetary values (if any)
    rfm = rfm[rfm['Monetary'] > 0]

    # --- RFM Scoring (1-5) ---
    # Recency: lower value is better (higher score)
    r_labels = range(5, 0, -1)
    # Frequency & Monetary: higher value is better (higher score)
    f_labels = m_labels = range(1, 6)

    # Use qcut for quartile-based scoring
    # For Frequency, rank method 'first' is used to handle ties consistently
    rfm['R'] = pd.qcut(rfm['Recency'], q=5, labels=r_labels, duplicates='drop')
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=f_labels, duplicates='drop')
    rfm['M'] = pd.qcut(rfm['Monetary'], q=5, labels=m_labels, duplicates='drop')

    # Convert scores to int (qcut might return categorical)
    rfm['R'] = rfm['R'].astype(int)
    rfm['F'] = rfm['F'].astype(int)
    rfm['M'] = rfm['M'].astype(int)

    # --- RFM Segmentation ---
    rfm['RFM_Segment'] = rfm.apply(assign_rfm_segment, axis=1)

    return rfm

# --- Helper Function for CLV Calculation ---
def calculate_clv(df_raw, customer_id_col, invoice_date_col, invoice_col, quantity_col, price_col):
    """
    Calculates historical Customer Lifetime Value (CLV) for each customer.

    Parameters:
    -----------
    df_raw : pd.DataFrame
        The raw DataFrame uploaded by the user.
    customer_id_col : str
        Name of the column containing customer IDs.
    invoice_date_col : str
        Name of the column containing invoice/transaction dates.
    invoice_col : str
        Name of the column containing invoice/transaction IDs.
    quantity_col : str
        Name of the column containing product quantities.
    price_col : str
        Name of the column containing product prices.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with 'CustomerID', 'TotalRevenue', 'PurchaseCount',
        'AvgOrderValue', 'PurchaseFrequency', 'CustomerLifespanDays', 'CLV' columns.
    """
    temp_df = df_raw.copy()

    required_cols = [customer_id_col, invoice_date_col, invoice_col, quantity_col, price_col]
    if not all(col in temp_df.columns for col in required_cols):
        raise ValueError("One or more required CLV columns are missing from the data.")
    
    temp_df = temp_df.dropna(subset=required_cols)
    if temp_df.empty:
        raise ValueError("No data left after dropping rows with missing CLV-related values.")

    try:
        temp_df[invoice_date_col] = pd.to_datetime(temp_df[invoice_date_col])
    except Exception as e:
        raise ValueError(f"Could not convert '{invoice_date_col}' to datetime. Error: {e}")

    try:
        temp_df[quantity_col] = pd.to_numeric(temp_df[quantity_col], errors='coerce')
        temp_df[price_col] = pd.to_numeric(temp_df[price_col], errors='coerce')
        temp_df = temp_df.dropna(subset=[quantity_col, price_col])
        temp_df = temp_df[temp_df[quantity_col] > 0]
    except Exception as e:
        raise ValueError(f"Could not convert quantity or price to numeric. Error: {e}")

    temp_df['Revenue'] = temp_df[quantity_col] * temp_df[price_col]

    # Aggregate data per customer
    clv_df = temp_df.groupby(customer_id_col).agg(
        TotalRevenue=('Revenue', 'sum'),
        PurchaseCount=(invoice_col, 'nunique'),
        FirstPurchaseDate=(invoice_date_col, 'min'),
        LastPurchaseDate=(invoice_date_col, 'max')
    ).reset_index()

    clv_df.rename(columns={customer_id_col: 'CustomerID'}, inplace=True)

    # Calculate Average Order Value (AOV)
    clv_df['AvgOrderValue'] = clv_df['TotalRevenue'] / clv_df['PurchaseCount']

    # Calculate Customer Lifespan in Days
    clv_df['CustomerLifespanDays'] = (clv_df['LastPurchaseDate'] - clv_df['FirstPurchaseDate']).dt.days

    # For customers with only one purchase, lifespan is 0. We might consider a minimum lifespan or adjust.
    # For simplicity, if lifespan is 0, set it to 1 to avoid division by zero for frequency.
    clv_df['CustomerLifespanDays'] = clv_df['CustomerLifespanDays'].replace(0, 1)

    # Calculate Purchase Frequency (per day of lifespan)
    clv_df['PurchaseFrequency'] = clv_df['PurchaseCount'] / clv_df['CustomerLifespanDays']

    # Calculate CLV (simple historical model)
    # CLV = Average Order Value * Purchase Frequency * Customer Lifespan
    # Note: Purchase Frequency here is total purchases / lifespan.
    # So, CLV = AvgOrderValue * (PurchaseCount / CustomerLifespanDays) * CustomerLifespanDays
    # Which simplifies to CLV = AvgOrderValue * PurchaseCount = TotalRevenue
    # This is the historical CLV based on total spend.
    # For a predictive CLV, more complex models (e.g., probabilistic) are needed.
    clv_df['CLV'] = clv_df['TotalRevenue'] # Simple historical CLV

    # Add a CLV_Segment based on quartiles
    if len(clv_df) > 0:
        clv_df['CLV_Segment'] = pd.qcut(clv_df['CLV'], q=4, labels=['Low CLV', 'Medium CLV', 'High CLV', 'Very High CLV'], duplicates='drop')
    else:
        clv_df['CLV_Segment'] = pd.Series(dtype='object') # Empty series if no data

    return clv_df[['CustomerID', 'TotalRevenue', 'PurchaseCount', 'AvgOrderValue', 'PurchaseFrequency', 'CustomerLifespanDays', 'CLV', 'CLV_Segment']]


# --- Function to create visualizations (for clustering) ---
def create_clustering_visualizations(clustered_df, original_df, features, analysis_type):
    """
    Generates a list of Plotly figures for customer segmentation analysis.

    Parameters:
    -----------
    clustered_df : pd.DataFrame
        DataFrame containing customer data with a 'Cluster' column.
    original_df : pd.DataFrame
        The raw uploaded DataFrame (used for Quantity * Price calculation).
    features : list
        List of column names (features) used for clustering.
    analysis_type : str
        The type of analysis performed ('general_clustering' or 'rfm_analysis').

    Returns:
    --------
    list
        A list of Plotly graph_objects.Figure objects.
    """
    figures = [] # Initialize list to store all figures

    # Visualization 1: Pie Chart - Customer Distribution by Segment (K-Means Clusters)
    fig1 = px.pie(clustered_df, names='Cluster', title='Customer Distribution Across K-Means Segments',
                 color='Cluster', color_discrete_sequence=px.colors.qualitative.Bold)
    # Set textinfo to 'none' for no text on slices, and explicitly set hoverinfo
    fig1.update_traces(textinfo='none', hoverinfo='label+percent', pull=[0.05] * len(clustered_df['Cluster'].unique()))
    fig1.update_layout(legend_title_text='K-Means Segment')
    figures.append(fig1)
    
    # Visualization 2: Bar Chart - Feature Importance (Mean values by K-Means cluster)
    cluster_means = clustered_df.groupby('Cluster')[features].mean().reset_index()
    fig2 = px.bar(cluster_means.melt(id_vars='Cluster', value_vars=features),
                 x='variable', y='value', color='Cluster', barmode='group',
                 title='Average Feature Values for Each K-Means Segment',
                 labels={'variable': 'Feature', 'value': 'Average Value'},
                 color_discrete_sequence=px.colors.qualitative.Bold)
    figures.append(fig2)
    
    # Visualization 3: Radar Chart - Profile of each K-Means cluster
    cluster_profiles = clustered_df.groupby('Cluster')[features].mean()
    
    # Normalize data for radar chart to show relative strengths
    scaler = StandardScaler()
    cluster_profiles_scaled = pd.DataFrame(scaler.fit_transform(cluster_profiles),
                                         index=cluster_profiles.index,
                                         columns=cluster_profiles.columns)
    
    fig3 = go.Figure()
    for cluster in sorted(cluster_profiles_scaled.index):
        fig3.add_trace(go.Scatterpolar(
            r=cluster_profiles_scaled.loc[cluster].values,
            theta=features,
            fill='toself',
            name=f'K-Means Segment {cluster}'
        ))
    
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="K-Means Segment Profiles (Relative Strengths of Features)"
    )
    figures.append(fig3)
    
    # Visualization 4: PCA - 2D projection of K-Means clusters or 3D for RFM
    fig4 = go.Figure()
    if len(features) >= 3 and all(f in ['Recency', 'Frequency', 'Monetary'] for f in features):
        # Specific 3D scatter for RFM
        fig4 = px.scatter_3d(clustered_df, x='Recency', y='Frequency', z='Monetary', color='Cluster',
                             title='3D Visualization of K-Means Segments (Recency, Frequency, Monetary)',
                             color_discrete_sequence=px.colors.qualitative.Bold)
        fig4.update_layout(scene = dict(
                            xaxis_title='Recency (Days)',
                            yaxis_title='Frequency (Orders)',
                            zaxis_title='Monetary (Value)'))
    elif len(features) >= 2:
        # Apply PCA to reduce dimensionality to 2D
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(clustered_df[features])
        
        pca_df = pd.DataFrame({
            'PCA1': features_pca[:, 0],
            'PCA2': features_pca[:, 1],
            'Cluster': clustered_df['Cluster']
        })
        
        explained_var = pca.explained_variance_ratio_
        fig4 = px.scatter(pca_df, x='PCA1', y='PCA2', color='Cluster',
                        title=f'K-Means Segments: 2D PCA Projection<br> (Variance Explained by PC1: {explained_var[0]:.2%}, PC2: {explained_var[1]:.2%})',
                        opacity=0.7, color_discrete_sequence=px.colors.qualitative.Bold)
        
        # Add density contours to highlight cluster regions
        for cluster in sorted(pca_df['Cluster'].unique()):
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            fig4.add_trace(go.Histogram2dContour(
                x=cluster_data['PCA1'],
                y=cluster_data['PCA2'],
                colorscale='Blues',
                showscale=False,
                opacity=0.3,
                line=dict(width=0),
                hoverinfo='none'
            ))
    else:
        fig4 = px.scatter(x=[0], y=[0], title="PCA requires at least 2 features")
    figures.append(fig4)
    
    # Visualization 5: Parallel Coordinates - All features by K-Means cluster
    fig5 = px.parallel_coordinates(clustered_df, color='Cluster',
                               dimensions=features,
                               color_continuous_scale=px.colors.diverging.Tealrose,
                               title='Parallel Coordinates Plot: Feature Distribution by K-Means Segment')
    figures.append(fig5)

    # Visualization 6: Bar Chart - RFM Segment Distribution (Existing)
    fig6_rfm_segment_dist = go.Figure()
    if analysis_type == 'rfm_analysis' and 'RFM_Segment' in clustered_df.columns:
        rfm_segment_counts = clustered_df['RFM_Segment'].value_counts().reset_index()
        rfm_segment_counts.columns = ['RFM_Segment', 'Count']
        fig6_rfm_segment_dist = px.bar(rfm_segment_counts, x='RFM_Segment', y='Count',
                      title='Customer Count by RFM Segment Category',
                      color='RFM_Segment', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig6_rfm_segment_dist.update_layout(xaxis_title="RFM Segment", yaxis_title="Number of Customers")
    else:
        fig6_rfm_segment_dist.add_annotation(text="RFM Segment Distribution (Available with RFM Analysis)",
                                          xref="paper", yref="paper", showarrow=False, font=dict(size=16))
        fig6_rfm_segment_dist.update_layout(title_text='Customer Count by RFM Segment Category')
    figures.append(fig6_rfm_segment_dist)

    # Visualization 7: Pie Chart - Distribution of Monetary Ranges (Existing, but now conditional)
    fig7_monetary_ranges = go.Figure()
    if analysis_type == 'rfm_analysis' and 'Monetary' in clustered_df.columns and not clustered_df['Monetary'].empty:
        try:
            df_temp = clustered_df.copy()
            df_temp['Monetary'] = pd.to_numeric(df_temp['Monetary'], errors='coerce')
            df_temp = df_temp.dropna(subset=['Monetary'])

            if not df_temp['Monetary'].empty:
                df_temp['Order_Amount_Range'] = pd.qcut(
                    df_temp['Monetary'],
                    q=4,
                    labels=['Low Value', 'Medium Value', 'High Value', 'Very High Value'],
                    duplicates='drop'
                )
                df_temp = df_temp.dropna(subset=['Order_Amount_Range']) # Drop any NaNs from pd.qcut
                if not df_temp['Order_Amount_Range'].empty:
                    fig7_monetary_ranges = px.pie(df_temp, names='Order_Amount_Range', title='Distribution of Customers by Monetary Value Ranges (RFM)',
                                                 color='Order_Amount_Range', color_discrete_sequence=px.colors.qualitative.Pastel)
                    # Set textinfo to 'none' for no text on slices, and explicitly set hoverinfo
                    fig7_monetary_ranges.update_traces(textinfo='none', hoverinfo='label+percent', pull=[0.05] * len(df_temp['Order_Amount_Range'].unique()))
                    fig7_monetary_ranges.update_layout(legend_title_text='Monetary Range')
                else:
                    fig7_monetary_ranges = go.Figure().add_annotation(text="Monetary data insufficient for ranges after cleaning.",
                                                                     xref="paper", yref="paper", showarrow=False,
                                                                     font=dict(size=14, color="red"))
                    fig7_monetary_ranges.update_layout(title_text='Distribution of Customers by Monetary Value Ranges (RFM)')
            else:
                fig7_monetary_ranges = go.Figure().add_annotation(text="Monetary data not available for Monetary Ranges.",
                                                                 xref="paper", yref="paper", showarrow=False,
                                                                 font=dict(size=14, color="gray"))
                fig7_monetary_ranges.update_layout(title_text='Distribution of Customers by Monetary Value Ranges (RFM)')
        except Exception as e:
            print(f"Warning: Could not create Monetary Range pie chart. Error: {e}")
            fig7_monetary_ranges = go.Figure().add_annotation(text=f"Error creating Monetary Range chart: {e}",
                                                             xref="paper", yref="paper", showarrow=False,
                                                             font=dict(size=14, color="red"))
            fig7_monetary_ranges.update_layout(title_text='Distribution of Customers by Monetary Value Ranges (RFM)')
    else:
        fig7_monetary_ranges = go.Figure().add_annotation(text="Distribution of Monetary Ranges (Available with RFM Analysis)",
                                                         xref="paper", yref="paper", showarrow=False,
                                                         font=dict(size=16))
        fig7_monetary_ranges.update_layout(title_text='Distribution of Customers by Monetary Value Ranges (RFM)')
    figures.append(fig7_monetary_ranges)

    # NEW Visualization 8: Pie Chart - Distribution of Order Amounts (Quantity * Price) for General Clustering
    fig8_order_amount_qp_ranges = go.Figure()
    if analysis_type == 'general_clustering' and 'Quantity' in original_df.columns and 'Price' in original_df.columns:
        try:
            df_temp_qp = original_df.copy()
            df_temp_qp['Quantity'] = pd.to_numeric(df_temp_qp['Quantity'], errors='coerce')
            df_temp_qp['Price'] = pd.to_numeric(df_temp_qp['Price'], errors='coerce')
            df_temp_qp = df_temp_qp.dropna(subset=['Quantity', 'Price'])
            df_temp_qp = df_temp_qp[df_temp_qp['Quantity'] > 0] # Filter out non-positive quantities

            if not df_temp_qp.empty:
                df_temp_qp['Order_Amount'] = df_temp_qp['Quantity'] * df_temp_qp['Price']
                
                price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
                # Ensure labels match the bins correctly
                labels = [f'${price_range[i]}-{price_range[i+1]-1}' for i in range(len(price_range)-1)]
                labels.append(f'${price_range[-1]}+') # For the last bin

                df_temp_qp['Order_Amount_Range_QP'] = pd.cut(
                    df_temp_qp['Order_Amount'],
                    bins=price_range + [np.inf], # Add infinity for the last bin
                    labels=labels,
                    right=False, # Interval is [a, b)
                    include_lowest=True # Include 0 in the first bin
                )
                
                # Drop any NaNs that might result from pd.cut if values fall outside defined bins
                df_temp_qp = df_temp_qp.dropna(subset=['Order_Amount_Range_QP'])

                if not df_temp_qp['Order_Amount_Range_QP'].empty:
                    fig8_order_amount_qp_ranges = px.pie(df_temp_qp, names='Order_Amount_Range_QP',
                                                         title='Distribution of Transactions by Order Amount Ranges',
                                                         color='Order_Amount_Range_QP',
                                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                    # Set textinfo to 'none' for no text on slices, and explicitly set hoverinfo
                    fig8_order_amount_qp_ranges.update_traces(textinfo='none', hoverinfo='label+percent', pull=[0.05] * len(df_temp_qp['Order_Amount_Range_QP'].unique()))
                    fig8_order_amount_qp_ranges.update_layout(legend_title_text='Order Amount Range')
                else:
                    fig8_order_amount_qp_ranges = go.Figure().add_annotation(text="Order Amount (Q*P) data insufficient for ranges after cleaning.",
                                                                             xref="paper", yref="paper", showarrow=False,
                                                                             font=dict(size=14, color="red"))
                    fig8_order_amount_qp_ranges.update_layout(title_text='Distribution of Transactions by Order Amount Ranges')
            else:
                fig8_order_amount_qp_ranges = go.Figure().add_annotation(text="No valid Quantity/Price data for Order Amount (Q*P) ranges.",
                                                                         xref="paper", yref="paper", showarrow=False,
                                                                         font=dict(size=14, color="gray"))
                fig8_order_amount_qp_ranges.update_layout(title_text='Distribution of Transactions by Order Amount Ranges')
        except Exception as e:
            print(f"Warning: Could not create Order Amount (Q*P) Range pie chart. Error: {e}")
            fig8_order_amount_qp_ranges = go.Figure().add_annotation(text=f"Error creating Order Amount (Q*P) Range chart: {e}",
                                                                     xref="paper", yref="paper", showarrow=False,
                                                                     font=dict(size=14, color="red"))
            fig8_order_amount_qp_ranges.update_layout(title_text='Distribution of Transactions by Order Amount Ranges')
    else:
        fig8_order_amount_qp_ranges = go.Figure().add_annotation(text="Order Amount Distribution (Q*P) (Available with General Clustering)",
                                                                 xref="paper", yref="paper", showarrow=False,
                                                                 font=dict(size=16))
        fig8_order_amount_qp_ranges.update_layout(title_text='Distribution of Transactions by Order Amount Ranges')
    figures.append(fig8_order_amount_qp_ranges)

    # NEW Visualization 9: Scatter plot - Frequency vs. Recency
    fig9_freq_vs_recency = go.Figure()
    if analysis_type == 'rfm_analysis' and 'Recency' in clustered_df.columns and 'Frequency' in clustered_df.columns:
        fig9_freq_vs_recency = px.scatter(clustered_df, x='Recency', y='Frequency', color='Cluster',
                                         title='Customer Frequency vs. Recency by K-Means Segment',
                                         labels={'Recency': 'Recency (Days Since Last Purchase)', 'Frequency': 'Frequency (Number of Purchases)'},
                                         color_discrete_sequence=px.colors.qualitative.Bold)
    else:
        fig9_freq_vs_recency.add_annotation(text="Frequency vs. Recency (Available with RFM Analysis)",
                                          xref="paper", yref="paper", showarrow=False, font=dict(size=16))
        fig9_freq_vs_recency.update_layout(title_text='Customer Frequency vs. Recency by K-Means Segment')
    figures.append(fig9_freq_vs_recency)

    # NEW Visualization 10: Scatter plot - Frequency vs. Monetary
    fig10_freq_vs_monetary = go.Figure()
    if analysis_type == 'rfm_analysis' and 'Frequency' in clustered_df.columns and 'Monetary' in clustered_df.columns:
        fig10_freq_vs_monetary = px.scatter(clustered_df, x='Monetary', y='Frequency', color='Cluster',
                                          title='Customer Frequency vs. Monetary Value by K-Means Segment',
                                          labels={'Monetary': 'Monetary Value (Total Spend)', 'Frequency': 'Frequency (Number of Purchases)'},
                                          color_discrete_sequence=px.colors.qualitative.Bold)
    else:
        fig10_freq_vs_monetary.add_annotation(text="Frequency vs. Monetary (Available with RFM Analysis)",
                                           xref="paper", yref="paper", showarrow=False, font=dict(size=16))
        fig10_freq_vs_monetary.update_layout(title_text='Customer Frequency vs. Monetary Value by K-Means Segment')
    figures.append(fig10_freq_vs_monetary)

    # NEW Visualization 11: Scatter plot - Monetary vs. Recency
    fig11_monetary_vs_recency = go.Figure()
    if analysis_type == 'rfm_analysis' and 'Monetary' in clustered_df.columns and 'Recency' in clustered_df.columns:
        fig11_monetary_vs_recency = px.scatter(clustered_df, x='Recency', y='Monetary', color='Cluster',
                                            title='Customer Monetary Value vs. Recency by K-Means Segment',
                                            labels={'Recency': 'Recency (Days Since Last Purchase)', 'Monetary': 'Monetary Value (Total Spend)'},
                                            color_discrete_sequence=px.colors.qualitative.Bold)
    else:
        fig11_monetary_vs_recency.add_annotation(text="Monetary vs. Recency (Available with RFM Analysis)",
                                              xref="paper", yref="paper", showarrow=False, font=dict(size=16))
        fig11_monetary_vs_recency.update_layout(title_text='Customer Monetary Value vs. Recency by K-Means Segment')
    figures.append(fig11_monetary_vs_recency)


    return figures

# --- Basic Sentiment Lexicon (for demonstration) ---
# In a real application, you'd use a more comprehensive lexicon or a pre-trained model.
positive_words = set([
    "good", "great", "excellent", "amazing", "fantastic", "superb", "love", "happy",
    "satisfied", "recommend", "positive", "awesome", "perfect", "brilliant", "enjoy",
    "delightful", "impressed", "wonderful", "fabulous", "pleased", "smooth", "efficient"
])

negative_words = set([
    "bad", "poor", "terrible", "horrible", "disappointing", "hate", "unhappy",
    "dissatisfied", "negative", "awful", "broken", "slow", "expensive", "issue",
    "problem", "bug", "frustrating", "waste", "unreliable", "difficult", "clunky"
])

def analyze_sentiment(text):
    """
    Performs basic sentiment analysis on a given text using a simple lexicon.
    Returns 'Positive', 'Negative', or 'Neutral'.
    """
    if not isinstance(text, str):
        return 'Neutral' # Handle non-string inputs

    cleaned_text = re.sub(r'[^\w\s]', '', text).lower()
    words = word_tokenize(cleaned_text)
    
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)

    if positive_score > negative_score:
        return 'Positive'
    elif negative_score > positive_score:
        return 'Negative'
    else:
        return 'Neutral'

# --- Helper Function for Cohort Analysis ---
def calculate_cohort_retention(df_raw, customer_id_col, order_date_col):
    """
    Performs cohort analysis and calculates retention rates.

    Parameters:
    -----------
    df_raw : pd.DataFrame
        The raw DataFrame uploaded by the user.
    customer_id_col : str
        Name of the column containing customer IDs.
    order_date_col : str
        Name of the column containing order/transaction dates.

    Returns:
    --------
    tuple: (pd.DataFrame, pd.DataFrame)
        A tuple containing:
        - cohort_counts: DataFrame with the number of active customers per cohort per month.
        - retention_matrix: DataFrame with retention rates.
        Returns None if essential columns are missing or data is invalid.
    """
    temp_df = df_raw.copy()

    required_cols = [customer_id_col, order_date_col]
    if not all(col in temp_df.columns for col in required_cols):
        raise ValueError("One or more required Cohort Analysis columns are missing from the data.")
    
    temp_df = temp_df.dropna(subset=required_cols)
    if temp_df.empty:
        raise ValueError("No data left after dropping rows with missing Cohort Analysis values.")

    # Convert date column to datetime
    try:
        temp_df[order_date_col] = pd.to_datetime(temp_df[order_date_col])
    except Exception as e:
        raise ValueError(f"Could not convert '{order_date_col}' to datetime. Error: {e}")

    # Extract order month
    temp_df['OrderMonth'] = temp_df[order_date_col].dt.to_period('M')

    # Determine the cohort (first purchase month) for each customer
    cohort_df = temp_df.groupby(customer_id_col)['OrderMonth'].min().reset_index()
    cohort_df.rename(columns={'OrderMonth': 'CohortMonth'}, inplace=True)
    
    # Merge cohort month back to the original transactions
    df_merged = temp_df.merge(cohort_df, on=customer_id_col)

    # Calculate the difference in months between OrderMonth and CohortMonth
    df_merged['CohortIndex'] = (df_merged['OrderMonth'] - df_merged['CohortMonth']).apply(lambda x: x.n)

    # Count active customers per cohort per month
    cohort_counts = df_merged.groupby(['CohortMonth', 'CohortIndex'])[customer_id_col].nunique().reset_index()
    cohort_counts.rename(columns={customer_id_col: 'CustomerCount'}, inplace=True)

    # Create a pivot table for retention
    cohort_pivot = cohort_counts.pivot_table(index='CohortMonth',
                                             columns='CohortIndex',
                                             values='CustomerCount')

    # Calculate retention matrix
    cohort_sizes = cohort_pivot.iloc[:, 0] # Size of each cohort (month 0)
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100 # Convert to percentage

    return cohort_counts, retention_matrix


# --- Dash App Layout ---
app.layout = dbc.Container([
    dbc.Row([
        # Sidebar for controls
        dbc.Col([
            html.H2("Customer Segmentation", style={"textAlign": "center", "marginBottom": 20}),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.I(className="fas fa-file-upload me-2"),
                    "Drag & Drop or Click to Upload CSV"
                ]),
                style={
                    'width': '100%', 'height': '80px', 'lineHeight': '80px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'marginBottom': '15px'
                },
                multiple=False
            ),
            dbc.Toast( # Toast notification for successful upload
                "Upload Successful! Ready to configure segmentation.",
                id="upload-toast",
                header="Success",
                is_open=False,
                dismissable=True,
                icon="success",
                duration=4000,
                style={"position": "fixed", "top": 10, "right": 10, "width": 350, "zIndex": 9999},
            ),
            html.Hr(),

            html.H5("Choose Analysis Type:", className="mb-2"),
            dcc.RadioItems(
                id='analysis-type-radio',
                options=[
                    {'label': '  General Clustering (Numeric Features)', 'value': 'general_clustering'},
                    {'label': '  RFM Analysis & Clustering', 'value': 'rfm_analysis'},
                    {'label': '  Word Occurrence Analysis', 'value': 'word_occurrence'},
                    {'label': '  Apriori Analysis', 'value': 'apriori_analysis'},
                    {'label': '  CLV Analysis', 'value': 'clv_analysis'}, # New option for CLV
                    {'label': '  Recommendation Engine (Basic)', 'value': 'recommendation_engine'}, # New option for Recommendation Engine
                    {'label': '  Sentiment Analysis (Basic)', 'value': 'sentiment_analysis'}, # New option for Sentiment Analysis
                    {'label': '  Cohort Analysis', 'value': 'cohort_analysis'} # New option for Cohort Analysis
                ],
                value='general_clustering', # Default selected
                inline=False,
                className="mb-3"
            ),
            html.Hr(),

            # General Clustering Feature Selection (conditionally displayed)
            html.Div(id='general-clustering-feature-selection', style={'display': 'block'}, children=[
                html.H5("Select Features for General Clustering:", className="mb-3"),
                dcc.Dropdown(
                    id='general-features-dropdown',
                    options=[], # Populated by callback
                    multi=True,
                    placeholder="Select numeric features for clustering",
                    className="mb-2"
                ),
                html.Hr()
            ]),

            # RFM column selection (conditionally displayed)
            html.Div(id='rfm-column-selection', style={'display': 'none'}, children=[
                html.H5("Map RFM Columns:", className="mb-3"),
                html.P("Select the corresponding columns from your uploaded CSV for RFM analysis:"),
                dbc.Row([
                    dbc.Col(html.Label("Customer ID Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='customer-id-dropdown', placeholder="Select Customer ID", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Invoice Date Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='invoice-date-dropdown', placeholder="Select Invoice Date", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Invoice/Trans. ID Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='invoice-dropdown', placeholder="Select Invoice ID", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Quantity Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='quantity-dropdown', placeholder="Select Quantity", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Price Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='price-dropdown', placeholder="Select Price", className="mb-2", value=None, options=[]), width=7)
                ]),
                html.Hr()
            ]),

            # CLV column selection (NEW, conditionally displayed)
            html.Div(id='clv-column-selection', style={'display': 'none'}, children=[
                html.H5("Map CLV Columns:", className="mb-3"),
                html.P("Select the corresponding columns from your uploaded CSV for CLV analysis:"),
                dbc.Row([
                    dbc.Col(html.Label("Customer ID Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='clv-customer-id-dropdown', placeholder="Select Customer ID", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Invoice Date Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='clv-invoice-date-dropdown', placeholder="Select Invoice Date", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Invoice/Trans. ID Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='clv-invoice-dropdown', placeholder="Select Invoice ID", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Quantity Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='clv-quantity-dropdown', placeholder="Select Quantity", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Price Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='clv-price-dropdown', placeholder="Select Price", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Button([
                    html.I(className="fas fa-dollar-sign me-2"),
                    "Run CLV Analysis"
                ], id="run-clv-analysis", color="success", className="w-100 mb-3"),
                html.Hr()
            ]),

            # Word Occurrence Section (conditionally displayed)
            html.Div(id='word-occurrence-section', style={'display': 'none'}, children=[
                html.H5("Analyze Word Occurrence:", className="mb-3"),
                html.P("Select a text column to find trending words (e.g., product descriptions):"),
                dcc.Dropdown(
                    id='text-column-dropdown',
                    options=[], # Populated by callback
                    placeholder="Select a text column",
                    className="mb-2"
                ),
                dbc.Button([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Analyze Word Occurrence"
                ], id="run-word-occurrence", color="secondary", className="w-100 mb-3"),
                html.Hr()
            ]),

            # Apriori Analysis Section (new, conditionally displayed)
            html.Div(id='apriori-analysis-section', style={'display': 'none'}, children=[
                html.H5("Apriori Algorithm Settings:", className="mb-3"),
                html.P("Select columns for transactional data:"),
                dbc.Row([
                    dbc.Col(html.Label("Transaction ID Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='apriori-transaction-id-dropdown', placeholder="Select Transaction ID", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Item ID Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='apriori-item-id-dropdown', placeholder="Select Item ID", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Minimum Support (0.0-1.0):"), width=7),
                    dbc.Col(dcc.Input(id='min-support-input', type='number', value=0.01, min=0.001, max=1.0, step=0.001, className="mb-2"), width=5)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Minimum Confidence (0.0-1.0):"), width=7),
                    dbc.Col(dcc.Input(id='min-confidence-input', type='number', value=0.5, min=0.0, max=1.0, step=0.01, className="mb-2"), width=5)
                ]),
                dbc.Button([
                    html.I(className="fas fa-sitemap me-2"),
                    "Run Apriori Analysis"
                ], id="run-apriori", color="success", className="w-100 mb-3"),
                html.Hr()
            ]),

            # Recommendation Engine Section (NEW, conditionally displayed)
            html.Div(id='recommendation-engine-section', style={'display': 'none'}, children=[
                html.H5("Item-Based Recommendations:", className="mb-3"),
                html.P("Select an item to see what other items customers frequently bought with it (requires Apriori Analysis to be run first):"),
                dcc.Dropdown(
                    id='recommendation-item-dropdown',
                    options=[], # Populated by callback after Apriori runs
                    placeholder="Select an item for recommendations",
                    className="mb-2"
                ),
                dbc.Button([
                    html.I(className="fas fa-lightbulb me-2"),
                    "Get Recommendations"
                ], id="run-recommendation-engine", color="info", className="w-100 mb-3"),
                html.Hr()
            ]),

            # Sentiment Analysis Section (NEW, conditionally displayed)
            html.Div(id='sentiment-analysis-section', style={'display': 'none'}, children=[
                html.H5("Perform Sentiment Analysis:", className="mb-3"),
                html.P("Select a text column containing reviews or feedback:"),
                dcc.Dropdown(
                    id='sentiment-text-column-dropdown',
                    options=[], # Populated by callback
                    placeholder="Select a text column for sentiment analysis",
                    className="mb-2"
                ),
                dbc.Button([
                    html.I(className="fas fa-smile me-2"),
                    "Run Sentiment Analysis"
                ], id="run-sentiment-analysis", color="warning", className="w-100 mb-3"),
                html.Hr()
            ]),

            # Cohort Analysis Section (NEW, conditionally displayed)
            html.Div(id='cohort-analysis-section', style={'display': 'none'}, children=[
                html.H5("Cohort Analysis Settings:", className="mb-3"),
                html.P("Select columns for cohort analysis:"),
                dbc.Row([
                    dbc.Col(html.Label("Customer ID Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='cohort-customer-id-dropdown', placeholder="Select Customer ID", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Order Date Column:"), width=5),
                    dbc.Col(dcc.Dropdown(id='cohort-order-date-dropdown', placeholder="Select Order Date", className="mb-2", value=None, options=[]), width=7)
                ]),
                dbc.Button([
                    html.I(className="fas fa-users me-2"),
                    "Run Cohort Analysis"
                ], id="run-cohort-analysis", color="primary", className="w-100 mb-3"),
                html.Hr()
            ]),
            
            dbc.Button([
                html.I(className="fas fa-play me-2"),
                "Run Segmentation"
            ], id="run-segmentation", color="primary",
            className="w-100 mb-3"),
            dbc.Button([
                html.I(className="fas fa-file-pdf me-2"),
                "Download Report (PDF)"
            ], id="download-report-button", color="info", className="w-100 mb-3"),
            dcc.Download(id="download-pdf-report"),
            html.Div(id='error-message', style={'color': 'red', 'marginTop': 10}),
            
            html.Div(id='loading-output'), # For showing loading message during processing

            html.Div(id='run-output-message', className="mt-3"), # Feedback on run
            html.Hr(),
            html.Div([
                html.H5("Segment Filter", className="mb-2"),
                dcc.Dropdown(id='cluster-dropdown', placeholder="Select Segment"),
                html.Div(id='selected-segment-info', className="mt-3")
            ], className="mt-4")
        ], width=3, style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "10px"}),

        # Main content area for dashboard
        dbc.Col([
            html.H2("Customer Segmentation Dashboard", style={"textAlign": "center", "marginBottom": 20}),
            dbc.Spinner(
                html.Div([
                    dbc.Tabs(id="main-tabs", active_tab="tab-preview", children=[
                        dbc.Tab([
                            html.Div(id='data-preview', className="mt-3"),
                        ], label="Data Preview", tab_id="tab-preview"),
                        dbc.Tab([
                            html.Div(id='cluster-summary', className="mt-3"),
                        ], label="Segment Summary", tab_id="tab-summary"),
                        dbc.Tab([
                            html.Div(id='cluster-graphs', className="mt-3"),
                        ], label="Visualizations", tab_id="tab-viz"),
                        dbc.Tab([ # New tab for Word Occurrence
                            html.Div(id='word-occurrence-graphs', className="mt-3"),
                        ], label="Word Occurrence", tab_id="tab-word-occurrence"),
                        dbc.Tab([ # New tab for Apriori results
                            html.Div(id='apriori-results-content', className="mt-3"),
                        ], label="Apriori Results", tab_id="tab-apriori-results"),
                        dbc.Tab([ # New tab for CLV results
                            html.Div(id='clv-results-content', className="mt-3"),
                        ], label="CLV Results", tab_id="tab-clv-results"),
                        dbc.Tab([ # New tab for Recommendation Engine results
                            html.Div(id='recommendation-results-content', className="mt-3"),
                        ], label="Recommendations", tab_id="tab-recommendation-results"),
                        dbc.Tab([ # New tab for Sentiment Analysis results
                            html.Div(id='sentiment-results-content', className="mt-3"),
                        ], label="Sentiment Results", tab_id="tab-sentiment-results"),
                        dbc.Tab([ # New tab for Cohort Analysis results
                            html.Div(id='cohort-results-content', className="mt-3"),
                        ], label="Cohort Results", tab_id="tab-cohort-results"),
                    ]),
                ]),
                size="lg", color="primary", fullscreen=False, id="loading-spinner"
            )
        ], width=9)
    ])
], fluid=True, className="mt-4")

# --- Callbacks ---

# Callback to update the dashboard content based on user interactions
@app.callback(
    Output('data-preview', 'children'),
    Output('cluster-summary', 'children'),
    Output('cluster-dropdown', 'options'),
    Output('error-message', 'children'),
    Output('cluster-graphs', 'children'),
    Output('selected-segment-info', 'children'),
    Output('run-output-message', 'children'), # For feedback after running segmentation
    Input('upload-data', 'contents'),
    Input('run-segmentation', 'n_clicks'),
    Input('cluster-dropdown', 'value'),
    State('upload-data', 'filename'),
    State('analysis-type-radio', 'value'), # New state for analysis type
    # New states for RFM column selections
    State('customer-id-dropdown', 'value'),
    State('invoice-date-dropdown', 'value'),
    State('invoice-dropdown', 'value'),
    State('quantity-dropdown', 'value'),
    State('price-dropdown', 'value'),
    # New state for selected general features
    State('general-features-dropdown', 'value')
)
def update_dashboard(
    contents, run_clicks, selected_cluster, filename, analysis_type,
    customer_col_val, date_col_val, invoice_col_val, quantity_col_val, price_col_val,
    selected_general_features
):
    global df, clustered_data, feature_columns_global, generated_figures
    ctx = dash.callback_context

    # Handle initial load explicitly by returning default empty components
    if not ctx.triggered:
        return (
            html.Div([
                html.H4("Data Preview: No file uploaded"),
                html.P("Please upload a CSV file to begin. Once uploaded, its first 10 rows and a summary will appear here.")
            ]), # data-preview
            html.Div("Segmentation summary will appear here after you run the analysis."), # cluster-summary
            [], # cluster-dropdown options
            "", # error-message
            html.Div("Visualizations will appear here after segmentation."), # cluster-graphs
            "", # selected-segment-info
            "" # run-output-message
        )

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    preview_output = dash.no_update
    cluster_summary_output = dash.no_update
    dropdown_options_output = dash.no_update
    error_message_output = ""
    graph_output = dash.no_update
    segment_info_output = dash.no_update
    run_message_output = "" # Initialize run message

    # Logic for file upload
    if triggered_id == 'upload-data':
        if contents is None:
            raise PreventUpdate
        try:
            content_type, content_string = contents.split(',')
            decoded_bytes = base64.b64decode(content_string) 
            decoded_str = None
            try:
                decoded_str = decoded_bytes.decode('utf-8')
            except UnicodeDecodeError:
                decoded_str = decoded_bytes.decode('latin-1')

            try:
                df = pd.read_csv(io.StringIO(decoded_str))
            except pd.errors.ParserError:
                df = pd.read_csv(io.StringIO(decoded_str), delimiter=';')
            
            # --- New: Detailed Dataset Info ---
            data_info = pd.DataFrame({
                'Column': df.columns,
                'Dtype': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            }).reset_index(drop=True)

            preview_output = html.Div([
                html.H4(f"Data Preview: {filename}"),
                dash_table.DataTable(
                    data=df.head(10).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                    style_cell={'textAlign': 'left'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                ),
                html.H4("Dataset Summary Info:"),
                dash_table.DataTable(
                    data=data_info.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in data_info.columns],
                    style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                    style_cell={'textAlign': 'left'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                ),
                html.Div([
                    html.Strong("Overall Dataset Metrics:"),
                    html.Ul([
                        html.Li(f"Total Rows: {df.shape[0]}"),
                        html.Li(f"Total Columns: {df.shape[1]}"),
                        html.Li(f"Numeric Columns: {len(df.select_dtypes(include=['float64', 'int64']).columns)}"),
                        html.Li(f"Categorical Columns: {len(df.select_dtypes(include=['object', 'category']).columns)}")
                    ])
                ], className="mt-3")
            ])
            # Reset other outputs on new file upload
            cluster_summary_output = html.Div("Upload successful. Click 'Run Segmentation' to proceed.")
            graph_output = html.Div("Visualizations will appear here after segmentation.")
            dropdown_options_output = []
            segment_info_output = ""


        except Exception as e:
            error_message_output = f"Error processing file: {e}"

    # Logic for running segmentation
    elif triggered_id == 'run-segmentation':
        if df.empty:
            error_message_output = "Please upload a CSV file first."
            raise PreventUpdate
        else:
            try:
                time.sleep(1) # Simulate processing time

                if analysis_type == 'rfm_analysis':
                    # Validate RFM column selections
                    required_rfm_cols = [customer_col_val, date_col_val, invoice_col_val, quantity_col_val, price_col_val]
                    if not all(col and col in df.columns for col in required_rfm_cols):
                        error_message_output = "Please select all required RFM columns from the dropdowns."
                        raise PreventUpdate

                    # Perform RFM calculation
                    rfm_df = calculate_rfm(df.copy(), customer_col_val, date_col_val, invoice_col_val, quantity_col_val, price_col_val)
                    
                    # Use RFM features for clustering
                    features_to_cluster = ['Recency', 'Frequency', 'Monetary']
                    
                    if rfm_df.empty or len(rfm_df) < 2:
                        error_message_output = "Not enough data for RFM analysis after cleaning. Please check your data."
                        raise PreventUpdate

                    # Standardize RFM features
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(rfm_df[features_to_cluster])

                    # Apply KMeans on RFM scores
                    n_clusters = min(3, len(rfm_df) - 1)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    rfm_df['Cluster'] = kmeans.fit_predict(features_scaled)
                    
                    clustered_data = rfm_df.copy()
                    feature_columns_global = features_to_cluster
                    run_message_output = dbc.Alert("RFM Analysis and Clustering Completed Successfully!", color="success")

                elif analysis_type == 'general_clustering': # Explicitly handle general clustering
                    if not selected_general_features:
                        error_message_output = "Please select at least two numeric features for general clustering."
                        raise PreventUpdate

                    df_cleaned = df.dropna(subset=selected_general_features) # This line already drops nulls
                    features_to_cluster = [col for col in selected_general_features if col in df_cleaned.select_dtypes(include=['float64', 'int64']).columns]
                    
                    if len(features_to_cluster) < 2:
                        error_message_output = "Need at least 2 valid numeric columns for general clustering from your selection. Please check your data."
                        raise PreventUpdate
                    if len(df_cleaned) < 2:
                        error_message_output = "Not enough data for general clustering after cleaning. Please upload a dataset with more rows or select features with fewer missing values."
                        raise PreventUpdate
                    
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(df_cleaned[features_to_cluster])
                    
                    n_clusters = min(3, len(df_cleaned) - 1)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    df_cleaned['Cluster'] = kmeans.fit_predict(features_scaled)
                    
                    clustered_data = df_cleaned.copy()
                    feature_columns_global = features_to_cluster
                    run_message_output = dbc.Alert("General Clustering Completed Successfully!", color="success")
                else: # If analysis_type is 'word_occurrence', 'apriori_analysis', 'clv_analysis', 'recommendation_engine', 'sentiment_analysis', or 'cohort_analysis', prevent segmentation run
                    error_message_output = "Please select 'General Clustering' or 'RFM Analysis & Clustering' to run segmentation."
                    raise PreventUpdate


                # Prepare dropdown options for clusters
                dropdown_options_output = [{"label": f"Segment {i}", "value": i} for i in sorted(clustered_data['Cluster'].unique())]

                # Calculate cluster statistics
                cluster_stats = clustered_data.groupby('Cluster').size().reset_index(name='Count')
                cluster_stats['Percentage'] = cluster_stats['Count'] / cluster_stats['Count'].sum()
                
                # Prepare cluster summary
                cluster_means = clustered_data.groupby('Cluster')[feature_columns_global].mean().round(2)
                
                # --- RFM Segment Summary (New) ---
                rfm_segment_summary_output = html.Div([]) # Initialize empty
                if analysis_type == 'rfm_analysis' and 'RFM_Segment' in clustered_data.columns:
                    rfm_segment_counts = clustered_data['RFM_Segment'].value_counts().reset_index()
                    rfm_segment_counts.columns = ['RFM_Segment', 'Count']
                    rfm_segment_counts['Percentage'] = rfm_segment_counts['Count'] / rfm_segment_counts['Count'].sum()

                    rfm_segment_profiles = clustered_data.groupby('RFM_Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)

                    rfm_segment_summary_output = html.Div([
                        html.H4("RFM Segment Distribution"),
                        dash_table.DataTable(
                            data=rfm_segment_counts.to_dict('records'),
                            columns=[
                                {"name": "RFM Segment", "id": "RFM_Segment"},
                                {"name": "Count", "id": "Count"},
                                {"name": "Percentage", "id": "Percentage", "type": "numeric", "format": {"specifier": ".1%"}}
                            ],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'center'},
                            style_header={'fontWeight': 'bold'},
                        ),
                        html.Hr(),
                        html.H4("RFM Segment Profiles (Avg. R, F, M)"),
                        dash_table.DataTable(
                            data=rfm_segment_profiles.reset_index().to_dict('records'),
                            columns=[{"name": i, "id": i} for i in rfm_segment_profiles.reset_index().columns],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'center'},
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ],
                            style_header={'fontWeight': 'bold'},
                        ),
                        html.Div([ # New div for the download button
                            dbc.Button([
                                html.I(className="fas fa-download me-2"),
                                "Download Customer List by RFM Segment"
                            ], id="download-rfm-customers-button", color="secondary", className="mt-3 mb-3"),
                            dcc.Download(id="download-rfm-customers-csv"), # New download component
                        ])
                    ])


                cluster_summary_output = html.Div([
                    html.H4("K-Means Segment Distribution"), # Renamed for clarity
                    dash_table.DataTable(
                        data=cluster_stats.to_dict('records'),
                        columns=[
                            {"name": "Segment", "id": "Cluster"},
                            {"name": "Count", "id": "Count"},
                            {"name": "Percentage", "id": "Percentage", "type": "numeric", "format": {"specifier": ".1%"}}
                        ],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center'},
                        style_header={'fontWeight': 'bold'},
                    ),
                    html.Hr(),
                    html.H4("K-Means Segment Profiles"), # Renamed for clarity
                    dash_table.DataTable(
                        data=cluster_means.reset_index().to_dict('records'),
                        columns=[{"name": i, "id": i} for i in cluster_means.reset_index().columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        style_header={'fontWeight': 'bold'},
                    ),
                    html.Hr(), # Separator between K-Means and RFM summaries
                    rfm_segment_summary_output # Add the new RFM segment summary here
                ])
                
                # Generate visualizations, passing analysis_type and original_df
                generated_figures = create_clustering_visualizations(clustered_data, df, feature_columns_global, analysis_type)
                
                # Dynamically build graph_output based on analysis_type
                key_insights_children = [
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=generated_figures[0])], width=6),
                        dbc.Col([dcc.Graph(figure=generated_figures[1])], width=6)
                    ]),
                ]
                # Only add RFM-specific graphs if RFM analysis was performed
                if analysis_type == 'rfm_analysis':
                    key_insights_children.append(
                        dbc.Row([
                            dbc.Col([dcc.Graph(figure=generated_figures[5])], width=12) # RFM Segment Distribution
                        ])
                    )
                    key_insights_children.append(
                        dbc.Row([
                            dbc.Col([dcc.Graph(figure=generated_figures[6])], width=6), # RFM Monetary Range Pie Chart
                            dbc.Col([dcc.Graph(figure=generated_figures[8])], width=6)  # Frequency vs Recency
                        ])
                    )
                    key_insights_children.append(
                        dbc.Row([
                            dbc.Col([dcc.Graph(figure=generated_figures[9])], width=6),  # Frequency vs Monetary
                            dbc.Col([dcc.Graph(figure=generated_figures[10])], width=6) # Monetary vs Recency
                        ])
                    )
                    # Move the 3D RFM plot to the end of RFM-specific graphs
                    key_insights_children.append(
                        dbc.Row([
                            dbc.Col([dcc.Graph(figure=generated_figures[3])], width=12), # PCA/3D RFM
                            dbc.Col(html.Div([
                                html.H5("Understanding PCA (Principal Component Analysis):"),
                                html.P("PCA is a dimensionality reduction technique that transforms a set of correlated variables into a smaller set of uncorrelated variables called principal components (PCs)."),
                                html.Ul([
                                    html.Li("The first principal component (PCA1) accounts for the largest possible variance in the data."),
                                    html.Li("Each subsequent principal component (PCA2, etc.) accounts for the highest remaining variance and is orthogonal to the preceding components."),
                                    html.Li("In this 2D plot, we project your multi-dimensional customer data onto the two principal components that capture the most variance. This helps visualize clusters that might not be apparent in higher dimensions.")
                                ]),
                                html.P("The 'Explained Variance' percentages indicate how much of the total data variability is captured by each principal component shown. Higher percentages mean the components are better at representing the original data.")
                            ], className="mt-3 p-3 bg-light rounded"), width=12)
                        ])
                    )
                elif analysis_type == 'general_clustering':
                    key_insights_children.append(
                        dbc.Row([
                            dbc.Col([dcc.Graph(figure=generated_figures[3])], width=6), # PCA
                            dbc.Col([dcc.Graph(figure=generated_figures[7])], width=6) # NEW: Order Amount (Q*P) Range Pie Chart
                        ])
                    )
                    # Add PCA explanation for general clustering as well
                    key_insights_children.append(
                        dbc.Row([
                            dbc.Col(html.Div([
                                html.H5("Understanding PCA (Principal Component Analysis):"),
                                html.P("PCA is a dimensionality reduction technique that transforms a set of correlated variables into a smaller set of uncorrelated variables called principal components (PCs)."),
                                html.Ul([
                                    html.Li("The first principal component (PCA1) accounts for the largest possible variance in the data."),
                                    html.Li("Each subsequent principal component (PCA2, etc.) accounts for the highest remaining variance and is orthogonal to the preceding components."),
                                    html.Li("In this 2D plot, we project your multi-dimensional customer data onto the two principal components that capture the most variance. This helps visualize clusters that might not be apparent in higher dimensions.")
                                ]),
                                html.P("The 'Explained Variance' percentages indicate how much of the total data variability is captured by each principal component shown. Higher percentages mean the components are better at representing the original data.")
                            ], className="mt-3 p-3 bg-light rounded"), width=12)
                        ])
                    )

                graph_output = html.Div([
                    dbc.Tabs([
                        dbc.Tab(key_insights_children, label="Key Insights"),
                        dbc.Tab([
                            dbc.Row([
                                dbc.Col([dcc.Graph(figure=generated_figures[2])], width=12), # Radar Chart
                            ]),
                            dbc.Row([
                                dbc.Col([dcc.Graph(figure=generated_figures[4])], width=12) # Parallel Coordinates
                            ])
                        ], label="Advanced Analysis")
                    ])
                ])

            except Exception as e:
                error_message_output = f"Segmentation Error: {e}"
                run_message_output = dbc.Alert(f"Segmentation Failed: {e}", color="danger")


    # Logic for filtering by cluster dropdown
    elif triggered_id == 'cluster-dropdown':
        if clustered_data.empty:
            error_message_output = "Please run segmentation first."
        elif selected_cluster is None:
            # Re-display the full data preview with detailed info when filter is cleared
            data_info = pd.DataFrame({
                'Column': df.columns,
                'Dtype': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            }).reset_index(drop=True)

            preview_output = html.Div([
                html.H4("Data Preview: All Customers"),
                dash_table.DataTable(
                    data=df.head(10).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                ),
                html.H4("Dataset Summary Info:"),
                dash_table.DataTable(
                    data=data_info.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in data_info.columns],
                    style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                ),
                html.Div([
                    html.Strong("Overall Dataset Metrics:"),
                    html.Ul([
                        html.Li(f"Total Rows: {df.shape[0]}"),
                        html.Li(f"Total Columns: {df.shape[1]}"),
                        html.Li(f"Numeric Columns: {len(df.select_dtypes(include=['float64', 'int64']).columns)}"),
                        html.Li(f"Categorical Columns: {len(df.select_dtypes(include=['object', 'category']).columns)}")
                    ])
                ], className="mt-3")
            ])
            segment_info_output = "" # Clear segment info
        else:
            filtered = clustered_data[clustered_data['Cluster'] == selected_cluster]
            
            # Columns to display in the filtered preview, including RFM scores/segment if available
            display_cols = [col for col in filtered.columns if col not in ['Recency', 'Frequency', 'Monetary'] or col in feature_columns_global]
            # Ensure CustomerID is always included if it exists in filtered data
            if 'CustomerID' in filtered.columns and 'CustomerID' not in display_cols:
                display_cols.insert(0, 'CustomerID') # Add at the beginning for clarity
            if 'R' in filtered.columns: display_cols.extend(['R', 'F', 'M', 'RFM_Segment'])
            display_cols = list(dict.fromkeys(display_cols)) # Remove duplicates while preserving order

            preview_output = html.Div([
                html.H4(f"K-Means Segment {selected_cluster} Customer Sample"),
                dash_table.DataTable(
                    data=filtered[display_cols].head(10).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in display_cols],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={'fontWeight': 'bold'},
                )
            ])
            
            segment_size = len(filtered)
            segment_percentage = segment_size / len(clustered_data) * 100
            
            segment_info_list = [
                html.Li([
                    html.Strong(f"{col}: "), 
                    f"{filtered[col].mean():.2f} (avg)"
                ]) for col in filtered.select_dtypes(include=['float64', 'int64']).columns if col in feature_columns_global
            ]
            
            # Add RFM scores/segment to info if available
            if 'R' in filtered.columns:
                avg_r = filtered['R'].mean().round(2)
                avg_f = filtered['F'].mean().round(2)
                avg_m = filtered['M'].mean().round(2)
                most_common_rfm_segment = filtered['RFM_Segment'].mode()[0] if not filtered['RFM_Segment'].empty else 'N/A'
                
                segment_info_list.extend([
                    html.Li([html.Strong("Avg. R Score: "), f"{avg_r}"]),
                    html.Li([html.Strong("Avg. F Score: "), f"{avg_f}"]),
                    html.Li([html.Strong("Avg. M Score: "), f"{avg_m}"]),
                    html.Li([html.Strong("Most Common RFM Segment: "), f"{most_common_rfm_segment}"]),
                ])


            segment_info_output = html.Div([
                html.H6(f"K-Means Segment {selected_cluster} Stats:"),
                html.P([
                    html.Strong("Size: "), f"{segment_size} customers ({segment_percentage:.1f}%)"
                ]),
                html.Hr(),
                html.H6("Key Characteristics:"),
                html.Ul(segment_info_list)
            ])

    return preview_output, cluster_summary_output, dropdown_options_output, error_message_output, graph_output, segment_info_output, run_message_output

# Callback to toggle visibility of RFM, General Clustering, Word Occurrence, Apriori, CLV and Recommendation sections
@app.callback(
    [Output('rfm-column-selection', 'style'),
     Output('general-clustering-feature-selection', 'style'),
     Output('word-occurrence-section', 'style'),
     Output('apriori-analysis-section', 'style'),
     Output('clv-column-selection', 'style'), # New output for CLV section
     Output('recommendation-engine-section', 'style'), # New output for Recommendation Engine section
     Output('sentiment-analysis-section', 'style'), # New output for Sentiment Analysis section
     Output('cohort-analysis-section', 'style'), # New output for Cohort Analysis section
     Output('run-segmentation', 'style')], # Adjust visibility of run segmentation button
    Input('analysis-type-radio', 'value')
)
def toggle_analysis_type_sections(analysis_type):
    hide_all = {'display': 'none'}
    show_block = {'display': 'block'}
    show_inline_block = {'display': 'inline-block', 'width': '100%', 'marginTop': '10px', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'padding': '10px 20px'} # Style for run button

    rfm_style = hide_all
    general_style = hide_all
    word_occurrence_style = hide_all
    apriori_style = hide_all
    clv_style = hide_all # New style for CLV
    recommendation_style = hide_all # New style for Recommendation Engine
    sentiment_style = hide_all # New style for Sentiment Analysis
    cohort_style = hide_all # New style for Cohort Analysis
    run_segmentation_button_style = hide_all # Hide by default

    if analysis_type == 'rfm_analysis':
        rfm_style = show_block
        run_segmentation_button_style = show_inline_block
    elif analysis_type == 'general_clustering':
        general_style = show_block
        run_segmentation_button_style = show_inline_block
    elif analysis_type == 'word_occurrence':
        word_occurrence_style = show_block
        run_segmentation_button_style = {'display': 'none'} # Hide segmentation button for word occurrence
    elif analysis_type == 'apriori_analysis':
        apriori_style = show_block
        run_segmentation_button_style = {'display': 'none'} # Hide segmentation button for apriori
    elif analysis_type == 'clv_analysis': # New condition for CLV
        clv_style = show_block
        run_segmentation_button_style = {'display': 'none'} # Hide segmentation button for CLV
    elif analysis_type == 'recommendation_engine': # New condition for Recommendation Engine
        recommendation_style = show_block
        run_segmentation_button_style = {'display': 'none'} # Hide segmentation button for recommendation engine
    elif analysis_type == 'sentiment_analysis': # New condition for Sentiment Analysis
        sentiment_style = show_block
        run_segmentation_button_style = {'display': 'none'} # Hide segmentation button for sentiment analysis
    elif analysis_type == 'cohort_analysis': # New condition for Cohort Analysis
        cohort_style = show_block
        run_segmentation_button_style = {'display': 'none'} # Hide segmentation button for cohort analysis

    return rfm_style, general_style, word_occurrence_style, apriori_style, clv_style, recommendation_style, sentiment_style, cohort_style, run_segmentation_button_style


# Callback to populate column dropdown options after file upload for all sections
@app.callback(
    [Output('customer-id-dropdown', 'options'),
     Output('invoice-date-dropdown', 'options'),
     Output('invoice-dropdown', 'options'),
     Output('quantity-dropdown', 'options'),
     Output('price-dropdown', 'options'),
     Output('general-features-dropdown', 'options'),
     Output('text-column-dropdown', 'options'),
     Output('apriori-transaction-id-dropdown', 'options'),
     Output('apriori-item-id-dropdown', 'options'),
     Output('clv-customer-id-dropdown', 'options'), # New output for CLV
     Output('clv-invoice-date-dropdown', 'options'), # New output for CLV
     Output('clv-invoice-dropdown', 'options'), # New output for CLV
     Output('clv-quantity-dropdown', 'options'), # New output for CLV
     Output('clv-price-dropdown', 'options'), # New output for CLV
     Output('sentiment-text-column-dropdown', 'options'), # New output for Sentiment Analysis
     Output('cohort-customer-id-dropdown', 'options'), # New output for Cohort Analysis
     Output('cohort-order-date-dropdown', 'options')], # New output for Cohort Analysis
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def populate_all_column_options(contents, filename):
    global df # Ensure global df is updated
    if contents:
        content_type, content_string = contents.split(',')
        decoded_bytes = base64.b64decode(content_string) 
        decoded_str = None
        try:
            decoded_str = decoded_bytes.decode('utf-8')
        except UnicodeDecodeError:
            decoded_str = decoded_bytes.decode('latin-1')

        try:
            temp_df = pd.read_csv(io.StringIO(decoded_str))
        except pd.errors.ParserError:
            temp_df = pd.read_csv(io.StringIO(decoded_str), delimiter=';')
        
        df = temp_df.copy() # Update the global df here

        all_columns = [{'label': col, 'value': col} for col in df.columns]
        numeric_columns = [{'label': col, 'value': col} for col in df.select_dtypes(include=['float64', 'int64']).columns]
        # For text columns, we can just provide all object/string columns
        text_columns = [{'label': col, 'value': col} for col in df.select_dtypes(include=['object', 'string']).columns]


        # Return the same options for all RFM, Apriori, and CLV dropdowns, and the numeric options for general features
        return (
            all_columns, all_columns, all_columns, all_columns, all_columns, # RFM
            numeric_columns, # General Clustering
            text_columns, # Word Occurrence
            all_columns, all_columns, # Apriori
            all_columns, all_columns, all_columns, all_columns, all_columns, # CLV
            text_columns, # Sentiment Analysis
            all_columns, all_columns # Cohort Analysis
        )
    return [[] for _ in range(17)] # Return empty lists if no file is uploaded (15 existing + 2 new for Cohort)


# Callback for showing upload success toast
@app.callback(
    Output('upload-toast', 'is_open'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def show_upload_success(contents):
    return bool(contents)

# Callback for downloading the PDF report
@app.callback(
    Output("download-pdf-report", "data"),
    Input("download-report-button", "n_clicks"),
    State('analysis-type-radio', 'value'), # Get current analysis type
    prevent_initial_call=True
)
def download_report(n_clicks, analysis_type):
    if n_clicks is None:
        raise PreventUpdate
    
    # Only allow PDF download for clustering analyses
    if analysis_type not in ['general_clustering', 'rfm_analysis']:
        print("PDF report is only available for General Clustering or RFM Analysis.")
        raise PreventUpdate

    if clustered_data.empty or not feature_columns_global or not generated_figures:
        print("No data or visualizations available to generate report. Please upload data and run segmentation first.")
        raise PreventUpdate
    
    pdf_bytes = generate_segment_report(clustered_data, feature_columns_global, generated_figures)
    
    return dcc.send_bytes(pdf_bytes, "customer_segmentation_report.pdf")

# New callback for downloading RFM customer list
@app.callback(
    Output("download-rfm-customers-csv", "data"),
    Input("download-rfm-customers-button", "n_clicks"),
    prevent_initial_call=True
)
def download_rfm_customers(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    
    # Ensure RFM analysis was performed and data is available
    if 'RFM_Segment' not in clustered_data.columns or clustered_data.empty:
        # In a real app, you might want to show a toast/error message here
        print("RFM analysis not performed or data not available for download.")
        return None # Return None to prevent download
    
    # Explicitly check for 'CustomerID' before proceeding
    if 'CustomerID' not in clustered_data.columns:
        print("CustomerID column not found in clustered data. Cannot generate RFM customer list.")
        return None # Return None to prevent download

    # Get unique RFM segments
    rfm_segments = clustered_data['RFM_Segment'].unique()
    
    # Create a dictionary to hold customer IDs for each segment
    segment_customer_ids = {segment: [] for segment in rfm_segments}
    
    # Populate the dictionary
    for _, row in clustered_data.iterrows():
        segment_customer_ids[row['RFM_Segment']].append(row['CustomerID'])
        
    # Find the maximum length among all segment lists to pad shorter lists
    max_len = max(len(ids) for ids in segment_customer_ids.values()) if segment_customer_ids else 0
    
    # Create a dictionary for the new DataFrame, padding shorter lists with None
    padded_data = {
        segment: ids + [None] * (max_len - len(ids))
        for segment, ids in segment_customer_ids.items()
    }
    
    # Create the DataFrame
    customers_for_download_df = pd.DataFrame(padded_data)
    
    # Convert to CSV string
    csv_string = customers_for_download_df.to_csv(index=False, encoding='utf-8')
    
    # Return as a downloadable file
    return dcc.send_string(csv_string, "rfm_segmented_customers_column_wise.csv")

# Callback for Word Occurrence Analysis
@app.callback(
    Output('word-occurrence-graphs', 'children'),
    Output('main-tabs', 'active_tab', allow_duplicate=True), # To switch to the Word Occurrence tab
    Input('run-word-occurrence', 'n_clicks'),
    State('text-column-dropdown', 'value'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_word_occurrence_graph(n_clicks, text_column, contents):
    if n_clicks is None or text_column is None or contents is None:
        raise PreventUpdate

    if not contents:
        return html.Div("Please upload a CSV file first."), "tab-word-occurrence"

    if not text_column:
        return html.Div("Please select a text column for analysis."), "tab-word-occurrence"

    try:
        content_type, content_string = contents.split(',')
        decoded_bytes = base64.b64decode(content_string) 
        decoded_str = None
        try:
            decoded_str = decoded_bytes.decode('utf-8')
        except UnicodeDecodeError:
            decoded_str = decoded_bytes.decode('latin-1')

        try:
            temp_df = pd.read_csv(io.StringIO(decoded_str))
        except pd.errors.ParserError:
            temp_df = pd.read_csv(io.StringIO(decoded_str), delimiter=';')
        
        if text_column not in temp_df.columns:
            return html.Div(f"Selected column '{text_column}' not found in the uploaded data."), "tab-word-occurrence"

        # Text cleaning and word frequency calculation
        text_data = temp_df[text_column].dropna().astype(str)
        
        if text_data.empty:
            return html.Div(f"No valid text data found in column '{text_column}' for analysis."), "tab-word-occurrence"

        # Combine all text into a single string
        all_text = " ".join(text_data.tolist())
        
        # Remove punctuation and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', '', all_text).lower()
        
        # Tokenize words
        words = word_tokenize(cleaned_text)
        
        # Remove stop words and single-character words
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 1]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Get top N words (e.g., top 20) for the first graph
        top_20_words = word_counts.most_common(20)
        
        # Get all words (or top 50 for readability) for the second horizontal graph
        # Limiting to top 50 for better visualization, as plotting all unique words can be overwhelming
        all_words_for_plot = word_counts.most_common(50) 

        if not top_20_words and not all_words_for_plot:
            return html.Div("No significant words found after cleaning and filtering."), "tab-word-occurrence"

        # Figure for Top 20 Words (vertical bar chart)
        fig_top_20 = go.Figure()
        if top_20_words:
            words_df_top_20 = pd.DataFrame(top_20_words, columns=['Word', 'Frequency'])
            fig_top_20 = px.bar(words_df_top_20, x='Word', y='Frequency',
                                title=f'Top 20 Word Frequencies in "{text_column}"',
                                labels={'Word': 'Word', 'Frequency': 'Frequency'},
                                color='Frequency', color_continuous_scale=px.colors.sequential.Viridis)
            fig_top_20.update_layout(xaxis={'categoryorder':'total descending'}) # Sort bars in decreasing order
        else:
            fig_top_20.add_annotation(text="No top 20 words found.", xref="paper", yref="paper", showarrow=False)
            fig_top_20.update_layout(title_text=f'Top 20 Word Frequencies in "{text_column}"')

        # Figure for All Word Frequencies (horizontal bar chart, limited to top 50)
        fig_all_words = go.Figure()
        if all_words_for_plot:
            words_df_all = pd.DataFrame(all_words_for_plot, columns=['Word', 'Frequency'])
            fig_all_words = px.bar(words_df_all, x='Frequency', y='Word', orientation='h',
                                   title=f'Word Frequencies in "{text_column}" (Top {len(all_words_for_plot)})',
                                   labels={'Word': 'Word', 'Frequency': 'Frequency'},
                                   color='Frequency', color_continuous_scale=px.colors.sequential.Viridis)
            # For horizontal bar charts, 'total ascending' means the longest bar is at the top
            fig_all_words.update_layout(yaxis={'categoryorder':'total ascending'}) 
        else:
            fig_all_words.add_annotation(text="No words found for frequency analysis.", xref="paper", yref="paper", showarrow=False)
            fig_all_words.update_layout(title_text=f'Word Frequencies in "{text_column}"')


        # New section: How trending words can boost sales and drive business decisions
        business_insights_section = html.Div([
            html.H4("Leveraging Trending Words for Business Growth"),
            html.P("Understanding trending words from customer feedback, product descriptions, or reviews can provide invaluable insights for boosting sales and driving strategic business decisions:"),
            html.Ul([
                html.Li([
                    html.Strong("Product Development & Enhancement: "),
                    "Identify gaps in your product offerings or areas for improvement by noting frequently requested features or common complaints. Trending words can highlight emerging market needs."
                ]),
                html.Li([
                    html.Strong("Marketing & Advertising Optimization: "),
                    "Incorporate popular keywords and phrases into your marketing campaigns, ad copy, and SEO strategies to resonate more effectively with your target audience and improve search visibility."
                ]),
                html.Li([
                    html.Strong("Content Strategy: "),
                    "Create engaging content (blog posts, social media, articles) around trending topics or words that customers are actively searching for or discussing. This drives organic traffic and establishes thought leadership."
                ]),
                html.Li([
                    html.Strong("Customer Service Improvement: "),
                    "Analyze trending issues or questions from support tickets to proactively address common customer pain points, develop FAQs, or train support staff on recurring themes."
                ]),
                html.Li([
                    html.Strong("Sales Forecasting & Inventory Management: "),
                    "Anticipate demand for certain products or features by tracking the rise of related keywords. This can inform inventory decisions and prevent stockouts or overstocking."
                ]),
                html.Li([
                    html.Strong("Competitive Analysis: "),
                    "Monitor trending words associated with competitors to understand their strengths, weaknesses, and customer perceptions, allowing you to refine your own positioning."
                ])
            ], className="mb-4")
        ], className="mt-4")


        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_top_20), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_all_words), width=12)
            ]),
            business_insights_section # Add the new section here
        ]), "tab-word-occurrence"

    except Exception as e:
        return html.Div(f"Error performing word occurrence analysis: {e}"), "tab-word-occurrence"

# New Callback for Apriori Analysis
@app.callback(
    Output('apriori-results-content', 'children'),
    Output('main-tabs', 'active_tab', allow_duplicate=True), # To switch to the Apriori Results tab
    Output('recommendation-item-dropdown', 'options'), # Output to populate recommendation dropdown
    Input('run-apriori', 'n_clicks'),
    State('apriori-transaction-id-dropdown', 'value'),
    State('apriori-item-id-dropdown', 'value'),
    State('min-support-input', 'value'),
    State('min-confidence-input', 'value'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def run_apriori_analysis(n_clicks, transaction_col, item_col, min_support, min_confidence, contents):
    global rules_df # Make rules_df global
    if n_clicks is None:
        raise PreventUpdate

    if not contents:
        return html.Div("Please upload a CSV file first."), "tab-apriori-results", dash.no_update

    if not transaction_col or not item_col:
        return html.Div("Please select both Transaction ID and Item ID columns for Apriori analysis."), "tab-apriori-results", dash.no_update

    try:
        content_type, content_string = contents.split(',')
        decoded_bytes = base64.b64decode(content_string) 
        decoded_str = None
        try:
            decoded_str = decoded_bytes.decode('utf-8')
        except UnicodeDecodeError:
            decoded_str = decoded_bytes.decode('latin-1')

        try:
            temp_df = pd.read_csv(io.StringIO(decoded_str))
        except pd.errors.ParserError:
            temp_df = pd.read_csv(io.StringIO(decoded_str), delimiter=';')
        
        if transaction_col not in temp_df.columns or item_col not in temp_df.columns:
            return html.Div(f"Selected columns '{transaction_col}' or '{item_col}' not found in the uploaded data."), "tab-apriori-results", dash.no_update

        # Prepare data for Apriori
        # Drop rows with missing values in the relevant columns
        transactions_df = temp_df.dropna(subset=[transaction_col, item_col])
        
        if transactions_df.empty:
            return html.Div("No valid data for Apriori analysis after dropping missing values."), "tab-apriori-results", dash.no_update

        # Group items by transaction ID
        transactions = transactions_df.groupby(transaction_col)[item_col].apply(list).tolist()

        # Use TransactionEncoder to one-hot encode the transactions
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        # Run Apriori algorithm
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            return html.Div(f"No frequent itemsets found with minimum support of {min_support}. Try lowering the support value."), "tab-apriori-results", dash.no_update

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        # Sort rules by lift for better insights
        rules = rules.sort_values(by='lift', ascending=False)
        rules_df = rules.copy() # Store rules globally

        # Extract all unique items for the recommendation dropdown
        all_items = set()
        for itemset in frequent_itemsets['itemsets']:
            for item in itemset:
                all_items.add(item)
        recommendation_dropdown_options = [{'label': item, 'value': item} for item in sorted(list(all_items))]


        # Format itemsets for display
        frequent_itemsets_display = frequent_itemsets.copy()
        frequent_itemsets_display['itemsets'] = frequent_itemsets_display['itemsets'].apply(lambda x: ', '.join(list(x)))
        
        rules_display = rules.copy()
        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))


        # --- Network Graph Visualization ---
        nodes = set()
        for _, row in rules.iterrows():
            # Add all antecedents and consequents as nodes
            for item in row['antecedents']:
                nodes.add(item)
            for item in row['consequents']:
                nodes.add(item)
        nodes = list(nodes)

        # Assign positions for nodes (e.g., in a circle)
        num_nodes = len(nodes)
        node_positions = {node: (np.cos(2 * np.pi * i / num_nodes), np.sin(2 * np.pi * i / num_nodes))
                          for i, node in enumerate(nodes)}

        # Create a list to hold all edge traces
        edge_traces = []

        # Iterate through rules to create individual edge traces
        for _, rule in rules.iterrows():
            # For network graph, we typically visualize A -> B, so we take the first item
            # from antecedents and consequents for simplicity.
            # For more complex antecedents/consequents (multiple items),
            # this might need more sophisticated visualization (e.g., hypergraphs or different layout).
            if len(rule['antecedents']) > 0 and len(rule['consequents']) > 0:
                from_node = next(iter(rule['antecedents']))
                to_node = next(iter(rule['consequents']))

                x0, y0 = node_positions[from_node]
                x1, y1 = node_positions[to_node]

                # Normalize lift for line width, handle potential division by zero
                min_lift = rules['lift'].min()
                max_lift = rules['lift'].max()
                
                if max_lift == min_lift: # Avoid division by zero if all lifts are the same
                    lift_norm = 0.5 # Default to medium thickness
                else:
                    lift_norm = (rule['lift'] - min_lift) / (max_lift - min_lift)
                
                line_width = 1 + lift_norm * 4 # Min width 1, max 5

                edge_trace = go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    line=dict(width=line_width, color='#888'),
                    hoverinfo='text',
                    text=f"Rule: {', '.join(list(rule['antecedents']))} -> {', '.join(list(rule['consequents']))}<br>Confidence: {rule['confidence']:.2f}<br>Lift: {rule['lift']:.2f}",
                    mode='lines',
                    showlegend=False
                )
                edge_traces.append(edge_trace)

        # Create node trace
        node_x = [pos[0] for pos in node_positions.values()]
        node_y = [pos[1] for pos in node_positions.values()]
        node_text = list(nodes)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                size=15,
                color='skyblue',
                line_width=2
            ),
            textposition="top center"
        )

        # Combine all traces for the figure
        all_traces = edge_traces + [node_trace]

        fig_network = go.Figure(data=all_traces,
                                layout=go.Layout(
                                    title='Network Graph of Association Rules (Lift as Edge Thickness)',
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20,l=5,r=5,t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                ))
        
        # Display results in Dash tables and graph
        apriori_content = [
            html.H4("Frequent Itemsets"),
            dash_table.DataTable(
                id='frequent-itemsets-table',
                columns=[{"name": i, "id": i} for i in frequent_itemsets_display.columns],
                data=frequent_itemsets_display.to_dict('records'),
                style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                page_size=10,
            ),
            html.H4("Association Rules"),
            dash_table.DataTable(
                id='association-rules-table',
                columns=[{"name": i, "id": i} for i in rules_display.columns],
                data=rules_display.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                page_size=10,
            ),
            dcc.Graph(figure=fig_network), # Add the network graph
            html.Div([
                html.H5("Understanding the Metrics:"),
                html.P([
                    html.Strong("Support: "),
                    "Indicates how frequently an itemset appears in the dataset. A higher support value means the itemset is more common."
                ]),
                html.P([
                    html.Strong("Confidence: "),
                    "Measures how often items in the consequent (RHS) appear in transactions that already contain the antecedent (LHS). It's the conditional probability P(RHS | LHS)."
                ]),
                html.P([
                    html.Strong("Lift: "),
                    "Indicates how much more likely the consequent is to be purchased when the antecedent is purchased, compared to when the consequent is purchased independently. A lift value > 1 suggests a positive association, < 1 suggests a negative association, and = 1 suggests no association."
                ]),
                html.H5("How to interpret the Network Graph:"),
                html.Ul([
                    html.Li("Each circle represents an item found in your transactions."),
                    html.Li("An arrow from Item A to Item B indicates an association rule: 'If A is bought, then B is also likely to be bought' (A -> B)."),
                    html.Li("The thickness of the arrow represents the 'Lift' of the rule. Thicker arrows indicate stronger, more interesting associations. A higher lift means the co-occurrence is less likely to be by chance.")
                ])
            ], className="mt-4 mb-4"),
            html.Div([
                html.H4("Conclusions from Association Rules"),
                html.P(f"Based on the Apriori analysis with a minimum support of {min_support:.2f} and minimum confidence of {min_confidence:.2f}, the following insights can be derived:"),
                html.Ul([
                    html.Li(f"A total of {len(frequent_itemsets)} frequent itemsets were identified."),
                    html.Li(f"A total of {len(rules)} association rules were generated."),
                    html.Li("Key insights from the top association rules (sorted by Lift):")
                ] + [
                    html.Li(f"Customers who bought '{rule['antecedents']}' are likely to also buy '{rule['consequents']}' (Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}). This suggests a strong purchasing pattern.")
                    for index, rule in rules.head(5).iterrows()
                ] if not rules.empty else [html.Li("No strong association rules found with the given thresholds. Consider adjusting minimum support or confidence.")])
            ])
        ]


        return html.Div(apriori_content), "tab-apriori-results", recommendation_dropdown_options

    except Exception as e:
        return html.Div(f"Error performing Apriori analysis: {e}"), "tab-apriori-results", dash.no_update

# New Callback for CLV Analysis
@app.callback(
    Output('clv-results-content', 'children'),
    Output('main-tabs', 'active_tab', allow_duplicate=True), # To switch to the CLV Results tab
    Input('run-clv-analysis', 'n_clicks'),
    State('clv-customer-id-dropdown', 'value'),
    State('clv-invoice-date-dropdown', 'value'),
    State('clv-invoice-dropdown', 'value'),
    State('clv-quantity-dropdown', 'value'),
    State('clv-price-dropdown', 'value'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def run_clv_analysis(n_clicks, customer_col, date_col, invoice_col, quantity_col, price_col, contents):
    global clv_data
    if n_clicks is None:
        raise PreventUpdate

    if not contents:
        return html.Div("Please upload a CSV file first."), "tab-clv-results"

    required_cols = [customer_col, date_col, invoice_col, quantity_col, price_col]
    if not all(col for col in required_cols):
        return html.Div("Please select all required columns for CLV analysis."), "tab-clv-results"

    try:
        content_type, content_string = contents.split(',')
        decoded_bytes = base64.b64decode(content_string) 
        decoded_str = None
        try:
            decoded_str = decoded_bytes.decode('utf-8')
        except UnicodeDecodeError:
            decoded_str = decoded_bytes.decode('latin-1')

        try:
            temp_df = pd.read_csv(io.StringIO(decoded_str))
        except pd.errors.ParserError:
            temp_df = pd.read_csv(io.StringIO(decoded_str), delimiter=';')
        
        if not all(col in temp_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in temp_df.columns]
            return html.Div(f"Missing required columns in uploaded data: {', '.join(missing)}"), "tab-clv-results"

        clv_data = calculate_clv(temp_df.copy(), customer_col, date_col, invoice_col, quantity_col, price_col)
        
        if clv_data.empty:
            return html.Div("No valid data for CLV analysis after cleaning. Please check your data."), "tab-clv-results"

        # --- CLV Visualizations ---
        clv_figures = []

        # 1. Histogram of CLV Distribution
        fig_clv_dist = px.histogram(clv_data, x='CLV', nbins=50,
                                    title='Distribution of Customer Lifetime Value (CLV)',
                                    labels={'CLV': 'Customer Lifetime Value'},
                                    color_discrete_sequence=px.colors.qualitative.Plotly)
        fig_clv_dist.update_layout(xaxis_title="CLV ($)", yaxis_title="Number of Customers")
        clv_figures.append(fig_clv_dist)

        # 2. Pie Chart of CLV Segments
        if 'CLV_Segment' in clv_data.columns and not clv_data['CLV_Segment'].empty:
            fig_clv_segment_pie = px.pie(clv_data, names='CLV_Segment',
                                         title='Distribution of Customers by CLV Segment',
                                         color='CLV_Segment',
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_clv_segment_pie.update_traces(textinfo='percent+label', pull=[0.05]*len(clv_data['CLV_Segment'].unique()))
            clv_figures.append(fig_clv_segment_pie)
        else:
            fig_clv_segment_pie = go.Figure().add_annotation(text="CLV Segment distribution not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=16))
            fig_clv_segment_pie.update_layout(title_text='Distribution of Customers by CLV Segment')
            clv_figures.append(fig_clv_segment_pie)


        # 3. Bar Chart: Average CLV per CLV Segment
        if 'CLV_Segment' in clv_data.columns and not clv_data['CLV_Segment'].empty:
            avg_clv_per_segment = clv_data.groupby('CLV_Segment')['CLV'].mean().reset_index()
            fig_avg_clv_segment = px.bar(avg_clv_per_segment, x='CLV_Segment', y='CLV',
                                        title='Average CLV per CLV Segment',
                                        labels={'CLV': 'Average CLV ($)', 'CLV_Segment': 'CLV Segment'},
                                        color='CLV_Segment', color_discrete_sequence=px.colors.qualitative.Vivid)
            clv_figures.append(fig_avg_clv_segment)
        else:
            fig_avg_clv_segment = go.Figure().add_annotation(text="Average CLV per segment not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=16))
            fig_avg_clv_segment.update_layout(title_text='Average CLV per CLV Segment')
            clv_figures.append(fig_avg_clv_segment)


        # 4. Scatter Plot: PurchaseCount vs. TotalRevenue (colored by CLV Segment)
        if 'CLV_Segment' in clv_data.columns and not clv_data['CLV_Segment'].empty:
            fig_purchase_revenue = px.scatter(clv_data, x='PurchaseCount', y='TotalRevenue', color='CLV_Segment',
                                            title='Purchase Count vs. Total Revenue by CLV Segment',
                                            labels={'PurchaseCount': 'Number of Purchases', 'TotalRevenue': 'Total Revenue ($)'},
                                            hover_data=['CustomerID', 'CLV'],
                                            color_discrete_sequence=px.colors.qualitative.Bold)
            clv_figures.append(fig_purchase_revenue)
        else:
            fig_purchase_revenue = go.Figure().add_annotation(text="Purchase Count vs. Total Revenue not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=16))
            fig_purchase_revenue.update_layout(title_text='Purchase Count vs. Total Revenue by CLV Segment')
            clv_figures.append(fig_purchase_revenue)


        clv_content = [
            html.H4("Customer Lifetime Value (CLV) Summary"),
            html.P("Customer Lifetime Value (CLV) represents the total revenue a business can reasonably expect from a single customer account throughout their relationship. Understanding CLV helps in making strategic decisions about marketing, sales, product development, and customer support."),
            html.Hr(),
            html.H5("CLV Metrics Table (Sample)"),
            dash_table.DataTable(
                id='clv-data-table',
                columns=[{"name": col, "id": col, "type": "numeric", "format": {"specifier": ".2f"}} if clv_data[col].dtype in ['float64', 'int64'] else {"name": col, "id": col} for col in clv_data.columns],
                data=clv_data.round(2).head(10).to_dict('records'),
                style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                page_size=10,
            ),
            html.Div([ # New div for the download button
                dbc.Button([
                    html.I(className="fas fa-download me-2"),
                    "Download Customer List by CLV Segment"
                ], id="download-clv-customers-button", color="secondary", className="mt-3 mb-3"),
                dcc.Download(id="download-clv-customers-csv"), # New download component
            ]),
            html.Div([ # New div for the download button for full CLV data
                dbc.Button([
                    html.I(className="fas fa-file-csv me-2"),
                    "Download Full CLV Metrics Table"
                ], id="download-full-clv-data-button", color="secondary", className="mt-3 mb-3"),
                dcc.Download(id="download-full-clv-data-csv"), # New download component
            ]),
            html.H5("CLV Visualizations"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=clv_figures[0]), width=6), # CLV Distribution
                dbc.Col(dcc.Graph(figure=clv_figures[1]), width=6)  # CLV Segment Pie
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=clv_figures[2]), width=6), # Avg CLV per Segment
                dbc.Col(dcc.Graph(figure=clv_figures[3]), width=6)  # Purchase Count vs Total Revenue
            ]),
            html.Div([
                html.H5("Interpreting CLV Results:"),
                html.Ul([
                    html.Li([
                        html.Strong("High CLV Customers: "),
                        "These are your most valuable customers. Focus on retention strategies, loyalty programs, and personalized upselling/cross-selling to maximize their value."
                    ]),
                    html.Li([
                        html.Strong("Low CLV Customers: "),
                        "Identify reasons for low CLV. Is it due to low purchase frequency, low average order value, or short customer lifespan? Develop strategies to increase their engagement or average spend."
                    ]),
                    html.Li([
                        html.Strong("CLV Segments: "),
                        "The segmentation helps in tailoring marketing campaigns. For example, 'Very High CLV' customers might respond well to exclusive offers, while 'Low CLV' customers might need re-engagement campaigns."
                    ]),
                    html.Li([
                        html.Strong("Data Trends: "),
                        "Observe the distribution of CLV. Is it skewed? Are there distinct groups? This can inform your overall customer strategy."
                    ])
                ])
            ], className="mt-4 mb-4")
        ]

        return html.Div(clv_content), "tab-clv-results"

    except Exception as e:
        return html.Div(f"Error performing CLV analysis: {e}"), "tab-clv-results"

# New callback for downloading CLV customer list
@app.callback(
    Output("download-clv-customers-csv", "data"),
    Input("download-clv-customers-button", "n_clicks"),
    prevent_initial_call=True
)
def download_clv_customers(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    
    # Ensure CLV analysis was performed and data is available
    if 'CLV_Segment' not in clv_data.columns or clv_data.empty:
        print("CLV analysis not performed or data not available for download.")
        return None # Return None to prevent download
    
    # Explicitly check for 'CustomerID' before proceeding
    if 'CustomerID' not in clv_data.columns:
        print("CustomerID column not found in CLV data. Cannot generate CLV customer list.")
        return None # Return None to prevent download

    # Get unique CLV segments
    clv_segments = clv_data['CLV_Segment'].unique()
    
    # Create a dictionary to hold customer IDs for each segment
    segment_customer_ids = {segment: [] for segment in clv_segments}
    
    # Populate the dictionary
    for _, row in clv_data.iterrows():
        segment_customer_ids[row['CLV_Segment']].append(row['CustomerID'])
        
    # Find the maximum length among all segment lists to pad shorter lists
    max_len = max(len(ids) for ids in segment_customer_ids.values()) if segment_customer_ids else 0
    
    # Create a dictionary for the new DataFrame, padding shorter lists with None
    padded_data = {
        segment: ids + [None] * (max_len - len(ids))
        for segment, ids in segment_customer_ids.items()
    }
    
    # Create the DataFrame
    customers_for_download_df = pd.DataFrame(padded_data)
    
    # Convert to CSV string
    csv_string = customers_for_download_df.to_csv(index=False, encoding='utf-8')
    
    # Return as a downloadable file
    return dcc.send_string(csv_string, "clv_segmented_customers_column_wise.csv")

# New callback for downloading the full CLV metrics table
@app.callback(
    Output("download-full-clv-data-csv", "data"),
    Input("download-full-clv-data-button", "n_clicks"),
    prevent_initial_call=True
)
def download_full_clv_data(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    
    if clv_data.empty:
        print("No CLV data available to download.")
        return None
    
    csv_string = clv_data.to_csv(index=False, encoding='utf-8')
    return dcc.send_string(csv_string, "full_clv_metrics_table.csv")


# Callback for Recommendation Engine
@app.callback(
    Output('recommendation-results-content', 'children'),
    Output('main-tabs', 'active_tab', allow_duplicate=True), # To switch to the Recommendations tab
    Input('run-recommendation-engine', 'n_clicks'),
    State('recommendation-item-dropdown', 'value'),
    prevent_initial_call=True
)
def get_recommendations(n_clicks, selected_item):
    if n_clicks is None:
        raise PreventUpdate

    if rules_df.empty:
        return html.Div("Please run Apriori Analysis first to generate association rules."), "tab-recommendation-results"

    if not selected_item:
        return html.Div("Please select an item from the dropdown to get recommendations."), "tab-recommendation-results"

    # Filter rules where the selected item is an antecedent
    # We need to check if the selected_item is present in the frozenset of antecedents
    # Convert frozensets to lists of strings for comparison
    relevant_rules = rules_df[rules_df['antecedents'].apply(lambda x: selected_item in x)]
    
    if relevant_rules.empty:
        return html.Div(f"No recommendations found for '{selected_item}'. Try selecting a different item or adjusting Apriori parameters."), "tab-recommendation-results"

    # Sort by confidence and then lift for best recommendations
    relevant_rules = relevant_rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])

    # Format antecedents and consequents for display
    recommendations_display = relevant_rules.copy()
    recommendations_display['antecedents'] = recommendations_display['antecedents'].apply(lambda x: ', '.join(list(x)))
    recommendations_display['consequents'] = recommendations_display['consequents'].apply(lambda x: ', '.join(list(x)))

    recommendation_content = [
        html.H4(f"Recommendations for Customers Who Bought: '{selected_item}'"),
        html.P("Based on the Apriori analysis, here are items frequently purchased together:"),
        dash_table.DataTable(
            id='recommendations-table',
            columns=[
                {"name": "If Customer Bought", "id": "antecedents"},
                {"name": "Then Also Bought", "id": "consequents"},
                {"name": "Confidence", "id": "confidence", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "Lift", "id": "lift", "type": "numeric", "format": {"specifier": ".2f"}}
            ],
            data=recommendations_display.to_dict('records'),
            style_table={'overflowX': 'auto', 'marginBottom': '20px'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            page_size=10,
        ),
        html.Div([
            html.H5("How to Use These Recommendations:"),
            html.Ul([
                html.Li([
                    html.Strong("Cross-Selling: "),
                    "Suggest 'consequent' items to customers who have 'antecedent' items in their cart or purchase history."
                ]),
                html.Li([
                    html.Strong("Product Bundling: "),
                    "Create bundles of frequently co-purchased items to increase average order value."
                ]),
                html.Li([
                    html.Strong("Store Layout Optimization: "),
                    "Arrange products that are frequently bought together closer in physical stores or on e-commerce platforms."
                ]),
                html.Li([
                    html.Strong("Targeted Marketing: "),
                    "Design marketing campaigns that highlight these product relationships to relevant customer segments."
                ])
            ])
        ], className="mt-4 mb-4")
    ]

    return html.Div(recommendation_content), "tab-recommendation-results"

# New Callback for Sentiment Analysis
@app.callback(
    Output('sentiment-results-content', 'children'),
    Output('main-tabs', 'active_tab', allow_duplicate=True), # To switch to the Sentiment Results tab
    Input('run-sentiment-analysis', 'n_clicks'),
    State('sentiment-text-column-dropdown', 'value'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def run_sentiment_analysis(n_clicks, text_column, contents):
    if n_clicks is None:
        raise PreventUpdate

    if not contents:
        return html.Div("Please upload a CSV file first."), "tab-sentiment-results"

    if not text_column:
        return html.Div("Please select a text column for sentiment analysis."), "tab-sentiment-results"

    try:
        content_type, content_string = contents.split(',')
        decoded_bytes = base64.b64decode(content_string) 
        decoded_str = None
        try:
            decoded_str = decoded_bytes.decode('utf-8')
        except UnicodeDecodeError:
            decoded_str = decoded_bytes.decode('latin-1')

        try:
            temp_df = pd.read_csv(io.StringIO(decoded_str))
        except pd.errors.ParserError:
            temp_df = pd.read_csv(io.StringIO(decoded_str), delimiter=';')
        
        if text_column not in temp_df.columns:
            return html.Div(f"Selected column '{text_column}' not found in the uploaded data."), "tab-sentiment-results"

        # Create a DataFrame for sentiment analysis results
        sentiment_df = temp_df[[text_column]].copy()
        sentiment_df.rename(columns={text_column: 'Text'}, inplace=True)
        sentiment_df['Sentiment'] = sentiment_df['Text'].apply(analyze_sentiment)
        
        # Drop rows where text was NaN and resulted in 'Neutral' if that's not desired
        sentiment_df = sentiment_df.dropna(subset=['Text'])

        if sentiment_df.empty:
            return html.Div(f"No valid text data found in column '{text_column}' for sentiment analysis after cleaning."), "tab-sentiment-results"

        # Calculate sentiment distribution
        sentiment_counts = sentiment_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        sentiment_counts['Percentage'] = sentiment_counts['Count'] / sentiment_counts['Count'].sum()

        # Create Pie Chart for Sentiment Distribution
        fig_sentiment_pie = px.pie(sentiment_counts, names='Sentiment', values='Count',
                                   title='Overall Sentiment Distribution',
                                   color='Sentiment',
                                   color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'})
        fig_sentiment_pie.update_traces(textinfo='percent+label', pull=[0.05]*len(sentiment_counts['Sentiment'].unique()))

        sentiment_content = [
            html.H4("Sentiment Analysis Results"),
            html.P(f"Analysis performed on the '{text_column}' column."),
            html.Hr(),
            html.H5("Sentiment Distribution Summary"),
            dash_table.DataTable(
                id='sentiment-summary-table',
                columns=[
                    {"name": "Sentiment", "id": "Sentiment"},
                    {"name": "Count", "id": "Count"},
                    {"name": "Percentage", "id": "Percentage", "type": "numeric", "format": {"specifier": ".1%"}}
                ],
                data=sentiment_counts.to_dict('records'),
                style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            ),
            dcc.Graph(figure=fig_sentiment_pie),
            html.H5("Sample of Analyzed Text with Sentiment"),
            dash_table.DataTable(
                id='sentiment-sample-table',
                columns=[
                    {"name": "Original Text", "id": "Text"},
                    {"name": "Assigned Sentiment", "id": "Sentiment"}
                ],
                data=sentiment_df[['Text', 'Sentiment']].head(20).to_dict('records'), # Show first 20 samples
                style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'minWidth': '150px', 'maxWidth': '300px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                page_size=10,
            ),
            html.Div([
                html.H5("Leveraging Sentiment Analysis for Business:"),
                html.Ul([
                    html.Li([
                        html.Strong("Customer Satisfaction: "),
                        "Quickly gauge overall customer satisfaction from reviews and feedback."
                    ]),
                    html.Li([
                        html.Strong("Product/Service Improvement: "),
                        "Identify specific features or aspects that generate strong positive or negative reactions, guiding product development."
                    ]),
                    html.Li([
                        html.Strong("Brand Monitoring: "),
                        "Track public perception of your brand over time and react to negative trends promptly."
                    ]),
                    html.Li([
                        html.Strong("Targeted Marketing: "),
                        "Understand customer preferences and pain points to tailor marketing messages more effectively."
                    ]),
                    html.Li([
                        html.Strong("Early Warning System: "),
                        "Detect emerging issues or crises by monitoring a sudden increase in negative sentiment."
                    ])
                ])
            ], className="mt-4 mb-4")
        ]

        return html.Div(sentiment_content), "tab-sentiment-results"

    except Exception as e:
        return html.Div(f"Error performing sentiment analysis: {e}"), "tab-sentiment-results"

# New Callback for Cohort Analysis
@app.callback(
    Output('cohort-results-content', 'children'),
    Output('main-tabs', 'active_tab', allow_duplicate=True), # To switch to the Cohort Results tab
    Input('run-cohort-analysis', 'n_clicks'),
    State('cohort-customer-id-dropdown', 'value'),
    State('cohort-order-date-dropdown', 'value'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def run_cohort_analysis(n_clicks, customer_id_col, order_date_col, contents):
    if n_clicks is None:
        raise PreventUpdate

    if not contents:
        return html.Div("Please upload a CSV file first."), "tab-cohort-results"

    if not customer_id_col or not order_date_col:
        return html.Div("Please select both Customer ID and Order Date columns for Cohort Analysis."), "tab-cohort-results"

    try:
        content_type, content_string = contents.split(',')
        decoded_bytes = base64.b64decode(content_string) 
        decoded_str = None
        try:
            decoded_str = decoded_bytes.decode('utf-8')
        except UnicodeDecodeError:
            decoded_str = decoded_bytes.decode('latin-1')

        try:
            temp_df = pd.read_csv(io.StringIO(decoded_str))
        except pd.errors.ParserError:
            temp_df = pd.read_csv(io.StringIO(decoded_str), delimiter=';')
        
        if customer_id_col not in temp_df.columns or order_date_col not in temp_df.columns:
            return html.Div(f"Selected columns '{customer_id_col}' or '{order_date_col}' not found in the uploaded data."), "tab-cohort-results"

        cohort_counts, retention_matrix = calculate_cohort_retention(temp_df.copy(), customer_id_col, order_date_col)

        if retention_matrix.empty:
            return html.Div("No valid data for Cohort Analysis after cleaning. Please check your data or date range."), "tab-cohort-results"

        # Format retention matrix for display
        retention_display = retention_matrix.reset_index()
        retention_display['CohortMonth'] = retention_display['CohortMonth'].astype(str) # Convert Period to string for display

        # Create heatmap using go.Heatmap
        fig_retention_heatmap = go.Figure(data=go.Heatmap(
            z=retention_matrix.values,
            x=retention_matrix.columns.astype(str), # Ensure x-axis labels are strings
            y=retention_matrix.index.astype(str),   # Ensure y-axis labels are strings
            colorscale='Greens',
            text=retention_matrix.round(1).astype(str).values, # Text to display on heatmap cells
            texttemplate="%{text}%", # Format text to show percentage
            hoverinfo='text'
        ))
        
        fig_retention_heatmap.update_layout(
            title='Customer Retention Rate by Cohort',
            xaxis_title='Cohort Index (Months Since First Purchase)',
            yaxis_title='Cohort Month',
            height=600,
            # Add annotations for better readability, similar to text_auto
            annotations=[
                go.layout.Annotation(
                    x=col_idx,
                    y=row_idx,
                    text=f"{retention_matrix.iloc[row_idx, col_idx]:.1f}%",
                    showarrow=False,
                    font=dict(color="black" if retention_matrix.iloc[row_idx, col_idx] < 50 else "white") # Adjust text color based on cell value
                )
                for row_idx, row_val in enumerate(retention_matrix.index)
                for col_idx, col_val in enumerate(retention_matrix.columns)
                if not pd.isna(retention_matrix.iloc[row_idx, col_idx]) # Only add text for non-NaN values
            ]
        )


        cohort_content = [
            html.H4("Cohort Analysis Results"),
            html.P("Cohort analysis helps understand customer behavior and retention over time. Each row represents a cohort (customers acquired in a specific month), and columns represent months since their first purchase."),
            html.Hr(),
            html.H5("Customer Retention Matrix (%)"),
            dash_table.DataTable(
                id='retention-matrix-table',
                columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ".1f"}} if i != 'CohortMonth' else {"name": i, "id": i} for i in retention_display.columns],
                data=retention_display.to_dict('records'),
                style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                page_size=10,
            ),
            dcc.Graph(figure=fig_retention_heatmap),
            html.Div([
                html.H5("Interpreting Cohort Analysis:"),
                html.Ul([
                    html.Li([
                        html.Strong("Cohort Month: "),
                        "The month when a group of customers made their first purchase."
                    ]),
                    html.Li([
                        html.Strong("Cohort Index (Months Since First Purchase): "),
                        "Represents the number of months passed since the cohort's first purchase. Index 0 is the acquisition month, Index 1 is the next month, and so on."
                    ]),
                    html.Li([
                        html.Strong("Retention Rate: "),
                        "The percentage of customers from a specific cohort who are still active (made a purchase) in a subsequent month. A higher percentage indicates better retention."
                    ]),
                    html.Li([
                        html.Strong("Heatmap Interpretation: "),
                        "Darker shades typically indicate higher retention rates. You can observe trends: do retention rates drop sharply after the first month? Do certain cohorts retain better than others?"
                    ]),
                    html.Li([
                        html.Strong("Business Insights: "),
                        "Identify if recent cohorts have better or worse retention than older ones. This can inform marketing effectiveness, product changes, and customer success initiatives."
                    ])
                ])
            ], className="mt-4 mb-4")
        ]

        return html.Div(cohort_content), "tab-cohort-results"

    except Exception as e:
        return html.Div(f"Error performing Cohort Analysis: {e}"), "tab-cohort-results"


# Run the Dash application
if __name__ == '__main__':
    app.run(debug=True)

