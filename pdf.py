import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pandas as pd
import plotly.io as pio
import base64
from datetime import datetime
import os # Import os for environment check

def generate_segment_report(df, features, figures):
    """
    Generate a PDF report for the segmentation analysis
    
    Parameters:
    -----------
    df : DataFrame
        Data with cluster assignments
    features : list
        List of feature column names
    figures : list
        List of plotly figures
    
    Returns:
    --------
    bytes object containing the PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=1, parent=styles['Heading1']))
    styles.add(ParagraphStyle(name='Heading2Centered', alignment=1, fontSize=14, spaceAfter=6, parent=styles['h2']))
    styles.add(ParagraphStyle(name='NormalSmall', fontSize=9, parent=styles['Normal']))
    
    Story = []

    print("PDF Generation: Starting report generation...")

    # Title Page
    Story.append(Paragraph("Customer Segmentation Report", styles['Center']))
    Story.append(Spacer(1, 0.2 * inch))
    Story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Center']))
    Story.append(Spacer(1, 0.5 * inch))
    Story.append(Paragraph("This report provides an overview of customer segments identified using various analytical techniques.", styles['Normal']))
    Story.append(Spacer(1, 2 * inch))
    print("PDF Generation: Title page added.")

    # Executive Summary (Page 2)
    Story.append(Spacer(1, 0.5 * inch)) # Add space to push content to next page or lower
    Story.append(Paragraph("1. Executive Summary", styles['h1']))
    Story.append(Spacer(1, 0.1 * inch))
    Story.append(Paragraph(
        "This report details the customer segmentation derived from the provided dataset. By grouping customers with similar characteristics, businesses can develop targeted marketing strategies, improve customer engagement, and optimize resource allocation. The analysis identifies distinct customer segments based on their behavioral patterns.",
        styles['Normal']
    ))
    Story.append(Spacer(1, 0.2 * inch))
    print("PDF Generation: Executive Summary added.")

    # Segmentation Overview
    Story.append(Paragraph("2. Segmentation Overview", styles['h1']))
    Story.append(Spacer(1, 0.1 * inch))
    Story.append(Paragraph("K-Means Segment Distribution:", styles['h2']))
    
    # K-Means Segment Table
    if 'Cluster' in df.columns:
        cluster_counts = df['Cluster'].value_counts(normalize=True).reset_index()
        cluster_counts.columns = ['K-Means Segment', 'Percentage']
        cluster_counts['Percentage'] = cluster_counts['Percentage'].apply(lambda x: f"{x:.1%}")
        
        table_data = [cluster_counts.columns.tolist()] + cluster_counts.values.tolist()
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        t = Table(table_data)
        t.setStyle(table_style)
        Story.append(t)
        Story.append(Spacer(1, 0.2 * inch))
        print("PDF Generation: K-Means Segment Distribution table added.")
    else:
        Story.append(Paragraph("K-Means Cluster information not available.", styles['NormalSmall']))
        print("PDF Generation: K-Means Cluster information not available (skipped table).")

    # RFM Segment Distribution (if applicable)
    if 'RFM_Segment' in df.columns:
        Story.append(Paragraph("RFM Segment Distribution:", styles['h2']))
        rfm_counts = df['RFM_Segment'].value_counts(normalize=True).reset_index()
        rfm_counts.columns = ['RFM Segment', 'Percentage']
        rfm_counts['Percentage'] = rfm_counts['Percentage'].apply(lambda x: f"{x:.1%}")
        
        table_data_rfm = [rfm_counts.columns.tolist()] + rfm_counts.values.tolist()
        table_style_rfm = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        t_rfm = Table(table_data_rfm)
        t_rfm.setStyle(table_style_rfm)
        Story.append(t_rfm)
        Story.append(Spacer(1, 0.2 * inch))
        print("PDF Generation: RFM Segment Distribution table added.")
    else:
        Story.append(Paragraph("RFM Segment information not available.", styles['NormalSmall']))
        print("PDF Generation: RFM Segment information not available (skipped table).")


    # K-Means Segment Profiles
    if 'Cluster' in df.columns and features:
        Story.append(Paragraph("3. K-Means Segment Profiles", styles['h1']))
        Story.append(Spacer(1, 0.1 * inch))
        Story.append(Paragraph("Average values of features by K-Means segment:", styles['Normal']))
        
        cluster_profiles = df.groupby('Cluster')[features].mean().round(2).reset_index()
        table_data_profiles = [cluster_profiles.columns.tolist()] + cluster_profiles.values.tolist()
        
        table_style_profiles = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        t_profiles = Table(table_data_profiles)
        t_profiles.setStyle(table_style_profiles)
        Story.append(t_profiles)
        Story.append(Spacer(1, 0.2 * inch))
        print("PDF Generation: K-Means Segment Profiles table added.")
    else:
        Story.append(Paragraph("K-Means Cluster profile information not available.", styles['NormalSmall']))
        print("PDF Generation: K-Means Cluster profile information not available (skipped table).")

    # Visualizations
    Story.append(Paragraph("4. Visualizations", styles['h1']))
    Story.append(Spacer(1, 0.1 * inch))
    print(f"PDF Generation: Attempting to add {len(figures)} visualizations.")

    for i, fig in enumerate(figures):
        if fig is None: # Skip if the figure is None (e.g., if RFM graph was not generated)
            print(f"PDF Generation: Skipping figure {i+1} as it is None.")
            continue
        try:
            print(f"PDF Generation: Converting figure {i+1} ('{fig.layout.title.text if fig.layout.title.text else 'Untitled'}') to image...")
            # Convert Plotly figure to a static image (PNG)
            img_bytes = pio.to_image(fig, format="png", engine="kaleido", width=800, height=500, scale=2) # Increased scale for better resolution
            img_data = io.BytesIO(img_bytes)
            
            # Create ReportLab Image object
            # Calculate width to fit page, maintaining aspect ratio
            img = Image(img_data)
            img_width, img_height = img.drawWidth, img.drawHeight
            aspect = img_height / float(img_width)
            
            # Max width for an image on a letter page (approx 6.5 inches)
            max_width = 6.5 * inch 
            if img_width > max_width:
                img_width = max_width
                img_height = img_width * aspect

            img.drawWidth = img_width
            img.drawHeight = img_height
            
            Story.append(img)
            Story.append(Spacer(1, 0.1 * inch))
            Story.append(Paragraph(f"Figure {i+1}: {fig.layout.title.text if fig.layout.title.text else 'Segmentation Plot'}", styles['NormalSmall']))
            Story.append(Spacer(1, 0.2 * inch))
            print(f"PDF Generation: Successfully added figure {i+1} to report.")
        except Exception as e:
            print(f"PDF Generation Error: Could not convert figure {i+1} ('{fig.layout.title.text if fig.layout.title.text else 'Untitled'}') to image. Error: {e}")
            Story.append(Paragraph(f"Figure {i+1} could not be generated due to an error: {e}", styles['NormalSmall']))
            Story.append(Spacer(1, 0.2 * inch))

    # Build the PDF
    print("PDF Generation: Attempting to build the PDF document...")
    try:
        doc.build(Story)
        buffer.seek(0)
        print("PDF Generation: PDF document built successfully.")
        return buffer.getvalue()
    except Exception as e:
        print(f"PDF Generation Error: Error building PDF document: {e}")
        # Re-raise the exception to be caught by the Dash callback if needed for debugging
        raise

