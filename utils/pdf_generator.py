import os
import datetime
from fpdf import FPDF
from PIL import Image
import tempfile
import cv2

class ReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'PlantGuard AI - Diagnostic Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(original_img, heatmap_img, prediction_data, disease_info):
    """
    Generates a PDF report for a prediction.
    
    Args:
        original_img: PIL Image of the uploaded leaf.
        heatmap_img: numpy array of the Grad-CAM heatmap superimposed image.
        prediction_data: dict containing prediction results.
        disease_info: dict containing disease information.
        
    Returns:
        str: Path to the generated PDF file.
    """
    pdf = ReportPDF()
    pdf.add_page()
    
    # Title & Metadata
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    
    status = "Healthy" if prediction_data['is_healthy'] else "Diseased"
    pdf.cell(0, 10, f'Predicted Class: {prediction_data["predicted_class"].replace("_", " ")}', 0, 1)
    pdf.cell(0, 10, f'Status: {status}', 0, 1)
    pdf.cell(0, 10, f'Confidence: {prediction_data["confidence"] * 100:.2f}%', 0, 1)
    
    pdf.ln(10)
    
    # Images (We need to save them temporarily to add to PDF)
    temp_dir = tempfile.gettempdir()
    orig_path = os.path.join(temp_dir, "orig_leaf.jpg")
    heat_path = os.path.join(temp_dir, "heat_leaf.jpg")
    
    original_img.convert('RGB').save(orig_path)
    
    if heatmap_img is not None:
        cv2.imwrite(heat_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
    
    # Add images
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(95, 10, 'Original Image:', 0, 0, 'C')
    if heatmap_img is not None:
        pdf.cell(95, 10, 'AI Focus Area (Grad-CAM):', 0, 1, 'C')
    else:
        pdf.ln(10)
        
    y_before_images = pdf.get_y()
    pdf.image(orig_path, x=10, y=y_before_images, w=80)
    
    if heatmap_img is not None:
        pdf.image(heat_path, x=110, y=y_before_images, w=80)
    
    pdf.set_y(y_before_images + 85) # Move cursor below images
    
    # Disease Info
    if not prediction_data['is_healthy']:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Disease Information', 0, 1)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Description:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 8, disease_info.get("description", "N/A"))
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Symptoms:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 8, disease_info.get("symptoms", "N/A"))
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Causes:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 8, disease_info.get("causes", "N/A"))
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Prevention/Care:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 8, disease_info.get("prevention", "N/A"))
        
    # Clean up temp files
    try:
        os.remove(orig_path)
        if heatmap_img is not None:
            os.remove(heat_path)
    except Exception:
        pass
        
    report_path = os.path.join(temp_dir, f"PlantGuard_Report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
    pdf.output(report_path)
    return report_path
