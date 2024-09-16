import os
from fpdf import FPDF
from PyPDF2 import PdfMerger

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Machine Learning Model Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path, width, height):
        if os.path.exists(image_path):
            self.image(image_path, x=None, y=None, w=width, h=height)
        else:
            print(f"Warning: The image at {image_path} was not found.")
        self.ln()

class ReportGenerator:
    def __init__(self, results, model_name, data_type='real', compression_ratio=None, original_data_size=None, smote_data_size=None, enn_data_size=None, num_nodes=None):
        self.results = results
        self.model_name = model_name
        self.data_type = data_type  # 'real' or 'synthetic'
        self.compression_ratio = compression_ratio
        self.original_data_size = original_data_size
        self.smote_data_size = smote_data_size
        self.enn_data_size = enn_data_size
        self.num_nodes = num_nodes

    @staticmethod
    def save_with_incremental_filename(path, extension=".pdf"):
        counter = 1
        base, ext = os.path.splitext(path)
        while os.path.exists(path):
            path = f"{base}_{counter}{ext}"
            counter += 1
        return path

    def generate_pdf(self, output_dir, save_dir):
        pdf = PDF()
        pdf.add_page()
        pdf.chapter_title('Introduction')
        pdf.chapter_body(f'This report contains the results of the {self.model_name} model applied to the {self.data_type} dataset.')

        pdf.chapter_title('Data Overview')
        pdf.chapter_body(f'The dataset contains sensor data and response values from multiple nodes.')
        if self.compression_ratio is not None:
            pdf.chapter_body(f"Compression Ratio: {self.compression_ratio:.2f}")
        if self.original_data_size is not None:
            pdf.chapter_body(f"Original Data Size: {self.original_data_size}")
        if self.smote_data_size is not None:
            pdf.chapter_body(f"SMOTE Data Size: {self.smote_data_size}")
        if self.enn_data_size is not None:
            pdf.chapter_body(f"ENN Data Size: {self.enn_data_size}")
        if self.num_nodes is not None:
            pdf.chapter_body(f"Number of Nodes: {self.num_nodes}")

        pdf.chapter_title(f'{self.model_name} Results ({self.data_type.capitalize()} Data)')

        # Paths for images
        confusion_matrix_path = os.path.join(output_dir, f'{self.model_name}_confusion_matrix_{self.data_type}.png')
        classification_report_path = os.path.join(output_dir, f'{self.model_name}_classification_report_{self.data_type}.png')

        # Save images and check existence
        if os.path.exists(confusion_matrix_path):
            print(f"Adding image {confusion_matrix_path} to PDF...")
            pdf.add_image(confusion_matrix_path, width=180, height=100)
        else:
            print(f"Error: Confusion matrix image not found at {confusion_matrix_path}")

        if os.path.exists(classification_report_path):
            print(f"Adding image {classification_report_path} to PDF...")
            pdf.add_image(classification_report_path, width=180, height=100)
        else:
            print(f"Error: Classification report image not found at {classification_report_path}")

        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pdf_output_path = os.path.join(save_dir, f'ML_Model_Report_{self.model_name}_{self.data_type}.pdf')
        pdf_output_path = self.save_with_incremental_filename(pdf_output_path)
        pdf.output(pdf_output_path)
        return pdf_output_path

    @staticmethod
    def combine_pdfs(pdf_paths, output_path):
        merger = PdfMerger()
        for pdf in pdf_paths:
            merger.append(pdf)
        combined_pdf_path = ReportGenerator.save_with_incremental_filename(output_path)
        merger.write(combined_pdf_path)
        merger.close()
        return combined_pdf_path
