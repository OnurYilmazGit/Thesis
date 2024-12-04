from fpdf import FPDF
import pandas as pd
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Project Report', align='C', ln=True)
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path, x=None, y=None, w=0):
        if os.path.exists(image_path):
            self.image(image_path, x=x, y=y, w=w)
            self.ln(10)

    def add_table(self, csv_path):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            self.set_font('Arial', '', 10)
            for i in range(len(df)):
                for col in df.columns:
                    self.cell(40, 10, str(df[col][i]), border=1)
                self.ln(10)

    def add_summary(self, summary):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Summary', 0, 1, 'L')
        self.ln(4)
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, summary)
        self.ln()

def generate_pdf():
    pdf = PDFReport()

    # Cover page
    pdf.add_page()
    pdf.chapter_title('Step 8: Summary of Results')

    summary_text = """
    The report provides a summary of model performance and data processing results, including details such as
    accuracy, compression ratios, and the size of datasets used. Additionally, visualizations like confusion matrices 
    and cluster plots are included to illustrate the results.
    """
    pdf.add_summary(summary_text)

    # Summary table
    pdf.add_page()
    pdf.chapter_title('Summary Table')
    pdf.add_table('report/summary_results.csv')

    # Images (Confusion Matrices, Clusters, Core Set)
    pdf.add_page()
    pdf.chapter_title('Confusion Matrix - Full Data')
    pdf.add_image('report/confusion_matrix_full.png', w=100)

    pdf.add_page()
    pdf.chapter_title('Confusion Matrix - Core Set')
    pdf.add_image('report/confusion_matrix_core.png', w=100)

    pdf.add_page()
    pdf.chapter_title('Confusion Matrix - Core Set Model on Full Data')
    pdf.add_image('report/confusion_matrix_full_core.png', w=100)

    pdf.add_page()
    pdf.chapter_title('Cluster Visualization')
    pdf.add_image('report/clusters.png', w=150)

    pdf.add_page()
    pdf.chapter_title('Core Set Visualization')
    pdf.add_image('report/core_set.png', w=150)

    # Save the PDF
    pdf.output('report/Project_Report.pdf')

if __name__ == "__main__":
    generate_pdf()
