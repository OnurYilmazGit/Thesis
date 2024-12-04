from fpdf import FPDF

# Create instance of FPDF class
pdf = FPDF()

# Add a page
pdf.add_page()

# Set title font
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, txt="Onur YILMAZ", ln=True, align='C')

# Add contact details
pdf.set_font('Arial', '', 12)
pdf.cell(200, 10, txt="Munich, Germany | business.onuryilmaz@hotmail.com | linkedin.com/in/onur-yilmaz | github.com/OnurYilmazGit", ln=True, align='C')

# Add Objective
pdf.ln(10)
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Objective', ln=True)

pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, "Master's graduate in Informatics with extensive experience in DevOps, cloud infrastructure, and data-driven dashboard development. Proficient in Python, Jenkins, Grafana, and cloud technologies (AWS, Azure). Ready to contribute to innovative engineering teams by applying my expertise to optimize performance, automate infrastructure, and improve scalability.")

# Add Education
pdf.ln(5)
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Education', ln=True)

pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, "Technical University of Munich - M. Sc. Informatics", ln=True)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, "October 2024", ln=True)

pdf.set_font('Arial', 'I', 12)
pdf.cell(0, 10, "Master Thesis: Exploration of Machine Learning Dataset Compression on Time-Series Datasets", ln=True)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, "Explored dataset distillation techniques to optimize data volume while maintaining model accuracy, with a focus on dual-stage data augmentation pipelines. Utilized Python and deep learning frameworks (e.g., TensorFlow, PyTorch) to run extensive experiments on HPC workloads, optimizing computational resources and storage.")

pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, "Özyeğin University - Bachelor of Computer Science (Honors)", ln=True)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, "September 2015 - July 2021", ln=True)

# Add Work Experience
pdf.ln(5)
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Work Experience', ln=True)

# Intel Corporation
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, "Intel Corporation - Infrastructure and DevOps Intern", ln=True)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, "May 2022 – October 2023", ln=True)

pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, '''
- Automated Infrastructure & Monitoring: Reduced manual monitoring efforts by 98%, cutting test result retrieval time from 1 hour to 5 minutes by developing an automated infrastructure and test suite system.
- Advanced Dashboard Development: Created a Grafana dashboard integrated with Jenkins and InfluxDB, improving real-time data visibility and enhancing decision-making for test results and system inventory.
- Data Processing Pipeline: Automated data collection and processing from Jenkins, transforming raw data into a structured format suitable for efficient storage and analysis.
- Custom Visualization Solutions: Enhanced clarity of performance metrics and system inventory with tailored Grafana dashboards, enabling more informed decisions.
- Proactive Alerting System: Improved system reliability with automated alerts for test failures and performance issues, reducing downtime.
- Production-Grade Deployment: Ensured scalability and reliability of the dashboard by deploying it using Docker Swarm.''')

# Giantswarm Company
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, "Giantswarm Company - Cloud Native Working Student", ln=True)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, "November 2023 – August 2024", ln=True)

pdf.multi_cell(0, 10, '''
- Enhanced Security & Workflow Efficiency: Spearheaded Kubernetes package updates, improving security and streamlining workflows.
- Improved Cluster Reliability: Diagnosed and fixed failed Kubernetes nodes, ensuring high availability and stable performance.
- Optimized Resource Utilization: Managed customer clusters with Microsoft Azure services, maximizing resource efficiency.
- Strengthened Governance & Compliance: Developed and enforced new cluster policies with Go and Kubernetes, ensuring compliance.
- Refined Kubernetes Infrastructure: Implemented best practices for Kubernetes orchestration, improving system performance.''')

# Add University Projects
pdf.ln(5)
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'University Projects', ln=True)

pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, "Court Finder App - Tech Challenge Winner (July 2023)", ln=True)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, "Developed a web application using Python and JavaScript, integrating a decision tree for legal court decisions. Won the Tech Challenge.")

pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, "Air Drum - Bachelor Graduation Project (June 2021)", ln=True)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, "Designed a virtual drum kit using Python, OpenCV, and Kalman Filter, allowing users to play drums using body movements.")

# Add Technical Skills & Certifications
pdf.ln(5)
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Technical Skills & Certifications', ln=True)

pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, '''
- **Programming Languages:** Proficient in Python, C++, Java | Familiar with Go, Groovy, JavaScript, CSS/HTML  
- **Cloud Platforms:** Proficient in AWS, Microsoft Azure | Familiar with Google Cloud  
- **DevOps & Automation:** Jenkins, Grafana, Docker, Ansible, Kubernetes, CI/CD, Git/GitHub  
- **Databases & Monitoring:** InfluxDB, Advanced MySQL, Clickhouse, Snowflake  
- **Machine Learning & AI:** TensorFlow, OpenCV, YOLO  
- **Operating Systems:** Unix-based systems, Windows  
- **Automation & Scripting:** Automation, Debugging, Test-Driven Development, Virtualization
''')

pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, "Certifications:", ln=True)
pdf.cell(0, 10, "- AWS Certified Cloud Practitioner (Earned: June 2023)", ln=True)
pdf.cell(0, 10, "- Preparing for AWS Certified Solutions Architect – Associate Exam (Expected completion: December 2024)", ln=True)

# Add Soft Skills and Languages
pdf.ln(5)
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Soft Skills', ln=True)

pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, "Strong problem-solving abilities with a fast learning curve, demonstrated through quickly mastering new technologies and adapting to project needs. Independent, goal-oriented, and analytical, with proven DevOps problem-solving skills. Effective communicator and proactive team collaborator, consistently contributing to team success.")

pdf.ln(5)
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Languages', ln=True)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, "Fluent in English and Turkish | Learning German", ln=True)

# Add Extracurricular Activities
pdf.ln(5)
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Extracurricular Activities', ln=True)

pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, "Jazz Drumming", ln=True)
pdf.multi_cell(0, 10, "Dedicated 7 years to mastering jazz drumming, performing at various events and gigs.")

# Output PDF
pdf_file_path = "Onur_Yilmaz_Final_CV.pdf"
pdf.output(pdf_file_path)


