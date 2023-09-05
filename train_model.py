import spacy
from spacy.training.example import Example

# Load a pre-trained model or create a blank model
nlp = spacy.load("en_core_web_sm")
# OR: nlp = spacy.blank("en")

# Define your custom component (section detection) and add it to the pipeline
section_detection = nlp.create_pipe("ner")
nlp.add_pipe(section_detection)

# Load your annotated data (e.g., from a JSON file)
training_data = [
    ("Introduction is the first section.", {"entities": [(0, 12, "SECTION")]}),
    # Add more annotated examples
]

<<<<<<< HEAD
if __name__ == "__main__":
    # Step 2: Prepare the datasets
    csv_files = ['dataset/A SYLLABUS GENERATOR FOR THE COLLEGE.csv', 'dataset/A Web-based Announcement Management System via SMS for the College of Computer Studies.csv', 'dataset/Anonymous Restriction Application Using AES Algorithm.csv', 'dataset/Book1AFScan Android File Scanner and Translator with Optical.csv', 'dataset/CCS InfoCast- An Information Dissemination.csv',
                 'dataset/CCS Online Grading System.csv', 'dataset/CCS ONLINE THESIS MANAGEMENT SYSTEM.csv', 'dataset/Crime Analysis in 4th District of Laguna using JRip Algorithm.csv', 'dataset/eFort AN ELECTRONIC FACULTY PORTFOLIO MANAGEMENT SYSTEM.csv', 'dataset/FEMS  A LAN-BASED STUDENTS EVALUATION ON.csv', 'dataset/Jesus the Saviour Hospital Management Information System.csv', 
                 'dataset/LSPU CCS PAYMENT.csv', 'dataset/OFFICE MANAGEMENT SOLUTIONS FOR.csv', 'dataset/PORTABLE COOLER.csv', 'dataset/Premiumbikes Lending MangementSystem.csv', 'dataset/SMART EXAM CHECKER USING SPEED UP ROBUST FEATURE (SURF).csv', 'dataset/THERMOELECTRIC BBQ GRILL.csv', 'dataset/VIRUS ATTACK- THE DEVELOPMENT OF.csv', 'dataset/WEB-BASED SYLLABUS MANAGEMENT SYSTEM FOR COLLEGE OF COMPUTER.csv', 
                 'dataset/THE-DEVELOPMENT-OF-MUNICIPAL-PLANNING-AND-DEVELOPMENT-COORDINATOR-OFFICE-MPDCO.csv', 'dataset/THE-DEVELOPMENT-OF-MDRRMO-REPORT-MANAGEMENT-INFORMATION-SYSTEM.csv', 'dataset/THE-BIBLE-WARRIOR-VIRTUAL-REALITY.csv', 'dataset/TESDA-SCHOOL-DIVISION-PAYROLL-AND-MANAGEMENT-SYSTEM.csv', 'dataset/Sustainable-Livelihood-Program-Management-Information-System-with-SMS-Notification.csv', 
                 'dataset/SMP-Service-Management-Program-E-Learning-System.csv', 'dataset/SMART-OFFICE-Office-Automation-And-Security-Monitoring-System.csv', 'dataset/SCHOOL-EVENT-ATTENDANCE-MONITORING-SYSTEM-USING-FINGERPRINT-AND-SMS-NOTIFICATION.csv', 'dataset/SaveLIFE-Connecting-Blood-Donors-Mobile-Application.csv', 'dataset/SANTA-CRUZ-WATER-DISTRICT-CUSTOMER-RELATIONSHIP.csv', 
                 'dataset/RAINFALL-ADVISORY-SYSTEM-WITH-SMS-NOTIFICATION-FOR-MUNICIPAL-DISASTER-RISK-REDUCTION-AND-MANAGEMENT-OFFICE-SANTA-CRUZ-LAGUNA.csv', 'dataset/PINHS-INTERACTIVE-ONLINE-COURSEWARE-FOR-ARALING-PANLIPUNAN-WITH-VIRTUAL-ASSISTANCE.csv', 'dataset/PICK-TAP-AND-SNAP-A-COIN-OPERATED-PHOTO-BOOTH.csv', 'dataset/PICK-TAP-AND-SNAP-A-COIN-OPERATED-PHOTO-BOOTH.csv', 
                 'dataset/PESO-LAGUNA-INTEGRATED-WEBSITE-WITH-DATA-MAPPING.csv', 'dataset/PDRRMO-INFORMATION-MANAGEMENT-SYSTEM.csv', 'dataset/OSYAID-An-Online-Learning-Management-System-for-Out-of-School.csv', 'dataset/Online-Management-Information-System-for-Girl-Scouts-of-the-Philippines-Laguna-Council.csv', 'dataset/ONLINE-COURSEWARE-OF-DMRMNHS-IN-FILIPINO-SUBJECT.csv', 
                 'dataset/NSTP-WEB-PORTAL-STUDENT-INFORMATION-MANAGEMENT-SYSTEM-WITH-AUTO-GENERATED-QR-CODE.csv', 'dataset/Nexus-Point-A-College-of-Computer-Studies-Web-Portal.csv', 'dataset/MUNICIPALITY-OF-STA.-CRUZ-SCHOLARSHIP-PROGRAM-VALIDATION-WITH-SMS-NOTIFICATION.csv', 'dataset/TILA-T.L.E.-INTERACTIVE-LEARNING-AID.csv', 'dataset/MUNICIPALITY-OF-MAGDALENA-BILLING-AND-COLLECTION-WITH-METER-READING-APPLICATION-AND-SMS-NOTIFICATION.csv', 
                 'dataset/Tiwi-Food-Product-Company-Web-based-Human-Resource-Integrated-System.csv', 'dataset/TRIP-SA-LAGUNA-MOBILE-APPLICATION-SYSTEM-TRAVEL-ASSISTANT.csv', 'dataset/A-SALON-PROPRIETORS-SMART-ASSISTANT-SYSTEM.csv', 'dataset/Modernization-Program-for-Kingdom-Plantae-Identification-and-Proper-Care-with-the-Use-of-Mobile-Application-South-East-Asian-Plant.csv', 'dataset/MATRIX-ADVENTURES.csv', 
                 'dataset/Municipality-of-Santa-Cruz-Public-Cemetery-Record-Management-System-with-SMS-Notification.csv', 'dataset/Municipal-Tricycle-Franchise-Regulatory-Board-and-Business-Permit-Record-Management-System-MTFRBBPRMS.csv', 'dataset/MUNICIPAL-MEDICAL-ASSISTANCE-OFFICE-MEDICATION-MONITORING-SYSTEM-WITH-SMS-NOTIFICATION.csv', 'dataset/LSPU-WEB-BASED-RESEARCH-AND-DEVELOPMENT-MANAGEMENT-SYSTEM.csv', 
                 'dataset/LSPU-WEB-BASED-RESEARCH-AND-DEVELOPMENT-MANAGEMENT-SYSTEM.csv', 'dataset/LSPU-WEB-BASED-RESEARCH-AND-DEVELOPMENT-MANAGEMENT-SYSTEM.csv', 'dataset/LSPU-WEB-BASED-ISO-RECORD-MANAGEMENT-SYSTEM.csv', 'dataset/LSPU-System-Scholarship-and-Financial-Assistance-Record-Management-System-with-SMS-Announcement-and-Notification.csv', 'dataset/LSPU-STUDENTS-OJT-WEB-PORTAL.csv', 
                 'dataset/LSPU-SPORTS-EQUIPMENT-MONITORING-SYSTEM-USING-BIOMETRICS-AND-SMS-NOTIFICATION.csv', 'dataset/LSPU-SCC-Physical-Plant-and-Site-Development-Scheduling-Management-System.csv', 'dataset/LSPU-SCC-Asset-Management-System.csv', 'dataset/LPSU-ARCSS-LSPU-Social-Media-with-Thesis-Archive.csv', 'dataset/LDH-InfoKiosk-Laguna-Doctors-Hospital-Information-Kiosk.csv', 
                 'dataset/LAGUNA-STATE-POLYTECHNIC-UNIVERSITY-STA.CRUZ-CAMPUS-SAFETY-AND-SECURITY-MANAGEMENT-SYSTEM-USING-BARCODE-AND-SMS-TECHNOLOGY.csv', 'dataset/LAGUNA-STATE-POLYTECHNIC-UNIVERSITY-EXTENSION-AND-TRAINING-SERVICES-DOCUMENT-MANAGEMENT-SYSTEM.csv', 'dataset/LAGUNA-STATE-POLYTECHNIC-UNIVERSITY-CS-ONLINE-AN-ONLINE-CLERICAL-SERVICE-SYSTEM-FOR-THE-GUIDANCE-OFFICE.csv', 'dataset/LAGUNA-SHOPPE.csv', 
                 'dataset/Laguna-GENSERV-Cross-Platform-Web-and-Mobile-Application.csv', 'dataset/LABVIEW-LAN-Based-LSPU-SCC-CCS-Laboratory-Attendance-Monitoring-System-using-Finger-Print-Biometric-Technology.csv', 'dataset/LABVIEW-LAN-Based-LSPU-SCC-CCS-Laboratory-Attendance-Monitoring-System-using-Finger-Print-Biometric-Technology.csv', 'dataset/L-SMS-LSPU-SCC-SYLLABUS-MANAGEMENT-SYSTEM.csv', 
                 'dataset/JOBS-MANAGEMENT-AND-DISSEMINATION-SYSTEM-USING-SMS-TECHNOLOGY-FOR-PUBLIC-EMPLOYMENT-SERVICE-OFFICE-PESO-AT-SANTA-CRUZ-LAGUNA.csv', 'dataset/ITAid-A-LAN-Based-Courseware-for-College-of-Computer-Studies.csv', 'dataset/INTERGRATION-OF-COOPERATIVES-UNDER-PCDO-LOANING-MANAGEMENT-SYSTEM.csv', 'dataset/Instructional-Materials-IMs-Submission-and-Monitoring-System.csv', 
                 'dataset/INFORMATION-ASSIMILATION-LEARNING-TOOL-FOR-LUMBAN-NATIONAL-HIGH-SCHOOL.csv', 'dataset/IKNoSS-INTEGRATED-KNOWLEDGE-SUPERVISION-SYSTEM.csv', 'dataset/HUMAN-AR-AUGMENTED-HUMAN-INTERNAL-ORGANS.csv', 'dataset/FELICIANO-DENTAL-CLINIC-MANAGEMENT-SYSTEM-WITH-SMS.csv', 'dataset/FAMS-WEB-BASED-FARM-MONITORING-SYSTEM-WITH-3D-MAPPING-FOR-BRGY.-STA-CLARA-SUR-PILA-LAGUNA.csv', 
                 'dataset/FAMS-FACULTY-ATTENDANCE-MANAGEMENT-SYSTEM.csv', 'dataset/FACULTY-AVAILABILITY-AND-MONITORING-SYSTEM-USING-SMS-TECHNOLOGY.csv', 'dataset/Eyeguide-An-obstacle-detection-and-avoidance-for-blind-navigation.csv', 'dataset/EXECUTOR-MASSIVE-OPEN-WORLD-ROLE-PLAYING-GAME.csv', 'dataset/EXECUTOR-MASSIVE-OPEN-WORLD-ROLE-PLAYING-GAME.csv', 'dataset/EATSMART-A-SMART-DINING-SYSTEM.csv', 
                 'dataset/E-STENO-A-MOBILE-STENOGRAPHY.csv', 'dataset/E-Sked-Online-Scheduling-Management-System-of-Provincial-Population-Office-with-SMS-notification.csv', 'dataset/E-Report-Mo-An-Online-Crime-Reporting-System-for-Santa-Cruz-Municipal-Police-Station.csv', 'dataset/E-Mayor-Scheduling-Appointment-System.csv', 'dataset/E-HEALTH-VICTORIA-HEALTH-CENTER-ONLINE.csv', 
                 'dataset/e-GYNE-MOBILE-APPLICATION-FOR-WOMENS-HEALTH.csv', 'dataset/E-BNS-ONLINE-BARANGAY-NUTRITION-SCHOLAR.csv', 'dataset/DISASTERVILLE-ADROID-GAME.csv', 'dataset/CROSS-PLATFORM-UNIFIED-PRESIDENT-REPORT.csv', 'dataset/Crime-and-Incident-Mapping-of-Bay-Municipal-Police-Station.csv', 'dataset/COP-CCS-OJT-PORTAL.csv', 'dataset/COLLEGE-OF-COMPUTER-STUDIES-FILE-SUBMISSION-KIOSK.csv', 
                 'dataset/Class-Scheduler-An-Expert-System-Timetabling-for-CCS-Faculty-Members-using.csv', 'dataset/CHECKMATE-A-MOBILE-SCANNING-AND-SCORE-GENERATING-APPLICATION.csv', 'dataset/CCS-RESEARCH-LAB-MONITORING-SYSTEM-WITH-BARCODE-SCANNER.csv', 'dataset/CCS-eBASS-College-of-Computer-Studies-electronic-Bulletin-Announcement-System-using-SMS.csv', 'dataset/CALEMS-CCS-ADVANCE-LABORATORY-ELECTRIC-MANAGEMENT-SYSTEM.csv', 
                 'dataset/CAIHRMS-COMPUTER-AIDED-INSTRUCTION-FOR-HOTEL-RESERVATION-MANAGEMENT-SYSTEM.csv', 'dataset/Book1THEREPO-A-Thesis-Repository-Android-Based-Application.csv', 'dataset/Book1HR-Online-An-Online-Human-Resource-Management-Information-System-of-Laguna-State-Polytechnic-University-Santa-Cruz-Campus.csv', 'dataset/BAT-Budget-Office-Accounting-Office-Treasury-Office-Monitoring-Record-Management-System-of-Municipality-of-Sta.-Cruz-Laguna.csv', 
                 'dataset/BAT-Budget-Office-Accounting-Office-Treasury-Office-Monitoring-Record-Management-System-of-Municipality-of-Sta.-Cruz-Laguna.csv', 'dataset/AUTOMATED-CANTEEN-TRANSACTION-AND-GATE-PASS-MONITORING-USING-RFID-OF-ST.-MARYS-MONTESSORI.csv', 'dataset/AUGMENTED-ELEMENT-AN-INTERACTIVE-AUGMENTED-REALITY-GAME-OF-ELEMENTS.csv', 'dataset/ASADO-AUTOMATED-STUDENT-FILE-AND-DOCUMENT-ORGANIZER.csv', 'dataset/ADVANCE-LEARNING-FOR-CCS-SCC-USING-AR-AND-MODULE-SUPPORT.csv']
    df = load_datasets(csv_files)
=======
# Convert the training data into Example objects
examples = []
for text, annotations in training_data:
    example = Example.from_dict(nlp.make_doc(text), annotations)
    examples.append(example)
>>>>>>> parent of b876eff (up)

# Train the model
n_iter = 10  # Adjust the number of training iterations as needed
for _ in range(n_iter):
    for example in examples:
        nlp.update([example], losses={})

# Save the trained model
output_dir = "path_to_output_directory"
nlp.to_disk(output_dir)
