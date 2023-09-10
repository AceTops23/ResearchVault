import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from gensim import corpora, models
from nltk import ne_chunk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def load_datasets(csv_files):
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding='Windows-1252')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def preprocess_text(text):
    if isinstance(text, str):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        words = nltk.word_tokenize(text)
        processed_words = []
        for word in words:
            if word.isnumeric():
                processed_words.append("NUMERIC")
            else:
                stemmed_word = stemmer.stem(word)
                pos_tag = nltk.pos_tag([stemmed_word])[0][1]
                if pos_tag.startswith('NN'):
                    processed_words.append(stemmed_word)
        filtered_words = [word for word in processed_words if word.lower() not in stop_words]

        # Named Entity Recognition
        named_entities = ne_chunk(nltk.pos_tag(filtered_words))
        named_entities_filtered = [ne[0] if isinstance(ne, tuple) else ne for ne in named_entities]
        
        named_entities_str = [ne if isinstance(ne, str) else ne[0] for ne in named_entities_filtered if isinstance(ne, (str, tuple))]
        
        return " ".join(named_entities_str)  # Convert list of tokens to string
    else:
        return ""




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

    # Step 3: Data Preprocessing
    df["preprocessed_text"] = df["Text"].apply(preprocess_text)

    # Step 4: Drop rows with missing values
    df.dropna(subset=["preprocessed_text", "Label"], inplace=True)

    # Step 5: Topic Modeling
    
    texts = df["preprocessed_text"].tolist()
    tokenized_texts = [text.split() for text in texts] 
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=5, passes=15)

    
    vectorizer = TfidfVectorizer()
    preprocessed_texts = df["preprocessed_text"].tolist()  
    X = vectorizer.fit_transform(preprocessed_texts) 
    y = df["Label"]

    # Step 7: Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # Step 8: Evaluation
    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    # Step 9: Saving the Model
    joblib.dump(classifier, 'trained_model_with_tfidf.joblib')

