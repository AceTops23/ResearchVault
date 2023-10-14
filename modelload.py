import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load your dataset
print("Loading dataset...")
df = pd.read_csv('Annotations.csv')

# Define a hierarchical classification model
class HierarchicalBERT(torch.nn.Module):
    def __init__(self, num_imrad_classes, num_subsection_classes, num_subsubsection_classes):
        super(HierarchicalBERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_imrad_classes)
        self.subsection_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_subsection_classes)
        self.subsubsection_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_subsubsection_classes)
        
        # Store the labels used during training
        self.imrad_labels = ["INTRODUCTION", "METHODS", "RESULTS AND DISCUSSION", "RECOMMENDATIONS", "SUMMARY"]
        
        print("IMRAD Labels:", self.imrad_labels)
        
        self.subsection_labels = [
            "Research Design",
            "Locale of the Study",
            "Applied Concepts and Techniques",
            "Algorithm Analysis",
            "Data Collection Methods",
            "Data Model Generation",
            "Data Collection",
            "Pre-Processing",
            "Feature Extraction",
            "Tokenization",
            "Stemming",
            "Delete Stop Words",
            "Vectorization",
            "Lowercasing",
            "Lemmatization",
            "Numeric and Special Character Removal",
            "System Development Methodology",
            "Software and Hardware Tools Used",
            "System Architecture",
            "Prototype Output",
            "RESEARCH OBJECTIVE 1:",
            "RESEARCH OBJECTIVE 2:",
            "RESEARCH OBJECTIVE 3:",
            "RESEARCH OBJECTIVE 4:",
            "RESEARCH OBJECTIVE 5:",
            "Overall Record of Actual Testing"
        ]
        
        
        print("Subsection Labels:", self.subsection_labels)

        self.subsubsection_labels = [
            "Algorithm Testing Haar Cascade and Local Binary Patterns",
            "Analysis",
            "Analysis and Quick Design",
            "Brainstorm",
            "Build Prototype",
            "Coding",
            "Consultation",
            "Customer Evaluation",
            "Customized Frequently Asked Questions (FAQs)",
            "Data Collection",
            "Data Preparation Process",
            "Data Preprocessing",
            "Deployment",
            "Design",
            "Development",
            "Doing List",
            "Done List",
            "Feature Extraction",
            "Feature Vectors",
            "Haar Cascade",
            "Hardware",
            "Image Gathering",
            "Image Recognition",
            "Implementation",
            "Interview",
            "Library Research",
            "Local Data",
            "Machine Learning Techniques",
            "Maintenance",
            "Model Development",
            "Model Evaluation",
            "Model Implementation",
            "Model Requirement",
            "Natural Language Processing Techniques",
            "Object Detection",
            "Object Recognition",
            "On Hold List",
            "Online Research",
            "Online Survey",
            "Prediction and Authentication",
            "Priority List",
            "Prototype Cycles",
            "Quality Assurance",
            "Questionnaire",
            "Quick Design and Prototype Cycles",
            "Requirement Gathering",
            "Review and Update",
            "Review List",
            "Roboflow",
            "Semi-Supervised Machine Learning",
            "Software",
            "Supervised Learning",
            "Synthetic Data",
            "Testing",
            "Text Classification",
            "Text Recognition",
            "To Do List",
            "Train Recognizer Model"
        ]
        print("Subsubsection Labels:", self.subsubsection_labels)

    def forward(self, input_ids, attention_mask):
        imrad_logits = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        subsection_logits = self.subsection_classifier(input_ids=input_ids, attention_mask=attention_mask).logits
        subsubsection_logits = self.subsubsection_classifier(input_ids=input_ids, attention_mask=attention_mask).logits
        return imrad_logits, subsection_logits, subsubsection_logits

    def get_labels(self):
        # Return the stored labels
        return self.imrad_labels, self.subsection_labels, self.subsubsection_labels
    
# Tokenizer and model initialization
print("Initializing tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128  # Adjust this based on your data and hardware constraints
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# Define the number of classes for each level of hierarchy
num_imrad_classes = len(df['IMRAD Section'].unique())
num_subsection_classes = len(df['Subsection'].unique()) + 1  # +1 for "none"
num_subsubsection_classes = len(df['Sub subsection'].unique()) + 1  # +1 for "none"

# Print the number of classes for each level of hierarchy
print(f"Number of IMRAD classes: {num_imrad_classes}")
print(f"Number of subsection classes: {num_subsection_classes}")
print(f"Number of subsubsection classes: {num_subsubsection_classes}")
    
# Initialize the model
print("Initializing the model...")
model = HierarchicalBERT(num_imrad_classes, num_subsection_classes, num_subsubsection_classes)
model.to(device)
print("Model initialized successfully.")
