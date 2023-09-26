"""
This script preprocesses and trains a model on a dataset with hierarchical text data
and references to figures and tables.
"""
import os
import io
import pandas as pd
import pytesseract
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from PIL import Image

# Define the path to the Figures and Tables folders
FIGURES_FOLDER = 'Figures'
TABLES_FOLDER = 'Tables'

# Ensure that the Figures and Tables folders exist
if not os.path.exists(FIGURES_FOLDER):
    os.makedirs(FIGURES_FOLDER)

if not os.path.exists(TABLES_FOLDER):
    os.makedirs(TABLES_FOLDER)

# Load your dataset from Annotations.csv
data = pd.read_csv('Annotations.csv')

# Assuming you have two columns 'Subsections' and 'Subsubsections' in your CSV file

data = data[(data['Subsection'] != 'N/A') & (data['Sub subsection'] != 'N/A')]

# Replace all NaN elements with 0s.
data = data.fillna('missing')

# Split the Subsections and Subsubsections columns into lists of strings
data['Subsection'] = data['Subsection'].apply(lambda x: x.split(',') if isinstance(x, str) else [x])
data['Sub subsection'] = data['Sub subsection'].apply(lambda x: x.split(',') if isinstance(x, str) else [x])

# Find the maximum lengths for Subsections and Subsubsections
max_subsection_length = data['Subsection'].apply(len).max()
max_subsubsection_length = data['Sub subsection'].apply(len).max()

# Pad the Subsections and Subsubsections columns
data['Subsection'] = data['Subsection'].apply(lambda x: x + ['<PAD>'] * (max_subsection_length - len(x)))
data['Sub subsection'] = data['Sub subsection'].apply(lambda x: x + ['<PAD>'] * (max_subsubsection_length - len(x)))

# Convert the DataFrame columns to NumPy arrays
padded_subsection = np.array(data['Subsection'].tolist())
padded_subsubsection = np.array(data['Sub subsection'].tolist())

# Create a list of dictionaries with the padded data
padded_data = [{'Subsection': subsection, 'Subsub section': subsubsection}
               for subsection, subsubsection in zip(padded_subsection, padded_subsubsection)]

# Print or use the padded_data as needed
print(padded_data)

# Create a dictionary to map unique titles to dataset IDs
title_to_id = {title: idx for idx, title in enumerate(data['Title'].unique())}

# Add a new column 'Dataset_ID' to your DataFrame
data['Dataset_ID'] = data['Title'].map(title_to_id)

# Load the pretrained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to preprocess references to figures and tables
def preprocess_references(text, figure_reference, table_reference):
    """
    Preprocesses text by replacing figure and table references with placeholders.

    Args:
        text (str): The text to preprocess.
        figure_reference (str or None): The reference to a figure.
        table_reference (str or None): The reference to a table.

    Returns:
        str: The preprocessed text with references replaced by placeholders.
    """
    # Convert figure_reference and table_reference to strings if they are not already
    figure_reference = str(figure_reference)
    table_reference = str(table_reference)

    # Replace "N/A" with an empty string
    figure_reference = "" if figure_reference == "N/A" else figure_reference
    table_reference = "" if table_reference == "N/A" else table_reference

    # Replace figure references with corresponding filenames
    text = text.replace(figure_reference, f'[FIGURE:{figure_reference}]')
    # Replace table references with corresponding filenames
    text = text.replace(table_reference, f'[TABLE:{table_reference}]')
    return text


# Define a function to preprocess image files (adjust as needed)
def preprocess_image(image_path):
    """
    Preprocesses an image by loading it from the given path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray or None: The preprocessed image as a NumPy array or None if the file does not exist.
    """
    if image_path:
        # Load and preprocess image
        image = Image.open(image_path)
        # Convert the image to a NumPy array
        image_array = np.array(image)
        return image_array
    else:
        return None

# Define a function to preprocess table files (adjust as needed)
def preprocess_table(tables_path):
    """
    Preprocesses a table from a CSV file or PNG image located at the given path.

    Args:
        table_path (str): The path to the CSV file or PNG image containing the table data.

    Returns:
        DataFrame or None: The preprocessed table data as a DataFrame or None if the file does not exist.
    """
    if table_path:
        if table_path.endswith('.csv'):
            # Load and preprocess the table data from a CSV file
            table_data = pd.read_csv(tables_path)
        elif table_path.endswith('.png'):
            # Load and preprocess the table data from a PNG image using OCR
            try:
                image = Image.open(tables_path)
                # Perform OCR to extract text from the image
                table_text = pytesseract.image_to_string(image)
                # Convert the extracted text into a DataFrame (adjust as needed)
                table_data = pd.read_csv(io.StringIO(table_text), delimiter='\t')
            except Exception as e:
                print(f"Error processing table image: {e}")
                table_data = None
        else:
            print("Unsupported table format. Supported formats: .csv, .png")
            table_data = None

        return table_data
    else:
        return None

# Handle "N/A" values in text columns
def preprocess_text(text):
    """
    Preprocesses text by replacing "N/A" with an empty string.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text with "N/A" replaced by an empty string.
    """
    if text == "N/A":
        return ""
    return text

# Tokenize and preprocess the hierarchical text data, including references to figures and tables
max_seq_length = 256  

tokenized_data = []

for _, row in data.iterrows():
    imrad_section = preprocess_text(row['IMRAD Sections'])
    subsection = preprocess_text(row['Subsection'])
    sub_subsection = preprocess_text(row['Sub subsection'])
    section_content = row['Section Content']
    figure_filename = str(row['Figure Filename'])  # Ensure it's a string
    table_filename = str(row['Table Filename'])    # Ensure it's a string
    figure_reference = preprocess_text(row['Figure Reference'])
    table_reference = preprocess_text(row['Table Reference'])

    # Load and preprocess the figure (if available)
    figure_path = None
    if figure_filename and isinstance(figure_filename, str):
        figure_path = os.path.join(FIGURES_FOLDER, figure_filename)
    figure_features = preprocess_image(figure_path) if figure_path and os.path.exists(figure_path) else []

    # Load and preprocess the table (if available)
    table_path = None
    if table_filename and isinstance(table_filename, str):
        table_path = os.path.join(TABLES_FOLDER, table_filename)
    table_features = preprocess_table(table_path) if table_path and os.path.exists(table_path) else []

    # Handle references to figures and tables by replacing them with placeholders
    section_content = preprocess_references(section_content, figure_reference, table_reference)

    # Concatenate hierarchical text with appropriate separators
    full_text = f"{imrad_section} [SEP] {subsection} [SEP] {sub_subsection} [SEP] {section_content}"

    # Tokenize the text
    encoding = tokenizer.encode_plus(
        full_text,
        max_length=max_seq_length,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']


    # Load and preprocess the figure (if available)
    figure_path = os.path.join(FIGURES_FOLDER, figure_filename)
    figure_features = preprocess_image(figure_path) if os.path.exists(figure_path) else []

    # Load and preprocess the table (if available)
    table_path = os.path.join(TABLES_FOLDER, table_filename)
    table_features = preprocess_table(table_path) if os.path.exists(table_path) else []

    tokenized_data.append({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'figure_features': figure_features,
        'table_features': table_features,
        'dataset_id': row['Dataset_ID']
    })


# Split the dataset into train and validation sets
train_data, val_data = train_test_split(tokenized_data, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    """
    A custom dataset class for loading and processing hierarchical text data with associated features.

    Args:
        data (list): A list of dictionaries containing preprocessed data.

    Attributes:
        data (list): The list of preprocessed data dictionaries.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, figure_features, table_features, and dataset_id.
        """
        sample = {
            'input_ids': torch.tensor(self.data[idx]['input_ids']),
            'attention_mask': torch.tensor(self.data[idx]['attention_mask']),
            'figure_features': self.data[idx]['figure_features'],
            'dataset_id': self.data[idx]['dataset_id']
        }

        # Attempt to load 'table_features' and preprocess it
        table_features = self.data[idx]['table_features']
        if table_features is not None:
            if isinstance(table_features, (pd.DataFrame, list)):
                # Assuming 'table_features' is either a Pandas DataFrame or a list
                if isinstance(table_features, pd.DataFrame):
                    # Convert Pandas DataFrame to a NumPy array
                    # Handle non-numeric values by converting them to NaN
                    table_features = table_features.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
                elif isinstance(table_features, list):
                    # Convert list elements to float32
                    # Handle non-numeric values by converting them to NaN
                    table_features = [pd.to_numeric(item, errors='coerce') for item in table_features]
                    table_features = np.array(table_features, dtype=np.float32)

                # Convert NumPy array or list to a tensor
                sample['table_features'] = torch.tensor(table_features)
            else:
                raise ValueError(f"'table_features' has an unsupported data type: {type(table_features)}")
        else:
            sample['table_features'] = None  # Set it to None if it's not available

        return sample

# Create DataLoaders
batch_size = 32  # You can adjust this as needed
train_dataset = CustomDataset(train_data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

val_dataset = CustomDataset(val_data)
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

from transformers import BertForSequenceClassification
import torch

class HierarchicalBertModel(torch.nn.Module):
    """
    A custom hierarchical BERT model for sequence classification.

    Args:
        num_datasets (int): The number of unique datasets or classes for classification.
    """

    def __init__(self, num_datasets):
        super(HierarchicalBertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_datasets)

    def forward(self, input_ids, attention_mask):
        """
        Perform forward pass through the model.

        Args:
            input_ids (Tensor): Input tensor containing token IDs.
            attention_mask (Tensor): Input tensor containing attention mask.

        Returns:
            logits (Tensor): Logits output from the model.
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.logits


# Initialize the HBM model
num_datasets = len(data['Title'].unique())
model = HierarchicalBertModel(num_datasets)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 5  # You can adjust this as needed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}', unit=' batches'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dataset_ids = batch['dataset_id'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, dataset_ids)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0

    for batch in tqdm(val_dataloader, desc=f'Validation', unit=' batches'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dataset_ids = batch['dataset_id'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, dataset_ids)
            val_loss += loss.item()

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        val_correct += (preds == dataset_ids).sum().item()

    avg_val_loss = val_loss / len(val_dataloader)
    accuracy = val_correct / len(val_data)

    print(f'Epoch {epoch + 1}:')
    print(f'Training Loss: {avg_train_loss:.4f}')
    print(f'Validation Loss: {avg_val_loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'hierarchical_bert_model.pth')