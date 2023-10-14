import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the CSV file into a DataFrame
df = pd.read_csv('annotations.csv')

# Create a dictionary to map IDs to their corresponding section/subsection/sub-subsection
imrad_to_subsection = {}
subsection_to_subsubsection = {}

# Iterate through each row in the DataFrame to identify mappings
for _, row in df.iterrows():
    # Map IMRAD Section ID to IMRAD Section
    imrad_to_subsection[row['IMRAD Section ID']] = row['IMRAD Section']
    
    # Map Subsection ID to IMRAD Section ID
    subsection_to_subsubsection[row['Subsection ID']] = row['IMRAD Section ID']

# Initialize lists to store filled values
imrad_sections = []
subsections = []
sub_subsections = []
section_contents = []

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Iterate through each row in the DataFrame to create hierarchical structure and tokenize
for _, row in df.iterrows():
    # Append IMRAD Section
    imrad_sections.append(imrad_to_subsection.get(row['IMRAD Section ID'], None))

    # Append Subsection
    subsections.append(row['Subsection'])

    # Append Sub-subsection
    sub_subsections.append(row['Sub subsection'])

    # Check for missing (NaN) values in 'Section Content'
    section_content = row['Section Content']
    if pd.notna(section_content):  # Check if it's not NaN
        section_content_tokens = tokenizer.encode(section_content, add_special_tokens=True, truncation=True, max_length=512)
    else:
        # Handle missing values (e.g., replace with an empty list)
        section_content_tokens = []
    
    # Append tokenized Section Content
    section_contents.append(section_content_tokens)

# Create a new DataFrame with the hierarchical structure and tokenized Section Content
hierarchical_df = pd.DataFrame({
    'Dataset ID': df['ID'],
    'IMRAD Section': imrad_sections,
    'Subsection': subsections,
    'Sub subsection': sub_subsections,
    'Section Content Tokens': section_contents
})

# Print the final hierarchy structure and the resulting hierarchical DataFrame
unique_imrad_sections = hierarchical_df['IMRAD Section'].unique()
for imrad_section in unique_imrad_sections:
    print(imrad_section)
    subsections_for_imrad = hierarchical_df[hierarchical_df['IMRAD Section'] == imrad_section]['Subsection'].unique()
    for subsection in subsections_for_imrad:
        print(f"    - {subsection}")
        sub_subsections_for_subsection = hierarchical_df[
            (hierarchical_df['IMRAD Section'] == imrad_section) &
            (hierarchical_df['Subsection'] == subsection)
        ]['Sub subsection'].unique()
        for sub_subsection in sub_subsections_for_subsection:
            print(f"         - {sub_subsection}")
    print()

# Split the dataset into training, evaluation, and test sets
# Adjust the test_size and random_state parameters as needed
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
test_df = train_test_split(eval_df, test_size=0.5, random_state=42)

# Define BERT models for each level of the hierarchy
num_labels_imrad = len(df['IMRAD Section'].unique())
num_labels_subsection = len(df['Subsection'].unique())
num_labels_sub_subsection = len(df['Sub subsection'].unique())

model_imrad = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels_imrad)
model_subsection = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels_subsection)
model_sub_subsection = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels_sub_subsection)

optimizer_imrad = AdamW(model_imrad.parameters(), lr=1e-5)
optimizer_subsection = AdamW(model_subsection.parameters(), lr=1e-5)
optimizer_sub_subsection = AdamW(model_sub_subsection.parameters(), lr=1e-5)

# Tokenize the section content and add the tokens to a new column in the DataFrame
df['Section Content Tokens'] = df['Section Content'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512, pad_to_max_length=True))

# Split the DataFrame into train, eval, and test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Print column names
print(df.columns)

# Define your DataFrame
dataframe = pd.DataFrame()

# Print column names
print(dataframe.columns)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create a label encoder
label_encoder = LabelEncoder()


# Fit the label encoder and transform 'sub_subsection' into numerical labels
sub_subsection = label_encoder.fit_transform(df['Sub subsection ID'])

print("Type of sub_subsection:", type(sub_subsection))
print("Shape of sub_subsection:", sub_subsection.shape)
print(sub_subsection)


# Ensure sub_subsection is one-dimensional
sub_subsection = sub_subsection.ravel()

sub_subsection_list = sub_subsection.tolist()

# Convert to a PyTorch tensor
labels = torch.tensor(sub_subsection, dtype=torch.int32).to(device)

class IMRaDDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        section_content_tokens = self.dataframe.iloc[idx]['Section Content Tokens']
        imrad_section = self.dataframe.iloc[idx]['IMRAD Section ID']
        subsection = self.dataframe.iloc[idx]['Subsection ID']
        sub_subsection = self.dataframe.iloc[idx]['Sub subsection ID']
        
        return section_content_tokens, imrad_section, subsection, sub_subsection

# Create datasets
train_dataset = IMRaDDataset(train_df)
eval_dataset = IMRaDDataset(eval_df)
test_dataset = IMRaDDataset(test_df)

# Create an instance of IMRaDDataset with df
dataset = IMRaDDataset(df)

# Now you can use 'dataset' in your DataLoader and training loop
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Move models to the device
model_imrad = model_imrad.to(device)
model_subsection = model_subsection.to(device)
model_sub_subsection = model_sub_subsection.to(device)

# Number of training epochs
num_epochs = 10

criterion = torch.nn.CrossEntropyLoss()


# Train model_sub_subsection
model_sub_subsection.train()  # Set the model to training mode
for batch in train_dataloader:
    inputs, imrad_section, subsection, sub_subsection = batch
    # Move batch to device
    inputs = [input.to(device) for input in inputs]
    labels = torch.tensor(sub_subsection, dtype=torch.int32).to(device)
    
    # Check if "Sub subsection" is applicable
    if not pd.isnull(labels):
        # Forward pass
        outputs = model_sub_subsection(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer_sub_subsection.zero_grad()
        loss.backward()
        optimizer_sub_subsection.step()

# Train model_subsection for data where "Sub subsection" is not applicable
model_subsection.train()
for batch in train_dataloader:
    inputs, _, labels, _ = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Check if "Subsection" is applicable and "Sub subsection" is not applicable
    if pd.isnull(labels):
        outputs = model_subsection(inputs)
        
        loss = criterion(outputs, labels)
        
        optimizer_subsection.zero_grad()
        loss.backward()
        optimizer_subsection.step()

# Train model_imrad for data where both "Sub subsection" and "Subsection" are not applicable
model_imrad.train()
for batch in train_dataloader:
    inputs, labels, _, _ = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Check if both "Sub subsection" and "Subsection" are not applicable
    if pd.isnull(labels):
        outputs = model_imrad(inputs)
        
        loss = criterion(outputs, labels)
        
        optimizer_imrad.zero_grad()
        loss.backward()
        optimizer_imrad.step()
        

# Start training loop
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch + 1}/{num_epochs}')
    
    # Train model_sub_subsection
    model_sub_subsection.train()  # Set the model to training mode
    for batch in train_dataloader:
        # Move batch to device
        inputs, _, _, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Check if "Sub subsection" is applicable
        if not pd.isnull(labels):
            # Forward pass
            outputs = model_sub_subsection(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer_sub_subsection.zero_grad()
            loss.backward()
            optimizer_sub_subsection.step()

    # Train model_subsection for data where "Sub subsection" is not applicable
    model_subsection.train()
    for batch in train_dataloader:
        inputs, _, labels, _ = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Check if "Subsection" is applicable and "Sub subsection" is not applicable
        if pd.isnull(labels):
            outputs = model_subsection(inputs)
            
            loss = criterion(outputs, labels)
            
            optimizer_subsection.zero_grad()
            loss.backward()
            optimizer_subsection.step()

    # Train model_imrad for data where both "Sub subsection" and "Subsection" are not applicable
    model_imrad.train()
    for batch in train_dataloader:
        inputs, labels, _, _ = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Check if both "Sub subsection" and "Subsection" are not applicable
        if pd.isnull(labels):
            outputs = model_imrad(inputs)
            
            loss = criterion(outputs, labels)
            
            optimizer_imrad.zero_grad()
            loss.backward()
            optimizer_imrad.step()