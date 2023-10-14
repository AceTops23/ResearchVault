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

df['Subsection'].replace('None', 'Not Specified', inplace=True)
df['Sub subsection'].replace('None', 'Not Specified', inplace=True)

# Initialize separate LabelEncoders for each column
imrad_label_encoder = LabelEncoder()
subsection_label_encoder = LabelEncoder()
subsubsection_label_encoder = LabelEncoder()

# Fit the LabelEncoders to your 'IMRAD Section', 'Subsection', and 'Sub subsection' columns
df['IMRAD Section'] = imrad_label_encoder.fit_transform(df['IMRAD Section'])
df['Subsection'] = subsection_label_encoder.fit_transform(df['Subsection'])
df['Sub subsection'] = subsubsection_label_encoder.fit_transform(df['Sub subsection'])

# Define a custom dataset for BERT input
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Section Content']
        subsection = self.data.iloc[idx]['Subsection']
        subsubsection = self.data.iloc[idx]['Sub subsection']
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'IMRAD Section': self.data.iloc[idx]['IMRAD Section'],  # Include 'IMRAD Section' here
            'Subsection': subsection,  # Include 'Subsection' in the batch data
            'Sub subsection': subsubsection  # Include 'Subsection' in the batch data
        }

# Define a hierarchical classification model
class HierarchicalBERT(torch.nn.Module):
    def __init__(self, num_imrad_classes, num_subsection_classes, num_subsubsection_classes):
        super(HierarchicalBERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_imrad_classes)
        self.subsection_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_subsection_classes)
        self.subsubsection_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_subsubsection_classes)

        # Store the labels used during training
        self.imrad_labels = []
        self.subsection_labels = []

        self.subsubsection_labels = []

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

# Split the data into training and validation sets
print("Splitting the data into training and validation sets...")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create label encoders for each level of hierarchy
imrad_label_encoder = LabelEncoder()
subsection_label_encoder = LabelEncoder()
subsubsection_label_encoder = LabelEncoder()

# Fit the label encoders to your training dataset
print("Fitting label encoders to training dataset...")
train_df['IMRAD Section'] = imrad_label_encoder.fit_transform(train_df['IMRAD Section'])
train_df['Subsection'] = subsection_label_encoder.fit_transform(train_df['Subsection'])
train_df['Sub subsection'] = subsubsection_label_encoder.fit_transform(train_df['Sub subsection'])

# Transform the labels in the validation dataset
print("Transforming labels in the validation dataset...")
val_df['IMRAD Section'] = val_df['IMRAD Section'].map(lambda s: imrad_label_encoder.transform([s])[0] if s in imrad_label_encoder.classes_ else -1)
val_df['Subsection'] = val_df['Subsection'].map(lambda s: subsection_label_encoder.transform([s])[0] if s in subsection_label_encoder.classes_ else -1)
val_df['Sub subsection'] = val_df['Sub subsection'].map(lambda s: subsubsection_label_encoder.transform([s])[0] if s in subsubsection_label_encoder.classes_ else -1)

# Create dataloaders
batch_size = 16  # Adjust the batch size based on your hardware
train_dataset = CustomDataset(train_df, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(val_df, tokenizer, max_length)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
print("Initializing the model...")
model = HierarchicalBERT(num_imrad_classes, num_subsection_classes, num_subsubsection_classes)
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

if __name__ == "__main__":
    # Training code
    num_epochs = 15  # Adjust this based on your needs
    print("Training the model...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Extract the numerical label columns as PyTorch tensors
            imrad_labels_batch = torch.tensor(batch['IMRAD Section'], dtype=torch.long).to(device)
            subsection_labels_batch = torch.tensor(batch['Subsection'], dtype=torch.long).to(device)
            subsubsection_labels_batch = torch.tensor(batch['Sub subsection'], dtype=torch.long).to(device)

            # Store the labels in the model
            model.imrad_labels.extend(imrad_labels_batch.tolist())
            model.subsection_labels.extend(subsection_labels_batch.tolist())
            model.subsubsection_labels.extend(subsubsection_labels_batch.tolist())

            optimizer.zero_grad()

            imrad_logits, subsection_logits, subsubsection_logits = model(input_ids, attention_mask)

            imrad_loss = criterion(imrad_logits, imrad_labels_batch)
            subsection_loss = criterion(subsection_logits, subsection_labels_batch)
            subsubsection_loss = criterion(subsubsection_logits, subsubsection_labels_batch)

            loss = imrad_loss + subsection_loss + subsubsection_loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Print debug information
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

    # Initialize lists to store true and predicted labels
    true_imrad_labels = []
    true_subsection_labels = []
    true_subsubsection_labels = []

    predicted_imrad_labels = []
    predicted_subsection_labels = []
    predicted_subsubsection_labels = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Extract the labels from the batch
            imrad_labels_batch = torch.tensor(batch['IMRAD Section'], dtype=torch.long).to(device)
            subsection_labels_batch = torch.tensor(batch['Subsection'], dtype=torch.long).to(device)
            subsubsection_labels_batch = torch.tensor(batch['Sub subsection'], dtype=torch.long).to(device)
            
            imrad_logits, subsection_logits, subsubsection_logits = model(input_ids, attention_mask)

            # Append the true labels
            true_imrad_labels.extend(imrad_labels_batch.tolist())
            true_subsection_labels.extend(subsection_labels_batch.tolist())
            true_subsubsection_labels.extend(subsubsection_labels_batch.tolist())

            # Append the predicted labels
            predicted_imrad_labels.extend(imrad_logits.argmax(dim=1).tolist())
            predicted_subsection_labels.extend(subsection_logits.argmax(dim=1).tolist())
            predicted_subsubsection_labels.extend(subsubsection_logits.argmax(dim=1).tolist())

    # Calculate accuracy, precision, recall, and F1-score for each level of hierarchy
    accuracy_imrad = accuracy_score(true_imrad_labels, predicted_imrad_labels)
    precision_imrad = precision_score(true_imrad_labels, predicted_imrad_labels, average='weighted')
    recall_imrad = recall_score(true_imrad_labels, predicted_imrad_labels, average='weighted')
    f1_imrad = f1_score(true_imrad_labels, predicted_imrad_labels, average='weighted')

    accuracy_subsection = accuracy_score(true_subsection_labels, predicted_subsection_labels)
    precision_subsection = precision_score(true_subsection_labels, predicted_subsection_labels, average='weighted')
    recall_subsection = recall_score(true_subsection_labels, predicted_subsection_labels, average='weighted')
    f1_subsection = f1_score(true_subsection_labels, predicted_subsection_labels, average='weighted')

    accuracy_subsubsection = accuracy_score(true_subsubsection_labels, predicted_subsubsection_labels)
    precision_subsubsection = precision_score(true_subsubsection_labels, predicted_subsubsection_labels, average='weighted')
    recall_subsubsection = recall_score(true_subsubsection_labels, predicted_subsubsection_labels, average='weighted')
    f1_subsubsection = f1_score(true_subsubsection_labels, predicted_subsubsection_labels, average='weighted')

    # Print the evaluation metrics
    print("IMRAD Classification Metrics:")
    print(f"Accuracy: {accuracy_imrad:.4f}")
    print(f"Precision: {precision_imrad:.4f}")
    print(f"Recall: {recall_imrad:.4f}")
    print(f"F1-Score: {f1_imrad:.4f}")

    print("\nSubsection Classification Metrics:")
    print(f"Accuracy: {accuracy_subsection:.4f}")
    print(f"Precision: {precision_subsection:.4f}")
    print(f"Recall: {recall_subsection:.4f}")
    print(f"F1-Score: {f1_subsection:.4f}")

    print("\nSub-subsection Classification Metrics:")
    print(f"Accuracy: {accuracy_subsubsection:.4f}")
    print(f"Precision: {precision_subsubsection:.4f}")
    print(f"Recall: {recall_subsubsection:.4f}")
    print(f"F1-Score: {f1_subsubsection:.4f}")

    # Make predictions on the entire dataset
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in DataLoader(val_dataset, batch_size=batch_size, shuffle=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            imrad_logits, subsection_logits, subsubsection_logits = model(input_ids, attention_mask)

            imrad_predictions = imrad_logits.argmax(dim=1).tolist()
            subsection_predictions = subsection_logits.argmax(dim=1).tolist()
            subsubsection_predictions = subsubsection_logits.argmax(dim=1).tolist()

            # Append the predictions to the list
            predictions.extend(zip(imrad_predictions, subsection_predictions, subsubsection_predictions))

    # Assign the predictions back to the DataFrame
    val_df['Predicted_IMRAD'] = [imrad_class for imrad_class, _, _ in predictions]
    val_df['Predicted_Subsection'] = [subsection_class for _, subsection_class, _ in predictions]
    val_df['Predicted_Subsubsection'] = [subsubsection_class for _, _, subsubsection_class in predictions]

    # Save the DataFrame with predictions to a CSV file
    val_df.to_csv('Annotations_with_predictions.csv', index=False)
    torch.save(model.state_dict(), "fine_tuned_bert_model.pth")