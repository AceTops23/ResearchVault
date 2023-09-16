import pandas as pd
import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess your IMRAD dataset
file_paths = ['dataset/A SYLLABUS GENERATOR FOR THE COLLEGE.csv',]

imrad_data = pd.concat([pd.read_csv(path, encoding='Windows-1252') for path in file_paths], ignore_index=True)

# Drop rows with missing 'imrad_section' values
imrad_data.dropna(subset=['Label'], inplace=True)

# Assuming you have 'text' and 'imrad_section' columns in your CSV file
text_data = imrad_data['Text'].astype(str).tolist()
imrad_labels = imrad_data['Label'].tolist()

# Map IMRAD labels to numerical values
imrad_label_map = {'Introduction': 0, 'Method': 1, 'Result': 2, 'Discussion': 3}
imrad_labels = [imrad_label_map[label] for label in imrad_labels]

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Tokenize and preprocess data
encoded_data = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt')
input_ids = encoded_data['input_ids']
attention_mask = encoded_data['attention_mask']

# Convert labels to PyTorch tensor
labels = torch.Tensor(imrad_labels)

# Create data loader
dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)  # Define the test_loader

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids_batch, attention_mask_batch, labels_batch = batch
        outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
        logits = outputs.logits
        loss = loss_fn(logits, labels_batch.long())  # Cast labels to Long data type
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
predicted_labels_list = []
true_labels_list = []

with torch.no_grad():
    for batch in test_loader:
        input_ids_batch, attention_mask_batch, labels_batch = batch
        outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
        predicted_labels = outputs.logits.argmax(dim=1)
        # Collect predicted and true labels for later evaluation
        predicted_labels_list.extend(predicted_labels.tolist())
        true_labels_list.extend(labels_batch.tolist())
        
        total += labels_batch.size(0)
        correct += (predicted_labels == labels_batch).sum().item()
        # Print the shape and content of predicted_sections
        print("Predicted Labels Shape:", predicted_labels.shape)
        print("Predicted Labels Content:", predicted_labels)
        
        # Collect predicted and true labels for later evaluation
        predicted_labels_list.extend(predicted_labels.tolist())
        true_labels_list.extend(labels_batch.tolist())


accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# Calculate precision, recall, F1-score
report = classification_report(true_labels_list, predicted_labels_list, target_names=imrad_label_map.keys())
print("Classification Report:\n", report)

# Generate confusion matrix

conf_matrix = confusion_matrix(true_labels_list, predicted_labels_list)
print("Confusion Matrix:\n", conf_matrix)

# Save the entire model
save_dir = "models/BERT Model"  
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)  # Save the tokenizer too
torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))  # Save optimizer state
torch.save(loss_fn.state_dict(), os.path.join(save_dir, 'loss_fn.pt'))  # Save loss function state
torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': total_loss}, os.path.join(save_dir, 'checkpoint.pth'))  # Save training checkpoint
print(f"Trained model and related components saved to {save_dir}")