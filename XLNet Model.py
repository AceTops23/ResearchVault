import pandas as pd
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download the set of stop words the first time
import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

file_paths = ['Annotations.csv']

imrad_data = pd.concat([pd.read_csv(path, encoding='utf-8').replace('N/A', None) for path in file_paths], ignore_index=True)

# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if not word in stop_words]
    text = ' '.join(text)
    
    return text

# Apply preprocessing to 'Section Content'
imrad_data['Section Content'] = imrad_data['Section Content'].apply(preprocess_text)


imrad_data = pd.concat([pd.read_csv(path, encoding='utf-8').replace('N/A', None) for path in file_paths], ignore_index=True)

# Create a mapping from unique titles to integers
unique_titles = imrad_data['Title'].unique()
title_map = {title: i for i, title in enumerate(unique_titles)}

# Map titles to integers
imrad_titles = imrad_data['Title'].map(title_map).tolist()

imrad_data['Label'] = imrad_data[['IMRAD Sections', 'Subsection', 'Sub subsection']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

imrad_data = imrad_data[imrad_data['Label'].notna() & (imrad_data['Label'] != '')]

unique_labels = imrad_data['Label'].unique()
label_map = {label: i for i, label in enumerate(unique_labels)}
imrad_labels = imrad_data['Label'].map(label_map).tolist()

num_labels = len(unique_labels)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

text_data_list = []
for i, row in imrad_data.iterrows():
    text_data = row['Section Content']
    figure_references = row['Figure Reference']
    if isinstance(figure_references, str):
        figure_references = figure_references.split(', ')
        for figure_reference in figure_references:
            text_data = text_data.replace(figure_reference, f"<{figure_reference.replace(' ', '_').upper()}>")

    table_references = row['Table Reference']
    if isinstance(table_references, str):
        table_references = table_references.split(', ')
        for table_reference in table_references:
            text_data = text_data.replace(table_reference, f"<{table_reference.replace(' ', '_').upper()}>")
    
    text_data_list.append(text_data)

encoded_data = tokenizer(text_data_list, padding=True, truncation=True, return_tensors='pt')
input_ids = encoded_data['input_ids']
attention_mask = encoded_data['attention_mask']

labels = torch.Tensor(imrad_labels)

assert input_ids.shape[0] == attention_mask.shape[0] == labels.shape[0]

dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)  

optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids_batch, attention_mask_batch, labels_batch = batch
        outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
        logits = outputs.logits
        loss = loss_fn(logits, labels_batch.long())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

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
        predicted_labels_list.extend(predicted_labels.tolist())
        true_labels_list.extend(labels_batch.tolist())
        
        total += labels_batch.size(0)
        correct += (predicted_labels == labels_batch).sum().item()
        print("Predicted Labels Shape:", predicted_labels.shape)
        print("Predicted Labels Content:", predicted_labels)
        
        predicted_labels_list.extend(predicted_labels.tolist())
        true_labels_list.extend(labels_batch.tolist())


accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

unique_test_labels = set(true_labels_list + predicted_labels_list)

target_names = [label for label in label_map.keys() if label_map[label] in unique_test_labels]

report = classification_report(true_labels_list, predicted_labels_list, target_names=target_names)
print("Classification Report:\n", report)


conf_matrix = confusion_matrix(true_labels_list, predicted_labels_list)
print("Confusion Matrix:\n", conf_matrix)

save_dir = "models/test BERT Model"     
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)  
torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))  
torch.save(loss_fn.state_dict(), os.path.join(save_dir, 'loss_fn.pt')) 
torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': total_loss}, os.path.join(save_dir, 'checkpoint.pth'))  # Save training checkpoint
print(f"Trained model and related components saved to {save_dir}")