import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.model_selection import train_test_split
# Assuming you have a custom dataset class for EEG and spectrogram data
from custom_dataset import CustomDataset

# Sample paths and labels (replace with your dataset paths and labels)
eeg_paths = ['path/to/eeg1.npy', 'path/to/eeg2.npy', ...]
spectrogram_paths = ['path/to/spectrogram1.png', 'path/to/spectrogram2.png', ...]
labels = [0, 1, ...]

# Split the data into train and validation sets
eeg_train, eeg_val, spec_train, spec_val, labels_train, labels_val = train_test_split(
    eeg_paths, spectrogram_paths, labels, test_size=0.2, random_state=42
)

# Initialize ViT feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create custom datasets
train_dataset = CustomDataset(eeg_train, spec_train, labels_train, transform)
val_dataset = CustomDataset(eeg_val, spec_val, labels_val, transform)

# Create DataLoader
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model.to(device)

# Set up optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        eeg_data, spec_data, labels = batch
        eeg_data, spec_data, labels = eeg_data.to(device), spec_data.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(eeg_data, spec_data)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

    # Validation after each epoch
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in val_dataloader:
            eeg_data, spec_data, labels = batch
            eeg_data, spec_data, labels = eeg_data.to(device), spec_data.to(device), labels.to(device)

            outputs = model(eeg_data, spec_data)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}")

# Save the trained model if needed
torch.save(model.state_dict(), 'model1.pth')
