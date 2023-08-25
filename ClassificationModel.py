import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(device)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(64 * (353 // 8) * (353 // 8), 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 32),
        #     nn.ReLU(inplace=True),
        # )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (353 // 8) * (353 // 8), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Additional outputs
        self.fc_posture = nn.Linear(64, 1)

        self.relu = nn.ReLU()

        # Initialize weights
        # self.initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv_layers(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc_layers(x))

        return self.fc_posture(x)  # out_hand_presence

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                init.constant_(layer.weight, 0.1)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0.1)


class CustomDataset(Dataset):
    def __init__(self, image_folders, label_files, transform=None):
        self.image_folders = image_folders
        self.label_files = label_files
        self.transform = transform

        self.labels_dict = {}  # Dictionary to store (session_id, image_id) to label mapping

        for label_file in label_files:
            with open(label_file, 'r') as file:
                image_id = 0  # Initialize the image_id for the current session
                for line in file:
                    label = int(line.strip())  # Assuming the label file format is "label"
                    self.labels_dict[(session_id, image_id)] = label
                    image_id += 1

        self.image_paths = []
        for folder in image_folders:
            image_names = os.listdir(folder)
            folder_image_paths = [os.path.join(folder, name) for name in image_names if name.startswith("color_Frame")]
            self.image_paths.extend(folder_image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert('RGB')

        # # Convert the PIL image to a NumPy array and apply cropping
        image = np.array(image)
        # image = image[93:446, 132:485, :]

        # Extract the image ID from the image name
        image_id_str = image_name.split('#')[1].split('.')[0].strip()
        image_id = int(image_id_str)

        session_id = int(os.path.basename(os.path.dirname(image_path)).split('_')[1])
        if (session_id, image_id) in self.labels_dict:
            label = self.labels_dict[(session_id, image_id)]
        else:
            print("THERE IS AN ISSUE HERE")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float)

        return image, label


def show_samples(dataset, num_samples=1):
    fig, axs = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))

    for i in range(num_samples):
        image, label = dataset[i * 3]
        image = transforms.ToPILImage()(image)  # Convert the tensor image to a PIL Image
        label_value = label  # Get the numeric value of the label

        # Convert the PIL image to a Numpy array
        image = np.array(image)

        # Display the image with its associated label as the title
        axs[i].imshow(image)  # Use axs[i] to specify the correct subplot
        axs[i].set_title(f"Label: {label_value}")
        axs[i].axis('off')
    plt.show()


# Define the image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transformations as needed (e.g., normalization)
])

image_folders = [
    "/home/ac913/PycharmProjects/appChallenge/unlabeled_data",
]


# Create the label files for all the session folders
def create_label_files(image_folders):
    label_files = []
    for image_path in image_folders:
        label_files.append(os.path.join("/home/ac913/Python", f"keyframes_" + image_path.split("_")[-1].split("/")[0] + ".txt"))
    return label_files


combined_label_files = create_label_files(image_folders)
train_dataset = CustomDataset(image_folders, combined_label_files, transform=transform)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = CNNModel().to(device)  # Replace with your CNNModel initialization code

# show_samples(train_dataset, num_samples=10)

hand_presence_criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100

hand_presence_losses = []
# model.load_state_dict(torch.load("/home/alanjiach/catkin_ws/src/FANUC_Stream_Motion_Controller_CPP/ProbalisticCNN/TrainedModels/BestModel10Frompt2.pth"))
model.train()  # Set the model to training mode
for epoch in range(num_epochs):
    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_accuracy_hand_presence = 0.0
    threshold = 0.5
    total_correct_hand_presence = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        hand_presence_labels = labels  # Use the first column in the labels as the hand presence label
        hand_presence_labels = hand_presence_labels.float().to(device)

        optimizer.zero_grad()
        # Forward pass
        out_hand_presence = model(images.to(device))

        # Calculate the hand presence classification loss
        hand_presence_loss = hand_presence_criterion(out_hand_presence.squeeze(), hand_presence_labels)
        # Backpropagation and update weights
        hand_presence_loss.backward()
        optimizer.step()

        hand_presence_losses.append(hand_presence_loss.item())

        if epoch % 5 == 0:
            epochs = range(1, len(hand_presence_losses) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, hand_presence_losses, label="Hand Presence Loss")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
            plt.grid(True)
            # plt.show(block=False)
            plt.savefig(
                "/home/alanjiach/catkin_ws/src/FANUC_Stream_Motion_Controller_CPP/ProbalisticCNN/TrainedModels/training_loss_" + str(
                    current_datetime) + ".png")

            model_filename = f"model_{current_datetime}.pth"
            torch.save(model.state_dict(),
                       "/home/alanjiach/catkin_ws/src/FANUC_Stream_Motion_Controller_CPP/ProbalisticCNN/TrainedModels/AimingFor99/" + model_filename)
        print(batch_idx)

        predicted_hand_presence = (out_hand_presence >= threshold).int()
        # print(predicted_hand_presence)
        # print(hand_presence_labels)

        for i in range(predicted_hand_presence.shape[0]):
            if (predicted_hand_presence[i] == hand_presence_labels[i]):
                total_correct_hand_presence += 1
            total_samples += 1
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {hand_presence_loss.item():.4f}, Time: {current_datetime}")
    print(f"Training Accuracy: {total_correct_hand_presence / total_samples}")


test_image_folders = [
    "/home/ac913/PycharmProjects/appChallenge/",
]

test_combined_label_files = create_label_files(test_image_folders)
test_dataset = CustomLegoDataset(test_image_folders, test_combined_label_files, transform=transform)

# Create DataLoader for testing dataset
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

# model.load_state_dict(torch.load("/home/alanjiach/catkin_ws/src/FANUC_Stream_Motion_Controller_CPP/ProbalisticCNN/TrainedModels/model_2023-08-07 18:06:31.pth"))
model.eval()

total_accuracy_hand_presence = 0.0
num_batches = 0

with torch.no_grad():
    threshold = 0.5
    total_correct_hand_presence = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(test_loader):
        hand_presence_labels = labels  # Use the 5th column in the labels as the hand presence label
        hand_presence_labels = hand_presence_labels.float().to(device)

        # Forward pass
        out_hand_presence = model(images.to(device))
        predicted_hand_presence = (out_hand_presence >= threshold).int()
        # print(predicted_hand_presence)
        # print(hand_presence_labels)

        for i in range(predicted_hand_presence.shape[0]):
            if (predicted_hand_presence[i] == hand_presence_labels[i]):
                total_correct_hand_presence += 1
            total_samples += 1
        # correct_hand_presence = (predicted_hand_presence == hand_presence_labels).int().sum()
        # total_correct_hand_presence += c/orrect_hand_presence.item()
        # total_samples += labels.size(0)
        num_batches += 1

        # Accumulate loss for evaluation
        # total_accuracy_hand_presence += hand_presence_loss.item()
print(total_correct_hand_presence, total_samples)
# average_accuracy_hand_presence = total_accuracy_hand_presence / num_batches
accuracy_hand_presence = total_correct_hand_presence / total_samples

# print(f"Testing Loss Hand Presence (BCELoss): {average_accuracy_hand_presence:.4f}")
print(f"Accuracy Hand Presence: {accuracy_hand_presence:.4f}")
