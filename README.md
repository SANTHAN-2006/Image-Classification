# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

![image](https://github.com/user-attachments/assets/f2921d2e-27ef-4d74-9a68-527391570d63)

## DESIGN STEPS

### STEP 1: Problem Statement
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

### STEP 2:Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.
### STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.
### STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.
### STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.
### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.


## PROGRAM

### Name: K SANTHAN KUMAR
### Register Number: 212223240065
```python
class CNNClassifier(nn.Module):
    def __init__(self): 
        super(CNNClassifier, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) 
        x = self.pool(torch.relu(self.conv2(x))) 
        x = self.pool(torch.relu(self.conv3(x))) 
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        x = self.fc3(x)
        return x
```

```python
# Initialize model
model = CNNClassifier()

# Initialize loss function
criterion = nn.CrossEntropyLoss()

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
# Train the model

def train_model(model, train_loader, criterion, optimizer, num_epochs=3, device="cuda"):
    print('Name: K SANTHAN KUMAR')
    print('Register Number: 212223240065')

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

![image](https://github.com/user-attachments/assets/d30e6a59-55e5-4a97-ac50-df2c7a292f76)

### Confusion Matrix

![image](https://github.com/user-attachments/assets/5b2a52c7-476d-4f23-888c-7769b3c76812)

### Classification Report

![image](https://github.com/user-attachments/assets/3bda535b-031b-46c4-a4c1-906f34f50738)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/987544be-4ab0-429c-85e3-80d63daa0d91)

## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
