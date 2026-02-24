# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Include the problem statement and Dataset


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.


## PROGRAM

### Name: BAVYA SRI B

### Register Number: 212224230034
```PY
# Load Pretrained Model and Modify for Transfer Learning

model=models.vgg19(weights=VGG19_Weights.DEFAULT)


# Modify the final fully connected layer to match the dataset classes

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)


# Include the Loss function and optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss=0.0
        with torch.no_grad():
          for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())
            val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Dharini.S")
    print("Register Number: 212224040072")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
# Train the model
# Write your code here
train_model(model,train_loader,test_loader)

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="817" height="801" alt="image" src="https://github.com/user-attachments/assets/c7bda7fb-eead-487a-9d59-06d94d016cef" />


## Confusion Matrix

<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/d4e85e5d-0360-47a1-8d95-1b00fbecb05f" />


## Classification Report

<img width="471" height="220" alt="image" src="https://github.com/user-attachments/assets/91eab271-7982-4ec9-aeaa-6dc17f79c26f" />

### New Sample Data Prediction

<img width="398" height="401" alt="image" src="https://github.com/user-attachments/assets/a2f0fc9f-f0d4-478a-82bb-ab93b7119bed" />

<img width="382" height="391" alt="image" src="https://github.com/user-attachments/assets/3da37e15-ec11-4606-8f2f-d0c378db0ebd" />

## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
