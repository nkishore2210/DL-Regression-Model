# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: KISHORE N

### Register Number: 212222240049

```
# Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Generate Dataset
torch.manual_seed(0)
x = torch.arange(1, 51, dtype=torch.float32).view(-1, 1)
noise = torch.randn(x.size()) * 5
y = 2 * x + 1 + noise

# Step 2: Define Model
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

model = Model(1, 1)

# Step 3: Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Step 4: Train the Model
loss_values = []
epochs = 100

for epoch in range(epochs):
    model.train()
    
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Step 5 & 6: Visualizations
model.eval()
with torch.no_grad():
    predicted = model(x)

fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

# Loss Curve
axs[0].plot(loss_values, label='Training Loss', color='blue')
axs[0].set_title('Training Loss vs Iterations')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('MSE Loss')
axs[0].legend()
axs[0].grid(True)

# Best Fit Line
axs[1].scatter(x.numpy(), y.numpy(), label='Original Data', color='blue')
axs[1].plot(x.numpy(), predicted.numpy(), label='Best Fit Line', color='red')
axs[1].set_title('Best Fit Line')
axs[1].set_xlabel('Input X')
axs[1].set_ylabel('Target Y')
axs[1].legend()
axs[1].grid(True)

plt.show()

# Step 7: Make Prediction
new_input = torch.tensor([[55.0]])
predicted_output = model(new_input).item()
print("\n" + "="*40)
print(f"Prediction for input {new_input.item()}: {predicted_output:.2f}")
print("="*40)


```

### Dataset Information
![image](https://github.com/user-attachments/assets/2c88aa6b-54e2-46fb-bfcb-81d282d91efd)



## OUTPUT
### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/0887eb70-6269-46f2-a10d-6d2d825cfb84)

### Best Fit line plot
![image](https://github.com/user-attachments/assets/e16d5596-394f-498a-8a5f-54556cfd5e9d)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/2310b2d6-776a-4d7a-93a4-28d8502a1b86)


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
