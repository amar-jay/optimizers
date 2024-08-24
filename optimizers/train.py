from simple_dataset import Data
from simple_model import LinearRegressionModel
import torch
import torch.nn.functional as F 
from sgd import StochasticGradientDescent
from torch.utils.data import DataLoader



dataset = Data()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
model = LinearRegressionModel(1,1)
optimizer = StochasticGradientDescent(model.parameters(), lr=0.01)


NUM_EPOCHS = 100

# training loop
for epoch in range(NUM_EPOCHS):
    for x, y in dataloader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.1:
            break
        if loss is not None:
            print(f'Epoch {epoch}, Loss {loss.item()}')


# inference
accuracy = 0

for idx in torch.linspace(-10, 10, 100):
    x = torch.tensor([idx])
    y_hat = model(x)
    actual = torch.tensor(dataset[idx])
    accuracy += torch.abs(y_hat - actual)
    print(f'Prediction: {y_hat.item()}, Actual: {actual.item()}')

print(f'Accuracy: {accuracy*100/100}%')
