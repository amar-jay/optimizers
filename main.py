import pytorch_lightning
import torch
from dataset import get_dataloader
from model import CNNClassifier
# print(pytorch_lightning.cuda)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Device: ", device)
# print("\n\n")
# model = CNNClassifier()
# model.to(device)
# print("num of parameters: ", model.num_parameters())
# res = model(torch.randn((4, 1, 28,28)))
# print("result", res)
train_loader, _ = get_dataloader()
for inputs, labels in train_loader:
    print(inputs.shape, labels.shape)
    break