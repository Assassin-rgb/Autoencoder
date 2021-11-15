import torch
import torch.nn as nn
import torch.nn.functional as F


# flatten features
def flatten(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class AutoEncoder(nn.Module):
    def __init__(self, mode):
        super(AutoEncoder, self).__init__()
        if mode == 1:
            # define layers model1
            self.encoder_fc1 = nn.Linear(784, 256)
            self.encoder_fc2 = nn.Linear(256, 128)
            self.decoder_fc1 = nn.Linear(128, 256)
            self.decoder_fc2 = nn.Linear(256, 784)
            # forward
            self.forward = self.model1
        elif mode == 2:
            # define layers model2
            self.encoder_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten(start_dim=1)
            self.encoder_fc1 = nn.Linear(32*7*7, 100)
            self.decoder_fc1 = nn.Linear(100, 32*7*7)
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 7, 7))
            self.decoder_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.decoder_conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.decoder_conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
            # forward
            self.forward = self.model2
        else:
            print("Invalid mode ", mode, " selected. Select between 1-2")
            exit(0)

    # model using only fully connected layers
    def model1(self, x):
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        x = F.relu(self.decoder_fc1(x))
        x = torch.sigmoid(self.decoder_fc2(x))
        return x

    # model using convolution layers
    def model2(self, x):
        x = self.pool1(F.relu(self.encoder_conv1(x)))
        x = self.pool2(F.relu(self.encoder_conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.decoder_fc1(x))
        x = self.unflatten(x)
        x = F.relu(self.decoder_conv1(x))
        x = self.upsample1(x)
        x = F.relu(self.decoder_conv2(x))
        x = self.upsample2(x)
        x = torch.sigmoid(self.decoder_conv3(x))
        return x
