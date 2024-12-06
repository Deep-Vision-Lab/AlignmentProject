import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


vector_length = 512

# Slicing function to create windowed views without resizing
def slicing_window(image, window_width=10):
    batches, channels, height, width = image.shape
    sliced_windows = []

    for i in range(batches):
        sliced_windows_per_batch = []
        for x in range(0, width, window_width):
            window_end = x + window_width
            window = image[i, :, :, x:window_end]
            if window.shape[2] < window_width:  # Pad if the window is smaller than required
                # Move zero_array to the same device as window
                zero_array = torch.zeros(channels, height, window_width - window.shape[2], device=window.device)
                window = torch.cat([window, zero_array], dim=2)
            sliced_windows_per_batch.append(window)
        sliced_windows.append(torch.stack(sliced_windows_per_batch))

    return torch.stack(sliced_windows)



# Define simpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self,height,width):
        super(SimpleCNN, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)
        
        # Final fully connected layer to get the desired output shape [numofwindows, 512]
        # Assuming after pooling the spatial size is small enough
        self.fc = nn.Linear(256 * height * width, vector_length)  # Modify the size depending on the input image size

    def forward(self, x):
        # Input x shape: [numofwindows, channels, height, width]
        x = F.relu(self.conv1(x))  # Apply first conv layer
        
        x = F.relu(self.conv2(x))  # Apply second conv layer

        # Flatten the output to feed into the fully connected layer
        x = torch.flatten(x, start_dim=1)

        # Apply the final fully connected layer
        x = self.fc(x)
        
        return x  # Output shape: [numofwindows, 512]


# Initialize VGG model with adaptive pooling
def initialize_vgg_with_adaptive_pooling(vgg_type='vgg16'):
    if vgg_type == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif vgg_type == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise ValueError("VGG type not supported. Choose 'vgg16' or 'vgg19'.")

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, vector_length)
    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.MaxPool2d):
            model.features[i] = nn.AdaptiveAvgPool2d((7, 7))
    return model


# Initialize ResNet model with output dimension 512
def initialize_resnet(resnet_type='resnet18'):
    if resnet_type == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif resnet_type == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif resnet_type == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif resnet_type == 'resnet101':
        model = models.resnet101(pretrained=True)
    else:
        raise ValueError("ResNet type not supported. Choose 'resnet18', 'resnet34', 'resnet50', or 'resnet101'.")

    model.fc = nn.Linear(model.fc.in_features, vector_length)
    return model


# Model selector function
def select_model(model_name, height, width):
    if model_name == 'simpleCNN':
        return SimpleCNN(height, width)
    elif model_name == 'ResNet18':
        return initialize_resnet('resnet18')
    elif model_name == 'ResNet34':
        return initialize_resnet('resnet34')
    elif model_name == 'ResNet50':
        return initialize_resnet('resnet50')
    elif model_name == 'ResNet101':
        return initialize_resnet('resnet101')
    elif model_name == 'vgg16':
        return initialize_vgg_with_adaptive_pooling('vgg16')
    elif model_name == 'vgg19':
        return initialize_vgg_with_adaptive_pooling('vgg19')
    else:
        raise ValueError(
            "Model not supported. Choose from 'simpleCNN', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'vgg16' or 'vgg19'.")

def Run_CNN(model,seq1,seq2):

    batches_1 = [model(seq1[i]) for i in range(seq1.shape[0])]
    batch1 = torch.stack(batches_1, dim=0)

    batches_2 = [model(seq2[i]) for i in range(seq2.shape[0])]
    batch2 = torch.stack(batches_2, dim=0)

    return batch1, batch2

# # Sample input tensors
# seq1 = torch.randn(4, 3, 50, 150)
# seq2 = torch.randn(4, 3, 50, 154)
# print(f'Before slicing window: {seq1.shape}')
#
# # Slice windows without resizing
# seq1 = slicing_window(seq1)
# seq2 = slicing_window(seq2)
# print(f'After slicing window: {seq1.shape}')
#

#
# print(f'batch1 shape: {batch1.shape}')
# print(f'batch2 shape: {batch2.shape}')