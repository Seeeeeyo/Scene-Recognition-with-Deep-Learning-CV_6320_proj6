import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
    to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one

    Download pretrained alexnet using pytorch's API (Hint: see the import
    statements)
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################
    pretrained_model = alexnet(pretrained=True)

    self.cnn_layers = pretrained_model.features

    fc_layers = list(pretrained_model.classifier.children())[:-1]

    fc_layers.append(nn.Linear(in_features=4096, out_features=15))

    self.fc_layers = nn.Sequential(*fc_layers)

    for param in self.cnn_layers.parameters():
        param.requires_grad = False
    for param in self.fc_layers[:-1].parameters():
        param.requires_grad = False

    for param in self.fc_layers[-1].parameters():
        param.requires_grad = True

    # Define the loss function
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    ###########################################################################
    # Student code end
    ###########################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''

    model_output = None
    x = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images

    ###########################################################################
    # Student code begin
    ###########################################################################
    x = self.cnn_layers(x)
    x = x.view(x.size(0), -1)
    model_output = self.fc_layers(x)
    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
