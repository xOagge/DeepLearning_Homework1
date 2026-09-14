#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import time
import utils


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """ Define a vanilla multiple-layer FFN with `layers` hidden layers 
        Args:
            n_classes (int)
            n_features (int)
            hidden_size (int)
            layers (int)
            activation_type (str)
            dropout (float): dropout probability
        """
        super().__init__()

        #  --- Define activation functions mapping ---

        # Map activation_type (str) to PyTorch definitions of respective activation functions
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
        }
        activation_function = None
        # verify if the activation_type (str) is defined in the map. Otherwise, default ReLU
        if activation_type.lower() not in activation_map:
            print(
                f"activation_type: '{activation_type}', is not defined. "
                f"activation_type will be defaulted to ReLU."
                f"Available options: {list(activation_map.keys())}"
            )
            activation_function = activation_map['relu']
        #define activation function respective to activation_type (str) by taking its .lower() version
        else:
            activation_function = activation_map[activation_type.lower()]

        #  --- Define number of inputs and outputs for each layer---

        #first Hidden layer: Inputs n_features, the number of features of a single input.
        #Other Hidden layer: Inputs hidden_size, input layer and hidden layer output hidden_size number of outputs
        #Other Hidden layer: Outputs hidden_size, which is the number of neurons in the hidden layer, so it strictly outputs hidden_size number of outputs
        #Output layer: Inputs hidden_size, number of outputs of the last hidden layer
        #Output layer: Outputs n_classes, number of classes we are classifying with this FFN
        in_sizes = [n_features] + [hidden_size] * layers
        out_sizes = [hidden_size] * layers + [n_classes]

        #  --- Build the Network structure ---
        #holds the sequence of "commands" that should be done in the model (ex. Linear, activation, dropout, Linear, ...)
        ffn_sequence  = []

        # Iterate through input and output sizes to generate linear layers with defined n_in inputs and n_out outputs
        # i is the index of the layer we are creating in a loop
        for i, (n_in, n_out) in enumerate(zip(in_sizes, out_sizes)):
            # create linear layer and add to network_layers
            ffn_sequence.append(nn.Linear(n_in, n_out))
            
            # if the current layer index is equal to the last possible index, given by len(in_sizes) - 1 or len(out_sizes) - 1
            # then we are currently iterating the output layer, which we dont want to apply Dropout or activation function after
            is_output_layer = (i == len(in_sizes) - 1)

            #if not iterating output layer, add activation and dropout
            if not is_output_layer:
                ffn_sequence.append(activation_function)
                if dropout > 0.0: ffn_sequence.append(nn.Dropout(dropout))

        self.ffn = nn.Sequential(*ffn_sequence)

    #wont be called directly, but model(x) takes use of this forward method. default definition in nn.Module exists but not built
    def forward(self, x, **kwargs):
        """ Compute a forward pass through the FFN
        Args:
            x (torch.Tensor): a batch of examples (batch_size x n_features)
        Returns:
            scores (torch.Tensor)
        """
        #outpu will be format (batch_size x n_classes). On each row, each value is a score associated
        #witha  respective class n_j, j the column index
        return self.ffn(x)
    
    
def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """ Do an update rule with the given minibatch
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
    Returns:
        loss (float)
    """

    #clean the optimizer before calculating gradient for the current batch
    optimizer.zero_grad()
    # calls forward method which activates the sequential. shape: (n_examples x n_classes), values are scores for the classes
    outputs = model(X, **kwargs)
    # calculate total loss between predictions and true labels. We will return this value to see loss before training
    # loss is a 0-dimensions tensor
    loss = criterion(outputs, y)
    #applying backward() to tensor loss, torch backdates the used tensors to calculate loss, resulting in the dL/dW for each layer
    loss.backward()
    #updates model parameters according to W -> W - n * dL/dW. optimizer has access to the model parameters in its definition
    optimizer.step()

    return loss.item() #item is so that instead of 0-dim tensor we get float


def predict(model, X):
    """ Predict the labels for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
    Returns:
        preds: (n_examples)
    """
    # calls forward method which activates the sequential. shape: (n_examples x n_classes), values are scores for the classes
    scores = model(X)
    #the prediction is the argmax of the classes scores for a given input. (n_examples).
    # (row, col), dim = -1 is col, meaning that the comparison will be made along columns, comparing elements in the same row
    predicted_labels = scores.argmax(dim=-1)
    return predicted_labels


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """ Compute the loss and the accuracy for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        criterion: loss function
    Returns:
        loss, accuracy (Tuple[float, float])
    """

    #changes the model to evaluation mode, deactivates dropout and metrics calculated that can be used for optimization
    model.eval()
    #get output scores. shape: (n_examples x n_classes)
    outputs = model(X)
    # calculate total loss between predictions and true labels. We will return this value to see loss before training
    # loss is a 0-dimensions tensor, will return loss.item() so its a float
    loss = criterion(outputs, y)
    #more efficient to obtain predictions like this than using predict function, would recalculate outputs. shape: (n_examples)
    y_hat = outputs.argmax(dim=-1)
    #number of correct predictions. shapes match equal: (n_examples)
    n_correct_predictions = (y == y_hat).sum().item()
    #accuracy: number of correct predictions / number of total predictions. could do y_hat.shape[0] or y.shape[0]
    accuracy = n_correct_predictions / y_hat.shape[0]
    return loss.item(), accuracy

def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=30, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001) #learning rate -----------------
    parser.add_argument('-l2_decay', type=float, default=0.0) # l2 value / weight decay
    parser.add_argument('-dropout', type=float, default=0.0) #dropout value ---------------
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='emnist-letters.npz',)
    parser.add_argument('--model', type=str, default='ffn', help='Model name for plotting')
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    train_X, train_y = dataset.X, dataset.y # (changed) train_X -> X, train_y -> y 
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    print(f"N features: {n_feats}")
    print(f"N classes: {n_classes}")

    # initialize the model
    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        opt.hidden_size,
        opt.layers,
        opt.activation,
        opt.dropout
    )

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    start = time.time()

    model.eval()
    initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
    initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
    train_losses.append(initial_train_loss)
    train_accs.append(initial_train_acc)
    valid_losses.append(initial_val_loss)
    valid_accs.append(initial_val_acc)
    print('initial val acc: {:.4f}'.format(initial_val_acc))

    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        model.train()
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        model.eval()
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print('train loss: {:.4f} | val loss: {:.4f} | val acc: {:.4f}'.format(
            epoch_train_loss, val_loss, val_acc
        ))

        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print('Final test acc: {:.4f}'.format(test_acc))

    # plot
    config = (
        f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
        f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
        f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
    )

    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }

    accs = {
        "Train Accuracy": train_accs,
        "Valid Accuracy": valid_accs
    }

    # slice to skip initial evaluation
    plot_epochs = epochs  # epochs already starts from 1
    plot_losses = {k: v[1:] for k, v in losses.items()}
    plot_accs   = {k: v[1:] for k, v in accs.items()}

    output_folder = "Q2_outputs"

    plot(plot_epochs, plot_losses, filename=f'{output_folder}/losses-{config}.pdf')
    plot(plot_epochs, plot_accs, filename=f'{output_folder}/accs-{config}.pdf')
    print(f"Final Training Accuracy: {train_accs[-1]:.4f}")
    print(f"Best Validation Accuracy: {max(valid_accs):.4f}")


if __name__ == '__main__':
    main()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-epochs', default=30, type=int,
#                         help="""Number of epochs to train for. You should not
#                         need to change this value for your plots.""")
#     parser.add_argument('-batch_size', default=64, type=int,
#                         help="Size of training batch.")
#     parser.add_argument('-hidden_size', type=int, default=32)
#     parser.add_argument('-layers', type=int, default=1)
#     parser.add_argument('-learning_rate', type=float, default=0.001)
#     parser.add_argument('-l2_decay', type=float, default=0.0)
#     parser.add_argument('-dropout', type=float, default=0.0)
#     parser.add_argument('-activation',
#                         choices=['tanh', 'relu'], default='relu')
#     parser.add_argument('-optimizer',
#                         choices=['sgd', 'adam'], default='sgd')
#     parser.add_argument('-data_path', type=str, default='emnist-letters.npz',)
#     parser.add_argument('--model', type=str, default='ffn', help='Model name for plotting')
#     opt = parser.parse_args()

#     utils.configure_seed(seed=42)

#     data = utils.load_dataset(opt.data_path)
#     dataset = utils.ClassificationDataset(data)
#     train_dataloader = DataLoader(
#         dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
#     train_X, train_y = dataset.X, dataset.y # (changed) train_X -> X, train_y -> y 
#     dev_X, dev_y = dataset.dev_X, dataset.dev_y
#     test_X, test_y = dataset.test_X, dataset.test_y

#     n_classes = torch.unique(dataset.y).shape[0]  # 26
#     n_feats = dataset.X.shape[1]

#     print(f"N features: {n_feats}")
#     print(f"N classes: {n_classes}")

#     # initialize the model
#     model = FeedforwardNetwork(
#         n_classes,
#         n_feats,
#         opt.hidden_size,
#         opt.layers,
#         opt.activation,
#         opt.dropout
#     )

#     # get an optimizer
#     optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

#     optim_cls = optims[opt.optimizer]
#     optimizer = optim_cls(
#         model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
#     )

#     # get a loss criterion
#     criterion = nn.CrossEntropyLoss()

#     # training loop
#     epochs = torch.arange(1, opt.epochs + 1)
#     train_losses = []
#     train_accs = []
#     valid_losses = []
#     valid_accs = []

#     start = time.time()

#     model.eval()
#     initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
#     initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
#     train_losses.append(initial_train_loss)
#     train_accs.append(initial_train_acc)
#     valid_losses.append(initial_val_loss)
#     valid_accs.append(initial_val_acc)
#     print('initial val acc: {:.4f}'.format(initial_val_acc))

#     for ii in epochs:
#         print('Training epoch {}'.format(ii))
#         epoch_train_losses = []
#         model.train()
#         for X_batch, y_batch in train_dataloader:
#             loss = train_batch(
#                 X_batch, y_batch, model, optimizer, criterion)
#             epoch_train_losses.append(loss)

#         model.eval()
#         epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
#         _, train_acc = evaluate(model, train_X, train_y, criterion)
#         val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

#         print('train loss: {:.4f} | val loss: {:.4f} | val acc: {:.4f}'.format(
#             epoch_train_loss, val_loss, val_acc
#         ))

#         train_losses.append(epoch_train_loss)
#         train_accs.append(train_acc)
#         valid_losses.append(val_loss)
#         valid_accs.append(val_acc)

#     elapsed_time = time.time() - start
#     minutes = int(elapsed_time // 60)
#     seconds = int(elapsed_time % 60)
#     print('Training took {} minutes and {} seconds'.format(minutes, seconds))

#     _, test_acc = evaluate(model, test_X, test_y, criterion)
#     print('Final test acc: {:.4f}'.format(test_acc))

#     # plot
#     config = (
#         f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
#         f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
#         f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
#     )

#     losses = {
#         "Train Loss": train_losses,
#         "Valid Loss": valid_losses,
#     }

#     accs = {
#         "Train Accuracy": train_accs,
#         "Valid Accuracy": valid_accs
#     }

#     # slice to skip initial evaluation
#     plot_epochs = epochs  # epochs already starts from 1
#     plot_losses = {k: v[1:] for k, v in losses.items()}
#     plot_accs   = {k: v[1:] for k, v in accs.items()}

#     output_folder = "Q2_outputs"

#     plot(plot_epochs, plot_losses, filename=f'{output_folder}/losses-{config}.pdf')
#     plot(plot_epochs, plot_accs, filename=f'{output_folder}/accs-{config}.pdf')
#     print(f"Final Training Accuracy: {train_accs[-1]:.4f}")
#     print(f"Best Validation Accuracy: {max(valid_accs):.4f}")
