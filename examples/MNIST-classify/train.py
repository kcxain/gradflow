import numpy as np
from model import MLPResNet, softmax_loss
import gradflow.nn as nn
import gradflow as df


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    hit, total = 0, 0
    loss_func = nn.SoftmaxLoss()
    loss_all = 0
    if opt is not None:
        model.train()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            opt.reset_grad()
            loss = loss_func(output, y)
            loss_all += loss.numpy()
            loss.backward()
            opt.step()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    else:
        model.eval()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            loss = loss_func(output, y)
            loss_all += loss.numpy()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    acc = (total - hit) / total
    return acc, loss_all / (idx + 1)

def train_mnist(batch_size=100, epochs=10, optimizer=df.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    train_data = df.data.MNISTDataset(
        data_dir + '/train-images-idx3-ubyte.gz',
        data_dir + '/train-labels-idx1-ubyte.gz'
    )
    test_data = df.data.MNISTDataset(
        data_dir + '/t10k-images-idx3-ubyte.gz',
        data_dir + '/t10k-labels-idx1-ubyte.gz',
    )
    train_loader = df.data.DataLoader(train_data, batch_size)
    test_loader = df.data.DataLoader(test_data, batch_size)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
    test_acc, test_loss = epoch(test_loader, model)
    return (train_acc, train_loss, test_acc, test_loss)

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (df.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (df.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: df.Tensor[np.float32]
            W2: df.Tensor[np.float32]
    """
    m = X.shape[0]
    for i in range(0, m, batch):
        X_batch = X[i : i+batch]
        y_batch = y[i : i+batch]
        X_batch = df.Tensor(X_batch)
        Z1 = df.ops.relu(X_batch @ W1)
        Z = Z1 @ W2
        y_one_hot = np.zeros(Z.shape, dtype="float32")
        y_one_hot[np.arange(Z.shape[0]),y_batch] = 1
        loss = softmax_loss(Z, df.Tensor(y_one_hot))
        loss.backward()

        W1 = (W1 - lr * W1.grad).detach()
        W2 = (W2 - lr * W2.grad).detach()
    return W1, W2
