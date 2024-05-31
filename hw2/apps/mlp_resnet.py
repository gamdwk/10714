import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), 
                       nn.Linear(hidden_dim, dim), norm(dim))
    return nn.Sequential(nn.Residual(fn), nn.ReLU()) 
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for i in range(num_blocks)], 
        nn.Linear(hidden_dim, num_classes))
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    loss_fn = nn.SoftmaxLoss()
    error = 0
    n = 0
    losses = 0.0
    for batch in dataloader:
        X, y = batch[0], batch[1]
        h = model(X)
        loss = loss_fn(h, y)
        
        n += X.shape[0]
        err = h.numpy().argmax(axis=1) - y.numpy()
        error += np.sum(err != 0)
        losses += loss.numpy() * X.shape[0]
        
        if opt is not None:
            loss.backward()
            opt.step()
        
    return error/n, losses/n
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(data_dir + "/train-images-idx3-ubyte.gz", data_dir + "/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(data_dir +"/t10k-images-idx3-ubyte.gz",data_dir +"/t10k-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    input_dim = train_dataset[0][0].size
    num_class = train_dataset.num_classes
    model = MLPResNet(input_dim, hidden_dim, num_classes=num_class)
    opt = optimizer(model.parameters(), lr = lr, weight_decay=weight_decay)
    train_error = 0.0
    train_loss = 0.0
    
    for i in range(epochs):
        error_avg, loss_avg = epoch(train_dataloader, model=model, opt=opt)
        print(i, error_avg, loss_avg)
        if i == epochs - 1:
            train_error = error_avg
            train_loss = loss_avg
    test_error, test_loss = epoch(test_dataloader, model=model)
    print(test_error, test_loss)
    return (train_error, train_loss, test_error, test_loss)
    
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
