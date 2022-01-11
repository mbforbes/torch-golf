import code
from imgcat import imgcat
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

# generate y = mx + b + 𝒩(0, 𝜎²)
x = torch.arange(0, 100, 1.0).unsqueeze(1)  # (n,d) for pytorch
n = len(x)
y = 5 * x + 12 + torch.normal(0.0, 20.0, (n, 1))

# linear regression w/ gradient descent (whole dataset at once)
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
for epoch in range(20):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(f"epoch {epoch}, loss {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# plot data
sns.set()
sns.scatterplot(x=x.squeeze(1), y=y.squeeze(1))
imgcat(plt.gcf())
