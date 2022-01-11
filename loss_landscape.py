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
y = 5 * x + 0 + torch.normal(0.0, 20.0, (n, 1))

# linear regression
model = nn.Linear(1, 1, bias=False)
loss_fn = nn.MSELoss()
code.interact(local=dict(globals(), **locals()))
weights, losses = [], []
for w in torch.arange(0, 10, 0.2):
    model.weight = nn.Parameter(torch.Tensor([[w]]))
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    weights.append(w)
    losses.append(loss.item())
    # print(f"weight {w}, loss {loss.item()}")
    # loss.backward()

# analytic solution
# wa =  (\X^T\X)^{-1} \X^T\y
# HAHA YOU IDIOT ITS FIVE

# plot data
sns.set()
sns.scatterplot(x=weights, y=losses)
imgcat(plt.gcf())
