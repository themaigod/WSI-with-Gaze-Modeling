import torch
from torch.autograd import Function
from torch.autograd import Variable


class MIL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.tensor(2, dtype=torch.float, requires_grad=True)

    def forward(self, x):
        x1 = self.a * x
        y = torch.ones((1, 6))
        x2 = y.copy_(x1)
        x = self.a * x2
        return x, x1, x2


class BinarizedF(Function):

    def forward(self, ctx, *args, **kwargs):
        self.save_for_backward = ctx
        a = torch.ones_like(ctx)
        b = -torch.ones_like(ctx)
        output = torch.where(ctx >= 0, a, b)
        return output

    def backward(self, output_grad):
        input = self.save_for_backward
        input_abs = torch.abs(input)
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        input_grad = torch.where(input_abs <= 1, ones, zeros)
        return input_grad


if __name__ == '__main__':
    b = torch.tensor([0.2], requires_grad=True)
    a = torch.tensor(2, dtype=torch.float, requires_grad=True)
    e = torch.tensor(2, dtype=torch.float, requires_grad=True)
    # model = MIL()
    # c1, x1, x2 = model(b)
    # c = torch.mean(c1)
    # c.backward()
    v = b.clone()
    x1 = a * b
    y = torch.ones((1, 6))
    # x2 = y.copy_(x1)
    x2 = x1.expand((1, 2, 3))
    x = e * x2
    c = torch.mean(x)
    c.backward()

qe = self.result[0]
