# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
from solo.losses import mocov3_loss_func


def test_mocov3_loss():
    b, f = 32, 128
    query = torch.randn(b, f).requires_grad_()
    key = torch.randn(b, f).requires_grad_()

    loss = mocov3_loss_func(query, key, temperature=0.1)
    initial_loss = loss.item()
    assert loss != 0

    for _ in range(20):
        loss = mocov3_loss_func(query, key, temperature=0.1)
        loss.backward()
        query.data.add_(-0.5 * query.grad)
        key.data.add_(-0.5 * key.grad)

        query.grad = key.grad = None

    assert loss < initial_loss
