import unittest

import numpy as np
import torch
import torch.optim as optim
from nnsvs import mdn
from nnsvs.util import init_seed
from torch import nn


class MDN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, num_gaussians=30):
        super(MDN, self).__init__()
        self.first_linear = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.tanh = nn.Tanh()
        self.mdn = mdn.MDNLayer(hidden_dim, out_dim, num_gaussians=num_gaussians)

    def forward(self, x, lengths=None):
        out = self.tanh(self.first_linear(x))
        for hl in self.hidden_layers:
            out = self.tanh(hl(out))
        return self.mdn(out)


class TestMDN(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        init_seed(42)

        # generate data
        # Inverse model written in PRML Book p. 273
        # https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
        n = 2500
        self.d_in = 1
        self.d_out = 1

        x_train = np.random.uniform(0, 1, (n, self.d_in)).astype(np.float32)
        noise = np.random.uniform(-0.1, 0.1, (n, self.d_in)).astype(np.float32)
        y_train = x_train + 0.3 * np.sin(2 * np.pi * x_train) + noise

        self.x_train_inv = y_train
        self.y_train_inv = x_train

        self.x_test = np.array([0.0, 0.2, 0.5, 0.8, 1.0]).astype(np.float32)

        # [lower_limit, upper_limit] corresponding to x_test
        self.y_test_range = np.array(
            [[-0.5, 1], [-0.5, 2.0], [0.2, 0.9], [0.8, 1.0], [0.85, 1.05]]
        ).astype(np.float32)

        hidden_dim = 50
        num_gaussians = 30
        num_layers = 0

        self.batch_size = n

        use_cuda = torch.cuda.is_available()

        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = MDN(
            self.d_in,
            hidden_dim,
            self.d_out,
            num_layers=num_layers,
            num_gaussians=num_gaussians,
        ).to(self.device)

        learning_rate = 0.008
        self.opt = optim.Adam(self.model.parameters(), lr=learning_rate)

    def test_mdn_loss(self):
        # wrap up the inverse data as Variables
        x = torch.from_numpy(
            self.x_train_inv.reshape(self.batch_size, -1, self.d_in)
        ).to(
            self.device
        )  # (B, max(T), D_in)
        y = torch.from_numpy(
            self.y_train_inv.reshape(self.batch_size, -1, self.d_out)
        ).to(
            self.device
        )  # (B, max(T), D_out)
        for e in range(1000):
            self.model.zero_grad()
            pi, sigma, mu = self.model(x)
            loss = mdn.mdn_loss(pi, sigma, mu, y).mean()
            if e % 100 == 0:
                print(f"loss: {loss.data.item()}")
            loss.backward()
            self.opt.step()

    def test_mdn_get_most_probable_sigma_and_mu(self):
        self.test_mdn_loss()

        pi, sigma, mu = self.model(
            torch.from_numpy(self.x_test.reshape(1, -1, self.d_in)).to(self.device)
        )
        _, max_mu = mdn.mdn_get_most_probable_sigma_and_mu(pi, sigma, mu)
        max_mu = max_mu.squeeze(0).cpu().detach().numpy()
        print(max_mu.shape)

        for i, sample in enumerate(max_mu):
            lower_limit = self.y_test_range[i][0]
            upper_limit = self.y_test_range[i][1]
            assert lower_limit < sample and upper_limit > sample
            print(
                f"sample: {sample}, lower_limit: {lower_limit}, upper_limit: {upper_limit}"
            )

    def test_mdn_get_sample(self):
        self.test_mdn_loss()

        pi, sigma, mu = self.model(
            torch.from_numpy(self.x_test.reshape(1, -1, self.d_in)).to(self.device)
        )
        samples = mdn.mdn_get_sample(pi, sigma, mu).squeeze(0).cpu().detach().numpy()

        for i, sample in enumerate(samples):
            lower_limit = self.y_test_range[i][0]
            upper_limit = self.y_test_range[i][1]
            assert lower_limit < sample and upper_limit > sample
            print(
                f"sample: {sample}, lower_limit: {lower_limit}, upper_limit: {upper_limit}"
            )


if __name__ == "__main__":
    unittest.main()
