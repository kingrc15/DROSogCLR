# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleWeight(nn.Module):
    def __init__(self, latent_size, method, N):
        super().__init__()
        self.method = method
        if method == "Linear":
            self.p = nn.Linear(latent_size, 1)
        elif method == "Parameter":
            self.p = nn.Parameter(torch.ones((N, N)) / N)

    def forward(self, x=None, index=None):
        if self.method == "Linear":
            return self.p(x)
        elif self.method == "Parameter":
            self.p = torch.nn.functional.softmax(self.p, dim=0)
            return self.p[index, index]


class SimCLR(nn.Module):
    """
    Build a SimCLR model with a base encoder, and two MLPs

    """

    def __init__(
        self,
        base_encoder,
        dim=256,
        mlp_dim=2048,
        T=1.0,
        cifar_head=False,
        loss_type="dcl",
        N=50000,
        num_proj_layers=2,
        device=None,
        kl_weight=0,
        p_method="Linear",
        use_cov=False,
    ):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        cifar_head: special input layers for training cifar datasets (default: false)
        loss_type: dynamatic contrastive loss (dcl) or contrastive loss (cl) (default: dcl)
        N: number of samples in the dataset used for computing moving average (default: 50000)
        num_proj_layers: number of non-linear projection head (default: 2)
        """
        super(SimCLR, self).__init__()
        self.T = T
        self.N = N
        self.loss_type = loss_type
        self.kl_weight = kl_weight
        self.use_cov = use_cov

        self.p = SampleWeight(128, p_method, N)

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)

        # build non-linear projection heads
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        # update input heads if image_size < 32 (e.g., cifar)
        print("cifar head:", cifar_head)
        self.base_encoder.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.base_encoder.maxpool = nn.Identity()

        # sogclr
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if self.loss_type == "dcl":
            self.u = torch.zeros(N).reshape(-1, 1)  # .to(self.device)

        self.LARGE_NUM = 1e9

        self.kl_crit = nn.KLDivLoss(reduction="batchmean")

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def dynamic_contrastive_loss(
        self, hidden1, hidden2, index=None, gamma=0.99, distributed=True
    ):
        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(
            hidden2, p=2, dim=1
        )
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(
            torch.arange(batch_size, dtype=torch.long), batch_size * 2
        ).to(self.device)
        masks = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(
            self.device
        )

        logits_aa = torch.matmul(hidden1, hidden1_large.T)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T)
        logits_ba = torch.matmul(hidden2, hidden1_large.T)

        #  SogCLR
        neg_mask = 1 - labels
        logits_ab_aa = torch.cat([logits_ab, logits_aa], 1)
        logits_ba_bb = torch.cat([logits_ba, logits_bb], 1)

        neg_logits1 = torch.exp(logits_ab_aa / self.T) * neg_mask  # (B, 2B)
        neg_logits2 = torch.exp(logits_ba_bb / self.T) * neg_mask

        u1 = (1 - gamma) * self.u[index].cuda() + gamma * torch.sum(
            neg_logits1, dim=1, keepdim=True
        ) / (2 * (batch_size - 1))
        u2 = (1 - gamma) * self.u[index].cuda() + gamma * torch.sum(
            neg_logits2, dim=1, keepdim=True
        ) / (2 * (batch_size - 1))

        self.u[index] = u1.detach().cpu() + u2.detach().cpu()

        p_neg_weights1 = (neg_logits1 / u1).detach()
        p_neg_weights2 = (neg_logits2 / u2).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights, dro_weights):
            expsum_neg_logits = torch.sum(
                dro_weights * weights * logits, dim=1, keepdim=True
            ) / (2 * (batch_size - 1))
            normalized_logits = logits - expsum_neg_logits

            return -torch.sum((labels * normalized_logits), dim=1)

        p_1 = torch.nn.functional.softmax(
            self.p(x=hidden1.clone().detach(), index=index)
        )
        p_2 = torch.nn.functional.softmax(
            self.p(x=hidden2.clone().detach(), index=index)
        )

        if not self.use_cov:
            kl_1 = self.kl_crit(p_1, torch.ones_like(p_1) / p_1.size(0))
            kl_2 = self.kl_crit(p_2, torch.ones_like(p_2) / p_2.size(0))

            kl_loss = (kl_1 + kl_2) / 2

            dro_weights = 1

        else:
            p = torch.nn.functional.softmax(
                (p_1 @ p_2.T / batch_size).reshape(-1, 1), dim=0
            )

            kl_loss = self.kl_crit(p, torch.ones_like(p) / p.size(0))

            p = p.reshape(batch_size, batch_size)

            dro_weights = torch.cat([p, p], 1) * batch_size

            p_1 = 1 / batch_size
            p_2 = 1 / batch_size

        loss_a = (
            softmax_cross_entropy_with_logits(
                labels, logits_ab_aa, p_neg_weights1, dro_weights
            )
            * p_1
            * batch_size
        )
        loss_b = (
            softmax_cross_entropy_with_logits(
                labels, logits_ba_bb, p_neg_weights2, dro_weights
            )
            * p_2
            * batch_size
        )
        loss = (loss_a + loss_b).mean()

        return loss + kl_loss * self.kl_weight

    def contrastive_loss(
        self, hidden1, hidden2, index=None, gamma=None, distributed=True
    ):
        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(
            hidden2, p=2, dim=1
        )
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(
            torch.arange(batch_size, dtype=torch.long), batch_size * 2
        ).to(self.device)
        masks = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(
            self.device
        )

        logits_aa = torch.matmul(hidden1, hidden1_large.T) / self.T
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T) / self.T
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T) / self.T
        logits_ba = torch.matmul(hidden2, hidden1_large.T) / self.T

        logits_ab_aa = torch.cat([logits_ab, logits_aa], 1)
        logits_ba_bb = torch.cat([logits_ba, logits_bb], 1)

        def softmax_cross_entropy_with_logits(labels, logits):
            # logits = logits - torch.max(logits)
            expsum_neg_logits = torch.sum(torch.exp(logits), dim=1, keepdim=True)
            normalized_logits = logits - torch.log(expsum_neg_logits)
            return -torch.sum(labels * normalized_logits, dim=1)

        p_ab_aa = self.p(logits_ab_aa)
        p_ba_bb = self.p(logits_ba_bb)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa) * p_ab_aa
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb) * p_ba_bb
        loss = (loss_a + loss_b).mean()
        return loss

    def forward(self, x1, x2, index, gamma):
        # compute features
        h1 = self.base_encoder(x1)
        h2 = self.base_encoder(x2)

        if self.loss_type == "dcl":
            loss = self.dynamic_contrastive_loss(h1, h2, index, gamma)
        elif self.loss_type == "cl":
            loss = self.contrastive_loss(h1, h2)
        return loss


class SimCLR_ResNet(SimCLR):
    def _build_projector_and_predictor_mlps(
        self, dim, mlp_dim, num_proj_layers=2, kl_weight=0
    ):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc  # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(
            num_proj_layers, hidden_dim, mlp_dim, dim
        )
