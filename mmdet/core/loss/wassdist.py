import torch
import torch.nn as nn
import numpy as np

# Adapted from https://github.com/dfdazac/wassdistance
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, x_points, y_points, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.x_points = x_points
        self.y_points = y_points
        self.reduction = reduction

        # both marginals are fixed with equal weights
        self.mu = torch.empty(1, self.x_points, dtype=torch.float,
                              requires_grad=False).fill_(1.0 / self.x_points).squeeze()
        self.nu = torch.empty(1, self.y_points, dtype=torch.float,
                              requires_grad=False).fill_(1.0 / self.y_points).squeeze()
        self.u = torch.zeros_like(self.mu)
        self.v = torch.zeros_like(self.nu)

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        # both marginals are fixed with equal weights
        mu, nu = self.mu.to(x.device), self.nu.to(x.device)
        u, v = self.u.to(x.device), self.v.to(x.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        # C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        C = torch.norm(torch.abs(x_col - y_lin), 2, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

def wasserstein_loss(point_set_1, point_set_2, sinkhorn):
    '''Computation of optimal transport distance via sinkhorn algorithm.
    - Input:
        - point_set_1:	torch.Tensor	[..., num_points_1, point_dim] e.g. [bs, h, w, 1000, 2]; [bs, 1000, 2]; [1000, 2]
        - point_set_2:	torch.Tensor	[..., num_points_2, point_dim]
        		(the dimensions of point_set_2 except the last two should be the same as point_set_1)
    - Output:
        - distance:	torch.Tensor	[...] e.g. [bs, h, w]; [bs]; []
    '''

    assert point_set_1.dim() == point_set_2.dim()
    assert point_set_1.shape[-1] == point_set_2.shape[-1]
    if point_set_1.dim() <= 3:
        dist, _, _ = sinkhorn(point_set_1, point_set_2)
    else:
        point_dim = point_set_1.shape[-1]
        num_points_1, num_points_2 = point_set_1.shape[-2], point_set_2.shape[-2]
        point_set_1t = point_set_1.reshape((-1, num_points_1, point_dim))
        point_set_2t = point_set_2.reshape((-1, num_points_2, point_dim))
        dist_t, _, _ = sinkhorn(point_set_1t, point_set_2t)
        dist_dim = point_set_1.shape[:-2]
        dist = dist_t.reshape(dist_dim)
    return dist

if __name__ == '__main__':
    n_points_1, n_points_2 = 50, 100
    a = np.array([[np.random.rand(), 0] for i in range(n_points_1)])
    b = np.array([[np.random.rand(), 10] for i in range(n_points_2)])
    x = torch.tensor([[a], [a]], dtype=torch.float).squeeze(1)
    y = torch.tensor([[b], [b]], dtype=torch.float).squeeze(1)

    # a = np.array([[np.random.rand(), np.random.rand(), 0] for i in range(n_points_1)])
    # b = np.array([[np.random.rand(), np.random.rand(), 10] for i in range(n_points_2)])
    # x = torch.tensor([[a,a,a], [b,b,a]], dtype=torch.float).squeeze(1)
    # y = torch.tensor([[a,b,a], [b,b,b]], dtype=torch.float).squeeze(1)
    print(x.shape)
    print(y.shape)
    
    # sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    # dist, P, C = sinkhorn(x, y)
    import time
    start = time.time()
    for i in range(5):
        dist = wasserstein_loss(x.cuda(3), y.cuda(3))
    end = time.time()
    print((end - start)/ (5.0 * 2.0))
    print(dist.cpu().numpy())
    # import pdb
    # pdb.set_trace()

