import torch.optim

from optim.mixin import OptimMixin
from geoopt import ManifoldParameter, ManifoldTensor


class RiemannianAdam(OptimMixin, torch.optim.Adam):

    def __init__(self, *args, stabilize, **kwargs):
        super().__init__(*args, stabilize=stabilize, **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():

            for group in self.param_groups:

                if "step" not in group:
                    group["step"] = 0

                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                group["step"] += 1

                for point in group["params"]:
                    grad = point.grad

                    if grad is None:
                        continue

                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianAdam does not support sparse gradients, use SparseRiemannianAdam instead")
                    state = self.state[point]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(point)
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # actual step
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        grad.add_(point, alpha=weight_decay)

                    grad = manifold.egrad2rgrad(point, grad)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(manifold.component_inner(point, grad), alpha=1 - betas[1])

                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = max_exp_avg_sq
                    else:
                        denom = exp_avg_sq

                    bias_correction1 = 1 - betas[0] ** group["step"]
                    bias_correction2 = 1 - betas[1] ** group["step"]
                    step_size = learning_rate
                    direction = (exp_avg / bias_correction1).div_((denom / bias_correction2).sqrt().add_(eps))
                    if not isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        direction = direction + point * weight_decay
                    new_point, exp_avg_new = manifold.retr_transp(point, -step_size * direction, exp_avg)
                    point.copy_(new_point)
                    exp_avg.copy_(exp_avg_new)

                if group["stabilize"] is not None and group["step"] % group["stabilize"] == 0:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            state = self.state[p]
            if not state:
                continue
            manifold = p.manifold
            exp_avg = state["exp_avg"]
            p.copy_(manifold.projx(p))
            exp_avg.copy_(manifold.proju(p, exp_avg))
