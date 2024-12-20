from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Iterate over each parameter group
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            beta1, beta2 = betas
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            correct_bias = group["correct_bias"]

            # Iterate over each parameter in the group
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # Access or initialize the state for the parameter
                state = self.state[p]

                if len(state) == 0:
                    # Initialize state
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Increment the step count
                state['step'] += 1
                step = state['step']

                # Get the exponential moving averages
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if correct_bias:
                    # Compute bias-corrected first moment
                    bias_correction1 = 1 - beta1 ** step

                    # Compute bias-corrected second moment
                    bias_correction2 = 1 - beta2 ** step

                    # Compute step size with bias correction
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = lr

                # Compute denominator (sqrt of second moment + epsilon)
                denom = exp_avg_sq.sqrt().add_(eps)

                # Update the parameters with Adam step
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Apply decoupled weight decay (AdamW)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        return loss
    # def step(self, closure: Callable = None):
    #     loss = None
    #     if closure is not None:
    #         loss = closure()
    #
    #     for group in self.param_groups:
    #         for p in group["params"]:
    #             if p.grad is None:
    #                 continue
    #             grad = p.grad.data
    #             if grad.is_sparse:
    #                 raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
    #
    #             # State should be stored in this dictionary
    #             state = self.state[p]
    #
    #             # Access hyperparameters from the `group` dictionary
    #             alpha = group["lr"]
    #
    #             # Complete the implementation of AdamW here, reading and saving
    #             # your state in the `state` dictionary above.
    #             # The hyperparameters can be read from the `group` dictionary
    #             # (they are lr, betas, eps, weight_decay, as saved in the constructor).
    #             #
    #             # 1- Update first and second moments of the gradients
    #             # 2- Apply bias correction
    #             #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
    #             #     also given in the pseudo-code in the project description).
    #             # 3- Update parameters (p.data).
    #             # 4- After that main gradient-based update, update again using weight decay
    #             #    (incorporating the learning rate again).
    #
    #             ### TODO
    #             # raise NotImplementedError\
        # return loss
