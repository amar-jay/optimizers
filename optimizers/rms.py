from typing import Any
import torch

# https://www.geeksforgeeks.org/custom-optimizers-in-pytorch/
class CustomOptimizer(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr) 
        super().__init__(params, defaults) 
  
    def step(self, closure=None)->Any: 
        if closure is not None: 
            with torch.enable_grad(): 
                closure()

        for group in self.param_groups: 
            for p in group['params']: 
                if p.grad is None: 
                    continue
                #p.data -= group['lr']*p.grad.data
                p.data.add_(-p['lr'], p.grad)
  
# https://www.geeksforgeeks.org/gradient-descent-with-rmsprop-from-scratch/
class RMSProp(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3, alpha=0.9):
        defaults = dict(lr=lr, alpha=alpha)
        self.lr = lr
        self.alpha = alpha
        super().__init__(params, defaults) 
  
    def step(self, closure=None)->Any: 
        if closure is not None: 
            with torch.enable_grad(): 
                closure()

        for group in self.param_groups: 
            for p in group['params']: 
                if p.grad is None: 
                    continue

                param_state = self.state[p]
                if "acc_square_avg" not in param_state:
                    param_state["acc_square_avg"] = torch.zeros_like(p.grad.data)
                    print("Hey, new parameter")

                param_state["acc_square_avg"] = param_state["acc_square_avg"]*self.alpha + (1-self.alpha)*(p.grad.data ** 2)
                p.data.add_(p.grad / (torch.sqrt(param_state["acc_square_avg"]) + 1e-8),alpha=-group['lr'])


class GPTRMSProp(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None)->Any:
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                if "acc_square_avg" not in param_state:
                    param_state["acc_square_avg"] = torch.zeros_like(p.data)

                grad = p.grad.data
                acc_square_avg = param_state["acc_square_avg"]
                alpha = group["alpha"]
                eps = group["eps"]
                lr = group["lr"]

                # Update the accumulated squared gradients
                acc_square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                # Update the parameters
                p.data.addcdiv_(grad, torch.sqrt(acc_square_avg) + eps, value=-lr)
  

# Im so exhausted, so I told Llama to think about it and he came up with this
class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None)->Any:
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr'] * bias_correction1 / bias_correction2

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

