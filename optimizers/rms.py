from typing import Any
import torch

  
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



# generated by chatgpt
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
