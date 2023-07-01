import torch
from torch.func import grad,hessian,jacfwd,jacrev

def equation_of_motion(lagrangian, state, t=None):
  q, q_t = torch.split(state, 2,dim = -1)
  q_tt = (torch.linalg.pinv(hessian(lagrangian, 1)(q,q_t))
          @ (grad(lagrangian, 0)(q,q_t)
            - jacfwd(jacfwd(lagrangian, 1), 0)(q,q_t) @ q_t))
  return torch.cat([q_t, q_tt])


def learned_lagrangian(model):
    #function that takes the model and return the lagrangian
    def lagrangian(q,q_t):
      state  = torch.cat([q,q_t],dim = -1)
      return model(state)[0]
    return lagrangian


    