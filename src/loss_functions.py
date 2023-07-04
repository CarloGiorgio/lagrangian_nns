import torch
from torch.func import grad,hessian,jacfwd
import numpy as np
from functools import partial
from scipy.integrate import odeint

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

def equation_of_motion_integrate(lagrangian, state, t=None):
  q, q_t = torch.split(torch.Tensor(state), 1,dim = -1)
  q_tt = torch.matmul(torch.linalg.pinv(hessian(lagrangian, 1)(q,q_t)),(grad(lagrangian, 0)(q,q_t)- torch.matmul(jacfwd(jacfwd(lagrangian, 1), 0)(q,q_t), q_t)))
  #return np.array([q_t.cpu().detach().numpy(), q_tt.cpu().detach().numpy()])
  return np.concatenate([q_t.cpu().detach().numpy(), q_tt.cpu().detach().numpy()])



def prediction(model,x1,N=300,h=0.01):
  #integrate the equation of motion using the 
  #initial conditions and the integrator
  t1 = np.arange(0,N)*h
  f = odeint(partial(equation_of_motion_integrate,learned_lagrangian(model)),x1,t1)
  return f




def equation_of_motion_hamilton(hamiltonian,state,t = None):
    # equation of motion for relativisti particle
    # the thing it is changing is the conversion from 
    # derivative to canonical momenta
    q, q_t = torch.split(state, 2,dim = -1)
    
    #Move to canonical coordinates:
    p = q_t/(1.0-q_t**2)**(3/2.0)
    q = q
    
    conditionals = conditionals / 10.0 #Normalize
    p_t = -grad(hamiltonian, 0)(q, p, conditionals)
    #Move back to generalized coordinates:
    q_tt = p_t*(1-q_t**2)**(5.0/2)/(1.0+2*q_t**2)
    
    #Avoid nans by computing q_t afterwards:
    q_t = grad(hamiltonian, 1)(q, p, conditionals)
    return torch.cat([q_t,q_tt])


def raw_lagrangian_eom(lagrangian, state, t=None):
    #given the model, retunr the lagrangian density
    vlagrangian = torch.vmap(lagrangian, (0, 0), 0)

    #Evaluate the lagrangian from the density
    def lagrangian_fnc(q, q_t):
        
        q_min = torch.roll(q, shift=+1)
        q_plus = torch.roll(q, shift=-1)

        q_t_min = torch.roll(q_t, shift=+1)
        q_t_plus = torch.roll(q_t, shift=-1)
        
        all_q = torch.cat([q_min,q,q_plus],axis = -1)
        all_q_t = torch.cat([q_t_min, q_t, q_t_plus], axis=-1)
        return torch.sum(vlagrangian(all_q, all_q_t))
    
    def conv_fnc(q, q_t):
        #perform the equaion of motion
        q_tt = (torch.linalg.pinv(hessian(lagrangian_fnc, 1)(q,q_t))
          @ (grad(lagrangian_fnc, 0)(q,q_t)
            - jacfwd(jacfwd(lagrangian_fnc, 1), 0)(q,q_t) @ q_t))
        
        return torch.cat([q_t, q_tt])

    def fnc(state):
        #recast all in the right dimension
        q, q_t = torch.split(state, dim = 1)

        out = conv_fnc(q, q_t)
        out = torch.cat([out[0], out[1]])
        return out
    
    return fnc(state)
  
  

    