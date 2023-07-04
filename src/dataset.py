from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch import nn
from scipy.integrate import odeint


def harmonic_oscillator(state,t = 0,k = 1,m = 1):
  q,q_t = state
  q_tt = -k/m*q
  return np.array([q_t,q_tt])

def coupled_HO(state,t = None,alpha = 0.5):
  q1,q2,q_t1,q_t2 = state

  return np.array([q_t1,q_t2,-(q1+alpha*q2),-(q2+alpha*q1)])


def single_pendulum(state,t = None,l = 1,g = 9.81):
  q,q_t = state
  q_tt = -g/l*np.sin(q)
  return np.array([q_t,q_tt])


def double_pendulum(state,t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
  # evaluate analyticaly the velocity from the equation of motion
  # this is done by means of the equation of motion from the lagrangian
    
  t1, t2, w1, w2 = state
  a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
  a2 = (l1 / l2) * np.cos(t1 - t2)
  f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * np.sin(t1 - t2) - \
      (g / l1) * np.sin(t1)
  f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
  g1 = (f1 - a1 * f2) / (1 - a1 * a2)
  g2 = (f2 - a2 * f1) / (1 - a1 * a2)
  return np.array([w1,w2,g1,g2])

def relativistic_particle(state,t = 0,g = 9.81):
    #code for generating a relativistic trajectory, evaluating the acceleration of the 
    q,q_t = state
    q_tt = (1-q_t**2)**(5./2)/(1+2*q_t**2)*g
    return np.array([q_t,q_tt])


def generate_wave_equation(dx = 0.1):
    # generate a function for the creation of the 
    # function to integrate 
    def wave_equation(state):
        # equation of motion for the wave equation
        # the function returns the first and second derivative 
        # of the field
        q, q_t = torch.split(state, 2)

        q_plus = torch.roll(q, shift=-1)
        q_min = torch.roll(q, shift=+1)

        q_x = (q_plus - q_min) / (2 * dx)
        q_xx = (q_plus - 2 * q + q_min) / (2 * dx)
        
        #Wave equation with constraint:
        q_tt = q_xx
        return np.array([q_t,q_tt])
    return wave_equation

def normalize_angle(state):
    #constrain the angles between [-pi,pi]
    return np.concatenate(((state[:2]+np.pi)%(2*np.pi) - np.pi,state[2:]))

def v_norm_angle(x):
    # vectorized angle normalization
    return np.concatenate(((x[:,:2] + np.pi)%(2*np.pi) -np.pi,x[:,2:]),axis = 1)




f_analytical = double_pendulum
N = 4000
h = 0.01

#x0 = torch.Tensor([3*np.pi/7, 3*np.pi/4, 0, 0])
x0 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0])

#t = torch.arange(0,N,1)*h
t = np.arange(0,N,1)*h

#print(x0.size(),t.size())
x = odeint(f_analytical,x0,t)
x = v_norm_angle(x)

y = np.array(list(map(f_analytical,x)))

class SampleDataset(Dataset):
  def __init__(self,x,y):
    self.x = x
    self.y = y
  
  def __len__(self):
    return len(self.x)
  
  def __getitem__(self, idx):
    return self.x[idx],self.y[idx]
