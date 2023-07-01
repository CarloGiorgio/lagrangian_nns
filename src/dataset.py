from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch import nn
from scipy.integrate import odeint


def f_analytical(state,t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
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


def normalize_angle(state):
    #constrain the angles between [-pi,pi]
    
  return np.concatenate(((state[:2]+np.pi)%(2*np.pi) - np.pi,state[2:]))

def v_norm_angle(x):
    
  return np.concatenate(((x[:,:2] + np.pi)%(2*np.pi) -np.pi,x[:,2:]),axis = 1)

x0 = torch.Tensor([3*np.pi/7, 3*np.pi/4, 0, 0])
N = 4000
h = 0.01
t = torch.arange(0,N,1)*h
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


loss_func = nn.MSELoss()
metric_func = nn.L1Loss()
lr = 1e-4
#optimizer = torch.optim.Adam(model.parameters(),lr = lr)