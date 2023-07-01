from models import Network
from dataset import *
import time
from loss_functions import learned_lagrangian,equation_of_motion
from functools import partial

class SaveBestModel(object):
  def __init__(self,best_valid_loss = float('inf')):
    self.best_valid_loss = best_valid_loss
  
  def __call__(self,current_valid_loss,
               epoch,model,optimizer,loss,metric,filename = 'best_model.pth'):
    
    if current_valid_loss <self.best_valid_loss:
      self.best_valid_loss = current_valid_loss
      torch.save({
          'epoch':epoch+1,
          'model_state_dict':model.state_dict(),
          'optimizer_state_dict':optimizer.state_dict(),
          'loss':loss,
          'metric':metric
      },filename)
      return True
    else:
      return False


model = Network()

lr = 1e-3
loss_func = loss_func = nn.MSELoss()
metric_func = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters,lr = lr)

epochs = 2000

SBM = SaveBestModel()

train_loss = 0.0
metric_loss = 0.0

hist_vali_loss = []
hist_train_loss = []
hist_vali_metric = []
hist_train_metric = []

filename = 'best_model.pth'

print("Starting training!")
for epoch in range(epochs):
  train_loss = 0.0
  train_metric = 0.0 
  vali_loss = 0.0
  vali_metric = 0.0 
  
  counter = 0
  total_time = 0
  model.train()
  for xb,yb in train_dataloader:
    start = time.time()

    L = learned_lagrangian(model)
    out = torch.vmap(partial(equation_of_motion,L))(xb)

    optimizer.zero_grad()
    loss = loss_func(out,yb)
    metric = metric_func(out,yb)
    loss.backward()
    optimizer.step()

    counter +=1
    train_loss += loss.item()
    train_metric += metric.item()
    
    end = time.time() - start
    total_time += end

  train_loss /= counter
  train_metric /= counter
  hist_train_loss.append(train_loss)
  hist_train_metric.append(train_metric)


  counter = 0

  model.eval()
  with torch.no_grad():
    for xb,yb in vali_dataloader:

      start = time.time()

      L = learned_lagrangian(model)
      out = torch.vmap(partial(equation_of_motion,L))(xb)

      loss = loss_func(out,yb)
      metric = metric_func(out,yb)
      counter +=1
      vali_loss += loss.item()
      vali_metric += metric.item()
      counter += 1

      end = time.time() - start
      total_time += end
      

  vali_loss /= counter
  vali_metric /= counter

  hist_vali_loss.append(vali_loss)
  hist_vali_metric.append(vali_metric)

  print("epoch:%d time taken:%lf\nTrain loss:%lf Validation metric:%lf\nValidation loss:%lf Validation metric:%lf"%(epoch,total_time,train_loss,train_metric,vali_loss,vali_metric))

  if SBM(vali_loss,epoch,model,optimizer,loss_func,metric_func,filename):
    print("best model saved")
  print("\n")


