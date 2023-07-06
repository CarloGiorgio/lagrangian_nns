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

lr = 0.05
loss_func = nn.MSELoss()
metric_func = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters,lr = lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

epochs = 2000

def train(model,train_dataloader,vali_dataloader,
          epochs,loss_func,metric_func,optimizer,scheduler
          ,print_results = 20,filename = 'best_model.pth'):
  
  epoch_best_model = 0
  
  SBM = SaveBestModel()

  train_loss = 0.0

  hist_vali_loss = []
  hist_train_loss = []
  hist_vali_metric = []
  hist_train_metric = []

  print("Starting training!")
  try:
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
        
      scheduler.step()
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

      if SBM(vali_loss,epoch,model,optimizer,loss_func,metric_func,filename):
        epoch_best_model = epoch + 1

      if epoch%print_results == 0:
        print("epoch:%d time taken:%lf\nTrain loss:%lf Validation metric:%lf\nValidation loss:%lf Validation metric:%lf"%(epoch+1,total_time,train_loss,train_metric,vali_loss,vali_metric))
        print("Best model found at epoch: %d"%epoch_best_model)
        print("\n")
  except KeyboardInterrupt:
    print("Training interrupted at epoch %d\nBest model found epoch %d"%((epoch+1),epoch_best_model))
