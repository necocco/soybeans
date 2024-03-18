from PIL import Image
import os
from torchvision import transforms
from torchvision import datasets
from glob import glob
from natsort import natsorted
from torchvision.models import resnet34
from torchmetrics.functional import accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger



class Net(pl.LightningModule):
  def __init__(self):
   super().__init__()
   self.feature=resnet34(pretrained=True)
   self.fc=nn.Linear(1000,4)

  def forward(self,x):
    h = self.feature(x)
    h = self.fc(h)
    return h

  def training_step(self,batch,batch_idx):
    x,t=batch
    y=self(x)
    loss=F.cross_entropy(y,t)
    self.log('train_loss',loss,on_step=False,on_epoch=True)
    self.log('train_acc',accuracy(y.softmax(dim=-1),t,task='multiclass',num_classes=4,top_k=1),on_step=False,on_epoch=True)
    return loss

  def validation_step(self,batch,batch_idx):
    x,t=batch
    y=self(x)
    loss=F.cross_entropy(y,t)
    self.log('val_loss',loss,on_step=False,on_epoch=True)
    self.log('val_acc',accuracy(y.softmax(dim=-1),t,task='multiclass',num_classes=4,top_k=1),on_step=False,on_epoch=True)
    return loss

