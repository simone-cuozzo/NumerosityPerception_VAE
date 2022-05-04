# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def kaiming_uniform(m):
  if type(m)== nn.Conv2d or type(m)== nn.Linear or type(m)== nn.ConvTranspose2d:
    torch.nn.init.kaiming_uniform_(m.weight)
    if m.bias is not None:
      torch.nn.init.zeros_(m.bias)

def xavier_uniform(m):
  if type(m)== nn.Conv2d or type(m)== nn.Linear or type(m)== nn.ConvTranspose2d:
    torch.nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
      torch.nn.init.zeros_(m.bias)
      
def kaiming_normal(m):
  if type(m)== nn.Conv2d or type(m)== nn.Linear or type(m)== nn.ConvTranspose2d:
    torch.nn.init.kaiming_normal_(m.weight)
    if m.bias is not None:
      torch.nn.init.zeros_(m.bias)

def xavier_normal(m):
  if type(m)== nn.Conv2d or type(m)== nn.Linear or type(m)== nn.ConvTranspose2d:
    torch.nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
      torch.nn.init.zeros_(m.bias)