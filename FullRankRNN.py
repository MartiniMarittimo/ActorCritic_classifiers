import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
import random
import time
from contextlib import redirect_stdout


#seed = 3
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
##torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

class FullRankRNN(nn.Module): # FullRankRNN is a child class, nn.Module is the parent class

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha=0.2, rho=1, beta=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False,
                 wi_init=None, wo_init=None, wrec_init=None, si_init=None, so_init=None):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        """
        
        super(FullRankRNN, self).__init__()  #???
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.train_wi = train_wi #boolean (False)
        self.train_wrec = train_wrec #boolean (True)
        self.train_wo = train_wo #boolean (False)
        self.train_h0 = train_h0 #boolean (False)
        self.non_linearity = torch.nn.ReLU()
        self.actor = False

        self.wi = nn.Parameter(torch.empty((input_size, hidden_size), dtype=torch.float64)) #tensore 2D
        self.si = nn.Parameter(torch.empty((input_size), dtype=torch.float64)) #tensore 1D
        if train_wi:
            self.si.requires_grad = False 
        if not train_wi:
            self.wi.requires_grad = False
            #self.si.requires_grad = False #attention
            
        self.wrec = nn.Parameter(torch.empty((hidden_size, hidden_size), dtype=torch.float64))
        if not train_wrec:
            self.wrec.requires_grad = False
            
        self.wo = nn.Parameter(torch.empty((hidden_size, output_size), dtype=torch.float64))
        self.so = nn.Parameter(torch.empty((output_size), dtype=torch.float64))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
            #self.so.requires_grad = True #attention
            
        self.h0 = nn.Parameter(torch.empty((hidden_size), dtype=torch.float64)) 
        if not train_h0:
            self.h0.requires_grad = False

        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if wrec_init is None:
                self.wrec.normal_(std = rho/sqrt(hidden_size))
            else:
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std = 5/hidden_size)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            self.h0.zero_() #fills self tensor with zeros
            #self.wi.zero_()
        self.wi_full, self.wo_full = [None] * 2
        self.define_proxy_parameters()

        self.beta = beta

        
        
    def define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so
    
    
    
    def actor_critic(self, actor=False):
        
        if actor:
            self.actor = True
        #else:
            #self.wi.data = torch.ones(1,1)#, requires_grad=True)
            #self.wo.data = torch.ones(1,1)#, requires_grad=True)
        #    self.non_linearity = torch.nn.Tanh()

        
        
        
    def forward(self, input, return_dynamics=False, h0=None, time_step=0, trial_index=0, epoch=0):       
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        IMPORTANT --> the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape(batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, output tensor & trajectories tensor of shape(batch_size, #timesteps, #hidden_units)
        """
        
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if h0 is None:
            h = self.h0
        else:
            h = h0 
        r = self.non_linearity(h)
        self.define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, dtype=torch.float64, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, dtype=torch.float64, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, dtype=torch.float64, device=self.wrec.device)
            
        # forward loop
        for i in range(seq_len):
            h = h + self.alpha * (- h + input[:, i, :].matmul(self.wi_full) + r.matmul(self.wrec.t())) + \
            self.noise_std * noise[:, i, :]
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            if return_dynamics:
                trajectories[:, i, :] = h
        
        
        if self.actor:                        
            soft_output = nn.functional.softmax(output.clone()*self.beta, dim=-1) 
            if trial_index == 0 and time_step == 0:
                with open('data.txt', 'w') as f:
                    with redirect_stdout(f):
                        print("epoch ", epoch, ", trial ", trial_index+1)
                        print("z: ", output, "\nsoft_output: ", soft_output)
            else:
                with open('data.txt', 'a') as f:
                    with redirect_stdout(f):
                        print("epoch ", epoch, ", trial ", trial_index+1)
                        print("z: ", output, "\nsoft_output: ", soft_output)
            if return_dynamics:
                #print(soft_output)
                return soft_output, trajectories
            else:
                return soft_output
        
        else:
            if return_dynamics:
                return output, trajectories
            else:
                return output
