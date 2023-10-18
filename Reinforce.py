import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import copy
import time

import FullRankRNN as rnn
import MONKEYtask as mky



class REINFORCE:
    
    def __init__(self, deltaT=20., noise_std=0, alpha=0.2, beta=1,
                 name_load_actor=None, name_load_critic=None, seed=None,
                 train_wi_a=True, train_wrec_a=True, train_wo_a=True,
                 train_wi_c=True, train_wrec_c=True, train_wo_c=True,
                 v1s=None, v2s=None, p1s=None, p2s=None):        

        self.cuda = False
        self.device = torch.device('cpu')

        if seed is not None:
            torch.manual_seed(seed)
        self.actor_network = rnn.FullRankRNN(input_size=5, hidden_size=128, output_size=3,
                                             noise_std=noise_std, alpha=alpha, rho=0.8, beta=beta,
                                             train_wi=train_wi_a, train_wo=train_wo_a, train_wrec=train_wrec_a)
        
        if name_load_actor is not None:
            self.actor_network.load_state_dict(torch.load(name_load_actor, map_location=self.device))
        
        self.actor_network.actor_critic(actor=True)
        
        if seed is not None:
            torch.manual_seed(seed)
        self.critic_network = rnn.FullRankRNN(input_size=131, hidden_size=128, output_size=1,
                                              noise_std=noise_std, alpha=alpha, rho=0.8, beta=1,
                                              train_wi=train_wi_c, train_wo=train_wo_c, train_wrec=train_wrec_c)
        if name_load_critic is not None:
            self.critic_network.load_state_dict(torch.load(name_load_critic, map_location=self.device))

        self.critic_network.actor_critic(actor=False)
        
        self.task = mky.RandomDotMotion(dt=deltaT, v1s=v1s, v2s=v2s, p1s=p1s, p2s=p2s)
        
        self.hidden_size = 128
        
        self.coh_info = {"n0":0, "r0":0, "pos0": 0, "neg0": 0, "n6":0, "r6":0, "pos6": 0, "neg6": 0,\
                         "n12":0, "r12":0, "pos12": 0, "neg12": 0, "n25":0, "r25":0, "pos25": 0, "neg25": 0,\
                         "n51":0, "r51":0, "pos51": 0, "neg51": 0}
        self.trial = 0
        self.epochs = 0
        self.epoch = 0
        self.actions_t = torch.zeros(3, dtype=torch.float64, device=self.device)
        #self.actions_tt = []
        
        self.df_finale = self.task.dframe.copy()
        
# ===============================================================================================================
        
    def obj_function(self, log_action_probs, actions, cum_rho, values, entropies, n_trs, hyper_l):
        
        new_mask = torch.zeros(log_action_probs.size(), device=self.device)
        
        for i in range(len(actions)):
            action = actions[i]
            new_mask[i][action] = 1
            
        #assert torch.all(torch.eq(new_mask, full_mask))
        
        obj = (new_mask * log_action_probs)
        obj = obj.sum(dim=-1)
        obj = obj * (cum_rho - values)        
        obj = obj.sum(dim=-1) / (-n_trs) + hyper_l * entropies.sum() / (-n_trs)
        
        return obj
    
# =============================================================================================================== 
    
    def loss_mse(self, output, target, trial_begins, n_trs):
        
        loss = 0
        
        for i in range(n_trs):
            
            start = int(trial_begins[i])
            stop = int(trial_begins[i+1])
            T = stop - start
            
            trial_output = output[start:stop]
            trial_target = target[start:stop]
            
            L = (trial_output - trial_target).pow(2).sum(dim=-1) / T
            loss = loss + L
            
        loss = loss / n_trs

        return loss

# =============================================================================================================== 
    
    def experience(self, n_trs, training=False):
        
        #if not training:
        #    self.task.dframe.copy()
        
        device = self.device
        
        observations = []
        rewards = []
        stimuli = []
        
        log_action_probs = torch.unsqueeze(torch.zeros(3, device=device), 0)
        actions = []
        final_actions = []
        
        
        frates_actor = torch.unsqueeze(torch.zeros(128, device=device), 1)
        frates_col_actor = np.zeros((128,1))
        frates_critic = torch.unsqueeze(torch.zeros(128, device=device), 1)
        frates_col_critic = np.zeros((128,1))
        
        entropies = torch.zeros(0, device=device)
        #entropies = np.zeros(3)
        values = torch.zeros(0, device=device)
        global_values = []
        time_av_values = torch.zeros(0, device=device)
        time_av_values_col = torch.zeros(0, device=device)
        
        #gt = []
        #coh = []
        errors = []
        
        trial_index = 0
        time_step = -1
        trial_begins = [0]

        self.task.reset()
        action = 0
        
        h0_actor = torch.zeros(self.hidden_size, dtype=torch.float64, device=device)
        h0_critic = torch.zeros(self.hidden_size, dtype=torch.float64, device=device)

        while trial_index < n_trs: #ciclo su tutti i time-step in fila di tutti gli n_trs trials
            
            time_step += 1

            ob, rwd, done, info = self.task.step(action=action)
            observations.append(ob)
            rewards.append(rwd)
            ob = torch.tensor(np.array([ob]), dtype=torch.float64, device=device)
            ob = torch.unsqueeze(ob, 0) # tensor of size (1,1,3)
            
            action_probs, trajs = self.actor_network(ob, return_dynamics=True, h0=h0_actor,
                                                     time_step=time_step, trial_index=trial_index,
                                                     epoch=self.epoch)  
            log_probs = torch.log(action_probs)
            log_action_probs = torch.cat((log_action_probs, torch.unsqueeze(log_probs[0][0], 0)))
            
            p = action_probs[0][0].clone().detach().to(device=torch.device('cpu')).numpy()
            pip = np.random.uniform()
            if pip < 1:
                action = np.random.choice(np.arange(len(p)), p=p) # 0, 1, 2: fix, right, left
            else: 
                action = np.random.choice(np.arange(len(p)), p=np.array([1/3, 1/3, 1/3])) 
            actions.append(action)             
            if action == 0:
                self.actions_t[0] = 1
            elif action == 1:
                self.actions_t[1] = 1
            elif action == 2:
                self.actions_t[2] = 1
            #self.actions_tt.append(self.actions_t.clone())
            
            relu_trajs = self.actor_network.non_linearity(trajs[0][0])      
            frates_actor = torch.cat((frates_actor, torch.unsqueeze(relu_trajs.clone().detach(), 1)), dim=1)
            
            in_for_critic = torch.unsqueeze(torch.unsqueeze(torch.cat((self.actions_t.clone(), relu_trajs.detach())),0),0)
            self.actions_t.zero_()
            #in_for_critic = torch.unsqueeze(torch.unsqueeze(relu_trajs.detach(),0),0)
            
            value, trajs_critic = self.critic_network(in_for_critic, return_dynamics=True, h0=h0_critic)
            values = torch.cat((values, value[0][0]))  
            time_av_values = torch.cat((time_av_values, torch.unsqueeze(value[0][0].clone().detach(), 1)))  
            
            relu_trajs_critic = self.actor_network.non_linearity(trajs_critic[0][0])
            frates_critic = torch.cat((frates_critic, torch.unsqueeze(relu_trajs_critic.clone().detach(), 1)), dim=1)

            h0_actor = trajs  
            h0_critic = trajs_critic

            if info["new_trial"]:
                
                pp = action_probs[0][0].clone()
                p_r = pp[1] / (pp[1] + pp[2])
                p_l = pp[2] / (pp[1] + pp[2])
                trial_entropy = torch.unsqueeze(- p_r*torch.log(p_r) - p_l*torch.log(p_l), 0)
                entropies = torch.cat((entropies, trial_entropy))
               
                stimuli.append(info["stimuli"])
            
                if actions[-2] == 1:    
                    final_actions.append(1)
                elif actions[-2] == 2:
                    final_actions.append(-1)
                else:
                    final_actions.append(0)
                
                global_values.append(info["global_value"])
                
                if info["gt"] != 0:
                    if actions[-2] != info["gt"]:
                        errors.append(trial_index)
                
                trial_index += 1
                if (trial_index)%100 == 0:
                    print("iteration", trial_index)
                if training:
                    self.trial += 1
                    
                trial_begins.append(time_step+1)
                                    
                h0_actor.fill_(0)
                h0_critic.fill_(0)
                
                frates_actor = frates_actor[:, 1:]
                frates_actor = np.asarray(frates_actor)
                frates_actor = frates_actor[:, 4:29]
                frates_actor = frates_actor.mean(axis=1)
                frates_actor = frates_actor.reshape(-1, 1)
                frates_col_actor = np.concatenate((frates_col_actor, frates_actor), axis=1)
                frates_actor = torch.unsqueeze(torch.zeros(128, device=device), 1)
                
                frates_critic = frates_critic[:, 1:]
                frates_critic = np.asarray(frates_critic)
                frates_critic = frates_critic[:, 4:29]
                frates_critic = frates_critic.mean(axis=1)
                frates_critic = frates_critic.reshape(-1, 1)
                frates_col_critic = np.concatenate((frates_col_critic, frates_critic), axis=1)
                frates_critic = torch.unsqueeze(torch.zeros(128, device=device), 1)
                
                time_av_values = time_av_values[4:29]
                time_av_values = torch.unsqueeze(time_av_values.mean(), -1)
                time_av_values_col = np.concatenate((time_av_values_col, time_av_values))
                time_av_values = torch.zeros(0, device=device)


        errors = np.asarray(errors)

        log_action_probs = log_action_probs[1:]
        frates_col_actor = frates_col_actor[:, 1:]
        frates_col_critic = frates_col_critic[:, 1:]
        final_actions = np.asarray(final_actions)
        global_values = np.asarray(global_values)
        time_av_values_col = np.asarray(time_av_values_col)
        stimuli = np.asarray(stimuli)

        
        if not training:
            self.df_finale -= self.df_finale
            df_somma = self.task.dframe + self.task.complementary
            df_divisione = np.divide(self.task.dframe, df_somma, out=np.zeros_like(self.task.dframe), where=(df_somma != 0))        
            df_divisione = np.round(df_divisione, decimals=2)
            self.df_finale.values[:] = df_divisione

      
        return observations, rewards, actions, log_action_probs, entropies, values, trial_begins, errors, frates_col_actor, frates_col_critic, time_av_values_col, final_actions, global_values, stimuli
         
# =============================================================================================================== 
         
    def update_actor(self, optimizer_actor, obj_function):
        
        obj_function.backward()#retain_graph=True)
        optimizer_actor.step()
        obj_function.detach_()

# =============================================================================================================== 

    def update_critic(self, optimizer_critic, loss_mse):
    
        loss_mse.backward()
        optimizer_critic.step()
        loss_mse.detach_()

# =============================================================================================================== 
       
    def learning_step(self, optimizer_actor, optimizer_critic, epoch, n_trs, train_actor, train_critic, hyper_l): 
        
        begin = time.time() 
        
        device = self.device
        
        # TODO
        #if clip_gradient is not None:
        #        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient) 
        
        #with torch.no_grad():        
        
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        
        observations, rewards, actions, log_action_probs, entropies, values,\
        trial_begins, errors, frates_col_actor, frates_col_critic,\
        time_av_values_col, final_actions, global_values = self.experience(n_trs, training=True)

        cum_rho = np.zeros(0)
        tau_r = np.inf  # Song et al. set this value to 10s only for reaction time tasks
        trial_total_reward = []
        
        for i in range(n_trs):
            
            start = int(trial_begins[i])
            stop = int(trial_begins[i+1])
            
            trial_rewards = rewards[start:stop]
            trial_cumulative_rewards = []
            
            for j in range(len(trial_rewards)):
                
                disc_rew = [r*np.exp(-(i_r)/tau_r) for i_r, r in enumerate(trial_rewards[j+1:])]
                trial_cumulative_rewards.append(np.sum(disc_rew))
            
            trial_cumulative_rewards = np.array(trial_cumulative_rewards)
            trial_total_reward.append(trial_cumulative_rewards[0])
            cum_rho = np.concatenate((cum_rho, trial_cumulative_rewards))
                    
        cum_rho = torch.tensor(cum_rho, device=device)
        trial_total_reward = np.asarray(trial_total_reward)
        
        rewards = torch.Tensor(np.asarray(rewards))
        
        """ QUI NON MOLTO CHIARO DEL PERCHÈ DEBBA FARE DETACH() DAI VALUES... 
            È VERO CHE LI OTTENGO COME OUTPUT DI UNA RETE NEURALE CON PARAMETRI PHI, 
            MA È ANCHE VERO CHE ALL'ACTOR_OPTIMIZER, QUANDO LO INIZIALIZZO DENTRO LA TRAINING FUNCTION,
            GLI PASSO ESPLICITAMENTE SOLO I PARAMETRI THETA 
        """
        actions = np.asarray(actions)
        detached_values = values.clone().detach()#.numpy()
        
        obj_function = self.obj_function(log_action_probs, actions, cum_rho, detached_values,
                                         entropies, n_trs, hyper_l)
        if train_actor:
            self.update_actor(optimizer_actor, obj_function)
        
        loss_mse = self.loss_mse(cum_rho, values, trial_begins, n_trs)
        if train_critic:
            self.update_critic(optimizer_critic, loss_mse)
        
        #log_action_probs.detach_() ...?
        #values.detach_() ...?  
        
        #errors += self.trial
                
        return obj_function, loss_mse, trial_total_reward, errors
    
# ===============================================================================================================
    
    def training(self, n_trs, epochs, lr_a=1e-4, lr_c=1e-4, hyper_l=0, cuda=False, train_actor=True, train_critic=True):
   
        begin = time.time()
    
        self.cuda = cuda
        
        if self.cuda:
            if not torch.cuda.is_available():
                print("Warning: CUDA not available on this machine, switching to CPU")
                self.device = torch.device('cpu')
            else:
                print("Okay with CUDA")
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        device = self.device
        self.actor_network.to(device=device)
        self.critic_network.to(device=device)
        self.actions_t = torch.zeros(3, dtype=torch.float64, device=device)

        optimizer_actor = torch.optim.Adam(self.actor_network.parameters(), lr=lr_a)
        optimizer_critic = torch.optim.Adam(self.critic_network.parameters(), lr=lr_c)
        
        actor_rewards = []
        actor_errors = []
        critic_losses = []
        
        s_o = []
        s_o_critic = []
        
        self.epochs = epochs
        self.epoch = 0
       
        copied_actor = copy.deepcopy(self.actor_network.state_dict())
        copied_critic = copy.deepcopy(self.critic_network.state_dict())
        torch.save(copied_actor, "models/RL_actor_network_bef.pt".format(self.hidden_size))
        torch.save(copied_critic, "models/RL_critic_network_bef.pt".format(self.hidden_size))

        self.df_finale -= self.df_finale
        
        for epoch in range(epochs):   

            self.epoch += 1
           
            obj_function, loss_mse, trial_total_rewards, errors = self.learning_step(optimizer_actor,
                                                                                     optimizer_critic,
                                                                                     epoch, n_trs,
                                                                                     train_actor, train_critic,
                                                                                     hyper_l)
            actor_rewards.append(trial_total_rewards.sum()/n_trs)
            actor_errors.append(len(errors))
            critic_losses.append(loss_mse.detach())#.numpy())
        
            #s_o.append(self.actor_network.so.data.clone().detach())#.numpy())
            #s_o_critic.append(self.critic_network.so.data.clone().detach())#.numpy())
            
            if (epoch+1)%500 == 0 or epoch < 5:
                print("iteration", epoch+1, "- %.2f s so far" %((time.time()-begin)))
        
        df_somma = self.task.dframe + self.task.complementary
        df_divisione = np.divide(self.task.dframe, df_somma, out=np.zeros_like(self.task.dframe), where=(df_somma != 0))        
        df_divisione = np.round(df_divisione, decimals=2)
        self.df_finale.values[:] = df_divisione
        self.df_finale.to_pickle('training_heatmap.pkl')

        copied_actor2 = copy.deepcopy(self.actor_network.state_dict())
        copied_critic2 = copy.deepcopy(self.critic_network.state_dict())
        torch.save(copied_actor2, "models/RL_actor_network.pt".format(self.hidden_size))
        torch.save(copied_critic2, "models/RL_critic_network.pt".format(self.hidden_size))
        
        torch.save(actor_rewards, 'models/actor_rewards.pt')
        torch.save(actor_errors, 'models/actor_errors.pt')
        torch.save(critic_losses, 'models/critic_loss.pt')
        
        #torch.save(s_o, 'models/s_o.pt')
        #torch.save(s_o_critic, 'models/s_o_critic.pt')
            
        print("\nDEVICE: " + str(device) + ". It took %.2f m for %i epochs. %i trials per epoch." %((time.time()-begin)/60, epochs, n_trs))