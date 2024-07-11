import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        
        state = torch.tensor(np.array(state, dtype=float),dtype=torch.float)
        next_state =torch.tensor(np.array(next_state,dtype=float),dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.long)
        done = torch.tensor(done,dtype=torch.bool)#added

        if len(state.shape)==1:
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done =torch.unsqueeze(done,0)

        pred = self.model((state))
        target =pred.clone()
        next_pred = self.model(next_state)#
        max_next_pred = torch.max(next_pred,dim=1)[0]
        Q_new = reward +self.gamma*max_next_pred*(~done)
        target[range(len(action)),action]=Q_new

        """
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i]+self.gamma*torch.max(self.model(next_state[i]))
            action_index = action[i].item()
            target[i,action_index] = Q_new
        """
        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()
        self.optimizer.step()