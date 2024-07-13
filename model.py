import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        """
        Initialize the neural network with one hidden layer.
        :param input_size: Number of input features
        :param hidden_size: Number of neurons in the hidden layer
        :param output_size: Number of output neurons
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        """
        Forward pass through the network.
        :param x: Input tensor
        :return: Output tensor
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name = 'model.pth'):
        """
        Save the model parameters to a file.
        :param file_name: Name of the file to save the model
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        """
        Initialize the QTrainer with model, learning rate, and discount factor.
        :param model: Neural network model to train
        :param lr: Learning rate
        :param gamma: Discount factor for future rewards
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform a single training step for the model.
        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state after the action
        :param done: Whether the episode is done
        """
        
        state = torch.tensor(np.array(state, dtype=float),dtype=torch.float)
        next_state =torch.tensor(np.array(next_state,dtype=float),dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)
        done = torch.tensor(done,dtype=torch.bool)
        # Ensure inputs are batched
        if len(state.shape)==1:
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done =torch.unsqueeze(done,0)

        pred = self.model((state))
        target =pred.clone()

        with torch.no_grad():
            next_pred = self.model(next_state)
            max_next_pred = torch.max(next_pred, dim=1)[0]
            Q_new = reward + self.gamma * max_next_pred * (~done)
        """
        next_pred = self.model(next_state)
        max_next_pred = torch.max(next_pred,dim=1)[0]
        Q_new = reward +self.gamma*max_next_pred*(~done)"""

        # Update target Q-values
        target[range(len(action)),action]=Q_new
        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)

        # Backpropagation
        loss.backward()
        self.optimizer.step()