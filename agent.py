import torch
import random
import numpy as np
from collections import deque
from game_2048 import Game
from model import Linear_QNet, QTrainer
from helper import plot
MAX = 100000
BATCH_SIZE = 1000
LR = .01

class Agent:
    def __init__(self):
        self.num_games =0
        self.epsilon =0
        self.gamma =.9
        self.memory = deque(maxlen =MAX)
        self.model = Linear_QNet(16,256,4)
        self.trainer = QTrainer(self.model, lr= LR, gamma =self.gamma)
    
    def get_state(self,game):
        state = game.board.flat
        return np.array(state, dtype=int)
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    
    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones=zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        self.epsilon = 80 - self.num_games
        final_move=[0,0,0,0]
        if random.randint(0,200)<self.epsilon:
            move = random.randint(0,3)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move]=1
        final_move = np.atleast_1d(final_move)
        indices = np.where(final_move == 1)[0][0]
        return indices
    
def train():
    plot_scores =[]
    plot_mean_scores =[]
    total_score =0
    record=0
    agent =Agent()
    game = Game()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward,done,score = game.next_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old,final_move,reward,state_new,done)
        agent.remember(state_old,final_move,reward,state_new,done)
        if done:
            game.reset()
            agent.num_games+=1
            agent.train_long_memory()
            if score >record:
                record =score
                agent.model.save()
            print('Game: ',agent.num_games,'Score: ', score, 'Record: ', record)
            plot_scores.append(score)
            total_score +=score
            mean_score = total_score/ agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)

if __name__ =='__main__':
    train()