import torch
import random
import numpy as np
from collections import deque
from game_2048 import Game
from model import Linear_QNet, QTrainer
from helper import plot
MAX = 10000  # Maximum size of memory buffer
BATCH_SIZE = 100 # Batch size for training
LR = .01 # Learning rate

class Agent:
    def __init__(self):
        """
        Initialize the Agent with required attributes including model, trainer, and memory.
        """
        self.num_games =0
        self.epsilon =0
        self.gamma =.9
        self.memory = deque(maxlen =MAX)
        self.model = Linear_QNet(16,128,4)
        self.trainer = QTrainer(self.model, lr= LR, gamma =self.gamma)
    
    def get_state(self,game):
        """
        Get the current state of the game.
        :param game: Game object
        :return: Numpy array representing the flattened board state
        """
        return np.array(game.board.flat, dtype=int)
    
    def remember(self,state,action,reward,next_state,done):
        """
        Store the experience in the memory buffer.
        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state after the action
        :param done: Whether the game is over
        """
        self.memory.append((state,action,reward,next_state,done))
    
    def train_long_memory(self):
        """
        Train the model using experiences sampled from the memory buffer.
        """
        mini_sample = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory
        states, actions, rewards, next_states, dones = map(np.array, zip(*mini_sample))
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        """
        Train the model with a single experience.
        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state after the action
        :param done: Whether the game is over
        """
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        """
        Decide the action to take based on the current state using an epsilon-greedy strategy.
        :param state: Current state
        :return: Action to take
        """
        self.epsilon = 80 - self.num_games
        if random.randint(0,200)<self.epsilon:
            return random.randint(0,3)
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
            return torch.argmax(prediction).item()
    
def train():
    """
    Main training loop for the agent.
    """
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