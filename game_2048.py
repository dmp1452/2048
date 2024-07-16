
import pygame
import numpy as np
from enum import Enum

BLACK = (0,0,0)
BLOCK_SIZE = 160
pygame.init()


class Direction(Enum):
    #Enumeration for direction constants.
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 0

class Game:
    def __init__(self, w= 640,h = 690):
        """
        Initialize the game with a window of width w and height h.
        """
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('2048')
        self.reset()

    def reset(self):
        """
        Reset the game state, including the board, score, and frame iteration.
        Place two initial tiles on the board.
        """
        self.board = np.zeros((4,4))
        self.score = 0
        self.last_score =0
        self.frame_iteration =0
        self.update_ui()
        self.place()
        self.place()
    
    def place(self):
        """
        Place a new tile (2 or 4) in a random empty spot on the board.
        """
        zeroes = np.where(self.board==0)
        if len(zeroes[0])>0:
            zeroes = zip(zeroes[0],zeroes[1])
            zeroes =list(zeroes)
            index = np.random.choice(len(zeroes))
            tile =zeroes[index]
            x,y,number = tile[0],tile[1],np.random.choice((2,4),p=(.9,.1))
            self.board[x][y] = number

            font =pygame.font.Font(None,90)
            message_surface = font.render(str(number), True, (255,0,0))
            message_rect = message_surface.get_rect()
            message_rect.center=(y*BLOCK_SIZE+BLOCK_SIZE//2,x*BLOCK_SIZE+BLOCK_SIZE//2+50)
            self.display.blit(message_surface,message_rect)
    
    def next_step(self,action):
        """
        Execute the next step of the game based on the action provided.
        """
        self.frame_iteration+=1
        old_board = np.copy(self.board)
        self.move(action)
        self.update_ui()
        #pygame.time.delay(150)
        self.place()

        reward =0
        game_over = False

        if self.last_score>0:
            reward +=self.last_score

        if np.array_equal(old_board,self.board):
            reward-=5
        
        free_space_reward = np.sum(self.board == 0)
        reward += free_space_reward

        if not self.check_loss():
            game_over= True
            reward -=20

        return reward, game_over, self.score

    def move(self,action):
        """
        Move the tiles on the board in the specified direction.
        """
        if action ==Direction.LEFT.value:
            self.board=np.array([self.merge(row) for row in self.board])

        elif action == Direction.RIGHT.value:
            self.board =np.flip(self.board,axis =1)
            self.board=np.array([self.merge(row) for row in self.board])
            self.board =np.flip(self.board,axis =1)

        elif action == Direction.UP.value:
            self.board = np.flip(self.board, axis =1)
            self.board = np.transpose(self.board)
            self.board=np.array([self.merge(row) for row in self.board])
            self.board = np.transpose(self.board)
            self.board = np.flip(self.board, axis =1)

        else: #DOWN

            self.board = np.transpose(self.board)
            self.board = np.flip(self.board, axis =1)
            self.board=np.array([self.merge(row) for row in self.board])
            self.board = np.flip(self.board, axis =1)
            self.board = np.transpose(self.board)

    
    def merge(self,row):
        """
        Merge tiles in a single row to the left.
        """
        self.last_score= 0
        new_row = [num for num in row if num != 0]
        for i in range(len(new_row)-1):
            if new_row[i] ==new_row[i+1]:
                new_row[i] *=2
                self.score+=new_row[i]
                self.last_score+=new_row[i]
                new_row[i+1]=0
                i+=1
        new_row = [num for num in new_row if num !=0]
        return new_row + [0]*(len(row)- len(new_row))


    def check_loss(self):
        """
        Check if the game is lost (no moves available).
        """
        return 0 in self.board.flat or self.can_combine()
    
    def can_combine(self):
        """
        Check if any tiles can be combined.
        """
        for x in range(3):
            for y in range(3):
                if self.board[x][y]==self.board[x+1][y] or self.board[x][y]==self.board[x][y+1]:
                    return True
        for x in range(3,0,-1):
            if self.board[x][3]==self.board[x-1][3]or self.board[x][3]==self.board[x][2]:
                return True
        return False

    def update_ui(self):
        """
        Update the game's user interface.
        """
        self.display.fill(BLACK)
        font =pygame.font.Font(None,45)
        message_surface = font.render("Score: "+ str(self.score), True, (255,0,0))
        message_rect = message_surface.get_rect()
        message_rect.center=(320,25)
        self.display.blit(message_surface,message_rect)
        font =pygame.font.Font(None,90)
        for x in range(4):
            for y in range(4):
                number = self.board[x][y]
                pygame.draw.rect(self.display,(0,75,75),(y*BLOCK_SIZE,x*BLOCK_SIZE+50,BLOCK_SIZE,BLOCK_SIZE))
                pygame.draw.rect(self.display,BLACK,(y*BLOCK_SIZE,x*BLOCK_SIZE+50,BLOCK_SIZE-1,BLOCK_SIZE-1),1)
                if number !=0:
                    message_surface = font.render(str(int(number)), True, (205,0,0))
                    message_rect = message_surface.get_rect()
                    message_rect.center=(y*BLOCK_SIZE+BLOCK_SIZE//2,x*BLOCK_SIZE+BLOCK_SIZE//2+50)
                    self.display.blit(message_surface,message_rect)
        pygame.display.update()


#Used to play manually
"""
game = Game()
while True:
    for event in pygame.event.get():
        if event.type ==pygame.QUIT:
            pygame.quit()
            print("quit")
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                game.next_step(Direction.LEFT)
            if event.key == pygame.K_RIGHT:
                game.next_step(Direction.RIGHT)
            if event.key == pygame.K_UP:
                game.next_step(Direction.UP)
            if event.key == pygame.K_DOWN:
                game.next_step(Direction.DOWN)
    pygame.display.update() 
        
        """
        