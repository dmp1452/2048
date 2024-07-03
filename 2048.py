
import pygame
import numpy as np
import sys
from enum import Enum

BLACK = (0,0,0)
BLOCK_SIZE = 160
pygame.init()
font =pygame.font.Font(None,45)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Game:
    def __init__(self, w= 640,h = 690):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('2048')
        self.reset()

    def reset(self):
        self.board = np.zeros((4,4))
        self.score = 0
        self.frame_iteration =0
        self.update_ui()
        self.place()
        self.place()
    
    def place(self):
        zeroes = np.where(self.board==0)
        if len(zeroes[0])>0:
            zeroes = zip(zeroes[0],zeroes[1])
            zeroes =list(zeroes)
            index = np.random.choice(len(zeroes))
            tile =zeroes[index]
            x,y,number = tile[0],tile[1],np.random.choice((2,4),p=(.9,.1))
            self.board[x][y] = number

            message_surface = font.render(str(number), True, (255,0,0))
            message_rect = message_surface.get_rect()
            message_rect.center=(y*BLOCK_SIZE+BLOCK_SIZE//2,x*BLOCK_SIZE+BLOCK_SIZE//2+50)
            self.display.blit(message_surface,message_rect)
    
    def next_step(self,action):
        self.frame_iteration+=1
        self.move(action)
        self.update_ui()
        pygame.time.delay(150)
        self.place()
        if not self.check_loss():
            print("Game lost")

    def move(self,action):
        if action ==Direction.LEFT:
            self.board=np.array([self.merge(row) for row in self.board])

        elif action == Direction.RIGHT:
            self.board =np.flip(self.board,axis =1)
            self.board=np.array([self.merge(row) for row in self.board])
            self.board =np.flip(self.board,axis =1)

        elif action == Direction.UP:
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
        new_row = [num for num in row if num != 0]
        for i in range(len(new_row)-1):
            if new_row[i] ==new_row[i+1]:
                new_row[i] *=2
                self.score+=new_row[i]
                new_row[i+1]=0
                i+=1
        new_row = [num for num in new_row if num !=0]
        return new_row + [0]*(len(row)- len(new_row))


    def check_loss(self):
        return 0 in self.board.flat or self.can_combine()
    
    def can_combine(self):
        for x in range(3):
            for y in range(3):
                if self.board[x][y]==self.board[x+1][y] or self.board[x][y]==self.board[x][y+1]:
                    return True
        for x in range(3,0,-1):
            if self.board[x][3]==self.board[x-1][y]or self.board[x][3]==self.board[x][2]:
                return True
        return False

    def update_ui(self):
        self.display.fill(BLACK)
        message_surface = font.render("Score: "+ str(self.score), True, (255,0,0))
        message_rect = message_surface.get_rect()
        message_rect.center=(320,25)
        self.display.blit(message_surface,message_rect)
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
                #print("Left")
            if event.key == pygame.K_RIGHT:
                game.next_step(Direction.RIGHT)
                print("Right")
            if event.key == pygame.K_UP:
                game.next_step(Direction.UP)
                print("Up")
            if event.key == pygame.K_DOWN:
                game.next_step(Direction.DOWN)
                print("Down")
    pygame.display.update() 
        
        
        