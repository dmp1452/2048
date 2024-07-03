
import pygame
import numpy as np
import sys
from enum import Enum

BLACK = (0,0,0)
BLOCK_SIZE = 160
pygame.init()
font =pygame.font.Font(None,35)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Game:
    def __init__(self, w= 640,h = 680):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('2048')
        self.reset()
        print(self.board)
        self.update_ui()

    def reset(self):
        self.board = np.zeros((4,4))
        self.place()
        self.place()
    
    def place(self):
        zeroes = np.where(self.board==0)
        if zeroes:
            zeroes = zip(zeroes[0],zeroes[1])
            zeroes =list(zeroes)
            index = np.random.choice(len(zeroes))
            tile =zeroes[index]
            self.board[tile[0]][tile[1]] = np.random.choice((2,4),p=(.9,.1))



    def update_ui(self):
        self.display.fill(BLACK)
        for x in range(4):
            for y in range(4):
                number = self.board[x][y]
                pygame.draw.rect(self.display,(255,255,255),(y*BLOCK_SIZE,x*BLOCK_SIZE+40,BLOCK_SIZE,BLOCK_SIZE))
                pygame.draw.rect(self.display,BLACK,(y*BLOCK_SIZE,x*BLOCK_SIZE+40,BLOCK_SIZE-1,BLOCK_SIZE-1),1)
                if number !=0:
                    print(str(y),str(x))
                    message_surface = font.render(str(number), True, (200,50,50))
                    message_rect = message_surface.get_rect()
                    x_pos = x*BLOCK_SIZE+BLOCK_SIZE//2
                    y_pos = y*BLOCK_SIZE+BLOCK_SIZE//2+40
                    message_rect.center=(y*BLOCK_SIZE+BLOCK_SIZE//2,x*BLOCK_SIZE+BLOCK_SIZE//2+40)
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
                print("Left")
            if event.key == pygame.K_RIGHT:
                print("Right")
            if event.key == pygame.K_UP:
                print("Up")
            if event.key == pygame.K_DOWN:
                print("Down")
    pygame.display.update() 
        
        
        