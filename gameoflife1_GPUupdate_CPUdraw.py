import numba, pygame, copy, random, time
from numba import cuda
import numpy as np
#from numba.cuda import random

@cuda.jit
def update_board(array, n_list, n_array):

    x = cuda.threadIdx.x
    y = cuda.blockIdx.x

    if y < array.shape[0] and x < array[0].size:  # make sure in bound
        n = 0
        for x2,y2 in n_list:
            if y+y2 < array.shape[0] and y+y2 >= 0 and x+x2 < array[0].size and x+x2 >= 0:
                if array[y+y2][x+x2] == 1:
                    n+=1
        if array[y][x] == 0 and n == 3:
            n_array[y][x] = 1
        elif n != 3 and n != 2:
            n_array[y][x] = 0

def update_board_cpu(array, n_list, n_array):
    for y,row in enumerate(array):
        for x,v in enumerate(row):
            n=0
            for x2,y2 in n_list:
                if y+y2 < array.shape[0] and y+y2 >= 0 and x+x2 < array[0].size and x+x2 >= 0:
                    if array[y+y2][x+x2] == 1:
                        n+=1
            if array[y][x] == 0 and n == 3:
                n_array[y][x] = 1
            elif n != 3 and n != 2:
                n_array[y][x] = 0

def rand_board(array):
    for y,row in enumerate(array):
        for x,_ in enumerate(row):
            array[y][x] = random.randint(0,1)
            
WIDTH = 100
HEIGHT = 100
SCALE = [5,5]
board = np.array([np.array(list([0 for x in range(WIDTH)])) for y in range(HEIGHT)])
print(board.shape[0])
screen_width, screen_height = WIDTH*SCALE[0],HEIGHT*SCALE[1]

def main():
    global board

    pygame.init()
    pygame.display.set_caption("The Game of Life")

    screenf = pygame.display.set_mode([screen_width, screen_height])

    neighbor_list = []
    for y in range(-1,2):
        for x in range(-1,2):
            neighbor_list.append(np.array([x,y]))
    neighbor_list = np.array([x for x in neighbor_list if x[0]!=0 or x[1]!=0])

    print(board)
    #print(neighbor_list)
    update_board[HEIGHT, WIDTH](copy.copy(board), neighbor_list, board)
    print(board)

    last = time.time()

    PLAY = False

    GAME_RUN = True
    while GAME_RUN:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_run = False
                pygame.display.quit()
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                #screenf.blit(screenf, (452, 0))
                if event.key == pygame.K_f:
                    update_board[HEIGHT, WIDTH](copy.copy(board), neighbor_list, board)
                if event.key == pygame.K_c:
                    board = np.array([np.array(list([0 for x in range(WIDTH)])) for y in range(HEIGHT)])
                if event.key == pygame.K_r:
                    rand_board(board)
                if event.key == pygame.K_SPACE:
                    PLAY = not PLAY
                if event.key == pygame.K_t:
                    if PLAY != None:
                        PLAY = None
                    else:
                        PLAY = False

        s_time = time.time()
        if PLAY:
            update_board[HEIGHT, WIDTH](copy.copy(board), neighbor_list, board)
        elif PLAY == None:
            update_board_cpu(copy.copy(board), neighbor_list, board)
        print(f'board calc {time.time()-s_time}')

        if pygame.mouse.get_pressed()[0]:
            pos = [int(x/SCALE[i]) for i,x in enumerate(pygame.mouse.get_pos())]
            board[pos[1]][pos[0]] = 1
        if pygame.mouse.get_pressed()[1]:
            pos = [int(x/SCALE[i]) for i,x in enumerate(pygame.mouse.get_pos())]
            board[pos[1]][pos[0]] = 0


        screenf.fill(0)

        #draw_board[HEIGHT, WIDTH](screenf, board, SCALE)

        s_time = time.time()
        for y,row in enumerate(board):
            for x,t in enumerate(row):
                if t == 1:
                    pygame.draw.rect(screenf, (255,255,255), (x*SCALE[0], y*SCALE[1], SCALE[0], SCALE[1]))
        pygame.display.flip()

        print(f'draw {time.time()-s_time}')

        print(f'total {time.time()-last}\n')
        last = time.time()

if __name__ == "__main__":
    main()