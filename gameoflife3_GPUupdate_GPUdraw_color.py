import numba, copy, time, cv2
from numba import cuda
import numpy as np
from win32api import (GetKeyState)
from win32keycodes import VK_CODE
from numba.cuda import random
import random as random2
from PIL import ImageGrab

@cuda.jit
def update_board(array, n_list, n_array):

    x = cuda.threadIdx.x
    y = cuda.blockIdx.x

    if y < array.shape[0] and x < array[0].size:
        n = 0
        r = 0
        g = 0
        b = 0
        for x2,y2 in n_list:
            if y+y2 < array.shape[0] and y+y2 >= 0 and x+x2 < array[0].size and x+x2 >= 0:
                if array[y+y2][x+x2][0] != 0 or array[y+y2][x+x2][1] != 0 or array[y+y2][x+x2][2] != 0:
                    n+=1
                    r += array[y+y2][x+x2][0]
                    g += array[y+y2][x+x2][1]
                    b += array[y+y2][x+x2][2]
        if array[y][x][0] == 0 and array[y][x][1] == 0 and array[y][x][2] == 0 and n == 3:
            #n_array[y][x] = 1
            n_array[y][x][0] = r/n
            n_array[y][x][1] = g/n
            n_array[y][x][2] = b/n
        elif n != 3 and n != 2:
            n_array[y][x][0] = 0
            n_array[y][x][1] = 0
            n_array[y][x][2] = 0

'''def update_board_cpu(array, n_list, n_array):
    for y,row in enumerate(array):
        for x,v in enumerate(row):
            n=0
            for x2,y2 in n_list:
                if y+y2 < array.shape[0] and y+y2 >= 0 and x+x2 < array[0].size and x+x2 >= 0:
                    if array[y+y2][x+x2][0] != 0 and array[y+y2][x+x2][1] != 0 and array[y+y2][x+x2][2] != 0:
                        n+=1
            if array[y][x][0] == 0 and array[y][x][1] == 0 and array[y][x][2] == 0 and n == 3:
                n_array[y][x] = 1
            elif n != 3 and n != 2:
                n_array[y][x] = 0'''

def rand_board(array):
    for y,row in enumerate(array):
        for x,_ in enumerate(row):
            array[y][x] = np.array([random.randint(0,1)], np.float32)

@cuda.jit
def rand_color(array, black_percent, rng_states):

    x = cuda.threadIdx.x
    y = cuda.blockIdx.x
    thread_id = cuda.grid(1)

    if y < array.shape[0] and x < array[0].size:  # make sure in bound
        if random.xoroshiro128p_uniform_float32(rng_states, thread_id) < black_percent:
            array[y][x][0] = 0
            array[y][x][1] = 0
            array[y][x][2] = 0
        else:
            array[y][x][0] = random.xoroshiro128p_uniform_float32(rng_states, thread_id)
            array[y][x][1] = random.xoroshiro128p_uniform_float32(rng_states, thread_id)
            array[y][x][2] = random.xoroshiro128p_uniform_float32(rng_states, thread_id)

@cuda.jit
def rand_kill(array, black_percent, rng_states):

    x = cuda.threadIdx.x
    y = cuda.blockIdx.x
    thread_id = cuda.grid(1)

    if y < array.shape[0] and x < array[0].size:  # make sure in bound
        if random.xoroshiro128p_uniform_float32(rng_states, thread_id) < black_percent:
            array[y][x][0] = 0
            array[y][x][1] = 0
            array[y][x][2] = 0

KEY_DICT = {VK_CODE["f"]:False,VK_CODE["c"]:False,VK_CODE["space"]:False,VK_CODE["r"]:False}
def key_down(key):
    state = GetKeyState(key)
    if (state != 0) and (state != 1):
        if KEY_DICT[key]:
            return False
        KEY_DICT[key] = True
        return True
    else:
        KEY_DICT[key] = False
        return False
            
WIDTH,HEIGHT = 1000,1000
SCALE = [1,1]
board = np.array([np.array(list([np.array([0,0,0], np.float32) for x in range(WIDTH)])) for y in range(HEIGHT)])
print(board.shape[0])
screen_width, screen_height = WIDTH*SCALE[0],HEIGHT*SCALE[1]

inital_res = 1920,1080
bound = [x/2 - WIDTH/2 for i,x in enumerate(inital_res)] + [x/2 + HEIGHT/2 for i,x in enumerate(inital_res)]

def main():
    global board

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
    #true=cpu false=gpu(kinda)

    GAME_RUN = True
    while GAME_RUN:
        s_time = time.time()
        k = cv2.waitKey(1)
        if k == ord('f'):
            update_board[HEIGHT, WIDTH](copy.copy(board), neighbor_list, board)
        elif k == ord('c'):
            board = np.array([np.array(list([np.array([0,0,0], np.float32) for x in range(WIDTH)])) for y in range(HEIGHT)])
        elif k == ord('r'):
            rand_board(board)
        elif k == ord('t'):
            rand_color[HEIGHT, WIDTH](board,0.7,random.create_xoroshiro128p_states(WIDTH * HEIGHT, seed=random2.random()*100000))
        elif k == ord(' '):
            PLAY = not PLAY
        if key_down(VK_CODE["s"]):
            PLAY = False
            board = cv2.cvtColor(cv2.cvtColor(np.array(ImageGrab.grab(bound)), cv2.CV_16U), cv2.COLOR_BGR2RGB)
            rand_kill[HEIGHT, WIDTH](board, 0.7, random.create_xoroshiro128p_states(WIDTH * HEIGHT, seed=random2.random()*100000))
        if key_down(VK_CODE["q"]):
            GAME_RUN = False
            cv2.destroyAllWindows()
            break
        print(f'keys {time.time()-s_time}')

        if PLAY:
            s_time = time.time()
            update_board[HEIGHT, WIDTH](copy.copy(board), neighbor_list, board)
            print(f'board calc {time.time()-s_time}')

        '''if GetKeyState(VK_CODE[""]):
            pos = [int(x/SCALE[i]) for i,x in enumerate(pygame.mouse.get_pos())]
            board[pos[1]][pos[0]] = np.array([1,1,1])
        if pygame.mouse.get_pressed()[1]:
            pos = [int(x/SCALE[i]) for i,x in enumerate(pygame.mouse.get_pos())]
            board[pos[1]][pos[0]] = np.array([0,0,0])'''

        #draw_board[HEIGHT, WIDTH](screenf, board, SCALE)

        s_time = time.time()

        #board.reshape((board.size, -1))
        #print(board)
        frame = cv2.cvtColor(board, cv2.CV_16U)
        cv2.imshow("The Game of Life", frame)

        print(f'draw {time.time()-s_time}')

        print(f'total {time.time()-last}\n')
        last = time.time()

if __name__ == "__main__":
    main()