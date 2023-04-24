import cv2
import numpy as np
import time
class cv():
    def __init__(self) -> None:
        pass
    def namedWindow(self, name:str, flags:None = cv2.WINDOW_AUTOSIZE):
        cv2.namedWindow(name, flags)
    def imshow(self, name:str, img_array:np.ndarray):
        cv2.imshow(name, img_array)
    def destroyAllWindows(self):
        cv2.destroyAllWindows()
    def setMouseCallback(self, name:str, func:None):
        cv2.setMouseCallback(name, func)
    def waitkey(self, flags=0):
        return cv2.waitKey(flags)

class MineSweeper(cv):
    def __init__(self, h=15, w=15, mine_number=35) -> None:
        super(MineSweeper, self)
        self.namedWindow("MineSweeper")
        self.size = np.array([h, w])
        self.mine_map = np.zeros(self.size)
        self.zero_area = np.zeros_like(self.mine_map)
        self.playing_map = np.zeros(self.size) -2
        self.mine_number = mine_number
        self.flag = cv2.imread("./flag.png")
        self.zero_queue = []
        # position of mines
        all_corr = np.array(np.where(self.mine_map==0)).T
        point = np.random.choice(h*w, mine_number, replace=False)
        self.mine_map[all_corr[point].T[0], all_corr[point].T[1]] = -1

        # calcute the number of mines at every point
        kernel_size = (3, 3)
        kernel = np.ones(kernel_size)*-1
        mine_map_ = cv2.filter2D(self.mine_map, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
        self.mine_map[self.mine_map == 0] = mine_map_[self.mine_map == 0]
        self.mine_map = self.mine_map.astype(np.int8)
        self.zero_area[self.mine_map == 0] = 1
        # create a map to display
        self.answer_img = np.zeros([self.size[0]*40 + (self.size[0] + 1)*3, self.size[1]*40 + (self.size[1] + 1)*3, 3], dtype=np.uint8) + 150
        self.mask_img = np.zeros([self.size[0]*40 + (self.size[0] + 1)*3, self.size[1]*40 + (self.size[1] + 1)*3, 3], dtype=np.uint8) + 200
        lines_points_h = [[[0, (i*43) + 1], [self.answer_img.shape[1], (i*43) + 1]] for i in range(self.size[0]+1)]
        lines_points_v = [[[(i*43) + 1, 0], [(i*43) + 1, self.answer_img.shape[0]]] for i in range(self.size[1]+1)]
        for line_v in lines_points_v:
            self.answer_img = cv2.line(self.answer_img, line_v[0], line_v[1], (30, 30, 30), 3 )
            self.mask_img = cv2.line(self.mask_img, line_v[0], line_v[1], (30, 30, 30), 3 )

        for line_h in lines_points_h:
            self.answer_img = cv2.line(self.answer_img, line_h[0], line_h[1], (30, 30, 30), 3 )
            self.mask_img = cv2.line(self.mask_img, line_h[0], line_h[1], (30, 30, 30), 3 )

        colors = [(0,0,0), (200,0,0), (50,200,50), (0,0,200), (80,0,0), (0,60,80), (0,0,80), (0,0,80), (0,0,80), (0,0,80), (0,0,0)]
        text = ['', '1', '2', '3', '4', '5', '6', '7', '8', '!!']
        self.corr_maping = {}
        for i, nums in enumerate(self.mine_map):
            for j, num in enumerate(nums):
                self.corr_maping[f"{j}_{i}"] = [lines_points_v[j][0][0], lines_points_h[i][0][1]]
                cv2.putText(self.answer_img, text[num], [lines_points_v[j][0][0]+11, lines_points_h[i+1][0][1]-11], cv2.FONT_HERSHEY_SIMPLEX,
                            1, colors[num], 4)
        self.display_img = self.mask_img.copy()
    
    def found_zero(self, x, y):
        self.zero_queue.append([x, y])
        move_to = [[0, -1], [1, -1], [1, 0], [1, 1],
                 [0, 1], [-1, 1], [-1, 0], [-1, -1]]
        while len(self.zero_queue) != 0:
            queue = []
            for x, y in self.zero_queue:
                self.zero_area[y][x] = 2
                for move in move_to:
                    n_x, n_y = (x + move[0]), (y + move[1])
                    check = (n_x >= 0 and n_x < self.size[1] 
                            and n_y >= 0 and n_y < self.size[0])
                    if check and self.zero_area[n_y][n_x] != 1:
                        self.zero_area[n_y][n_x] = 2
                        check = False
                    next_pos = [n_x, n_y]
                    if check and next_pos not in queue:
                            queue.append(next_pos)
            self.zero_queue = queue
        
    def show_answer(self, x, y):
        for x, y in zip (x, y):
            i, j = self.corr_maping[f"{x}_{y}"]
            self.display_img[j:j+40, i:i+40, :] = (
                self.answer_img[j:j+40, i:i+40, :])
    
    def set_flag(self, x, y, set_):
        i, j = self.corr_maping[f"{x}_{y}"]
        if set_:
            self.display_img[j:j+40, i:i+40, :][self.flag != 255] = (
                self.flag[self.flag != 255])
        else:
            self.display_img[j:j+40, i:i+40, :] = (
                self.mask_img[j:j+40, i:i+40, :])
            
    def click_once(self, x, y, keyboard=False):
        game_end = False
        reward = 0
        if self.playing_map[y[0]][x[0]] == -2 and self.mine_map[y[0]][x[0]] != -1:
            reward = 1
        self.playing_map[y[0]][x[0]] = self.mine_map[y[0]][x[0]]
        self.show_answer(x, y)
        if self.mine_map[y[0]][x[0]] == 0:
            self.found_zero(x[0], y[0])
            y, x = np.where(self.zero_area >= 2)
            self.playing_map[self.zero_area >= 2] = self.mine_map[self.zero_area >= 2]
            self.show_answer(x, y)

        if len(self.playing_map[self.playing_map <= -2]) == self.mine_number:
            game_end = True
            text = "You Win!!"
            cv2.putText(self.display_img, text, [20, self.display_img.shape[0]//2], cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 200, 0), 10)
            self.setMouseCallback('MineSweeper', self.game_over)
            reward = 100
            if keyboard:
                print(text)
        
        
        if self.mine_map[y[0]][x[0]] == -1:
            game_end = True
            y, x = np.where(self.mine_map == -1)
            self.show_answer(x, y)
            self.playing_map[self.mine_map == -1] = self.mine_map[self.mine_map == -1]
            text = "Game Over!!"
            reward = -100
            cv2.putText(self.display_img, text, [20, self.display_img.shape[0]//2], cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 200), 10)
            self.setMouseCallback('MineSweeper', self.game_over)
            if keyboard:
                print(text)
        if keyboard:
            print(self.playing_map.astype(np.int8))

        return game_end, reward
    
    def game_over(self, event, x, y, flags, userdata):
        pass

    def show_xy(self, event=0, x=0, y=0,flags=0, userdata=None):
        if event == 1:
            x = [x//43, ]
            y = [y//43, ]
            
            if self.playing_map[y[0]][x[0]] == -3:
                return None
            self.click_once(x, y)

        elif event == 2 :
            x = x//43
            y = y//43
            if self.playing_map[y][x] >-2 :
                return None
            self.playing_map[y][x] = -self.playing_map[y][x] - 5
            self.set_flag(x, y, -self.playing_map[y][x] - 2)

        self.imshow("MineSweeper", self.display_img)
        return None
        
    def run_mouse(self):
        self.imshow("MineSweeper", self.display_img)
        while True:
            self.setMouseCallback('MineSweeper', self.show_xy)
            key = self.waitkey(0)
            if key == ord('q'):
                self.destroyAllWindows()
                return False
            if key == ord('r'):
                return True
    
    def run_keyboard(self):
        print(self.playing_map.astype(np.int8))
        game_end = False
        while not game_end:
            keyboard_in = input('輸入座標(x,y,l/r):').split(',')
            if keyboard_in[0] == 'q':
                self.destroyAllWindows()
                return None
            elif keyboard_in[0] == 'r':
                return True
            else:
                x, y, l_r = keyboard_in
            x = [int(x), ]
            y = [int(y), ]
            if l_r == 'l':
                if self.playing_map[y[0]][x[0]] == -3:
                    continue
                game_end, reward = self.click_once(x, y, True)

            elif l_r == 'r':
                self.playing_map[y[0]][x[0]] = -self.playing_map[y[0]][x[0]] - 5
                self.set_flag(x[0], y[0], -self.playing_map[y[0]][x[0]] - 2)
                reward = 0
                print(self.display_img)
            cv2.imwrite(f'./image/{time.time_ns()}.png',self.display_img)
            print(f"reward: {reward}")
        if input("again? (y/n):") == "y":
            return True
        else:
            return False

if __name__=='__main__':
    again = True
    while again:
        ms = MineSweeper(h=15, w=15, mine_number=10)
        # again = ms.run_keyboard()
        again = ms.run_mouse()