from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ['A-SZ','SZS-','---S','--ZG',],
    "8x8": [
        'A--SSS--',
        'ZZRZ----',
        'ZSZ----S',
        '--S-ZR--',
        '-SS--Z--',
        '--Z-----',
        '----SZ-Z',
        '-SZ-S--G',
    ],
}


def is_map_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "Z":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(size: int = 8, p: float = 0.8) -> List[str]:
    valid = False
    board = [] 

    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "Z"], (size, size), p=[p, 1 - p])
        board[0][0] = "A"
        board[-1][-1] = "G"
        valid = is_map_valid(board, size)
    return ["".join(x) for x in board]


class CrashBaracuta(Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 6,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=False,
    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        print(desc.shape)
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        self.initial_desc = desc.copy()

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"A").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.is_slippery = is_slippery
        for row in range(nrow):
            for col in range(ncol):
                s = self.convert_to_state(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GZ":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, * self.update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, * self.update_probability_matrix(row, col, a)))
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.death_img = None
        self.ice_img = None
        self.crash_images = None
        self.goal_img = None
        self.start_img = None

        self.rock_img = None
        self.presente_img = None

    def convert_to_state(self, row, col):
        return row * self.ncol + col

    def boundaryCheck(self, row, col ,a):
        if a == LEFT:
            col = col - 1
        elif a == DOWN:
            row = row + 1
        elif a == RIGHT:
            col = col + 1
        elif a == UP:
            row = row - 1

        return row<0 or row> self.nrow - 1 or col<0 or col> self.ncol - 1

    
    def increment_indices(self,row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return (row, col)


    def update_probability_matrix(self,row, col, action):
        newrow, newcol = self.increment_indices(row, col, action)

        out = self.boundaryCheck(row, col, action)

        newletter = self.desc[newrow, newcol]

        if out:
            reward = -20
        elif bytes(newletter)==b"S":
            reward = +10
        elif bytes(newletter)==b"Z":
            reward = -10
        elif bytes(newletter)==b"G":
            reward=0
        else:
            reward=-0.1
        terminated = bytes(newletter) in b"GZ"

        newstate = self.convert_to_state(row, col) if (out or bytes(newletter)== b"R") else self.convert_to_state(newrow, newcol)
        
        return newstate, reward, terminated

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        
        self.s = s
        self.lastaction = a

        row, col = divmod(s, self.ncol)
        
        if bytes(self.desc[row, col]) == b"S":
            pos = self.convert_to_state(row,col)
            self.desc[row, col] = b"-"
            
            for a in range(4):
                pos_row, pos_col = self.increment_indices(row, col,a)
                pos = self.convert_to_state(pos_row, pos_col)
                invers = [2, 3, 0, 1]
                pos_inverse = invers[a]
                
                if pos!=self.s:
                    self.P[pos][pos_inverse].pop(0)
                    if self.is_slippery:
                        self.P[pos][pos_inverse].append(
                        (1.0 / 3.0, *self.update_probability_matrix(pos_row, pos_col, pos_inverse)))
                    else:
                        self.P[pos][pos_inverse].append(
                        (1.0, *self.update_probability_matrix(pos_row, pos_col, pos_inverse)))


        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        self.desc = self.initial_desc.copy()

        self.nrow, self.ncol = nrow, ncol = self.desc.shape

        nA = 4
        nS = nrow * ncol
        self.nrow, self.ncol = nrow, ncol = self.desc.shape
        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.is_slippery = self.is_slippery
        for row in range(nrow):
            for col in range(ncol):
                s = self.convert_to_state(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = self.desc[row, col]
                    if letter in b"Z":
                        li.append((1.0, s, 0, True))
                    else:
                        if self.is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, * self.update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, * self.update_probability_matrix(row, col, a)))

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        elif self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _resetRender(self):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )


    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("The Last Of Us")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/nitro.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.death_img is None:
            file_name = path.join(path.dirname(__file__), "img/death.png")
            self.death_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/grass.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/door.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.rock_img is None:
            file_name = path.join(path.dirname(__file__), "img/rock.png")
            self.rock_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.presente_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.presente_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.crash_images is None:
            crash = [
                path.join(path.dirname(__file__), "img/crash_left.png"),
                path.join(path.dirname(__file__), "img/crash_down.png"),
                path.join(path.dirname(__file__), "img/crash_right.png"),
                path.join(path.dirname(__file__), "img/crash_up.png"),
            ]
            self.crash_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in crash
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"Z":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"A":
                    self.window_surface.blit(self.start_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.presente_img, pos)
                elif desc[y][x] == b"R":
                    self.window_surface.blit(self.rock_img, pos) 
                    
                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        crash_img = self.crash_images[last_action]

        if desc[bot_row][bot_col] == b"Z":
            self.window_surface.blit(self.death_img, cell_rect)
        else:
            self.window_surface.blit(crash_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
