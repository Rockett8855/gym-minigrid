import numpy as np

from gym_minigrid.minigrid import OBJECT_TO_IDX, Box, Grid
from gym_minigrid.pushergrid import PusherGridEnv
from gym_minigrid.register import register


class PusherEnv(PusherGridEnv):
    def __init__(self, n_agents, n_blocks, size=8, **kwargs):
        self.n_agents = n_agents
        self.n_blocks = n_blocks

        self.block_top = (2, 2)
        self.block_size = (size - 4, size - 4)

        super().__init__(
            n_agents,
            grid_size=size,
            max_steps=10 * size * size,
            **kwargs
        )

    def _place_agent_left(self, idx, color):
        top = (1, 1)
        size = (1, self.height-2)
        self.place_agent(idx, top=top, size=size, direction=0, color=color)

    def _place_agent_top(self, idx, color):
        top = (1, 1)
        size = (self.width-2, 1)
        self.place_agent(idx, top=top, size=size, direction=1, color=color)

    def _place_agent_right(self, idx, color):
        top = (self.width-2, 1)
        size = (1, self.height-2)
        self.place_agent(idx, top=top, size=size, direction=2, color=color)

    def _place_agent_bottom(self, idx, color):
        top = (1, self.height-2)
        size = (self.width-2, 1)
        self.place_agent(idx, top=top, size=size, direction=3, color=color)

    def _init_agent(self, idx, pos, color):
        """
        Puts an agent down, pos = 0(left column), 1(top row), 2(right column), 3(bottom row)
        pointing in the direction towards the center
        """
        if pos == 0:
            self._place_agent_left(idx, color)
        elif pos == 1:
            self._place_agent_top(idx, color)
        elif pos == 2:
            self._place_agent_right(idx, color)
        elif pos == 3:
            self._place_agent_bottom(idx, color)
        else:
            assert False, "unkown position"

    def _done(self):
        """
        Returns true when the goal position is reached.
        """
        # Finish criteria, or bad state criteria
        block_idx = self.grid.encode() == OBJECT_TO_IDX['box']
        # print(block_idx, self.goal_block_idx)
        # print(np.array_equal(block_idx, self.goal_block_idx))
        return np.array_equal(block_idx, self.goal_block_idx)

    def _reward(self, agent_i):
        """
        Returns a reward for a spefic agent
        """
        return 0

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        # place agents on the outskirts
        for i in range(self.n_agents):
            self._init_agent(i, i % 4, self._rand_color())

        # place blocks in the center area
        goal_grid = Grid(width, height)

        for i in range(self.n_blocks):
            self.place_obj(Box(), top=self.block_top, size=self.block_size, grid=goal_grid)
            self.place_obj(Box(), top=self.block_top, size=self.block_size, grid=self.grid)

        self.goal_grid_np = goal_grid.encode()
        self.goal_block_idx = self.goal_grid_np == OBJECT_TO_IDX['box']


class Pusher4Block5Env10x10(PusherEnv):
    def __init__(self, **kwargs):
        super().__init__(n_agents=4, n_blocks=5, size=10, **kwargs)


class Pusher4Block10Env20x20(PusherEnv):
    def __init__(self, **kwargs):
        super().__init__(n_agents=4, n_blocks=10, size=20, **kwargs)


class Pusher4Block40Env20x20(PusherEnv):
    def __init__(self, **kwargs):
        super().__init__(n_agents=4, n_blocks=40, size=20, **kwargs)


class Pusher8Block10Env20x20(PusherEnv):
    def __init__(self, **kwargs):
        super().__init__(n_agents=8, n_blocks=10, size=20, **kwargs)

class Pusher8Block40Env20x20(PusherEnv):
    def __init__(self, **kwargs):
        super().__init__(n_agents=8, n_blocks=40, size=20, **kwargs)

class Pusher8Block80Env20x20(PusherEnv):
    def __init__(self, **kwargs):
        super().__init__(n_agents=8, n_blocks=80, size=20, **kwargs)


register(
    id='MiniGrid-Pusher4-Block5-Env10x10-v0',
    entry_point='gym_minigrid.envs:Pusher4Block5Env10x10'
)

register(
    id='MiniGrid-Pusher4-Block10-Env20x20-v0',
    entry_point='gym_minigrid.envs:Pusher4Block10Env20x20'
)

register(
    id='MiniGrid-Pusher4-Block40-Env20x20-v0',
    entry_point='gym_minigrid.envs:Pusher4Block40Env20x20'
)

register(
    id='MiniGrid-Pusher8-Block10-Env20x20-v0',
    entry_point='gym_minigrid.envs:Pusher8Block10Env20x20'
)

register(
    id='MiniGrid-Pusher8-Block40-Env20x20-v0',
    entry_point='gym_minigrid.envs:Pusher8Block40Env20x20'
)

register(
    id='MiniGrid-Pusher8-Block80-Env20x20-v0',
    entry_point='gym_minigrid.envs:Pusher8Block80Env20x20'
)
