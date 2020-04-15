import numpy as np

from gym_minigrid.minigrid import OBJECT_TO_IDX, Box, Grid
from gym_minigrid.pushergrid import PusherGridEnv
from gym_minigrid.pushergame import PusherGame
from gym_minigrid.register import register


class PusherEnv(PusherGridEnv):
    def __init__(self, n_agents, n_blocks, size=8, **kwargs):
        self.n_agents = n_agents
        self.n_blocks = n_blocks

        super().__init__(
            n_agents,
            grid_size=size,
            max_steps=10 * size * size,
            **kwargs
        )

    def _done(self):
        """
        Returns true when the goal position is reached.
        """
        # Finish criteria, or bad state criteria
        block_idx = self.grid.encode() == OBJECT_TO_IDX['box']
        # print(block_idx, self.goal_block_idx)
        # print(np.array_equal(block_idx, self.goal_block_idx))
        return np.array_equal(block_idx, self.goal_block_idx)

    def _reward(self):
        """
        Returns a reward for a spefic agent
        """
        return 0

    def make_game(self, n_agents, n_blocks, width, height):
        self.game = PusherGame(n_blocks, n_agents, width, height, self._rand_int)
        self.game.random_game()

        goal = PusherGame(n_blocks, n_agents, width, height, self._rand_int)
        goal.random_game()

        self.goal_grid_np = goal.grid.encode()
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
