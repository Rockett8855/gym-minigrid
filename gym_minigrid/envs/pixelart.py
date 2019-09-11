from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class PusherEnv(MiniGridEnv):
    def __init__(self, size=8, agents_start_config)
        self.agents = agents_start_config
        
        super().__init__(grid_size=size, max_steps=4*size*size, see_through_walls=True)

