import math
import numpy as np

from .agent import PusherAgent, PusherActions
from .minigrid import Grid, Box


class PusherGame(object):
    def __init__(self, n_blocks, n_agents, width, height, rand_int_fn):
        self.n_blocks = n_blocks
        self.n_agents = n_agents
        self.grid = Grid(width, height)
        self.width = width
        self.height = height
        self.rand_int_fn = rand_int_fn

        self.actions = PusherActions

        self.grid.wall_rect(0, 0, width, height)

        self.block_top = (2, 2)
        self.block_size = (width - 4, height - 4)

        self.agents = [None] * n_agents
        self.unstep_buffer = []

    def random_game(self):
        # place agents on the outskirts
        for i in range(self.n_agents):
            self.init_agent(i, i % 4)

        for i in range(self.n_blocks):
            self.place_obj(Box(), top=self.block_top, size=self.block_size)

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        def has_agent(pos):
            for a in self.agents:
                if a is None:
                    continue
                if np.array_equal(pos, a.position):
                    return True
            return False

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise Exception('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self.rand_int_fn(top[0], min(top[0] + size[0], self.grid.width)),
                self.rand_int_fn(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if has_agent(pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def place_agent(
        self,
        agent_idx,
        top=None,
        size=None,
        direction=None,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """
        idx = agent_idx
        assert idx >= 0 and idx < len(self.agents)

        self.agents[idx] = None
        pos = self.place_obj(None, top, size, max_tries=max_tries)

        if direction is None:
            d = self._rand_int(0, 4)
        else:
            d = direction

        self.agents[idx] = PusherAgent(pos, d)

        return pos

    def _place_agent_left_column(self, idx):
        top = (1, 1)
        size = (1, self.height - 2)
        self.place_agent(idx, top=top, size=size, direction=0)

    def _place_agent_top_row(self, idx):
        top = (1, 1)
        size = (self.width - 2, 1)
        self.place_agent(idx, top=top, size=size, direction=1)

    def _place_agent_right_column(self, idx):
        top = (self.width - 2, 1)
        size = (1, self.height - 2)
        self.place_agent(idx, top=top, size=size, direction=2)

    def _place_agent_bottom_row(self, idx):
        top = (1, self.height - 2)
        size = (self.width - 2, 1)
        self.place_agent(idx, top=top, size=size, direction=3)

    def init_agent(self, idx, pos):
        """
        Puts an agent down, pos = 0(left column), 1(top row), 2(right column), 3(bottom row)
        pointing in the direction towards the center
        """
        if pos == 0:
            self._place_agent_left_column(idx)
        elif pos == 1:
            self._place_agent_top_row(idx)
        elif pos == 2:
            self._place_agent_right_column(idx)
        elif pos == 3:
            self._place_agent_bottom_row(idx)
        else:
            assert False, "unkown position"

    def agent_overlaps(self, pos):
        for a in self.agents:
            if np.array_equal(a.position, pos):
                return True
        return False

    def move_block(self, from_pos, to_pos, cell, unstep):
        self.grid.set(to_pos[0], to_pos[1], cell)
        self.grid.set(from_pos[0], from_pos[1], None)
        unstep['blocks'].append((from_pos, to_pos, cell))

    def try_push_block(self, block_pos, block_cell, direction):
        next_pos = block_pos + direction
        next_cell = self.grid.get(*next_pos)

        if self.agent_overlaps(next_pos):
            return False

        if type(next_cell) is Box:
            if self.try_push_block(next_pos, next_cell, direction):
                self.move_block(block_pos, next_pos, block_cell)
                return True
        elif next_cell is None or next_cell.can_overlap():
            self.move_block(block_pos, next_pos, block_cell)
            return True
        else:
            return False

    def try_move_and_push_fwd(self, agent, pos, unstep):
        cell = self.grid.get(*pos)

        if type(cell) is Box:
            # try and push the block one unit in the "forward" direction
            if not self.try_push_block(pos, cell, agent.dir_vec):
                # Couldn't move the blocks
                return False
            # The blocks get removed in reverse order, we want to undo in the
            # opposite order of movement.
            unstep["blocks"].reverse()
        return self.try_move_to(pos)

    def try_move_to(self, agent_idx, agent, pos, unstep):
        cell = self.grid.get(*pos)

        # If they overlap, don't move the agent
        if self.agent_overlaps(pos):
            return False

        if cell is None or cell.can_overlap():
            unstep['agent'] = (agent_idx, agent.position)
            agent.position = pos

    def step_agent(self, agent_idx, action):
        agent = self.agents[agent_idx]
        unstep = {'blocks': [], 'agent': None}

        # Move forward
        if action == self.actions.forward:
            fwd_pos = agent.front_pos
            self.try_move_and_push_fwd(agent, fwd_pos, unstep)
        # Move backward
        elif action == self.actions.backward:
            back_pos = agent.back_pos
            self.try_move_to(agent_idx, agent, back_pos, unstep)
        # Move left
        elif action == self.actions.left:
            left_pos = agent.left_pos
            self.try_move_to(agent_idx, agent, left_pos, unstep)
        # Move right
        elif action == self.actions.right:
            right_pos = agent.right_pos
            self.try_move_to(agent_idx, agent, right_pos, unstep)
        # None action (not used by default)
        elif action == self.actions.none:
            pass
        else:
            assert False, "unknown action {action}".format(action=action)

        self.unstep_buffer.append(unstep)

    def unstep(self):
        """
        For graph searching, it is useful to run a step, and then undo it
        for board analysis reasons.
        """
        if len(self.unstep_buffer) == 0:
            return

        unstep = self.unstep_buffer.pop()

        agent_u = unstep['agent']
        blocks_u = unstep['blocks']

        if agent_u:
            idx, from_pos = agent_u
            self.agents[idx].position = from_pos

        for b_u in blocks_u:
            from_pos, to_pos, cell = b_u
            self.grid.set(from_pos[0], from_pos[1], cell)
            self.grid.set(to_pos[0], to_pos[1], None)
