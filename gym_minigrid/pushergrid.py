import math
import gym
from enum import IntEnum
import numpy as np
from gym import spaces
from gym.utils import seeding

from .minigrid import DIR_TO_VEC, COLORS, COLOR_NAMES, CELL_PIXELS, Grid, Box


class Agent(object):
    def __init__(self, position, direction, agent_view_size, color):
        self.position = position
        self.direction = direction
        self.agent_view_size = agent_view_size
        self.color = color

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        assert self.direction >= 0 and self.direction < 4
        return DIR_TO_VEC[self.direction]

    @property
    def back_vec(self):
        """
        Get the vector pointing to the back of the agent.
        """
        dx, dy = self.dir_vec
        return np.array((-dx, -dy))

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """
        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def left_vec(self):
        """
        Get the vector pointing to the left of the agent.
        """
        dx, dy = self.dir_vec
        return np.array((dy, -dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        return self.position + self.dir_vec

    @property
    def back_pos(self):
        """
        Get the position of the cell that is behind the agent
        """
        return self.position + self.back_vec

    @property
    def right_pos(self):
        """
        Get the position of the cell that is to the right of the agent
        """
        return self.position + self.right_vec

    @property
    def left_pos(self):
        """
        Get the position of the cell that is to the left of the agent
        """
        return self.position + self.left_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.position
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.direction == 0:
            topX = self.position[0]
            topY = self.position[1] - self.agent_view_size // 2
        # Facing down
        elif self.direction == 1:
            topX = self.position[0] - self.agent_view_size // 2
            topY = self.position[1]
        # Facing left
        elif self.direction == 2:
            topX = self.position[0] - self.agent_view_size + 1
            topY = self.position[1] - self.agent_view_size // 2
        # Facing up
        elif self.direction == 3:
            topX = self.position[0] - self.agent_view_size // 2
            topY = self.position[1] - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size

        return (topX, topY, botX, botY)


class PusherGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second': 10
    }

    class PusherActions(IntEnum):
        left = 0
        right = 1
        forward = 2
        backward = 3

        none = 4

    def __init__(
        self,
        n_agents,
        actions=PusherActions,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        see_through_walls=False,
        seed=1337,
        agent_view_size=7,
    ):
        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = actions

        # Actions are discrete integer values
        self.action_space = spaces.MultiDiscrete([len(self.actions)] * n_agents)

        # Number of cells (width and height) in the agent view
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Renderer object used to render the whole grid (full-scale)
        self.grid_render = None

        # Renderer used to render observations (small-scale agent view)
        self.obs_render = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agents = [None] * n_agents

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):
        # Current position and direction of the agent
        self.agents = [None] * len(self.agents)

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.position is not None
            assert a.direction is not None

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*a.position)
            assert start_cell is None or start_cell.can_overlap()

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = []
        for a in self.agents:
            obs = self.gen_obs(a)
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall': 'W',
            'floor': 'F',
            'door': 'D',
            'key': 'K',
            'ball': 'A',
            'box': 'B',
            'goal': 'G',
            'lava': 'V',
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        def agent(i, j):
            for a in self.agents:
                if a.position[0] == i and a.position[1] == j:
                    return a
            return None

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                a = agent(i, j)
                if a is not None:
                    str += 2 * AGENT_DIR_TO_STR[a.direction]
                    continue

                c = self.grid.get(i, j)

                if c is None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=math.inf,
                  grid=None
                  ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """
        if grid is None:
            grid = self.grid

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (grid.width, grid.height)

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
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], grid.height))
            ))

            # Don't place the object on top of another object
            if grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if has_agent(pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        grid.set(pos[0], pos[1], obj)

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
        color='red',
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

        self.agents[idx] = Agent(pos, d, self.agent_view_size, COLORS[color])

        return pos

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid = Grid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def _step_agent(self, agent, action):
        def agent_overlaps(pos):
            for a in self.agents:
                if np.array_equal(a.position, pos):
                    return True
            return False

        def move_block(from_pos, to_pos, cell):
            self.grid.set(to_pos[0], to_pos[1], cell)
            self.grid.set(from_pos[0], from_pos[1], None)

        def try_push_block(block_pos, block_cell, direction):
            next_pos = block_pos + direction
            next_cell = self.grid.get(*next_pos)

            if agent_overlaps(next_pos):
                return False

            if type(next_cell) is Box:
                if try_push_block(next_pos, next_cell, direction):
                    move_block(block_pos, next_pos, block_cell)
                    return True
            elif next_cell is None or next_cell.can_overlap():
                move_block(block_pos, next_pos, block_cell)
                return True
            else:
                return False

        def try_move_and_push_fwd(pos):
            cell = self.grid.get(*pos)

            if type(cell) is Box:
                # try and push the block one unit in the "forward" direction
                if not try_push_block(pos, cell, agent.dir_vec):
                    # Couldn't move the blocks
                    return False
            return try_move_to(pos)

        def try_move_to(pos):
            cell = self.grid.get(*pos)

            # If they overlap, don't move the agent
            if agent_overlaps(pos):
                return False

            if cell is None or cell.can_overlap():
                agent.position = pos

        # Move forward
        if action == self.actions.forward:
            fwd_pos = agent.front_pos
            try_move_and_push_fwd(fwd_pos)
        # Move backward
        elif action == self.actions.backward:
            back_pos = agent.back_pos
            try_move_to(back_pos)
        # Move left
        elif action == self.actions.left:
            left_pos = agent.left_pos
            try_move_to(left_pos)
        # Move right
        elif action == self.actions.right:
            right_pos = agent.right_pos
            try_move_to(right_pos)
        # None action (not used by default)
        elif action == self.actions.none:
            pass
        else:
            assert False, f"unknown action {action}"

    def step(self, action):
        self.step_count += 1

        done = False

        for (ag, action) in zip(self.agents, action):
            self._step_agent(ag, action)

        obs = []
        reward = []
        for a in self.agents:
            obs.append(self.gen_obs(a))
            reward.append(self._reward(a))

        if self._done() and self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, {}

    def gen_obs_grid(self, agent):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = agent.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(agent.direction + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        return grid, vis_mask

    def gen_obs(self, agent):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid(agent)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': agent.direction,
        }

        return obs

    def get_obs_render(self, obs, tile_pixels=CELL_PIXELS // 2):
        """
        Render an agent observation for visualization
        """

        if self.obs_render is None:
            from gym_minigrid.rendering import Renderer
            self.obs_render = Renderer(
                self.agent_view_size * tile_pixels,
                self.agent_view_size * tile_pixels
            )

        r = self.obs_render

        r.beginFrame()

        grid = Grid.decode(obs)

        # Render the whole grid
        grid.render(r, tile_pixels)

        # Draw the agent
        ratio = tile_pixels / CELL_PIXELS
        r.push()
        r.scale(ratio, ratio)
        r.translate(
            CELL_PIXELS * (0.5 + self.agent_view_size // 2),
            CELL_PIXELS * (self.agent_view_size - 0.5)
        )
        r.rotate(3 * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 10),
            (12,  0),
            (-12, -10)
        ])
        r.pop()

        r.endFrame()

        return r.getPixmap()

    def render(self, mode='human', close=False, highlight=True):
        """
        Render the whole-grid human view
        """

        if close:
            if self.grid_render:
                self.grid_render.close()
            return

        if self.grid_render is None or self.grid_render.window is None:
            from gym_minigrid.rendering import Renderer
            self.grid_render = Renderer(
                self.width * CELL_PIXELS,
                self.height * CELL_PIXELS,
                True if mode == 'human' else False
            )

        r = self.grid_render

        if r.window:
            r.window.setText("Pusher Environment")

        r.beginFrame()

        # Render the whole grid
        self.grid.render(r, CELL_PIXELS)

        # Draw the agent
        for a in self.agents:
            r.push()
            r.translate(
                CELL_PIXELS * (a.position[0] + 0.5),
                CELL_PIXELS * (a.position[1] + 0.5)
            )
            r.rotate(a.direction * 90)
            r.setLineColor(*a.color)
            r.setColor(*a.color)
            r.drawPolygon([
                (12, -10),
                (-12, 0),
                (12, 10)
            ])
            r.pop()

            # Compute which cells are visible to the agent
            _, vis_mask = self.gen_obs_grid(a)

            # Compute the absolute coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = a.dir_vec
            r_vec = a.right_vec
            top_left = a.position + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

            # For each cell in the visibility mask
            if highlight:
                for vis_j in range(0, self.agent_view_size):
                    for vis_i in range(0, self.agent_view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_mask[vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                        # Highlight the cell
                        r.fillRect(
                            abs_i * CELL_PIXELS,
                            abs_j * CELL_PIXELS,
                            CELL_PIXELS,
                            CELL_PIXELS,
                            255, 255, 255, 50
                        )

        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()

        return r
