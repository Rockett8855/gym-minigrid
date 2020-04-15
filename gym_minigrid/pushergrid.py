import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from .agent import PusherActions
from .minigrid import COLOR_NAMES, CELL_PIXELS, Grid


class PusherGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second': 10
    }

    def __init__(
        self,
        n_agents,
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
        self.actions = PusherActions

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

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    @property
    def grid(self):
        return self.game.grid

    def make_game(self, n_agents, n_blocks):
        raise NotImplementedError

    def reset(self):
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self.make_game(self.n_agents, self.n_blocks, self.width, self.height)

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = []
        for a in self.game.agents:
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
            0: '<',
            1: '^',
            2: '>',
            3: 'V'
        }

        str = ''

        def agent(i, j):
            for a in self.game.agents:
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

    def step(self, action):
        self.step_count += 1

        done = False

        for i, action in enumerate(action):
            self.game.step_agent(i, action)

        positions = [a.position for a in self.game.agents]
        print(f"agent positions {positions}")

        obs = []
        reward = []
        for a in self.game.agents:
            obs.append(self.gen_obs(a))
            reward.append(self._reward())

        if self._done() and self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, {}

    def gen_obs_grid(self, agent):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = agent.get_view_exts(self.agent_view_size)

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
        for a in self.game.agents:
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
