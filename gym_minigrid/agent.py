import numpy as np

from enum import IntEnum

from .minigrid import COLORS, DIR_TO_VEC


class PusherActions(IntEnum):
    left = 0
    right = 1
    forward = 2
    backward = 3

    none = 4


class PusherAgent(object):
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction
        self.color = COLORS['teal']

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

    def get_view_coords(self, i, j, agent_view_size):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.position
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = agent_view_size
        hs = agent_view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def relative_coords(self, x, y, agent_view_size):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= agent_view_size or vy >= agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y, agent_view_size):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y, agent_view_size) is not None

    def get_view_exts(self, agent_view_size):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.direction == 0:
            topX = self.position[0]
            topY = self.position[1] - agent_view_size // 2
        # Facing down
        elif self.direction == 1:
            topX = self.position[0] - agent_view_size // 2
            topY = self.position[1]
        # Facing left
        elif self.direction == 2:
            topX = self.position[0] - agent_view_size + 1
            topY = self.position[1] - agent_view_size // 2
        # Facing up
        elif self.direction == 3:
            topX = self.position[0] - agent_view_size // 2
            topY = self.position[1] - agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return (topX, topY, botX, botY)
