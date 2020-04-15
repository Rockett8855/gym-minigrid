#!/usr/bin/env python3

from __future__ import division, print_function

import numpy as np
import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Pusher8-Block40-Env20x20-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name, seed=123)

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    global ACTIONS
    global AGENT_IDX

    ACTIONS = [env.actions.none] * env.n_agents
    AGENT_IDX = 0

    def keyDownCb(keyName):
        global ACTIONS
        global AGENT_IDX

        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = env.actions.none

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward
        elif keyName == 'DOWN':
            action = env.actions.backward

        elif keyName == 'RETURN':
            for _ in range(env.n_agents):
                env.game.unstep()
            return
            # action = env.actions.none
        elif keyName == 'SPACE':
            ACTIONS[AGENT_IDX] = env.actions.none
            AGENT_IDX = (AGENT_IDX + 1) % env.n_agents
            action = env.actions.none

        else:
            print("unknown key %s" % keyName)
            return

        ACTIONS[AGENT_IDX] = action

        obs, reward, done, info = env.step(ACTIONS)

        print(f'step={env.step_count}, reward={reward}')

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
