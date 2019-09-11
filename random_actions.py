#!/usr/bin/env python3

from __future__ import division, print_function

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
        default='MiniGrid-Pusher8-Block80-Env20x20-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

    renderer.window.setKeyDownCb(keyDownCb)

    def do_step():
        actions = env.action_space.sample()

        obs, reward, done, info = env.step(actions)

        print(f'step={env.step_count}, rewards={reward}')

        if done:
            print('done!')
            resetEnv()

    while True:
        env.render('human')
        time.sleep(0.1)

        do_step()

        # If the window was closed
        if renderer.window is None:
            break


if __name__ == "__main__":
    main()
