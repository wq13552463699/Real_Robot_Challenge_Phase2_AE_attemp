#!/usr/bin/env python3
"""Demo on how to run the simulation using the Gym environment

This demo creates a SimCubeTrajectoryEnv environment and runs one episode using
a dummy policy.
"""
from rrc_example_package import cube_trajectory_env
from rrc_example_package.example import PointAtTrajectoryPolicy


def main():
    # env = cube_trajectory_env.SimCubeTrajectoryEnv(
    #     goal_trajectory=None,  # passing None to sample a random trajectory
    #     action_type=cube_trajectory_env.ActionType.TORQUE,
    #     visualization=False,
    #     step_size=40
    # )
    env = cube_trajectory_env.CustomSimCubeEnv(visualisation=False)

    is_done = False
    observation = env.reset()
    t = 0
    
    print('Reset obs: {}'.format(observation))
    # print('Observation Space:')
    # print(env.observation_space)
    # print('\nAction space:')
    # print(env.action_space)
    # print('\nEpisode Length: {}'.format(env.task.EPISODE_LENGTH))
    # print('\nTrajectory: {}'.format(env.goal))
    
    for _ in range(1):
    # while not is_done:
        action = env.action_space.sample()
        # print('\nAction: {}'.format(action))
        observation, reward, is_done, info = env.step(action)
        print('Observation: {}'.format(observation))
        # print('\nInfo: {}'.format(info))
        t = info["time_index"]
        # print('t = {}'.format(t))
        # print('\nTrajectory: {}'.format(info["trajectory"]))
        # print('Reward: {}'.format(reward))
        print('\nState_obs.shape: {}'.format(observation["observation"].shape))
        break


if __name__ == "__main__":
    main()