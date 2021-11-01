#!/usr/bin/env python3
"""Simple demo on how to use the TriFingerPlatform interface."""
import time

import trifinger_simulation


if __name__ == "__main__":
    platform = trifinger_simulation.TriFingerPlatform(visualization=True)

    # Move the fingers to random positions
    while True:
        position = platform.spaces.robot_position.gym.sample()
        finger_action = platform.Action(position=position)

        # apply action for a few steps, so the fingers can move to the target
        # position and stay there for a while
        for _ in range(100):
            t = platform.append_desired_action(finger_action)
            # sleep after each step so that the visualization happens in real
            # time
            time.sleep(platform.get_time_step())

        # show the latest observations
        robot_observation = platform.get_robot_observation(t)
        print("Finger0 Joint Positions: %s" % robot_observation.position[:3])

        # the cube pose is part of the camera observation
        camera_observation = platform.get_camera_observation(t)
        cube_pose = camera_observation.object_pose
        print("Cube Position (x, y, z): %s" % cube_pose.position)
