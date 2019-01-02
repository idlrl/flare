#!/usr/bin/python


import sys
import matplotlib.pyplot as plt


if __name__ == "__main__":
    log_file = sys.argv[1]
    with open(log_file, "r") as f:
        lines = f.read().splitlines()

    rewards = [float(l.split()[-1][:-1]) for l in lines if 'total_reward' in l]

    plt.plot(rewards)
    plt.ylabel("Game reward")
    plt.xlabel("Number of total games (x100)")

    plt.savefig('/tmp/test.png')
