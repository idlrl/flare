# FLARE

## Design

FLARE is a reinforcement learning (RL) framework for training [embodied agents](https://en.wikipedia.org/wiki/Embodied_agent) with [PyTorch](https://pytorch.org/). The design philosophy of FLARE is to maximize its flexibility so that a *researcher* can easily apply it to a variety of scenarios or tasks. By "easily", we mean at least four properties:

1. The code is modular so that for a new problem the researcher might only need to change some (hopefully a small number of) modules of the existing code while keeping the rest unchanged. For example, when the user decides to switch from "Q-learning" to "SARSA", he/she probably only has to change the learning objective but not the network configuration (since both algorithms require outputting Q-values).

2. The code is extensible. Usually an embodied agent would have multiple tasks learned in parallel (e.g., language ability and vision ability). And sometimes some other unsupervised learning tasks might be added to facilitate feature learning. So FLARE should be able to easily combine several tasks that require distinct objective functions, or to easily add a new task to existing ones whenever the researcher feels necessary.

3. The support for complex computations due to the broad range of various algorithms that are involved in modeling an embodied agent. It should be convenient in FLARE to have computational branches of neural nets and conditional code behaviors. For this reason, we choose PyTorch because of its dynamic computational graph and that it blends well in the host language.

4. Extensible data I/O between an agent and different environments. In FLARE, the researcher only needs to spend a tiny effort on the change of agent-environment communication when the agent is placed in a new environment that has new sensor inputs or requires new actions.

## Prerequisites
FLARE is currently a pure Python framework which doesn't need build. Several dependent tools:
* [PyTorch](https://pytorch.org/)
* [Glog](https://pypi.org/project/glog/)
* [OpenAI Gym](https://gym.openai.com/) (for running examples)

That's it. Enjoy training agents with FLARE!

## Quick example
*TODO*
