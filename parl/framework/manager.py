#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from multiprocessing import Queue
from threading import Thread
from parl.common.logging import GameLogger
from parl.framework.computation_task import ComputationTask


class Manager(object):
    def __init__(self, ct_settings):
        self.agents = []
        self.cts = {}
        self.state_specs = {}
        self.wrappers = {}
        for name, setting in ct_settings.iteritems():
            self.cts[name] = ComputationTask(name, **setting)
            self.state_specs[name] = self.cts[name].get_state_specs()
            self.wrappers[name] = self.cts[name].wrapper
        self.logger = GameLogger(1, 100)

    def add_agent(self, agent):
        agent.id = len(self.agents)
        # `Agent` needs to know the state specs to prepare state data
        agent.make_initial_states(self.state_specs)
        self.agents.append(agent)
        for name, wrapper in self.wrappers.iteritems():
            agent.add_helper(
                wrapper.create_helper(agent.id), agent.pack_exps,
                agent.unpack_exp_seqs, agent.is_episode_end)
            agent.log_q = self.logger.log_q

    def remove_agent(self):
        self.agents[-1].running.value = 0
        self.agents[-1].join()
        self.agents.pop()

    def start(self):
        self.logger.start()
        for wrapper in self.wrappers.values():
            wrapper.run()
        for agent in self.agents:
            agent.start()

        while self.agents:
            self.agents[-1].join()
            self.agents.pop()
        for wrapper in self.wrappers.values():
            wrapper.stop()
        self.logger.running.value = False
        self.logger.join()
