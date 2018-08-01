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
from flare.framework.computation_task import ComputationTask
from parl.common.logging import GameLogger
import signal
import sys


class Manager(object):
    def __init__(self, ct_settings):
        """
            Initialize `Manager`. `ct_settings` is used to create
            `ComputationTask`; The parameters in `ct_settings` are for each
            `ComputationTask`.
        """
        self.agents = []
        self.cts = {}
        self.CDPs = {}
        for name, setting in ct_settings.iteritems():
            self.cts[name] = ComputationTask(name, **setting)
            self.CDPs[name] = self.cts[name].CDP
        self.logger = GameLogger(1, 100)

    def __signal_handler(self, sig, frame):
        # this is still not good, as we don't get a chance to normally stop
        # the processes.
        print "user signaled ctrl+c"
        for cdp in self.CDPs.values():
            cdp.stop()
        for agent in self.agents:
            agent.running.value = 0
            agent.join()

        self.logger.running.value = False
        self.logger.join()
        sys.exit(0)

    def add_agent(self, agent):
        agent.id = len(self.agents)
        # `Agent` needs to know the state specs to prepare state data
        agent.make_initial_states(
            {k: v.get_state_specs()
             for k, v in self.cts.iteritems()})
        self.agents.append(agent)
        for name, cdp in self.CDPs.iteritems():
            agent.add_agent_helper(
                cdp.create_agent_helper(agent.id),
                [s[0] for s in self.cts[name].get_input_specs()],
                [s[0] for s in self.cts[name].get_action_specs()],
                [s[0] for s in self.cts[name].get_state_specs()],
                [s[0] for s in self.cts[name].get_reward_specs()])
            agent.log_q = self.logger.log_q

    def remove_agent(self):
        self.agents[-1].running.value = 0
        self.agents[-1].join()
        self.agents.pop()

    def start(self):
        signal.signal(signal.SIGINT, self.__signal_handler)
        self.logger.start()
        for cdp in self.CDPs.values():
            cdp.run()
        for agent in self.agents:
            agent.start()

        while self.agents:
            self.agents[-1].join()
            self.agents.pop()
        for cdp in self.CDPs.values():
            cdp.stop()
        self.logger.running.value = False
        self.logger.join()
