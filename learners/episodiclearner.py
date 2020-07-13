from .learner import Learner, LinearEpsilon
from environments.environment import _Environment
from agents.agent import AgentWithNetworks

from itertools import count


LEARNER_N_RUNS = 1000
LEARNER_EPSILON = LinearEpsilon(0.9, 0.4, LEARNER_N_RUNS)
LEARNER_SAVE_FREQUENCY = 10
LEARNER_EVAL_FREQUENCY = 5
LEARNER_RANDOM_INIT_DECAY_RATE = 1
LEARNER_REWARD_SCALE = 1
LEARNER_TARGET_UPDATE = None
LEARNER_ITERATIONS = 200

class EpisodicLearner(Learner):

    def __init__(self, environment: _Environment, agent: AgentWithNetworks,
                 file_name=None,
                 n_runs=LEARNER_N_RUNS,
                 epsilon=LEARNER_EPSILON,
                 save_frequency=LEARNER_SAVE_FREQUENCY,
                 eval_frequency=LEARNER_EVAL_FREQUENCY,
                 reward_scale=LEARNER_REWARD_SCALE,
                 target_update=LEARNER_TARGET_UPDATE,
                 iterations=LEARNER_ITERATIONS):

        super(self.__class__, self).__init__(environment, agent, file_name, n_runs, epsilon, save_frequency, eval_frequency, reward_scale, target_update)

        self.iterations = iterations

    def run(self):
        try:
            total_counter = 0

            for epoch in range(self._n_runs):
                total_loss = 0
                total_reward = self.env.get_reward()
                epsilon = self.epsilon()
                if epoch % self._eval_freq == self._eval_freq-1:
                    self.evaluate()
                    self.env.reset()
                else:
                    self.env.random_init(self._random_init_range)

                if epoch % self._save_freq == self._save_freq-1:
                    self.save()

                print("Init state: %s"%self.env.state)
                for _ in count():
                    old_state = self.env.state

                    controls = self.agent.epsilon_greedy_action(old_state, epsilon)

                    self.env.step(controls)

                    if self.env.success:
                        new_state = None
                    else:
                        new_state = self.env.state

                    reward = self.env.get_reward()*self._reward_scale
                    total_reward += reward

                    self.agent.add_to_memory(old_state, controls, new_state, reward)

                    if self.env.done:
                        self.accumulated_reward.append(total_reward)

                        for _ in range(self.iterations):
                            total_counter += 1
                            self.agent.learn(None, None, None, None)
                            loss = self.agent.loss
                            if loss is not None:
                                total_loss += loss

                            if self.agent.soft_update_speed is None and total_counter % self._target_update == 0:
                                self.agent.hard_update()

                        self.accumulated_loss.append(total_loss)

                        break


                print("%s: %.1f%% Total Reward: %f" % (self.file_name.split('/')[-1], (epoch / self._n_runs * 100), total_reward))

        except KeyboardInterrupt:
            self.save()