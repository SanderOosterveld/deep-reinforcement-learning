from environments import ContinuousUpswingPendulum
from agents import TD3, DDPGAgent
from learners import Learner, EpisodicLearner
from learners.radam import RAdam

import gc
import threading

def run_N_times(env_kwargs={}, agent_kwargs={}, learner_kwargs={}, base_name=None, N_threads=5, agent_class=TD3,
                learner_class=Learner, default_name=None):
    threads = []
    for i in range(N_threads):
        environment = ContinuousUpswingPendulum(**env_kwargs)
        agent = agent_class(environment.nb_sensors, environment.nb_actuators, **agent_kwargs)

        file_name = base_name + "_" + str(i)
        learner = learner_class(environment, agent, file_name, **learner_kwargs)

        if i == 0 and default_name is not None:
            environment.store_defaults(default_name)
            agent.store_defaults(default_name)
            learner.store_defaults(default_name)

        threads.append(threading.Thread(target=learner.run))
        threads[i].start()


    for i in range(N_threads):
        threads[i].join()

    gc.collect()