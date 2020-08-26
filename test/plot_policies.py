from agents import TD3
import torch
import utils
import numpy as np
def main():
    k = 0
    for i in range(2):
        for j in range(5):
            agent = TD3(2,1)
            agent.load("_data/0826-plot-policies/_plot_1/default_" +str(i)+ "_" +str(j))
            agent.store_policy_value("_data/0826-plot-policies/matrices/"+str(k))
            f = open(utils.check_store_name("_data/0826-plot-policies/matrices/"+str(k)+"_evaluated"), 'w')
            eval_final = np.loadtxt("_data/0826-plot-policies/_plot_1/default_" +str(i)+ "_" +str(j)+"_evaluated_reward.csv")[-1]
            f.write(str(eval_final))
            k += 1
    torch.cuda.empty_cache()

    for i in range(2):
        for j in range(5):
            agent = TD3(2,1)
            agent.load("_data/0826-plot-policies/_plot_3/default_" +str(i)+ "_" +str(j))
            agent.store_policy_value("_data/0826-plot-policies/matrices/"+str(k))
            f = open(utils.check_store_name("_data/0826-plot-policies/matrices/"+str(k)+"_evaluated"), 'w')
            eval_final = np.loadtxt("_data/0826-plot-policies/_plot_1/default_" +str(i)+ "_" +str(j)+"_evaluated_reward.csv")[-1]
            f.write(str(eval_final))
            k += 1

    for i in range(2):
        for j in range(5):
            agent = TD3(2,1)
            agent.load("_data/0826-plot-policies/_plot_2/default_" +str(i)+ "_" +str(j))
            agent.store_policy_value("_data/0826-plot-policies/matrices/"+str(k))
            f = open(utils.check_store_name("_data/0826-plot-policies/matrices/"+str(k)+"_evaluated"), 'w')
            eval_final = np.loadtxt("_data/0826-plot-policies/_plot_1/default_" +str(i)+ "_" +str(j)+"_evaluated_reward.csv")[-1]
            f.write(str(eval_final))
            k += 1