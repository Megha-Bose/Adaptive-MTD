from utils import *
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns

iter = 10
timesteps = 1000
turning_point1 = 330
turning_point2 = 660
alphas = [0.0, 0.5, 1.0]
beta = 1
eps = 0.2

fplmtd_eta = 0.1
fplmtd_gamma = 0.007
fplmtd_lmax = 1000

discount_factor = 0.9
M = 200.0
methods = ["ATA-FMDP", "FPL-MTD", "EPS-GREEDY", "URS"]
random.seed(10)

def get_att_success_rate(state, typ, local_asr, remote_asr):
    if typ == 'unknown':
        return 1.0
    asr = 0
    source_node = typ[0]
    target_node = typ[1]
    if state[int(target_node)] == '1':
        if source_node == target_node:
            asr = local_asr
        else:
            asr = remote_asr
    return asr

def get_true_att_success_rate(state, typ, local_asr, remote_asr):
    asr = 0
    source_node = typ[0]
    target_node = typ[1]
    if typ == 'unknown':
        target_node = '0'
    if state[int(target_node)] == '1':
        if typ == 'unknown':
            asr = 1.0
        elif source_node == target_node:
            asr = local_asr
        else:
            asr = remote_asr
    return asr

def get_defender_loss_due_to_attack(state, typ, sys_loss):
    loss = 0
    source_node = typ[0]
    target_node = typ[1]
    if typ == 'unknown':
        target_node = '0'
    if state[int(target_node)] == '1':
        if typ == 'unknown':
            loss = 100
        else:
            loss = sys_loss
    return loss

def get_prob_att_type(ts, typ):
    prob = 0
    if ts == turning_point1:
        if typ == 'unknown':
            prob = 1.0
        if typ == '00':
            prob = 0.0
        if typ == '01':
            prob = 0.0
        if typ == '10':
            prob = 0.0
        if typ == '11':
            prob = 0.0
        return prob
    if typ == 'unknown':
        prob = 0.0
    if typ == '00':
        prob = 0.2
    if typ == '01':
        prob = 0.3
    if typ == '10':
        prob = 0.3
    if typ == '11':
        prob = 0.2
    return prob

def get_fplmtd_strategy(ts, actions, states, reward_estimates, current_fplmtd_state, switching_costs):
    fplmtd_reward_estimates = reward_estimates.copy()
    fplmtd_action = random.choice(actions)
    gamma1 = random.random()
    if(gamma1 <= fplmtd_gamma):
        fplmtd_action = random.choice(actions)
    else:
        if(num != 0):
            for state in states:
                fplmtd_reward_estimates[state] = fplmtd_reward_estimates[state]/ts
        for state in states:
            fplmtd_reward_estimates[state] -= np.random.exponential(fplmtd_eta)
        new_estimates = [fplmtd_reward_estimates[state] - switching_costs[(current_fplmtd_state, state)] for state in states]
        best_state = states[0]
        best_estimate = 0
        idx = 0
        for state in states:
            if best_estimate < new_estimates[idx]:
                best_state = state
                best_estimate = new_estimates[idx]
            idx+=1
        fplmtd_action =  best_state
    return fplmtd_action

def fplmtd_gr(ts, reward_estimates, actions, states, current_fplmtd_state, strat, util, switching_costs):
	fplmtd_reward_estimates = reward_estimates.copy()
	l = 1
	while(l < fplmtd_lmax):
		strat2 = get_fplmtd_strategy(ts, actions, states, fplmtd_reward_estimates, current_fplmtd_state, switching_costs)
		if(strat2 == strat):
			break
		l+=1
	fplmtd_reward_estimates[strat] += util*l
	return fplmtd_reward_estimates

if __name__ == "__main__":
    # Load domain
    state_factors, action_factors, scs = loader()
    attacker_types = ['00', '01', '10', '11', 'unknown']  
    state_vects = get_states_from_state_factors(state_factors)
    states = []
    for st in state_vects:
        states.append(state_to_state_one_hot_encoding(st, state_factors))
        print(st)
    print("\nStates: \n", states, state_factors)

    action_vects = get_actions_from_action_factors(action_factors)
    actions = []
    for act in action_vects:
        actions.append(action_to_action_one_hot_encoding(act, action_factors))
    print("\nActions: \n", actions, action_factors)

    switching_costs = {}
    i = 0
    for state1 in states:
        j = 0
        for state2 in states:
            switching_costs[(state1, state2)] = scs[i][j]
            j+=1
        i+=1
    print("\n Switching costs: \n", switching_costs)

    values = {}

    for alpha in alphas:
        num_attack_success = {}
        for method in methods:
            num_attack_success[method] = 0
        num_switches = {}
        for method in methods:
            num_switches[method] = 0

        cumulative_rewards_mdp = []
        cumulative_rewards_urs = []
        cumulative_rewards_fplmtd = []
        cumulative_rewards_eps_greedy = []

        cumulative_rewards_static = {}
        for state in states:
            cumulative_rewards_static[state] = []

        sum_reward_timesteps_mdp = [0]*timesteps
        sum_reward_timesteps_urs = [0]*timesteps
        sum_reward_timesteps_fplmtd = [0]*timesteps
        sum_reward_timesteps_eps_greedy = [0]*timesteps

        sum_reward_timesteps_static = {}
        for state in states:
            sum_reward_timesteps_static[state] = [0]*timesteps

        for it in range(iter):
            start_state = random.choice(states)
            current_mdp_state = start_state
            current_urs_state = start_state
            current_rl_state = start_state
            current_fplmtd_state = start_state
            current_eps_greedy_state = start_state

            reward_mdp = []
            reward_urs = []
            reward_rl = []
            reward_fplmtd = []
            reward_eps_greedy = []

            reward_static = {}
            for state in states:
                reward_static[state] = []

            prob_att_type = {}
            true_prob_att_type = {}
            prob_att_success = {}
            true_prob_att_success = {}
            attack_loss = {}

            num_success = {}

            local_asr = random.uniform(0.5, 0.6)
            remote_asr = random.uniform(0.2, 0.3)
            system_loss = random.uniform(60, 70)

            for state in states:
                prob_att_type[state] = {}
                prob_att_success[state] = {}
                true_prob_att_success[state] = {}
                attack_loss[state] = {}
                num_success[state] = {}
                for action in actions:
                    prob_att_type[state][action] = {}
                    prob_att_success[state][action] = {}
                    true_prob_att_success[state][action] = {}
                    attack_loss[state][action] = {}
                    num_success[state][action] = {}
                    for typ in attacker_types:
                        prob_att_type[state][action][typ] = 0.25
                        true_prob_att_type[typ] = get_prob_att_type(0, typ)
                        prob_att_success[state][action][typ] = get_att_success_rate(state_one_hot_encoding_to_state(action, state_factors), typ, local_asr, remote_asr)
                        true_prob_att_success[state][action][typ] = get_true_att_success_rate(state_one_hot_encoding_to_state(action, state_factors), typ, local_asr, remote_asr)
                        attack_loss[state][action][typ] = get_defender_loss_due_to_attack(state_one_hot_encoding_to_state(action, state_factors), typ, system_loss)
                        num_success[state][action][typ] = 0
            
            mdp_action_counts = {}
            urs_action_counts = {}
            fplmtd_action_counts = {}
            eps_greedy_action_counts = {}
            rl_action_counts = {}

            for action in actions:
                mdp_action_counts[action] = 0
                urs_action_counts[action] = 0
                rl_action_counts[action] = 0
                fplmtd_action_counts[action] = 0
            
            for state in states:
                eps_greedy_action_counts[state] = {}
                for action in actions:
                    eps_greedy_action_counts[state][action] = 1

            eps_greedy_state_weights = {}
            for state in states:
                eps_greedy_state_weights[state] = {}
                for action in actions:
                    eps_greedy_state_weights[state][action] = 0

            reward_estimates_fpl = {}
            for state in states:
                reward_estimates_fpl[state] = 0.0

            for num in range(timesteps):
                if num == turning_point1:
                    for state in states:
                        for action in actions:
                            for typ in attacker_types:
                                true_prob_att_type[typ] = get_prob_att_type(num, typ)
                if num == turning_point2:
                    for state in states:
                        for action in actions:
                            for typ in attacker_types:
                                true_prob_att_type[typ] = get_prob_att_type(num, typ)
                # Update predictor and recalculate policy
                for state in states:
                    for action in actions:
                        new_estimate = {}
                        for att_type in attacker_types:
                            new_estimate[att_type] = 0
                        for att_type in attacker_types:
                            if prob_att_success[state][action][att_type] > 0:
                                new_estimate[att_type] = float(num_success[state][action][att_type]) / (prob_att_success[state][action][att_type])
                        s = 0
                        for att_type in attacker_types:
                            s+=new_estimate[att_type]
                        for att_type in attacker_types:
                            val = 0
                            if s > 0:
                                val = new_estimate[att_type] / s
                            prob_att_type[state][action][att_type] = val

                # Solve LP using current attacker response predictor
                coefficients = []
                i = 0
                for state in states:
                    for action in actions:
                        coefficients.append([])
                        for j in range(0, 4):
                            coefficients[i].append(discount_factor*action[j]-state[j])
                        defender_loss = 0
                        for att_type in attacker_types:
                            defender_loss += prob_att_type[state][action][att_type]*prob_att_success[state][action][att_type]*attack_loss[state][action][att_type]
                        coefficients[i].append(M-defender_loss-alpha*switching_costs[(state, action)])
                        i+=1
                weights = lp_solve(coefficients, len(states)*len(actions), len(states)+1)
                values = {}
                for state in states:
                    value = 0
                    for idx in range(len(weights)):
                        value += weights[idx]*float(state[idx])
                    values[state] = value
                # Policy calculation
                policy = {}
                for state in states:
                    max_expected_return = -1e9
                    max_expected_return_action = actions[0]
                    for action in actions:
                        defender_loss = 0
                        for att_type in attacker_types:
                            defender_loss += prob_att_type[state][action][att_type]*prob_att_success[state][action][att_type]*attack_loss[state][action][att_type]
                        exp_return = M-defender_loss-alpha*switching_costs[(state, action)]+discount_factor*values[action]
                        if exp_return > max_expected_return:
                            max_expected_return = exp_return
                            max_expected_return_action = action
                    policy[state] = max_expected_return_action
                current_policy = policy

                for state in states:
                    for action in actions:
                        for typ in attacker_types:
                            num_success[state][action][att_type] = float(num_success[state][action][att_type]) / beta
            
                types = list(true_prob_att_type.keys())
                type_probabilities = list(true_prob_att_type.values())
                current_att_type = random.choices(types, type_probabilities)[0]
                current_mdp_att_type = current_att_type
                current_urs_att_type = current_att_type
                current_rl_att_type = current_att_type
                current_fplmtd_att_type = current_att_type
                current_eps_greedy_att_type = current_att_type
                current_static_att_type = current_att_type

                random_number = random.uniform(0, 1)
                
                # Fetch MDP def strategy
                current_mdp_action = current_policy[current_mdp_state]
                mdp_action_counts[current_mdp_action]+=1
                attack_success = 0
                if random_number <= true_prob_att_success[current_mdp_state][current_mdp_action][current_mdp_att_type]:
                    attack_success = 1
                    num_success[current_mdp_state][current_mdp_action][current_mdp_att_type] += 1.0
                    num_attack_success["ATA-FMDP"] += 1.0
                reward_mdp.append(M-(attack_success*attack_loss[current_mdp_state][current_mdp_action][current_mdp_att_type])-(alpha*switching_costs[(current_mdp_state, current_mdp_action)]))
                if current_mdp_state != current_mdp_action:
                    num_switches["ATA-FMDP"] += 1.0
                current_mdp_state = current_mdp_action

                # Fetch URS def strategy
                attack_success = 0
                current_urs_action = random.choice(actions)
                urs_action_counts[current_urs_action]+=1
                if random_number <= true_prob_att_success[current_urs_state][current_urs_action][current_urs_att_type]:
                    attack_success = 1
                    num_attack_success["URS"] += 1.0
                reward_urs.append(M-(attack_success*attack_loss[current_urs_state][current_urs_action][current_urs_att_type])-(alpha*switching_costs[(current_urs_state, current_urs_action)]))
                if current_urs_state != current_urs_action:
                    num_switches["URS"] += 1.0
                current_urs_state = current_urs_action
                
                # Fetch FPL-MTD def strategy
                current_fplmtd_action = get_fplmtd_strategy(num, actions, states, reward_estimates_fpl, current_fplmtd_state, switching_costs)
                attack_success = 0
                fplmtd_action_counts[current_fplmtd_action]+=1
                if random_number <= true_prob_att_success[current_fplmtd_state][current_fplmtd_action][current_fplmtd_att_type]:
                    attack_success = 1
                    num_attack_success["FPL-MTD"] += 1.0
                util = M-(attack_success*attack_loss[current_fplmtd_state][current_fplmtd_action][current_fplmtd_att_type])-(alpha*switching_costs[(current_fplmtd_state, current_fplmtd_action)])
                reward_fplmtd.append(util)
                reward_estimates_fpl = fplmtd_gr(num, reward_estimates_fpl, actions, states, current_fplmtd_state, current_fplmtd_action, util, switching_costs)
                if current_fplmtd_state != current_fplmtd_action:
                    num_switches["FPL-MTD"] += 1.0
                current_fplmtd_state = current_fplmtd_action
                
                # Fetch eps_greedy def strategy
                attack_success = 0
                greedy_action = states[0]
                max_reward_received = 0
                for action, reward_received in eps_greedy_state_weights[current_eps_greedy_state].items():
                    if max_reward_received < reward_received / eps_greedy_action_counts[current_eps_greedy_state][action]:
                        greedy_action = action
                        max_reward_received = reward_received / eps_greedy_action_counts[current_eps_greedy_state][action]
                current_eps_greedy_action = greedy_action
                random_number = random.uniform(0, 1)
                if random_number <= eps:
                    current_eps_greedy_action = random.choice(actions)
                eps_greedy_action_counts[current_eps_greedy_state][current_eps_greedy_action]+=1
                if random_number <= true_prob_att_success[current_eps_greedy_state][current_eps_greedy_action][current_eps_greedy_att_type]:
                    attack_success = 1
                    num_attack_success["EPS-GREEDY"] += 1.0
                eps_greedy_state_weights[current_eps_greedy_state][current_eps_greedy_action]+=(M-(attack_success*attack_loss[current_eps_greedy_state][current_eps_greedy_action][current_eps_greedy_att_type]))-(alpha*switching_costs[(current_eps_greedy_state, current_eps_greedy_action)])
                reward_eps_greedy.append(M-(attack_success*attack_loss[current_eps_greedy_state][current_eps_greedy_action][current_eps_greedy_att_type])-(alpha*switching_costs[(current_eps_greedy_state, current_eps_greedy_action)]))
                if current_eps_greedy_state != current_eps_greedy_action:
                    num_switches["EPS-GREEDY"] += 1.0
                current_eps_greedy_state = current_eps_greedy_action

                # Static Defense
                for state in states:
                    attack_success = 0
                    if random_number <= true_prob_att_success[state][state][current_static_att_type]:
                        attack_success = 1
                    reward_static[state].append(M-(attack_success*attack_loss[state][state][current_static_att_type])-(alpha*switching_costs[(state, state)]))

            idx = 0
            for reward in reward_mdp:
                sum_reward_timesteps_mdp[idx] += reward
                idx+=1
            idx = 0
            for reward in reward_urs:
                sum_reward_timesteps_urs[idx] += reward
                idx+=1
            idx = 0
            for reward in reward_fplmtd:
                sum_reward_timesteps_fplmtd[idx] += reward
                idx+=1
            idx = 0
            for reward in reward_eps_greedy:
                sum_reward_timesteps_eps_greedy[idx] += reward
                idx+=1
            for st in states:
                idx = 0
                for reward in reward_static[st]:
                    sum_reward_timesteps_static[st][idx] += reward
                    idx+=1

            cumulative_rewards_mdp.append(np.mean(reward_mdp))
            cumulative_rewards_urs.append(np.mean(reward_urs))
            cumulative_rewards_fplmtd.append(np.mean(reward_fplmtd))
            cumulative_rewards_eps_greedy.append(np.mean(reward_eps_greedy))
            for state in states:
                cumulative_rewards_static[state].append(np.mean(reward_static[state]))


        sns.set(style="whitegrid")
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['font.size'] = 14

        # Line plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(timesteps), y=[val / iter for val in sum_reward_timesteps_mdp], label="ATA-FMDP (ours)", color="r")
        sns.lineplot(x=range(timesteps), y=[val / iter for val in sum_reward_timesteps_fplmtd], label="FPL-MTD", color="orange", alpha=0.8)
        sns.lineplot(x=range(timesteps), y=[val / iter for val in sum_reward_timesteps_eps_greedy], label="EPS-GREEDY", color="forestgreen", alpha=0.5)
        sns.lineplot(x=range(timesteps), y=[val / iter for val in sum_reward_timesteps_urs], label="URS", color="cornflowerblue", alpha=0.5)
        plt.legend(fontsize="x-large")
        plt.yticks(np.arange(0, 210, 20))
        plt.ylabel("Average Defender Reward", fontsize=24)
        plt.xlabel("Timestep", fontsize=24)
        plt.tight_layout()
        plt.savefig("./graphs/ts_stochastic_evolve_alpha_" + str(alpha) + ".png")

        print()
        print("Alpha: ", alpha)
        print("Method: Mean Reward, Reward Std, Mean Num. Attack Success, Mean Num. Switches")
        print("Adaptive-Threat-Aware-FMDP: ", np.mean(cumulative_rewards_mdp), np.std(cumulative_rewards_mdp), num_attack_success["ATA-FMDP"]/iter, num_switches["ATA-FMDP"]/iter)
        print("FPL-MTD: ", np.mean(cumulative_rewards_fplmtd), np.std(cumulative_rewards_fplmtd), num_attack_success["FPL-MTD"]/iter, num_switches["FPL-MTD"]/iter)
        print("EPS-GREEDY: ", np.mean(cumulative_rewards_eps_greedy), np.std(cumulative_rewards_eps_greedy), num_attack_success["EPS-GREEDY"]/iter, num_switches["EPS-GREEDY"]/iter)
        print("URS: ", np.mean(cumulative_rewards_urs), np.std(cumulative_rewards_urs), num_attack_success["URS"]/iter, num_switches["URS"]/iter)
        print()

        k = 0
        max_stat_reward = 0
        min_stat_reward = R=M
        for st in states:
            tmp = np.mean(cumulative_rewards_static[st])
            max_stat_reward = max(tmp, max_stat_reward)
            min_stat_reward = min(tmp, min_stat_reward)
            print("Config: "+str(k+1), tmp, np.std(cumulative_rewards_static[st]))
            k+=1
        
        mean_values = [
            np.mean(cumulative_rewards_mdp),
            np.mean(cumulative_rewards_fplmtd),
            np.mean(cumulative_rewards_eps_greedy),
            np.mean(cumulative_rewards_urs)
        ]
        std_values = [
            np.std(cumulative_rewards_mdp),
            np.std(cumulative_rewards_fplmtd),
            np.std(cumulative_rewards_eps_greedy),
            np.std(cumulative_rewards_urs)
        ]

        # Bar plot
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=methods, y=mean_values, hue=methods, dodge=False, palette=["red", "grey", "grey", "grey"], errorbar=None)
        ax.errorbar(x=np.arange(len(methods)), y=mean_values, yerr=std_values, fmt='none', c='black', capsize=5)
        ax.set_ylabel('Average Defender Reward', fontsize=24)
        ax.set_xlabel('Method', fontsize=24)
        ax.tick_params(axis='x', labelsize=16)
        plt.yticks(np.arange(0, 210, 20))
        
        plt.axhline(y=max_stat_reward, color='black', linestyle='--')
        padding = 5
        plt.text(x=len(methods) - 1, y=max_stat_reward+padding, s="Best Static Config. in Hindsight", color='black', va='bottom', ha='right', fontsize=16)

        plt.axhline(y=min_stat_reward, color='black', linestyle='--')
        padding = 5
        plt.text(x=len(methods) - 1, y=min_stat_reward-padding, s="Worst Static Config. in Hindsight", color='black', va='top', ha='right', fontsize=16)

        plt.tight_layout()
        plt.savefig("./graphs/stochastic_evolve_alpha_" + str(alpha) + ".png")