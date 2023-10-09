from utils import *
import random
import matplotlib.pyplot as plt
from matplotlib import rcParams

iter = 10
timesteps = 1000
alphas = [0.0, 0.5, 1.0]
beta = 1.2
eps = 0.2

fplmtd_eta = 0.1
fplmtd_gamma = 0.007
fplmtd_lmax = 1000

discount_factor = 0.9
R = 200.0
methods = ["ATA-FMDP", "FPL-MTD", "EPS-GREEDY", "URS"]
random.seed(10)

def get_att_success_rate(state, typ, exploits):
    if typ == 'unknown':
        return 1.0
    asr = 0
    tech_num = 0
    for tech in state:
        if tech in exploits:
            tech_num += 1
            asr += exploits[tech][0]
    if tech_num == 0:
        return 0
    return asr / tech_num

def get_true_att_success_rate(state, typ, exploits):
    asr = 0
    tech_num = 0
    for tech in state:
        if tech in exploits:
            tech_num += 1
            asr += exploits[tech][0]
    if tech_num == 0:
        return 0
    return asr / tech_num

def get_defender_loss_due_to_attack(state, typ, exploits):
    if typ == 'unknown':
        return 100.0
    loss = 0
    tech_num = 0
    for tech in state:
        if tech in exploits:
            tech_num += 1
            loss += exploits[tech][1]
    if tech_num == 0:
        return 0
    return loss / tech_num

def get_prob_att_type(ts, typ):
    prob = 0
    if typ == 'none':
        prob = 0.0
    if typ == 'unknown':
        prob = 0.15
    if typ == 'DH':
        prob = 0.35
    if typ == 'MH':
        prob = 0.5
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
    db_switch = 0
    postgresql_switch_count = 0

    lang_switch = 0
    python_switch_count = 0

    # Load domain
    state_factors, action_factors, attacker_types, exploits, scs = loader()
    
    print("\nAttacker Types, Tech, (Attack Success Rate, Attack Impact): \n", exploits)
    
    state_vects = get_states_from_state_factors(state_factors)
    states = []
    for st in state_vects:
        states.append(state_to_state_one_hot_encoding(st, state_factors))

    print("\nStates: \n", states)

    action_vects = get_actions_from_action_factors(action_factors)
    actions = []
    for act in action_vects:
        actions.append(action_to_action_one_hot_encoding(act, action_factors))

    print("\nActions: \n", actions)

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
        cumulative_rewards_mdp = []
        cumulative_rewards_urs = []
        cumulative_rewards_fplmtd = []
        cumulative_rewards_eps_greedy = []

        sum_reward_timesteps_mdp = [0]*timesteps
        sum_reward_timesteps_urs = [0]*timesteps
        sum_reward_timesteps_fplmtd = [0]*timesteps
        sum_reward_timesteps_eps_greedy = [0]*timesteps

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

            prob_att_type = {}
            true_prob_att_type = {}
            prob_att_success = {}
            true_prob_att_success = {}
            attack_loss = {}

            num_success = {}

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
                        prob_att_success[state][action][typ] = get_att_success_rate(state_one_hot_encoding_to_state(action, state_factors), typ, exploits[typ])
                        true_prob_att_success[state][action][typ] = get_true_att_success_rate(state_one_hot_encoding_to_state(action, state_factors), typ, exploits[typ])
                        attack_loss[state][action][typ] = get_defender_loss_due_to_attack(state_one_hot_encoding_to_state(action, state_factors), typ, exploits[typ])
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
                eps_greedy_action_counts[action] = 0

            eps_greedy_state_weights = {}
            for state in states:
                eps_greedy_state_weights[state] = 0

            reward_estimates_fpl = {}
            for state in states:
                reward_estimates_fpl[state] = 0.0

            for num in range(timesteps):
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
                        coefficients[i].append(R-defender_loss-alpha*switching_costs[(state, action)])
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
                        exp_return = R-defender_loss-alpha*switching_costs[(state, action)]+discount_factor*values[action]
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

                random_number = random.uniform(0, 1)
                
                # Fetch MDP def strategy
                current_mdp_action = current_policy[current_mdp_state]
                mdp_action_counts[current_mdp_action]+=1
                attack_success = 0
                if random_number <= true_prob_att_success[current_mdp_state][current_mdp_action][current_mdp_att_type]:
                    attack_success = 1
                    num_success[current_mdp_state][current_mdp_action][current_mdp_att_type] += 1.0
                reward_mdp.append(R-(attack_success*attack_loss[current_mdp_state][current_mdp_action][current_mdp_att_type])-(alpha*switching_costs[(current_mdp_state, current_mdp_action)]))
                if current_mdp_state[2] != current_mdp_action[2]:
                    db_switch +=1
                    if current_mdp_state[2] == 0:
                        postgresql_switch_count += 1
                if current_mdp_state[0] != current_mdp_action[0]:
                    lang_switch +=1
                    if current_mdp_state[0] == 0:
                        python_switch_count += 1
                current_mdp_state = current_mdp_action
                current_mdp_state = current_mdp_action

                # Fetch URS def strategy
                attack_success = 0
                current_urs_action = random.choice(actions)
                urs_action_counts[current_urs_action]+=1
                if random_number <= true_prob_att_success[current_urs_state][current_urs_action][current_urs_att_type]:
                    attack_success = 1
                reward_urs.append(R-(attack_success*attack_loss[current_urs_state][current_urs_action][current_urs_att_type])-(alpha*switching_costs[(current_urs_state, current_urs_action)]))
                current_urs_state = current_urs_action
                
                # Fetch FPL-MTD def strategy
                current_fplmtd_action = get_fplmtd_strategy(num, actions, states, reward_estimates_fpl, current_fplmtd_state, switching_costs)
                attack_success = 0
                fplmtd_action_counts[current_fplmtd_action]+=1
                if random_number <= true_prob_att_success[current_fplmtd_state][current_fplmtd_action][current_fplmtd_att_type]:
                    attack_success = 1
                util = R-(attack_success*attack_loss[current_fplmtd_state][current_fplmtd_action][current_fplmtd_att_type])-(alpha*switching_costs[(current_fplmtd_state, current_fplmtd_action)])
                reward_fplmtd.append(util)
                reward_estimates_fpl = fplmtd_gr(num, reward_estimates_fpl, actions, states, current_fplmtd_state, current_fplmtd_action, util, switching_costs)
                current_fplmtd_state = current_fplmtd_action

                # Fetch eps_greedy def strategy
                attack_success = 0
                greedy_action = states[0]
                max_reward_received = 0
                for state, reward_received in eps_greedy_state_weights.items():
                    if max_reward_received < reward_received:
                        greedy_action = state
                        max_reward_received = reward_received
                current_eps_greedy_action = greedy_action
                random_number = random.uniform(0, 1)
                if random_number <= eps:
                    current_eps_greedy_action = random.choice(actions)
                eps_greedy_action_counts[current_eps_greedy_action]+=1
                if random_number <= true_prob_att_success[current_eps_greedy_state][current_eps_greedy_action][current_eps_greedy_att_type]:
                    attack_success = 1
                eps_greedy_state_weights[current_eps_greedy_action]+=(R-(attack_success*attack_loss[current_eps_greedy_state][current_eps_greedy_action][current_eps_greedy_att_type]))-(alpha*switching_costs[(current_eps_greedy_state, current_eps_greedy_action)])
                reward_eps_greedy.append(R-(attack_success*attack_loss[current_eps_greedy_state][current_eps_greedy_action][current_eps_greedy_att_type])-(alpha*switching_costs[(current_eps_greedy_state, current_eps_greedy_action)]))
                current_eps_greedy_state = current_eps_greedy_action

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
            cumulative_rewards_mdp.append(np.mean(reward_mdp))
            cumulative_rewards_urs.append(np.mean(reward_urs))
            cumulative_rewards_fplmtd.append(np.mean(reward_fplmtd))
            cumulative_rewards_eps_greedy.append(np.mean(reward_eps_greedy))

        rcParams['font.weight'] = 'bold'
        rcParams['font.size'] = 14

        plt.close('all')
        plt.plot(range(timesteps), [val / iter for val in sum_reward_timesteps_mdp], label="Adaptive-Threat-Aware-FMDP", color="r")
        plt.plot(range(timesteps), [val / iter for val in sum_reward_timesteps_fplmtd], label="FPL-MTD", color="orange")
        plt.plot(range(timesteps), [val / iter for val in sum_reward_timesteps_eps_greedy], label="EPS-GREEDY", color="forestgreen")
        plt.plot(range(timesteps), [val / iter for val in sum_reward_timesteps_urs], label="URS", color="cornflowerblue")
        plt.legend()
        plt.ylabel("Average Defender Reward")
        plt.xlabel("Timestep")
        plt.yticks(np.arange(0, 210, 10))
        plt.savefig("./graphs/non-evolving/ts_stochastic_non_evolve_alpha_" + str(alpha) + ".png")
        
        print()
        print("Alpha: ", alpha)
        print("Adaptive-Threat-Aware-FMDP: ", np.mean(cumulative_rewards_mdp), np.std(cumulative_rewards_mdp))
        print("FPL-MTD: ", np.mean(cumulative_rewards_fplmtd), np.std(cumulative_rewards_fplmtd))
        print("EPS-GREEDY: ", np.mean(cumulative_rewards_eps_greedy), np.std(cumulative_rewards_eps_greedy))
        print("URS: ", np.mean(cumulative_rewards_urs), np.std(cumulative_rewards_urs))

        mean_values = [np.mean(cumulative_rewards_mdp),
                    np.mean(cumulative_rewards_fplmtd),
                    np.mean(cumulative_rewards_eps_greedy),
                    np.mean(cumulative_rewards_urs)]
        std_values = [np.std(cumulative_rewards_mdp),
                    np.std(cumulative_rewards_fplmtd),
                    np.std(cumulative_rewards_eps_greedy),
                    np.std(cumulative_rewards_urs)]

        plt.close('all')
        fig, ax = plt.subplots()
        bars = ax.bar(methods, mean_values, yerr=std_values, capsize=5)
        bars[0].set_color('red')
        ax.set_ylabel('Average Defender Reward')
        ax.set_xlabel('Method')
        plt.yticks(np.arange(0, 210, 10))
        plt.tight_layout()
        plt.savefig("./graphs/non-evolving/stochastic_non_evolve_alpha_" + str(alpha) + ".png")

    print(db_switch, postgresql_switch_count, python_switch_count)