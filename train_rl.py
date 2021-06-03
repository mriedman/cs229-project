import numpy as np

def initialize_mdp_data(num_states):
    transition_counts = np.zeros((num_states, num_states, 2))
    transition_probs = np.ones((num_states, num_states, 2)) / num_states
    # Index zero is sum of rewards, index 1 is count of total num state is reached
    reward_counts = np.zeros((num_states, 2))
    reward = np.zeros(num_states)
    value = np.random.rand(num_states) * 0.1

    return {
        'transition_counts': transition_counts,
        'transition_probs': transition_probs,
        'reward_counts': reward_counts,
        'reward': reward,
        'value': value,
        'num_states': num_states,
    }


def choose_action(state, mdp_data):
    def eval_action(a):
        transition_probs = mdp_data['transition_probs'][state, :, a]
        return transition_probs @ mdp_data['value']

    action_vals = np.array(list(map(eval_action, [0, 1])))
    if action_vals[0] > action_vals[1]:
        return 0
    elif action_vals[1] > action_vals[0]:
        return 1
    else:
        return np.random.choice([0, 1])


def update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, reward):
    mdp_data['transition_counts'][state, new_state, action] += 1
    mdp_data['reward_counts'][new_state] += np.array([reward, 1])


def update_mdp_transition_probs_reward(mdp_data):
    transition_counts = mdp_data['transition_counts']
    for a in [0, 1]:
        for state in range(mdp_data['num_states']):
            sa_tc = transition_counts[state, :, a]
            if np.sum(sa_tc) == 0:
                continue
            mdp_data['transition_probs'][state, :, a] = sa_tc / np.sum(sa_tc)

    reward_counts = mdp_data['reward_counts']
    for state in range(mdp_data['num_states']):
        if reward_counts[state, 1] == 0:
            continue
        mdp_data['reward'][state] = reward_counts[state, 0] / reward_counts[state, 1]


def update_mdp_value(mdp_data, tolerance, gamma):
    num_iter = 0
    prev_value = None
    curr_value = mdp_data['value']
    while (prev_value is None or np.linalg.norm(curr_value - prev_value) > tolerance) and num_iter < 10:
        num_iter += 1
        prev_value = curr_value
        action_vals = np.array([mdp_data['transition_probs'][:, :, i] @ prev_value for i in [0, 1]])
        curr_value = mdp_data['reward'] + gamma * np.max(action_vals, axis=0)

    mdp_data['value'] = curr_value
    # print(num_iter == 1)
    return num_iter == 1
