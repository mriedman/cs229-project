from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment import pyhanabi
from adv_human import AdvancedHumanAgent
from collections import defaultdict
from typing import List
from decision_funcs import *
from train_rl import *
import numpy as np


rng = np.random.default_rng()

class RLDecisionFunctionAgent(Agent):

    def __init__(self, config):
        # Initialize
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = 8 # config.get('information_tokens', 8)
        # self.mistake_agent = AdvancedHumanMistakeAgent({'alpha': config.get('alpha2')})
        self.sim_agent = AdvancedHumanAgent({})
        self.last_move = None
        self.sim = 50
        self.ENCODE_LEN = 658
        self.dt = np.dtype([('color', np.int), ('rank', np.int)])
        self.cards = {}
        df_list = {'kss': k_means_semi_sup, 'em': em_predict}
        print('Decision Func:', config.get('decision_func'))
        self.decision_func = df_list[config.get('decision_func')]
        for i in range(5):
            for j in range(5):
                self.cards[(i,j)] = pyhanabi.HanabiCard(i, j)
        self.data = []
        self.clusters = np.array([rng.random(7) for _ in range(2)])
        self.labels = []
        self.rl_data = initialize_mdp_data(27 * 3 + 26)
        for j in self.rl_data:
            i = config.get('alpha')
            if j in ['transition_counts', 'transition_probs']:
                for k in range(2):
                    self.rl_data[j][:, :, k] = np.loadtxt("./rl_data/%s_%s_0-%02d_%d.txt" %
                                                          (config.get('decision_func'), j, int(i * 100), k))
            elif j != 'num_states':
                self.rl_data[j] = np.loadtxt("./rl_data/%s_%s_0-%02d.txt" %
                                             (config.get('decision_func'), j, int(i * 100)))
        self.last_sa = (0, 0)

    @staticmethod
    def playable_card(card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return card.rank() == fireworks[card.color()]

    def reset(self, config):
        self.config = config
        self.last_move = None
        for i in range(5):
            for j in range(5):
                self.cards[(i,j)] = pyhanabi.HanabiCard(i, j)

        update_mdp_transition_probs_reward(self.rl_data)
        update_mdp_value(self.rl_data, .01, 1)

        write_mdp_files(self.rl_data, self.config.get('alpha'), config.get('decision_func'))

    def gen_cards(self, observation: pyhanabi.HanabiObservation):
        if observation is None:
            return []
        self.last_move.flag = 0
        plausible = []
        for i in range(5):
            p_cols = []
            p_ranks = []
            for j in range(5):
                if self.last_move.card_knowledge()[0][i].color_plausible(j):
                    p_cols.append(j)
                if self.last_move.card_knowledge()[0][i].rank_plausible(j):
                    p_ranks.append(j)
            plausible.append([p_cols, p_ranks])
        rand_cards = np.array([list(zip(rng.choice(plausible[i][0], self.sim),
                                   rng.choice(plausible[i][1], self.sim))) for i in range(5)],
                              dtype=self.dt).T

        sim_moves = []
        for i in rand_cards:
            self.last_move.obs_cards = [[],[self.cards[tuple(j)] for j in i]]
            sim_moves.append(self.sim_agent.act(self.last_move))
        return sim_moves

    def move_features(self, l1: List[pyhanabi.HanabiMove], m2: pyhanabi.HanabiMove):
        ct = 0
        cts = defaultdict(int)
        type_ct = 0
        type_cts = defaultdict(int)
        for i in l1:
            ct += str(i) == str(m2)
            cts[str(i)] += 1
            if i.type() == m2.type():
                type_ct += 1
            type_cts[i.type()] += 1
        return ct, max(cts.values()), ct / (max(cts.values()) / len(l1)),\
               type_ct, ct / ((type_ct + 1) / len(l1)), max(type_cts.values()), type_ct / (max(type_cts.values()) / len(l1))

    def get_last_move(self, last_moves):
        if len(last_moves) == 0:
            return []
        if len(last_moves) == 1:
            if last_moves[0].move().type() >= 5:
                return []
            return last_moves[0].move()
        if last_moves[0].move().type() >= 5:
            return last_moves[1].move()
        return last_moves[0].move()

    def act(self, observation: pyhanabi.HanabiObservation):
        """Act based on an observation."""
        if observation.cur_player_offset() != 0:
            self.last_move = observation
            return None

        sim_moves = self.gen_cards(observation)
        last_move = self.get_last_move(observation.last_moves())

        xi = np.array(self.move_features(sim_moves, last_move))

        df_res = self.decision_func(xi)
        cur_state = df_res + 3 * (sum(observation.fireworks()) // 3) + 27 * observation.life_tokens()

        update_mdp_transition_counts_reward_counts(self.rl_data, self.last_sa[0], self.last_sa[1], cur_state, 0)
        rl_res = choose_action(cur_state, self.rl_data)

        move = self.sim_agent.act(observation, rl_res >= 1)
        return move

    def get_score(self, score):
        cur_state = 55 + score
        update_mdp_transition_counts_reward_counts(self.rl_data, self.last_sa[0], self.last_sa[1], cur_state, score)

    def update(self, observation: pyhanabi.HanabiObservation, move: pyhanabi.HanabiMove, flag: int):
        pass


def write_mdp_files(data, alpha, df):
    # num_states = 81 + 26
    a = data
    for i in [0,.01,.05,.08,.1,.13,.15,.2,.25,.3]:
        if i != alpha:
            continue
        for j in a:
            if j in ['transition_counts', 'transition_probs']:
                for k in range(2):
                    np.savetxt("./rl_data/%s_%s_0-%02d_%d.txt" % (df, j, int(i * 100), k), a[j][:,:,k])
            elif j != 'num_states':
                np.savetxt("./rl_data/%s_%s_0-%02d.txt" % (df, j, int(i * 100)), a[j])

'''a = initialize_mdp_data(81 + 26)
for i in [0,.01,.05,.08,.1,.13,.15,.2,.25,.3]:
    for j in ['em', 'kss']:
        write_mdp_files(a, i, j)'''
