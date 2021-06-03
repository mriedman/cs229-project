from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment import pyhanabi
from adv_human_mistake import AdvancedHumanMistakeAgent
from copy import deepcopy
from collections import defaultdict
from typing import List
import numpy as np


rng = np.random.default_rng()

class TrainingObserver(Agent):

    def __init__(self, config):
        # Initialize
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = 8 # config.get('information_tokens', 8)
        # self.opp_alpha = config.get('alpha')
        self.mistake_agent = AdvancedHumanMistakeAgent({'alpha': config.get('alpha2')})
        self.last_move = None
        self.sim = 50
        self.ENCODE_LEN = 658
        self.dt = np.dtype([('color', np.int), ('rank', np.int)])
        self.cards = {}
        for i in range(5):
            for j in range(5):
                self.cards[(i,j)] = pyhanabi.HanabiCard(i, j)
        self.data = []
        self.clusters = np.array([rng.random(7) for _ in range(2)])
        self.labels = []

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

    def gen_cards(self, observation: pyhanabi.HanabiObservation):
        if observation == None:
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
            sim_moves.append(self.mistake_agent.act(self.last_move))
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

        cluster = np.argmin(np.linalg.norm(self.clusters - xi, axis=1))

        move = self.mistake_agent.act(observation, cluster == 1)
        return move

    def act2(self, observation: pyhanabi.HanabiObservation):
        """Act based on an observation."""
        if observation.cur_player_offset() != 0:
            self.last_move = observation
            return None

        sim_moves = self.gen_cards(observation)
        last_move = self.get_last_move(observation.last_moves())

        xi = np.array(self.move_features(sim_moves, last_move))

        cluster = np.argmin(np.linalg.norm(self.clusters - xi, axis=1))

        move = self.mistake_agent.act(observation, cluster == 1)
        return move, cluster

    def act3(self, observation: pyhanabi.HanabiObservation, mistake: int):
        if self.last_move is None:
            self.last_move = observation
            return None

        """Act based on an observation."""

        sim_moves = self.gen_cards(observation)
        last_move = self.get_last_move(observation.last_moves())

        features = self.move_features(sim_moves, last_move)

        self.data.append(np.array(features) / len(sim_moves))
        self.labels.append(mistake)

        self.last_move = observation

    def update(self, observation: pyhanabi.HanabiObservation, move: pyhanabi.HanabiMove, flag: int):
        pass



