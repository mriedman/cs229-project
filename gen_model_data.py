# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from baseline_agent import BaselineAgent
from adv_human import AdvancedHumanAgent
from adv_human_mistake import AdvancedHumanMistakeAgent
# from adv_human_mistake2 import AdvancedHumanMistake2Agent
from training_observer import TrainingObserver
# from mdk2 import MDK2
import sys
import argparse
from random import seed
import numpy as np
import json


from hanabi_learning_environment import pyhanabi


class ModObservation(pyhanabi.HanabiObservation):
    def __init__(self):
        self.obs = None
        self.obs_cards = None
        moves = []
        for card_index in range(5):
            moves.append(pyhanabi.HanabiMove.get_play_move(card_index))
            moves.append(pyhanabi.HanabiMove.get_discard_move(card_index))
            moves.append(pyhanabi.HanabiMove.get_reveal_color_move(1, card_index))
            moves.append(pyhanabi.HanabiMove.get_reveal_rank_move(1, card_index))
        self.moves = moves
        self._observation = None
        self.flag = 1

    def observation(self):
        """Returns the C++ HanabiObservation object."""
        return self._observation

    def cur_player_offset(self):
        """Returns the player index of the acting player, relative to observer."""
        return self.flag

    def num_players(self):
        """Returns the number of players in the game."""
        return self.obs.num_players()

    def observed_hands(self):
        """Returns a list of all hands, with cards ordered oldest to newest.

         The observing player's cards are always invalid.
        """
        return self.obs_cards

    def card_knowledge(self):
        """Returns a per-player list of hinted card knowledge.

        Each player's entry is a per-card list of HanabiCardKnowledge objects.
        Each HanabiCardKnowledge for a card gives the knowledge about the cards
        accumulated over all past reveal actions.
        """
        return self.obs.card_knowledge()

    def discard_pile(self):
        """Returns a list of all discarded cards, in order they were discarded."""
        return self.obs.discard_pile()

    def fireworks(self):
        """Returns a list of fireworks levels by value, ordered by color."""
        return self.obs.fireworks()

    def deck_size(self):
        """Returns number of cards left in the deck."""
        return self.obs.deck_size()

    def last_moves(self):
        """Returns moves made since observing player last acted.

        Each entry in list is a HanabiHistoryItem, ordered from most recent
        move to oldest.  Oldest move is the last action made by observing
        player. Skips initial chance moves to deal hands.
        """
        return self.obs.last_moves()

    def information_tokens(self):
        """Returns the number of information tokens remaining."""
        return self.obs.information_tokens()

    def life_tokens(self):
        """Returns the number of information tokens remaining."""
        return self.obs.life_tokens()

    def legal_moves(self):
        """Returns list of legal moves for observing player.

        List is empty if cur_player() != 0 (observer is not currently acting).
        """
        return self.moves


    def card_playable_on_fireworks(self, color, rank):
        return self.obs.card_playable_on_fireworks(color, rank)

def run_game(game_parameters, agents, trainmodel=None, verbose=1, train=False):
    """Play a game, selecting random actions."""

    def print_state(state):
        """Print some basic information about the state."""
        print("")
        print("Current player: {}".format(state.cur_player()))
        print(state)

        # Example of more queries to provide more about this state. For
        # example, bots could use these methods to to get information
        # about the state in order to act accordingly.
        print("### Information about the state retrieved separately ###")
        print("### Information tokens: {}".format(state.information_tokens()))
        print("### Life tokens: {}".format(state.life_tokens()))
        print("### Fireworks: {}".format(state.fireworks()))
        print("### Deck size: {}".format(state.deck_size()))
        print("### Discard pile: {}".format(str(state.discard_pile())))
        print("### Player hands: {}".format(str(state.player_hands())))
        print("")

    def print_observation(observation):
        """Print some basic information about an agent observation."""
        print("--- Observation ---")
        print(observation)

        print("### Information about the observation retrieved separately ###")
        print("### Current player, relative to self: {}".format(
            observation.cur_player_offset()))
        print("### Observed hands: {}".format(observation.observed_hands()))
        print("### Card knowledge: {}".format(observation.card_knowledge()))
        print("### Discard pile: {}".format(observation.discard_pile()))
        print("### Fireworks: {}".format(observation.fireworks()))
        print("### Deck size: {}".format(observation.deck_size()))
        move_string = "### Last moves:"
        for move_tuple in observation.last_moves():
            move_string += " {}".format(move_tuple)
        print(move_string)
        print("### Information tokens: {}".format(observation.information_tokens()))
        print("### Life tokens: {}".format(observation.life_tokens()))
        print("### Legal moves: {}".format(observation.legal_moves()))
        print("--- EndObservation ---")

    def print_encoded_observations(encoder, state, num_players):
        print("--- EncodedObservations ---")
        print("Observation encoding shape: {}".format(encoder.shape()))
        print("Current actual player: {}".format(state.cur_player()))
        for i in range(num_players):
            print("Encoded observation for player {}: {}".format(
                i, encoder.encode(state.observation(i))))
        print("--- EndEncodedObservations ---")

    game = pyhanabi.HanabiGame(game_parameters)

    encoder = pyhanabi.ObservationEncoder(game)
    mod_obs = [ModObservation() for _ in range(len(agents) + train)]

    if verbose > 3:
      print(game.parameter_string(), end="")
    obs_encoder = pyhanabi.ObservationEncoder(
        game, enc_type=pyhanabi.ObservationEncoderType.CANONICAL)

    state = game.new_initial_state()
    last_mistake, move, mistake = None, None, None
    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            continue

        observation = state.observation(state.cur_player())

        for idx, agt in enumerate(agents):
            if idx == state.cur_player():
                last_mistake = mistake
                move, mistake = agt.act2(observation)
                # print('Move:', move, mistake)
            else:
                mod_obs[idx].obs = observation
                mod_obs[idx].obs_cards = None
                mod_obs[idx].flag = 1
                agt.act(mod_obs[idx])
        if train:
            mod_obs[-1].obs = observation
            mod_obs[-1].obs_cards = None
            mod_obs[-1].flag = 1
            trainmodel.act3(mod_obs[-1], last_mistake)

        state.apply_move(move)

        if verbose > 3:
            print_state(state)
            print_observation(observation)
            if verbose > 4:
                print_encoded_observations(obs_encoder, state, game.num_players())
            print("")
            print("Number of legal moves: {}".format(len(observation.legal_moves())))
        if verbose > 2:
            print("Agent has chosen: {}".format(move))
    if verbose > 2:
        print("")
        print("Game done. Terminal state:")
        print("")
        print(state)
        print("")
    if verbose > 1:
        print("score: {}".format(state.score()))
    return state.score()


reinforcement_agents = []

p = argparse.ArgumentParser(prog='PROG')
p.add_argument('foo')
for i in [('-p', 'players', 2, int), ('-c', 'colors', 5, int), ('-r', 'rank', 5, int), ('-hs', 'hand_size', 5, int),
        ('-i', 'max_information_tokens', 8, int), ('-l', 'max_life_tokens', 3, int), ('-s', 'seed', -1, int),
        ('-v', 'verbose', 0, int), ('-n', 'num_rounds', 1, int), ('-tr', 'training_rounds', 0, int),
        ('-al', 'alpha', 0.05, float), ('-a2', 'alpha2', 0, float), ('-a3', 'alpha3', 0, float), ('-t', 'train', 1, int)]:
    p.add_argument(i[0], dest=i[1], default=i[2], type=i[3])
p.add_argument('-a', dest='agents', default=['baseline', 'baseline'], nargs='*')
args = vars(p.parse_args(sys.argv))
agent_dict = {'baseline': BaselineAgent, 'advhm': AdvancedHumanMistakeAgent, 'advh': AdvancedHumanAgent,
              'mdk': TrainingObserver}

seed(1)
score_list = []
training_to_go = args['training_rounds']
agents=[]
labels = []
args['print'] = 0
for agent in args['agents']:
    agents.append(agent_dict[agent](args))
trainagt = TrainingObserver(args)

for _ in range(args['num_rounds']):
    if _%50 == 0 and args['verbose'] >= 0:
        print('#################'+str(_))
    if _ == args['num_rounds'] - 1:
        args['print'] = 1
    for agent in agents:
        agent.reset(args)
    trainagt.reset(args)
    score_list.append(run_game(args, agents, trainmodel=trainagt, verbose=args['verbose'], train=True))

if args['verbose'] >= 0:
    print(sum(score_list)/args['num_rounds'])
    agents = agents[:-1] + [trainagt]
    score_list=[]
    print('##################################')

    if args['train'] == 1:
        '''trainagt.k_means()
        print('Clusters: ')
        print(trainagt.clusters)
        np.savetxt('./clusters.txt', trainagt.clusters)'''
        data = np.concatenate([trainagt.data, np.array(trainagt.labels).reshape((-1,1))], axis=1)
        np.savetxt('./train_data.txt', data)

        trainagt = TrainingObserver(args)
        for _ in range(args['num_rounds']):
            if _%50 == 0 and args['verbose'] >= 0:
                print('#################'+str(_))
            if _ == args['num_rounds'] - 1:
                args['print'] = 1
            for agent in agents:
                agent.reset(args)
            trainagt.reset(args)
            score_list.append(run_game(args, agents, trainmodel=trainagt, verbose=args['verbose'], train=True))
        data = np.concatenate([trainagt.data, np.array(trainagt.labels).reshape((-1, 1))], axis=1)
        np.savetxt('./val_data.txt', data)

