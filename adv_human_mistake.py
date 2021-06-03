from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment import pyhanabi
import numpy as np


rng = np.random.default_rng()

class AdvancedHumanMistakeAgent(Agent):

    def __init__(self, config, *args, **kwargs):
        # Initialize
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)
        self.alpha = config.get('alpha')

    @staticmethod
    def playable_card(card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return card.rank() == fireworks[card.color()]

    def reset(self, config):
        self.config = config

    def act2(self, observation: pyhanabi.HanabiObservation, play: bool = True):
        """Act based on an observation."""
        if observation.cur_player_offset() != 0:
            return None

        # Make a mistake?

        if rng.random() < self.alpha:
            # print('M!')
            return rng.choice(observation.legal_moves(), 1)[0], 0

        # Check if there are any pending hints and play the card corresponding to
        # the hint.
        play_card = -1
        for move in observation.last_moves():
            if move.move().type() < 5:
                if move.move().type() == 3:
                    play_card = move.card_info_revealed()[-1]
                break
        if play_card >= 0 and observation.life_tokens() > 1 and play:
            move = pyhanabi.HanabiMove.get_play_move(play_card)
            return move, 1
        for card_index, hint in enumerate(observation.card_knowledge()[0]):
            if hint.color() is not None and hint.rank() is not None:
                if observation.card_playable_on_fireworks(hint.color(), hint.rank()):
                    move = pyhanabi.HanabiMove.get_play_move(card_index)
                    return move, 1

        # Check if it's possible to hint a card to your colleagues.
        fireworks = observation.fireworks()
        if observation.information_tokens() > 0:
            # Check if there are any playable cards in the hands of the cooperators.
            for player_offset in range(1, observation.num_players()):
                player_hand = observation.observed_hands()[player_offset]
                player_hints = observation.card_knowledge()[player_offset]
                # Check if the card in the hand of the cooperator is playable.
                for idx, tpl in enumerate(zip(player_hand, player_hints)):
                    card, hint = tpl
                    if AdvancedHumanMistakeAgent.playable_card(card,
                                                 fireworks) and hint.color() is None:
                        if not any(card1.color() == card.color() for card1 in player_hand[idx+1:]):
                            move = pyhanabi.HanabiMove.get_reveal_color_move(player_offset, card.color())
                            for i in observation.legal_moves():
                                if i.type()==move.type() and i.target_offset()==move.target_offset() and i.color()==move.color:
                                    return i, 2
                            return move, 2
                    if AdvancedHumanMistakeAgent.playable_card(card,
                                                 fireworks) and hint.rank() is None:
                        move = pyhanabi.HanabiMove.get_reveal_rank_move(player_offset, card.rank())
                        for i in observation.legal_moves():
                            if i.type()==move.type() and i.target_offset()==move.target_offset() and i.rank()==move.rank():
                                return i, 1
                        return move.to_dict()

        # If no card is hintable then discard or play.
        for i in observation.legal_moves():
            if i.type()==pyhanabi.HanabiMoveType.DISCARD:
                return i, 1
        return observation.legal_moves()[-1], 1

    def act(self, observation: pyhanabi.HanabiObservation, play: bool = True):
        """Act based on an observation."""
        if observation.cur_player_offset() != 0:
            return None

        # Make a mistake?

        if rng.random() < self.alpha:
            # print('M!')
            return rng.choice(observation.legal_moves(), 1)[0]

        # Check if there are any pending hints and play the card corresponding to
        # the hint.
        play_card = -1
        for move in observation.last_moves():
            if move.move().type() < 5:
                if move.move().type() == 3:
                    play_card = move.card_info_revealed()[-1]
                break
        if play_card >= 0: # and observation.life_tokens() > 1 and play:
            move = pyhanabi.HanabiMove.get_play_move(play_card)
            return move
        for card_index, hint in enumerate(observation.card_knowledge()[0]):
            if hint.color() is not None and hint.rank() is not None:
                if observation.card_playable_on_fireworks(hint.color(), hint.rank()):
                    move = pyhanabi.HanabiMove.get_play_move(card_index)
                    return move

        # Check if it's possible to hint a card to your colleagues.
        fireworks = observation.fireworks()
        if observation.information_tokens() > 0:
            # Check if there are any playable cards in the hands of the opponents.
            for player_offset in range(1, observation.num_players()):
                player_hand = observation.observed_hands()[player_offset]
                player_hints = observation.card_knowledge()[player_offset]
                # Check if the card in the hand of the opponent is playable.
                for idx, tpl in enumerate(zip(player_hand, player_hints)):
                    card, hint = tpl
                    if AdvancedHumanMistakeAgent.playable_card(card,
                                                 fireworks) and hint.color() is None:
                        if not any(card1.color() == card.color() for card1 in player_hand[idx+1:]):
                            move = pyhanabi.HanabiMove.get_reveal_color_move(player_offset, card.color())
                            for i in observation.legal_moves():
                                if i.type()==move.type() and i.target_offset()==move.target_offset() and i.color()==move.color:
                                    return i
                            return move
                    if AdvancedHumanMistakeAgent.playable_card(card,
                                                 fireworks) and hint.rank() is None:
                        move = pyhanabi.HanabiMove.get_reveal_rank_move(player_offset, card.rank())
                        for i in observation.legal_moves():
                            if i.type()==move.type() and i.target_offset()==move.target_offset() and i.rank()==move.rank():
                                return i
                        return move.to_dict()

        # If no card is hintable then discard or play.
        for i in observation.legal_moves():
            if i.type()==pyhanabi.HanabiMoveType.DISCARD:
                return i
        return observation.legal_moves()[-1]

#run_game({},[BaselineAgent({}) for _ in range(2)],2)