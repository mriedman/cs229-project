from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment import pyhanabi


class AdvancedHumanAgent(Agent):

    def __init__(self, config):
        # Initialize
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)

    @staticmethod
    def playable_card(card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return card.rank() == fireworks[card.color()]

    def reset(self, config):
        self.config = config

    def act(self, observation: pyhanabi.HanabiObservation, play = True):
        """Act based on an observation."""
        if observation.cur_player_offset() != 0:
            return None

        # Check if there are any pending hints and play the card corresponding to
        # the hint.
        play_card = -1
        for move in observation.last_moves():
            if move.move().type() < 5:
                if move.move().type() == 3:
                    play_card = move.card_info_revealed()[-1]
                break
        if play_card >= 0 and play: #and observation.life_tokens() > 1:
            move = pyhanabi.HanabiMove.get_play_move(play_card)
            return move
        for card_index, hint in enumerate(observation.card_knowledge()[0]):
            if hint.color() is not None and hint.rank() is not None:
                if observation.card_playable_on_fireworks(hint.color(),hint.rank()):
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
                    if AdvancedHumanAgent.playable_card(card,
                                                 fireworks) and hint.color() is None:
                        if not any(card1.color() == card.color() for card1 in player_hand[idx+1:]):
                            move = pyhanabi.HanabiMove.get_reveal_color_move(player_offset, card.color())
                            for i in observation.legal_moves():
                                if i.type()==move.type() and i.target_offset()==move.target_offset() and i.color()==move.color:
                                    return i
                            return move
                    if AdvancedHumanAgent.playable_card(card,
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

    def act2(self, observation):
        return self.act(observation), 1

#run_game({},[BaselineAgent({}) for _ in range(2)],2)