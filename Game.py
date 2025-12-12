from time import sleep
from collections import deque
import random
import torch
import torch.nn as nn
import numpy as np
from sympy.physics.units import action


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank} of {self.suit}" if self.rank != "Joker" else "Joker"

    def __repr__(self):
        return f"{self.rank} of {self.suit}" if self.rank != "Joker" else "Joker"

    def value(self):
        if self.rank == 'A':
            return 1
        elif self.rank == 'J':
            return 11
        elif self.rank == 'Q':
            return 12
        elif self.rank == 'K':
            return 13
        elif self.rank == 'Joker':
            return 20
        else:
            return int(self.rank)


class Deck:
    def __init__(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['A'] + [str(n) for n in range(2, 11)] + ['J', 'Q', 'K']
        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]
        self.cards += [Card(None, 'Joker') for _ in range(2)]
        random.shuffle(self.cards)

    def draw(self):
        return self.cards.pop() if self.cards else None

    def is_empty(self):
        return len(self.cards) == 0


class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.can_play = []

    def draw_card(self, deck, game):
        if deck.is_empty():
            game.replenish_deck()
        if not deck.is_empty():
            card = deck.draw()
            self.hand.append(card)
            return card
        return None

    def play_cards(self, top_card):
        joker_on_top = top_card.rank == 'Joker'

        # First find valid ranks in hand to potentially stack
        potential_plays = {}
        for card in self.hand:
            potential_plays.setdefault(card.rank, []).append(card)

        # print(potential_plays)

        # Try to play multiple cards of the same rank
        for rank, cards in potential_plays.items():
            if len(cards) > 1:
                # Check if any card in this rank can be played on top card
                can_play_rank = False
                for card in cards:
                    if (joker_on_top or
                            card.suit == top_card.suit or
                            card.rank == top_card.rank or
                            card.rank == 'Joker'):
                        can_play_rank = True
                        break

                if can_play_rank:
                    # Play all cards of the same rank
                    for card in cards:
                        self.hand.remove(card)
                    return cards  # List of cards played

        # Otherwise, fall back to single card play
        for card in self.hand:
            if joker_on_top or card.rank == top_card.rank or card.suit == top_card.suit or card.rank == 'Joker':
                self.hand.remove(card)
                return [card]
        return []

    def hand_value(self):
        return sum(card.value() for card in self.hand)

    def has_cards(self):
        return len(self.hand) > 0

class SimpleRLPlayer(Player):
    def __init__(self, name, epsilon=0.1, alpha=0.1, gamma=0.9):
        super().__init__(name)
        self.q_table = {}  # Dictionary: state -> {action: q_value}
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.last_state = None
        self.last_action = None

    def can_play_card(self, card, top_card):
        """Check if a single card can be played"""
        if not top_card:
            return True  # Any card can be played if there is no top card

        if card.rank == 'Joker':
            return True  # Joker can be played on any card

        if top_card.rank == 'Joker':
            return True  # Any card can be played on a Joker

        return (
            card.suit == top_card.suit or
            card.rank == top_card.rank or
            card.rank == 'Joker'
        )

    def get_playable_cards(self, top_card):
        """Get all cards that can be played individually"""
        playable = []
        for card in self.hand:
            if self.can_play_card(card, top_card):
                playable.append(card)

        return playable

    def get_valid_actions(self, top_card):
        """Get all possible actions (sets of cards to play)"""
        actions = []

        # 1. check for multiple cards of the same rank
        rank_groups = {}
        for card in self.hand:
            rank_groups.setdefault(card.rank, []).append(card)

        # Add actions for playing all cards of the same rank
        for rank, cards in rank_groups.items():
            if len(cards) > 1:
                # Check if at least one card in the group is playable
                for card in cards:
                    if self.can_play_card(card, top_card):
                        actions.append(tuple(sorted(cards, key=lambda c: str(c))))  # might not need to sort the cards

        # 2. add single card playing actions
        for card in self.hand:
            if self.can_play_card(card, top_card):
                actions.append(tuple(card))

        # 3. Add a draw card action represented by and empty tuple
        actions.append(())

        return actions

    def get_state_key(self, game):
        """Create a simplified state representation for the Q-table"""
        top_card = game.get_top_card()

        # Count the playable cards
        playable_cards = self.get_playable_cards(top_card)
        num_playable = len(playable_cards)

        # check for special cards in the hand
        has_joker = any(card.rank == 'Joker' for card in self.hand)
        has_special = any(card.rank in ['2', '5', '8', 'J'] for card in self.hand)

        # Opponent hand sizes
        opponent_hand_sizes = tuple(sorted(
            len(p.hand) for p in game.players if p != self
        ))

        # create the state key
        state_key = (
            len(self.hand),                         # Player hand size
            num_playable,                           # number of playable cards in player hand
            1 if has_joker else 0,                  # has joker
            1 if has_special else 0,                # has a special card
            top_card.rank if top_card else None,    # top card rank
            opponent_hand_sizes,                    # hand sizes of the opponents
            len(game.deck.cards) > 10,              # deck has cards
        )

        return state_key

    def choose_action(self, game):
        """Choose action to take using epsilon-greedy Q-learning"""
        state = self.get_state_key(game)
        actions = self.get_valid_actions(game.get_top_card())

        # Initialize Q-values for new state
        # this implementation means that the state has to match the previous exactly before any update occurs
        # the q-table will be very robust
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in actions} # getting actions available for this particular state

        # Add any new actions that might not be in the Q-table
        # not sure why we would need to add them again TODO: ask Deepseek why
        for action in actions:
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # explore random actions
            action = random.choice(actions)
        else:
            # Exploit: choose action with the highest Q-value
            max_q = max(self.q_table[state].values())
            best_actions = [a for a, q in self.q_table[state].items()
                            if q == max_q and a in actions]
            action = random.choice(best_actions) if best_actions else random.choice(actions)

        return action

    def execute_action(self, action, game):
        """Execute the chosen action in the game"""
        top_card = game.get_top_card()

        if action == ():
            card_drawn = self.draw_card(game.deck, game)
            if card_drawn:
                print(f"{self.name} chooses to draw {card_drawn}")
            return 0  # no forced draws from drawing

        # Play cards
        for card in action:
            if card in self.hand:
                self.hand.remove(card)
                game.playing_stack.append(card)

        print(f"{self.name} plays: {', '.join(str(c) for c in action)}")

        # Handle special card effects
        if action[0].rank == '2':
            return 2 * len(action)
        elif action[0].rank == '5':
            return 3 * len(action)
        elif action[0].rank == 'J':
            forced_draws = len(action)
            for p in game.players:
                if p != self:
                    for _ in range(forced_draws):
                        p.draw_card(game.deck, game)
            return 0
        elif action[0].rank == '8':
            return -len(action)

        return 0

    def learn(self, state, action, reward, next_state, game):
        """Update Q-table using q-learning"""
        if state not in self.q_table:
            return

        # Get max Q-value for next state
        next_max_q = 0
        if next_state in self.q_table and self.q_table[next_state]:
            next_actions = self.get_valid_actions(game.get_top_card())
            valid_q_values = [
                self.q_table[next_state].get(a, 0) for a in next_actions if a in self.q_table[next_state]
            ]
            next_max_q = max(valid_q_values) if valid_q_values else 0

        # Q-learning update
        # Q-values are calculated here
        old_q = self.q_table[state].get(action, 0)
        self.q_table[state][action] = old_q + self.alpha * (
            reward + self.gamma * next_max_q - old_q
        )

    def update(self, game, reward):
        """Called after an action is executed"""
        if self.last_state is not None and self.last_action is not None:
            next_state = self.get_state_key(game)
            self.learn(self.last_state, self.last_action, reward, next_state, game)

        # Store current state/action for next update
        self.last_state = self.get_state_key(game)
        self.last_action = self.choose_action(game) if self.hand else None


class Game:
    def __init__(self, player_names):
        self.deck = Deck()
        self.players = [Player(name) for name in player_names]
        self.playing_stack = []

        # deal 4 cards to each player
        for player in self.players:
            for _ in range(4):
                player.draw_card(self.deck, self)

        # Initialize playing stack with a non-special card
        while True:
            card = self.deck.draw()
            if card and card.rank not in ['2', '5', '8', 'J', 'Joker']:  # Avoid starting with special cards
                self.playing_stack.append(card)
                break

    def get_top_card(self):
        return self.playing_stack[-1] if self.playing_stack else None

    def replenish_deck(self):
        if len(self.playing_stack) > 1:
            print("\nDeck is empty. Replenishing from playing stack...")
            top_card = self.playing_stack.pop()  # Keep the top card
            self.deck.cards = self.playing_stack  # Rest go to deck
            random.shuffle(self.deck.cards)
            self.playing_stack = [top_card]  # Restore top card
        else:
            print("\nDeck and playing stack are empty. Cannot replenish.")

    def play_turn(self, player):
        top_card = self.get_top_card()
        if not top_card:
            # Draw a card to start if playing stack is empty
            card = self.deck.draw()
            if card:
                self.playing_stack.append(card)
                top_card = card

        print(f"\n{player.name}'s turn. Top card: {top_card}")
        print(f"{player.name}'s hand: {player.hand}")

        cards_played = player.play_cards(top_card)

        if cards_played:
            for card in cards_played:
                self.playing_stack.append(card)
            print(f"{player.name} plays: {', '.join(map(str, cards_played))}")

            # Handle extended draw rules for 2, 5, and J
            if cards_played[0].rank == '2':
                forced_draws = 2 * len(cards_played)
                print(f"Rule triggered: Next player must draw {forced_draws} cards!")
                return forced_draws

            elif cards_played[0].rank == '5':
                forced_draws = 3 * len(cards_played)
                print(f"Rule triggered: Next player must draw {forced_draws} cards!")
                return forced_draws

            elif cards_played[0].rank == 'J':
                forced_draws = len(cards_played)
                print(f"Rule triggered: All other players must draw {forced_draws} cards!")
                for p in self.players:
                    if p != player:
                        for _ in range(forced_draws):
                            p.draw_card(self.deck, self)
                return 0  # No need for further handling

            elif cards_played[0].rank == '8':
                skips = len(cards_played)
                print(f"Rule triggered: Next {skips} player(s) will miss their turn!")
                return -skips  # Negative return value signifies number of skipped turns

        else:
            new_card = player.draw_card(self.deck, self)
            if new_card:
                print(f"{player.name} has no playable card. Draws {new_card}")
            else:
                print(f"{player.name} has no playable card and the deck is empty.")
        return 0  # No forced draw

    def check_winner(self):
        for player in self.players:
            if not player.has_cards():
                return player
        return None

    def get_loser(self):
        return max(self.players, key=lambda p: p.hand_value())

    def start(self):
        current_player = 0
        skip_count = 0

        while True:
            player = self.players[current_player]

            # Handle skip turns
            if skip_count > 0:
                print(f"{player.name} is skipped!")
                skip_count -= 1
                current_player = (current_player + 1) % len(self.players)
                continue

            result = self.play_turn(player)

            if result > 0:  # Forced draws (2 or 5)
                next_player_index = (current_player + 1) % len(self.players)
                next_player = self.players[next_player_index]
                print(f"{next_player.name} must draw {result} cards!")
                for _ in range(result):
                    next_player.draw_card(self.deck, self)
                current_player = next_player_index

            elif result < 0:  # Skip turns (8s)
                skip_count = -result - 1  # Skip next players
                current_player = (current_player + 1) % len(self.players)
                continue

            sleep(0.8)

            winner = self.check_winner()
            if winner:
                print(f"\n{'=' * 50}")
                print(f"🎉 We have a winner: {winner.name} wins! 🎉")
                loser = self.get_loser()
                print(f"😞 {loser.name} lost the game with a total card value of {loser.hand_value()}")
                print(f"{'=' * 50}")
                break

            current_player = (current_player + 1) % len(self.players)


if __name__ == "__main__":
    game = Game(["Alice", "Bob", "Charles", "Derrick"])
    game.start()
