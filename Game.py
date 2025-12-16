from time import sleep
from collections import deque
from collections import defaultdict
import random
import pickle
import torch
import torch.nn as nn
import numpy as np


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

    def __eq__(self, other):
        return other.name == self.name

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

def create_default_dict():
    return defaultdict(float)

def create_default_q_table():
    return defaultdict(create_default_dict)

class SimpleRLPlayer(Player):
    def __init__(self, name, epsilon=0.1, alpha=0.1, gamma=0.9):
        super().__init__(name)
        self.q_table = create_default_q_table()  # Dictionary: state -> {action: q_value}
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.last_state = None
        self.last_action = None
        self.total_reward = 0
        self.memory = []

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
                actions.append((card,))

        # 3. Add a draw card action represented by and empty tuple
        actions.append(())

        return actions

    def get_state_key(self, game):
        """Create a simplified state representation for the Q-table"""
        top_card = game.get_top_card()

        # Count the playable cards
        playable_cards = self.get_playable_cards(top_card)
        num_playable = len(playable_cards)

        # Bucket hand size (instead of exact count)
        hand_size_bucket = self._bucket_hand_size(len(self.hand))

        # Bucket playable cards
        playable_bucket = self._bucket_playable(num_playable)

        # check for special cards in the hand
        has_joker = any(card.rank == 'Joker' for card in self.hand)
        has_special = any(card.rank in ['2', '5', '8', 'J'] for card in self.hand)

        # Bucket opponent threat level (instead of exact hand sizes)
        opponent_threat = self._bucket_opponent_threat(game)

        # Simplify top card rank to categories
        top_card_category = self._categorize_top_card(top_card)

        # Deck status (more generous bucketing)
        deck_status = self._bucket_deck_status(len(game.deck.cards))

        # Opponent hand sizes
        opponent_hand_sizes = tuple(sorted(
            len(p.hand) for p in game.players if p != self
        ))

        # create the state key
        state_key = (
            hand_size_bucket,                   # Player hand size
            playable_bucket,                    # number of playable cards in player hand
            1 if has_joker else 0,              # has joker
            1 if has_special else 0,            # has a special card
            top_card_category,                  # top card rank
            opponent_threat,                    # hand sizes of the opponents
            deck_status,                        # deck has cards
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

    def learn_from_step(self, reward, game):
        """Update Q-table based on last action"""
        if self.last_state is not None and self.last_action is not None:
            current_state = self.get_state_key(game)

            # Get max Q-value for current state
            next_max_q = 0
            if current_state in self.q_table and self.q_table[current_state]:
                next_max_q = max(self.q_table[current_state].values())

            # Q-learning update
            old_q = self.q_table[self.last_state][self.last_action]
            new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
            self.q_table[self.last_state][self.last_action] = new_q

            # Store for next update
            self.last_state = current_state
            self.last_action = None

        self.total_reward += reward

    def update(self, game, reward):
        """Called after an action is executed"""
        if self.last_state is not None and self.last_action is not None:
            next_state = self.get_state_key(game)
            self.learn(self.last_state, self.last_action, reward, next_state, game)

        # Store current state/action for next update
        self.last_state = self.get_state_key(game)
        self.last_action = self.choose_action(game) if self.hand else None

    def _bucket_hand_size(self, hand_size):
        """Bucket hand size into categories"""
        if hand_size <= 2:
            return "critical"  # Very few cards - close to winning!
        elif hand_size <= 5:
            return "low"
        elif hand_size <= 8:
            return "medium"
        else:
            return "high"

    def _bucket_playable(self, num_playable):
        """Bucket number of playable cards"""
        if num_playable == 0:
            return "none"
        elif num_playable == 1:
            return "one"
        elif num_playable <= 3:
            return "few"
        else:
            return "many"

    def _bucket_opponent_threat(self, game):
        """Assess opponent threat level based on their hand sizes"""
        opponent_hand_sizes = [len(p.hand) for p in game.players if p != self]

        if not opponent_hand_sizes:
            return "safe"

        min_opponent_cards = min(opponent_hand_sizes)

        if min_opponent_cards <= 2:
            return "critical"  # Someone is about to win!
        elif min_opponent_cards <= 4:
            return "danger"
        else:
            return "safe"

    def _categorize_top_card(self, top_card):
        """Categorize the top card into broader groups"""
        if not top_card:
            return None

        if top_card.rank in ['2', '5', '8', 'J', 'Joker']:
            return "special"
        elif top_card.rank in ['K', 'Q']:
            return "high"
        else:
            return "low"

    def _bucket_deck_status(self, deck_size):
        """Bucket deck size"""
        if deck_size > 20:
            return "plenty"
        elif deck_size > 5:
            return "low"
        else:
            return "critical"


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


class RLGame(Game):
    def __init__(self, player_names, use_rl=False):
        super().__init__(player_names)

        self.use_rl = use_rl

        # if use_rl:
        #     # Replace players with RL versions
        #     self.players = [SimpleRLPlayer(name) for name in player_names]
        #     # Deal cards to RL players
        #     for player in self.players:
        #         for _ in range(4):
        #             card = self.deck.draw()
        #             if card:
        #                 player.hand.append(card)

    # def play_turn(self, player):
    #     top_card = self.get_top_card()
    #     # TODO: adjust implementation for is a player's name ends with 's'
    #     print(f"\n{player.name}'s turn. Top card: {top_card}")
    #     print(f"Hand: {player.hand}")
    #
    #     if isinstance(player, SimpleRLPlayer):
    #         # RL player chooses action
    #         action = player.choose_action(self)
    #         result = player.execute_action(action, self)
    #
    #         # calculate reward for the RL player
    #         reward = self.calculate_reward(player, action, result)
    #         player.update(self, reward)
    #
    #     else:
    #         # Original non-RL player behaviour
    #         result = super().play_turn(player)
    #
    #     return result

    def play_turn(self, player):
        """Tracks rewards for RL players"""
        if isinstance(player, SimpleRLPlayer):
            return self.play_rl_turn(player)
        else:
            return super().play_turn(player)

    def play_rl_turn(self, player):
        """Special turn handling for RL agents with reward tracking"""
        top_card = self.get_top_card()
        if not top_card:
            # Initialize if needed
            card = self.deck.draw()
            if card:
                self.playing_stack.append(card)
                top_card = card

        print(f"  {player.name}'s turn. Top: {top_card}, Hand: {len(player.hand)} cards")

        # RL agent chooses and executes action
        action = player.choose_action(self)
        result = player.execute_action(action, self)

        # Calculate immediate reward
        reward = self.calculate_immediate_reward(player, action, result)

        # Update Q-learning
        current_state = player.get_state_key(self)
        player.learn_from_step(reward, self)

        return result
        

    def calculate_reward(self, player, action, result):
        """Calculate reward for the RL player"""
        reward = 0

        # Positive reward
        if action != ():
            reward += 1.0
            if len(action) > 1.0:
                reward += 1.0
            if action[0].rank in ['2', '5', '8', 'J', 'Joker']:
                reward += 2.0

        # Negative rewards
        if action == ():
            reward -= 0.5

        # Check for win
        if not player.has_cards():
            reward += 100.0

        # Penalty for high hand value
        # TODO: look to adjust this
        reward -= player.hand_value() * 0.01

        return reward

    def calculate_immediate_reward(self, player, action, result):
        """Calculate reward for RL agent's action"""
        reward = 0

        if action == ():  # Drew a card
            reward -= 0.3
        else:  # Played cards
            reward += 0.5

            # Bonus for playing multiple cards
            if len(action) > 1:
                reward += 0.5

            # Bonus for special cards
            if action[0].rank in ['2', '5', '8', 'J', 'Joker']:
                reward += 1.0

            # Big bonus for winning move
            if len(player.hand) == len(action):  # Played last cards
                reward += 10.0

        return reward

class RLTrainingSystem:
    def __init__(self):
        self.agent = None
        self.training_stats = []
        self.learning_curve = []

    def train_agent(self, num_episodes=100):
        """Train an RL agent over multiple episodes"""
        print("\n---------- Starting RL training -----------\n")

        self.agent = SimpleRLPlayer("RL_Learner", epsilon=0.3, alpha=0.2, gamma=0.95)

        for episode in range(num_episodes):
            print(f"\n=== Training episode {episode + 1}/{num_episodes} (ε={self.agent.epsilon:.3f}) ===")

            # Create new game with the SAME agent
            game = RLGame(["RL_Learner", "Bob", "Charles", "Derrick"], use_rl=False)

            # Replace the first player with our RL agent
            game.players[0] = self.agent

            # Ensure the agent's hand is empty and ready for new game
            self.agent.hand = []

            # Deal cards to the agent (bypassing the normal draw method)
            for _ in range(4):
                card = game.deck.draw()
                if card:
                    self.agent.hand.append(card)

            winner = self.play_episode(game, episode + 1)

            # Record statistics
            won = winner is not None and winner.name == "RL_Learner"
            self.training_stats.append({
                'episode': episode + 1,
                'won': won,
                'winner': winner.name if winner else 'Draw',
                'epsilon': self.agent.epsilon,
                'q_table_size': len(self.agent.q_table),
                'final_hand_size': len(self.agent.hand),
                'total_reward': self.agent.total_reward if hasattr(self.agent, 'total_reward') else 0
            })

            # Update learning curve (moving average of win rate)
            # TODO: ask why we wait until 10 episodes
            if len(self.training_stats) >= 10:
                recent_wins = sum(1 for s in self.training_stats[-10:] if s['won'])
                self.learning_curve.append(recent_wins / 10)

            # Decay exploration rate
            self.agent.epsilon = max(0.05, self.agent.epsilon * 0.99)

            # Reset agent's episodic memory (but keep the Q-table)
            self.agent.last_state = None
            self.agent.last_action = None
            self.agent.total_reward = 0

            # Show progress every 10 episodes
            if (episode + 1) % 10 == 0:
                self.show_progress(episode + 1)

    def play_episode(self, game, episode_num):
        """Play one complete game episode"""
        current_player = 0
        skip_count = 0
        turn_count = 0

        # Track if our agent made a move this turn (for reward calculation)
        agent_made_move = False

        while True:
            player = game.players[current_player]

            # skip handling
            if skip_count > 0:
                skip_count -= 1
                current_player = (current_player + 1) % len(game.players)
                continue

            turn_count += 1

            # Check if it's our agent's turn
            is_agent_turn = (player == self.agent)

            if is_agent_turn:
                print(f"   Agent's turn #{turn_count}: {len(player.hand)} cards")
                agent_made_move = True

            # play the turn
            # result variable stores if there will be a forced draw for the next player
            result = game.play_turn(player)

            # Check for winner
            winner = game.check_winner()
            if winner:
                print(f"  !! Winner after {turn_count} turns: {winner.name}")
                return winner

            if turn_count > 200:
                print(f"  Game terminated after {turn_count} turns (too long)")
                return None

            if game.deck.is_empty() and len(game.playing_stack) <= 1:
                print("  !! No more cards available")

            # Handle special card results
            if result > 0: # Forced draws
                next_idx = (current_player + 1) % len(game.players)
                next_player = game.players[next_idx]
                if is_agent_turn: # Agent forced opponent to draw
                    self.agent.total_reward += 0.5 * result # Bonus for forcing draws
                print(f"  {next_player.name} draws {result} cards!")
                for _ in range(result):
                    next_player.draw_card(game.deck, game)
                current_player = next_idx

            elif result < 0: # skip turns
                skip_count = -result - 1
                if is_agent_turn:  # Agent made opponent skip turns
                    self.agent.total_reward += 0.3 * (-result)
                current_player = (current_player + 1) % len(game.players)

            else:
                current_player = (current_player + 1) % len(game.players)

            # Give a small reward for surviving the round
            if is_agent_turn and agent_made_move:
                self.agent.total_reward += 0.1
                agent_made_move = False

    def show_progress(self, episode):
        """Display training progress"""
        if len(self.training_stats) < 10:
            return

        recent = self.training_stats[-10:]
        wins = sum(1 for s in recent if s['won'])

        print(f"\n{'=' * 50}")
        print(f" --- Training progress after {episode} episodes ---")
        print(f"\n{'=' * 50}")
        print(f"Win rate (last 10): {wins}/10 = {wins * 10}%")
        print(f"Exploration rate (ε): {self.agent.epsilon:.3f}")
        print(f"Learned states: {len(self.agent.q_table)}")

        # Show Q-Table sample if small enough
        if len(self.agent.q_table) < 20:
            print(f"\n Sample Q-values:")
            for i, (state, actions) in enumerate(list(self.agent.q_table.items())[:5]):
                print(f"        State {state}:")
                for action, q in list(actions.items())[:3]:
                    print(f"         {action}: {q:.2f}")

    def save_agent(self, filename="trained_card_player.pkl"):
        """Save the trained agent to file - FIXED VERSION"""
        # Convert defaultdict to regular dict for pickling
        if self.agent and hasattr(self.agent, 'q_table'):
            # Convert nested defaultdict to regular dict
            q_table_dict = {}
            for state, actions in self.agent.q_table.items():
                q_table_dict[state] = dict(actions)
            self.agent.q_table = q_table_dict

        # Save the data
        data = {
            'agent': self.agent,
            'training_stats': self.training_stats,
            'learning_curve': self.learning_curve,
            'agent_class': 'SimpleRLPlayer'  # Store class info
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Agent saved to {filename}")

        # Restore defaultdict structure if needed
        if self.agent and hasattr(self.agent, 'q_table'):
            self.agent.q_table = create_default_q_table()
            for state, actions in q_table_dict.items():
                self.agent.q_table[state].update(actions)

    def load_agent(self, filename="trained_card_player.pkl"):
        """Load a trained agent from file - FIXED VERSION"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.agent = data['agent']
        self.training_stats = data.get('training_stats', [])
        self.learning_curve = data.get('learning_curve', [])

        # Convert loaded dict back to defaultdict structure
        if self.agent and hasattr(self.agent, 'q_table'):
            q_table_dict = self.agent.q_table
            self.agent.q_table = create_default_q_table()
            for state, actions in q_table_dict.items():
                self.agent.q_table[state].update(actions)

        print(f"📂 Agent loaded from {filename}")
        print(f"   States learned: {len(self.agent.q_table)}")
        return self.agent

    def analyze_learning(self):
        """Analyze what the agent learned"""
        if not self.training_stats:
            print("No training data to analyze")
            return

        print(f"\n{'=' * 60}")
        print("🧠 Learning Analysis Report")
        print(f"{'=' * 60}")

        # Calculate win rates in different phases
        total_episodes = len(self.training_stats)
        quarter = total_episodes // 4

        phases = [
            ("First 25%", self.training_stats[:quarter]),
            ("Second 25%", self.training_stats[quarter:2 * quarter]),
            ("Third 25%", self.training_stats[2 * quarter:3 * quarter]),
            ("Final 25%", self.training_stats[3 * quarter:])
        ]

        for phase_name, phase_stats in phases:
            if phase_stats:
                wins = sum(1 for s in phase_stats if s['won'])
                total = len(phase_stats)
                win_rate = (wins / total) * 100 if total > 0 else 0
                print(f"{phase_name}: {wins}/{total} wins ({win_rate:.1f}%)")

        # Show final Q-table statistics
        print(f"\n📊 Final Knowledge Base:")
        print(f"  Total states learned: {len(self.agent.q_table)}")

        # Count actions per state
        actions_per_state = [len(actions) for actions in self.agent.q_table.values()]
        if actions_per_state:
            avg_actions = sum(actions_per_state) / len(actions_per_state)
            print(f"  Average actions per state: {avg_actions:.1f}")

        # Show highest Q-values
        print(f"\n🏆 Best Learned Strategies:")
        high_q_states = sorted(
            self.agent.q_table.items(),
            key=lambda x: max(x[1].values()) if x[1] else 0,
            reverse=True
        )[:3]

        for state, actions in high_q_states:
            if actions:
                best_action = max(actions.items(), key=lambda x: x[1])
                print(f"  State {state[:3]}... -> {best_action[0]}: Q={best_action[1]:.2f}")


# if __name__ == '__main__':
#     # Train RL agent against rule-based players
#     game = RLGame(['RL_agent', 'Bob', 'Charles', 'Derrick'], use_rl=True)
#
#     # Train for multiple episodes
#     for episode in range(100):
#         print(f"\n=== Episode {episode + 1} ===")
#         game.start()
#
#         # Reset game for next episode
#         if episode < 99:
#             game = RLGame(['RL_agent', 'Bob', 'Charles', 'Derrick'], use_rl=True)
#
#     # Test trained agent
#     print("\n=== Testing trained agent ===")
#     test_game = RLGame(['Trained_RL', 'Bob', 'Charles', 'Derrick'], use_rl=True)
#     test_game.start()


# if __name__ == "__main__":
#     game = Game(["Alice", "Bob", "Charles", "Derrick"])
#     game.start()

if __name__ == "__main__":
    # ========== TRAINING PHASE ==========
    print("🤖 CARD GAME RL TRAINING SYSTEM")
    print("=" * 50)

    trainer = RLTrainingSystem()

    # Train for 100 episodes
    trainer.train_agent(num_episodes=100)

    # Save the trained agent
    trainer.save_agent("trained_card_player.pkl")

    # Analyze learning
    trainer.analyze_learning()

    # ========== TESTING PHASE ==========
    print("\n" + "=" * 50)
    print("🧪 FINAL TEST WITH TRAINED AGENT")
    print("=" * 50)

    # Create a fresh game
    test_game = Game(["RL_Learner", "Expert_Bob", "Smart_Charlie", "Quick_Derrick"])

    # Load the trained agent
    trained_agent = trainer.load_agent("trained_card_player.pkl")
    trained_agent.name = "RL_Learner"
    trained_agent.epsilon = 0.9  # Minimal exploration for testing

    # Replace first player with trained agent
    test_game.players[0] = trained_agent

    # Reset agent's hand and deal fresh cards
    trained_agent.hand = []
    for _ in range(4):
        card = test_game.deck.draw()
        if card:
            trained_agent.hand.append(card)

    # Play the test match
    print("\n🎮 Starting Final Test Match...")
    test_game.start()

    # Show agent's performance
    print(f"\n📊 Trained Agent Final Stats:")
    print(f"  Final hand size: {len(trained_agent.hand)}")
    print(f"  Hand value: {trained_agent.hand_value()}")
    print(f"  Total Q-table states: {len(trained_agent.q_table)}")
