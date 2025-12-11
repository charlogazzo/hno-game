from cgitb import handler
from time import sleep

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


import random


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

    # def play_turn(self, player):
    #     print(f"\n{player.name}'s turn. Top card: {self.get_top_card()}")
    #     cards_played = player.play_cards(self.get_top_card())
    #
    #     if cards_played:
    #         for card in cards_played:
    #             self.playing_stack.append(card)
    #         print(f"{player.name} plays: {', '.join(map(str, cards_played))}")
    #
    #         # Handle extended draw rules for 2, 5, and J
    #         if cards_played[0].rank == '2':
    #             forced_draws = 2 * len(cards_played)
    #             print(f"Rule triggered: Next player must draw {forced_draws} cards!")
    #             return forced_draws
    #
    #         elif cards_played[0].rank == '5':
    #             forced_draws = 3 * len(cards_played)
    #             print(f"Rule triggered: Next player must draw {forced_draws} cards!")
    #             return forced_draws
    #
    #         elif cards_played[0].rank == 'J':
    #             forced_draws = len(cards_played)
    #             print(f"Rule triggered: All other players must draw {forced_draws} cards!")
    #             for p in self.players:
    #                 if p != player:
    #                     for _ in range(forced_draws):
    #                         p.draw_card(self.deck, self)
    #             return 0  # No need for further handling
    #         elif cards_played[0].rank == '8':
    #             skips = len(cards_played)
    #             print(f"Rule triggered: Next {skips} player(s) will miss their turn!")
    #             return -skips  # Negative return value signifies number of skipped turns
    #
    #     else:
    #         new_card = player.draw_card(self.deck, self)
    #         if new_card:
    #             print(f"{player.name} has no playable card. Draws {new_card}")
    #         else:
    #             print(f"{player.name} has no playable card and the deck is empty.")
    #     return 0  # No forced draw

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

    # def start(self):
    #     current_player = 0
    #     while True:
    #         player = self.players[current_player]
    #         result = self.play_turn(player)
    #
    #         if result > 0:  # Forced draws (2 or 5)
    #             next_player_index = (current_player + 1) % len(self.players)
    #             next_player = self.players[next_player_index]
    #             print(f"{next_player.name} must draw {result} cards!")
    #             for _ in range(result):
    #                 next_player.draw_card(self.deck, self)
    #
    #         elif result < 0:  # Skip turns (8s)
    #             skips = -result
    #             print(f"Skipping next {skips} player(s)...")
    #             current_player = (current_player + skips) % len(self.players)
    #
    #         sleep(0.8)
    #
    #         winner = self.check_winner()
    #         if winner:
    #             print(f"\n ---------- We have a winner -----------\n {winner.name} wins")
    #             loser = self.get_loser()
    #             print(f" -__- {loser.name} lost the game with a total card value of {loser.hand_value()}")
    #             break
    #
    #         else:
    #             current_player = (current_player + 1) % len(self.players)

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
