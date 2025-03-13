# cython: language_level=3

from alphazero.Game import GameState
from typing import List, Tuple, Any

# import diseatchess
import string

import numpy as np

BOARD_SIZE = 7
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
BLOCK = 2
inithealth = 2
attackpower = 1
healpower = 1
isDiagHeal = True
isDiagAttack = True

NUM_CHANNELS = 1
ACTION_SIZE = BOARD_SIZE ** 2
OBSERVATION_SIZE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
NUM_PLAYERS = 2
MAX_TURNS = BOARD_SIZE * BOARD_SIZE


class DEChess:
    def __init__(self):
        self.board = np.array([[{'type': EMPTY, 'health': 0} for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)])
        self.currentPlayer = PLAYER_X
        self.boardChanged = False

    def copy(self):
        new_game = DEChess()
        new_game.board = np.array([
            [{'type': cell['type'], 'health': cell['health']} for cell in row]
            for row in self.board
        ])
        new_game.currentPlayer = self.currentPlayer
        new_game.boardChanged = self.boardChanged
        return new_game

    def initialize_board(self):
        self.board = np.array([[{'type': EMPTY, 'health': 0} for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)])
        self.currentPlayer = PLAYER_X
        self.boardChanged = False

    def heal_rule(self, row, col):
        currentPlayer = self.board[row][col]['type']
        if currentPlayer == EMPTY or currentPlayer == BLOCK:
            return
        self.board[row][col]['health'] = inithealth
        n = 0
        directNeighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for dr, dc in directNeighbors:
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c]['type'] == currentPlayer:
                n += 1
        diagonalNeighbors = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        m = 0
        for dr, dc in diagonalNeighbors:
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c]['type'] == currentPlayer:
                m += 1
        if n >= 2:
            if m > 0 and isDiagHeal:
                self.board[row][col]['health'] = inithealth + (n + m) * healpower - 1
            else:
                self.board[row][col]['health'] = inithealth + n * healpower - 1

    def damage_rule(self, row, col, currentPlayer):
        if self.board[row][col]['type'] == EMPTY or self.board[row][col]['type'] == BLOCK:
            return
        opponent = PLAYER_O if currentPlayer == PLAYER_X else PLAYER_X
        n = 0
        directNeighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for dr, dc in directNeighbors:
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c]['type'] == opponent:
                n += 1
        diagonalNeighbors = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        m = 0
        for dr, dc in diagonalNeighbors:
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c]['type'] == opponent:
                m += 1
        if n >= 2:
            if m > 0 and isDiagAttack:
                self.board[row][col]['health'] -= (n + m) * attackpower
            else:
                self.board[row][col]['health'] -= n * attackpower

    def death_rule(self, row, col):
        if self.board[row][col]['type'] == EMPTY or self.board[row][col]['type'] == BLOCK:
            return
        if self.board[row][col]['health'] <= 0:
            self.board[row][col]['type'] = PLAYER_O if self.board[row][col]['type'] == PLAYER_X else PLAYER_X
            self.board[row][col]['health'] = 2
            self.boardChanged = True

    def block_rule(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                directNeighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
                xCount, oCount = 0, 0
                for dr, dc in directNeighbors:
                    r, c = i + dr, j + dc
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        if self.board[r][c]['type'] == PLAYER_X:
                            xCount += 1
                        elif self.board[r][c]['type'] == PLAYER_O:
                            oCount += 1
                if xCount >= 2 and oCount >= 2:
                    self.board[i][j]['type'] = BLOCK
                    self.board[i][j]['health'] = 0

    def refresh_board(self):
        self.boardChanged = False
        if not isDiagHeal or not isDiagAttack or attackpower != healpower:
            self.block_rule()
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] != EMPTY:
                    self.heal_rule(i, j)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == self.currentPlayer:
                    self.damage_rule(i, j, self.currentPlayer)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == self.currentPlayer:
                    self.death_rule(i, j)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] != EMPTY:
                    self.heal_rule(i, j)
        opponent = PLAYER_O if self.currentPlayer == PLAYER_X else PLAYER_X
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == opponent:
                    self.damage_rule(i, j, opponent)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == opponent:
                    self.death_rule(i, j)

    def is_board_full(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == EMPTY:
                    return False
        return True

    def count_pieces(self):
        xCount, oCount = 0, 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == PLAYER_X:
                    xCount += 1
                elif self.board[i][j]['type'] == PLAYER_O:
                    oCount += 1
        return xCount, oCount

    def determine_winner(self):
        if self.is_board_full():
            xCount, oCount = self.count_pieces()
            if xCount > oCount:
                return PLAYER_X
            elif oCount > xCount:
                return PLAYER_O
            else:
                return 'Draw'
        return None

    def get_state(self):
        state = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                cell = self.board[i][j]
                if cell['type'] == PLAYER_X:
                    state.append(1)
                elif cell['type'] == PLAYER_O:
                    state.append(-1)
                elif cell['type'] == BLOCK:
                    state.append(0)
                else:
                    state.append(0)
        return np.array(state)

    def get_valid_moves(self):
        valid_moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]['type'] == EMPTY:
                    valid_moves.append((i, j))
        return valid_moves

    def make_move(self, row, col):
        if self.board[row][col]['type'] == EMPTY:
            self.board[row][col]['type'] = self.currentPlayer
            self.board[row][col]['health'] = inithealth
            self.boardChanged = True
            while self.boardChanged:
                self.boardChanged = False
                self.refresh_board()
            winner = self.determine_winner()
            if winner is not None:
                return winner
            self.currentPlayer = PLAYER_O if self.currentPlayer == PLAYER_X else PLAYER_X
            return None
        else:
            raise RuntimeError('Cannot make a move')
        # return None


class Game(GameState):
    def __init__(self, _board = None):
        super().__init__(_board or self._get_board())

    def __eq__(self, other: 'Game') -> bool:
        return (
            np.asarray(self._board.get_state()) == np.asarray(other._board.get_state())
            and self._player == other._player
            and self.turns == other.turns
        )

    def __hash__(self) -> int:
        return hash(self._board.get_state().tobytes() + bytes([self._turns]) + bytes([self._player]))


    @staticmethod
    def _get_board():
        return DEChess()

    def clone(self) -> 'Game':
        # print(f"Clone: {self}")
        g = Game(_board=self._board.copy())
        g._player = self._player
        g._turns = self.turns
        # g.last_action = self.last_action
        return g

    @staticmethod
    def action_size() -> int:
        # print(f"Get ACTION_SIZE: {ACTION_SIZE}")
        return ACTION_SIZE

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        # print(f"Get OBSERVATION_SIZE: {OBSERVATION_SIZE}")
        return OBSERVATION_SIZE

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @staticmethod
    def max_turns() -> int:
        return MAX_TURNS

    @staticmethod
    def has_draw() -> bool:
        return True

    def play_action(self, action: int) -> None:
        super().play_action(action)
        move = (action // BOARD_SIZE, action % BOARD_SIZE)
        self._board.make_move(move[0], move[1])
        self._update_turn()

    def valid_moves(self) -> np.ndarray:
        valids = [0] * self.action_size()

        for x, y in self._board.get_valid_moves():
            valids[BOARD_SIZE * x + y] = 1

        # print(f"valids: {valids}")

        return np.array(valids, dtype=np.intc)

    def win_state(self) -> np.ndarray:
        # print("WIN STATE")

        result = [False] * (NUM_PLAYERS + 1)

        winner = self._board.determine_winner()

        if winner is not None:
            if winner == PLAYER_X:
                result[0] = True
            elif winner == PLAYER_O:
                result[1] = True
            else:
                result[2] = True

        return np.array(result, dtype=np.uint8)

    def observation(self) -> np.ndarray:
        state = self._board.get_state()
        return np.expand_dims(np.resize(np.asarray(state), (BOARD_SIZE, BOARD_SIZE)), axis=0)

    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        print(f"Symmetries: {pi}")
        raise RuntimeError("Symmetries not implemented.")
