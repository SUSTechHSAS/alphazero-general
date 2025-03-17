# cython: language_level=3
# distutils: language=c++

from alphazero.Game import GameState
from typing import List, Tuple, Any

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

BOARD_SIZE = 7

# Constants
DEF BOARD_SIZE = 7
DEF EMPTY = 0
DEF PLAYER_X = 1
DEF PLAYER_O = -1
DEF BLOCK = 2

DEF inithealth = 2
DEF attackpower = 1
DEF healpower = 1
DEF isDiagHeal = True
DEF isDiagAttack = True

cdef struct Cell:
    int type
    int health

cdef class DEChess:
    cdef:
        Cell** board
        int step
        int currentPlayer
        bint boardChanged
        readonly int init_health
        readonly int attack_power
        readonly int heal_power
        readonly bint is_diag_heal
        readonly bint is_diag_attack

    def __cinit__(self):
        # Allocate board memory
        self.board = <Cell**> malloc(BOARD_SIZE * sizeof(Cell *))
        for i in range(BOARD_SIZE):
            self.board[i] = <Cell *> malloc(BOARD_SIZE * sizeof(Cell))

        self.step = 0

        self.init_health = 2
        self.attack_power = 1
        self.heal_power = 1
        self.is_diag_heal = True
        self.is_diag_attack = True
        self.initialize_board()

    def __dealloc__(self):
        # Free board memory
        for i in range(BOARD_SIZE):
            free(self.board[i])
        free(self.board)

    def __reduce__(self):
        # Return state needed to reconstruct the object
        state = (
            # Board state as a 2D list of dicts
            [[{'type': self.board[i][j].type, 'health': self.board[i][j].health}
              for j in range(BOARD_SIZE)]
             for i in range(BOARD_SIZE)],
            self.step,
            self.currentPlayer,
            self.boardChanged,
            self.init_health,
            self.attack_power,
            self.heal_power,
            self.is_diag_heal,
            self.is_diag_attack
        )

        return (self.__class__, (), state)

    def __setstate__(self, state):
        # Reconstruct object from saved state
        board_state, step, current_player, board_changed, init_health, attack_power, heal_power, is_diag_heal, is_diag_attack = state

        # Allocate board memory
        self.board = <Cell**> malloc(BOARD_SIZE * sizeof(Cell *))
        for i in range(BOARD_SIZE):
            self.board[i] = <Cell *> malloc(BOARD_SIZE * sizeof(Cell))

        # Restore board state
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.board[i][j].type = board_state[i][j]['type']
                self.board[i][j].health = board_state[i][j]['health']

        # Restore other attributes
        self.step = step
        self.currentPlayer = current_player
        self.boardChanged = board_changed
        self.init_health = init_health
        self.attack_power = attack_power
        self.heal_power = heal_power
        self.is_diag_heal = is_diag_heal
        self.is_diag_attack = is_diag_attack

    cpdef void initialize_board(self):
        cdef int i, j
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.board[i][j].type = EMPTY
                self.board[i][j].health = 0
        self.currentPlayer = PLAYER_X
        self.boardChanged = False

    cdef inline void heal_rule(self, int row, int col):
        cdef:
            int currentPlayer = self.board[row][col].type
            int direct_count = 0
            int diag_count = 0
            int r, c, i
            int[4][2] direct_neighbors = [[-1, 0], [0, -1], [0, 1], [1, 0]]
            int[4][2] diag_neighbors = [[-1, -1], [-1, 1], [1, -1], [1, 1]]

        if currentPlayer == EMPTY or currentPlayer == BLOCK:
            return

        # Count direct neighbors
        for i in range(4):
            r = row + direct_neighbors[i][0]
            c = col + direct_neighbors[i][1]
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c].type == currentPlayer:
                direct_count += 1

        # Count diagonal neighbors
        if self.is_diag_heal:
            for i in range(4):
                r = row + diag_neighbors[i][0]
                c = col + diag_neighbors[i][1]
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c].type == currentPlayer:
                    diag_count += 1

        if direct_count >= 2:
            if diag_count > 0 and self.is_diag_heal:
                self.board[row][col].health = self.init_health + (direct_count + diag_count) * self.heal_power - 1
            else:
                self.board[row][col].health = self.init_health + direct_count * self.heal_power - 1
        else:
            self.board[row][col].health = self.init_health

    cdef inline void damage_rule(self, int row, int col, int currentPlayer):
        cdef:
            int opponent = PLAYER_O if currentPlayer == PLAYER_X else PLAYER_X
            int direct_count = 0
            int diag_count = 0
            int r, c, i
            int[4][2] direct_neighbors = [[-1, 0], [0, -1], [0, 1], [1, 0]]
            int[4][2] diag_neighbors = [[-1, -1], [-1, 1], [1, -1], [1, 1]]

        if self.board[row][col].type == EMPTY or self.board[row][col].type == BLOCK:
            return

        # Count direct opponents
        for i in range(4):
            r = row + direct_neighbors[i][0]
            c = col + direct_neighbors[i][1]
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c].type == opponent:
                direct_count += 1

        # Count diagonal opponents
        if self.is_diag_attack:
            for i in range(4):
                r = row + diag_neighbors[i][0]
                c = col + diag_neighbors[i][1]
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c].type == opponent:
                    diag_count += 1

        if direct_count >= 2:
            if diag_count > 0 and self.is_diag_attack:
                self.board[row][col].health -= (direct_count + diag_count) * self.attack_power
            else:
                self.board[row][col].health -= direct_count * self.attack_power

    cpdef void refresh_cell(self, int x, int y, int player):
        cdef:
            int opponent = PLAYER_O if player == PLAYER_X else PLAYER_X
            int r, c, i
            int[4][2] direct_neighbors = [[-1, 0], [0, -1], [0, 1], [1, 0]]
            int[4][2] diag_neighbors = [[-1, -1], [-1, 1], [1, -1], [1, 1]]

        if self.board[x][y].type == player:
            self.heal_rule(x, y)
            self.damage_rule(x, y, player)
            if self.board[x][y].health <= 0:
                self.board[x][y].type = -player
                self.board[x][y].health = self.init_health
                for i in range(4):
                    r = x + direct_neighbors[i][0]
                    c = y + direct_neighbors[i][1]
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        self.refresh_cell(r, c, player)
                for i in range(4):
                    r = x + diag_neighbors[i][0]
                    c = y + diag_neighbors[i][1]
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        self.refresh_cell(r, c, player)

    # Add to the DEChess class in diseatchess.pyx

    cpdef get_state(self):
        cdef:
            int i, j
            list state = []

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j].type == PLAYER_X:
                    state.append(1)
                elif self.board[i][j].type == PLAYER_O:
                    state.append(-1)
                else:
                    state.append(0)
        return np.array(state, dtype=np.int32)

    cpdef list get_valid_moves(self):
        cdef:
            int i, j
            list valid_moves = []

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j].type == EMPTY:
                    valid_moves.append((i, j))
        return valid_moves

    cpdef make_move(self, int row, int col):
        cdef:
            int opponent = PLAYER_O if self.currentPlayer == PLAYER_X else PLAYER_X
            int r, c, i
            int[4][2] direct_neighbors = [[-1, 0], [0, -1], [0, 1], [1, 0]]
            int[4][2] diag_neighbors = [[-1, -1], [-1, 1], [1, -1], [1, 1]]

        if self.board[row][col].type != EMPTY:
            raise RuntimeError('Cannot make a move')

        self.board[row][col].type = self.currentPlayer
        self.board[row][col].health = self.init_health

        self.refresh_cell(row, col, self.currentPlayer)

        if self.board[row][col].type == self.currentPlayer:
            for i in range(4):
                r = row + direct_neighbors[i][0]
                c = col + direct_neighbors[i][1]
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    self.refresh_cell(r, c, opponent)
            for i in range(4):
                r = row + diag_neighbors[i][0]
                c = col + diag_neighbors[i][1]
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    self.refresh_cell(r, c, opponent)

        self.step += 1

        winner = self.determine_winner()
        if winner is not None:
            return winner

        self.currentPlayer = PLAYER_O if self.currentPlayer == PLAYER_X else PLAYER_X
        return None

    cpdef determine_winner(self):
        cdef int x_count, o_count
        if self.is_board_full():
            x_count, o_count = self.count_pieces()
            if x_count > o_count:
                return PLAYER_X
            elif o_count > x_count:
                return PLAYER_O
            return 'Draw'
        return None

    cpdef bint is_board_full(self):
        return self.step == BOARD_SIZE * BOARD_SIZE

    cpdef tuple count_pieces(self):
        cdef:
            int i, j
            int x_count = 0
            int o_count = 0

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j].type == PLAYER_X:
                    x_count += 1
                elif self.board[i][j].type == PLAYER_O:
                    o_count += 1

        return x_count, o_count

    def __str__(self):
        cdef:
            int i, j
            str result = ""
            dict piece_map = {
                EMPTY: ".",
                PLAYER_X: "X",
                PLAYER_O: "O",
                BLOCK: "#"
            }

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = piece_map[self.board[i][j].type]
                health = self.board[i][j].health
                result += f"{piece}{health} "
            result += "\n"
        return result

    def copy(self):
        cdef:
            DEChess new_game = DEChess()
            int i, j

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                new_game.board[i][j].type = self.board[i][j].type
                new_game.board[i][j].health = self.board[i][j].health

        new_game.step = self.step
        new_game.currentPlayer = self.currentPlayer
        new_game.boardChanged = self.boardChanged
        new_game.init_health = self.init_health
        new_game.attack_power = self.attack_power
        new_game.heal_power = self.heal_power
        new_game.is_diag_heal = self.is_diag_heal
        new_game.is_diag_attack = self.is_diag_attack

        return new_game

NUM_CHANNELS = 1
ACTION_SIZE = BOARD_SIZE ** 2
OBSERVATION_SIZE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
NUM_PLAYERS = 2
MAX_TURNS = BOARD_SIZE * BOARD_SIZE

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
