# from hnefatafl.engine import Move, BoardGameException
# from alphazero.envs.hnefatafl.tafl_old import get_action
# from alphazero.envs.hnefatafl.fastafl import get_action
from alphazero.GenericPlayers import BasePlayer
from alphazero.Game import GameState
from alphazero.envs.diseatchess.diseatchess import BOARD_SIZE

import pyximport, numpy

pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from boardgame.errors import InvalidMoveError


class HumanPlayer(BasePlayer):
    @staticmethod
    def is_human() -> bool:
        return True

    def play(self, state: GameState):
        valid_moves = state.valid_moves()

        def string_to_action(player_inp: str) -> int:
            try:
                x, y = map(int, player_inp.split())
                return BOARD_SIZE * x + y
            except (ValueError, AttributeError, InvalidMoveError):
                return -1

        print(state.observation())

        action = string_to_action(input(f"Enter the move to play for the player {state.player}: "))
        while action == -1 or not valid_moves[action]:
            action = string_to_action(input(f"Illegal move (action={action}, "
                                            f"in valids: {bool(valid_moves[action])}). Enter a valid move: "))

        return action


