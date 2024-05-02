# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 33:
# 107502 Francisco Ferro Pereira
# 104182 Tiago Romão

import sys
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id
    
    # TODO: outros metodos da classe


class Board:
    """ Representação interna de uma grelha de PipeMania. """
    def __init__(self, board):
        self.board = board
        self.row_count = len(board)
        self.col_count = len(board[0]) if board else 0

    @staticmethod
    def parse_instance():
        """Lê a instância do problema do standard input (stdin)
        e retorna uma instância da classe Board.
        Por exemplo:
        $ python3 pipe_mania.py < input_T01
        > from sys import stdin
        > line = stdin.readline().split()
        """
        board = []
        for line in sys.stdin:
            row = line.strip().split()
            board.append(row)
        return Board(board)
    
    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """ Devolve os valores imediatamente acima e abaixo,
        respectivamente. """
        vertical_up = None
        vertical_down = None

        if row > 0:
            vertical_up = self.board[row - 1][col]

        if row < self.row_count - 1:  # Check against the upper bound of rows
            vertical_down = self.board[row + 1][col]

        return (vertical_up, vertical_down)


    
    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """ Devolve os valores imediatamente à esquerda e à direita,
        respectivamente. """
        horizontal_left = None
        horizontal_right = None

        if col > 0:
            horizontal_left = self.board[row][col - 1]

        if col < self.col_count - 1:  # Check against the upper bound of columns
            horizontal_right = self.board[row][col + 1]

        return (horizontal_left, horizontal_right)
    
    def get_value(self, row: int, col: int):
        """ Devolve o valor nas coordenadas dadas como argumento """
        return self.board[row][col]



class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass


board = Board.parse_instance() 
print(board.adjacent_vertical_values(0, 0))
print(board.adjacent_horizontal_values(0, 0))
print(board.adjacent_vertical_values(1, 1))
print(board.adjacent_horizontal_values(1, 1))

#my_problem = PipeMania(board)

#result = depth_first_tree_search(my_problem)