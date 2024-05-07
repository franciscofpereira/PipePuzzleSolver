# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 33:
# 107502 Francisco Ferro Pereira
# 104182 Tiago Romão

import sys
from copy import deepcopy

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
    
    def print_board(self):
        """Imprime a grelha do tabuleiro"""
        for row in range(self.row_count):
            for col in range(self.col_count):
                print(f"{self.board[row][col]}\t", end='')  # Tab spacing between elements
            print()  # Move to the next line after printing each row

        
    def translate_pipe(self, row: int, col: int):
        """ Devolve um tuplo do formato (CIMA, BAIXO, ESQUERDA, DIREITA) com entradas a 1 nas direções em
        em que o pipe é aberto e com entradas a 0 nas direções em que o pipe é fechado """

        pipe_type, orientation = self.board[row][col]

        pipe_translations = {
        'F': {'C': (1, 0, 0, 0), 'B': (0, 1, 0, 0), 'E': (0, 0, 1, 0), 'D': (0, 0, 0, 1)},
        'B': {'C': (1, 0, 1, 1), 'B': (0, 1, 1, 1), 'E': (1, 1, 1, 0), 'D': (1, 1, 0, 1)},
        'V': {'C': (1, 0, 1, 0), 'B': (0, 1, 0, 1), 'E': (0, 1, 1, 0), 'D': (1, 0, 0, 1)},
        'L': {'V': (1, 1, 0, 0), 'H': (0, 0, 1, 1)}
        }

        return pipe_translations[pipe_type][orientation]
    

    def is_connected(self, row: int, col: int):
        
        """ Retorna True se o pipe estiver conectado em todas as suas aberturas, falso caso contrário """
        
        pipe_type, orientation = self.board[row][col]
    
        pipe_below = self.translate_pipe(row+1, col) if row+1 < self.row_count else (0,0,0,0)
        pipe_above = self.translate_pipe(row-1, col) if row-1 >= 0 else (0,0,0,0)
        pipe_left = self.translate_pipe(row, col-1) if col-1 >= 0 else (0,0,0,0)
        pipe_right = self.translate_pipe(row, col+1) if col+1 < self.col_count else (0,0,0,0)

        
        #TODO: Caso em que pipe adjacente a pipe de fecho também é de fecho.
         
        if pipe_type == 'F':

            if pipe_type == 'F':
                if orientation == 'C' and row - 1 >= 0 and pipe_above[1] == 1 and self.board[row-1][col][0] != 'F':
                    return True
                elif orientation == 'B' and row + 1 < self.row_count and pipe_below[0] == 1 and self.board[row+1][col][0] != 'F':
                    return True
                elif orientation == 'E' and col + 1 < self.col_count and pipe_right[3] == 1 and self.board[row][col+1][0] != 'F':
                    return True
                elif orientation == 'D' and col - 1 >= 0 and pipe_left[2] == 1 and self.board[row][col-1][0] != 'F':
                    return True
        
        elif pipe_type == 'B':

            if orientation == 'C' and pipe_above[1] == 1 and pipe_left[3] == 1 and pipe_right[2] == 1:
                return True
            elif orientation == 'B' and pipe_below[0] == 1 and pipe_left[3] == 1 and pipe_right[2] == 1:
                return True
            elif orientation == 'E' and pipe_above[1] == 1 and pipe_below[0] == 1 and pipe_left[3] == 1:
                return True
            elif orientation == 'D' and pipe_above[1] == 1 and pipe_below[0] == 1 and pipe_right[2] == 1:
                return True
            
        elif pipe_type == 'V':

            if orientation == 'C' and pipe_above[1] == 1 and pipe_left[3] == 1:
                return True
            elif orientation == 'B' and pipe_below[0] == 1 and pipe_right[2] == 1:
                return True
            elif orientation == 'E' and pipe_below[0] == 1 and pipe_left[3] == 1:
                return True
            elif orientation == 'D' and pipe_above[1] == 1 and pipe_right[2] == 1:
                return True
        
        elif pipe_type == 'L':
            
            if orientation == 'H' and pipe_left[3] == 1 and pipe_right[2] == 1:
                return True
            elif orientation == 'V' and pipe_above[1] == 1 and pipe_below[0] == 1:
                return True
        
        return False

    def sum_connected_pipes(self):
        """Retorna o número de pipes conectados no Board."""
        
        connected_pipes_count = 0
        for i in range(0, self.row_count):
            for j in range(0, self.col_count):

                if self.is_connected(i,j):
                    connected_pipes_count += 1

        return connected_pipes_count

      
class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(PipeManiaState(board))
        

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento e faz pre-pruning em ações de pipes que estão nas bordas do puzzle."""
        
        actions = []

        for i in range(0,state.board.row_count):
            for j in range(0, state.board.col_count):

                if state.board.is_connected(i,j):
                    continue

                pipe_type, orientation = state.board.board[i][j]

                actions_to_remove = []

                # Check if we are on the first row
                if i == 0:
                    if pipe_type == 'F':
                        actions_to_remove.append('C')  
                    elif pipe_type == 'V':
                        actions_to_remove.append('C')
                        actions_to_remove.append('D')
                    elif pipe_type == 'B':
                        actions_to_remove.append('C')
                        actions_to_remove.append('E')
                        actions_to_remove.append('D')
                    elif pipe_type == 'L':
                        actions_to_remove.append('V')  
                
                # Check if we are on the leftmost column
                if j == 0:
                    if pipe_type == 'F':
                        actions_to_remove.append('E')
                    elif pipe_type == 'B':
                        actions_to_remove.append('C')
                        actions_to_remove.append('B')
                        actions_to_remove.append('E')
                    elif pipe_type == 'V':
                        actions_to_remove.append('C')
                        actions_to_remove.append('E')
                    elif pipe_type == 'L':
                        actions_to_remove.append('H')  

                # Check if we are on the rightmost column
                if j == state.board.col_count - 1:
                    if pipe_type == 'F':
                        actions_to_remove.append('D')  # Rightward rotation
                    elif pipe_type == 'B':
                        actions_to_remove.append('C')
                        actions_to_remove.append('B')
                        actions_to_remove.append('D')
                    elif pipe_type == 'V':
                        actions_to_remove.append('B')
                        actions_to_remove.append('D')
                    elif pipe_type == 'L':
                        actions_to_remove.append('H')

                # Check if we are on the bottom row
                if i == state.board.row_count - 1:
                    if pipe_type == 'F':
                        actions_to_remove.append('B')  # Downward rotation
                    elif pipe_type == 'B':
                        actions_to_remove.append('B')
                        actions_to_remove.append('E')
                        actions_to_remove.append('D')
                    elif pipe_type == 'V':
                        actions_to_remove.append('B')
                        actions_to_remove.append('E')
                    elif pipe_type == 'L':
                        actions_to_remove.append('V')

                # If pipe style is 'L', remove all rotation actions
                if pipe_type == 'L':
                    actions_to_remove += ['C', 'B', 'E', 'D']

                    # Add the opposite orientation as an action
                    if orientation == 'V':
                        actions_to_remove.append('V')
                    else:
                        actions_to_remove.append('H')
                else:
                    actions_to_remove += ['V', 'H']

                    if orientation == 'C':
                        actions_to_remove.append('C')
                    if orientation == 'B':
                        actions_to_remove.append('B')
                    if orientation == 'E':
                        actions_to_remove.append('E')
                    if orientation == 'D':
                        actions_to_remove.append('D')

                # Remove actions based on position and pipe_type
                possible_actions = ['C', 'B', 'E', 'D', 'V', 'H']
                for action in set(actions_to_remove):
                    possible_actions.remove(action)

                # Add remaining actions to the list
                for action in possible_actions:
                    actions.append((i, j, action))

                #print(f"Peça {state.board.board[i][j]} com coordenadas {i},{j} e temos {len(actions)} ações possíveis: ")
                #for l in actions:
                    #print(l)

        #print("Ação aplicada!")
        return actions

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        new_board = deepcopy(state.board)

        pipe_x, pipe_y, new_orientation = action

        pipe_type = new_board.board[pipe_x][pipe_y][0]
        updated_pipe = pipe_type + new_orientation
        new_board.board[pipe_x][pipe_y] = updated_pipe

        return PipeManiaState(new_board)


    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        
        board = state.board

        for i in range(0, board.row_count):
            for j in range(0, board.col_count):

                if not board.is_connected(i,j):
                    return False

        return True


    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*. Retorna número de pipes desconectados"""

        board = node.state.board
      
        disconnected_pipes = sum(not board.is_connected(i, j) for i in range(board.row_count) for j in range(board.col_count))
        #print(f"Este estado tem {disconnected_pipes} pipes desconectados")

        return disconnected_pipes

    # TODO: outros metodos da classe


if __name__ == "__main__":
    
    input_board = Board.parse_instance() 
    problem = PipeMania(input_board)
    
    solution = greedy_search(problem)
    if solution is not None:
       solution.state.board.print_board()

    
    




