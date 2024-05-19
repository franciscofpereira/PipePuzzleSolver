# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 33:
# 107502 Francisco Ferro Pereira
# 104182 Tiago Romão

import sys
from copy import deepcopy
import numpy as np

sys.path.append('/home/francisco/Documents/ProjetoIA/src/')
from Visualizador.visualizer2 import visualizer

from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
    depth_limited_search
)

class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        result = [tuple(self.board.board[i]) for i in range(0, self.board.row_count-1)]
        self.hash = hash(tuple(result))   
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id
    
    def __eq__(self, other_state):
        return isinstance(other_state, PipeManiaState) and other_state.board == self.board

    def __hash__(self) -> int:
        return self.hash       
                
        
    
    # TODO: outros metodos da classe


class Board:
    """ Representação interna de uma grelha de PipeMania. """
    def __init__(self, board):
        self.board = board
        self.row_count, self.col_count = self.board.shape
        self.total_pipe_ends = self.get_total_pipe_ends()
        self.domain = self.initialize_domains()
      
        
    def __eq__(self, other_board):
        return isinstance(other_board, Board) and (other_board.board == self.board).all()

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
    
        board_matrix = np.array(board)

        return Board(board_matrix)
    
    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """ Devolve os valores imediatamente acima e abaixo,
        respectivamente. """
        vertical_up = None
        vertical_down = None

        if row > 0:
            vertical_up = self.board[row - 1, col]

        if row < self.row_count - 1:  # Check against the upper bound of rows
            vertical_down = self.board[row + 1, col]

        return (vertical_up, vertical_down)


    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """ Devolve os valores imediatamente à esquerda e à direita,
        respectivamente. """
        horizontal_left = None
        horizontal_right = None

        if col > 0:
            horizontal_left = self.board[row, col - 1]

        if col < self.col_count - 1: 
            horizontal_right = self.board[row, col + 1]

        return (horizontal_left, horizontal_right)
    
    def get_value(self, row: int, col: int):
        """ Devolve o valor nas coordenadas dadas como argumento """
        return self.board[row,col]
    
    def print_board(self):
        """Imprime a grelha do tabuleiro"""
        for row in range(self.row_count):
            for col in range(self.col_count):
                if col == self.col_count-1:
                    print(f"{self.board[row,col]}\n", end='')  
                else:
                    print(f"{self.board[row,col]}\t", end='')  

    def print_board_with_colors(self):
        """Imprime a grelha do tabuleiro"""
        for row in range(self.row_count):
            for col in range(self.col_count):
                if len(self.domain[row][col]) == 1:
                    # Print in green
                    print("\033[32m" + f"{self.board[row][col]}", end='')
                else:
                    # Print in white
                    print("\033[0m" + f"{self.board[row][col]}", end='')

                if col == self.col_count - 1:
                    print()  # New line at the end of each row
                else:
                    print('\t', end='')  # Tab separator for columns
                
        
    def translate_pipe(self, row: int, col: int):
        """ Devolve um tuplo do formato (CIMA, BAIXO, ESQUERDA, DIREITA) com entradas a 1 nas direções em
        em que o pipe é aberto e com entradas a 0 nas direções em que o pipe é fechado """

        pipe_type, orientation = self.board[row,col]

        pipe_translations = {
        'F': {'C': (1, 0, 0, 0), 'B': (0, 1, 0, 0), 'E': (0, 0, 1, 0), 'D': (0, 0, 0, 1)},
        'B': {'C': (1, 0, 1, 1), 'B': (0, 1, 1, 1), 'E': (1, 1, 1, 0), 'D': (1, 1, 0, 1)},
        'V': {'C': (1, 0, 1, 0), 'B': (0, 1, 0, 1), 'E': (0, 1, 1, 0), 'D': (1, 0, 0, 1)},
        'L': {'V': (1, 1, 0, 0), 'H': (0, 0, 1, 1)}
        }

        return pipe_translations[pipe_type][orientation]
    

    def translate_pipe_no_coordinates(self, pipe: str):
        """ Devolve um tuplo do formato (CIMA, BAIXO, ESQUERDA, DIREITA) com entradas a 1 nas direções em
        em que o pipe é aberto e com entradas a 0 nas direções em que o pipe é fechado """

        pipe_type = pipe[0] 
        orientation = pipe[1]

        pipe_translations = {
        'F': {'C': (1, 0, 0, 0), 'B': (0, 1, 0, 0), 'E': (0, 0, 1, 0), 'D': (0, 0, 0, 1)},
        'B': {'C': (1, 0, 1, 1), 'B': (0, 1, 1, 1), 'E': (1, 1, 1, 0), 'D': (1, 1, 0, 1)},
        'V': {'C': (1, 0, 1, 0), 'B': (0, 1, 0, 1), 'E': (0, 1, 1, 0), 'D': (1, 0, 0, 1)},
        'L': {'V': (1, 1, 0, 0), 'H': (0, 0, 1, 1)}
        }

        return pipe_translations[pipe_type][orientation]


    def is_connected(self, row: int, col: int):
        """ Retorna um tuplo no formato (BOOL, INT) em que BOOL é True se a peça está conectada e False caso contrário.
        O valor de INT indica o número de aberturas da peça que estão conectadas. """
        
        connected_ends = 0
        pipe_type = self.board[row, col][0]
        
        neighbours = self.get_neighbors(row, col)

        for neighbour in neighbours:    
            x_offset = neighbour[0] - row 
            y_offset = neighbour[1] - col 
            if self.check_compatibility(row, col, x_offset, y_offset):
                connected_ends += 1

        if pipe_type == 'F':
            return (connected_ends == 1, connected_ends)
        elif pipe_type == 'B':
            return (connected_ends == 3, connected_ends)
        elif pipe_type == 'V':
            return (connected_ends == 2, connected_ends)
        elif pipe_type == 'L':
            return (connected_ends == 2, connected_ends)
        else:
            return (False, connected_ends)


    def check_compatibility(self, row: int, col: int, x_offset: int, y_offset):
        
        ''' Verifica se o pipe é compatível com a peça adjacente. A posição da peça adjacente é expressa por x_offset
        quando à esquerda ou direita ou por y_offset quando acima ou abaixo.'''
        
        p1 = self.translate_pipe(row,col)
        p2 = self.translate_pipe(row + x_offset, col + y_offset)  if  (0 <= row + x_offset < self.row_count and 0 <= col + y_offset < self.col_count) else (0,0,0,0)

        pipe1_type = self.board[row, col][0]
        pipe2_type = self.board[row+x_offset,col+y_offset][0] if  (0 <= row + x_offset < self.row_count and 0 <= col + y_offset < self.col_count) else (0,0,0,0)

        # Two close ended pipes are never compatible
        if pipe1_type == 'F' and pipe2_type == 'F':
            return False
        
        offset_tuple = (x_offset, y_offset)

        # UP
        if offset_tuple == (-1,0):
            return p1[0] and p2[1]
        
        # DOWN
        elif offset_tuple == (1,0):
            return p1[1] and p2[0]
        
        # LEFT
        elif offset_tuple == (0,-1):
            return p1[2] and p2[3]
        
        # RIGHT
        elif offset_tuple == (0,1):
            return p1[3] and p2[2]

        return False
    

    def get_neighbors(self, row: int, col: int):
        ''' Retorna uma lista com tuplos com as coordenadas dos pipes vizinhos nas direções em que o pipe tem aberturas '''
        
        pipe = self.translate_pipe(row, col)
        
        neighbours = []

        if pipe[0] and row - 1 >= 0:
            neighbours.append((row - 1, col))
           

        if pipe[1] and row + 1 < self.row_count:
            neighbours.append((row + 1, col))

        if pipe[2] and col - 1 >= 0:
            neighbours.append((row, col - 1))

        if pipe[3] and col + 1 < self.col_count:
            neighbours.append((row, col + 1))

        return neighbours
    
    def get_neighbor_in_direction(self, row: int, col: int, direction: int):

        # UP
        if direction == 0 and row-1 >= 0:
            return (row-1,col)
        # DOWN
        elif direction == 1 and row + 1 < self.row_count:
            return (row+1,col)
        # LEFT
        elif direction == 2 and col - 1 >= 0:
            return (row,col-1)
        
        # RIGHT
        elif direction == 3 and col + 1 < self.col_count:
            return (row,col+1)

        
    def get_non_neighbors(self, row: int, col: int):
        ''' Retorna uma lista com tuplos com as coordenadas dos pipes adjacentes nas direções em que o pipe não tem aberturas '''
        
        pipe = self.translate_pipe(row, col)
        
        neighbours = []

        if not pipe[0] and row - 1 >= 0:
            neighbours.append((row - 1, col))
           

        if not pipe[1] and row + 1 < self.row_count:
            neighbours.append((row + 1, col))

        if not pipe[2] and col - 1 >= 0:
            neighbours.append((row, col - 1))

        if not pipe[3] and col + 1 < self.col_count:
            neighbours.append((row, col + 1))

        return neighbours

    def is_neighbor(self, row: int, col: int, neighbor_row: int, neighbor_col: int):

        neighbours = self.get_neighbors(row,col)

        return (neighbor_row,neighbor_col) in neighbours
    
    def get_total_pipe_ends(self):
        ''' Retorna o número de pipe ends no tabuleiro. '''

        pipe_ends = 0
        for row in range(self.row_count):
            for col in range(self.col_count):

                pipe_type = self.board[row,col][0]

                if pipe_type == 'F':
                    pipe_ends += 1

                elif pipe_type == 'B':
                    pipe_ends += 3
                
                elif pipe_type == 'V' or pipe_type == 'L':
                    pipe_ends += 2

        return pipe_ends


    def initialize_domains1(self):

        domain = [[[] for _ in range(self.col_count)] for _ in range(self.row_count)]

        for i in range(0,self.row_count):
            for j in range(0, self.col_count):

                pipe_type = self.board[i,j][0]

                if pipe_type == 'F':
                    domain[i,j] = ['FC','FB','FE','FD']
                if pipe_type == 'V':
                    domain[i,j] = ['VC','VB','VE','VD']
                if pipe_type == 'B':
                    domain[i,j] = ['BC','BB','BE','BD']
                if pipe_type == 'L':
                    domain[i,j] = ['LH','LV']
                
        return domain

    def edge_constraint(self, row: int, col: int):

        ''' Verifica se o pipe em questão respeita as restrições das bordas do tabuleiro '''
        
        pipe = self.translate_pipe(row, col)

        # top row
        if row == 0 and pipe[0]:
            return False
        # bottom row
        if row == self.row_count -1 and pipe[1]:
            return False
        # leftmost col
        if col == 0 and pipe[2]:
            return False
        # rightmost col
        if col == self.col_count - 1 and pipe[3]:
            return False

        return True


    def initialize_domains(self):

        domain = np.empty((self.row_count, self.col_count), dtype=object)

        for i in range(0,self.row_count):
            for j in range(0, self.col_count):

                pipe_type = self.board[i,j][0]

                if pipe_type == 'F':

                    if (i,j) == (0,0):
                        domain[i,j] = ['FB','FD']
                    elif (i,j) == (0,self.col_count-1):
                        domain[i,j] = ['FB','FE']
                    elif (i,j) == (self.row_count-1,0):
                        domain[i,j] = ['FC','FD']
                    elif (i,j) == (self.row_count-1,self.col_count-1):
                        domain[i,j] = ['FC','FE']
                    elif i == 0:
                        domain[i,j] = ['FB','FE','FD']
                    elif i == self.row_count-1:
                        domain[i,j] = ['FC','FE','FD']
                    elif j == 0:
                        domain[i,j] = ['FB','FC','FD']
                    elif j == self.col_count-1:
                        domain[i,j] = ['FB','FC','FE'] 
                    else:
                        domain[i,j] = ['FC','FB','FE','FD']
                
                if pipe_type == 'V':

                    if (i,j) == (0,0):
                        domain[i,j] = ['VB']
                    elif (i,j) == (0,self.col_count-1):
                        domain[i,j] = ['VE']
                    elif (i,j) == (self.row_count-1,0):
                        domain[i,j] = ['VD']
                    elif (i,j) == (self.row_count-1,self.col_count-1):
                        domain[i,j] = ['VC']
                    elif i == 0:
                        domain[i,j] = ['VB','VE']
                    elif i == self.row_count-1:
                        domain[i,j] = ['VC','VD']
                    elif j == 0:
                        domain[i,j] = ['VB','VD']
                    elif j == self.col_count-1:
                        domain[i,j] = ['VE','VC']
                    else:
                        domain[i,j] = ['VC','VB','VE','VD']
                    
                if pipe_type == 'B':
                    
                    if i == 0:
                        domain[i,j] = ['BB']
                    elif i == self.row_count-1:
                        domain[i,j] = ['BC']
                    elif j == 0:
                        domain[i,j] = ['BD']
                    elif j == self.col_count-1:
                        domain[i,j] = ['BE']
                    else:
                        domain[i,j] = ['BC','BB','BE','BD']

                if pipe_type == 'L':
                    
                    if i == 0:
                        domain[i,j] = ['LH']
                    elif i == self.row_count-1:
                        domain[i,j] = ['LH']
                    elif j == 0:
                        domain[i,j] = ['LV']
                    elif j == self.col_count-1:
                        domain[i,j] = ['LV']
                    else:
                        domain[i,j] = ['LH','LV']
                
                self.board[i][j] = domain[i,j][0]

        return domain

    def edge_constraint(self, row: int, col: int):

        ''' Verifica se o pipe em questão respeita as restrições das bordas do tabuleiro '''
        
        pipe = self.translate_pipe(row, col)

        # top row
        if row == 0 and pipe[0]:
            return False
        # bottom row
        if row == self.row_count -1 and pipe[1]:
            return False
        # leftmost col
        if col == 0 and pipe[2]:
            return False
        # rightmost col
        if col == self.col_count - 1 and pipe[3]:
            return False

        return True
    

    def corner_constraints(self, row: int, col:int):

        ''' Verifica se o pipe em questão respeita as restrições nos cantos do tabuleiro '''
        pipe = self.translate_pipe(row,col)
        
        pipe_type = self.board[row,col][0]

        if pipe_type not in ('F','V'):
            return False

        # top left 
        if (row,col) == (0,0) and (pipe[0] or pipe[2]):
            return False

        # top right
        if (row,col) == (0,self.col_count-1) and (pipe[0] or pipe[3]):
            return False

        # bottom left
        if (row, col) == (self.row_count-1,0) and (pipe[1] or pipe[2]):
            return False
        
        # bottom right
        if (row,col) == (self.row_count-1, self.col_count-1) and (pipe[1] or pipe[3]):
            return False
        
        return True


    def iterate_borders1(self):
        rows, cols = (self.col_count-1, self.row_count-1)
        for layer in range((min(rows, cols) + 1) // 2):
            # Iterate over the top border
            for col in range(layer, cols - layer):
                print(self.board[layer, col])
            # Iterate over the right border
            for row in range(layer + 1, rows - layer):
                print(self.board[row, cols - layer - 1])
            # Iterate over the bottom border
            for col in range(cols - layer - 2, layer - 1, -1):
                print(self.board[rows - layer - 1, col])
            # Iterate over the left border
            for row in range(rows - layer - 2, layer, -1):
                print(self.board[row,layer])

    def iterate_borders(self):
        rows, cols = self.row_count, self.col_count
        coordinates = []
        for layer in range((min(rows, cols) + 1) // 2):
            # Iterate over the top border
            for col in range(layer, cols - layer):
                coordinates.append((layer, col))
            # Iterate over the right border
            for row in range(layer + 1, rows - layer):
                coordinates.append((row, cols - layer - 1))
            # Iterate over the bottom border
            for col in range(cols - layer - 2, layer - 1, -1):
                coordinates.append((rows - layer - 1, col))
            # Iterate over the left border
            for row in range(rows - layer - 2, layer, -1):
                coordinates.append((row, layer))

            result = [self.simplify_domains(c[0], c[1]) for c in coordinates]
            #result = [(c[0], c[1]) for c in coordinates]
        



    def simplify_domains(self, row: int, col: int):
        ''' Função que ao receber as coordenadas de um pipe simplifica o seu domínio e o dos seus vizinhos, propagando restrições '''
        # pipe is optimal
        if len(self.domain[row, col]) == 1:

            neighbours = self.get_neighbors(row, col)

            # removes all incompatible values from neighbouring domains
            for n in neighbours:
                temp_list = self.get_neighbors(n[0], n[1]).copy()
                for val in self.domain[n[0], n[1]]:
                    self.set_pipe(n[0], n[1], val)
                    if not self.check_compatibility(row, col, n[0] - row, n[1] - col):
                        temp_list.remove(val)
                if len(temp_list) != len(self.domain[n[0]][n[1]]):
                    self.domain[n[0]][n[1]] = temp_list


            non_neighbours = self.get_non_neighbors(row, col)
            # removes all values from non neighbouring domains that point towards the optimal pipe
            for n in non_neighbours:
                temp_list = self.get_neighbors(n[0], n[1]).copy()
                for val in self.domain[n[0], n[1]]:
                    self.set_pipe(n[0], n[1], val)
                    if self.points_towards(n[0], n[1], row, col):
                        temp_list.remove(val)
                if len(temp_list) != len(self.domain[n[0]][n[1]]):
                    self.domain[n[0]][n[1]] = temp_list

        # pipe is not optimal
        else:

            directions = self.needs_connection(row, col)

            # checks if pipe has an obligatory connection, if so removes neighbors incompatible with that connection
            if len(directions) > 0:

                for d in directions:
                    n = self.get_neighbor_in_direction(row, col, d)
                    temp_list = self.domain[n[0]][n[1]].copy()

                    if n is not None:

                        for val in self.domain[n[0], n[1]]:
                            self.set_pipe(n[0], n[1], val)
                            if not self.check_compatibility(row, col, n[0] - row, n[1] - col):
                                temp_list.remove(val)
                        if len(temp_list) != len(self.domain[n[0]][n[1]]):
                            self.domain[n[0]][n[1]] = temp_list

            neighbours = self.get_neighbors(row, col)

            # if the neighbor is optimal and the pipe is not in its neighbors, than we can remove all values from the domain that point to the neighbor
            for n in neighbours:
                temp_list = self.domain[n[0]][n[1]].copy()
                if len(self.domain[n[0], n[1]]) == 1 and (row, col) not in self.get_neighbors(n[0], n[1]):
                    for val in self.domain[row, col]:
                        self.set_pipe(row, col, val)
                        if self.points_towards(row, col, n[0], n[1]):
                            temp_list.remove(val)
                        if len(temp_list) != len(self.domain[n[0]][n[1]]):
                            self.domain[n[0]][n[1]] = temp_list


    def fix_corners(self):
        ''' Corrige a rotação das peças dos cantos do tabuleiro '''
        
        top_left = self.board[0][0]
        top_right = self.board[0][self.col_count-1]
        bottom_left = self.board[self.row_count-1][0]
        bottom_right = self.board[self.row_count-1][self.col_count-1]

        if top_left[0] == 'V':
            self.board[0][0] = 'VB'
        else:
            
            if not self.is_connected(0,0)[0]:
                self.board[0][0] = 'FB'

        if top_right[0] == 'V':
            self.board[0][self.col_count-1] = 'VE'
        else:
            if not self.is_connected(0,self.col_count-1)[0]:
                self.board[0][self.col_count-1] = 'FB'

        if bottom_left[0] == 'V':
            self.board[self.row_count-1][0] = 'VD'
        else:
            if not self.is_connected(self.row_count-1,0)[0]:
                self.board[self.row_count-1][0] = 'FC'

        if bottom_right[0] == 'V':
            self.board[self.row_count-1][self.col_count-1] = 'VC'
        else:
            if not self.is_connected(self.row_count-1,self.col_count-1)[0]:
                self.board[self.row_count-1][self.col_count-1] = 'FC'

        return self


    def fix_edges(self):
        
        # bottom and top row
        for col in range(0, self.col_count):
            top_row = self.board[0][col]
            bottom_row = self.board[self.row_count - 1][col]

            if col > 0 and col < self.col_count - 1:
                if top_row == 'FC':
                    self.board[0][col] = 'FB'
                elif top_row == 'VC' or top_row == 'VD':
                    self.board[0][col] = 'VB'
                elif top_row[0] == 'B':
                    self.board[0][col] = 'BB'
                elif top_row[0] == 'L':
                    self.board[0][col] = 'LH'

                if bottom_row == 'FB':
                    self.board[self.row_count - 1][col] = 'FC'
                elif bottom_row == 'VB'or bottom_row == 'VE':
                    self.board[self.row_count - 1][col] = 'VC'
                elif bottom_row[0] == 'B':
                    self.board[self.row_count - 1][col] = 'BC'
                elif bottom_row[0] == 'L':
                    self.board[self.row_count - 1][col] = 'LH'

        # left and right columns            
        for row in range(0, self.row_count):
            left_col = self.board[row][0]
            right_col = self.board[row][self.col_count - 1]

            if row > 0 and row < self.row_count - 1:
                if left_col == 'FE':
                    self.board[row][0] = 'FB'
                elif left_col == 'VE' or left_col == 'VC':
                    self.board[row][0] = 'VB'
                elif left_col[0] == 'B':
                    self.board[row][0] = 'BD'
                elif left_col[0] == 'L':
                    self.board[row][0] = 'LV'
                
                if right_col == 'FD':
                    self.board[row][self.col_count - 1] = 'FB'
                elif right_col[0] == 'VD' or right_col == 'VB':
                    self.board[row][self.col_count - 1] = 'VC'
                elif right_col[0] == 'B':
                    self.board[row][self.col_count - 1] = 'BE'
                elif right_col[0] == 'L':
                    self.board[row][self.col_count - 1] = 'LV'

        self.fix_corners()

        return self
    
    def points_towards(self, row1: int, col1: int, row2: int, col2: int ):

        ''' Função que retorna True se o pipe1 aponta para o pipe2 '''
        
        relative_pos = (row2-row1, col2 - col1)
        p1 = self.translate_pipe(row1,col1)
        
        # UP
        if relative_pos == (-1, 0) and p1[0]:
            return True
        
        # DOWN
        elif relative_pos == (1, 0) and p1[1]:
            return True
        
        # LEFT
        elif relative_pos == (0, -1) and p1[2]:
            return True
        
        # RIGHT
        elif relative_pos == (0, 1) and p1[3]: 
            return True

        return False
    
    def needs_connection(self, row: int, col: int):
        '''Função que verifica se todos os valores no domínio do pipe têm aberturas numa direção comum, retornando os indices dessas direções
        nos tuplos do formato (CIMA, BAIXO, ESQUERDA, DIREITA) num array'''

        domain = self.domain[row,col]
        domain_translation = [self.translate_pipe_no_coordinates(val) for val in domain]

        directions = []

        # Needs connection UP
        if all(p[0] for p in domain_translation):
            directions.append(0)
        # Needs connection DOWN
        if all(p[1] for p in domain_translation):
            directions.append(1)
        # Needs connection LEFT
        if all(p[2] for p in domain_translation):
            directions.append(2)
        # Needs connection RIGHT
        if all(p[3] for p in domain_translation):
            directions.append(3)

        return directions


    def force_action(self, row: int, col: int):
        ''' Função que aplica a ação para peças que têm apenas um possível valor. '''
        self.board[row,col] = self.domain[row,col][0]

        return self
    
    def set_pipe(self, row: int, col: int, pipe: str):

        ''' Função que coloca um determinado pipe numa determinada posição do tabuleiro '''
        self.board[row,col] = pipe

        return self

    def check_constraints(self, row: int, col: int, neighbour_row: int, neighbour_col: int):
        return self.check_compatibility(row, col, neighbour_row-row, neighbour_col-col)
        


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

                if len(state.board.domain[i,j]) == 1:
                    continue

                for value in state.board.domain[i,j]:

                    if value != state.board.board[i][j]:
                        actions.append((i,j,value[1]))
                
                #print(f"Peça {state.board.board[i][j]} com coordenadas {i},{j} e temos {len(actions)} ações possíveis: ")
                #for l in actions:
                    #print(l)

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
        all_connected = all(board.is_connected(i, j)[0] for i in range(board.row_count) for j in range(board.col_count))

        return all_connected


    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*. Retorna número de pipe ends desconectadas"""

        board = node.state.board
        connected_pipe_ends = sum(board.is_connected(i, j)[1] for i in range(board.row_count) for j in range(board.col_count))
    
        return board.total_pipe_ends - connected_pipe_ends

    # TODO: outros metodos da classe



if __name__ == "__main__":
    
    input_board = Board.parse_instance() 

    input_board.fix_edges()

    simplified_board = input_board.iterate_borders()

    #arc_consistency = input_board.ac3()

    #if arc_consistency == False:
     #  print("There is no solution to this problem")
     #  exit(1)

    #simplified_board.print_board_with_colors()
    #visualizer(simplified_board.board, None)
    

    problem = PipeMania(simplified_board)
    state = PipeManiaState(simplified_board)
    #solution =  depth_limited_search(problem, input_board.row_count**2)
   
    while(not problem.goal_test(simplified_board)):
        simplified_board.iterate_borders()

   
   
    #solution = astar_search(problem)
    #if solution is not None:
     #   print("Solution: ") 'VE'])
    state.board.print_board()
     #   solution.state.board.print_board()
    visualizer(state.board.board, None)


    
    
    




