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
from Visualizador.visualizer3 import visualizer

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



pipe_translations = {
        'F': {'C': (1, 0, 0, 0), 'B': (0, 1, 0, 0), 'E': (0, 0, 1, 0), 'D': (0, 0, 0, 1)},
        'B': {'C': (1, 0, 1, 1), 'B': (0, 1, 1, 1), 'E': (1, 1, 1, 0), 'D': (1, 1, 0, 1)},
        'V': {'C': (1, 0, 1, 0), 'B': (0, 1, 0, 1), 'E': (0, 1, 1, 0), 'D': (1, 0, 0, 1)},
        'L': {'V': (1, 1, 0, 0), 'H': (0, 0, 1, 1)}
        }

possible_values = {
        'F': ['FC','FB', 'FE', 'FD'],
        'B': ['BC','BB', 'BE', 'BD'],
        'V': ['VC','VB', 'VE', 'VD'],
        'L': ['LV','LH'],
        } 



class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        result = [tuple(map(tuple, row)) for row in self.board.board]
        self.hash = hash(tuple(result))   
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id
    
    def __eq__(self, other_state):
        return isinstance(other_state, PipeManiaState) and other_state.board == self.board

    def __hash__(self) -> int:
        return self.hash       
                
          
class Board:
    """ Representação interna de uma grelha de PipeMania. """
    def __init__(self, board):
        self.board = board
        self.row_count, self.col_count = len(self.board), len(self.board[0])
        self.optimal = 0
    
    @property
    def num_pipes_to_visit(self):
        return (self.board[:, :, 1] == 0).sum()


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
            row_with_flags = [(pipe, 0) for pipe in row]
            board.append(row_with_flags)
    
        board_matrix = np.array(board, dtype=object)

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
                    print(f"{self.board[row,col][0]}\n", end='')  
                else:
                    print(f"{self.board[row,col][0]}\t", end='')  


    def print_board_with_colors(self):
        """Imprime a grelha do tabuleiro"""
        for row in range(self.row_count):
            for col in range(self.col_count):
                if self.board[row,col][1] == 2:
                    # Print in green
                    print("\033[32m" + f"{self.board[row][col][0]}", end='')
                
                elif self.board[row,col][1] == 0:
                    # Print in white
                    print("\033[0m" + f"{self.board[row][col][0]}", end='')
                else:
                    # Print in yellow
                    print("\033[33m" + f"{self.board[row][col][0]}", end='')
                if col == self.col_count - 1:
                    print()  
                else:
                    print('\t', end='')  



















    def translate_pipe(self, pipe: str):
        """ Devolve um tuplo do formato (CIMA, BAIXO, ESQUERDA, DIREITA) com entradas a 1 nas direções em
        em que o pipe é aberto e com entradas a 0 nas direções em que o pipe é fechado """
        return pipe_translations[pipe[0]][pipe[1]]




    def is_connected(self, row: int, col: int, pipe: str):
        """ Retorna um tuplo no formato (BOOL, INT) em que BOOL é True se a peça está conectada e False caso contrário.
        O valor de INT indica o número de aberturas da peça que estão conectadas. """
        
        connected_ends = 0
        pipe_type = pipe[0]
        
        neighbours = self.get_neighbors(row, col, pipe)

        for n in neighbours:    
            if self.check_compatibility(row, col, pipe, n[0], n[1], self.board[n[0],n[1]][0]):
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


    def get_neighbors(self, row: int, col: int, pipe: str):
        ''' Retorna uma lista com tuplos com as coordenadas dos pipes vizinhos nas direções em que o pipe tem aberturas '''
        
        pipe = self.translate_pipe(pipe)
        
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
    

    def get_adjacent(self, row: int, col: int):

        ''' Retorna uma lista com tuplos com as coordenadas dos pipes adjacentes '''
        adjacent = []

        # UP
        if row-1 >= 0:
            adjacent.append((row-1,col))
        # DOWN
        if row + 1 < self.row_count:
             adjacent.append((row+1,col))
        # LEFT
        if col - 1 >= 0:
            adjacent.append((row, col-1))
        # RIGHT
        if col + 1 < self.col_count:
            adjacent.append((row, col+1))
        
        return adjacent
                                    
    

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


    def neighbors_are_optimal(self, row: int, col: int):

        neighbors = self.get_neighbors(row,col)
        all_optimal = True

        for (x,y) in neighbors:
            if self.board[x,y][1] != 2:
                all_optimal = False
                return all_optimal
            
        return all_optimal


    def get_non_neighbors(self, row: int, col: int, pipe: str):
        ''' Retorna uma lista com tuplos com as coordenadas dos pipes adjacentes nas direções em que o pipe não tem aberturas '''
        
        pipe = self.translate_pipe(pipe)
        
        non_neighbours = []

        if not pipe[0] and row - 1 >= 0:
            non_neighbours.append((row - 1, col))
           
        if not pipe[1] and row + 1 < self.row_count:
            non_neighbours.append((row + 1, col))

        if not pipe[2] and col - 1 >= 0:
            non_neighbours.append((row, col - 1))

        if not pipe[3] and col + 1 < self.col_count:
            non_neighbours.append((row, col + 1))

        return non_neighbours

    
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
        
        return self

  
    def check_constraints(self, row: int, col: int, pipe: str):

        """
        Check constraints for the pipe at the given position.
        
        Parameters:
        - row: Row index of the pipe.
        - col: Column index of the pipe.
        - pipe: Pipe value we are considering
        
        Returns:
        - 2 if the pipe is in its optimal configuration.
        - 1 if we are uncertain about the optimality.
        - 0 if the pipe is in an incorrect configuration.
    """
        
        # Pipe is on the edge of the board
        if row == 0 or col == 0 or row == self.row_count - 1 or col == self.col_count - 1:
            # Pipe configuration must meet edge constraints
            if not self.edge_constraint(row, col, pipe):
                return 0
            
            # Adjacent pipes
            adjacent = self.get_adjacent(row, col)

            for (a_row, a_col) in adjacent:
                adjacent_confidence = self.board[a_row, a_col][1]
                adjacent_pipe = self.board[a_row,a_col][0]

                # If adjacent pipe is on the edge and optimal
                if (a_row == 0 or a_col == 0 or a_row == self.row_count - 1 or a_col == self.col_count - 1) and adjacent_confidence == 2:
                   
                    if self.points_towards(a_row, a_col, row, col, adjacent_pipe):
                        # they match
                        if self.check_compatibility(row, col, pipe, a_row, a_col, adjacent_pipe):
                            return 2
                        # pipe does not point towards optimal adjacent pipe
                        else:
                            return 0
                    else:
                        # adjacent pipe does not point to pipe but pipe points to adjacent
                        if self.points_towards(row, col, a_row, a_col, pipe):
                            return 0
                        # both don't point to each other
                        else:

                            if pipe[0] != 'F':
                                return 2
                            else:
                                return 1

                # If adjacent pipe is optimal and not on the edge
                elif adjacent_confidence == 2:
                    if self.points_towards(a_row, a_col, row, col, adjacent_pipe): 
                        if not self.check_compatibility(row, col, pipe, a_row, a_col, adjacent_pipe):
                            return 0
                    else:
                        if self.points_towards(row, col, a_row, a_col, pipe):
                           return 0
                        
                elif adjacent_confidence == 1:
                    
                    if self.points_towards(a_row, a_col, row, col, adjacent_pipe):
                        if not self.check_compatibility(row, col, pipe, a_row, a_col, adjacent_pipe):
                            return 0
                    else:
                        if self.points_towards(row,col,a_row,a_col,pipe):
                            return 0

            # If no clear optimal or incorrect configuration, return uncertain
            return 1
                


        # Pipe is not on the edge
        adjacent = self.get_adjacent(row, col)
        
        # Assume pipe is optimal unless proven otherwise
        optimal = True
        uncertain = False
        num_pipe_ends = self.get_num_pipe_ends(pipe)

        for (a_row, a_col) in adjacent:
            adjacent_pipe_confidence = self.board[a_row, a_col][1]
            adjacent_pipe = self.board[a_row,a_col][0]
            connected_ends = 0
            
            # Adjacent pipe is optimal
            if adjacent_pipe_confidence == 2:
                # Check if adjacent pipe points towards current pipe
                if self.points_towards(a_row, a_col, row, col, adjacent_pipe):
                    # Check compatibility between pipes
                    if not self.check_compatibility(row, col, pipe, a_row, a_col, adjacent_pipe):
                        return 0  # Not compatible, so incorrect
                    else:
                        connected_ends += 1
                        if connected_ends == num_pipe_ends:
                            return 2
                        
               # If adjacent doesn't point to current but current points to adjacent, it's wrong
                elif self.points_towards(row, col, a_row, a_col, pipe):
                    return 0
                    
            elif adjacent_pipe_confidence == 1:
                uncertain = True
                if self.points_towards(a_row, a_col, row, col, adjacent_pipe):
                    if not self.check_compatibility(row, col, pipe, a_row, a_col, adjacent_pipe):
                        return 0

        # If there was an uncertain adjacent pipe, current pipe is uncertain
        if uncertain:
            return 1
        
        # If all adjacent pipes are optimal and compatible, current pipe is optimal
        if optimal:
            return 2

        # Default to uncertain if no conclusive determination
        return 1


    def edge_constraint(self, row: int, col: int, pipe: str):

        ''' Verifica se o pipe em questão respeita as restrições das bordas do tabuleiro '''
        
        pipe = self.translate_pipe(pipe)

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
    

    def corner_constraints(self, row: int, col: int, pipe: str):

        ''' Verifica se o pipe em questão respeita as restrições nos cantos do tabuleiro '''
        pipe = self.translate_pipe(pipe)
        
        pipe_type = pipe[0]

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
    
    def check_compatibility(self, row1: int, col1: int, pipe1: str, row2: int, col2: int, pipe2: str):
        
        ''' Verifica se o pipe1 é compatível com o pipe2. A posição do pipe2 em relação ao pipe1 é calculada 
        e guardada no offset tuple. Retorna True se forem compatíveis e False caso contrário '''
        
        p1 = self.translate_pipe(pipe1)
        p2 = self.translate_pipe(pipe2)    

        pipe1_type = pipe1[0]
        pipe2_type = pipe2[0]

        # Two close ended pipes are never compatible
        if pipe1_type == 'F' and pipe2_type == 'F':
            return False
        
        offset_tuple = (row2-row1, col2-col1)

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
    
    def get_possible_values(self, row: int, col: int):
        
        pipe_type = self.board[row,col][0][0]
        return possible_values[pipe_type]
        
    def points_towards(self, row1: int, col1: int, row2: int, col2: int, pipe1: str ):

        ''' Função que retorna True se o pipe1 aponta para o pipe2 '''
        
        relative_pos = (row2-row1, col2 - col1)
        p1 = self.translate_pipe(pipe1)
        
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
    


    def get_num_possible_pipes(self, pipe: str):

        ''' Retorna o número do mesmo tipo que tem abertura na mesma direção que o pipe atual '''
        
        pipe_type = pipe[0]
        pipe_orientation = pipe[1]
        
        pipe_translations = [self.translate_pipe(pipe) for pipe in possible_values[pipe_type]]

        if pipe_orientation == 'C':
            return sum(p[0] for p in pipe_translations)
        elif pipe_orientation == 'B':
            return sum(p[1] for p in pipe_translations)
        elif pipe_orientation == 'E':
            return sum(p[2] for p in pipe_translations)
        elif pipe_orientation == 'D':
            return sum(p[3] for p in pipe_translations)
    
        return 0
    
    def get_num_pipe_ends(self, pipe: str):

        pipe_type = pipe[0]

        if pipe_type == 'F':
            return 1
        elif pipe_type == 'B':
            return 3
        else:
            return 2
    
    
    def fix_edges(self):
        
        # bottom and top row
        for col in range(0, self.col_count):
            top_row = self.board[0,col]
            bottom_row = self.board[self.row_count - 1, col]

            if col > 0 and col < self.col_count - 1:

                if top_row[0][0] == 'F':
                    self.board[0, col] = ('FB',1)
                elif top_row[0][0] == 'V':
                    self.board[0, col] = ('VB',1)
                elif top_row[0][0] == 'B':
                    self.board[0, col] = ('BB',2)
                    self.optimal += 1
                elif top_row[0][0] == 'L':
                    self.board[0, col] = ('LH', 2)
                    self.optimal += 1
                
                if bottom_row[0][0] == 'F':
                    self.board[self.row_count - 1,col] = ('FC',1)
                elif bottom_row[0][0] == 'V':
                    self.board[self.row_count - 1, col] = ('VC',1)
                elif bottom_row[0][0] == 'B':
                    self.board[self.row_count - 1,col] = ('BC',2)
                    self.optimal += 1
                elif bottom_row[0][0] == 'L':
                    self.board[self.row_count - 1,col] = ('LH',2)
                    self.optimal += 1
                
        # left and right columns            
        for row in range(0, self.row_count):
            left_col = self.board[row,0]
            right_col = self.board[row,self.col_count - 1]

            if row > 0 and row < self.row_count -1:
                
                if left_col[0][0] == 'F':
                    self.board[row, 0] = ('FB',1)
                elif left_col[0][0] == 'V':
                    self.board[row, 0] = ('VD',1)
                elif left_col[0][0] == 'B':
                    self.board[row, 0] = ('BD',2)
                    self.optimal += 1
                elif left_col[0][0] == 'L':
                    self.board[row, 0] = ('LV',2)
                    self.optimal += 1
                
                if right_col[0][0] == 'F':
                    self.board[row, self.col_count-1] = ('FB',1)
                elif right_col[0][0] == 'V':
                    self.board[row, self.col_count-1] = ('VE',1)
                elif right_col[0][0] == 'B':
                    self.board[row, self.col_count - 1] = ('BE',2)
                    self.optimal += 1
                elif right_col[0][0] == 'L':
                    self.board[row, self.col_count - 1] = ('LV',2)
                    self.optimal += 1
        
        self.fix_corners()

        return self

    def fix_corners(self):

        if self.board[0,0][0][0] == 'V':
            self.board[0,0] = ('VB',2)
            self.optimal += 1
        if self.board[0,0][0][0] == 'F':
            self.board[0,0] = ('FB',1)
        if self.board[0,self.col_count-1][0][0] == 'V':
            self.board[0,self.col_count-1] = ('VE',2)
            self.optimal += 1
        if self.board[0,self.col_count-1][0][0] == 'F':
            self.board[0,self.col_count-1] = ('FB',1)
        if self.board[self.row_count-1,0][0][0] == 'V':
            self.board[self.row_count-1,0] = ('VD',2)
            self.optimal += 1
        if self.board[self.row_count-1,0][0][0] == 'F':
            self.board[self.row_count-1,0] = ('FC',1)
        if self.board[self.row_count-1,self.col_count-1][0][0] == 'V':
            self.board[self.row_count-1,self.col_count-1] = ('VC',2)
            self.optimal += 1
        if self.board[self.row_count-1,self.col_count-1][0][0] == 'F':
            self.board[self.row_count-1,self.col_count-1] = ('FC',1)
        
        return self
    

class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(PipeManiaState(board))
        

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento e pruning a ações que não respeitam as constraints."""
        
        optimal_actions  = []
        uncertain_actions = []
        coordinates_frequency = {}

        for row in range(state.board.row_count):
            for col in range(state.board.col_count):
                

                # 0 
                # 1
                # 2


                #if not (row == 0 or col == 0 or row == state.board.row_count - 1 or col == state.board.col_count - 1):
                #    continue
                # Prioritize pipes that haven't been assigned a value yet
                if state.board.num_pipes_to_visit > 0 and state.board.board[row,col][1] != 0:
                   continue
                # If the pipe is optimal we don't move it
                elif state.board.board[row,col][1] == 2:
                    continue
                
                possible_values = state.board.get_possible_values(row, col)
                current_orientation = state.board.board[row,col][0]

                # gets all possible values for pipe
                for new_orientation in possible_values:

                    # if it satisfies the constraints we can append to the actions array
                    constraint_res = state.board.check_constraints(row, col, new_orientation)
                        
                    if constraint_res == 2:
                            optimal_actions.append((row,col,new_orientation[1]))
                            break
                    
                    elif constraint_res == 1:
                        if new_orientation != current_orientation:
                            uncertain_actions.append((row,col,new_orientation[1]))
                            coordinates_frequency[(row, col)] = coordinates_frequency.get((row, col), 0) + 1

                    else:
                        continue

        # Sort uncertain actions by frequency of coordinates
        uncertain_actions.sort(key=lambda action: coordinates_frequency.get((action[0], action[1]), 0))

        actions = []

        actions.append(len(optimal_actions))
        actions.extend(optimal_actions)
        actions.extend(uncertain_actions)

        return [] if len(optimal_actions) == 0 and len(uncertain_actions) == 0 else [actions] 


    def result(self, state: PipeManiaState, actions):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        new_board = deepcopy(state.board)

        num_optimal_actions = actions[0]

        # if there are no optimal actions applies the first uncertain action
        if num_optimal_actions == 0:
            pipe_x, pipe_y, new_orientation = actions[1]
            pipe_type = new_board.board[pipe_x, pipe_y][0][0]
            updated_pipe = pipe_type + new_orientation
            new_board.board[pipe_x,pipe_y] = (updated_pipe, 1)
            print(f"pipe at {pipe_x,pipe_y} rotated. New value is {updated_pipe}")
            
        else:

            # applies optimal actions 
            for i in range(1,num_optimal_actions+1):

                pipe_x, pipe_y, new_orientation = actions[i]
                pipe_type = new_board.board[pipe_x, pipe_y][0][0]
                updated_pipe = pipe_type + new_orientation
                new_board.board[pipe_x,pipe_y] = (updated_pipe, 2)
                new_board.optimal += 1
                print(f"pipe at {pipe_x,pipe_y} rotated. Optimal value is {updated_pipe}")

        return PipeManiaState(new_board)


    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""

        return state.board.optimal == state.board.row_count**2


    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*. Retorna número de pipe ends desconectadas"""

        board = node.state.board
        connected_pipe_ends = sum(board.is_connected(i, j)[1] for i in range(board.row_count) for j in range(board.col_count))
    
        return board.total_pipe_ends - connected_pipe_ends

    # TODO: outros metodos da classe



if __name__ == "__main__":
    
    input_board = Board.parse_instance() 

    input_board.fix_edges()

    input_board.print_board_with_colors()


    #visualizer(input_board.board, None)
    

    problem = PipeMania(input_board)
    #solution =  depth_limited_search(problem, input_board.row_count**2)
    solution = depth_first_tree_search(problem)
    if solution is not None:
        print("Solution: ")
        solution.state.board.print_board()
        visualizer(solution.state.board.board, None)


    
    
    




