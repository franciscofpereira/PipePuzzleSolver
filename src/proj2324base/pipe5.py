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

pipe_domains = {
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
        self.starting_points = []
        self.starting_points_index = 0

    @property
    def optimal(self):
        return (self.board[:, :, 1] == 1).sum()

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
                if self.board[row,col][1] == 1:
                    # Print in green
                    print("\033[32m" + f"{self.board[row][col][0]}", end='')
                else:
                    # Print in white
                    print("\033[0m" + f"{self.board[row][col][0]}", end='')
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
    
    def all_adjacent_optimal(self, row: int, col: int):

        adjacent = self.get_adjacent(row,col)
        all_optimal = True

        for (x,y) in adjacent:
            if not self.board[x,y][1]:
                all_optimal = False
                return all_optimal
            
        return all_optimal
                                    
    
    def get_possible_values(self ,row1 ,col1 ,pipe1 ,row2 ,col2):

        ''' Função que recebe pipe1 que está ótimo e posição do seu adjacente pipe2 e retorna os valores de pipe2 que encaixam localmente '''
        
        pipe_domain = self.get_domain(row2, col2)
        possible_values = []

        for value in pipe_domain:

            if row2 == 0 or row2 == self.row_count-1 or col2 == 0 or col2 == self.col_count-1:
                
                if not self.edge_constraint(row2, col2, value):
                    continue
            
            if self.check_compatibility(row1, col1, pipe1, row2, col2, value):
                possible_values.append(value)

        return possible_values
        


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
        
        # if both don't point to each other
        if not self.points_towards(row1,col1,row2,col2, pipe1) and not self.points_towards(row2,col2,row1,col1,pipe2):
            return True
        
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
    

    def get_suboptimal_adjacents(self, row1: int, col1: int , pipe1: str):
        
        ''' Função que recebe um pipe e retorna as coordenadas dos adjacentes que estão desconectados '''
        adjacents = self.get_adjacent(row1,col1)
        suboptimal_adjacents = []
        for (row2,col2) in adjacents:
            # if adjacent not optimal we add it to the list
            if not self.board[row2,col2][1]:
                suboptimal_adjacents.append((row2,col2))
        return suboptimal_adjacents


    def get_domain(self, row: int, col: int):
        pipe_type = self.board[row,col][0][0]
        return pipe_domains[pipe_type]
        

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
                    self.board[0, col] = ('FB',0)
                elif top_row[0][0] == 'V':
                    self.board[0, col] = ('VB',0)
                elif top_row[0][0] == 'B':
                    self.board[0, col] = ('BB',1)
                    self.starting_points.append((0,col))
                elif top_row[0][0] == 'L':
                    self.board[0, col] = ('LH', 1)
                    self.starting_points.append((0,col))
               
                
                if bottom_row[0][0] == 'F':
                    self.board[self.row_count - 1,col] = ('FC',0)
                elif bottom_row[0][0] == 'V':
                    self.board[self.row_count - 1, col] = ('VC',0)
                elif bottom_row[0][0] == 'B':
                    self.board[self.row_count - 1,col] = ('BC',1)
                    self.starting_points.append((self.row_count - 1,col))
                elif bottom_row[0][0] == 'L':
                    self.board[self.row_count - 1,col] = ('LH',1)
                    self.starting_points.append((self.row_count - 1,col))
                
        # left and right columns            
        for row in range(0, self.row_count):
            left_col = self.board[row,0]
            right_col = self.board[row,self.col_count - 1]

            if row > 0 and row < self.row_count -1:
                
                if left_col[0][0] == 'F':
                    self.board[row, 0] = ('FB',0)
                elif left_col[0][0] == 'V':
                    self.board[row, 0] = ('VD',0)
                elif left_col[0][0] == 'B':
                    self.board[row, 0] = ('BD',1)
                    self.starting_points.append((row,0))
                elif left_col[0][0] == 'L':
                    self.board[row, 0] = ('LV',1)
                    self.starting_points.append((row,0))
                
                if right_col[0][0] == 'F':
                    self.board[row, self.col_count-1] = ('FB',0)
                elif right_col[0][0] == 'V':
                    self.board[row, self.col_count-1] = ('VE',0)
                elif right_col[0][0] == 'B':
                    self.board[row, self.col_count - 1] = ('BE',1)
                    self.starting_points.append((row, self.col_count - 1))
                elif right_col[0][0] == 'L':
                    self.board[row, self.col_count - 1] = ('LV',1)
                    self.starting_points.append((row, self.col_count - 1))
        
        self.fix_corners()

        return self

    def fix_corners(self):

        if self.board[0,0][0][0] == 'V':
            self.board[0,0] = ('VB',1)
            self.starting_points.append((0,0))
        if self.board[0,0][0][0] == 'F':
            self.board[0,0] = ('FB',0)
        if self.board[0,self.col_count-1][0][0] == 'V':
            self.board[0,self.col_count-1] = ('VE',1)
            self.starting_points.append((0, self.col_count - 1))
        if self.board[0,self.col_count-1][0][0] == 'F':
            self.board[0,self.col_count-1] = ('FB',0)
        if self.board[self.row_count-1,0][0][0] == 'V':
            self.board[self.row_count-1,0] = ('VD',1)
            self.starting_points.append((self.row_count-1, 0))
        if self.board[self.row_count-1,0][0][0] == 'F':
            self.board[self.row_count-1,0] = ('FC',0)
        if self.board[self.row_count-1,self.col_count-1][0][0] == 'V':
            self.board[self.row_count-1,self.col_count-1] = ('VC',1)
            self.starting_points.append((self.row_count - 1, self.col_count - 1))
        if self.board[self.row_count-1,self.col_count-1][0][0] == 'F':
            self.board[self.row_count-1,self.col_count-1] = ('FC',0)
        
        return self
    

    def get_actions_recursively(self, row1: int, col1: int):

        actions = []
        uncertain_actions = []
        
        # optimal pipe
        pipe1 = self.board[row1,col1][0]

        # gets adjacents that are not optimal yet
        suboptimal_adjacents = self.get_suboptimal_adjacents(row1,col1,pipe1)

        if not suboptimal_adjacents:
            return actions

        # for each adjacent disconnected pipe call recursively on its adjacents 
        for (row2,col2) in suboptimal_adjacents:

            possible_values = self.get_possible_values(row1,col1,pipe1,row2,col2)

            # if there is only one possible value that means it's optimal so we call the function recursively on that pipe
            if len(possible_values) == 1:
                self.starting_points.append((row2,col2))

                # removes from uncertain actions if the newly optimal pipe is there
                actions = [action for action in actions if not (action[0] == row2 and action[1] == col2)]

                self.set_pipe(row2,col2,possible_values[0])
                actions.extend(self.get_actions_recursively(row2,col2))

            # if we have more than one possibility we break out of the function and let search decide
            else:
                
                uncertain_actions = [(row2,col2,pipe[1]) for pipe in possible_values]
                actions.extend(uncertain_actions)
                return actions
              
        return actions

    
    def set_pipe(self, row: int, col: int, pipe: str):

        ''' Função que coloca um determinado pipe numa determinada posição do tabuleiro '''
        self.board[row,col] = (pipe,1)

    
    

class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(PipeManiaState(board))
        

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento e faz pruning a ações que não respeitam as constraints."""

        
        if not state.board.starting_points:
            print("ficámos sem starting points")
            return []
        
        if state.board.starting_points_index >= len(state.board.starting_points):
            starting_points_index = 0

        row,col = state.board.starting_points[state.board.starting_points_index]
        actions = state.board.get_actions_recursively(row,col)


        # chooses a new starting point to search paths from the next time
        state.board.starting_points_index += 1
        
        starting_points_copy = [(row,col) for (row,col) in state.board.starting_points if not state.board.all_adjacent_optimal(row,col)]
        state.board.starting_points = starting_points_copy

        return [] if actions == [] else actions


    def result(self, state: PipeManiaState, action):
        
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        new_board = deepcopy(state.board)
        
        row, col, new_orientation = action
        pipe_type = new_board.board[row, col][0][0]
        updated_pipe = pipe_type + new_orientation
        new_board.board[row,col] = (updated_pipe, 1)
        new_board.starting_points.append((row,col))
        print(f"pipe at {row,col} rotated. Optimal value is {updated_pipe}")
        
        return PipeManiaState(new_board)



    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
  
        visited = [[False] * state.board.col_count for _ in range(state.board.row_count)]
        
        for row in range(state.board.row_count):
            for col in range(state.board.col_count):

                if not state.board.is_connected(row,col,state.board.board[row,col][0])[0]:
                    return False

                if not visited[row][col]:
                    # Start DFS traversal from unvisited position
                    cluster_size = self.dfs(state, visited, row, col)
                    
                    # Not goal state if cluster is smaller than the board
                    if cluster_size < state.board.row_count**2:
                        return False  
        
        return True
    

    def dfs(self, state: PipeManiaState, visited, row, col):
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        visited[row][col] = True
        
        cluster_size = 1  
        
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check if neighbor is within bounds and unvisited
            if 0 <= new_row < state.board.row_count and 0 <= new_col < state.board.col_count and not visited[new_row][new_col]:
                if state.board.is_connected(new_row, new_col, state.board.board[new_row,new_col][0])[0]:
                    # Continue DFS traversal recursively
                    cluster_size += self.dfs(state, visited, new_row, new_col)
        
        return cluster_size


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


    
    
    




