# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 33:
# 107502 Francisco Ferro Pereira
# 104182 Tiago Romão

import sys
import os
from copy import deepcopy
import numpy as np
from collections import OrderedDict, deque

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from Visualizador.visualizer import visualizer

from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
    depth_limited_search,
    depth_first_graph_search
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
        self.spiral_order = self.iterate_borders()
        self.next_actions = None
        self.board_size = self.row_count**2
        self.total_pipe_ends = self.get_total_pipe_ends()
        
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
        print()

    def get_domain(self, row: int, col: int):
        pipe_type = self.board[row,col][0][0]
        return pipe_domains[pipe_type]

    
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
    

    def get_suboptimal_adjacents(self, row1: int, col1: int , pipe1: str):
        
        ''' Retorna uma lista com tuplos das coordenadas dos adjacentes que não estão ótimos '''
        adjacents = self.get_adjacent(row1,col1)
        suboptimal_adjacents = []

        for (row2,col2) in adjacents:
            if not self.board[row2,col2][1]:
                suboptimal_adjacents.append((row2,col2))
        return suboptimal_adjacents
    
                            
    def get_possible_values(self ,row1 ,col1 ,pipe1 ,row2 ,col2):

        ''' Função que recebe pipe1 que está ótimo e posição do seu adjacente pipe2 e retorna os valores de pipe2 que encaixam localmente com pipe1 '''
        
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

        # if both don't point to each other
        if not self.points_towards(row1,col1,row2,col2, pipe1) and not self.points_towards(row2,col2,row1,col1,pipe2):
            return True

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
    

    def board_pre_processing(self):

        ''' Função que faz pré-processamento das bordas do tabuleiro e inicializa o dicionário com as ações possíveis '''
        
        # bottom and top row
        for col in range(0, self.col_count):
            top_row = self.board[0,col]
            bottom_row = self.board[self.row_count - 1, col]

            if col > 0 and col < self.col_count - 1:

                if top_row[0][0] == 'B':
                    self.board[0, col] = ('BB',1)
                elif top_row[0][0] == 'L':
                    self.board[0, col] = ('LH', 1)
                    
                if bottom_row[0][0] == 'B':
                    self.board[self.row_count - 1,col] = ('BC',1)   
                elif bottom_row[0][0] == 'L':
                    self.board[self.row_count - 1,col] = ('LH',1)
                   
                
        # left and right columns            
        for row in range(0, self.row_count):
            left_col = self.board[row,0]
            right_col = self.board[row,self.col_count - 1]

            if row > 0 and row < self.row_count -1:
                
                if left_col[0][0] == 'B':
                    self.board[row, 0] = ('BD',1)
                elif left_col[0][0] == 'L':
                    self.board[row, 0] = ('LV',1)
                
                if right_col[0][0] == 'B':
                    self.board[row, self.col_count - 1] = ('BE',1)
                elif right_col[0][0] == 'L':
                    self.board[row, self.col_count - 1] = ('LV',1)
        
        self.fix_corners()

        self.next_actions = OrderedDict()
        edge_coordinates = self.iterate_outer_border()

        # iterates over the outer edge
        for (row1,col1) in edge_coordinates:
            
            # if pipe is optimal and has not optimal adjacent pipes get actions for those adjacent pipes
            if self.board[row1, col1][1] == 1:
                    suboptimal_adjacent = self.get_suboptimal_adjacents(row1, col1, self.board[row1,col1][0])
                    if not suboptimal_adjacent:
                        continue

                    # for each adjacent non optimal pipe we get its possible values and append to the list
                    for (adj_row, adj_col) in suboptimal_adjacent:
                        if (adj_row, adj_col) not in self.next_actions:
                            self.next_actions[(adj_row, adj_col)] = set()
                        
                        possible_values = self.get_possible_values(row1, col1, self.board[row1, col1][0], adj_row, adj_col)
                          
                        for pipe in possible_values:
                            self.next_actions[(adj_row,adj_col)].add((pipe))
                                                   
        return self


    def propagate_constraints(self):

        ''' Função que propaga restrições no tabuleiro. Para cada peça ótima no tabuleiro, olha para as adjacentes e vê os valores possíveis localmente. '''
        
        past_iteration = set()
        current_iteration = set()
        current_iteration.add(0)
        
        # if these two lists are equal, then we have reached a point where we didn't infer anything on the board
        while(past_iteration != current_iteration):

            past_iteration = current_iteration.copy()
            
            # current iteration stores the coordinates of the pipes we could not infer anything on
            current_iteration = set()

            # iterate through board in spirals
            for (row,col) in self.spiral_order:
 
                # if we already have one possible value on the dictionary, we apply it and remove the entry
                if self.next_actions.get((row,col)) and len(self.next_actions.get((row, col))) == 1:
                    self.board[row,col][0] =  next(iter(self.next_actions[(row,col)]))
                    self.board[row,col][1] = 1
                    del self.next_actions[(row,col)]
                    continue

                # if the pipe is optimal
                if self.board[row, col][1] == 1:
                    
                    # if it has adjacent pipes that are not optimal we try to infer something
                    suboptimal_adjacent = self.get_suboptimal_adjacents(row, col, self.board[row,col][0])
                    if not suboptimal_adjacent:
                        continue
                    
                    for (adj_row, adj_col) in suboptimal_adjacent:

                        
                        possible_values = self.get_possible_values(row, col, self.board[row, col][0], adj_row, adj_col)
                        
                        # if there is only one possible value we apply it and consider the pipe optimal
                        if len(possible_values) == 1:
                            self.board[adj_row,adj_col][0] = possible_values[0]
                            self.board[adj_row,adj_col][1] = 1
                            if self.next_actions.get((adj_row, adj_col)):
                                del self.next_actions[(adj_row,adj_col)]
                       
                        # if there is more than one possible value we intersect them with the possible values already stored
                        else:

                            if not self.next_actions.get((adj_row, adj_col)):
                                self.next_actions[(adj_row,adj_col)]  = set(possible_values)
                            else:
                                self.next_actions[(adj_row, adj_col)] = set(possible_values) & self.next_actions[(adj_row, adj_col)]

                            current_iteration.add((adj_row,adj_col))
                
            if not current_iteration:
                break

        if self.optimal == self.board_size:
            return self.board
            
        return self.board            
                        

    def fix_corners(self):

        if self.board[0,0][0][0] == 'V':
            self.board[0,0] = ('VB',1)
        if self.board[0,self.col_count-1][0][0] == 'V':
            self.board[0,self.col_count-1] = ('VE',1)
        if self.board[self.row_count-1,0][0][0] == 'V':
            self.board[self.row_count-1,0] = ('VD',1)
        if self.board[self.row_count-1,self.col_count-1][0][0] == 'V':
            self.board[self.row_count-1,self.col_count-1] = ('VC',1)
        
        return self
    


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

        return coordinates
    

    def iterate_outer_border(self):
        rows, cols = self.row_count, self.col_count
        coordinates = []
        layer = 0  # Only consider the outermost layer

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

        return coordinates


    def get_total_pipe_ends(self):
        ''' Retorna o número de pipe ends no tabuleiro. '''

        pipe_ends = 0
        for row in range(self.row_count):
            for col in range(self.col_count):

                pipe_type = self.board[row][col][0]

                if pipe_type == 'F':
                    pipe_ends += 1

                elif pipe_type == 'B':
                    pipe_ends += 3
                
                elif pipe_type == 'V' or pipe_type == 'L':
                    pipe_ends += 2

        return pipe_ends


class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(PipeManiaState(board))
        

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento e faz pruning a ações que não respeitam as constraints."""

        actions = []

        # if dictionary empty return empty list
        if not state.board.next_actions:
            return actions
        
        # sort dictionary by number of possible actions per coordinate
        state.board.next_actions = OrderedDict(sorted(state.board.next_actions.items(), key=lambda item: len(item[1])))
        
        copy = deepcopy(state.board.next_actions)

        (key, val) = next(iter(state.board.next_actions.items()))

        row = key[0] 
        col = key[1] 

        possible_actions = val

        # if the set has length one, actions is optimal
        if(len(possible_actions) == 1):
            actions.append((row,col,next(iter(state.board.next_actions[(row,col)]))))
            del copy[(row,col)]
                  
        else:
            val_list = list(val)
            for pipe in val_list:
                actions.append((row,col,pipe))
            
            del copy[(row,col)]
            
        # update board dictionary
        state.board.next_actions = copy 

        return actions

        
    def result(self, state: PipeManiaState, action):
        
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        new_board = deepcopy(state.board)

        row, col, new_orientation = action
        new_board.board[row,col] = (new_orientation, 1)

        adjacents = new_board.get_suboptimal_adjacents(row, col, new_board.board[row,col][0])

        if adjacents:

            for (adj_row,adj_col) in adjacents:

                possible_values = new_board.get_possible_values(row,col, new_board.board[row,col][0], adj_row, adj_col)
                
                if not new_board.next_actions.get((adj_row,adj_col)):
                    new_board.next_actions[(adj_row,adj_col)] = set(possible_values)
                else:
                    new_board.next_actions[(adj_row, adj_col)] = set(possible_values) & new_board.next_actions[(adj_row, adj_col)]

                    
        return PipeManiaState(new_board)
    


    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
  
        visited = [[False] * state.board.col_count for _ in range(state.board.row_count)]
         
        if not visited[0][0]:
            
            cluster_size = self.bfs(state, visited, 0, 0)
            
            # Not goal state if cluster is smaller than the board
            if cluster_size < state.board.board_size:
                return False  
        
        return True
    

    def bfs(self, state: PipeManiaState, visited, row, col):
    
        queue = deque([(row, col)])
        visited[row][col] = True
        cluster_size = 1  

        while queue:
            row1,col1 = queue.popleft()
            neighbors = state.board.get_neighbors(row1,col1,state.board.board[row1,col1][0])
            for n_row, n_col in neighbors:
                
                # Check if neighbor is within bounds and unvisited
                if 0 <= n_row < state.board.row_count and 0 <= n_col < state.board.col_count and not visited[n_row][n_col]:
                    if state.board.is_connected(n_row, n_col, state.board.board[n_row, n_col][0])[0]:
                        visited[n_row][n_col] = True
                        queue.append((n_row, n_col))
                        cluster_size += 1
                    else:
                        return 0
            
        return cluster_size


    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*. Retorna número de pipe ends desconectadas"""

        board = node.state.board
        connected_pipe_ends = sum(board.is_connected(i, j, board.board[i,j][0])[1] for i in range(board.row_count) for j in range(board.col_count))
    
        return board.total_pipe_ends - connected_pipe_ends

    

if __name__ == "__main__":
    
    input_board = Board.parse_instance() 

    input_board.board_pre_processing()
    input_board.board = input_board.propagate_constraints()

    problem = PipeMania(input_board)

    solution =  depth_limited_search(problem, input_board.board_size)
    
    if solution is not None:
        solution.state.board.print_board()
        visualizer(solution.state.board.board, None)
    else:
        print("a solution deu none")


    
    
    




