'''
    Yeabkal Wubshit
    Sudoku Solver using Backtracking and SAT Solver.

    *** REQUIREMENTS ***
    To solve a Sudoku puzzle through a SAT solver, you need to have either:
    1) PySAT module installed (https://pypi.org/project/python-sat/) [PREFERRED WAY]
    2) `minisat` command line tool 

    *** Usage ***
    import sudoku

    # 2D list representation of the board to solve.
    # Empty cells should be represented by 0s.
    board = [...]

    solution_approach = sudoku.SOLUTION_TYPE_BACKTRACKING 
    # Or solution_approach = sudoku.SOLUTION_TYPE_SAT_SOLVER to use a SAT solver.

    solution = sudoku.solve_sudoku(board, solution_approach)
    if solultion != None:
        # the board is solved.
    else:
        # the board cannot be solved.

'''
PYSAT_EXISTS = 0
MINISAT_COMMAND_LINE_TOOL_EXISTS = 1
NO_SAT_SOLVER_DETECTED = 2

MINISAT_COMMAND_LINE_COMMAND = 'minisatt'

SAT_SOLVER_STATUS_ON_MACHINE = NO_SAT_SOLVER_DETECTED

import math
from os import listdir
from os.path import isfile, join
import time
import random
from os import system
from shutil import which

try:
    from pysat.solvers import Glucose3
    SAT_SOLVER_STATUS_ON_MACHINE = PYSAT_EXISTS
except:
    if which(MINISAT_COMMAND_LINE_COMMAND) is not None:
        SAT_SOLVER_STATUS_ON_MACHINE = MINISAT_COMMAND_LINE_TOOL_EXISTS

TEST_SUDOKU_FILES_DIRECTORY = './sudoku_data'

SOLUTION_TYPE_BACKTRACKING = 0
SOLUTION_TYPE_SAT_SOLVER = 1

# Temp files to be used if `minisat` command line tool is utilized.
SUDOKU_CNF_FILE = 'del.cnf'	
MINISAT_RESULT_FILE = 'sol.txt'	
OUTPUT_BUFFER = 'del.txt'

PRINT_SPACE = '          '

# Tidy print constants
CHECK_MARK = '\033[92m' + '\033[1m' + u'\u2713' + '\033[0m' # Green + Bold + CheckMark + EndColor
FAIL_MARK = '\033[91m' + '\033[1m' + 'X' + '\033[0m' # Red + Bold + X + EndColor

NUM_FILES_TO_RUN_TESTS_ON = 10

class Sudoku:
    def __init__(self, dim = 9):
        self.dim = dim
        self.sat_solver = Glucose3()
        self.minisat_clauses = []
        self.possible_values = [(i + 1) for i in range(self.dim)]
        # Flattened representation of the Sudoku Board.
        self.backing_array = [0 for i in range(self.dim * self.dim)]
        # possibilities[i] contains all values that can potentially go to the ith cell.
        self.possibilities = [set([i for i in range(1, self.dim + 1)]) for j in range(self.dim * self.dim)]
        # rows[i] contains all the cells that are in the ith row.
        self.rows = [set() for i in range(self.dim)]
        # columns[i] contains all the cells that are in the ith column.
        self.columns = [set() for j in range(self.dim)]
        # subsudokus[i] contains all the cells that are in the ith subsudoku
        # (where a "subsudoku" is a sqrt(dim) * sqrt(dim) section within the board).
        self.subsudokus = [set() for k in range(self.dim)]
        # Contains the cells that are empty, in order by which they are to be fed to the recursion.
        self.cells_to_fill = []
        # memo[i] contains a tuple of the form '(cell, value)'
        # This saves a restriction that is brought by filling the ith cell.
        # The specific restriction saved is: "`value` cannot be placed at `cell`".
        # It is important to save such restrictions since we need to undo them once we clear the assignment that
        # originally resulted them.
        self.memo = {i:[] for i in range(self.dim * self.dim)}

        # Populate the rows, columns, and subsudokus lists.
        for i in range(self.dim * self.dim):
            row, col = self.get_coord_for_id(i)
            subsudoku = self.get_subsudoku_for_id(i)
            self.rows[row].add(i)
            self.columns[col].add(i)
            self.subsudokus[subsudoku].add(i)

    # Fills the Sudoku board with the values in the `vals` 2D list.
    # Returns True if initialization succeeds, returns False otherwise
    def initialize(self, vals):
        if vals is None or len(vals) != self.dim or vals[0] is None or len(vals[0]) != self.dim:
            return False
        for i in range(self.dim):
            for j in range(self.dim):
                self.set(i, j, vals[i][j])
        return True

    # Returns a 2D list representation of the Sudoku board.
    def get_matrix(self):
        return [self.backing_array[i*self.dim:i*self.dim+self.dim] for i in range(self.dim)]

    # Captures a tidy print representation of the Sudoku board.
    def capture_tidy_print(self, is_solution=False):
        capture = []
        sqrt = int(math.sqrt(self.dim))
        printable = self.get_matrix()
        top_bottom_rows = '* '
        for i in range(self.dim):
            top_bottom_rows += '- '
            if (i + 1) % sqrt == 0:
                top_bottom_rows += '* '

        for row in range(self.dim):
            if row == 0:
               capture.append(top_bottom_rows)
            to_print = '| '
            for col in range(self.dim):
                char_to_print = ''
                if printable[row][col] != 0:
                    char_to_print = str(printable[row][col])
                    if is_solution and self.old_copy[row][col] == 0:
                        char_to_print = '\033[92m' + '\033[1m' + char_to_print + '\033[0m'
                else:
                    char_to_print = FAIL_MARK
                to_print += char_to_print + ' '
                if (col + 1) % sqrt == 0:
                    to_print += '| '
            capture.append(to_print)
            if (row + 1) % sqrt == 0:
                capture.append(top_bottom_rows)

        return capture
    
    def print_tidy_capture(self, capture):
        for captured_line in capture:
            print(captured_line)
    
    def print_old_and_new_captures(self, oldc, newc):
        for i in range(len(oldc)):
            if i == 6: # Middle section...print arrow.
                print(oldc[i] + '=========>' + newc[i])
            else:
                print(oldc[i] + PRINT_SPACE + newc[i])

    # Returns the subsudoku where the cell (x,y) is located in.
    def get_subsudoku(self, x, y):
        sqrt = int(math.sqrt(self.dim))
        x_new = x // sqrt
        y_new = y // sqrt
        return x_new * sqrt + y_new

    # Returns the coordinate that `the_id` represents.
    def get_coord_for_id(self, the_id):
        return the_id // self.dim, the_id % self.dim

    # Returns the subsudoku where the cell (x,y) represented by `the_id` is located in.
    def get_subsudoku_for_id(self, the_id):
        x, y = self.get_coord_for_id(the_id)
        return self.get_subsudoku(x, y)

    # Returns the value at location (x,y)
    def get(self, x, y):
        return self.get_for_id(self.get_id(x, y))

    # Returns the value at the location (x,y) represented by `the_id`.
    def get_for_id(self, the_id):
        return self.backing_array[the_id]

    # Sets the value at (x,y) to `val`.
    def set(self, x, y, val):
        self.set_for_id(self.get_id(x, y), val)

    # Sets the value at the location (x,y) represented by `the_id` to `val`.
    def set_for_id(self, the_id, val):
        row, col = self.get_coord_for_id(the_id)
        subsudoku = self.get_subsudoku_for_id(the_id)

        if val not in self.possibilities[the_id]:
            return

        self.backing_array[the_id] = val
        self.memo[the_id] = []

        for cell in self.columns[col]:
            if val in self.possibilities[cell]:
                self.memo[the_id].append((cell, val))
                self.possibilities[cell].remove(val)
        for cell in self.rows[row]:
            if val in self.possibilities[cell]:
                self.memo[the_id].append((cell, val))
                self.possibilities[cell].remove(val)
        for cell in self.subsudokus[subsudoku]:
            if val in self.possibilities[cell]:
                self.memo[the_id].append((cell, val))
                self.possibilities[cell].remove(val)

    # Clears the value at (x,y).
    def clear(self, x, y):
        self.clear_for_id(self.get_id(x, y))

    # Clears the value at the location (x,y) represented by `the_id`.
    def clear_for_id(self, the_id):
        self.backing_array[the_id] = 0
        for pair in self.memo[the_id]:
            self.possibilities[pair[0]].add(pair[1])
        self.memo[the_id] = []

    # Returns the id that reperesents the location (x,y).
    def get_id(self, x, y):
        return self.dim * x + y

    # Checks if the Sudoku board is valid.
    def check(self):
        # Check rows.
        for row in range(self.dim):
            seen = set()
            for col in range(0, self.dim):
                val = self.get(row, col)
                if val is None or val in seen:
                    return False
                seen.add(val)

        # Check columns.
        for col in range(self.dim):
            seen = set()
            for row in range(0, self.dim):
                val = self.get(row, col)
                if val is None or val in seen:
                    return False
                seen.add(val)

        # Check 3x3 subsudokus.
        sqrt = int(math.sqrt(self.dim))
        for corner_y in range(0, self.dim, sqrt):
            for corner_x in range(0, self.dim, sqrt):
                seen = set()
                for i in range(sqrt):
                    for j in range(sqrt):
                        val = self.get(corner_x + i, corner_y + j)
                        if val is None or val in seen:
                            return False
                        seen.add(val)
        # All good!
        return True

    # Gives the code for a number that goes at (row, col) in the board.
    def code(self, num, row, col):
        return num * 100 + row * 10 + col + 1

    # Decodes the code to give the number, row, and column that the code represents.
    def decode(self, code):
        code -= 1
        col = code % 10
        code //= 10
        row = code % 10
        code //= 10
        num = code

        return (num, row, col)

    # Attempt to solve the Sudoku.,
    # Returns True if the board is successfully solved, False if otherwise.
    def solve_backtrack(self):
        self.save_old_copy()
        # Account for the clues.

        for the_id in range(self.dim * self.dim):
            if self.backing_array[the_id] != 0:
                val = self.backing_array[the_id]
                row, col = self.get_coord_for_id(the_id)
                subsudoku = self.get_subsudoku_for_id(the_id)

                # Make `val` unavailable for reuse in the same column, row, and subsudoku (3x3 grid).
                for cell in self.columns[col]:
                    if val in self.possibilities[cell]:
                        self.possibilities[cell].remove(val)
                for cell in self.rows[row]:
                    if val in self.possibilities[cell]:
                        self.possibilities[cell].remove(val)
                for cell in self.subsudokus[subsudoku]:
                    if val in self.possibilities[cell]:
                        self.possibilities[cell].remove(val)
            else:
                self.cells_to_fill.append(the_id)

        # Nothing to solve. All cells filled.
        if len(self.cells_to_fill) == 0:
            return True

        for i in self.possible_values:
            if self.fill(0, i):
                return True

        return False

    # Helper for backtrack solver.
    # Attempts to set the value for a given cell (at position `index` in the list of the cells to fill) to `val`
    # If the attempt is successful, it continues to fill the next cell to be filled.
    def fill(self, index, val):
        if index == len(self.cells_to_fill): # Every cell has been filled.
            return True
        the_id = self.cells_to_fill[index] # The current cell id which is to be filled.
        if val not in self.possibilities[the_id]: # The value we are trying to put into the cell cannot be used at that location.
            return False

        if self.get_for_id(the_id) == 0: # Check if cell is empty before overriding it.
            self.set_for_id(the_id, val)

        for i in self.possible_values:
            if self.fill(index + 1, i):
                return True

        self.clear_for_id(the_id)

        return False

    # Takes a 2D representation of a sudoku board and prints its solution.
    def generate_cnf_clauses(self):
        # Make sure that the pre-filled clues are considered (enforced)...
        for the_id in range(self.dim * self.dim):
            num = self.get_for_id(the_id)
            if num is not None and num != 0:
                x, y = self.get_coord_for_id(the_id)
                self.add_clause([self.code(num, x, y)])
        
        # Enforce only 1 entry of all the entries that can go to a cell is included.
        for row in range(self.dim):
            for col in range(self.dim):
                for i in range(1,self.dim):
                    for j in range(2,self.dim + 1):
                        self.add_clause([-i, -j])


        # Enforce at least one of all the entries of every number that can go to a cell is included.
        for row in range(self.dim):
            for col in range(self.dim):
                clause = []
                for num in range(1,self.dim + 1):
                    num_code = self.code(num, row, col)
                    clause.append(num_code)
                self.add_clause(clause)
        
        sqrt = int(math.sqrt(self.dim))
        # Enforce no repeat of any number within a 3x3 subsudoku.
        for num in range(1,10):
            for cy in range(0, self.dim, sqrt):
                for cx in range(0, self.dim, sqrt):
                    grid3x3 = []
                    for ydiff in range(sqrt):
                        for xdiff in range(sqrt):
                            y = cy + ydiff
                            x = cx + xdiff
                            num_code = self.code(num, y, x)
                            grid3x3.append(num_code)
                    
                    for i in range(len(grid3x3) - 1):
                        for j in range(i + 1, len(grid3x3)):
                            self.add_clause([-grid3x3[i], -grid3x3[j]])
                    
        for num in range(1, self.dim + 1):
            for row in range(self.dim):
                row_elems = []
                col_elems = []
                for col in range(self.dim):
                    num_code_row = self.code(num, row, col)
                    row_elems.append(num_code_row)

                    num_code_column = self.code(num, col, row)
                    col_elems.append(num_code_column)

                # Enforce that a number appears at least once in a row.
                clause = []
                for elem in row_elems:
                    clause.append(elem)
                self.sat_solver.add_clause(clause)

                # Enforce no repeatition of any number within a row.
                for i in range(len(row_elems) - 1):
                    for j in range(i + 1, len(row_elems)):
                        self.add_clause([-row_elems[i], -row_elems[j]])

                # Enforce that a number appears at least once in a column.
                clause = []
                for elem in col_elems:
                    clause.append(elem)
                self.add_clause(clause)

                # Enforce no repeatition of any number within a column.
                for i in range(len(col_elems) - 1):
                    for j in range(i + 1, len(col_elems)):
                        self.add_clause([-col_elems[i], -col_elems[j]])

    def add_clause(self, clause):
        if SAT_SOLVER_STATUS_ON_MACHINE == PYSAT_EXISTS:
            self.sat_solver.add_clause(clause)
        elif SAT_SOLVER_STATUS_ON_MACHINE == MINISAT_COMMAND_LINE_TOOL_EXISTS:
            self.minisat_clauses.append(' '.join([str(x) for x in clause]) + ' 0')

    def write_clauses_to_file(self):	
        cnf = open(SUDOKU_CNF_FILE, 'w')	
        sqrt = int(math.sqrt(self.dim))	
        num_variables = (self.dim**sqrt)	

        cnf.write('p cnf %d %d\n' % (num_variables, len(self.minisat_clauses)))	
        for clause in self.minisat_clauses:	
            cnf.write(clause + '\n')	
        cnf.close()	

    # Runs the minisat command line tool.
    def run_minisat(self):	
        system("minisat %s %s > %s"%(SUDOKU_CNF_FILE, MINISAT_RESULT_FILE, OUTPUT_BUFFER))	
        return [int(x) for x in (open(MINISAT_RESULT_FILE, 'r').readlines()[1].split())]
    
    def solve_sat_solver(self):
        if SAT_SOLVER_STATUS_ON_MACHINE == NO_SAT_SOLVER_DETECTED:
            print('\033[91m' + 'No applicable SAT solver detected on your machine.\n \
                Please, install PySAT (pip3 install python-sat) and retry running this program.' + '\033[0m')
            return False

        self.save_old_copy()
        self.generate_cnf_clauses()
        solution = []

        if SAT_SOLVER_STATUS_ON_MACHINE == PYSAT_EXISTS:
            if not self.sat_solver.solve():
                return False
            solution = self.sat_solver.get_model()
        else:
            self.write_clauses_to_file()
            solution = self.run_minisat()

            system("rm %s"%SUDOKU_CNF_FILE)
            system("rm %s"%MINISAT_RESULT_FILE)
            system("rm %s"%OUTPUT_BUFFER)

        selected_cells = [x for x in solution if x > 0]
        
        for i in selected_cells:
            num, row, column = self.decode(i)        
            self.set_for_id(self.get_id(row, column), num)
                

        return True
    
    def save_old_copy(self):
        self.old_copy = self.get_matrix()

# Takes a 2D list (9x9) representation of a sudoku board and attempts to solve it (empty cells are represented by 0s).
# Returns a matrix representing the solution for the input board (as a 9x9 as a 2D list) if the board is solved,
# or a board that is the same as the input board if the sudoku is not solved.
def solve_sudoku(sudoku_board, solution_type):
    sudoku = Sudoku()
    if not sudoku.initialize(sudoku_board):
        print("Initialization Error")
        return None
    
    pre_solution_capture = sudoku.capture_tidy_print()
    success = False
    if solution_type == SOLUTION_TYPE_BACKTRACKING:
        success = sudoku.solve_backtrack()
    elif solution_type == SOLUTION_TYPE_SAT_SOLVER:
        if SAT_SOLVER_STATUS_ON_MACHINE == NO_SAT_SOLVER_DETECTED:
            print('\033[91m' + 'No applicable SAT solver detected on your machine.' + '\033[0m')
            return None
        success = sudoku.solve_sat_solver()

    if success:
        assert (sudoku.check())
        post_solution_capture = sudoku.capture_tidy_print(True)
        sudoku.print_old_and_new_captures(pre_solution_capture, post_solution_capture)
        print(CHECK_MARK)
        return sudoku.get_matrix()
    else:
        print("Sudoku is not solvable.")
        print(FAIL_MARK)
        return None
 

def solve_sudoku_backtracking(sudoku_board):
    return solve_sudoku(sudoku_board, SOLUTION_TYPE_BACKTRACKING)

def solve_sudoku_sat_solver(sudoku_board):
    return solve_sudoku(sudoku_board, SOLUTION_TYPE_SAT_SOLVER)

def count_clues(sudoku_board):
    clues = 0
    for row in sudoku_board:
        for cell in row:
            if cell != 0:
                clues += 1

    return clues

def reduce_clues(sudoku_board, target_clues):
    clues = count_clues(sudoku_board)
    if clues <= target_clues:
        return

    clue_cells = []
    for row in range(len(sudoku_board)):
        for col in range(len(sudoku_board[0])):
            if sudoku_board[row][col] != 0:
                clue_cells.append((row, col))

    random.shuffle(clue_cells)
    num_clues_to_cancel = clues - target_clues

    for i in range(num_clues_to_cancel):
        clue_to_cancel = clue_cells[i]
        sudoku_board[clue_to_cancel[0]][clue_to_cancel[1]] = 0


# Reads sudoku data from txt files and solves them by the given solution type.
# Times the tests.
# Sudoku data obtained from the University of Vaasa's Sudoku Research Page (http://lipas.uwasa.fi/~timan/sudoku/)
def run_tests(solution_type, num_files_to_execute):
    prefix = "./sudoku_data"
    sudoku_files = [f for f in listdir(prefix)][-num_files_to_execute:]
    boards = []
    for file_name in sudoku_files:
        file = open(join(prefix, file_name), "r")
        board = []
        for line in file.readlines():
            board.append([int(x) for x in line.split()])
            if len(board[-1]) == 0:
                board.pop()
        boards.append(board)

    min_time_taken, max_time_taken = float('inf'), -1
    solved_sudokus_count = 0
    start_time = time.time()

    for index, board in enumerate(boards):
        print(sudoku_files[index])
        print("Clues: %d" % (count_clues(board)))
        curr_start_time = time.time()
        if solve_sudoku(board, solution_type) is not None:
            solved_sudokus_count += 1
        time_taken = time.time() - curr_start_time
        max_time_taken = max(max_time_taken, time_taken)
        min_time_taken = min(min_time_taken, time_taken)
        
    total_time_taken = time.time() - start_time
    average_solve_time = (total_time_taken)/solved_sudokus_count

    solution_type_str = ''
    if solution_type == SOLUTION_TYPE_BACKTRACKING:
        solution_type_str = 'BACKTRACKING'
    else:
        solution_type_str = 'SAT SOLVER'
    print('\033[1m' + 'ANALYSIS FOR ' + solution_type_str + ' SOLUTION.' + '\033[0m')
    print("Total time taken for %d sudokus: %ss." % (solved_sudokus_count, round(total_time_taken, 3)))
    print("Failed solve attempts: %d" % (NUM_FILES_TO_RUN_TESTS_ON - solved_sudokus_count))
    print("Average time taken per sudoku: %ss." % (round(average_solve_time, 3)))
    print("Min time per sudoku: %ss." % (round(min_time_taken, 3)))
    print("Max time per sudoku: %ss." % (round(max_time_taken, 3)))

    return total_time_taken

if __name__ == '__main__':
    total_time_backtracking = run_tests(SOLUTION_TYPE_BACKTRACKING, NUM_FILES_TO_RUN_TESTS_ON)
    print("--------------------------")
    if SAT_SOLVER_STATUS_ON_MACHINE != NO_SAT_SOLVER_DETECTED:
        total_time_sat_solver = run_tests(SOLUTION_TYPE_SAT_SOLVER, NUM_FILES_TO_RUN_TESTS_ON)
        print("--------------------------")
        print('SAT Solver is performing %.3fx better than backtracking.' % (total_time_backtracking/total_time_sat_solver))
    else:
        print('\033[91m' + 'No applicable SAT solver detected on your machine.\n \
Please, install PySAT (pip3 install python-sat) and retry running this program.' + '\033[0m')
