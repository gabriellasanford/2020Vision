# Yeabkal Wubshit
# Sudoku Solver using Backtracking.

import math
from os import listdir
from os.path import isfile, join
import time
import random

class Sudoku:
    def __init__(self, dim = 9):
        self.dim = dim
        self.possible_values = [(i + 1) for i in range(self.dim)]
        # Flattened representation of the Sudoku Board.
        self.backing_array = [None for i in range(self.dim * self.dim)]
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

    # Prints the Sudoku board.
    def print_tidy(self):
        sqrt = int(math.sqrt(self.dim))
        printable = self.get_matrix()
        top_bottom_rows = '* '
        for i in range(self.dim):
            top_bottom_rows += '- '
            if (i + 1) % sqrt == 0:
                top_bottom_rows += '* '

        for row in range(self.dim):
            if row == 0:
               print(top_bottom_rows)
            to_print = '| '
            for col in range(self.dim):
                to_print += str(printable[row][col]) + ' '
                if (col + 1) % sqrt == 0:
                    to_print += '| '
            print(to_print)
            if (row + 1) % sqrt == 0:
                print(top_bottom_rows)
        print()

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
        self.backing_array[the_id] = None
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

    # Attempt to solve the Sudoku.,
    # Returns True if the board is successfully solved, False if otherwise.
    def solve(self):
        # Account for the clues.
        for the_id in range(self.dim * self.dim):
            if self.backing_array[the_id] is not None:
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

    # Attempts to set the value for a given cell (at position `index` in the list of the cells to fill) to `val`
    # If the attempt is successful, it continues to fill the next cell to be filled.
    def fill(self, index, val):
        if index == len(self.cells_to_fill): # Every cell has been filled.
            return True
        the_id = self.cells_to_fill[index] # The current cell id which is to be filled.
        if val not in self.possibilities[the_id]: # The value we are trying to put into the cell cannot be used at that location.
            return False

        if self.get_for_id(the_id) is None: # Check if cell is empty before overriding it.
            self.set_for_id(the_id, val)

        for i in self.possible_values:
            if self.fill(index + 1, i):
                return True

        self.clear_for_id(the_id)

        return False

# Takes a 2D list (9x9) representation of a sudoku board and attempts to solve it (empty cells are represented by 0s).
# Returns a matrix representing the solution for the input board (as a 9x9 as a 2D list) if the board is solved,
# or a board that is the same as the input board if the sudoku is not solved.
def solve_sudoku(sudoku_board):
    sudoku = Sudoku()
    if not sudoku.initialize(sudoku_board):
        print("Initialization Error")
        return None
    if sudoku.solve():
        assert (sudoku.check())
        sudoku.print_tidy()
    else:
        print("Sudoku is not solvable.")

    return sudoku.get_matrix()

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


# Reads sudoku data from txt files and solves them.
# Times the tests.
# Sudoku data obtained from the University of Vaasa's Sudoku Research Page (http://lipas.uwasa.fi/~timan/sudoku/)
def run_tests():
    prefix = "./sudoku_data"
    sudoku_files = [f for f in listdir(prefix)]
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
        solve_sudoku(board)
        time_taken = time.time() - curr_start_time
        max_time_taken = max(max_time_taken, time_taken)
        min_time_taken = min(min_time_taken, time_taken)
        solved_sudokus_count += 1
    total_time_taken = time.time() - start_time
    average_solve_time = (total_time_taken)/solved_sudokus_count
    print("Total time taken for %d sudokus: %ss." % (solved_sudokus_count, round(total_time_taken, 3)))
    print("Average time taken per sudoku: %ss." % (round(average_solve_time, 3)))
    print("Min time per sudoku: %ss." % (round(min_time_taken, 3)))
    print("Max time per sudoku: %ss." % (round(max_time_taken, 3)))

run_tests()