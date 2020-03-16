"""
Thanks to Paul Devlin for providing most of the infrastructure here!

For your midterm solution:
You should have a function that takes a greyscale image as input, and returns
a list of four 2-tuples containing the four (x, y) corners of the Sudoku board.

How to use:
Change the "from" line below to include
* the name of the file containing your function,
* and the name of your function.
For example, my function   sudoku_bounds  is in (my) file  sudoku_grid_find_Rob.py

Then run this file, for example by typing "python midterm_grader.py" on the command line.
Note that this file should reside in the same directory as the sudoku_midterm folder
(but not *within* the sudoku_midterm folder). The sudoku_corners.txt file should be
in the sudoku_midterm folder.
"""
from sudoku_grid_find_Rob import sudoku_bounds as bounds

import os
from typing import Dict, List, Tuple
import cv2


class MidtermGrader:
    def __init__(self):
        self.image_directory: str = "sudoku_midterm"
        self.label_file: str = self.image_directory + "/sudoku_corners.txt"
        self.keypoint_radius: int = 4
        self.current_filename: str = ""
        self.current_image = None
        self.labels: Dict[str, List[Tuple[int, int]]]
        labels_string: str = self.read_labels().strip()
        if not labels_string:
            self.labels: Dict[str, List[Tuple[int, int]]] = {}
        else:
            self.labels: Dict[str, List[Tuple[int, int]]] = eval(labels_string)

    def read_labels(self) -> str:
        if not os.path.exists(self.label_file):
            return ""
        with open(self.label_file) as label_input_file:
            label_string = label_input_file.readline()
        return label_string

    def grade(self):
        cv2.namedWindow("Visualizing Your Answers")
        for filename in os.listdir(self.image_directory):
            print("Grader: ", filename)
            if filename.find("png") == -1: continue # read only png files
            self.current_filename = filename
            corners = self.labels[self.current_filename]
            self.current_image = cv2.imread(self.image_directory + "/" + filename, cv2.IMREAD_GRAYSCALE)
            student_corners = bounds(self.current_image)

            # Draw corners. Correct answers are black. Student answers are white.
            for c in student_corners:
                cv2.circle(self.current_image, tuple(map(int, c)), 7, (255, 0, 0), 5)
            for l in corners:
                cv2.circle(self.current_image, l, 4, (0, 0, 255), 3)
            cv2.imshow("Visualizing Your Answers", self.current_image)
            cv2.waitKey(200) # Modify this line to pause longer between images
        cv2.destroyAllWindows()

    

if __name__ == "__main__":
    midterm_grader: MidtermGrader = MidtermGrader()
    midterm_grader.grade()
