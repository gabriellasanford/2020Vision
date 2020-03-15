"""
How to use: for each image, click on the keypoints, then press any key on your keyboard to move to the next image.
Your progress is saved as you go along. You can quit at any time by terminating the program and when you re-run the
program it will pick up where you left off.
The labels are saved in label_file as a string representation of a python dict.
"""
import os
from typing import Dict, List, Tuple

import cv2


class KeypointLabeler:
    def __init__(self):
        self.image_directory: str = "sudoku_midterm"
        self.label_file: str = "sudoku_corners.txt"
        self.keypoint_radius: int = 4
        self.current_filename: str = ""
        self.current_image = None
        self.labels: Dict[str, List[Tuple[int, int]]]
        labels_string: str = self.read_labels().strip()
        if not labels_string:
            self.labels: Dict[str, List[Tuple[int, int]]] = {}
        else:
            self.labels: Dict[str, List[Tuple[int, int]]] = eval(labels_string)

    def save_label(self, event: int, x: int, y: int, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            cv2.circle(self.current_image, (x, y), self.keypoint_radius, (255, 0, 0), -1)
            cv2.imshow("label_me_please", self.current_image)
            self.labels[self.current_filename].append((x, y))

    def label(self):
        cv2.namedWindow("label_me_please")
        for filename in os.listdir(self.image_directory):
            self.current_filename = filename
            if self.current_filename in self.labels:
                continue
            self.labels[self.current_filename] = []
            cv2.setMouseCallback("label_me_please", self.save_label)
            self.current_image = cv2.imread(self.image_directory + "/" + filename)
            cv2.imshow("label_me_please", self.current_image)
            cv2.waitKey()
            self.write_labels()

    def read_labels(self) -> str:
        if not os.path.exists(self.label_file):
            return ""
        with open(self.label_file) as label_output_file:
            label_string = label_output_file.readline()
        return label_string

    def write_labels(self):
        with open(self.label_file, "w") as label_output_file:
            label_output_file.write(str(self.labels))


if __name__ == "__main__":
    keypoint_labeler: KeypointLabeler = KeypointLabeler()
    keypoint_labeler.label()
