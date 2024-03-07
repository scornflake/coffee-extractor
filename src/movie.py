import time

import cv2


class Movie:
    def __init__(self, movie_filename):
        # Open the movie file, using opencv
        self.cap = cv2.VideoCapture(movie_filename)
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration(self):
        duration = self.frame_count / self.frame_rate
        return duration

    def hhmmss(self, frame_number, filename_safe: bool = False) -> str:
        hhmmss = time.strftime('%H:%M:%S', time.gmtime(frame_number / self.frame_rate))
        if filename_safe:
            hhmmss = hhmmss.replace(":", "")
        return hhmmss

    def get_frame_number(self, frame_number):
        # Validate that frame is >= 0 and less than the self.frame_count
        if frame_number < 0 or frame_number >= self.frame_count:
            raise ValueError(f"Frame number {frame_number} is out of range")

        # Get a specific frame from the movie
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        # convert image to CV_32f
        # print(f"FRAME properties: {frame.shape}, depth: {frame.dtype}, type: {type(frame)}, pixel format: {frame[0, 0]}")
        return frame
