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

    def get_series_of_quantized_frame_numbers(self, total_wanted_frames, frame_offset, quantized: bool = True):
        def compute_frame_number(index):
            frames_per_second = self.frame_rate
            total_frames = self.frame_count - frame_offset

            frames_per_index = total_frames / total_wanted_frames
            frame_number = frame_offset + int(index * frames_per_index)

            # a good frame lasts 2 seconds
            # the target indicator is 1 sec.  So, each reading is 3s in duration.
            # number of frames in 3s is 90
            # quantize to 3 seconds
            if quantized:
                frames_per_3_seconds = int(frames_per_second * 3)
                frame_number = int(frame_number / frames_per_3_seconds) * frames_per_3_seconds
                return frame_offset + frame_number
            return frame_number

        return [compute_frame_number(index) for index in range(total_wanted_frames)]

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
