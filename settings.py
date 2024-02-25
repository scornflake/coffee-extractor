# Input file specification as JSON
# {
#   "movie_file": "movie.mp4",
#   "output_dir": "output",
#   "first_crack_start": "00:05:00",
#   "first_crack_end": "00:05:30",
#   "second_crack_start": "00:06:00",
#   "second_crack_end": "00:06:30",
#   "target_temp": 242,
#   "digital_area": {
#     "left": 450,
#     "top": 780,
#     "width": 110,
#     "height": 60
#   }
# }
import json
import os

from digital_area import Area


class Settings:
    def __init__(self, spec_filename):
        # Read the input spec
        with open(spec_filename, 'r') as f:
            self.input_spec = json.load(f)

        self.movie_file = self.input_spec["movie_file"]
        self.output_dir = self.input_spec["output_dir"] or "."
        self.lcd_quad_skew = self.input_spec["lcd_quad_skew"] or 0
        self.target_temp = self.input_spec["target_temp"] or 0
        self.digital_area = Area.from_json(self.input_spec["digital_area"])
        self.first_crack_start = self.input_spec["first_crack_start"]
        self.first_crack_end = self.input_spec["first_crack_end"]
        self.second_crack_start = self.input_spec["second_crack_start"]
        self.second_crack_end = self.input_spec["second_crack_end"]
        self.target_temp = self.input_spec["target_temp"]

        self.validate()

    def write_to_file(self):
        # Write the input spec to a file
        with open(os.path.join(self.output_dir, 'input_spec.json'), 'w') as f:
            json.dump(self.input_spec, f, indent=2)

    def validate(self):
        # all the parameters in the input file specification must exist and be valid
        if not self.movie_file:
            print('Error: movie_file is required')
            exit(1)

        if not os.path.exists(self.movie_file):
            print(f'Error: movie_file {self.movie_file} does not exist')
            exit(1)

        if not self.output_dir:
            print('Error: output_dir is required')
            exit(1)

        if not os.path.exists(self.output_dir):
            print(f'Error: output_dir {self.output_dir} does not exist')
            exit(1)

        if not self.first_crack_start:
            print('Error: first_crack_start is required')
            exit(1)

        if not self.first_crack_end:
            print('Error: first_crack_end is required')
            exit(1)

        if not self.second_crack_start:
            print('Error: second_crack_start is required')
            exit(1)

        if not self.second_crack_end:
            print('Error: second_crack_end is required')
            exit(1)
