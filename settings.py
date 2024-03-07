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


class HSV:
    def __init__(self, h, s, v):
        self.h = h
        self.s = s
        self.v = v

    def to_json(self):
        return {
            "h": self.h,
            "s": self.s,
            "v": self.v,
        }

    @staticmethod
    def from_json(json):
        return HSV(json["h"], json["s"], json["v"])


class Settings:
    def __init__(self, spec_filename):
        # Read the input spec
        with open(spec_filename, 'r') as f:
            self.input_spec = json.load(f)

        self.ensure_input_spec_has_all_fields()

        self.spec_filename = spec_filename

        self.identifier = self.input_spec["identifier"]
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

        self.low_threshold = HSV.from_json(self.input_spec["lcd_low_threshold"])
        self.upper_threshold = HSV.from_json(self.input_spec["lcd_upper_threshold"])
        self.lcd_blur_amount = self.input_spec["lcd_blur_amount"] or 3

        self.validate()

    def save_values_to_input_spec(self):
        # Save the values to the input spec
        self.input_spec["identifier"] = self.identifier
        self.input_spec["movie_file"] = self.movie_file
        self.input_spec["output_dir"] = self.output_dir
        self.input_spec["lcd_quad_skew"] = self.lcd_quad_skew
        self.input_spec["target_temp"] = self.target_temp
        self.input_spec["digital_area"] = self.digital_area.to_json()
        self.input_spec["first_crack_start"] = self.first_crack_start
        self.input_spec["first_crack_end"] = self.first_crack_end
        self.input_spec["second_crack_start"] = self.second_crack_start
        self.input_spec["second_crack_end"] = self.second_crack_end
        self.input_spec["target_temp"] = self.target_temp

        self.input_spec["lcd_low_threshold"] = self.low_threshold.to_json()
        self.input_spec["lcd_upper_threshold"] = self.upper_threshold.to_json()
        self.input_spec["lcd_blur_amount"] = self.lcd_blur_amount

    def write_to_file(self):
        # Write the input spec to a file
        self.save_values_to_input_spec()
        folder_for_spec = os.getcwd()
        with open(os.path.join(folder_for_spec, self.spec_filename), 'w') as f:
            json.dump(self.input_spec, f, indent=2)

    def validate(self):
        # all the parameters in the input file specification must exist and be valid
        if not self.movie_file:
            print('Error: movie_file is required')
            exit(1)

        if not self.identifier:
            print('Error: identifier is required')
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

    def output_filename(self, label, filename: str = None, extension: str = None):
        if filename is not None:
            actual_filename = f'{label}_{self.identifier}_{filename}.{extension}'
        else:
            actual_filename = f'{label}_{self.identifier}.{extension}'
        full_path = os.path.join(self.output_dir, label, actual_filename)
        # check folder exists
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        return full_path

    def clear_files_for(self, label):
        folder = os.path.join(self.output_dir, label)
        # Ensure folder exists
        if not os.path.exists(folder):
            os.makedirs(folder)
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if file_path.__contains__(self.identifier):
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

    def ensure_input_spec_has_all_fields(self):
        # Ensure that all fields are present in the input spec
        if "lcd_quad_skew" not in self.input_spec:
            self.input_spec["lcd_quad_skew"] = -16

        if "lcd_blur_amount" not in self.input_spec:
            self.input_spec["lcd_blur_amount"] = 3

        if "target_temp" not in self.input_spec:
            self.input_spec["target_temp"] = 242

        if "lcd_low_threshold" not in self.input_spec:
            self.input_spec["lcd_low_threshold"] = HSV(70, 200, 100).to_json()

        if "lcd_upper_threshold" not in self.input_spec:
            self.input_spec["lcd_upper_threshold"] = HSV(100, 255, 255).to_json()

        if "digital_area" not in self.input_spec:
            self.input_spec["digital_area"] = Area(470, 795, 90, 42).to_json()

        if "output_dir" not in self.input_spec:
            self.input_spec["output_dir"] = "."

        # First and second crack values
        if "first_crack_start" not in self.input_spec:
            self.input_spec["first_crack_start"] = "00:05:00"

        if "first_crack_end" not in self.input_spec:
            self.input_spec["first_crack_end"] = "00:05:30"

        if "second_crack_start" not in self.input_spec:
            self.input_spec["second_crack_start"] = "00:06:00"

        if "second_crack_end" not in self.input_spec:
            self.input_spec["second_crack_end"] = "00:06:30"
