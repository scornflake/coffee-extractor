import tkinter

import cv2
from PIL import ImageTk, Image

from movie import Movie
from settings import Settings


# Basic UI that has a set of controls on the left (west), and a canvas taking up the rest of the right hand (east) side.
class Editor:
    def __init__(self, master, the_settings: Settings):
        self.video_canvas = None
        self.imagetk = None
        self.image = None
        self.master = master
        self.settings = the_settings
        self.master.title("Editor")

        self.movie = Movie(self.settings.movie_file)
        self.video_frame = None

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        self.controls = tkinter.Frame(self.master)
        self.controls.grid(row=0, column=0, sticky="nsew")
        self.editing_controls(self.controls)

        self.main_root_canvas = tkinter.Frame(self.master, bg="red")
        self.main_root_canvas.grid(row=0, column=1, sticky="nsew")
        # self.canvas_view(self.main_root_canvas)

        self.update_canvas()

    def canvas_view(self, master):
        self.video_canvas = tkinter.Canvas(master)
        self.video_canvas.grid(row=0, column=1, sticky="nsew")
        return self.video_canvas

    def update_canvas(self):
        # Generate a frame from the movie and the digital area
        a_video_frame = self.movie.get_frame_number(400)

        corrected = cv2.cvtColor(a_video_frame, cv2.COLOR_BGR2RGB)
        # remove alpha channel
        corrected = corrected[:, :, :3]

        self.image = Image.fromarray(corrected)
        self.imagetk = ImageTk.PhotoImage(self.image)
        self.video_canvas = tkinter.Label(self.main_root_canvas, image=self.imagetk)
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self.video_canvas.image = self.imagetk

    # Basic controls to adjust the canvas. left/top/width/height boxes.
    def editing_controls(self, master):
        # Make up a frame, then put the left/top/width/height boxes into it.
        # Its a vertical layout with labels for each entry box.
        # The labels are right aligned to the text boxes.
        # The text boxes are left aligned.

        # The frame
        frame = tkinter.Frame(master)
        frame.grid(row=0, column=0, sticky="nsew")

        # The labels
        left_label = tkinter.Label(frame, text="Left")
        left_label.grid(row=0, column=0, sticky="e")
        top_label = tkinter.Label(frame, text="Top")
        top_label.grid(row=1, column=0, sticky="e")
        width_label = tkinter.Label(frame, text="Width")
        width_label.grid(row=2, column=0, sticky="e")
        height_label = tkinter.Label(frame, text="Height")
        height_label.grid(row=3, column=0, sticky="e")

        # The text boxes
        left_entry = tkinter.Entry(frame)
        left_entry.grid(row=0, column=1, sticky="w")
        top_entry = tkinter.Entry(frame)
        top_entry.grid(row=1, column=1, sticky="w")
        width_entry = tkinter.Entry(frame)
        width_entry.grid(row=2, column=1, sticky="w")
        height_entry = tkinter.Entry(frame)
        height_entry.grid(row=3, column=1, sticky="w")

        # Fill in UI with values from settings
        left_entry.insert(0, self.settings.digital_area.left)
        top_entry.insert(0, self.settings.digital_area.top)
        width_entry.insert(0, self.settings.digital_area.width)
        height_entry.insert(0, self.settings.digital_area.height)

        # Return the frame
        return frame


import args

# Load the settings from the input_spec.json file
settings = Settings(args.args.input_spec)

# Make it go!
root = tkinter.Tk()
Editor(root, settings)
root.mainloop()
