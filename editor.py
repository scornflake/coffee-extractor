import tkinter

from settings import Settings


# Basic UI that has a set of controls on the left (west), and a canvas taking up the rest of the right hand (east) side.
class Editor:
    def __init__(self, master, the_settings: Settings):
        self.master = master
        self.settings = the_settings
        self.master.title("Editor")

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        self.controls = tkinter.Frame(self.master)
        self.controls.grid(row=0, column=0, sticky="nsew")
        self.editing_controls(self.controls)

        self.canvas = tkinter.Canvas(self.master, bg="red")
        self.canvas.grid(row=0, column=1, sticky="nsew")
        self.canvas_view(self.canvas)

    # A canvas, that'll hold a picture of what we're doing
    def canvas_view(self, master):
        canvas = tkinter.Canvas(master)
        canvas.grid(row=0, column=1, sticky="nsew")

        return canvas

    # Generate a frame from the movie and the digital area

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
