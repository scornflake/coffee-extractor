import tkinter

from src import args
import cv2
from PIL import ImageTk, Image

from extraction import parse_int_via_tesseract, extract_lcd_and_ready_for_teseract2, extract_lcd_and_ready_for_teseract
from movie import Movie
from settings import Settings


# Basic UI that has a set of controls on the left (west), and a canvas taking up the rest of the right hand (east) side.
class Editor:
    def __init__(self, master, the_settings: Settings):
        self.video_frame_seconds_label = None
        self.lcd_temp_label = None
        self.current_video_frame = None
        self.lcd_preview_canvas = None
        self.step_forward_button = None
        self.step_back_button = None
        self.slider = None
        self.save_button = None
        self.overlay_polygon = None
        self.ui_overlay_canvas = None
        self.video_canvas = None
        self.video_photo = None
        self.video_corrected_image = None
        self.master = master
        self.settings = the_settings
        self.master.title("Editor")
        self.video_frame_number = 1000

        self.movie = Movie(self.settings.movie_file)
        self.video_frame = None

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        self.controls = tkinter.Frame(self.master)
        self.controls.grid(row=0, column=0, sticky="nsew")
        self.editing_controls(self.controls)

        self.main_root_canvas = tkinter.Frame(self.master, bg="red")
        self.main_root_canvas.grid(row=0, column=1, sticky="nsew")
        self.create_video_canvas_view(self.main_root_canvas)

        self.put_movie_frame_onto_video_canvas()

        # self.step_forward()
        # self.step_forward()
        # self.step_forward()

    def create_video_canvas_view(self, master):
        # The canvas takes up most of the view
        self.video_canvas = tkinter.Canvas(master, width=1920, height=1080)
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        # Framing for buttons and slider
        button_frame = tkinter.Frame(master)
        button_frame.grid(row=1, column=0, sticky="nsew")

        # Add a slider underneath the canvas
        self.slider = tkinter.Scale(button_frame, from_=0, to=self.movie.frame_count, orient="horizontal")
        self.slider.grid(row=0, column=2, sticky="nsew")
        # Make slider take up remainder of the space
        button_frame.grid_columnconfigure(2, weight=1)

        # When the slider is moved, update the video canvas
        def slider_moved(event):
            self.video_frame_number = event.widget.get()
            self.put_movie_frame_onto_video_canvas()

        self.slider.bind("<ButtonRelease-1>", slider_moved)

        # To the left of the slider, add some buttons that let you step forward and backwards quickly
        self.step_back_button = tkinter.Button(button_frame, text="<<", command=self.step_back)
        self.step_back_button.grid(row=0, column=0, sticky="nsew")
        self.step_forward_button = tkinter.Button(button_frame, text=">>", command=self.step_forward)
        self.step_forward_button.grid(row=0, column=1, sticky="nsew")
        self.video_frame_seconds_label = tkinter.Label(button_frame, text="0.0")
        self.video_frame_seconds_label.grid(row=0, column=3, sticky="nsew")

        return self.video_canvas

    def step_back(self):
        self.video_frame_number = max(0, self.video_frame_number - self.movie.frame_count / 40)
        # update the slider to match
        self.slider.set(self.video_frame_number)
        self.put_movie_frame_onto_video_canvas()

    def step_forward(self):
        self.video_frame_number = min(self.movie.frame_count, self.video_frame_number + self.movie.frame_count / 40)
        self.slider.set(self.video_frame_number)
        self.put_movie_frame_onto_video_canvas()

    @property
    def video_seconds(self):
        return self.video_frame_number / self.movie.frame_rate

    def create_ui_overlay_view(self, parent):
        self.ui_overlay_canvas = tkinter.Canvas(parent, width=1920, height=1080)
        self.ui_overlay_canvas.grid(row=0, column=0, sticky="nsew")

    def create_or_update_ui_overlay(self):
        bottom_left = self.settings.digital_area.bottom_left_skewed(self.settings.lcd_quad_skew)
        bottom_right = self.settings.digital_area.bottom_right_skewed(self.settings.lcd_quad_skew)
        if self.overlay_polygon:
            self.video_canvas.delete(self.overlay_polygon)
            self.overlay_polygon = None

        if self.overlay_polygon is None:
            self.overlay_polygon = self.video_canvas.create_polygon(
                self.settings.digital_area.top_left,
                self.settings.digital_area.top_right,
                bottom_right,
                bottom_left,
                fill="",
                outline="green",
                width=3,
            )

        self.update_lcd_preview()

    def put_movie_frame_onto_video_canvas(self):
        # Generate a frame from the movie and the digital area
        self.current_video_frame = self.movie.get_frame_number(self.video_frame_number)
        corrected = cv2.cvtColor(self.current_video_frame, cv2.COLOR_BGR2RGB)

        self.video_corrected_image = Image.fromarray(corrected)
        self.video_photo = ImageTk.PhotoImage(self.video_corrected_image)
        self.video_canvas.create_image(0, 0, image=self.video_photo, anchor="nw")
        self.video_canvas.image = self.video_photo

        self.create_or_update_ui_overlay()
        self.video_frame_seconds_label.config(text=f"{self.video_seconds:.2f}")
        self.update_lcd_preview()

    # Basic controls to adjust the canvas. left/top/width/height boxes.
    def editing_controls(self, parent):
        # Make up a frame, then put the left/top/width/height boxes into it.
        # Its a vertical layout with labels for each entry box.
        # The labels are right aligned to the text boxes.
        # The text boxes are left aligned.

        # The frame
        frame = tkinter.Frame(parent)
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
        skew_label = tkinter.Label(frame, text="Skew")
        skew_label.grid(row=4, column=0, sticky="e")

        # The text boxes
        left_entry = tkinter.Entry(frame)
        left_entry.grid(row=0, column=1, sticky="w")
        top_entry = tkinter.Entry(frame)
        top_entry.grid(row=1, column=1, sticky="w")
        width_entry = tkinter.Entry(frame)
        width_entry.grid(row=2, column=1, sticky="w")
        height_entry = tkinter.Entry(frame)
        height_entry.grid(row=3, column=1, sticky="w")
        skew_entry = tkinter.Entry(frame)
        skew_entry.grid(row=4, column=1, sticky="w")

        # Fill in UI with values from settings
        left_entry.insert(0, self.settings.digital_area.left)
        top_entry.insert(0, self.settings.digital_area.top)
        width_entry.insert(0, self.settings.digital_area.width)
        height_entry.insert(0, self.settings.digital_area.height)
        skew_entry.insert(0, self.settings.lcd_quad_skew)

        # Now put in a flexible spacer (vertical)
        frame.grid_rowconfigure(5, weight=1)

        self.lcd_tuning_ui(frame)

        # The bottom, we have a horizontal row of buttons (for save, cancel, etc)
        button_frame = tkinter.Frame(frame)
        button_frame.grid(row=6, column=0, sticky="sew")
        self.save_button = tkinter.Button(button_frame, text="Save", command=self.save_settings)
        self.save_button.grid(row=0, column=0, sticky="nsew")
        # button_frame should have padding 10 pixels around all edges
        button_frame.grid(padx=10, pady=10)

        # When updating the value for skew_save it to settings and update the ui
        def update_skew(event):
            def update_skew_v(value):
                self.settings.lcd_quad_skew = value
                self.create_or_update_ui_overlay()

            self.get_int_from_entry(event, update_skew_v)

        # If any of the left/top/width/height are modified, save to appropriate settings var and update UI
        def update_left(event):
            def update_left_v(value):
                self.settings.digital_area.left = value
                self.create_or_update_ui_overlay()

            self.get_int_from_entry(event, update_left_v)

        def update_top(event):
            def update_top_v(value):
                self.settings.digital_area.top = value
                self.create_or_update_ui_overlay()

            self.get_int_from_entry(event, update_top_v)

        def update_width(event):
            def update_width_v(value):
                self.settings.digital_area.width = value
                self.create_or_update_ui_overlay()

            self.get_int_from_entry(event, update_width_v)

        def update_height(event):
            def update_height_v(value):
                self.settings.digital_area.height = value
                self.create_or_update_ui_overlay()

            self.get_int_from_entry(event, update_height_v)

        skew_entry.bind("<KeyRelease>", update_skew)
        left_entry.bind("<KeyRelease>", update_left)
        top_entry.bind("<KeyRelease>", update_top)
        width_entry.bind("<KeyRelease>", update_width)
        height_entry.bind("<KeyRelease>", update_height)

        return frame

    def get_int_from_entry(self, event, handler):
        value = event.widget.get()
        try:
            int_value = int(value)
            handler(int_value)
        except ValueError:
            print(f"Invalid value {value} for event...")
            return

    def save_settings(self):
        print("Saved settings")
        self.settings.write_to_file()

    def lcd_tuning_ui(self, parent):
        # Create our own frame to put everything into
        frame = tkinter.Frame(parent)
        frame.grid(row=5, column=0, columnspan=2, sticky="nsew")

        # Put in a canvas, that's big enough to show the LCD preview
        pct = 0.15
        self.lcd_preview_canvas = tkinter.Canvas(frame, width=1920 * pct, height=1080 * pct, background="black")
        self.lcd_preview_canvas.grid(row=0, column=0, sticky="nsew")

        # Add a label, to show the parsed lcd temp value
        self.lcd_temp_label = tkinter.Label(frame, text="LCD Temp: ")
        self.lcd_temp_label.grid(row=1, column=0, sticky="nsew")

        # Add UI controls to allow adjustment of the lower and upper HSV
        # The labels
        low_h_label = tkinter.Label(frame, text="Low H")
        low_h_label.grid(row=2, column=0, sticky="e")
        low_s_label = tkinter.Label(frame, text="Low S")
        low_s_label.grid(row=3, column=0, sticky="e")
        low_v_label = tkinter.Label(frame, text="Low V")
        low_v_label.grid(row=4, column=0, sticky="e")
        upper_h_label = tkinter.Label(frame, text="Upper H")
        upper_h_label.grid(row=5, column=0, sticky="e")
        upper_s_label = tkinter.Label(frame, text="Upper S")
        upper_s_label.grid(row=6, column=0, sticky="e")
        upper_v_label = tkinter.Label(frame, text="Upper V")
        upper_v_label.grid(row=7, column=0, sticky="e")
        lcd_blur_label = tkinter.Label(frame, text="Blur")
        lcd_blur_label.grid(row=8, column=1, sticky="e")

        # The text boxes
        low_h_entry = tkinter.Entry(frame)
        low_h_entry.grid(row=2, column=1, sticky="w")
        low_s_entry = tkinter.Entry(frame)
        low_s_entry.grid(row=3, column=1, sticky="w")
        low_v_entry = tkinter.Entry(frame)
        low_v_entry.grid(row=4, column=1, sticky="w")
        upper_h_entry = tkinter.Entry(frame)
        upper_h_entry.grid(row=5, column=1, sticky="w")
        upper_s_entry = tkinter.Entry(frame)
        upper_s_entry.grid(row=6, column=1, sticky="w")
        upper_v_entry = tkinter.Entry(frame)
        upper_v_entry.grid(row=7, column=1, sticky="w")
        lcd_blur_entry = tkinter.Entry(frame)
        lcd_blur_entry.grid(row=8, column=1, sticky="w")

        # Populate with the values from the settings
        low_h_entry.insert(0, self.settings.low_threshold.h)
        low_s_entry.insert(0, self.settings.low_threshold.s)
        low_v_entry.insert(0, self.settings.low_threshold.v)
        upper_h_entry.insert(0, self.settings.upper_threshold.h)
        upper_s_entry.insert(0, self.settings.upper_threshold.s)
        upper_v_entry.insert(0, self.settings.upper_threshold.v)
        lcd_blur_entry.insert(0, self.settings.lcd_blur_amount)

        # bind to the keyreleased, and update settings.
        def update_low_h(event):
            def update_low_h_v(value):
                self.settings.low_threshold.h = value
                self.update_lcd_preview()

            self.get_int_from_entry(event, update_low_h_v)

        def update_low_s(event):
            def update_low_s_v(value):
                self.settings.low_threshold.s = value
                self.update_lcd_preview()

            self.get_int_from_entry(event, update_low_s_v)

        def update_low_v(event):
            def update_low_v_v(value):
                self.settings.low_threshold.v = value
                self.update_lcd_preview()

            self.get_int_from_entry(event, update_low_v_v)

        def update_upper_h(event):
            def update_upper_h_v(value):
                self.settings.upper_threshold.h = value
                self.update_lcd_preview()

            self.get_int_from_entry(event, update_upper_h_v)

        def update_upper_s(event):
            def update_upper_s_v(value):
                self.settings.upper_threshold.s = value
                self.update_lcd_preview()

            self.get_int_from_entry(event, update_upper_s_v)

        def update_upper_v(event):
            def update_upper_v_v(value):
                self.settings.upper_threshold.v = value
                self.update_lcd_preview()

            self.get_int_from_entry(event, update_upper_v_v)

        def update_lcd_blur(event):
            def update_lcd_blur_v(value):
                self.settings.lcd_blur_amount = value
                self.update_lcd_preview()

            self.get_int_from_entry(event, update_lcd_blur_v)

        low_h_entry.bind("<KeyRelease>", update_low_h)
        low_s_entry.bind("<KeyRelease>", update_low_s)
        low_v_entry.bind("<KeyRelease>", update_low_v)
        upper_h_entry.bind("<KeyRelease>", update_upper_h)
        upper_s_entry.bind("<KeyRelease>", update_upper_s)
        upper_v_entry.bind("<KeyRelease>", update_upper_v)
        lcd_blur_entry.bind("<KeyRelease>", update_lcd_blur)

    def update_lcd_preview(self):
        if not self.lcd_preview_canvas:
            return

        # Using the current video frame, perform the same operations as the extractor
        opencv_image = extract_lcd_and_ready_for_teseract(self.current_video_frame, self.video_frame_number, self.settings)
        # opencv_image = extract_lcd_and_ready_for_teseract2(self.current_video_frame, self.video_frame_number, self.settings)
        if opencv_image is None:
            self.lcd_temp_label.config(text="NO IMAGE GENERATED")
            return

        # Convert to a PIL image
        pil_image = Image.fromarray(opencv_image)
        # Convert to a PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        # Put the photo onto the canvas
        self.lcd_preview_canvas.create_image(0, 0, image=photo, anchor="nw")
        self.lcd_preview_canvas.image = photo

        # Try to get the number, via tesseract
        parsed_value = parse_int_via_tesseract(opencv_image)
        if parsed_value is None:
            self.lcd_temp_label.config(text="LCD Temp: NONE")
        else:
            self.lcd_temp_label.config(text=f"LCD Temp: {parsed_value}")


# Load the settings from the input_spec.json file
settings = Settings(args.args.input_spec)


def save_and_quit():
    editor.save_settings()
    root.quit()


# Make it go!
root = tkinter.Tk()
editor = Editor(root, settings)
root.winfo_toplevel().title(f"Editing for {settings.movie_file}, {settings.identifier}")

menubar = tkinter.Menu(root)
mac_app_menu = tkinter.Menu(menubar, name="apple")
menubar.add_cascade(menu=mac_app_menu)
root.createcommand("tk::mac::Quit", save_and_quit)
root.protocol("WM_DELETE_WINDOW", save_and_quit)

root.mainloop()
