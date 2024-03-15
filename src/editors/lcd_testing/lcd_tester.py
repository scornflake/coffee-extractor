import os
import tkinter

import cv2
from PIL import ImageTk, Image

import args
from editors.components import AreaEditingUI, TuningUI
from extraction import extract_lcd_and_ready_for_tesseract, parse_int_via_tesseract
from movie import Movie
from settings import Settings

settings = Settings(args.args.input_spec)


class CellParts:
    def __init__(self, parent, lcd_image):
        self.parent = parent
        self.lcd_image = lcd_image


class LCDCell(tkinter.Frame):
    def __init__(self, settings_provider, frame_number: int, movie: Movie, master=None, cnf=None, **kwargs):
        self.pil_image = None
        self.fix_button = None
        self.indicator_size = 15
        self.indicator_label = None
        self.frame_number = frame_number
        self.frame = None
        self.movie = movie
        self.tesseract_value = None
        self.tesseract_label = None
        self.index = int(kwargs.pop("index", 0))
        self.settings_provider = settings_provider

        super().__init__(master, cnf, **kwargs)
        self.lcd_image = None
        self.actual_label = None
        self.actual_entry = None
        self.create_widgets()

        self.load_frame()

        print(f"[{self.index}]", end="")

    def load_frame(self, use_cache: bool = True):
        path_to_cached_frame = f"/tmp/lcd_editor_frame_{self.frame_number}.png"
        if use_cache and os.path.exists(path_to_cached_frame):
            self.frame = cv2.imread(path_to_cached_frame)
        else:
            self.frame = self.movie.get_frame_number(self.frame_number)
            cv2.imwrite(path_to_cached_frame, self.frame)

    def create_widgets(self):
        self.lcd_image = tkinter.Canvas(self, width=320, height=150, bg="pink")
        self.lcd_image.grid(row=0, column=0, columnspan=5, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.actual_label = tkinter.Label(self, text=f"Actual: {self.index + 1}")
        self.actual_label.grid(row=1, column=0, sticky="w")
        self.actual_entry = tkinter.Entry(self, width=5)
        self.actual_entry.grid(row=1, column=1, sticky="w")
        value_for_lcd = settings.get_true_value_for_lcd(self.index)
        if value_for_lcd is not None:
            self.actual_entry.insert(0, value_for_lcd)

        self.indicator_size = 19
        self.indicator_label = tkinter.Canvas(self, width=self.indicator_size, height=self.indicator_size)
        self.indicator_label.grid(row=1, column=2, padx=5, pady=5)

        self.fix_button = tkinter.Button(self, text="Fix", command=self.find_non_target_number)
        self.fix_button.grid(row=1, column=4)

        def save_to_settings(event):
            settings = self.settings_provider()
            settings.set_true_value_for_lcd(self.index, self.actual_entry.get())
            settings.write_to_file()

        def refresh_the_image(event):
            self.refresh_image()

        self.actual_entry.bind("<Enter>", refresh_the_image)
        self.actual_entry.bind("<Return>", save_to_settings)
        self.actual_entry.bind("<Tab>", save_to_settings)

        self.tesseract_label = tkinter.Label(self, text=f"Tess: ???")
        self.tesseract_label.grid(row=1, column=3)

        # self.refresh_image()
        # self.lcd_image.after(100 + (400 * self.index), self.refresh_image)

    def find_non_target_number(self):
        # Step at most 180 frames forward, looking for a non-target number, steps of 5
        for i in range(0, 180, 5):
            print(f"Trying to find non-target number at index {self.frame_number + i}...")
            frame = self.movie.get_frame_number(self.frame_number + i)
            extracted_image = extract_lcd_and_ready_for_tesseract(frame, 0, self.settings_provider())
            tesseract_value = parse_int_via_tesseract(extracted_image)
            if tesseract_value is not None and tesseract_value != settings.target_temp and tesseract_value < settings.target_temp:
                print(f"Found non-target value {tesseract_value} at frame {self.frame_number + i}")
                self.frame_number = self.frame_number + i
                settings.frame_numbers[self.index] = self.frame_number
                settings.write_to_file()

                self.load_frame(use_cache=False)
                return

    @property
    def values_matches_correctly(self):
        is_correct_value = False
        try:
            if self.tesseract_value is None:
                return False
            the_val = settings.get_true_value_for_lcd(self.index)
            if the_val is None:
                return False
            is_correct_value = int(self.tesseract_value) == int(the_val)
        except ValueError:
            pass
        return is_correct_value

    def update_tess_value_label(self):
        self.tesseract_label.config(text=f"Tess: {self.tesseract_value}")

        is_correct_value = self.values_matches_correctly

        # self.indicator_label is green if correct, red otherwise
        self.indicator_label.delete("all")
        self.indicator_label.create_oval(2, 2, self.indicator_size - 2, self.indicator_size - 2, fill="green" if is_correct_value else "red")

    def refresh_image(self, get_new_frame: bool = False):
        settings = self.settings_provider()

        if self.frame is None or get_new_frame:
            self.frame = self.movie.get_frame_number(self.frame_number)

        self.extracted_image = extract_lcd_and_ready_for_tesseract(self.frame, 0, settings)
        if self.extracted_image is None:
            self.tesseract_value = "???"
        else:
            self.tesseract_value = parse_int_via_tesseract(self.extracted_image)
            if self.tesseract_value is None:
                self.tesseract_value = "???"

        print(f"Value for {self.index} is {self.tesseract_value}")

        self.update_tess_value_label()

        # Respond to changing of widget size
        def resize_image_to_widget(event):
            # Before conversion, resize extracted_image to fit the canvas
            current_canvas_size = (self.lcd_image.winfo_width(), self.lcd_image.winfo_height())
            extracted_image = cv2.resize(self.extracted_image, current_canvas_size)
            self.pil_image = Image.fromarray(extracted_image)
            photo_image = ImageTk.PhotoImage(self.pil_image)
            self.lcd_image.create_image(0, 0, image=photo_image, anchor=tkinter.NW)
            self.lcd_image.image = photo_image

        self.lcd_image.bind("<Configure>", resize_image_to_widget)

        resize_image_to_widget(None)



class LCDEditor(tkinter.Frame):
    def __init__(self, master=None, cnf=None, **kwargs):
        self.statistics_label = None
        self.lcd_grid_views = None
        self.lcd_preview_canvas = None
        self.properties_panel = None
        self.grid_cols = 6
        self.grid_rows = 6

        self.movie = None
        if cnf is None:
            cnf = {}
        # Pop settings from kwargs
        self.settings = kwargs.pop("settings")
        super().__init__(master, cnf, width=1200, height=600, **kwargs)
        self.grid(row=0, column=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # background green
        # self.config(bg="green")
        self.create_widgets()

    @property
    def num_grid_items(self):
        return self.grid_cols * self.grid_rows

    def create_widgets(self):
        self.lcd_preview_canvas = None
        self.movie = Movie(self.settings.absolute_movie_file)

        # Create a properties panel, taking up the full left size, full height
        self.properties_panel = tkinter.Frame(self)  # ,  bg="red")
        self.properties_panel.grid(row=0, column=0, sticky="nsew")
        # grow vertically
        self.grid_columnconfigure(0, weight=0, minsize=100)
        # self.properties_panel.grid_rowconfigure(0, weight=1)
        self.create_properties_panel()

        # Create a row x col grid, that we can fill in with video previews
        self.lcd_preview_canvas = tkinter.Frame(self)
        self.lcd_preview_canvas.grid(row=0, column=1, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.lcd_grid_views = []

        # # Lets look at 5*5 video frames from the movie
        total_frames = self.movie.frame_count
        frames_to_generate = self.settings.frame_numbers
        if len(frames_to_generate) != self.num_grid_items:
            frames_to_generate = self.movie.get_series_of_quantized_frame_numbers(self.num_grid_items, self.settings.frame_offset)
            self.settings.frame_numbers = frames_to_generate
        for i in range(self.num_grid_items):
            column = (i % self.grid_rows)
            row = i // self.grid_cols
            frame_number = frames_to_generate[i]
            cell = LCDCell(frame_number=frame_number, movie=self.movie, settings_provider=self.settings_provider, master=self.lcd_preview_canvas, index=i)
            cell.grid(row=row, column=column, sticky="nsew", padx=5, pady=5)
            self.lcd_preview_canvas.grid_rowconfigure(row, weight=1)
            self.lcd_preview_canvas.grid_columnconfigure(column, weight=1)
            # make cell yellow
            # cell.config(bg="yellow")
            self.lcd_grid_views.append(cell)

        # Now a row underneath, for statistics
        self.statistics_label = tkinter.Label(self, text="Statistics")
        self.statistics_label.grid(row=2, column=0, columnspan=2, sticky="s")

        self.slow_lcd_refresh()

    def settings_provider(self):
        return self.settings

    def save_settings(self):
        self.settings.write_to_file()

    def create_properties_panel(self):
        area_ui = AreaEditingUI(master=self.properties_panel, the_settings=self.settings, update_preview_callback=self.update_everything)
        area_ui.grid(row=0, column=0, sticky="nw")

        tuning_props = TuningUI(master=self.properties_panel, update_preview_callback=self.update_for_tesseract_and_get_new_value, change_on_enter_only=True,
                                the_settings=self.settings)
        tuning_props.grid(row=1, column=0, sticky="nw")

        # Followed by a 'save' button
        save_button = tkinter.Button(self.properties_panel, text="Save", command=self.save_settings)
        save_button.grid(row=100, column=0, sticky="ne")

    def update_everything(self):
        self.update_for_tesseract_and_get_new_value()

    def update_for_tesseract_and_get_new_value(self):
        self.slow_lcd_refresh(immediate=True)

    def slow_lcd_refresh(self, immediate: bool = False):
        for cell in self.lcd_grid_views:
            cell.tesseract_value = None

        # Refresh the images slowly, so the UI is still responsive
        def load_next_empty_cell():
            for cell in self.lcd_grid_views:
                if cell.tesseract_value is None:
                    cell.refresh_image()
                    self.after(20, load_next_empty_cell)
                    self.recompute_stats()
                    return

        self.after(0 if immediate else 1000, load_next_empty_cell)

    def recompute_stats(self):
        # Recompute the statistics
        number_of_cells_correct = 0
        for cell in self.lcd_grid_views:
            if cell.values_matches_correctly:
                number_of_cells_correct += 1

        self.statistics_label.config(text=f"Statistics: {number_of_cells_correct}/{self.num_grid_items} correct = {number_of_cells_correct / self.num_grid_items * 100:.0f}%")


print("Loading LCD Editor...")
root = tkinter.Tk()
editor = LCDEditor(master=root, settings=settings)
root.winfo_toplevel().title(f"LCD Editing for {settings.absolute_movie_file}, {settings.identifier}")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)


# root.config(bg="darkblue")


def save_and_quit():
    editor.save_settings()
    root.quit()


menubar = tkinter.Menu(root)
mac_app_menu = tkinter.Menu(menubar, name="apple")
menubar.add_cascade(menu=mac_app_menu)
root.createcommand("tk::mac::Quit", save_and_quit)
root.protocol("WM_DELETE_WINDOW", save_and_quit)

root.mainloop()
