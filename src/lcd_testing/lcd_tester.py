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

    def load_frame(self):
        path_to_cached_frame = f"/tmp/lcd_editor_frame_{self.frame_number}.png"
        if os.path.exists(path_to_cached_frame):
            self.frame = cv2.imread(path_to_cached_frame)
        else:
            self.frame = self.movie.get_frame_number(self.frame_number)
            cv2.imwrite(path_to_cached_frame, self.frame)

    def create_widgets(self):
        self.lcd_image = tkinter.Canvas(self, width=320, height=150, bg="black")
        self.lcd_image.grid(row=0, column=0, columnspan=5, sticky="e")

        self.actual_label = tkinter.Label(self, text=f"Actual: {self.index + 1}")
        self.actual_label.grid(row=1, column=0)
        self.actual_entry = tkinter.Entry(self)
        self.actual_entry.grid(row=1, column=1)
        value_for_lcd = settings.get_true_value_for_lcd(self.index)
        if value_for_lcd is not None:
            self.actual_entry.insert(0, value_for_lcd)

        self.indicator_label = tkinter.Canvas(self, width=20, height=20)
        self.indicator_label.grid(row=1, column=2)

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

    def update_tess_value_label(self):
        self.tesseract_label.config(text=f"Tess: {self.tesseract_value}")

        is_correct_value = False
        try:
            is_correct_value = int(self.tesseract_value) == int(settings.get_true_value_for_lcd(self.index))
        except ValueError:
            pass
        # self.indicator_label is green if correct, red otherwise
        self.indicator_label.delete("all")
        if is_correct_value:
            self.indicator_label.create_oval(0, 0, 20, 20, fill="green")
        else:
            self.indicator_label.create_oval(0, 0, 20, 20, fill="red")

    def refresh_image(self, get_new_frame: bool = False):
        settings = self.settings_provider()

        if self.frame is None or get_new_frame:
            self.frame = self.movie.get_frame_number(self.frame_number)

        extracted_image = extract_lcd_and_ready_for_tesseract(self.frame, 0, settings)
        if extracted_image is None:
            self.tesseract_value = "???"
        else:
            self.tesseract_value = parse_int_via_tesseract(extracted_image)
            if self.tesseract_value is None:
                self.tesseract_value = "???"

        print(f"Value for {self.index} is {self.tesseract_value}")

        pil_image = Image.fromarray(extracted_image)
        photo_image = ImageTk.PhotoImage(pil_image)
        self.lcd_image.create_image(0, 0, image=photo_image, anchor=tkinter.NW)
        self.lcd_image.image = photo_image

        self.update_tess_value_label()


class LCDEditor(tkinter.Frame):
    def __init__(self, master=None, cnf=None, **kwargs):
        self.statistics_label = None
        self.lcd_grid_views = None
        self.lcd_preview_canvas = None
        self.properties_panel = None
        self.grid_cols = 5
        self.grid_rows = 5

        self.movie = None
        if cnf is None:
            cnf = {}
        # Pop settings from kwargs
        self.settings = kwargs.pop("settings")
        super().__init__(master, cnf, width=1200, height=600, **kwargs)
        self.grid(row=0, column=0)
        self.create_widgets()

    @property
    def num_grid_items(self):
        return self.grid_cols * self.grid_rows

    def create_widgets(self):
        self.lcd_preview_canvas = None
        self.movie = Movie(self.settings.absolute_movie_file)

        # Create a properties panel, 100px wide, taking up the full left size, full height
        self.properties_panel = tkinter.Frame(self)
        self.properties_panel.grid(row=0, column=0)
        self.create_properties_panel()

        # Create a 5x5 grid, that we can fill in with video previews
        self.lcd_preview_canvas = tkinter.Frame(self)
        self.lcd_preview_canvas.grid(row=0, column=1)

        self.lcd_grid_views = []

        # Lets look at 5*5 video frames from the movie
        total_frames = self.movie.frame_count
        frames_per_index = total_frames / 25
        for i in range(self.num_grid_items):
            column = (i % self.grid_rows)
            row = i // self.grid_cols

            frame_number = int(i * frames_per_index)
            cell = LCDCell(frame_number=frame_number, movie=self.movie, settings_provider=self.settings_provider, master=self.lcd_preview_canvas, index=i)
            cell.grid(row=row, column=column)
            self.lcd_grid_views.append(cell)

        # Now a row underneath, for statistics
        self.statistics_label = tkinter.Label(self, text="Statistics")
        self.statistics_label.grid(row=1, column=0, columnspan=2)

        self.slow_lcd_refresh()

    def settings_provider(self):
        return self.settings

    def save_settings(self):
        self.settings.write_to_file()

    def create_properties_panel(self):
        area_ui = AreaEditingUI(master=self.properties_panel, the_settings=self.settings, update_preview_callback=self.update_everything)
        area_ui.grid(row=0, column=0, sticky="nw")

        tuning_props = TuningUI(master=self.properties_panel, update_preview_callback=self.update_for_tesseract_and_get_new_vaule, the_settings=self.settings)
        tuning_props.grid(row=1, column=0, sticky="nw")

        # Followed by a 'save' button
        save_button = tkinter.Button(self.properties_panel, text="Save", command=self.save_settings)
        save_button.grid(row=100, column=0)

    def update_everything(self):
        for cell in self.lcd_grid_views:
            cell.refresh_image()

    def update_for_tesseract_and_get_new_vaule(self):
        for cell in self.lcd_grid_views:
            cell.refresh_image()

    def slow_lcd_refresh(self):
        # Refresh the images slowly, so the UI is still responsive
        def load_next_empty_cell():
            for cell in self.lcd_grid_views:
                if cell.tesseract_value is None:
                    cell.refresh_image()
                    self.after(50, load_next_empty_cell)
                    return

        self.after(1000, load_next_empty_cell)


print("Loading LCD Editor...")
root = tkinter.Tk()
editor = LCDEditor(master=root, settings=settings)
root.winfo_toplevel().title(f"LCD Editing for {settings.absolute_movie_file}, {settings.identifier}")


def save_and_quit():
    editor.save_settings()
    root.quit()


menubar = tkinter.Menu(root)
mac_app_menu = tkinter.Menu(menubar, name="apple")
menubar.add_cascade(menu=mac_app_menu)
root.createcommand("tk::mac::Quit", save_and_quit)
root.protocol("WM_DELETE_WINDOW", save_and_quit)

root.mainloop()
