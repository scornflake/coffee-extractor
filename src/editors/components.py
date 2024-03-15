import tkinter
from settings import Settings


class EditingMixin:
    def get_int_from_entry(self, event, handler):
        value = event.widget.get()
        try:
            int_value = int(value)
            handler(int_value)
        except ValueError:
            print(f"Invalid value {value} for event...")
            return


class AreaEditingUI(tkinter.Frame, EditingMixin):
    def __init__(self, master, the_settings: Settings, update_preview_callback, change_on_enter_only: bool = False, *args, **kwargs):
        self.settings = the_settings
        self.change_on_enter_only = change_on_enter_only
        self.update_preview_callback = update_preview_callback
        super().__init__(master, *args, **kwargs)
        self.create_widgets()

    def create_widgets(self):
        # The labels
        left_label = tkinter.Label(self, text="Left")
        left_label.grid(row=0, column=0, sticky="e")
        top_label = tkinter.Label(self, text="Top")
        top_label.grid(row=1, column=0, sticky="e")
        width_label = tkinter.Label(self, text="Width")
        width_label.grid(row=2, column=0, sticky="e")
        height_label = tkinter.Label(self, text="Height")
        height_label.grid(row=3, column=0, sticky="e")
        skew_label = tkinter.Label(self, text="Skew")
        skew_label.grid(row=4, column=0, sticky="e")

        # The text boxes
        left_entry = tkinter.Entry(self)
        left_entry.grid(row=0, column=1, sticky="w")
        top_entry = tkinter.Entry(self)
        top_entry.grid(row=1, column=1, sticky="w")
        width_entry = tkinter.Entry(self)
        width_entry.grid(row=2, column=1, sticky="w")
        height_entry = tkinter.Entry(self)
        height_entry.grid(row=3, column=1, sticky="w")
        skew_entry = tkinter.Entry(self)
        skew_entry.grid(row=4, column=1, sticky="w")

        # Fill in UI with values from settings
        left_entry.insert(0, self.settings.digital_area.left)
        top_entry.insert(0, self.settings.digital_area.top)
        width_entry.insert(0, self.settings.digital_area.width)
        height_entry.insert(0, self.settings.digital_area.height)
        skew_entry.insert(0, self.settings.lcd_quad_skew)

        # When updating the value for skew_save it to settings and update the ui
        def update_skew(event):
            def update_skew_v(value):
                self.settings.lcd_quad_skew = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_skew_v)

        # If any of the left/top/width/height are modified, save to appropriate settings var and update UI
        def update_left(event):
            def update_left_v(value):
                self.settings.digital_area.left = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_left_v)

        def update_top(event):
            def update_top_v(value):
                self.settings.digital_area.top = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_top_v)

        def update_width(event):
            def update_width_v(value):
                self.settings.digital_area.width = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_width_v)

        def update_height(event):
            def update_height_v(value):
                self.settings.digital_area.height = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_height_v)

        binding_name = self.change_on_enter_only and "<Return>" or "<KeyRelease>"
        skew_entry.bind(binding_name, update_skew)
        left_entry.bind(binding_name, update_left)
        top_entry.bind(binding_name, update_top)
        width_entry.bind(binding_name, update_width)
        height_entry.bind(binding_name, update_height)


class TuningUI(tkinter.Frame, EditingMixin):
    def __init__(self, master, the_settings: Settings, update_preview_callback, change_on_enter_only: bool = False,
                 *args, **kwargs):
        self.settings = the_settings
        self.change_on_enter_only = change_on_enter_only
        self.update_preview_callback = update_preview_callback
        super().__init__(master, *args, **kwargs)
        self.create_widgets()

    def create_widgets(self):
        # Add UI controls to allow adjustment of the lower and upper HSV
        # The labels
        low_h_label = tkinter.Label(self, text="Low H")
        low_h_label.grid(row=2, column=0, sticky="e")
        low_s_label = tkinter.Label(self, text="Low S")
        low_s_label.grid(row=3, column=0, sticky="e")
        low_v_label = tkinter.Label(self, text="Low V")
        low_v_label.grid(row=4, column=0, sticky="e")
        upper_h_label = tkinter.Label(self, text="Upper H")
        upper_h_label.grid(row=5, column=0, sticky="e")
        upper_s_label = tkinter.Label(self, text="Upper S")
        upper_s_label.grid(row=6, column=0, sticky="e")
        upper_v_label = tkinter.Label(self, text="Upper V")
        upper_v_label.grid(row=7, column=0, sticky="e")
        lcd_blur_label = tkinter.Label(self, text="Blur")
        lcd_blur_label.grid(row=8, column=1, sticky="e")

        # The text boxes
        low_h_entry = tkinter.Entry(self)
        low_h_entry.grid(row=2, column=1, sticky="w")
        low_s_entry = tkinter.Entry(self)
        low_s_entry.grid(row=3, column=1, sticky="w")
        low_v_entry = tkinter.Entry(self)
        low_v_entry.grid(row=4, column=1, sticky="w")
        upper_h_entry = tkinter.Entry(self)
        upper_h_entry.grid(row=5, column=1, sticky="w")
        upper_s_entry = tkinter.Entry(self)
        upper_s_entry.grid(row=6, column=1, sticky="w")
        upper_v_entry = tkinter.Entry(self)
        upper_v_entry.grid(row=7, column=1, sticky="w")
        lcd_blur_entry = tkinter.Entry(self)
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
                self.update_preview_callback()

            self.get_int_from_entry(event, update_low_h_v)

        def update_low_s(event):
            def update_low_s_v(value):
                self.settings.low_threshold.s = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_low_s_v)

        def update_low_v(event):
            def update_low_v_v(value):
                self.settings.low_threshold.v = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_low_v_v)

        def update_upper_h(event):
            def update_upper_h_v(value):
                self.settings.upper_threshold.h = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_upper_h_v)

        def update_upper_s(event):
            def update_upper_s_v(value):
                self.settings.upper_threshold.s = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_upper_s_v)

        def update_upper_v(event):
            def update_upper_v_v(value):
                self.settings.upper_threshold.v = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_upper_v_v)

        def update_lcd_blur(event):
            def update_lcd_blur_v(value):
                self.settings.lcd_blur_amount = value
                self.update_preview_callback()

            self.get_int_from_entry(event, update_lcd_blur_v)

        binding_name = self.change_on_enter_only and "<Return>" or "<KeyRelease>"

        low_h_entry.bind(binding_name, update_low_h)
        low_s_entry.bind(binding_name, update_low_s)
        low_v_entry.bind(binding_name, update_low_v)
        upper_h_entry.bind(binding_name, update_upper_h)
        upper_s_entry.bind(binding_name, update_upper_s)
        upper_v_entry.bind(binding_name, update_upper_v)
        lcd_blur_entry.bind(binding_name, update_lcd_blur)
