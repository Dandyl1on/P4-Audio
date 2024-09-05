import customtkinter as ctk
from tkinter import filedialog, Canvas
from PIL import Image, ImageTk

# COLORS
BACKGROUND = '#282828'
STROKE = '#312E2E'
PANEL_BG = '#535353'
PANELS = '#353535'

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.geometry('1000x600')
        self.title('Image Viewer')
        self.minsize(800, 500)

        # Store scaling state for each image
        self.image_states = {}
        self.photo_images = {}  # Store references to PhotoImage objects

        # Layout
        self.rowconfigure(0, weight=1, uniform='a')
        self.rowconfigure(1, weight=1, uniform='a')
        self.rowconfigure(2, weight=1, uniform='a')
        self.rowconfigure(3, weight=1, uniform='a')
        self.rowconfigure(4, weight=1, uniform='a')
        self.columnconfigure(0, weight=1, uniform='a')
        self.columnconfigure(1, weight=1, uniform='a')
        self.columnconfigure(2, weight=1, uniform='a')
        self.columnconfigure(3, weight=1, uniform='a')
        self.columnconfigure(4, weight=1, uniform='a')
        self.columnconfigure(5, weight=1, uniform='a')
        self.columnconfigure(6, weight=1, uniform='a')

        FrameColor(self, color=PANEL_BG, x=0, y=0, strY=5, strX=7)
        self.image_import = ImageImport(self, self.import_image, color=BACKGROUND, x=1, y=1, strX=4, strY=3,
                                        paddingX=10, paddingY=10)
        FrameColor(self, color=PANELS, x=0, y=0, strY=1, strX=1)
        FrameColor(self, color=PANELS, x=0, y=4, strY=1, strX=1)
        FrameColor(self, color=PANELS, x=6, y=0, strY=1, strX=1)
        FrameColor(self, color=PANELS, x=6, y=4, strY=1, strX=1)

        # Create a container frame for the image and label
        self.image_container = ctk.CTkFrame(self, fg_color=BACKGROUND)
        self.image_container.grid(row=0, column=5, columnspan=2, rowspan=2, sticky='nsew', padx=10, pady=10)

        # Create label for the image name
        self.image_label = ctk.CTkLabel(self.image_container, text='', text_color='white', font=('Arial', 12))
        self.image_label.pack(side='top', fill='x')

        # Create canvas for the smaller image
        self.small_image_canvas = Canvas(self.image_container, background=BACKGROUND, bd=0, highlightthickness=0, relief='ridge')
        self.small_image_canvas.pack(side='top', fill='both', expand=True)

        # Create a frame to hold the buttons
        self.button_frame = ctk.CTkFrame(self.image_container, fg_color=BACKGROUND)
        self.button_frame.pack(side='bottom', fill='x', pady=(5, 0))

        self.play_audio = ctk.CTkButton(self.button_frame, text='Play audio', command=self.play_og_audio)
        self.view_image = ctk.CTkButton(self.button_frame, text='View image', command=self.view_og_image)

        self.hide_elements()  # Initially hide elements

        # Initialize menu
        self.menu = Menu(self)
        self.menu.grid(row=0, column=1, columnspan=2, sticky='nsew', padx=10, pady=10)
        self.menu.grid_remove()  # Hide the menu initially

        # Add a NewTab button
        self.new_tab_button = NewTab(self, self.create_new_tab)
        self.new_tab_button.grid(row=0, column=3, sticky='ne', padx=10, pady=10)
        self.new_tab_button.grid_remove()  # Hide initially

        self.current_tab = None  # Keep track of the currently selected tab
        self.current_image = None  # Keep track of the currently displayed image

        # Start polling for tab changes
        self.check_tab_change()

        self.mainloop()

    def import_image(self, path):
        self.original = Image.open(path)
        image_name = self.original.filename.split('/')[-1]  # Extract the image name from the file path

        # Initialize image state
        self.image_states[image_name] = {
            'image': self.original,
            'ratio': self.original.size[0] / self.original.size[1],
            'width': self.original.size[0],
            'height': self.original.size[1]
        }

        self.current_image = self.image_states[image_name]

        # Convert image to PhotoImage and store the reference
        self.photo_images[image_name] = ImageTk.PhotoImage(self.original)

        self.image_import.grid_forget()
        self.image_output = ImageOutput(self, self.resize_image)
        self.close_button = CloseOutput(self, self.Close_edit)

        # Show the elements
        self.show_elements()

        # Update the label with the image name
        self.image_label.configure(text=image_name)

        # Display the smaller image
        self.display_small_image()

        # Create a new tab with the image name and the image itself
        self.menu.add_tab(image_name, self.original)
        self.menu.grid()  # Show the menu

        # Show the NewTab button
        self.new_tab_button.grid()

        # Set the current tab to the new one
        self.current_tab = image_name
        self.menu.set_current(self.current_tab)

    def display_small_image(self):
        if not self.current_image:
            return

        # Update the canvas dimensions
        self.small_image_canvas.update_idletasks()  # Ensure the canvas dimensions are updated
        canvas_width = self.small_image_canvas.winfo_width()
        canvas_height = self.small_image_canvas.winfo_height()

        # Retrieve current image state
        image_state = self.current_image
        image_ratio = image_state['ratio']

        # Calculate the new size while maintaining aspect ratio
        if canvas_width / canvas_height > image_ratio:
            new_height = canvas_height
            new_width = int(new_height * image_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / image_ratio)

        # Resize the image
        resized_image = image_state['image'].resize((new_width, new_height))

        # Convert resized image to PhotoImage and store the reference
        self.photo_images['small_image'] = ImageTk.PhotoImage(resized_image)

        # Clear the canvas and display the image centered
        self.small_image_canvas.delete('all')
        self.small_image_canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2,
                                             anchor='nw', image=self.photo_images['small_image'])

    def Close_edit(self):
        # Hide image and close button
        self.image_output.grid_forget()
        self.close_button.place_forget()
        self.hide_elements()
        self.small_image_canvas.delete("all")  # Clear the smaller image canvas
        self.image_import = ImageImport(self, self.import_image, color=BACKGROUND, x=1, y=1, strY=3, strX=4,
                                        paddingX=10, paddingY=10)

    def create_new_tab(self):
        """Open a file dialog to select an image, then create a new tab with the image's name."""
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")])
        if path:
            # Import the image
            self.import_image(path)

    def resize_image(self, event):
        # Current canvas ratio
        canvas_ratio = event.width / event.height

        # Resize
        if canvas_ratio > self.current_image['ratio']:  # If canvas is wider than the image
            self.current_image['height'] = int(event.height)
            self.current_image['width'] = int(self.current_image['height'] * self.current_image['ratio'])
        else:  # If canvas is taller than the image
            self.current_image['width'] = int(event.width)
            self.current_image['height'] = int(self.current_image['width'] / self.current_image['ratio'])

        self.place_image()

    def place_image(self):
        self.image_output.delete('all')
        resized_image = self.current_image['image'].resize((self.current_image['width'], self.current_image['height']))
        # Convert resized image to PhotoImage and store the reference
        self.photo_images['main_image'] = ImageTk.PhotoImage(resized_image)
        self.image_output.create_image(self.image_output.winfo_width() / 2, self.image_output.winfo_height() / 2, image=self.photo_images['main_image'])

    def show_elements(self):
        self.image_label.pack(side='top', fill='x')
        self.small_image_canvas.pack(side='top', fill='both', expand=True)
        self.play_audio.pack(side='left', padx=5, pady=5)
        self.view_image.pack(side='left', padx=5, pady=5)

    def hide_elements(self):
        self.image_label.pack_forget()
        self.small_image_canvas.pack_forget()
        self.play_audio.pack_forget()
        self.view_image.pack_forget()

    def play_og_audio(self):
        print("Original audio is being played")

    def view_og_image(self):
        print("Original image is being viewed")

    def check_tab_change(self):
        """Check for tab changes and update if necessary."""
        current_tab = self.menu.get_current()
        if current_tab != self.current_tab:
            self.current_tab = current_tab
            self.current_image = self.image_states.get(current_tab)
            if self.current_image:
                self.place_image()
                self.display_small_image()  # Update the small image when tab changes

        self.after(100, self.check_tab_change)  # Check every 100 milliseconds

class FrameColor(ctk.CTkFrame):
    def __init__(self, parent, color, x, y, strX, strY):
        super().__init__(master=parent, fg_color=color)
        self.grid(column=x, columnspan=strX, row=y, rowspan=strY, sticky='nsew')


class ImageImport(ctk.CTkFrame):
    def __init__(self, parent, import_func, color, x, y, strX, strY, paddingX, paddingY):
        super().__init__(master=parent, fg_color=color)
        self.grid(column=x, columnspan=strX, row=y, rowspan=strY, padx=paddingX, pady=paddingY, sticky='nsew')
        self.import_func = import_func

        ctk.CTkButton(self, text='Open image', command=self.open_dialog).pack(expand=True)

    def open_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")])
        if path:
            self.import_func(path)


class ImageOutput(Canvas):
    def __init__(self, parent, resize_image):
        super().__init__(master=parent, background=BACKGROUND, bd=0, highlightthickness=0, relief='ridge')
        self.grid(row=1, column=1, columnspan=4, rowspan=3, padx=13, pady=13, sticky='nsew')
        self.bind('<Configure>', resize_image)


class CloseOutput(ctk.CTkButton):
    def __init__(self, parent, close_func):
        super().__init__(master=parent, command=close_func, text='x', text_color='white', fg_color='transparent',
                         width=40, height=40,
                         corner_radius=0,
                         hover_color='red'
                         )
        self.place(relx=0.99, rely=0.01, anchor='ne')


class Menu(ctk.CTkTabview):
    def __init__(self, parent):
        super().__init__(master=parent)
        self.tabs = {}  # Dictionary to store tab names and their images

    def add_tab(self, tab_name, image):
        """Add a new tab with the given name and store the image."""
        self.add(tab_name)
        self.tabs[tab_name] = image
        self.set(tab_name)

    def set_current(self, tab_name):
        """Select the tab with the given name."""
        if tab_name in self.tabs:
            self.set(tab_name)  # This should set the active tab
            # No direct event handling; state change handled in polling method
        else:
            print(f"Tab '{tab_name}' does not exist.")

    def get_image_for_tab(self, tab_name):
        """Get the image associated with the given tab."""
        return self.tabs.get(tab_name)

    def get_current(self):
        """Get the name of the currently selected tab."""
        return self.get()  # Assuming `get` returns the current tab name


class NewTab(ctk.CTkButton):
    def __init__(self, parent, new_tab_func):
        super().__init__(master=parent, command=new_tab_func, text='+', text_color='white', fg_color='transparent',
                         width=30, height=30,
                         corner_radius=0,
                         hover_color=PANELS
                         )
        self.grid(row=0, column=3, sticky='ne', padx=10, pady=10)  # Adjusted size and position

# Create a global instance of the app to access from the Menu class
app_instance = App()
