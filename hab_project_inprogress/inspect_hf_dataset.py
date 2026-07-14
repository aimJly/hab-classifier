"""
This is separate from the other files that retrieve the data and train the model. So the code here isn't needed to 
run the rest of the project. This project brightens the images to show the shape of the algal bloom 
and shows multiple images side by side in a separate window. 
"""
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk


class ImageViewer(tk.Tk):
    """
    A simple graphical desktop application built with Tkinter.
    It allows you to visually inspect the raw multispectral satellite image bands 
    (like B06, B07, B08) for a specific geographic sample.
    """
    def __init__(self, folder: str, limit: int = 5):
        """
        Initializes the application window, locates the required satellite band files 
        within the target folder, and builds the basic user interface layout.
        """
        super().__init__()
        self.title("HAB Sample Images")
        self.geometry("1600x700")

        self.folder_path = Path(folder)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        # Define the specific satellite sensor bands we care about looking at
        self.band_names = ["B06_raw", "B07_raw", "B08_raw", "B8A_raw", "B11_raw", "B12_raw"]
        self.image_paths = []
        
        # Scan the folder to find existing files that match our target bands
        for band_name in self.band_names:
            for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                candidate = self.folder_path / f"{band_name}{ext}"
                if candidate.exists():
                    self.image_paths.append(candidate)
                    break

        if not self.image_paths:
            raise FileNotFoundError(f"No requested band images found in {self.folder_path}")

        self.photo_images = []
        self._load_images()

        # Set up the visual UI canvas where the images will be drawn
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        self.label = ttk.Label(main, text=self.folder_path.name, font=("Segoe UI", 12, "bold"))
        self.label.pack(anchor=tk.W, pady=(0, 8))

        self.canvas = tk.Canvas(main, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.show_current()

    def _load_images(self):
        """
        Opens the discovered image files, converts them to standard RGB, and shrinks them 
        into smaller 220x220 thumbnails so they fit neatly on the screen without crashing your memory.
        """
        for img_path in self.image_paths:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img.thumbnail((220, 220))
                self.photo_images.append(ImageTk.PhotoImage(img))

    def show_current(self):
        """
        Draws the loaded thumbnails onto the canvas in a grid layout (max 3 images per row).
        It also adds the filename as a white text label beneath each image so you know which band is which.
        """
        self.canvas.delete("all")
        x = 10
        y = 10
        for idx, img in enumerate(self.photo_images):
            self.canvas.create_image(x, y, anchor=tk.NW, image=img)
            self.canvas.create_text(x, y + 220 + 8, anchor=tk.NW, text=self.image_paths[idx].name, fill="white")
            
            # Shift the next image to the right, or drop down to a new row if we've placed 3 images
            x += 240
            if idx % 3 == 2:
                x = 10
                y += 260
                
        # Tell the canvas how far it needs to be able to scroll
        self.canvas.config(scrollregion=self.canvas.bbox("all"))


# Entry point: Runs the GUI application when you execute this script directly
if __name__ == "__main__":
    # Hardcoded path pointing to a specific downloaded sample in the Hugging Face cache
    sample_folder = r"C:\Users\mly72\.cache\huggingface\hub\datasets--kostaspic--alfitrite-inland-waters-hab-sentinel2\snapshots\88dda2242db358dc2464ef14a00e548acbd8d558\data\1000_2019-05-15"
    app = ImageViewer(sample_folder)
    app.mainloop()