"""Run viewer.py in the project's Python environment to browse data/."""
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from wafer_data import (discover, capture_metadata, read_frame, pixel_value,
                        display_limits, render_pixels, image_coordinates)


class Viewer(tk.Tk):
    def __init__(self, folder: Path):
        super().__init__()
        self.title('WaferPL — TIFF viewer')
        self.geometry('1400x900')
        self.minsize(950, 620)
        self.frame = None
        self.capture = None
        self.entries = {}
        self.photo = None
        self.display_size = (0, 0)
        self.resize_job = None
        self.folder = folder
        self.low = tk.StringVar(value='0')
        self.high = tk.StringVar(value='65535')
        self.gamma = tk.StringVar(value='1')
        self.invert = tk.BooleanVar()
        self.zoom = tk.StringVar(value='Вписать')
        self.page = tk.StringVar(value='1')
        self.status = tk.StringVar(value='Выберите TIFF')
        self.heading = tk.StringVar(value='WaferPL')
        self.build_ui()
        self.after(50, self.scan)

    def build_ui(self):
        top = ttk.Frame(self, padding=6)
        top.pack(fill='x')
        ttk.Button(top, text='Открыть папку…', command=self.choose_folder).pack(side='left')
        ttk.Button(top, text='Обновить', command=self.scan).pack(side='left', padx=6)
        ttk.Label(top, textvariable=self.heading).pack(side='left', padx=8)
        split = ttk.Panedwindow(self, orient='horizontal')
        split.pack(fill='both', expand=True)
        left = ttk.Frame(split, width=380)
        right = ttk.Frame(split)
        split.add(left, weight=1)
        split.add(right, weight=4)
        self.tree = ttk.Treeview(left, show='tree', selectmode='browse')
        tree_scroll = ttk.Scrollbar(left, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')
        self.tree.bind('<<TreeviewSelect>>', self.select)
        controls = ttk.Frame(right, padding=6)
        controls.pack(fill='x')
        for label, variable in (('Мин.', self.low), ('Макс.', self.high), ('Гамма', self.gamma)):
            ttk.Label(controls, text=label).pack(side='left', padx=3)
            entry = ttk.Entry(controls, textvariable=variable, width=9)
            entry.pack(side='left')
            entry.bind('<Return>', lambda event: self.draw())
        ttk.Button(controls, text='Применить', command=self.draw).pack(side='left', padx=4)
        ttk.Checkbutton(controls, text='Инверсия', variable=self.invert, command=self.draw).pack(side='left')
        tools = ttk.Frame(right, padding=(6, 0, 6, 6))
        tools.pack(fill='x')
        ttk.Button(tools, text='Авто 1–99,5%', command=lambda: self.set_limits(True)).pack(side='left')
        ttk.Button(tools, text='Весь диапазон', command=lambda: self.set_limits(False)).pack(side='left', padx=4)
        ttk.Label(tools, text='Масштаб').pack(side='left', padx=4)
        zoom = ttk.Combobox(tools, textvariable=self.zoom, state='readonly', width=9,
                            values=('Вписать', '25%', '50%', '100%', '200%'))
        zoom.pack(side='left')
        zoom.bind('<<ComboboxSelected>>', lambda event: self.draw())
        ttk.Label(tools, text='Страница').pack(side='left', padx=6)
        self.pages = ttk.Combobox(tools, textvariable=self.page, state='readonly', width=5)
        self.pages.pack(side='left')
        self.pages.bind('<<ComboboxSelected>>', lambda event: self.load())
        viewport = ttk.Frame(right)
        viewport.pack(fill='both', expand=True)
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(viewport, background='#20242b', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        sx = ttk.Scrollbar(viewport, orient='horizontal', command=self.canvas.xview)
        sy = ttk.Scrollbar(viewport, orient='vertical', command=self.canvas.yview)
        sx.grid(row=1, column=0, sticky='ew')
        sy.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        self.canvas.bind('<Motion>', self.inspect_pixel)
        self.canvas.bind('<Button-1>', self.inspect_pixel)
        self.canvas.bind('<Leave>', lambda event: self.status.set('Наведите курсор на изображение для чтения пикселя'))
        self.canvas.bind('<Configure>', self.schedule_resize)
        self.canvas.bind('<ButtonPress-2>', lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind('<B2-Motion>', lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        meta_frame = ttk.Frame(right)
        meta_frame.pack(fill='x')
        self.metadata = tk.Text(meta_frame, height=9, wrap='word', state='disabled')
        meta_scroll = ttk.Scrollbar(meta_frame, command=self.metadata.yview)
        self.metadata.configure(yscrollcommand=meta_scroll.set)
        self.metadata.pack(side='left', fill='x', expand=True)
        meta_scroll.pack(side='right', fill='y')
        ttk.Label(self, textvariable=self.status, padding=6).pack(fill='x')

    def choose_folder(self):
        folder = filedialog.askdirectory(initialdir=str(self.folder))
        if folder:
            self.folder = Path(folder)
            self.scan()

    def clear_image(self):
        self.frame = None
        self.photo = None
        self.display_size = (0, 0)
        self.canvas.delete('all')
        self.canvas.configure(scrollregion=(0, 0, 0, 0))
        self.pages.configure(values=())
        self.show_metadata('')
        self.heading.set('WaferPL')
        self.status.set('Выберите TIFF')

    def scan(self):
        self.clear_image()
        self.capture = None
        self.tree.delete(*self.tree.get_children())
        self.entries.clear()
        try:
            if not self.folder.is_dir():
                raise ValueError(f'Папка не найдена: {self.folder}')
            captures = discover(self.folder)
        except (OSError, ValueError) as error:
            messagebox.showerror('Чтение папки', str(error))
            return
        groups = {}
        initial = None
        for capture in captures:
            key = (capture.path.parent, capture.sample)
            if key not in groups:
                relative = capture.path.parent.relative_to(self.folder)
                label = f'{capture.sample} | {relative}'
                groups[key] = self.tree.insert('', 'end', text=label, open=True)
            kind = 'RAW16' if capture.raw else 'TIFF экспорт'
            node = self.tree.insert(groups[key], 'end', text=f'[{kind}] {capture.path.name}')
            self.entries[node] = capture
            if initial is None or (capture.raw and '261009_Mono16' in str(capture.path.parent)
                                   and '261009_Mono16' not in str(self.entries[initial].path.parent)):
                initial = node
        self.status.set(f'TIFF: {len(captures)}. RAW16 — для измерений; экспорт — для просмотра.')
        if initial:
            self.tree.selection_set(initial)
            self.tree.see(initial)
        else:
            self.show_metadata('В выбранной папке и её подпапках TIFF не найдены.')

    def select(self, event=None):
        selection = self.tree.selection()
        if not selection or selection[0] not in self.entries:
            return
        self.capture = self.entries[selection[0]]
        self.page.set('1')
        self.load()

    def load(self):
        if self.capture is None:
            return
        page_index = int(self.page.get()) - 1
        self.clear_image()
        try:
            self.frame = read_frame(self.capture.path, page_index)
            self.pages.configure(values=tuple(str(i + 1) for i in range(self.frame.page_count)))
            self.page.set(str(page_index + 1))
            self.invert.set(self.frame.white_is_zero)
            self.gamma.set('1')
            pixels = self.frame.pixels
            h, w = pixels.shape[:2]
            kind = 'RAW16: значения TIFF' if self.capture.raw else 'ЭКСПОРТ: значения TIFF, не исходный сигнал прибора'
            self.heading.set(f'{self.capture.sample} — {self.capture.mode} — {w}×{h}, {pixels.dtype}')
            details = [str(self.capture.path), kind,
                       f'Образец: {self.capture.sample} ({self.capture.sample_source})',
                       f'Режим из имени файла: {self.capture.mode}',
                       'Контраст, гамма и инверсия меняют только отображение.',
                       'SCN: параметры протокола; числовые коды не преобразованы в длины волн.',
                       'exposure_time приведён как в SCN; единица в XML не указана.']
            details.extend(f'{key}: {value}' for key, value in self.frame.tags.items())
            details.extend(f'{key}: {value}' for key, value in capture_metadata(self.capture).items())
            self.show_metadata('\n'.join(details))
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
            self.set_limits(True)
        except Exception as error:
            self.clear_image()
            self.status.set('Не удалось открыть выбранный TIFF')
            messagebox.showerror('Чтение TIFF', f'{self.capture.path.name}\n{error}')

    def show_metadata(self, text):
        self.metadata.configure(state='normal')
        self.metadata.delete('1.0', 'end')
        self.metadata.insert('1.0', text)
        self.metadata.configure(state='disabled')

    def set_limits(self, auto):
        if self.frame is None:
            return
        low, high = display_limits(self.frame.pixels, auto)
        self.low.set(f'{low:.8g}')
        self.high.set(f'{high:.8g}')
        self.draw()

    def schedule_resize(self, event=None):
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(180, self.finish_resize)

    def finish_resize(self):
        self.resize_job = None
        if self.zoom.get() == 'Вписать':
            self.draw()

    def draw(self):
        if self.frame is None:
            return
        try:
            low, high, gamma = (float(v.get().replace(',', '.')) for v in (self.low, self.high, self.gamma))
            rendered = render_pixels(self.frame.pixels, low, high, gamma, self.invert.get())
            h, w = rendered.shape[:2]
            if self.zoom.get() == 'Вписать':
                scale = min(max(1, self.canvas.winfo_width()) / w,
                            max(1, self.canvas.winfo_height()) / h)
            else:
                scale = float(self.zoom.get().rstrip('%')) / 100
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            image = Image.fromarray(rendered).resize(size, Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(image)
        except (ValueError, MemoryError, tk.TclError) as error:
            self.status.set(f'Не удалось обновить отображение: {error}')
            return
        self.photo = photo
        self.display_size = size
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, *size))
        self.status.set(f'Шкала {low:g}…{high:g}; гамма {gamma:g}. Наведите курсор для чтения исходного пикселя.')

    def inspect_pixel(self, event):
        if self.frame is None:
            return
        pixels = self.frame.pixels
        coordinates = image_coordinates(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y),
                                        self.display_size, (pixels.shape[1], pixels.shape[0]))
        if coordinates is None:
            self.status.set('За пределами изображения')
            return
        x, y = coordinates
        value = pixel_value(pixels, x, y)
        label = 'Значение RAW16' if self.capture.raw else 'Значение экспорта TIFF'
        self.status.set(f'x={x}, y={y} (с нуля) | {label}: {value} | {pixels.dtype} | страница {self.page.get()}')


if __name__ == '__main__':
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'data'
    Viewer(folder).mainloop()
