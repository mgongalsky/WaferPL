"""RAW-only sample navigation and additive channel overlays."""
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QColorDialog, QComboBox, QDockWidget, QHBoxLayout,
                             QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from viewer import Viewer
from wafer_data import (capture_metadata, discover, display_limits, image_coordinates,
                        pixel_value, read_frame, render_pixels)


@dataclass
class ChannelStyle:
    low: float
    high: float
    gamma: float
    invert: bool
    color: tuple[int, int, int]


def compose_overlay(layers):
    """Combine colors with per-component headroom instead of clipping sums."""
    if not layers:
        return None
    frames = [frame for frame, style in layers]
    height, width = max((f.pixels.shape for f in frames), key=lambda shape: shape[0] * shape[1])
    orientation = frames[0].tags.get('Orientation', '1')
    for frame in frames:
        h, w = frame.pixels.shape
        if w * height != width * h:
            raise ValueError('У каналов разные пропорции кадра: оверлей требует выравнивания')
        if frame.tags.get('Orientation', '1') != orientation:
            raise ValueError('У каналов разная ориентация TIFF')
    result = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.maximum(1.0, np.sum([np.array(style.color) / 255.0 for frame, style in layers], axis=0))
    for frame, style in layers:
        gray = render_pixels(frame.pixels, style.low, style.high, style.gamma, style.invert)
        h, w = gray.shape
        ys = np.arange(height, dtype=np.int64) * h // height
        xs = np.arange(width, dtype=np.int64) * w // width
        scaled = gray[ys[:, None], xs[None, :]].astype(np.float32)
        for component in range(3):
            result[..., component] += scaled * (style.color[component] / 255.0 / weights[component])
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


class ChannelViewer(Viewer):
    def __init__(self, folder):
        self.samples = {}
        self.sample_keys = []
        self.sample_key = None
        self.selected_modes = set()
        self.styles = {}
        self.colors = {}
        self.layers = {}
        self.cache = {}
        self.choices = {}
        self.page_indices = {}
        self.active_mode = None
        self.channel_items = {}
        self.output_size = (0, 0)
        self.updating = False
        self.initialized = False
        super().__init__(folder)

    def build_ui(self):
        from PyQt6.QtWidgets import (
            QCheckBox, QFormLayout, QLineEdit, QPlainTextEdit, QScrollArea,
            QSizePolicy, QTabWidget, QToolButton,
        )
        from viewer import ImageView

        self.fixed_modes = set()
        self.style_frames = {}
        self.auto_settings = {}
        self.setWindowTitle('WaferPL — RAW channels')
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        navigation = QHBoxLayout()
        navigation.setSpacing(2)
        layout.addLayout(navigation)

        def arrow(text, tooltip, callback):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            navigation.addWidget(button)
            return button

        self.sidebar_toggle = QToolButton()
        self.sidebar_toggle.setText('☰')
        self.sidebar_toggle.setToolTip('Показать или скрыть настройки')
        self.sidebar_toggle.setCheckable(True)
        navigation.addWidget(self.sidebar_toggle)
        arrow('←', 'Предыдущий образец — Shift+←', lambda: self.move_sample(-1))
        self.sample_selector = QComboBox()
        self.sample_selector.setToolTip('Образец')
        self.sample_selector.setMinimumWidth(0)
        self.sample_selector.setMinimumContentsLength(5)
        self.sample_selector.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.sample_selector.activated.connect(lambda index: self.tree.setCurrentItem(self.tree.topLevelItem(index)))
        navigation.addWidget(self.sample_selector, 2)
        arrow('→', 'Следующий образец — Shift+→', lambda: self.move_sample(1))
        navigation.addSpacing(6)
        arrow('◀', 'Предыдущий канал — Alt+←', lambda: self.cycle_channel(-1))
        self.channel_selector = QComboBox()
        self.channel_selector.setToolTip('Одиночный канал; состав оверлея выбирается в боковой панели')
        self.channel_selector.setMinimumWidth(0)
        self.channel_selector.setMinimumContentsLength(5)
        self.channel_selector.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.channel_selector.activated.connect(self.choose_single_channel)
        navigation.addWidget(self.channel_selector, 1)
        arrow('▶', 'Следующий канал — Alt+→', lambda: self.cycle_channel(1))
        self.view = ImageView()
        self.view.setMinimumSize(0, 0)
        layout.addWidget(self.view, 1)
        self.view.pixel_hovered.connect(self.inspect_pixel)
        self.view.pointer_left.connect(
            lambda: self.statusBar().showMessage('Наведите курсор на изображение для чтения пикселя'))

        # Keep the sample selection model used by navigation, without a second visible list.
        self.tree = QTreeWidget(self)
        self.tree.hide()
        self.tree.currentItemChanged.connect(self.select)
        self.sidebar = QDockWidget('Управление', self)
        self.sidebar.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.sidebar.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.sidebar.setMinimumWidth(210)
        self.sidebar.setMaximumWidth(290)
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setMinimumWidth(0)
        self.sidebar.setWidget(self.sidebar_tabs)

        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(6, 6, 6, 6)
        self.heading = QLabel('WaferPL')
        self.heading.setWordWrap(True)
        self.heading.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        settings_layout.addWidget(self.heading)
        self.add_button(settings_layout, 'Открыть папку…', self.choose_folder)
        self.add_button(settings_layout, 'Обновить', self.scan)
        self.auto_checkbox = QCheckBox('Авто по снимку')
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.setToolTip(
            'Процентили 0,5–99,5% и гамма по медиане. Отключите для фиксированной шкалы канала.')
        self.auto_checkbox.clicked.connect(self.set_auto_mode)
        settings_layout.addWidget(self.auto_checkbox)
        form = QFormLayout()
        settings_layout.addLayout(form)
        self.low = QLineEdit('0')
        self.high = QLineEdit('65535')
        self.gamma = QLineEdit('1')
        for label, entry in (('Минимум', self.low), ('Максимум', self.high), ('Гамма', self.gamma)):
            entry.setMinimumWidth(0)
            form.addRow(label, entry)
            entry.returnPressed.connect(self.draw)
        self.invert = QCheckBox('Инверсия')
        self.invert.clicked.connect(self.draw)
        settings_layout.addWidget(self.invert)
        self.add_button(settings_layout, 'Применить', self.draw)
        self.add_button(settings_layout, 'Авто: детали', lambda: self.set_limits(True))
        self.add_button(settings_layout, 'Весь диапазон', lambda: self.set_limits(False))
        display_form = QFormLayout()
        settings_layout.addLayout(display_form)
        self.zoom = QComboBox()
        self.zoom.addItems(['Вписать', '25%', '50%', '100%', '200%'])
        self.zoom.currentTextChanged.connect(self.view.set_zoom)
        display_form.addRow('Масштаб', self.zoom)
        self.pages = QComboBox()
        self.pages.activated.connect(self.load)
        display_form.addRow('Страница', self.pages)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        settings_layout.addWidget(self.notice)
        settings_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(0, 0)
        scroll.setWidget(settings)
        self.sidebar_tabs.addTab(scroll, 'Шкала')

        channels_page = QWidget()
        channels_layout = QVBoxLayout(channels_page)
        channels_layout.setContentsMargins(6, 6, 6, 6)
        self.channels = QTreeWidget()
        self.channels.setHeaderLabels(['Канал', 'Цвет'])
        self.channels.setMinimumSize(0, 0)
        self.channels.setColumnWidth(0, 150)
        self.channels.itemChanged.connect(self.toggle_channel)
        self.channels.currentItemChanged.connect(self.activate_channel)
        self.channels.itemDoubleClicked.connect(self.pick_color)
        channels_layout.addWidget(self.channels, 1)
        self.color_button = self.add_button(channels_layout, 'Выбрать цвет…', self.pick_active_color)
        self.color_button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        channels_layout.addWidget(QLabel('Съёмка активного канала:'))
        self.capture_choice = QComboBox()
        self.capture_choice.setMinimumContentsLength(8)
        self.capture_choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.capture_choice.activated.connect(self.change_capture)
        channels_layout.addWidget(self.capture_choice)
        self.add_button(channels_layout, 'Только активный канал', self.only_active)
        self.sidebar_tabs.addTab(channels_page, 'Каналы')
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.metadata.setMinimumSize(0, 0)
        self.sidebar_tabs.addTab(self.metadata, 'Данные')
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sidebar)
        self.resizeDocks([self.sidebar], [240], Qt.Orientation.Horizontal)
        self.sidebar_toggle.setChecked(True)
        self.sidebar_toggle.toggled.connect(self.sidebar.setVisible)
        self.sidebar.visibilityChanged.connect(self.sidebar_toggle.setChecked)

        self.shortcuts = []
        for key, delta in [('Shift+Left', -1), ('Shift+Up', -1), ('Shift+Right', 1), ('Shift+Down', 1)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda step=delta: self.move_sample(step))
            self.shortcuts.append(shortcut)
        for key, delta in [('Alt+Left', -1), ('Alt+Right', 1)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda step=delta: self.cycle_channel(step))
            self.shortcuts.append(shortcut)
        self.statusBar().showMessage('Выберите RAW16')

    def scan(self):
        self.scan_timer.stop()
        previous = self.sample_key
        self.updating = True
        self.tree.clear()
        self.channels.clear()
        self.channel_items.clear()
        self.samples.clear()
        self.cache.clear()
        self.layers.clear()
        self.sample_key = None
        self.capture = None
        self.clear_image()
        self.output_size = (0, 0)
        try:
            if not self.folder.is_dir():
                raise ValueError(f'Папка не найдена: {self.folder}')
            for capture in discover(self.folder):
                if capture.raw:
                    key = (capture.path.parent, capture.sample)
                    self.samples.setdefault(key, {}).setdefault(capture.mode, []).append(capture)
            self.sample_keys = list(self.samples)
            modes = sorted({mode for sample in self.samples.values() for mode in sample})
            palette = [(255, 80, 110), (70, 220, 255), (100, 255, 110),
                       (255, 190, 50), (185, 110, 255), (255, 255, 255), (80, 120, 255)]
            for i, mode in enumerate(modes):
                self.colors.setdefault(mode, palette[i % len(palette)])
                item = QTreeWidgetItem(self.channels, [mode, '●'])
                item.setData(0, Qt.ItemDataRole.UserRole, mode)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Unchecked)
                item.setForeground(1, QColor(*self.colors[mode]))
                self.channel_items[mode] = item
            for key in self.sample_keys:
                parent, sample = key
                item = QTreeWidgetItem(self.tree, [f'{sample} | {parent.relative_to(self.folder)}'])
                item.setData(0, Qt.ItemDataRole.UserRole, key)
            if self.sample_keys:
                target = previous if previous in self.samples else next(
                    (key for key in self.sample_keys if '261009_Mono16' in str(key[0])), self.sample_keys[0])
                if not self.initialized:
                    self.selected_modes = {next(iter(self.samples[target]))}
                    self.initialized = True
                if self.active_mode not in modes:
                    self.active_mode = next((mode for mode in modes if mode in self.selected_modes), modes[0])
                self.tree.setCurrentItem(self.tree.topLevelItem(self.sample_keys.index(target)))
                self.sample_key = target
            else:
                self.active_mode = None
        except (OSError, ValueError) as error:
            self.notice.setText(str(error))
        finally:
            self.updating = False
        if self.sample_key is not None:
            self.refresh_sample()
        else:
            self.sample_keys = []
            self.capture_choice.clear()
            self.notice.setText('RAW16 не найдены. Обычные TIFF скрыты.')

    def select(self, current, previous=None):
        if self.updating or current is None:
            return
        key = current.data(0, Qt.ItemDataRole.UserRole)
        if key not in self.samples:
            return
        self.sample_key = key
        self.cache.clear()
        self.refresh_sample()

    def move_sample(self, step):
        if self.sample_key not in self.sample_keys:
            return
        index = self.sample_keys.index(self.sample_key)
        index = max(0, min(len(self.sample_keys) - 1, index + step))
        self.tree.setCurrentItem(self.tree.topLevelItem(index))

    def selected_capture(self, mode):
        captures = self.samples.get(self.sample_key, {}).get(mode, [])
        if not captures:
            return None
        index = min(self.choices.get((self.sample_key, mode), 0), len(captures) - 1)
        return captures[index]

    def frame_for(self, mode):
        from normalization import detail_settings

        capture = self.selected_capture(mode)
        if capture is None:
            return None
        page = self.page_indices.get(capture.path, 0)
        key = (capture.path, page)
        if key not in self.cache:
            frame = read_frame(capture.path, page)
            if frame.pixels.ndim != 2 or frame.pixels.dtype != np.dtype('uint16'):
                raise ValueError(f'{capture.path.name}: ожидается монохромный uint16 RAW')
            self.cache[key] = frame
        frame = self.cache[key]
        if mode not in self.styles or (mode not in self.fixed_modes and self.style_frames.get(mode) != key):
            stat = capture.path.stat()
            stats_key = (key, stat.st_mtime_ns, stat.st_size)
            if stats_key not in self.auto_settings:
                self.auto_settings[stats_key] = detail_settings(frame.pixels, frame.white_is_zero)
            low, high, gamma = self.auto_settings[stats_key]
            self.styles[mode] = ChannelStyle(low, high, gamma, frame.white_is_zero, self.colors[mode])
            self.style_frames[mode] = key
        return frame

    def refresh_sample(self):
        self.updating = True
        available = self.samples.get(self.sample_key, {})
        for mode, item in self.channel_items.items():
            item.setText(0, mode if mode in available else f'{mode} — отсутствует')
            item.setCheckState(0, Qt.CheckState.Checked if mode in self.selected_modes else Qt.CheckState.Unchecked)
        if self.active_mode in self.channel_items:
            self.channels.setCurrentItem(self.channel_items[self.active_mode])
        self.updating = False
        self.update_active()
        self.render_selection()

    def toggle_channel(self, item, column):
        if self.updating or column != 0:
            return
        mode = item.data(0, Qt.ItemDataRole.UserRole)
        if item.checkState(0) == Qt.CheckState.Checked:
            self.selected_modes.add(mode)
        else:
            self.selected_modes.discard(mode)
        self.render_selection()

    def activate_channel(self, current, previous=None):
        if self.updating or current is None:
            return
        self.active_mode = current.data(0, Qt.ItemDataRole.UserRole)
        self.update_active()

    def update_active(self):
        self.sample_selector.clear()
        for parent, sample in self.sample_keys:
            self.sample_selector.addItem(f'{sample} | {parent.name}')
        if self.sample_key in self.sample_keys:
            self.sample_selector.setCurrentIndex(self.sample_keys.index(self.sample_key))
        self.channel_selector.clear()
        for mode in self.channel_items:
            missing = mode not in self.samples.get(self.sample_key, {})
            self.channel_selector.addItem(mode + (' — отсутствует' if missing else ''), mode)
        self.channel_selector.setCurrentIndex(self.channel_selector.findData(self.active_mode))
        self.auto_checkbox.setChecked(self.active_mode not in self.fixed_modes)
        self.frame = None
        self.capture = self.selected_capture(self.active_mode)
        self.capture_choice.clear()
        self.pages.clear()
        self.metadata.clear()
        for capture in self.samples.get(self.sample_key, {}).get(self.active_mode, []):
            self.capture_choice.addItem(capture.path.name)
        if self.capture is not None:
            self.capture_choice.setCurrentIndex(self.choices.get((self.sample_key, self.active_mode), 0))
            try:
                self.frame = self.frame_for(self.active_mode)
                self.pages.addItems([str(i + 1) for i in range(self.frame.page_count)])
                self.pages.setCurrentIndex(self.page_indices.get(self.capture.path, 0))
                style = self.styles[self.active_mode]
                self.low.setText(f'{style.low:.8g}')
                self.high.setText(f'{style.high:.8g}')
                self.gamma.setText(f'{style.gamma:g}')
                self.invert.setChecked(style.invert)
                details = [str(self.capture.path), f'Образец: {self.capture.sample} ({self.capture.sample_source})',
                           'Авто: шкала отдельно для каждого снимка; яркости разных снимков не сопоставимы напрямую.',
                           'Для сравнения яркости отключите авто и задайте одинаковые границы и гамму.',
                           'SCN: коды фильтров без преобразования в длины волн; единица exposure_time не указана.']
                details.extend(f'{key}: {value}' for key, value in self.frame.tags.items())
                details.extend(f'{key}: {value}' for key, value in capture_metadata(self.capture).items())
                self.metadata.setPlainText('\n'.join(details))
            except Exception as error:
                self.metadata.setPlainText(str(error))
        else:
            self.metadata.setPlainText('Активный канал отсутствует у этого образца.')
        for widget in (self.low, self.high, self.gamma, self.invert, self.pages):
            widget.setEnabled(self.frame is not None)
        self.color_button.setEnabled(self.active_mode is not None)
        if self.active_mode in self.colors:
            color = QColor(*self.colors[self.active_mode])
            self.color_button.setText(f'Цвет {self.active_mode}: {color.name()}…')

    def change_capture(self, index):
        self.choices[(self.sample_key, self.active_mode)] = index
        self.update_active()
        self.render_selection()

    def load(self, page_index=0):
        if self.capture is not None:
            self.page_indices[self.capture.path] = page_index
            self.cache.clear()
            self.update_active()
            self.render_selection()

    def cycle_channel(self, step):
        modes = list(self.samples.get(self.sample_key, {}))
        if not modes:
            return
        index = modes.index(self.active_mode) if self.active_mode in modes else -1
        self.active_mode = modes[(index + step) % len(modes)]
        self.only_active()

    def only_active(self):
        if self.active_mode is not None:
            self.selected_modes = {self.active_mode}
            self.refresh_sample()

    def pick_color(self, item, column):
        if column != 1:
            return
        mode = item.data(0, Qt.ItemDataRole.UserRole)
        color = QColorDialog.getColor(QColor(*self.colors[mode]), self, f'Цвет канала {mode}')
        if color.isValid():
            self.set_channel_color(mode, (color.red(), color.green(), color.blue()))

    def pick_active_color(self):
        if self.active_mode in self.channel_items:
            self.pick_color(self.channel_items[self.active_mode], 1)

    def set_channel_color(self, mode, color):
        self.colors[mode] = color
        if mode in self.styles:
            self.styles[mode].color = color
        self.channel_items[mode].setForeground(1, QColor(*color))
        self.update_active()
        self.render_selection()

    def set_limits(self, auto):
        if self.frame is None:
            return
        if auto:
            self.set_auto_mode(True)
            return
        try:
            low, high = display_limits(self.frame.pixels, False)
            self.low.setText(f'{low:.8g}')
            self.high.setText(f'{high:.8g}')
            self.gamma.setText('1')
            self.draw()
        except ValueError as error:
            self.statusBar().showMessage(str(error))

    def draw(self):
        if self.frame is None:
            return
        try:
            low, high, gamma = (float(entry.text().replace(',', '.')) for entry in (self.low, self.high, self.gamma))
            if not all(np.isfinite(v) for v in (low, high, gamma)) or high <= low or gamma <= 0:
                raise ValueError('Максимум должен быть больше минимума; гамма > 0')
            self.styles[self.active_mode] = ChannelStyle(low, high, gamma, self.invert.isChecked(),
                                                         self.colors[self.active_mode])
        except ValueError as error:
            self.statusBar().showMessage(f'Не удалось обновить отображение: {error}')
            return
        self.fixed_modes.add(self.active_mode)
        self.auto_checkbox.setChecked(False)
        self.render_selection()

    def render_selection(self):
        self.layers = {}
        problems = []
        for mode in sorted(self.selected_modes):
            try:
                frame = self.frame_for(mode)
                if frame is None:
                    problems.append(f'{mode}: отсутствует')
                else:
                    self.layers[mode] = frame
            except Exception as error:
                problems.append(f'{mode}: {error}')
        self.heading.setText(self.sample_key[1] if self.sample_key else 'WaferPL')
        try:
            image = compose_overlay([(frame, self.styles[mode]) for mode, frame in self.layers.items()])
            if image is None:
                self.view.clear_image()
                self.output_size = (0, 0)
            else:
                self.view.set_pixels(image)
                self.output_size = (image.shape[1], image.shape[0])
        except (ValueError, MemoryError) as error:
            self.view.clear_image()
            self.output_size = (0, 0)
            self.layers.clear()
            problems.append(str(error))
        notes = [' + '.join(sorted(self.layers)) or 'Каналы не отображаются']
        if any(mode not in self.fixed_modes for mode in self.layers):
            notes.append('Автошкала по снимку: для сравнения деталей, не абсолютной яркости.')
        if len(self.layers) > 1:
            notes.append('Оверлей по границам кадра, без регистрации; вклады цветов нормированы без обрезки суммы.')
        notes.extend(problems)
        self.notice.setText('\n'.join(notes))
        self.statusBar().showMessage('Наведите курсор: значения исходного RAW каждого показанного канала.')

    def inspect_pixel(self, scene_x, scene_y):
        # Pick the displayed cell, then the exact raw cell used in nearest-neighbour expansion.
        cell = image_coordinates(scene_x, scene_y, self.output_size, self.output_size)
        if cell is None:
            self.statusBar().showMessage('За пределами изображения')
            return
        readings = []
        for mode, frame in self.layers.items():
            size = (frame.pixels.shape[1], frame.pixels.shape[0])
            x, y = image_coordinates(cell[0], cell[1], self.output_size, size)
            readings.append(f'{mode}: x={x}, y={y}, RAW16={pixel_value(frame.pixels, x, y)}')
        self.statusBar().showMessage(' | '.join(readings))

    def choose_single_channel(self, index):
        mode = self.channel_selector.itemData(index)
        if mode is not None:
            self.active_mode = mode
            self.only_active()

    def set_auto_mode(self, enabled):
        if self.active_mode is None:
            return
        if enabled:
            self.fixed_modes.discard(self.active_mode)
            self.style_frames.pop(self.active_mode, None)
        else:
            self.fixed_modes.add(self.active_mode)
        self.update_active()
        self.render_selection()
