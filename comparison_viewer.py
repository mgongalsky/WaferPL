"""Shared sample tree and controls for equal-sized comparison views."""
from dataclasses import replace

from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QSizePolicy, QSplitter, QStackedWidget, QToolButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)
from pathlib import Path
from channel_viewer import ChannelViewer


class HostedPanel(ChannelViewer):
    state_changed = pyqtSignal()

    def scan(self):
        super().scan()
        self.state_changed.emit()

    def update_active(self):
        super().update_active()
        self.state_changed.emit()

    def render_selection(self):
        super().render_selection()
        self.state_changed.emit()

    def choose_folder(self):
        self.host.open_folder()

    def frame_for(self, mode):
        host = getattr(self, 'host', None)
        locked = host is not None and host.scale_lock.isChecked()
        if locked and mode not in host.shared_styles:
            style = host.fit_shared_style(mode)
            if style is not None:
                host.shared_styles[mode] = style
        if locked and mode in host.shared_styles:
            self.styles[mode] = replace(host.shared_styles[mode])
            self.colors[mode] = self.styles[mode].color
            self.fixed_modes.add(mode)
        return super().frame_for(mode)

    def draw(self):
        if not self.host.scale_lock.isChecked():
            return super().draw()
        if self.frame is None:
            return
        import math
        from channel_viewer import ChannelStyle
        try:
            low, high, gamma = (float(entry.text().replace(',', '.')) for entry in (self.low, self.high, self.gamma))
            if not all(math.isfinite(value) for value in (low, high, gamma)) or high <= low or gamma <= 0:
                raise ValueError('Максимум должен быть больше минимума; гамма > 0')
        except ValueError as error:
            self.statusBar().showMessage(f'Не удалось обновить отображение: {error}')
            return
        style = ChannelStyle(low, high, gamma, self.invert.isChecked(), self.colors[self.active_mode])
        self.host.publish_scale(self.active_mode, style)

    def set_auto_mode(self, enabled):
        if not self.host.scale_lock.isChecked():
            return super().set_auto_mode(enabled)
        if enabled and self.active_mode is not None:
            style = self.host.fit_shared_style(self.active_mode)
            if style is not None:
                self.host.publish_scale(self.active_mode, style)
        self.auto_checkbox.setChecked(False)

    def set_channel_color(self, mode, color):
        if not self.host.scale_lock.isChecked():
            return super().set_channel_color(mode, color)
        self.frame_for(mode)
        style = self.host.shared_styles.get(mode)
        if style is None:
            from channel_viewer import ChannelStyle
            style = ChannelStyle(0, 65535, 1, False, color)
        self.host.publish_scale(mode, replace(style, color=color))

    def only_active(self):
        if not self.host.scale_lock.isChecked():
            return super().only_active()
        if self.active_mode is not None:
            self.selected_modes = {self.active_mode}
            self.host.sync_channels(self)

    def toggle_channel(self, item, column):
        if self.updating or column != 0:
            return
        if not self.host.scale_lock.isChecked():
            return super().toggle_channel(item, column)
        mode = item.data(0, Qt.ItemDataRole.UserRole)
        if item.checkState(0) == Qt.CheckState.Checked:
            self.selected_modes.add(mode)
        else:
            self.selected_modes.discard(mode)
        self.host.sync_channels(self)

    def activate_channel(self, current, previous=None):
        if self.updating or current is None:
            return
        if not self.host.scale_lock.isChecked():
            return super().activate_channel(current, previous)
        self.active_mode = current.data(0, Qt.ItemDataRole.UserRole)
        self.host.sync_channels(self)

    def cycle_channel(self, step):
        if not self.host.scale_lock.isChecked():
            return super().cycle_channel(step)
        # Use one stable order on both sides, including channels missing in one sample.
        modes = sorted({mode for panel in (self.host.left, self.host.right)
                        if panel is not None for mode in panel.channel_items})
        if not modes:
            return
        if self.active_mode in modes:
            index = (modes.index(self.active_mode) + step) % len(modes)
        else:
            index = 0 if step >= 0 else len(modes) - 1
        self.active_mode = modes[index]
        self.only_active()

    def inspect_pixel(self, scene_x, scene_y):
        super().inspect_pixel(scene_x, scene_y)
        # Publish every hover, including repeated values on an inactive panel.
        sample = self.sample_key[1] if self.sample_key else 'Нет образца'
        self.host.statusBar().showMessage(
            f'{self.accessibleName()} · {sample} | {self.statusBar().currentMessage()}')


class ComparisonViewer(QMainWindow):
    def __init__(self, folder):
        super().__init__()
        self.setWindowTitle('WaferPL — сравнение образцов')
        self.resize(1500, 950)
        self.folder = Path(folder)
        self.right = None
        self.active = None
        self.syncing = False
        self.tree_signature = None
        self.tree_items = {}
        self.fit_timer = QTimer(self)
        self.fit_timer.setSingleShot(True)
        self.fit_timer.timeout.connect(self.equalize_views)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.navigation = QHBoxLayout()
        self.navigation.setSpacing(3)
        layout.addLayout(self.navigation)
        self.compare = QCheckBox('Две панели')
        self.compare.toggled.connect(self.set_comparison)
        self.navigation.addWidget(self.compare)
        self.navigation.addSpacing(10)
        self.navigation.addWidget(QLabel('Образец'))
        self.sample_previous = self.nav_button('←', 'Предыдущий образец — Shift+←', lambda: self.active.move_sample(-1))
        self.sample_next = self.nav_button('→', 'Следующий образец — Shift+→', lambda: self.active.move_sample(1))
        self.sample_selector = self.selector()
        self.sample_selector.activated.connect(self.choose_sample)
        self.navigation.addWidget(self.sample_selector, 2)
        self.navigation.addWidget(QLabel('Канал'))
        self.channel_previous = self.nav_button('←', 'Предыдущий канал — Alt+←', lambda: self.active.cycle_channel(-1))
        self.channel_next = self.nav_button('→', 'Следующий канал — Alt+→', lambda: self.active.cycle_channel(1))
        self.channel_selector = self.selector()
        self.channel_selector.activated.connect(self.choose_channel)
        self.navigation.addWidget(self.channel_selector, 1)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        layout.addWidget(self.splitter, 1)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel('Образцы — активное изображение')
        self.tree.currentItemChanged.connect(self.tree_selected)
        self.tree_dock = QDockWidget('Образцы', self)
        self.tree_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.tree_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.tree_dock.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.tree_dock)
        self.controls = QStackedWidget()
        self.controls_dock = QDockWidget('Управление', self)
        self.controls_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.controls_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.controls_dock.setMinimumWidth(210)
        self.controls_dock.setMaximumWidth(320)
        self.controls_dock.setWidget(self.controls)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.controls_dock)
        for dock, text in ((self.tree_dock, 'Дерево'), (self.controls_dock, 'Настройки')):
            button = QToolButton()
            button.setDefaultAction(dock.toggleViewAction())
            button.setText(text)
            self.navigation.addWidget(button)
        self.left = self.make_panel('Левая панель', self.folder)
        self.splitter.addWidget(self.left)
        self.active = self.left
        self.controls.setCurrentWidget(self.left.sidebar_tabs)
        self.resizeDocks([self.tree_dock, self.controls_dock], [220, 250], Qt.Orientation.Horizontal)
        self.shortcuts = []
        for key, delta in [('Shift+Left', -1), ('Shift+Up', -1), ('Shift+Right', 1), ('Shift+Down', 1)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda step=delta: self.active.move_sample(step))
            self.shortcuts.append(shortcut)
        for key, delta in [('Alt+Left', -1), ('Alt+Right', 1)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda step=delta: self.active.cycle_channel(step))
            self.shortcuts.append(shortcut)
        self.sync_ui()

    def nav_button(self, text, tooltip, callback):
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        self.navigation.addWidget(button)
        return button

    @staticmethod
    def selector():
        combo = QComboBox()
        combo.setMinimumWidth(0)
        combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        return combo

    def make_panel(self, title, folder):
        from PyQt6.QtWidgets import QScrollArea
        from scale_legend import ScaleLegend

        if not hasattr(self, 'scale_lock'):
            self.shared_styles = {}
            self.scale_lock = QToolButton()
            self.scale_lock.setText('🔓 Шкалы')
            self.scale_lock.setCheckable(True)
            self.scale_lock.setToolTip('Общая шкала из активной панели: границы, гамма, инверсия и цвет каждого канала')
            self.scale_lock.toggled.connect(self.set_scale_lock)
            self.navigation.insertWidget(1, self.scale_lock)
        panel = HostedPanel(folder)
        panel.host = self
        panel.setWindowFlags(Qt.WindowType.Widget)
        panel.setMinimumSize(0, 0)
        panel.setAccessibleName(title)
        navigation = panel.centralWidget().layout().itemAt(0).layout()
        for index in range(navigation.count()):
            widget = navigation.itemAt(index).widget()
            if widget is not None:
                widget.hide()
        panel.caption = QLabel(title)
        panel.caption.setFixedHeight(24)
        panel.caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout = panel.centralWidget().layout()
        layout.insertWidget(0, panel.caption)
        layout.removeWidget(panel.view)
        image_row = QHBoxLayout()
        image_row.setContentsMargins(0, 0, 0, 0)
        image_row.setSpacing(2)
        image_row.addWidget(panel.view, 1)
        panel.scale_legend = ScaleLegend()
        legend_scroll = QScrollArea()
        legend_scroll.setWidget(panel.scale_legend)
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setFixedWidth(146)
        legend_scroll.setMinimumHeight(0)
        legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        image_row.addWidget(legend_scroll)
        layout.addLayout(image_row, 1)
        panel.removeDockWidget(panel.sidebar)
        panel.sidebar_tabs.setParent(self.controls)
        self.controls.addWidget(panel.sidebar_tabs)
        panel.sidebar.hide()
        panel.statusBar().hide()
        panel.statusBar().messageChanged.connect(lambda text, p=panel: self.show_panel_status(p, text))
        for shortcut in panel.shortcuts:
            shortcut.setEnabled(False)
        for widget in (panel.view, panel.view.viewport(), panel.caption):
            widget.setProperty('comparison_panel', title)
            widget.installEventFilter(self)
        panel.state_changed.connect(self.sync_ui)
        panel.state_changed.connect(self.update_scale_legends)
        return panel

    def eventFilter(self, watched, event):
        title = watched.property('comparison_panel')
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.FocusIn):
            for panel in (getattr(self, 'left', None), self.right):
                if panel is not None and panel.accessibleName() == title:
                    self.activate_panel(panel)
                    break
        if event.type() == QEvent.Type.Resize and title:
            self.fit_timer.start(0)
        if event.type() == QEvent.Type.Leave and title:
            self.statusBar().showMessage('Наведите курсор на любое изображение для чтения RAW16')
        return super().eventFilter(watched, event)

    def activate_panel(self, panel):
        if panel is self.right and not self.compare.isChecked():
            return
        if panel is self.active:
            return
        tab = self.controls.currentWidget().currentIndex() if self.controls.currentWidget() is not None else 0
        self.active = panel
        panel.sidebar_tabs.setCurrentIndex(tab)
        self.controls.setCurrentWidget(panel.sidebar_tabs)
        self.sync_ui()
        self.show_panel_status(panel, panel.statusBar().currentMessage())

    def show_panel_status(self, panel, text):
        if panel is self.active:
            self.statusBar().showMessage(text)

    def sync_ui(self):
        if self.syncing or self.active is None:
            return
        self.syncing = True
        try:
            panel = self.active
            signature = tuple(panel.sample_keys)
            if signature != self.tree_signature:
                self.tree.blockSignals(True)
                self.tree.clear()
                self.tree_items.clear()
                groups = {}
                for key in signature:
                    parent, sample = key
                    if parent not in groups:
                        groups[parent] = QTreeWidgetItem(self.tree, [parent.name])
                        groups[parent].setExpanded(True)
                    item = QTreeWidgetItem(groups[parent], [sample])
                    item.setData(0, Qt.ItemDataRole.UserRole, key)
                    self.tree_items[key] = item
                self.tree_signature = signature
                self.tree.blockSignals(False)
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(self.tree_items.get(panel.sample_key))
            if panel.sample_key in self.tree_items:
                self.tree.scrollToItem(self.tree_items[panel.sample_key])
            self.tree.blockSignals(False)
            self.sample_selector.clear()
            for key in panel.sample_keys:
                self.sample_selector.addItem(f'{key[1]} | {key[0].name}', key)
            self.sample_selector.setCurrentIndex(
                panel.sample_keys.index(panel.sample_key) if panel.sample_key in panel.sample_keys else -1)
            self.channel_selector.clear()
            for mode in panel.channel_items:
                missing = mode not in panel.samples.get(panel.sample_key, {})
                self.channel_selector.addItem(mode + (' — отсутствует' if missing else ''), mode)
            self.channel_selector.setCurrentIndex(self.channel_selector.findData(panel.active_mode))
            self.controls_dock.setWindowTitle(f'Управление — {panel.accessibleName()}')
            for candidate in (getattr(self, 'left', None), self.right):
                if candidate is None:
                    continue
                sample = candidate.sample_key[1] if candidate.sample_key else 'Нет образца'
                channels = ' + '.join(sorted(candidate.selected_modes)) or 'Нет каналов'
                text = f'{candidate.accessibleName()}: {sample} · {channels}'
                candidate.caption.setText(text)
                candidate.caption.setToolTip(text)
                candidate.caption.setStyleSheet(
                    'background: #285c8a; color: white; padding-left: 6px;' if candidate is panel else 'padding-left: 6px;')
        finally:
            self.syncing = False
        self.fit_timer.start(0)

    def tree_selected(self, item, previous=None):
        if self.syncing or item is None:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key in self.active.samples:
            index = self.active.sample_keys.index(key)
            self.active.tree.setCurrentItem(self.active.tree.topLevelItem(index))

    def choose_sample(self, index):
        key = self.sample_selector.itemData(index)
        if key in self.active.samples:
            self.active.tree.setCurrentItem(self.active.tree.topLevelItem(self.active.sample_keys.index(key)))

    def choose_channel(self, index):
        mode = self.channel_selector.itemData(index)
        if mode is not None:
            self.active.active_mode = mode
            self.active.only_active()

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Открыть папку', str(self.folder))
        if folder:
            self.folder = Path(folder)
            for panel in (self.left, self.right):
                if panel is not None:
                    panel.folder = self.folder
                    panel.scan()
            self.sync_ui()

    def set_comparison(self, enabled):
        if enabled and self.right is None:
            self.right = self.make_panel('Правая панель', self.left.folder)
            self.splitter.addWidget(self.right)
            self.right.scan_timer.stop()
            self.right.scan()
            self.right.selected_modes = set(self.left.selected_modes)
            self.right.active_mode = self.left.active_mode
            self.right.colors = dict(self.left.colors)
            self.right.styles = {mode: replace(style) for mode, style in self.left.styles.items()}
            self.right.fixed_modes = set(self.left.fixed_modes)
            self.right.style_frames.clear()
            self.right.channels.blockSignals(True)
            for mode, item in self.right.channel_items.items():
                item.setForeground(1, QColor(*self.right.colors[mode]))
            self.right.channels.blockSignals(False)
            if self.left.sample_key in self.right.samples:
                index = self.right.sample_keys.index(self.left.sample_key)
                self.right.tree.setCurrentItem(self.right.tree.topLevelItem(index))
            self.right.refresh_sample()
        if self.right is not None:
            if not enabled:
                self.activate_panel(self.left)
            self.right.setVisible(enabled)
        if enabled:
            for panel in (self.left, self.right):
                panel.zoom.setCurrentText('Вписать')
            self.splitter.handle(1).setEnabled(False)
        self.sync_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'fit_timer'):
            self.fit_timer.start(0)

    def equalize_views(self):
        panels = [self.left]
        if self.compare.isChecked() and self.right is not None:
            panels.append(self.right)
            sizes = self.splitter.sizes()
            if len(sizes) == 2 and abs(sizes[0] - sizes[1]) > 1:
                half = max(1, sum(sizes) // 2)
                self.splitter.setSizes([half, half])
        common = (min(p.view.viewport().width() for p in panels),
                  min(p.view.viewport().height() for p in panels))
        for panel in panels:
            panel.view.fit_viewport_size = common if len(panels) == 2 else None
            if panel.view.zoom_mode == 'Вписать':
                panel.view.apply_zoom()

    def set_scale_lock(self, enabled):
        self.scale_lock.setText('🔒 Шкалы' if enabled else '🔓 Шкалы')
        if enabled:
            if not self.compare.isChecked():
                self.compare.setChecked(True)
            source = self.active
            modes = {mode for panel in (self.left, self.right)
                     if panel is not None for mode in panel.channel_items}
            styles = {}
            for mode in sorted(modes):
                style = self.fit_shared_style(mode)
                if style is not None:
                    styles[mode] = style
            self.shared_styles = styles
            self.apply_shared_scales()
            self.sync_channels(source)
        else:
            # Preserve the last shared mapping until explicit auto or manual editing.
            self.shared_styles.clear()
            for panel in (self.left, self.right):
                if panel is not None:
                    panel.update_active()
        self.update_scale_legends()

    def apply_shared_scales(self):
        for panel in (self.left, self.right):
            if panel is None:
                continue
            for mode, style in self.shared_styles.items():
                panel.styles[mode] = replace(style)
                panel.colors[mode] = style.color
                panel.fixed_modes.add(mode)
            panel.channels.blockSignals(True)
            try:
                for mode, item in panel.channel_items.items():
                    item.setForeground(1, QColor(*panel.colors[mode]))
            finally:
                panel.channels.blockSignals(False)
            panel.update_active()
            panel.render_selection()
        self.update_scale_legends()

    def publish_scale(self, mode, style):
        self.shared_styles[mode] = replace(style)
        self.apply_shared_scales()

    def update_scale_legends(self):
        self.scale_lock.setToolTip(
            'При включении шкала учитывает оба снимка. Шкалы и каналы связаны; образцы выбираются независимо.')
        for panel in (getattr(self, 'left', None), self.right):
            if panel is not None:
                panel.scale_legend.set_scales(panel, self.scale_lock.isChecked())
                if self.scale_lock.isChecked():
                    panel.auto_checkbox.setChecked(False)
                    panel.auto_checkbox.setToolTip(
                        'Авто один раз учитывает оба снимка и фиксирует общую шкалу. Гамма — по снимку с большей верхней границей.')
                else:
                    panel.auto_checkbox.setToolTip(
                        'Автоматическая шкала отдельно для каждого снимка.')
        if self.scale_lock.isChecked() and self.right is not None:
            if set(self.left.layers) != set(self.right.layers):
                self.statusBar().showMessage(
                    'Не все выбранные каналы доступны в обеих панелях. Отсутствующие каналы отмечены в списке и на шкалах.')

    def sync_channels(self, source):
        if not self.scale_lock.isChecked() or getattr(self, 'syncing_channels', False):
            return
        self.syncing_channels = True
        try:
            selected = set(source.selected_modes)
            active_mode = source.active_mode
            for panel in (self.left, self.right):
                if panel is not None:
                    panel.selected_modes = set(selected)
                    panel.active_mode = active_mode
                    panel.refresh_sample()
        finally:
            self.syncing_channels = False
        self.sync_ui()
        self.update_scale_legends()

    def fit_shared_style(self, mode):
        from normalization import detail_settings
        from channel_viewer import ChannelStyle

        candidates = []
        for panel in (getattr(self, 'left', None), self.right):
            if panel is None:
                continue
            try:
                # Bypass the shared-style override while collecting source statistics.
                frame = ChannelViewer.frame_for(panel, mode)
                if frame is not None:
                    limits = detail_settings(frame.pixels, frame.white_is_zero)
                    candidates.append((panel, frame, limits))
            except (OSError, ValueError, IndexError):
                continue
        if not candidates:
            return None
        # Every image contributes its own robust limits regardless of its resolution.
        low = min(entry[2][0] for entry in candidates)
        high = max(entry[2][1] for entry in candidates)
        reference = max(candidates, key=lambda entry: (entry[2][1], -entry[2][0]))
        gamma = reference[2][2]
        source = next((entry for entry in candidates if entry[0] is self.active), reference)
        color = source[0].colors[mode]
        return ChannelStyle(low, high, gamma, source[1].white_is_zero, color)
