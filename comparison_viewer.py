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
        panel = HostedPanel(folder)
        panel.host = self
        panel.setWindowFlags(Qt.WindowType.Widget)
        panel.setMinimumSize(0, 0)
        panel.setAccessibleName(title)
        # Only image and a one-line caption remain in each equal-sized pane.
        navigation = panel.centralWidget().layout().itemAt(0).layout()
        for index in range(navigation.count()):
            widget = navigation.itemAt(index).widget()
            if widget is not None:
                widget.hide()
        panel.caption = QLabel(title)
        panel.caption.setFixedHeight(24)
        panel.caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        panel.centralWidget().layout().insertWidget(0, panel.caption)
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
        return panel

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.FocusIn):
            title = watched.property('comparison_panel')
            for panel in (getattr(self, 'left', None), self.right):
                if panel is not None and panel.accessibleName() == title:
                    self.activate_panel(panel)
                    break
        if event.type() == QEvent.Type.Resize and watched.property('comparison_panel'):
            self.fit_timer.start(0)
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
            self.sample_selector.setCurrentIndex(panel.sample_keys.index(panel.sample_key) if panel.sample_key in panel.sample_keys else -1)
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
                candidate.caption.setStyleSheet('background: #285c8a; color: white; padding-left: 6px;' if candidate is panel else 'padding-left: 6px;')
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
