"""Large alignment workspace with display colors and reusable parameters."""
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QBoxLayout, QColorDialog, QComboBox, QFileDialog,
                             QHBoxLayout, QLabel, QPushButton, QScrollArea,
                             QSizePolicy, QSplitter, QVBoxLayout, QWidget)
from registration_viewer import RegistrationDialog
from registration_preview import compose_layers
from registration_session import make_session, validate_session, save_session, load_session


class RegistrationWorkspace(RegistrationDialog):
    def __init__(self, left_frame, right_frame, titles, parent=None, sources=None):
        # Base construction schedules its first preview after this setup completes.
        self.colors = [(255, 0, 0), (0, 255, 255)]
        self.sources = sources
        self.session_path = ''
        super().__init__(left_frame, right_frame, titles, parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        self.resize(1550, 1000)
        root = self.layout()
        items = [root.takeAt(0) for _ in range(root.count())]
        help_item, editors_item, actions_item, display_item, status_item, legend_item, output_item = items
        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        root.addWidget(self.workspace_splitter)

        sidebar = QWidget()
        side = QVBoxLayout(sidebar)
        side.addWidget(help_item.widget())
        self.editors.layout().setDirection(QBoxLayout.Direction.TopToBottom)
        side.addWidget(editors_item.widget(), 1)
        # Stack action buttons so the sidebar can stay narrow.
        actions = actions_item.layout()
        actions.setDirection(QBoxLayout.Direction.TopToBottom)
        side.addLayout(actions)
        files = QHBoxLayout()
        self.save_button = QPushButton('Сохранить параметры…')
        self.load_button = QPushButton('Загрузить параметры…')
        self.save_button.clicked.connect(self.choose_save)
        self.load_button.clicked.connect(self.choose_load)
        for button in (self.save_button, self.load_button):
            button.setEnabled(sources is not None)
            if sources is None:
                button.setToolTip('Откройте совмещение из панели сравнения для привязки к TIFF-файлам')
            files.addWidget(button)
        side.addLayout(files)
        legend_item.widget().setText('Цвета и нормирование используются только для просмотра. '
                                     'Параметры сохраняются в координатах исходных TIFF; RAW не меняется. '
                                     'Средняя кнопка мыши перемещает увеличенное изображение.')
        side.addWidget(legend_item.widget())
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setMinimumWidth(360)
        self.sidebar_scroll.setWidget(sidebar)
        self.workspace_splitter.addWidget(self.sidebar_scroll)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        self.display_mode.setItemText(1, 'Два цвета')
        result_layout.addLayout(display_item.layout())
        color_row = QHBoxLayout()
        self.color_buttons = []
        for index in range(2):
            button = QPushButton()
            button.clicked.connect(lambda checked=False, side=index: self.choose_color(side))
            color_row.addWidget(button)
            self.color_buttons.append(button)
        color_row.addStretch()
        color_row.addWidget(QLabel('Масштаб'))
        self.output_zoom = QComboBox()
        self.output_zoom.addItems(['Вписать', '50%', '100%', '200%'])
        self.output_zoom.setToolTip('Масштаб изображения предпросмотра, не исходного RAW')
        self.output_zoom.currentTextChanged.connect(self.output.set_zoom)
        color_row.addWidget(self.output_zoom)
        result_layout.addLayout(color_row)
        self.output.setMinimumSize(160, 160)
        self.output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        result_layout.addWidget(output_item.widget(), 1)
        result_layout.addWidget(status_item.widget())
        self.workspace_splitter.addWidget(result_panel)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([480, 1050])
        self.update_color_buttons()

    def update_color_buttons(self):
        if not getattr(self, '_color_controls_initialized', False):
            self._color_controls_initialized = True
            self.display_mode.setItemText(self.display_mode.findData('blend'), 'Серое смешивание')
            self.display_mode.setItemText(self.display_mode.findData('color'), 'Цветное наложение')
            self.display_mode.setCurrentIndex(self.display_mode.findData('color'))
            self.display_mode.setToolTip(
                'Красный и голубой показывают разные снимки; совпадающие яркости дают серый или белый. Серое смешивание отключает окрашивание.')
            self.align.setText('Совместить по характерным точкам')
            self.align.setToolTip(
                'SIFT + RANSAC: поворот ±15° и поправка центра до 12% радиуса. Размер пластины фиксирован. Неудачный поиск сохраняет ручное положение.')
            help_label = self.sidebar_scroll.widget().layout().itemAt(0).widget()
            help_label.setText(
                'Обведите пластину кругом на каждом снимке. Центр — перемещение; край — общий радиус; голубая ручка — поворот к общей детали. Поиск по точкам уточняет поворот и положение центра, сохраняя размер. Изменившиеся детали могут не совпадать.')
            self.matches_button = QPushButton('Показать найденные пары…')
            self.matches_button.setEnabled(False)
            self.matches_button.clicked.connect(self.show_matches)
            self.sidebar_scroll.widget().layout().addWidget(self.matches_button)
        # Also called after loading: don't claim evidence for another transform.
        if hasattr(self, 'matches_button'):
            self.matches_button.setEnabled(
                getattr(self, 'feature_evidence', None) is not None
                and self.result == getattr(self, 'feature_evidence_transform', None))
        for index, button in enumerate(self.color_buttons):
            color = QColor(*self.colors[index])
            label = 'Левый' if index == 0 else 'Правый'
            button.setText(f'{label}: {color.name()}…')
            button.setToolTip('Выбрать цвет снимка и включить цветное наложение')
            foreground = '#000000' if color.lightness() > 140 else '#ffffff'
            button.setStyleSheet(f'background-color: {color.name()}; color: {foreground}; padding: 5px;')

    def choose_color(self, index):
        color = QColorDialog.getColor(QColor(*self.colors[index]), self,
                                      'Цвет левого снимка' if index == 0 else 'Цвет правого снимка')
        if color.isValid():
            self.colors[index] = (color.red(), color.green(), color.blue())
            self.update_color_buttons()
            self.display_mode.setCurrentIndex(self.display_mode.findData('color'))
            self.render_layers()

    def render_layers(self):
        mode = self.display_mode.currentData()
        self.opacity.setEnabled(True)
        self.opacity.setToolTip(
            'Баланс снимков: 0% — только левый, 50% — оба, 100% — только правый. Работает в сером и цветном режимах.')
        self.opacity_label.setText(f'{self.opacity.value()}%')
        if self.preview_layers is not None:
            self.output.set_pixels(compose_layers(self.preview_layers, mode, self.opacity.value() / 100, self.colors))

    def build_preview(self, limit):
        # Use the space available to the large result panel, not a 450px thumbnail.
        size = max(self.output.viewport().width(), self.output.viewport().height())
        return super().build_preview(max(limit, min(2400, max(900, size))))

    def set_busy(self, busy):
        super().set_busy(busy)
        self.save_button.setEnabled(not busy and self.sources is not None)
        self.load_button.setEnabled(not busy and self.sources is not None)

    def session_document(self):
        transform = self.result or self.initial_transform()
        return make_session(self.sources, transform,
                            (self.left.view.orientation, self.right.view.orientation),
                            self.display_mode.currentData(), self.opacity.value(), self.colors,
                            refined=self.result is not None)

    def save_parameters(self, path):
        if self.worker is not None:
            raise ValueError('Дождитесь завершения поиска поворота')
        # Never overwrite one of the input measurements through the parameter picker.
        destination = Path(path).resolve()
        if self.sources is not None and any(destination == Path(source['path']).resolve() for source in self.sources):
            raise ValueError('Нельзя сохранять параметры поверх исходного TIFF')
        document = self.session_document()
        save_session(destination, document)
        self.session_path = str(destination)
        self.status.setText(f'Параметры сохранены: {destination.name}')

    def load_parameters(self, path):
        if self.worker is not None:
            raise ValueError('Дождитесь завершения поиска поворота')
        document = load_session(path, self.sources)
        transform, orientations, display, refined = validate_session(document, self.sources)
        # All file validation precedes changes to the controls.
        self.preview_timer.stop()
        for editor, circle, angle in zip((self.left, self.right),
                                         (transform.left, transform.right), orientations):
            editor.view.drag = None
            editor.draw_button.setChecked(False)
            editor.set_circle(circle)
            editor.angle.blockSignals(True)
            editor.angle.setValue(angle)
            editor.angle.blockSignals(False)
            editor.view.orientation = angle
            editor.view.fit_circle()
            editor.view.viewport().update()
        self.result = transform if refined else None
        self.colors = [tuple(color) for color in display['colors']]
        self.display_mode.blockSignals(True)
        self.opacity.blockSignals(True)
        self.display_mode.setCurrentIndex(self.display_mode.findData(display['mode']))
        self.opacity.setValue(display['opacity'])
        self.display_mode.blockSignals(False)
        self.opacity.blockSignals(False)
        self.update_color_buttons()
        self.session_path = str(path)
        if self.build_preview(900):
            self.status.setText(f'Параметры загружены: {Path(path).name}. Исходные файлы и страницы совпадают.')

    def choose_save(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить параметры совмещения',
                                              self.session_path or 'alignment.json', 'Параметры совмещения (*.json)')
        if not path:
            return
        if not Path(path).suffix:
            path += '.json'
        try:
            self.save_parameters(path)
        except (OSError, ValueError, TypeError) as error:
            self.status.setText(f'Не удалось сохранить параметры: {error}')

    def choose_load(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Загрузить параметры совмещения',
                                              self.session_path, 'Параметры совмещения (*.json)')
        if not path:
            return
        try:
            self.load_parameters(path)
        except (OSError, ValueError, TypeError) as error:
            self.status.setText(f'Не удалось загрузить параметры: {error}')

    def start(self):
        from feature_registration_ui import FeatureWorker
        if self.worker is not None or self.closing:
            return
        self.preview_timer.stop()
        self.build_preview(900)
        self.set_busy(True)
        self.status.setText('Ищем характерные точки SIFT и проверяем пары RANSAC… '
                            'Поворот ±15°, поправка центра до 12% радиуса; размер фиксирован.')
        self.worker = FeatureWorker(self.images, self.initial_transform(), self)
        self.worker.completed.connect(self.show_feature_result)
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self.search_finished)
        self.worker.start()

    def show_feature_result(self, evidence):
        from registration import Circle, Registration
        if self.closing or self.worker is None or self.worker.isInterruptionRequested():
            return
        t = evidence.transform
        # Absorb the correction into the circle: controls, masks and JSON then
        # describe exactly the same mapping, with no invisible residual shift.
        right = Circle(t.right.x + t.dx, t.right.y + t.dy, t.right.radius)
        orientation = (self.left.view.orientation + t.angle + 180) % 360 - 180
        self.right.set_circle(right)
        self.right.angle.blockSignals(True)
        self.right.angle.setValue(orientation)
        self.right.angle.blockSignals(False)
        self.right.view.orientation = orientation
        self.right.view.viewport().update()
        self.result = Registration(t.left, right, t.angle, score=t.score, overlap=t.overlap)
        self.feature_evidence = evidence
        self.feature_evidence_transform = self.result
        self.matches_button.setEnabled(True)
        if self.build_preview(900):
            self.status.setText(
                f'Совмещение по точкам: угол {t.angle:.2f}°; поправка центра '
                f'Δx={t.dx:.2f}, Δy={t.dy:.2f} px. '
                f'Подтверждено {len(evidence.errors)} из {evidence.matches} пар; '
                f'ошибка RMS {evidence.rms:.2f} px правого TIFF. '
                'Проверьте найденные пары. Размер пластины не изменён.')

    def show_matches(self):
        from feature_registration_ui import show_correspondences
        evidence = getattr(self, 'feature_evidence', None)
        if evidence is None or self.result != getattr(self, 'feature_evidence_transform', None):
            self.status.setText('Для текущих параметров нет подтверждённых пар. Запустите поиск по точкам.')
            return
        show_correspondences(self, self.images, evidence)

    def invalidate(self):
        self.feature_evidence = None
        self.feature_evidence_transform = None
        if hasattr(self, 'matches_button'):
            self.matches_button.setEnabled(False)
        super().invalidate()
