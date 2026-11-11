"""Графический интерфейс приложения (Tkinter)."""

from __future__ import annotations

import base64
import os
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

import numpy as np

from . import camera, image_io, processing, validation

APP_TITLE = "Обработка изображений — вариант №7"
CANVAS_BACKGROUND = "#3c3f41"
CANVAS_PADDING = 10
CHANNEL_ALL = "all"
NO_IMAGE_TEXT = "Изображение не выбрано"
HINT_COLOR = "#666666"


class ImageApp(tk.Tk):
    """Главное окно приложения."""

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)

        # Исходное изображение (для сброса) и текущее (после операций).
        self._original: Optional[np.ndarray] = None
        self._current: Optional[np.ndarray] = None

        # Данные о том, как изображение размещено на холсте.
        self._photo: Optional[tk.PhotoImage] = None
        self._scale = 1.0
        self._origin: Tuple[int, int] = (0, 0)
        self._redraw_job: Optional[str] = None

        # Элементы управления, доступные только при выбранном изображении.
        self._image_widgets: List[ttk.Widget] = []

        self._channel = tk.StringVar(value=CHANNEL_ALL)
        self._camera_index = tk.StringVar(value="0")
        self._angle = tk.StringVar(value="45")
        self._circle_x = tk.StringVar()
        self._circle_y = tk.StringVar()
        self._circle_radius = tk.StringVar()
        self._circle_filled = tk.BooleanVar(value=False)
        self._status = tk.StringVar()
        self._size_info = tk.StringVar()
        self._cursor_info = tk.StringVar()

        self._build_ui()
        self._set_controls_enabled(False)

        # Минимальная высота окна подбирается так, чтобы панель управления
        # помещалась целиком при любых системных шрифтах и темах оформления.
        self.update_idletasks()
        min_height = max(600, self.winfo_reqheight())
        self.minsize(900, min_height)
        self.geometry("1100x%d" % max(700, min_height))
        self._set_status(
            "Готово к работе. Откройте изображение или сделайте снимок "
            "с веб-камеры."
        )

    # ------------------------------------------------------------------
    # Построение интерфейса
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        panel = ttk.Frame(self, padding=10)
        panel.grid(row=0, column=0, sticky="ns")

        self._canvas = tk.Canvas(
            self, background=CANVAS_BACKGROUND, highlightthickness=0
        )
        self._canvas.grid(row=0, column=1, sticky="nsew")
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<Motion>", self._on_mouse_move)
        self._canvas.bind("<Leave>", self._on_mouse_leave)
        self._canvas.bind("<Button-1>", self._on_canvas_click)

        ttk.Separator(self, orient="horizontal").grid(
            row=1, column=0, columnspan=2, sticky="ew"
        )
        status_bar = ttk.Frame(self, padding=(10, 4))
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        status_bar.columnconfigure(0, weight=1)
        ttk.Label(status_bar, textvariable=self._status, anchor="w").grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Label(status_bar, textvariable=self._cursor_info, anchor="e").grid(
            row=0, column=1, sticky="e", padx=(10, 0)
        )
        ttk.Label(status_bar, textvariable=self._size_info, anchor="e").grid(
            row=0, column=2, sticky="e", padx=(10, 0)
        )

        self._build_source_section(panel)
        self._build_channel_section(panel)
        self._build_negative_section(panel)
        self._build_rotation_section(panel)
        self._build_circle_section(panel)
        self._build_file_section(panel)

    def _section(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill="x", pady=(0, 8))
        return frame

    def _build_source_section(self, panel: ttk.Frame) -> None:
        section = self._section(panel, "Источник изображения")
        ttk.Button(
            section, text="Открыть файл (PNG, JPG)…", command=self._open_file
        ).pack(fill="x")

        row = ttk.Frame(section)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(
            row, text="Снимок с веб-камеры", command=self._capture_from_camera
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(row, text="№").pack(side="left", padx=(6, 2))
        ttk.Spinbox(
            row,
            from_=0,
            to=camera.MAX_CAMERA_INDEX,
            width=3,
            textvariable=self._camera_index,
        ).pack(side="left")

    def _build_channel_section(self, panel: ttk.Frame) -> None:
        section = self._section(panel, "Цветовой канал")
        options = (
            ("Все каналы", CHANNEL_ALL),
            ("Красный", "red"),
            ("Зелёный", "green"),
            ("Синий", "blue"),
        )
        for position, (title, value) in enumerate(options):
            button = ttk.Radiobutton(
                section,
                text=title,
                value=value,
                variable=self._channel,
                command=self._on_channel_change,
            )
            button.grid(
                row=position // 2,
                column=position % 2,
                sticky="w",
                padx=(0, 12),
            )
            self._image_widgets.append(button)

    def _build_negative_section(self, panel: ttk.Frame) -> None:
        section = self._section(panel, "Негатив")
        button = ttk.Button(
            section, text="Показать негатив", command=self._apply_negative
        )
        button.pack(fill="x")
        self._image_widgets.append(button)

    def _build_rotation_section(self, panel: ttk.Frame) -> None:
        section = self._section(panel, "Вращение")
        row = ttk.Frame(section)
        row.pack(fill="x")
        ttk.Label(row, text="Угол, градусы:").pack(side="left")
        self._angle_entry = ttk.Entry(row, width=8, textvariable=self._angle)
        self._angle_entry.pack(side="left", padx=6)
        self._angle_entry.bind(
            "<Return>", lambda _event: self._apply_rotation()
        )
        button = ttk.Button(
            row, text="Повернуть", command=self._apply_rotation
        )
        button.pack(side="left", fill="x", expand=True)
        ttk.Label(
            section,
            text="Положительный угол — против часовой стрелки, "
            "отрицательный — по часовой.",
            wraplength=250,
            foreground=HINT_COLOR,
        ).pack(anchor="w", pady=(4, 0))
        self._image_widgets.extend([self._angle_entry, button])

    def _build_circle_section(self, panel: ttk.Frame) -> None:
        section = self._section(panel, "Красный круг")
        grid = ttk.Frame(section)
        grid.pack(fill="x")

        def add_entry(
            title: str, variable: tk.StringVar, row: int, column: int
        ) -> None:
            ttk.Label(grid, text=title).grid(
                row=row, column=column, sticky="w", pady=2
            )
            entry = ttk.Entry(grid, width=7, textvariable=variable)
            entry.grid(
                row=row, column=column + 1, sticky="w", padx=(4, 10), pady=2
            )
            entry.bind("<Return>", lambda _event: self._apply_circle())
            self._image_widgets.append(entry)

        add_entry("X центра:", self._circle_x, 0, 0)
        add_entry("Y центра:", self._circle_y, 0, 2)
        add_entry("Радиус:", self._circle_radius, 1, 0)
        check = ttk.Checkbutton(
            grid, text="Закрасить круг", variable=self._circle_filled
        )
        check.grid(row=1, column=2, columnspan=2, sticky="w")

        button = ttk.Button(
            section, text="Нарисовать круг", command=self._apply_circle
        )
        button.pack(fill="x", pady=(4, 0))
        ttk.Label(
            section,
            text="Координаты и радиус — в пикселях. Щелчок по изображению "
            "подставляет координаты центра.",
            wraplength=250,
            foreground=HINT_COLOR,
        ).pack(anchor="w", pady=(4, 0))
        self._image_widgets.extend([check, button])

    def _build_file_section(self, panel: ttk.Frame) -> None:
        section = self._section(panel, "Результат")
        reset = ttk.Button(
            section, text="Сбросить изменения", command=self._reset
        )
        reset.pack(side="left", fill="x", expand=True)
        save = ttk.Button(section, text="Сохранить как…", command=self._save)
        save.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self._image_widgets.extend([reset, save])

    # ------------------------------------------------------------------
    # Служебные методы
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status.set(text)

    def _show_error(self, title: str, message: str) -> None:
        """Показать ошибку в строке состояния и в диалоговом окне."""
        self._set_status("Ошибка: " + message.splitlines()[0])
        messagebox.showerror(title, message, parent=self)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "!disabled" if enabled else "disabled"
        for widget in self._image_widgets:
            widget.state([state])

    def _require_image(self) -> bool:
        if self._current is not None:
            return True
        self._set_status("Сначала выберите изображение.")
        messagebox.showwarning(
            "Нет изображения",
            "Сначала откройте изображение из файла или сделайте снимок "
            "с веб-камеры.",
            parent=self,
        )
        return False

    def report_callback_exception(self, exc, val, tb):  # type: ignore
        """Не дать непредвиденной ошибке завершить приложение."""
        traceback.print_exception(exc, val, tb)
        self._show_error(
            "Непредвиденная ошибка",
            "%s: %s\n\nПриложение продолжит работу."
            % (getattr(exc, "__name__", exc), val),
        )

    def _set_image(self, image: np.ndarray, description: str) -> None:
        """Сделать image новым исходным изображением."""
        self._original = image
        self._current = image.copy()
        self._channel.set(CHANNEL_ALL)
        self._set_controls_enabled(True)

        height, width = image.shape[:2]
        self._circle_x.set(str(width // 2))
        self._circle_y.set(str(height // 2))
        self._circle_radius.set(str(max(1, min(width, height) // 4)))

        self._render()
        self._set_status(
            "Загружено изображение: %s (%d x %d пикселей)."
            % (description, width, height)
        )

    def _displayed_image(self) -> Optional[np.ndarray]:
        """Изображение, которое видит пользователь (с учётом канала)."""
        if self._current is None:
            return None
        channel = self._channel.get()
        if channel == CHANNEL_ALL:
            return self._current
        return processing.extract_channel(self._current, channel)

    # ------------------------------------------------------------------
    # Отрисовка
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Перерисовать холст: вписать изображение в доступную область."""
        self._redraw_job = None
        canvas = self._canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)

        image = self._displayed_image()
        if image is None:
            self._photo = None
            self._size_info.set("")
            canvas.create_text(
                width // 2, height // 2, text=NO_IMAGE_TEXT, fill="#bbbbbb"
            )
            return

        data, scale = image_io.encode_preview(
            image, width - 2 * CANVAS_PADDING, height - 2 * CANVAS_PADDING
        )
        # Ссылка на PhotoImage хранится в атрибуте, иначе сборщик мусора
        # удалит изображение и холст останется пустым.
        photo = tk.PhotoImage(data=base64.b64encode(data))
        self._photo = photo
        self._scale = scale
        self._origin = (
            (width - photo.width()) // 2,
            (height - photo.height()) // 2,
        )
        canvas.create_image(
            self._origin[0], self._origin[1], image=photo, anchor="nw"
        )

        image_height, image_width = image.shape[:2]
        self._size_info.set(
            "%d x %d пикс., масштаб %d%%"
            % (image_width, image_height, round(scale * 100))
        )

    def _on_canvas_resize(self, _event: object) -> None:
        # Перерисовка откладывается, чтобы не выполняться на каждое
        # промежуточное событие при изменении размера окна.
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(50, self._render)

    def _canvas_to_image(
        self, canvas_x: int, canvas_y: int
    ) -> Optional[Tuple[int, int]]:
        """Перевести координаты холста в координаты пикселя изображения."""
        if self._current is None or self._photo is None or self._scale <= 0:
            return None
        dx, dy = canvas_x - self._origin[0], canvas_y - self._origin[1]
        if dx < 0 or dy < 0:
            return None
        x, y = int(dx / self._scale), int(dy / self._scale)
        height, width = self._current.shape[:2]
        if x >= width or y >= height:
            return None
        return x, y

    def _on_mouse_move(self, event: "tk.Event[tk.Canvas]") -> None:
        position = self._canvas_to_image(event.x, event.y)
        self._cursor_info.set(
            "" if position is None else "x=%d, y=%d" % position
        )

    def _on_mouse_leave(self, _event: object) -> None:
        self._cursor_info.set("")

    def _on_canvas_click(self, event: "tk.Event[tk.Canvas]") -> None:
        position = self._canvas_to_image(event.x, event.y)
        if position is None:
            return
        self._circle_x.set(str(position[0]))
        self._circle_y.set(str(position[1]))
        self._set_status(
            "Центр круга выбран щелчком: x=%d, y=%d. Укажите радиус и нажмите "
            "«Нарисовать круг»." % position
        )

    # ------------------------------------------------------------------
    # Выбор изображения
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Выберите изображение",
            filetypes=[
                (
                    "Изображения PNG и JPG",
                    ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"),
                ),
                ("Все файлы", "*"),
            ],
        )
        if not path:
            self._set_status("Выбор файла отменён.")
            return
        try:
            image = image_io.load_image(path)
        except image_io.ImageFileError as error:
            self._show_error("Не удалось открыть изображение", str(error))
            return
        self._set_image(image, "файл «%s»" % os.path.basename(path))

    def _capture_from_camera(self) -> None:
        try:
            index = validation.parse_int(
                self._camera_index.get(),
                "Номер камеры",
                0,
                camera.MAX_CAMERA_INDEX,
            )
        except validation.InputError as error:
            self._show_error("Некорректный номер камеры", str(error))
            return

        self._set_status("Подключение к веб-камере №%d…" % index)
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            frame = camera.capture_frame(index)
        except camera.CameraError as error:
            self._show_error(
                "Веб-камера недоступна",
                "%s\n\n%s" % (error, camera.TROUBLESHOOTING),
            )
            return
        finally:
            self.configure(cursor="")
        self._set_image(frame, "снимок с веб-камеры №%d" % index)

    # ------------------------------------------------------------------
    # Операции над изображением
    # ------------------------------------------------------------------

    def _on_channel_change(self) -> None:
        if self._current is None:
            return
        self._render()
        channel = self._channel.get()
        if channel == CHANNEL_ALL:
            self._set_status(
                "Показано изображение со всеми цветовыми каналами."
            )
        else:
            self._set_status(
                "Показан %s канал изображения."
                % processing.CHANNEL_TITLES[channel]
            )

    def _apply_negative(self) -> None:
        if not self._require_image() or self._current is None:
            return
        self._current = processing.negative(self._current)
        self._render()
        self._set_status(
            "Показан негатив изображения. "
            "Повторное нажатие вернёт прежние цвета."
        )

    def _apply_rotation(self) -> None:
        if not self._require_image() or self._current is None:
            return
        try:
            angle = validation.parse_angle(self._angle.get())
            rotated = processing.rotate(self._current, angle)
        except (validation.InputError, processing.ProcessingError) as error:
            self._show_error("Не удалось повернуть изображение", str(error))
            self._angle_entry.focus_set()
            return

        self._current = rotated
        self._render()
        height, width = rotated.shape[:2]
        if angle == 0:
            self._set_status("Угол 0° — изображение не изменилось.")
            return
        direction = (
            "против часовой стрелки" if angle > 0 else "по часовой стрелке"
        )
        self._set_status(
            "Изображение повёрнуто на %g° %s. Новый размер: %d x %d пикселей."
            % (abs(angle), direction, width, height)
        )

    def _apply_circle(self) -> None:
        if not self._require_image() or self._current is None:
            return
        height, width = self._current.shape[:2]
        try:
            x, y, radius = validation.parse_circle(
                self._circle_x.get(),
                self._circle_y.get(),
                self._circle_radius.get(),
                width,
                height,
            )
            result = processing.draw_circle(
                self._current, x, y, radius, filled=self._circle_filled.get()
            )
        except (validation.InputError, processing.ProcessingError) as error:
            self._show_error(
                "Не удалось нарисовать круг",
                "%s\n\nРазмер изображения: %d x %d пикселей."
                % (error, width, height),
            )
            return

        self._current = result
        self._render()
        self._set_status(
            "Нарисован красный круг: центр (%d, %d), радиус %d пикселей."
            % (x, y, radius)
        )

    # ------------------------------------------------------------------
    # Сброс и сохранение
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        if not self._require_image() or self._original is None:
            return
        self._current = self._original.copy()
        self._channel.set(CHANNEL_ALL)
        self._render()
        self._set_status("Изменения сброшены, показано исходное изображение.")

    def _save(self) -> None:
        if not self._require_image():
            return
        image = self._displayed_image()
        if image is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Сохранить изображение",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not path:
            self._set_status("Сохранение отменено.")
            return
        try:
            saved_path = image_io.save_image(path, image)
        except image_io.ImageFileError as error:
            self._show_error("Не удалось сохранить изображение", str(error))
            return
        self._set_status("Изображение сохранено: %s" % saved_path)
