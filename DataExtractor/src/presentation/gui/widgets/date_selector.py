import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import Callable, Optional


class DateSelector(ttk.Frame):
    """
    Widget personalizado para seleccion de fechas sin dependencias externas.

    Utiliza solo tkinter/ttk para maxima compatibilidad.
    """

    def __init__(self, parent, default_date: Optional[datetime] = None, **kwargs):
        """
        Inicializa el selector de fecha.

        Args:
            parent: Widget padre
            default_date: Fecha por defecto a mostrar
            **kwargs: Argumentos adicionales para Frame
        """
        super().__init__(parent, **kwargs)

        self.callback: Optional[Callable] = None

        if default_date is None:
            default_date = datetime.now()

        self._setup_ui(default_date)

    def _setup_ui(self, default_date: datetime):
        """Configura los componentes del selector."""
        self.year_var = tk.StringVar(value=str(default_date.year))
        self.month_var = tk.StringVar(value=str(default_date.month))
        self.day_var = tk.StringVar(value=str(default_date.day))

        year_label = ttk.Label(self, text="Año:")
        year_label.grid(row=0, column=0, padx=(0, 5))

        self.year_spinbox = ttk.Spinbox(
            self,
            from_=2010,
            to=2050,
            textvariable=self.year_var,
            width=6
        )
        self.year_spinbox.grid(row=0, column=1, padx=5)
        self.year_var.trace_add('write', self._on_change)

        month_label = ttk.Label(self, text="Mes:")
        month_label.grid(row=0, column=2, padx=(10, 5))

        self.month_spinbox = ttk.Spinbox(
            self,
            from_=1,
            to=12,
            textvariable=self.month_var,
            width=4
        )
        self.month_spinbox.grid(row=0, column=3, padx=5)
        self.month_var.trace_add('write', self._on_change)

        day_label = ttk.Label(self, text="Día:")
        day_label.grid(row=0, column=4, padx=(10, 5))

        self.day_spinbox = ttk.Spinbox(
            self,
            from_=1,
            to=31,
            textvariable=self.day_var,
            width=4
        )
        self.day_spinbox.grid(row=0, column=5, padx=5)
        self.day_var.trace_add('write', self._on_change)

        today_button = ttk.Button(
            self,
            text="Hoy",
            command=self._set_today,
            width=6
        )
        today_button.grid(row=0, column=6, padx=(10, 0))

    def _on_change(self, *args):
        """Callback cuando cambia alguna fecha."""
        if self.callback:
            self.callback()

    def _set_today(self):
        """Establece la fecha a hoy."""
        today = datetime.now()
        self.year_var.set(str(today.year))
        self.month_var.set(str(today.month))
        self.day_var.set(str(today.day))
        self._on_change()

    def get_date(self) -> datetime:
        """
        Obtiene la fecha seleccionada.

        Returns:
            Objeto datetime con la fecha seleccionada
        """
        try:
            year = int(self.year_var.get())
            month = int(self.month_var.get())
            day = int(self.day_var.get())

            month = max(1, min(12, month))

            if month in [1, 3, 5, 7, 8, 10, 12]:
                max_day = 31
            elif month in [4, 6, 9, 11]:
                max_day = 30
            else:
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                    max_day = 29
                else:
                    max_day = 28

            day = max(1, min(max_day, day))

            return datetime(year, month, day)
        except (ValueError, TypeError):
            return datetime.now()

    def set_date(self, date: datetime):
        """
        Establece la fecha del selector.

        Args:
            date: Fecha a establecer
        """
        self.year_var.set(str(date.year))
        self.month_var.set(str(date.month))
        self.day_var.set(str(date.day))
        self._on_change()

    def set_callback(self, callback: Callable):
        """
        Establece un callback para cuando cambie la fecha.

        Args:
            callback: Funcion a llamar cuando cambie la fecha
        """
        self.callback = callback
