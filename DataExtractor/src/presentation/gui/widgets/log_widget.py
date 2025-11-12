import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime


class LogWidget(scrolledtext.ScrolledText):
    """
    Widget personalizado para mostrar logs en la interfaz.

    Proporciona funcionalidad para agregar mensajes con timestamps
    y mantener un limite de lineas para evitar consumo excesivo de memoria.
    """

    def __init__(self, parent, max_lines=1000, **kwargs):
        """
        Inicializa el widget de logs.

        Args:
            parent: Widget padre
            max_lines: Numero maximo de lineas a mantener en el log
            **kwargs: Argumentos adicionales para ScrolledText
        """
        super().__init__(parent, **kwargs)
        self.max_lines = max_lines
        self.config(state='disabled')

    def add_log(self, message: str, level: str = "INFO"):
        """
        Agrega un mensaje al log.

        Args:
            message: Mensaje a agregar
            level: Nivel del log (INFO, WARNING, ERROR, SUCCESS)
        """
        self.config(state='normal')

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}\n"

        self.insert(tk.END, formatted_message)

        lines = int(self.index('end-1c').split('.')[0])
        if lines > self.max_lines:
            self.delete('1.0', f'{lines - self.max_lines}.0')

        self.see(tk.END)

        self.config(state='disabled')
        self.update_idletasks()

    def clear_logs(self):
        """Limpia todos los logs."""
        self.config(state='normal')
        self.delete('1.0', tk.END)
        self.config(state='disabled')

    def log_info(self, message: str):
        """Agrega un mensaje de nivel INFO."""
        self.add_log(message, "INFO")

    def log_warning(self, message: str):
        """Agrega un mensaje de nivel WARNING."""
        self.add_log(message, "WARNING")

    def log_error(self, message: str):
        """Agrega un mensaje de nivel ERROR."""
        self.add_log(message, "ERROR")

    def log_success(self, message: str):
        """Agrega un mensaje de nivel SUCCESS."""
        self.add_log(message, "SUCCESS")
