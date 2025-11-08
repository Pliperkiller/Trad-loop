import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta
import threading
import os

from ...application import DataExtractionService
from ...domain import MarketConfig, MarketType, Timeframe
from .widgets import LogWidget, DateSelector


class MainWindow:
    """
    Ventana principal de la aplicacion.

    Proporciona la interfaz grafica para configurar y ejecutar
    la extraccion de datos historicos de mercados financieros.
    """

    def __init__(self, root):
        """
        Inicializa la ventana principal.

        Args:
            root: Ventana raiz de tkinter
        """
        self.root = root
        self.root.title("Market Data Extractor")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        self.service = DataExtractionService()
        self.is_extracting = False

        self._setup_ui()
        self._load_initial_data()

    def _setup_ui(self):
        """Configura todos los componentes de la interfaz."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        ttk.Label(main_frame, text="Exchange:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.exchange_var = tk.StringVar()
        self.exchange_combo = ttk.Combobox(main_frame, textvariable=self.exchange_var, state='readonly')
        self.exchange_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        self.exchange_combo.bind('<<ComboboxSelected>>', self._on_exchange_changed)
        row += 1

        ttk.Label(main_frame, text="Tipo de Mercado:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.market_type_var = tk.StringVar()
        self.market_type_combo = ttk.Combobox(main_frame, textvariable=self.market_type_var, state='readonly')
        self.market_type_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        row += 1

        ttk.Label(main_frame, text="Simbolo (ej: BTC/USDT):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.symbol_var = tk.StringVar(value="BTC/USDT")
        self.symbol_entry = ttk.Entry(main_frame, textvariable=self.symbol_var)
        self.symbol_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        row += 1

        ttk.Label(main_frame, text="Temporalidad:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.timeframe_var = tk.StringVar()
        self.timeframe_combo = ttk.Combobox(main_frame, textvariable=self.timeframe_var, state='readonly')
        self.timeframe_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        row += 1

        ttk.Label(main_frame, text="Fecha Inicio:").grid(row=row, column=0, sticky=tk.W, pady=5)
        default_start = datetime.now() - timedelta(days=30)
        self.start_date_picker = DateSelector(main_frame, default_date=default_start)
        self.start_date_picker.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        row += 1

        ttk.Label(main_frame, text="Fecha Fin:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.end_date_picker = DateSelector(main_frame, default_date=datetime.now())
        self.end_date_picker.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        row += 1

        ttk.Label(main_frame, text="Ruta de Salida:").grid(row=row, column=0, sticky=tk.W, pady=5)
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        output_frame.columnconfigure(0, weight=1)

        self.output_path_var = tk.StringVar()
        self.output_path_entry = ttk.Entry(output_frame, textvariable=self.output_path_var)
        self.output_path_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.browse_button = ttk.Button(output_frame, text="Examinar...", command=self._browse_output_path)
        self.browse_button.grid(row=0, column=1, padx=(5, 0))
        row += 1

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=15)

        self.extract_button = ttk.Button(
            button_frame,
            text="Extraer Datos",
            command=self._start_extraction
        )
        self.extract_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancelar",
            command=self._cancel_extraction,
            state='disabled'
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        self.clear_log_button = ttk.Button(
            button_frame,
            text="Limpiar Logs",
            command=self._clear_logs
        )
        self.clear_log_button.pack(side=tk.LEFT, padx=5)
        row += 1

        ttk.Label(main_frame, text="Logs:").grid(row=row, column=0, sticky=tk.W, pady=(10, 5))
        row += 1

        self.log_widget = LogWidget(
            main_frame,
            height=15,
            width=80,
            wrap=tk.WORD,
            font=('Consolas', 9)
        )
        self.log_widget.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(row, weight=1)

    def _load_initial_data(self):
        """Carga los datos iniciales en los combos."""
        exchanges = self.service.get_available_exchanges()
        self.exchange_combo['values'] = exchanges
        if exchanges:
            self.exchange_combo.current(0)
            self._on_exchange_changed()

        self.log_widget.info("Aplicacion iniciada correctamente")
        self.log_widget.info(f"Exchanges disponibles: {', '.join(exchanges)}")

    def _on_exchange_changed(self, event=None):
        """Callback cuando se cambia el exchange seleccionado."""
        exchange_name = self.exchange_var.get()
        if not exchange_name:
            return

        market_types = self.service.get_supported_market_types(exchange_name)
        market_type_values = [mt.value for mt in market_types]
        self.market_type_combo['values'] = market_type_values
        if market_type_values:
            self.market_type_combo.current(0)

        timeframes = self.service.get_supported_timeframes(exchange_name)
        timeframe_values = [tf.value for tf in timeframes]
        self.timeframe_combo['values'] = timeframe_values
        if timeframe_values:
            self.timeframe_combo.current(2)

        self.log_widget.info(f"Exchange cambiado a: {exchange_name}")

    def _browse_output_path(self):
        """Abre un dialogo para seleccionar la ruta de salida."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=self._get_suggested_filename()
        )
        if filename:
            self.output_path_var.set(filename)

    def _get_suggested_filename(self) -> str:
        """Genera un nombre de archivo sugerido."""
        symbol = self.symbol_var.get().replace('/', '_')
        timeframe = self.timeframe_var.get()
        start = self.start_date_picker.get_date().strftime('%Y%m%d')
        end = self.end_date_picker.get_date().strftime('%Y%m%d')
        return f"{symbol}_{timeframe}_{start}_{end}.csv"

    def _clear_logs(self):
        """Limpia el widget de logs."""
        self.log_widget.clear_logs()
        self.log_widget.info("Logs limpiados")

    def _start_extraction(self):
        """Inicia el proceso de extraccion en un hilo separado."""
        if self.is_extracting:
            messagebox.showwarning("Advertencia", "Ya hay una extraccion en progreso")
            return

        if not self._validate_inputs():
            return

        self.is_extracting = True
        self.extract_button.config(state='disabled')
        self.cancel_button.config(state='normal')

        self.log_widget.info("=" * 60)
        self.log_widget.info("Iniciando extraccion de datos...")

        thread = threading.Thread(target=self._run_extraction, daemon=True)
        thread.start()

    def _validate_inputs(self) -> bool:
        """Valida las entradas del usuario."""
        if not self.symbol_var.get().strip():
            messagebox.showerror("Error", "Debe ingresar un simbolo")
            return False

        if not self.output_path_var.get().strip():
            messagebox.showerror("Error", "Debe seleccionar una ruta de salida")
            return False

        start_date = datetime.combine(self.start_date_picker.get_date(), datetime.min.time())
        end_date = datetime.combine(self.end_date_picker.get_date(), datetime.max.time())

        if start_date >= end_date:
            messagebox.showerror("Error", "La fecha de inicio debe ser anterior a la fecha de fin")
            return False

        return True

    def _run_extraction(self):
        """Ejecuta la extraccion de datos."""
        try:
            config = self._build_config()

            success, message = self.service.extract_market_data(
                config,
                progress_callback=self._progress_callback
            )

            if success:
                self.log_widget.success(message)
                self.root.after(0, lambda: messagebox.showinfo("Exito", message))
            else:
                self.log_widget.error(message)
                self.root.after(0, lambda: messagebox.showerror("Error", message))

        except Exception as e:
            error_msg = f"Error inesperado: {str(e)}"
            self.log_widget.error(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        finally:
            self.is_extracting = False
            self.root.after(0, lambda: self.extract_button.config(state='normal'))
            self.root.after(0, lambda: self.cancel_button.config(state='disabled'))

    def _build_config(self) -> MarketConfig:
        """Construye la configuracion de extraccion desde los inputs."""
        start_date = datetime.combine(self.start_date_picker.get_date(), datetime.min.time())
        end_date = datetime.combine(self.end_date_picker.get_date(), datetime.max.time())

        market_type_str = self.market_type_var.get()
        market_type = MarketType(market_type_str)

        timeframe_str = self.timeframe_var.get()
        timeframe = Timeframe(timeframe_str)

        return MarketConfig(
            exchange=self.exchange_var.get(),
            symbol=self.symbol_var.get(),
            market_type=market_type,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            output_path=self.output_path_var.get()
        )

    def _progress_callback(self, current: int, total: int, message: str):
        """Callback para actualizar el progreso."""
        self.root.after(0, lambda: self.log_widget.info(f"Progreso {current}/{total}: {message}"))

    def _cancel_extraction(self):
        """Cancela la extraccion en progreso."""
        if messagebox.askyesno("Confirmar", "Esta seguro de cancelar la extraccion?"):
            self.log_widget.warning("Cancelacion solicitada (no implementada completamente)")

    def run(self):
        """Inicia el loop principal de la aplicacion."""
        self.root.mainloop()
