import tkinter as tk
from .gui import MainWindow


def run_app():
    """
    Punto de entrada para ejecutar la aplicacion.

    Crea la ventana raiz de tkinter e inicializa la aplicacion.
    """
    root = tk.Tk()
    app = MainWindow(root)
    app.run()


if __name__ == "__main__":
    run_app()
