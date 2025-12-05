import sys
import os
import torch
import signal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy,
    QMessageBox
)
from PyQt5.QtCore import QProcess, QSettings, Qt, pyqtSignal, QProcessEnvironment

# --- Server Process Management ---

class ServerProcess(QProcess):
    """
    A simple QProcess wrapper for the single server instance.
    """
    # log_signal is connected to the GUI's append_log method
    def __init__(self, parent=None, log_signal=None):
        super().__init__(parent)
        self.log_signal = log_signal
        
        # Connect signals to handle output
        self.readyReadStandardOutput.connect(self._handle_stdout)
        self.readyReadStandardError.connect(self._handle_stderr)

    def _handle_stdout(self):
        # Read and decode all standard output data
        data = self.readAllStandardOutput().data().decode('utf-8', errors='ignore').strip()
        if data and self.log_signal:
            self.log_signal.emit(f"[SERVER][STDOUT] {data}", 'normal')

    def _handle_stderr(self):
        # Read and decode all standard error data (Flask logs usually go here)
        data = self.readAllStandardError().data().decode('utf-8', errors='ignore').strip()
        if data and self.log_signal:
            self.log_signal.emit(f"[SERVER][STDERR] {data}", 'error')
            
    def kill_server(self):
        """Attempts to terminate the server process."""
        if self.state() == QProcess.Running:
            if self.terminate():
                return True
            else:
                self.kill()
                return False

# --- GUI Application ---

class ServerGUI(QMainWindow):
    # Signal for logging from the QProcess thread to the main thread
    log_signal = pyqtSignal(str, str) # text, level

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero Training Server Manager (PyQt5)")
        self.setGeometry(100, 100, 1000, 700)
        
        self.server_process = ServerProcess(self, self.log_signal)
        self.server_process.finished.connect(self.handle_process_finished)
        self.settings = QSettings("AlphaZeroAL", "ServerGUI")
        
        self.log_signal.connect(self.append_log)
        
        self._init_default_args()
        self._setup_ui()
        self._load_settings()
        self.update_status_label()

    def _init_default_args(self):
        # --- Determine Default Device (CUDA/CPU) ---
        DEFAULT_DEVICE = 'cpu'
        try:
            if torch.cuda.is_available():
                DEFAULT_DEVICE = 'cuda'
        except Exception:
            # Fallback if torch is not installed or import fails
            self.append_log("Warning: Could not check CUDA availability. Default device set to 'cpu'.", 'warning')
        # --- Determine Default Device END ---

        # Define all parameters, their English labels, defaults, and widget types from server.py
        self.params = {
            # Core Arguments
            '--host': {'label': 'Host IP', 'type': QLineEdit, 'default': '0.0.0.0'},
            '--port': {'label': 'Port', 'type': QSpinBox, 'default': 7718, 'range': (1024, 65535)},
            '-d': {'label': 'Device (cuda/cpu)', 'type': QLineEdit, 'default': DEFAULT_DEVICE},
            '-e': {'label': 'Environment', 'type': QLineEdit, 'default': 'Connect4'},
            '-m': {'label': 'Model Type (CNN)', 'type': QLineEdit, 'default': 'CNN'},
            '--name': {'label': 'Pipeline Name', 'type': QLineEdit, 'default': 'AZ'},
            
            # MCTS/Pipeline Hyperparameters
            '-n': {'label': 'MCTS Simulations', 'type': QSpinBox, 'default': 100, 'range': (1, 100000)},
            '-c': {'label': 'C_puct Init', 'type': QDoubleSpinBox, 'default': 1.25, 'range': (0.1, 10.0), 'decimals': 3},
            '-a': {'label': 'Dirichlet Alpha', 'type': QDoubleSpinBox, 'default': 0.7, 'range': (0.0, 1.0), 'decimals': 3},
            '--discount': {'label': 'Discount Factor', 'type': QDoubleSpinBox, 'default': 0.99, 'range': (0.0, 1.0), 'decimals': 3},
            '-t': {'label': 'Softmax Temperature', 'type': QDoubleSpinBox, 'default': 1, 'range': (0.0, 5.0), 'decimals': 3},
            '--mcts_n': {'label': 'Pure MCTS Playouts', 'type': QSpinBox, 'default': 1000, 'range': (10, 100000)},
            '--thres': {'label': 'Win Rate Threshold', 'type': QDoubleSpinBox, 'default': 0.65, 'range': (0.5, 1.0), 'decimals': 3},
            '--num_eval': {'label': 'Evaluation Games', 'type': QSpinBox, 'default': 50, 'range': (1, 500)},
            '--interval': {'label': 'Eval Interval', 'type': QSpinBox, 'default': 10, 'range': (1, 100)},
            
            # Training/Buffer Hyperparameters
            '--lr': {'label': 'Learning Rate', 'type': QDoubleSpinBox, 'default': 1e-3, 'range': (1e-6, 1e-1), 'decimals': 6, 'single_step': 1e-4},
            '-b': {'label': 'Batch Size', 'type': QSpinBox, 'default': 512, 'range': (32, 4096)},
            '--buf': {'label': 'Buffer Size', 'type': QSpinBox, 'default': 5000, 'range': (100, 100000)},
            '--n_play': {'label': 'N Playout', 'type': QSpinBox, 'default': 1, 'range': (1, 10)},
            
            # Cache/Misc
            '--no-cache': {'label': 'Disable Transposition Table', 'type': QCheckBox, 'default': False, 'flag': True},
            '--cache_size': {'label': 'Transposition Table Size', 'type': QSpinBox, 'default': 5000, 'range': (0, 100000)},
            '--pause': {'label': 'Start Paused', 'type': QCheckBox, 'default': False, 'flag': True},
        }
        self.widgets = {}

    def _setup_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        # --- Left Side: Controls and Parameters ---
        left_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(400) 

        # 1. Control Box
        control_box = QGroupBox("Control Panel")
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("🚀 Start Server")
        self.start_button.clicked.connect(self.start_server)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("🛑 Stop Server")
        self.stop_button.clicked.connect(self.stop_server)
        
        control_layout.addWidget(self.stop_button)
        control_box.setLayout(control_layout)
        left_layout.addWidget(control_box)

        # 2. Status Box
        status_box = QGroupBox("Status")
        status_layout = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        status_box.setLayout(status_layout)
        left_layout.addWidget(status_box)

        # 3. Parameter Configuration
        param_box = QGroupBox("Server Parameter Configuration")
        param_layout = QGridLayout()
        
        row = 0
        for key, config in self.params.items():
            label = QLabel(f"{config['label']}:")
            param_layout.addWidget(label, row, 0)
            
            widget_type = config['type']
            widget = widget_type()

            if widget_type == QSpinBox:
                widget.setRange(config['range'][0], config['range'][1])
                widget.setValue(config['default'])
                if config.get('single_step'):
                     widget.setSingleStep(config['single_step'])
            elif widget_type == QDoubleSpinBox:
                widget.setRange(config['range'][0], config['range'][1])
                widget.setDecimals(config.get('decimals', 2))
                widget.setValue(config['default'])
                # Custom single step for LR
                if config.get('single_step'):
                    widget.setSingleStep(config['single_step'])
                elif key == '--c_init' or key == '--alpha':
                    widget.setSingleStep(0.01)
            elif widget_type == QCheckBox:
                widget.setChecked(config['default'])
            elif widget_type == QLineEdit:
                widget.setText(config['default'])
            
            param_layout.addWidget(widget, row, 1)
            self.widgets[key] = widget
            row += 1
            
        param_box.setLayout(param_layout)
        left_layout.addWidget(param_box)
        left_layout.addStretch(1) 

        # --- Right Side: Log Area ---
        right_layout = QVBoxLayout()
        
        log_box = QGroupBox("Server Log Output")
        log_layout = QVBoxLayout()
        
        # Clear Log Button
        self.clear_log_button = QPushButton("🗑️ Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        log_layout.addWidget(self.clear_log_button)
        
        # Log Text Edit
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        log_layout.addWidget(self.log_text_edit)
        log_box.setLayout(log_layout)
        right_layout.addWidget(log_box)

        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)

        self.setCentralWidget(central_widget)
    
    def clear_log(self):
        """Clears the content of the log text box."""
        self.log_text_edit.clear()
        self.append_log("--- Log manually cleared ---", 'info')

    def _load_settings(self):
        """Loads saved parameter values from QSettings."""
        for key, config in self.params.items():
            widget = self.widgets[key]
            value = self.settings.value(key, config['default'])
            if value is not None:
                if isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(value == 'true' or value is True)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
            
    def _save_settings(self):
        """Saves current parameter values to QSettings."""
        for key, config in self.params.items():
            widget = self.widgets[key]
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                self.settings.setValue(key, widget.value())
            elif isinstance(widget, QCheckBox):
                self.settings.setValue(key, widget.isChecked())
            elif isinstance(widget, QLineEdit):
                self.settings.setValue(key, widget.text())

    def get_server_args(self):
        """Collects command-line arguments for server.py from GUI controls."""
        args = []
        for key, config in self.params.items():
            widget = self.widgets[key]
            
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                value = widget.value()
                # Special handling for scientific notation (like learning rate)
                if key == '--lr':
                    # Format to scientific notation or fixed-point with max decimals
                    args.extend([key, f"{value:.6g}"]) 
                else:
                    args.extend([key, str(value)])
            elif isinstance(widget, QLineEdit):
                value = widget.text().strip()
                if value:
                    args.extend([key, value])
            elif isinstance(widget, QCheckBox):
                if config.get('flag') and widget.isChecked():
                    args.append(key)
        
        return args

    def update_status_label(self):
        """Updates the status label and controls button enablement."""
        is_running = self.server_process.state() == QProcess.Running
        status_text = "Running" if is_running else "Stopped"
        status_color = "green" if is_running else "red"
        
        self.status_label.setText(f"Server Status: <span style='color:{status_color};font-weight:bold;'>{status_text}</span>")
        
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)

        # Disable parameter modification while running
        for key in self.widgets:
            self.widgets[key].setEnabled(not is_running)

    def start_server(self):
        """Starts the server process."""
        if self.server_process.state() == QProcess.Running:
            self.append_log("Server is already running.", 'error')
            return

        self._save_settings() 
        args = self.get_server_args()
        
        if not os.path.exists("server.py"):
             self.append_log("Error: server.py not found. Ensure it is in the same directory.", 'error')
             return

        python_executable = sys.executable 

        self.append_log(f"--- Starting Server ---", 'info')
        
        # FIX: Use '-u' flag to force unbuffered output for real-time logging
        command_list = ['-u', 'server.py'] + args
        self.append_log(f"Command: {python_executable} {' '.join(command_list)}", 'info')

        self.server_process.start(python_executable, command_list)
        
        if not self.server_process.waitForStarted(2000): 
            self.append_log(f"Server failed to start: {self.server_process.errorString()}", 'error')
        
        self.update_status_label()

    def stop_server(self):
        """Stops the server process."""
        if self.server_process.state() != QProcess.Running:
            self.append_log("Server is not running.", 'warning')
            return
            
        self.append_log("--- Requesting Server to stop ---", 'warning')
        
        if self.server_process.kill_server():
            self.append_log("Termination signal sent to server (Terminate).", 'warning')
        else:
            self.append_log("Server forcefully killed (Kill).", 'error')
        
        QApplication.processEvents() 
        self.update_status_label()


    def handle_process_finished(self, exit_code, exit_status):
        """Handles the server process finishing."""
        status_name = "Unknown"
        if exit_status == QProcess.NormalExit:
            status_name = "Normal Exit (Code: 0)" if exit_code == 0 else f"Normal Exit (Non-zero Code: {exit_code})"
            level = 'info' if exit_code == 0 else 'warning'
        elif exit_status == QProcess.CrashExit:
            status_name = "Crash Exit (Crashed/Error Exit)"
            level = 'error'

        self.append_log(f"Server Process Finished. Exit Status: {status_name}.", level)
        self.update_status_label()
        
    def append_log(self, text, level='normal'):
        """Appends log text to the text box and scrolls to the bottom."""
        color_map = {
            'normal': 'black',
            'error': 'red',
            'warning': 'orange',
            'success': 'green',
            'info': 'blue'
        }
        color = color_map.get(level, 'black')
        
        html = f'<span style="color:{color};">{text}</span><br>'
        self.log_text_edit.insertHtml(html)
        self.log_text_edit.ensureCursorVisible()

    def closeEvent(self, event):
        """Stops the server process and saves settings before closing the window."""
        if self.server_process.state() == QProcess.Running:
            reply = QMessageBox.question(self, 'Confirm Exit',
                "The server is running. Do you want to stop it before exiting?", 
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

            if reply == QMessageBox.Yes:
                self.stop_server()
                QApplication.processEvents()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return

        self._save_settings()
        event.accept()

if __name__ == '__main__':
    # Ensure torch is available before starting the app (it's used in _init_default_args)
    try:
        if 'torch' not in sys.modules:
            import torch
    except ImportError:
        pass # The warning will be logged in _init_default_args

    app = QApplication(sys.argv)
    gui = ServerGUI()
    gui.show()
    sys.exit(app.exec_())