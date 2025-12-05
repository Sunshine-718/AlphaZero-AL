import sys
import os
import signal
import torch 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy,
    QMessageBox
)
from PyQt5.QtCore import QProcess, QSettings, Qt, pyqtSignal

# --- QProcess Management ---

class ClientProcess(QProcess):
    """
    An extended QProcess class for starting and monitoring a client.py instance.
    It includes the instance_id for log distinction.
    """
    def __init__(self, instance_id, parent=None, log_signal=None, process_finished_signal=None):
        super().__init__(parent)
        self.instance_id = instance_id
        self.log_signal = log_signal
        self.process_finished_signal = process_finished_signal
        
        self.readyReadStandardOutput.connect(self._handle_stdout)
        self.readyReadStandardError.connect(self._handle_stderr)
        self.finished.connect(self._handle_finished)

    def _handle_stdout(self):
        data = self.readAllStandardOutput().data().decode('utf-8', errors='ignore').strip()
        if data and self.log_signal:
            self.log_signal.emit(f"[{self.instance_id}][STDOUT] {data}", 'normal')

    def _handle_stderr(self):
        data = self.readAllStandardError().data().decode('utf-8', errors='ignore').strip()
        if data and self.log_signal:
            self.log_signal.emit(f"[{self.instance_id}][STDERR] {data}", 'error')
            
    def _handle_finished(self, exitCode, exitStatus):
        if self.process_finished_signal:
            self.process_finished_signal.emit(self.instance_id)
        
    def kill_process(self):
        """Sends a termination signal."""
        if self.state() == QProcess.Running:
            if self.terminate():
                return True
            else:
                self.kill()
                return False


class ActorGUI(QMainWindow):
    # Signals for inter-thread communication
    log_signal = pyqtSignal(str, str) # text, level
    process_finished_signal = pyqtSignal(int) # instance_id

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero Actor Client Manager (PyQt5)")
        self.setGeometry(100, 100, 1000, 700)
        self.processes = {}
        self.settings = QSettings("AlphaZeroAL", "ActorClientGUI")
        
        self.log_signal.connect(self.append_log)
        self.process_finished_signal.connect(self.handle_process_finished)
        
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
            self.append_log("Warning: Could not check CUDA availability. Default device set to 'cpu'.", 'warning')
        # --- Determine Default Device END ---

        # Define all parameters, their English labels, defaults, and widget types
        self.params = {
            # GUI Specific Parameter
            'num_instances': {'label': 'Parallel Instances', 'type': QSpinBox, 'default': 1, 'range': (1, 100)},
            # client.py Arguments
            '-n': {'label': 'Simulations (MCTS)', 'type': QSpinBox, 'default': 100, 'range': (1, 100000)},
            '--host': {'label': 'Server Host IP', 'type': QLineEdit, 'default': '127.0.0.1'},
            '--port': {'label': 'Server Port', 'type': QSpinBox, 'default': 7718, 'range': (1, 65535)},
            '--c_init': {'label': 'C_puct Init', 'type': QDoubleSpinBox, 'default': 1.25, 'range': (0.1, 10.0), 'decimals': 3},
            '--alpha': {'label': 'Dirichlet Alpha', 'type': QDoubleSpinBox, 'default': 0.7, 'range': (0.0, 1.0), 'decimals': 3},
            '--n_play': {'label': 'Games per Update', 'type': QSpinBox, 'default': 1, 'range': (1, 100)},
            '--discount': {'label': 'Discount Factor', 'type': QDoubleSpinBox, 'default': 0.99, 'range': (0.0, 1.0), 'decimals': 3},
            '--temp': {'label': 'Softmax Temperature', 'type': QDoubleSpinBox, 'default': 1.0, 'range': (0.0, 5.0), 'decimals': 3},
            '--tempD': {'label': 'Temperature Discount', 'type': QDoubleSpinBox, 'default': 0.93, 'range': (0.0, 1.0), 'decimals': 3},
            '--model': {'label': 'Model Type (CNN/ViT)', 'type': QLineEdit, 'default': 'CNN'},
            '--device': {'label': 'Device (cuda/cpu)', 'type': QLineEdit, 'default': DEFAULT_DEVICE}, 
            '--env': {'label': 'Environment Name', 'type': QLineEdit, 'default': 'Connect4'},
            '--retry': {'label': 'Retry Attempts', 'type': QSpinBox, 'default': 3, 'range': (0, 100)},
            '--no-cache': {'label': 'Disable Transposition Table', 'type': QCheckBox, 'default': False, 'flag': True},
            '--cache_size': {'label': 'Transposition Table Size', 'type': QSpinBox, 'default': 5000, 'range': (0, 100000)},
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
        
        self.start_button = QPushButton("🚀 Start All Clients")
        self.start_button.clicked.connect(self.start_clients)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("🛑 Stop All Clients")
        self.stop_button.clicked.connect(self.stop_all_clients)
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
        param_box = QGroupBox("Client Parameter Configuration")
        param_layout = QGridLayout()
        
        row = 0
        for key, config in self.params.items():
            label = QLabel(f"{config['label']}:")
            param_layout.addWidget(label, row, 0)
            
            widget_type = config['type']
            widget = widget_type()

            if widget_type in (QSpinBox, QDoubleSpinBox):
                widget.setRange(config['range'][0], config['range'][1])
                if widget_type == QDoubleSpinBox:
                    widget.setDecimals(config.get('decimals', 2))
                widget.setValue(config['default'])
                # Set step increments
                if key == '--c_init' or key == '--alpha':
                    widget.setSingleStep(0.01)
                elif key == '--cache_size':
                    widget.setSingleStep(100)
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
        
        # Log Box Group
        log_box = QGroupBox("Client Log Output")
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

    def get_client_args(self):
        """Collects command-line arguments for client.py from GUI controls."""
        args = []
        num_instances = self.widgets['num_instances'].value()

        for key, config in self.params.items():
            if key == 'num_instances':
                continue

            widget = self.widgets[key]
            
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                value = widget.value()
                args.extend([key, str(value)])
            elif isinstance(widget, QLineEdit):
                value = widget.text().strip()
                if value:
                    args.extend([key, value])
            elif isinstance(widget, QCheckBox):
                if config.get('flag') and widget.isChecked():
                    args.append(key)
        
        return num_instances, args

    def update_status_label(self):
        """Updates the status label and controls button enablement."""
        running_count = sum(1 for p in self.processes.values() if p.state() == QProcess.Running)
        total_count = self.widgets['num_instances'].value()
        self.status_label.setText(f"Running Instances: {running_count} / {len(self.processes)} (Target: {total_count})")
        
        is_running = running_count > 0
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)

        # Disable parameter modification while running
        for key in self.widgets:
            self.widgets[key].setEnabled(not is_running)

    def start_clients(self):
        """Starts all client processes."""
        if self.processes:
            self.append_log("Process list is not empty, please stop all processes first.", 'error')
            return

        self._save_settings() 
        num_instances, args = self.get_client_args()
        
        if not os.path.exists("client.py"):
             self.append_log("Error: client.py not found. Ensure it is in the same directory.", 'error')
             return

        python_executable = sys.executable 

        self.append_log(f"--- Preparing to start {num_instances} clients ---", 'info')
        self.append_log(f"Client Arguments: {' '.join(args)}", 'info')

        for i in range(1, num_instances + 1):
            instance_id = i
            process = ClientProcess(instance_id, self, self.log_signal, self.process_finished_signal)
            
            command_args = ['client.py'] + args
            
            self.append_log(f"Client {instance_id}: Starting...", 'normal')
            
            process.start(python_executable, command_args)
            
            if not process.waitForStarted(2000): 
                self.append_log(f"Client {instance_id} failed to start. {process.errorString()}", 'error')
                continue
                
            self.processes[instance_id] = process
            
        self.update_status_label()

    def stop_all_clients(self):
        """Stops all running client processes."""
        self.append_log("--- Requesting all clients to stop ---", 'warning')
        
        pids_to_stop = list(self.processes.keys()) 
        
        for instance_id in pids_to_stop:
            process = self.processes.get(instance_id)
            if process and process.state() == QProcess.Running:
                if process.kill_process():
                     self.append_log(f"Client {instance_id}: Termination signal sent (Terminate).", 'warning')
                else:
                     self.append_log(f"Client {instance_id}: Forcefully killed (Kill).", 'error')
        
        QApplication.processEvents() 
        
        if not self.processes:
            self.append_log("All clients stopped.", 'success')
            self.update_status_label()

    def handle_process_finished(self, instance_id):
        """Handles the process finished signal."""
        if instance_id in self.processes:
            process = self.processes[instance_id]
            exit_code = process.exitCode()
            exit_status = process.exitStatus()
            
            status_name = "Unknown"
            if exit_status == QProcess.NormalExit:
                status_name = "NormalExit (Code: 0)" if exit_code == 0 else "NormalExit (Non-zero Code)"
            elif exit_status == QProcess.CrashExit:
                status_name = "CrashExit (Crashed/Error Exit)"
                
            self.append_log(f"Client {instance_id} finished. Exit Code: {exit_code}, Status: {status_name}.", 'info')
            
            del self.processes[instance_id]
            
        self.update_status_label()
        
        if not self.processes and self.start_button.isEnabled() == False:
            self.append_log("--- All clients stopped or exited. ---", 'success')


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
        """Stops all processes and saves settings before closing the window."""
        if self.processes:
            reply = QMessageBox.question(self, 'Confirm Exit',
                "There are clients still running. Do you want to stop them before exiting?", 
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

            if reply == QMessageBox.Yes:
                self.stop_all_clients()
                QApplication.processEvents()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return

        self._save_settings()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ActorGUI()
    gui.show()
    sys.exit(app.exec_())