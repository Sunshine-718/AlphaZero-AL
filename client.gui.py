import sys
import os
import signal
import torch 
import re 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy,
    QMessageBox, QScrollArea, QFrame 
)
from PyQt5.QtCore import QProcess, QSettings, Qt, pyqtSignal, QTimer, QProcessEnvironment

UNBOUNDED_INT = 2000000000

class ClientProcess(QProcess):
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
        
    def kill_client(self):
        if self.state() == QProcess.Running:
            if self.terminate():
                return True
            else:
                self.kill()
                return False

class ActorGUI(QMainWindow):
    log_signal = pyqtSignal(str, str)
    process_finished_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero Actor Client Manager (PyQt5)")
        self.setGeometry(150, 150, 1000, 700)
        self.processes = {}
        self.next_instance_id = 1
        self.settings = QSettings("AlphaZeroAL", "ActorClientGUI")
        self.log_signal.connect(self.append_log)
        self.process_finished_signal.connect(self.handle_process_finished)
        
        self.total_client_uploaded_bytes = 0
        self.total_client_downloaded_bytes = 0
        self.traffic_log_pattern = re.compile(r"\[\[TRAFFIC_LOG::(UPLOAD|DOWNLOAD)::\+::(\d+)\]\]")
        
        self.traffic_update_timer = QTimer(self)
        self.traffic_update_timer.setInterval(500) 
        self.traffic_update_timer.timeout.connect(self._update_client_traffic_display)
        self.traffic_update_timer.start() 
        
        self._init_default_args()
        self._setup_ui()
        self._load_settings()
        self.update_status_label()

    def format_bytes(self, bytes_num):
        if bytes_num is None or bytes_num == 0: return "0 B"
        bytes_num = float(bytes_num)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_num < 1024.0: return f"{bytes_num:.2f} {unit}"
            bytes_num /= 1024.0
        return f"{bytes_num:.2f} PB"

    def _update_client_traffic_display(self):
        self.value_total_downloaded.setText(self.format_bytes(self.total_client_downloaded_bytes))
        self.value_total_uploaded.setText(self.format_bytes(self.total_client_uploaded_bytes))

    def reset_traffic_stats(self):
        self.total_client_uploaded_bytes = 0
        self.total_client_downloaded_bytes = 0
        self._update_client_traffic_display()
        self.append_log("--- Client traffic statistics reset ---", 'info')

    def _init_default_args(self):
        DEFAULT_DEVICE = 'cpu'
        try:
            if torch.cuda.is_available(): DEFAULT_DEVICE = 'cuda'
        except Exception: pass

        self.params_groups = [
            ("🔌 1. Connection & Client Setup", {
                '--host': {'label': 'Server Host IP', 'type': QLineEdit, 'default': '127.0.0.1'},
                '--port': {'label': 'Server Port', 'type': QSpinBox, 'default': 7718, 'range': (1, 65535)},
                'num_instances': {'label': 'Parallel Instances', 'type': QSpinBox, 'default': 1, 'range': (1, 100)},
                '--device': {'label': 'Device (cuda/cpu)', 'type': QLineEdit, 'default': DEFAULT_DEVICE}, 
                '--retry': {'label': 'Retry Attempts', 'type': QSpinBox, 'default': 3, 'range': (0, UNBOUNDED_INT)}, 
            }),
            ("🧠 2. MCTS Core Parameters", {
                '-n': {'label': 'Simulations (MCTS)', 'type': QSpinBox, 'default': 100, 'range': (1, UNBOUNDED_INT)}, 
                '--c_init': {'label': 'C_puct Init', 'type': QDoubleSpinBox, 'default': 1.25, 'range': (0.1, 10.0), 'decimals': 3},
                '--alpha': {'label': 'Dirichlet Alpha', 'type': QDoubleSpinBox, 'default': 0.7, 'range': (0.0, 1.0), 'decimals': 3},
            }),
            ("🔥 3. Policy & Game Flow", {
                '--temp': {'label': 'Softmax Temperature', 'type': QDoubleSpinBox, 'default': 1.0, 'range': (0.0, UNBOUNDED_INT), 'decimals': 3}, 
                '--tempD': {'label': 'Temperature Discount', 'type': QDoubleSpinBox, 'default': 0.93, 'range': (0.0, 1.0), 'decimals': 3},
                '--discount': {'label': 'Discount Factor', 'type': QDoubleSpinBox, 'default': 0.99, 'range': (0.0, 1.0), 'decimals': 3},
                '--n_play': {'label': 'Games per Update', 'type': QSpinBox, 'default': 1, 'range': (1, UNBOUNDED_INT)}, 
            }),
            ("🎮 4. Env, Model & Cache", {
                '--env': {'label': 'Environment Name', 'type': QLineEdit, 'default': 'Connect4'},
                '--model': {'label': 'Model Type (CNN/ViT)', 'type': QLineEdit, 'default': 'CNN'},
                '--cache_size': {'label': 'Transposition Table Size', 'type': QSpinBox, 'default': 5000, 'range': (0, UNBOUNDED_INT)}, 
                '--no-cache': {'label': 'Disable Transposition Table', 'type': QCheckBox, 'default': False, 'flag': True},
            })
        ]
        self.params = {k: v for _, group in self.params_groups for k, v in group.items()}
        self.widgets = {}
        self.cache_default_size = self.params['--cache_size']['default'] 

    def _setup_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setMinimumWidth(350) 

        # 1. Control Box
        control_box = QGroupBox("Control Panel")
        control_layout = QGridLayout()
        
        control_layout.addWidget(QLabel("Number of Clients:"), 0, 0)
        self.num_clients_widget = QSpinBox()
        self.num_clients_widget.setRange(1, 100)
        self.num_clients_widget.setValue(1)
        self.num_clients_widget.setEnabled(False) 
        control_layout.addWidget(self.num_clients_widget, 0, 1)

        self.start_button = QPushButton("🚀 Start Clients")
        self.start_button.clicked.connect(self.start_clients)
        control_layout.addWidget(self.start_button, 1, 0)
        
        self.stop_button = QPushButton("🛑 Stop All Clients")
        self.stop_button.clicked.connect(self.stop_all_clients)
        control_layout.addWidget(self.stop_button, 1, 1)

        self.reset_button = QPushButton("🔄 Reset Parameters")
        self.reset_button.clicked.connect(self.reset_parameters)
        control_layout.addWidget(self.reset_button, 2, 0)
        
        self.reset_traffic_button = QPushButton("🗑️ Reset Traffic")
        self.reset_traffic_button.clicked.connect(self.reset_traffic_stats)
        control_layout.addWidget(self.reset_traffic_button, 2, 1)
        
        control_box.setLayout(control_layout)
        left_layout.addWidget(control_box)

        # 2. Status Box
        status_box = QGroupBox("Clients Status & Network Traffic") 
        status_layout = QVBoxLayout() 
        
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        traffic_group = QGroupBox("Network Traffic (Clients Aggregate)")
        traffic_layout = QGridLayout()
        
        self.label_total_downloaded = QLabel("⬇️ Total Download:")
        self.value_total_downloaded = QLabel("0 B")
        self.value_total_downloaded.setStyleSheet("font-weight: bold; color: blue;")
        
        self.label_total_uploaded = QLabel("⬆️ Total Upload:")
        self.value_total_uploaded = QLabel("0 B")
        self.value_total_uploaded.setStyleSheet("font-weight: bold; color: green;")
        
        traffic_layout.addWidget(self.label_total_downloaded, 0, 0)
        traffic_layout.addWidget(self.value_total_downloaded, 0, 1)
        traffic_layout.addWidget(self.label_total_uploaded, 1, 0)
        traffic_layout.addWidget(self.value_total_uploaded, 1, 1)
        
        traffic_group.setLayout(traffic_layout)
        status_layout.addWidget(traffic_group)
        
        status_box.setLayout(status_layout)
        left_layout.addWidget(status_box)

        # 3. Parameter Configuration
        param_container = QWidget()
        param_layout = QGridLayout(param_container)
        row = 0
        for group_title, params_dict in self.params_groups:
            separator_label = QLabel(f"<b>{group_title}</b>")
            separator_label.setStyleSheet("margin-top: 5px; margin-bottom: 2px;")
            param_layout.addWidget(separator_label, row, 0, 1, 2)
            row += 1
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            param_layout.addWidget(line, row, 0, 1, 2)
            row += 1
            for key, config in params_dict.items():
                label_text = f"{config['label']}:"
                widget_type = config['type']
                if widget_type in (QSpinBox, QDoubleSpinBox) and 'range' in config:
                    min_val, max_val = config['range']
                    if widget_type == QSpinBox:
                        min_str = str(min_val)
                        max_str = "∞" if max_val == UNBOUNDED_INT else str(max_val)
                    else:
                        min_str = f"{min_val:.6g}" 
                        max_str = "∞" if max_val == UNBOUNDED_INT else f"{max_val:.6g}"
                    if max_val == UNBOUNDED_INT: label_text = f"{config['label']} (Min {min_str}):"
                    elif min_val != -UNBOUNDED_INT: label_text = f"{config['label']} ({min_str}-{max_str}):"
                label = QLabel(label_text)
                param_layout.addWidget(label, row, 0)
                widget = widget_type()
                if widget_type == QSpinBox:
                    min_val = config['range'][0]
                    max_val = config['range'][1]
                    if max_val == UNBOUNDED_INT: max_val = int(1e9)
                    widget.setRange(min_val, max_val) 
                    widget.setValue(config['default'])
                    if config.get('single_step'): widget.setSingleStep(config['single_step'])
                    elif key == '--cache_size': widget.setSingleStep(100)
                elif widget_type == QDoubleSpinBox:
                    max_val = config['range'][1] if config['range'][1] < UNBOUNDED_INT else 1e9
                    widget.setRange(config['range'][0], max_val)
                    widget.setDecimals(config.get('decimals', 3))
                    widget.setValue(config['default'])
                    if key in ('--c_init', '--alpha', '--tempD'): widget.setSingleStep(0.01)
                    elif key == '--temp': widget.setSingleStep(0.1)
                elif widget_type == QCheckBox:
                    widget.setChecked(config['default'])
                elif widget_type == QLineEdit:
                    widget.setText(config['default'])
                param_layout.addWidget(widget, row, 1)
                self.widgets[key] = widget
                if widget_type in (QSpinBox, QDoubleSpinBox):
                    widget.valueChanged.connect(lambda val, k=key: self._log_parameter_change(k, val))
                    if key == 'num_instances': widget.valueChanged.connect(lambda val: self.num_clients_widget.setValue(val))
                elif widget_type == QCheckBox:
                    widget.stateChanged.connect(lambda state, k=key: self._log_parameter_change(k, state))
                elif widget_type == QLineEdit:
                    widget.editingFinished.connect(lambda w=widget, k=key: self._log_parameter_change(k, w.text()))
                row += 1
        param_container.setLayout(param_layout) 
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) 
        scroll_area.setWidget(param_container) 
        param_group_box = QGroupBox("Client Parameter Configuration")
        param_group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        param_group_box_layout = QVBoxLayout(param_group_box)
        param_group_box_layout.addWidget(scroll_area)
        left_layout.addWidget(param_group_box)

        self.cache_size_spinbox = self.widgets['--cache_size']
        self.no_cache_checkbox = self.widgets['--no-cache']
        self.cache_size_spinbox.valueChanged.connect(self._sync_size_to_check)
        self.no_cache_checkbox.stateChanged.connect(self._sync_check_to_size)
        if self.cache_size_spinbox.value() == 0: self.no_cache_checkbox.setChecked(True)

        right_layout = QVBoxLayout()
        log_box = QGroupBox("Client Log Output")
        log_layout = QVBoxLayout()
        self.clear_log_button = QPushButton("🗑️ Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        log_layout.addWidget(self.clear_log_button)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.log_text_edit)
        log_box.setLayout(log_layout)
        right_layout.addWidget(log_box)
        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)
        self.setCentralWidget(central_widget)
        
    def _log_parameter_change(self, key, value):
        widget = self.widgets[key]
        display_value = str(value)
        if isinstance(widget, QCheckBox): display_value = "Enabled" if value == Qt.Checked else "Disabled"
        elif isinstance(widget, QDoubleSpinBox): display_value = f"{value:.4g}"
        label = self.params[key]['label']
        self.append_log(f"[PARAM CHANGE] '{label}' ({key}) set to: {display_value}", 'info')

    def _sync_size_to_check(self, value):
        checkbox = self.no_cache_checkbox
        checkbox.blockSignals(True)
        if value == 0:
            if not checkbox.isChecked(): checkbox.setChecked(True)
        elif value > 0:
            if checkbox.isChecked(): checkbox.setChecked(False)
        checkbox.blockSignals(False)

    def _sync_check_to_size(self, state):
        spinbox = self.cache_size_spinbox
        spinbox.blockSignals(True)
        if state == Qt.Checked: spinbox.setValue(0)
        elif state == Qt.Unchecked:
            if spinbox.value() == 0: spinbox.setValue(self.cache_default_size)
        spinbox.blockSignals(False)
    
    def clear_log(self):
        self.log_text_edit.clear()
        self.append_log("--- Log manually cleared ---", 'info')

    def _load_settings(self):
        for key, config in self.params.items():
            widget = self.widgets[key]
            widget.blockSignals(True)
            value = self.settings.value(key, config.get('default'))
            if value is not None:
                if isinstance(widget, QSpinBox):
                    try: widget.setValue(int(float(value)))
                    except (ValueError, TypeError): widget.setValue(config['default'])
                elif isinstance(widget, QDoubleSpinBox):
                    try: widget.setValue(float(value))
                    except (ValueError, TypeError): widget.setValue(config['default']) 
                elif isinstance(widget, QCheckBox):
                    if isinstance(value, str): widget.setChecked(value.lower() == 'true')
                    else: widget.setChecked(bool(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
            widget.blockSignals(False)
        self.num_clients_widget.setValue(self.widgets['num_instances'].value())

    def _save_settings(self):
        for key, config in self.params.items():
            widget = self.widgets[key]
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                self.settings.setValue(key, widget.value())
            elif isinstance(widget, QCheckBox):
                self.settings.setValue(key, widget.isChecked())
            elif isinstance(widget, QLineEdit):
                self.settings.setValue(key, widget.text())

    def get_client_args(self):
        args = []
        num_instances = self.widgets['num_instances'].value()
        for key, config in self.params.items():
            if key == 'num_instances': continue
            widget = self.widgets[key]
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                value = widget.value()
                args.extend([key, str(value)])
            elif isinstance(widget, QLineEdit):
                value = widget.text().strip()
                if value: args.extend([key, value])
            elif isinstance(widget, QCheckBox):
                if config.get('flag') and widget.isChecked(): args.append(key)
        return num_instances, args

    def update_status_label(self):
        running_count = sum(1 for p in self.processes.values() if p.state() == QProcess.Running)
        total_count = self.widgets['num_instances'].value()
        self.status_label.setText(f"Running Instances: {running_count} / {len(self.processes)} (Target: {total_count})")
        is_running = running_count > 0
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)
        self.reset_button.setEnabled(not is_running)
        for key in self.widgets: self.widgets[key].setEnabled(not is_running)

    def start_clients(self):
        if any(p.state() == QProcess.Running for p in self.processes.values()):
            self.append_log("Some clients are already running. Please stop them first.", 'error')
            return
        self.processes.clear() 
        self._save_settings() 
        num_instances, args = self.get_client_args()
        if not os.path.exists("client.py"):
             self.append_log("Error: client.py not found. Ensure it is in the same directory.", 'error')
             return
        python_executable = sys.executable 
        self.append_log(f"--- Preparing to start {num_instances} clients ---", 'info')
        self.append_log(f"Client Arguments: {' '.join(args)}", 'info')
        for i in range(1, num_instances + 1):
            instance_id = f"Client-{self.next_instance_id}"
            process = ClientProcess(instance_id, self, self.log_signal, self.process_finished_signal)
            env = QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONPATH", os.getcwd())
            env.insert("PYTHONIOENCODING", "utf-8")
            process.setProcessEnvironment(env)
            command_args = ['-u', 'client.py'] + args 
            self.append_log(f"Client {instance_id}: Starting...", 'normal')
            process.start(python_executable, command_args)
            if not process.waitForStarted(2000): 
                self.append_log(f"Client {instance_id} failed to start. {process.errorString()}", 'error')
            else:
                self.processes[instance_id] = process
                self.next_instance_id += 1
        self.update_status_label()

    def stop_all_clients(self):
        if not self.processes:
            QMessageBox.information(self, "Info", "No clients are currently running.")
            return
        self.append_log("Stopping all clients...", 'info')
        pids_to_kill = list(self.processes.keys())
        for instance_id in pids_to_kill:
            process = self.processes.get(instance_id)
            if process and process.state() == QProcess.Running:
                process.kill_client()
        QApplication.processEvents() 
        self.update_status_label()

    def handle_process_finished(self, instance_id):
        if instance_id in self.processes:
            self.processes.pop(instance_id, None)
            self.append_log(f"Client {instance_id} stopped or exited.", 'warning')
        self.update_status_label()
        if not self.processes and self.start_button.isEnabled() == False:
            self.append_log("--- All clients stopped or exited. ---", 'success')

    def append_log(self, text, level='normal'):
        match = self.traffic_log_pattern.search(text)
        if match:
            direction = match.group(1)
            try:
                bytes_added = int(match.group(2))
                if direction == 'UPLOAD': self.total_client_uploaded_bytes += bytes_added
                elif direction == 'DOWNLOAD': self.total_client_downloaded_bytes += bytes_added
                return 
            except ValueError: pass 
        
        bg_color = self.log_text_edit.palette().color(self.log_text_edit.backgroundRole())
        is_dark_mode = (bg_color.red() + bg_color.green() + bg_color.blue()) < 382
        if is_dark_mode:
            color_map = {'normal': 'white', 'error': '#FF6B6B', 'warning': '#FFCC66', 'success': '#8BC34A', 'info': '#66A5FF'}
            fallback_color = 'white'
        else:
            color_map = {'normal': 'black', 'error': 'red', 'warning': 'orange', 'success': 'green', 'info': 'blue'}
            fallback_color = 'black'
        color = color_map.get(level, fallback_color)
        html = f'<span style="color:{color};">{text}</span><br>'
        self.log_text_edit.insertHtml(html)
        self.log_text_edit.ensureCursorVisible()

    def reset_parameters(self):
        reply = QMessageBox.question(self, 'Confirm Reset', "Reset parameters to default?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No: return
        self.cache_size_spinbox.blockSignals(True)
        self.no_cache_checkbox.blockSignals(True)
        self.append_log("Resetting parameters...", 'warning')
        for key, config in self.params.items():
            widget = self.widgets[key]
            default = config['default']
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)): widget.setValue(default)
            elif isinstance(widget, QCheckBox): widget.setChecked(default)
            elif isinstance(widget, QLineEdit): widget.setText(str(default))
        self.cache_size_spinbox.blockSignals(False)
        self.no_cache_checkbox.blockSignals(False)
        if self.cache_default_size == 0: self.no_cache_checkbox.setChecked(True)
        else: self.no_cache_checkbox.setChecked(False)

    def closeEvent(self, event):
        if any(p.state() == QProcess.Running for p in self.processes.values()):
            reply = QMessageBox.question(self, 'Confirm Exit', "Clients are running. Stop before exit?", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
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