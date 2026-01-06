import sys
import os
import torch
import signal
import requests
import json
import re 
import time # <<< ADDED: 导入 time 模块
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy,
    QMessageBox, QFrame, QScrollArea 
)
from PyQt5.QtCore import QProcess, QSettings, Qt, pyqtSignal, QProcessEnvironment, QTimer

UNBOUNDED_INT = 2000000000

class ServerProcess(QProcess):
    log_signal = pyqtSignal(str, str)
    
    def __init__(self, parent=None, log_signal=None):
        super().__init__(parent)
        self.log_signal = log_signal
        self.readyReadStandardOutput.connect(self._handle_stdout)
        self.readyReadStandardError.connect(self._handle_stderr)

    def _handle_stdout(self):
        data = self.readAllStandardOutput().data().decode('utf-8', errors='ignore').strip()
        if data and self.log_signal:
            # 所有来自 server.py 的 STDOUT 都通过信号发送到 GUI 的 append_log
            self.log_signal.emit(f"[SERVER][STDOUT] {data}", 'normal')

    def _handle_stderr(self):
        data = self.readAllStandardError().data().decode('utf-8', errors='ignore').strip()
        if data and self.log_signal:
            self.log_signal.emit(f"[SERVER][STDERR] {data}", 'error')
            
    def kill_server(self):
        if self.state() == QProcess.Running:
            if self.terminate():
                return True
            else:
                self.kill()
                return False

class ServerGUI(QMainWindow):
    log_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero Training Server Manager (PyQt5)")
        self.setGeometry(100, 100, 1000, 700)
        
        self.server_process = ServerProcess(self, self.log_signal)
        self.server_process.finished.connect(self.handle_process_finished)
        self.settings = QSettings("AlphaZeroAL", "ServerGUI")
        
        self.total_server_received_bytes = 0 
        self.total_server_sent_bytes = 0 
        self.traffic_log_pattern = re.compile(r"\[\[TRAFFIC_LOG::(RECEIVED|SENT)::\+::(\d+)\]\]") 
        
        # <<< ADDED: Runtime tracking variables and timer
        self.server_start_time = None
        self.runtime_timer = QTimer(self)
        self.runtime_timer.setInterval(1000) # 1 second
        self.runtime_timer.timeout.connect(self._update_runtime_display)
        # >>>

        self.traffic_timer = QTimer(self)
        self.traffic_timer.setInterval(500) 
        self.traffic_timer.timeout.connect(self._update_server_traffic_display) 
        
        self.server_process.started.connect(self.traffic_timer.start)
        self.server_process.finished.connect(self.traffic_timer.stop)
        
        self.log_signal.connect(self.append_log)
        
        self._init_default_args()
        self._setup_ui()
        self._load_settings()
        self.update_status_label()
        self._update_runtime_display() # Initialize the display

    def format_bytes(self, bytes_num):
        if bytes_num is None or bytes_num == 0:
            return "0 B"
        bytes_num = float(bytes_num)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_num < 1024.0:
                return f"{bytes_num:.2f} {unit}"
            bytes_num /= 1024.0
        return f"{bytes_num:.2f} PB"
        
    def _update_server_traffic_display(self):
        self.value_total_received.setText(self.format_bytes(self.total_server_received_bytes))
        self.value_total_sent.setText(self.format_bytes(self.total_server_sent_bytes))

    # <<< ADDED: Method to update run time display
    def _update_runtime_display(self):
        if self.server_process.state() == QProcess.Running and self.server_start_time is not None:
            elapsed_seconds = int(time.time() - self.server_start_time)
            # Format as HH:MM:SS
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime_str = f"Run Time: {hours:02d}:{minutes:02d}:{seconds:02d} ⏱️"
            self.runtime_label.setText(runtime_str)
        elif self.server_start_time is not None:
            # Show final duration when stopped
            elapsed_seconds = int(time.time() - self.server_start_time)
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime_str = f"Total Time: {hours:02d}:{minutes:02d}:{seconds:02d} (Stopped)"
            self.runtime_label.setText(runtime_str)
        else:
            self.runtime_label.setText("Run Time: N/A")
    # >>>
            
    def _init_default_args(self):
        # ... (unchanged) ...
        DEFAULT_DEVICE = 'cpu'
        try:
            if torch.cuda.is_available():
                DEFAULT_DEVICE = 'cuda'
        except Exception:
            pass

        self.params_groups = [
            ("🔌 1. Connection & Core Setup", {
                '--host': {'label': 'Host IP', 'type': QLineEdit, 'default': '0.0.0.0'},
                '--port': {'label': 'Port', 'type': QSpinBox, 'default': 7718, 'range': (1024, 65535)},
                '-d': {'label': 'Device (cuda/cpu)', 'type': QLineEdit, 'default': DEFAULT_DEVICE},
                '-e': {'label': 'Environment', 'type': QLineEdit, 'default': 'Connect4'},
                '-m': {'label': 'Model Type (CNN)', 'type': QLineEdit, 'default': 'CNN'},
                '--name': {'label': 'Pipeline Name', 'type': QLineEdit, 'default': 'AZ'},
            }),
            ("🧠 2. MCTS & Training Core", {
                '-n': {'label': 'MCTS Simulations/Action', 'type': QSpinBox, 'default': 100, 'range': (1, UNBOUNDED_INT)}, 
                '-b': {'label': 'Batch Size', 'type': QSpinBox, 'default': 512, 'range': (1, UNBOUNDED_INT)}, 
                '--buf': {'label': 'Buffer Size', 'type': QSpinBox, 'default': 100000, 'range': (1, UNBOUNDED_INT)}, 
                '--lr': {'label': 'Learning Rate', 'type': QDoubleSpinBox, 'default': 3e-3, 'range': (1e-6, 1e-1), 'decimals': 6, 'single_step': 1e-4},
                '-c': {'label': 'C_puct Init', 'type': QDoubleSpinBox, 'default': 1.25, 'range': (0.1, 10.0), 'decimals': 3, 'single_step': 0.01},
                '-a': {'label': 'Dirichlet Alpha', 'type': QDoubleSpinBox, 'default': 0.7, 'range': (0.0, 1.0), 'decimals': 3, 'single_step': 0.01},
                '-t': {'label': 'Softmax Temperature', 'type': QDoubleSpinBox, 'default': 1, 'range': (0.0, UNBOUNDED_INT), 'decimals': 3, 'single_step': 0.1}, 
                '--discount': {'label': 'Discount Factor', 'type': QDoubleSpinBox, 'default': 0.99, 'range': (0.0, 1.0), 'decimals': 3, 'single_step': 0.01},
            }),
            ("🔥 3. Evaluation & Flow Control", {
                '--n_play': {'label': 'Games per Update (n_play)', 'type': QSpinBox, 'default': 1, 'range': (1, UNBOUNDED_INT)},
                '--mcts_n': {'label': 'Pure MCTS Playouts (mcts_n)', 'type': QSpinBox, 'default': 1000, 'range': (1, UNBOUNDED_INT)}, 
                '--num_eval': {'label': 'Evaluation Games (num_eval)', 'type': QSpinBox, 'default': 50, 'range': (1, UNBOUNDED_INT)},
                '--interval': {'label': 'Eval Interval', 'type': QSpinBox, 'default': 10, 'range': (1, UNBOUNDED_INT)},
                '--thres': {'label': 'Win Rate Threshold', 'type': QDoubleSpinBox, 'default': 0.65, 'range': (0.5, 1.0), 'decimals': 3, 'single_step': 0.01},
                '--pause': {'label': 'Start Paused (Training)', 'type': QCheckBox, 'default': False, 'flag': True},
            }),
            ("💾 4. Caching Settings", {
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
        # ... (unchanged) ...
        control_box = QGroupBox("Control Panel")
        control_layout = QGridLayout() 
        
        self.start_button = QPushButton("🚀 Start Server")
        self.start_button.clicked.connect(self.start_server)
        control_layout.addWidget(self.start_button, 0, 0)
        
        self.stop_button = QPushButton("🛑 Stop Server")
        self.stop_button.clicked.connect(self.stop_server)
        control_layout.addWidget(self.stop_button, 0, 1)
        
        self.reset_button = QPushButton("🔄 Reset Parameters")
        self.reset_button.clicked.connect(self.reset_parameters)
        control_layout.addWidget(self.reset_button, 1, 0)
        
        self.reset_traffic_button = QPushButton("🗑️ Reset Traffic")
        self.reset_traffic_button.clicked.connect(self.reset_traffic_stats)
        control_layout.addWidget(self.reset_traffic_button, 1, 1)
        
        control_box.setLayout(control_layout)
        left_layout.addWidget(control_box)


        # 2. Status Box
        status_box = QGroupBox("Status & Network Traffic") 
        status_layout = QVBoxLayout() 
        
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        # <<< ADDED: Runtime Label
        self.runtime_label = QLabel("Run Time: N/A")
        status_layout.addWidget(self.runtime_label)
        # >>>
        
        traffic_group = QGroupBox("Server Traffic")
        traffic_layout = QGridLayout()
        
        self.label_total_received = QLabel("⬇️ Total Received (Client Upload):") 
        self.value_total_received = QLabel("0 B")
        self.value_total_received.setStyleSheet("font-weight: bold; color: blue;")
        
        self.label_total_sent = QLabel("⬆️ Total Sent (Client Download):") 
        self.value_total_sent = QLabel("0 B")
        self.value_total_sent.setStyleSheet("font-weight: bold; color: green;")
        
        traffic_layout.addWidget(self.label_total_received, 0, 0)
        traffic_layout.addWidget(self.value_total_received, 0, 1)
        traffic_layout.addWidget(self.label_total_sent, 1, 0)
        traffic_layout.addWidget(self.value_total_sent, 1, 1)
        
        traffic_group.setLayout(traffic_layout)
        status_layout.addWidget(traffic_group)
        
        status_box.setLayout(status_layout)
        left_layout.addWidget(status_box)

        # 3. Parameter Configuration
        # ... (unchanged) ...
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

                    if max_val == UNBOUNDED_INT:
                        label_text = f"{config['label']} (Min {min_str}):"
                    elif min_val != -UNBOUNDED_INT: 
                        label_text = f"{config['label']} ({min_str}-{max_str}):"

                label = QLabel(label_text)
                param_layout.addWidget(label, row, 0)
                
                widget = widget_type()

                if widget_type == QSpinBox:
                    min_val = config['range'][0]
                    max_val = config['range'][1]
                    if max_val == UNBOUNDED_INT:
                        max_val = int(1e9)
                    widget.setRange(min_val, max_val) 
                    widget.setValue(config['default'])
                    if config.get('single_step'):
                         widget.setSingleStep(config['single_step'])
                    elif key == '--cache_size':
                        widget.setSingleStep(100)

                elif widget_type == QDoubleSpinBox:
                    max_val = config['range'][1] if config['range'][1] < UNBOUNDED_INT else 1e9
                    widget.setRange(config['range'][0], max_val)
                    widget.setDecimals(config.get('decimals', 3))
                    widget.setValue(config['default'])
                    if config.get('single_step'):
                        widget.setSingleStep(config['single_step'])
                    elif key in ('-c', '-a'):
                        widget.setSingleStep(0.01)

                elif widget_type == QCheckBox:
                    widget.setChecked(config['default'])
                elif widget_type == QLineEdit:
                    widget.setText(config['default'])
                
                param_layout.addWidget(widget, row, 1)
                self.widgets[key] = widget
                
                if widget_type in (QSpinBox, QDoubleSpinBox):
                    widget.valueChanged.connect(lambda val, k=key: self._log_parameter_change(k, val))
                elif widget_type == QCheckBox:
                    widget.stateChanged.connect(lambda state, k=key: self._log_parameter_change(k, state))
                elif widget_type == QLineEdit:
                    widget.editingFinished.connect(lambda w=widget, k=key: self._log_parameter_change(k, w.text()))
                
                row += 1
        
        param_container.setLayout(param_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(param_container) 
        
        param_group_box = QGroupBox("Server Parameter Configuration")
        param_group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        param_group_box_layout = QVBoxLayout(param_group_box)
        param_group_box_layout.addWidget(scroll_area)
        
        left_layout.addWidget(param_group_box)
        
        # ... (unchanged) ...
        self.cache_size_spinbox = self.widgets['--cache_size']
        self.no_cache_checkbox = self.widgets['--no-cache']
        self.cache_size_spinbox.valueChanged.connect(self._sync_size_to_check)
        self.no_cache_checkbox.stateChanged.connect(self._sync_check_to_size)
        if self.cache_size_spinbox.value() == 0:
             self.no_cache_checkbox.setChecked(True)

        right_layout = QVBoxLayout()
        log_box = QGroupBox("Server Log Output")
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

    def reset_traffic_stats(self):
        # ... (unchanged) ...
        if self.server_process.state() != QProcess.Running:
            QMessageBox.warning(self, "Warning", "Server is not running. Cannot reset traffic stats.")
            self.append_log("Reset Traffic failed: Server is not running.", 'error')
            return
        
        self.total_server_received_bytes = 0
        self.total_server_sent_bytes = 0
        self._update_server_traffic_display() 
        
        try:
            host = self.widgets['--host'].text()
            port = self.widgets['--port'].value()
            url = f'http://{host}:{port}/reset_traffic'
            self.append_log(f"Attempting to reset server traffic stats via POST to {url}...", 'info')
            
            r = requests.post(url, timeout=2)
            
            if r.status_code == 200 and r.json().get('status') == 'success':
                self.append_log("Server traffic statistics successfully reset to 0 (GUI and Backend).", 'success')
            else:
                self.append_log(f"Server traffic statistics reset in GUI. WARNING: Backend reset via HTTP failed. Status: {r.status_code}.", 'warning')
                
        except requests.exceptions.ConnectionError:
            self.append_log("Server traffic statistics reset in GUI. WARNING: Could not connect to the server for backend reset.", 'warning')
        except Exception as e:
            self.append_log(f"An unexpected error occurred during traffic reset: {e}", 'error')

    def _fetch_server_traffic_stats(self):
        pass

    def _log_parameter_change(self, key, value):
        # ... (unchanged) ...
        widget = self.widgets[key]
        display_value = str(value)
        if isinstance(widget, QCheckBox):
            display_value = "Enabled" if value == Qt.Checked else "Disabled"
        elif isinstance(widget, QDoubleSpinBox):
            display_value = f"{value:.4g}"
        label = self.params[key]['label']
        self.append_log(f"[PARAM CHANGE] '{label}' ({key}) set to: {display_value}", 'info')

    def _sync_size_to_check(self, value):
        # ... (unchanged) ...
        checkbox = self.no_cache_checkbox
        checkbox.blockSignals(True)
        if value == 0:
            if not checkbox.isChecked(): checkbox.setChecked(True)
        elif value > 0:
            if checkbox.isChecked(): checkbox.setChecked(False)
        checkbox.blockSignals(False)

    def _sync_check_to_size(self, state):
        # ... (unchanged) ...
        spinbox = self.cache_size_spinbox
        spinbox.blockSignals(True)
        if state == Qt.Checked: spinbox.setValue(0)
        elif state == Qt.Unchecked:
            if spinbox.value() == 0: spinbox.setValue(self.cache_default_size)
        spinbox.blockSignals(False)
    
    def clear_log(self):
        # ... (unchanged) ...
        self.log_text_edit.clear()
        self.append_log("--- Log manually cleared ---", 'info')

    def _load_settings(self):
        # ... (unchanged) ...
        for key, config in self.params.items():
            widget = self.widgets[key]
            widget.blockSignals(True)
            value = self.settings.value(key, config.get('default'))
            if value is not None:
                if isinstance(widget, QSpinBox):
                    try:
                        widget.setValue(int(float(value)))
                    except (ValueError, TypeError):
                        widget.setValue(config['default'])
                elif isinstance(widget, QDoubleSpinBox):
                    try:
                        widget.setValue(float(value))
                    except (ValueError, TypeError):
                        widget.setValue(config['default']) 
                elif isinstance(widget, QCheckBox):
                    if isinstance(value, str): widget.setChecked(value.lower() == 'true')
                    else: widget.setChecked(bool(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
            widget.blockSignals(False)
            
    def _save_settings(self):
        # ... (unchanged) ...
        for key, config in self.params.items():
            widget = self.widgets[key]
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                self.settings.setValue(key, widget.value())
            elif isinstance(widget, QCheckBox):
                self.settings.setValue(key, widget.isChecked())
            elif isinstance(widget, QLineEdit):
                self.settings.setValue(key, widget.text())

    def get_server_args(self):
        # ... (unchanged) ...
        args = []
        for key, config in self.params.items():
            widget = self.widgets[key]
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                value = widget.value()
                if key == '--lr': args.extend([key, f"{value:.6g}"]) 
                else: args.extend([key, str(value)])
            elif isinstance(widget, QLineEdit):
                value = widget.text().strip()
                if value: args.extend([key, value])
            elif isinstance(widget, QCheckBox):
                if config.get('flag') and widget.isChecked():
                    args.append(key)
        return args

    def start_server(self):
        if self.server_process.state() == QProcess.Running:
            QMessageBox.warning(self, "Warning", "Server is already running.")
            return
        command_list = self.get_server_args()
        python_executable = sys.executable
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", os.getcwd())
        env.insert("PYTHONIOENCODING", "utf-8")
        self.server_process.setProcessEnvironment(env)
        self.append_log(f"Starting server: {python_executable} {' '.join(['-u', 'server.py'] + command_list)}", 'info')
        # 使用 -u 确保 STDOUT 不被缓存，及时输出
        self.server_process.start(python_executable, ['-u', 'server.py'] + command_list)
        if not self.server_process.waitForStarted(2000): 
            self.append_log(f"Server failed to start: {self.server_process.errorString()}", 'error')
        else:
             self.traffic_timer.start()
             # <<< MODIFIED: Record start time and start runtime timer
             self.server_start_time = time.time() 
             self.runtime_timer.start()
             # >>>
        self.update_status_label()

    def stop_server(self):
        if self.server_process.state() == QProcess.Running:
            self.append_log("Stopping server...", 'info')
            self.server_process.kill_server()
            self.server_process.waitForFinished(3000)
        self.update_status_label()

    def handle_process_finished(self, exit_code, exit_status):
        self.append_log(f"Server finished with code {exit_code}.", 'warning')
        self.traffic_timer.stop()
        # <<< MODIFIED: Stop runtime timer and update display
        self.runtime_timer.stop()
        self._update_runtime_display() 
        # >>>
        self.update_status_label()

    def update_status_label(self):
        # ... (unchanged) ...
        state = self.server_process.state()
        if state == QProcess.Running:
            self.status_label.setText("Status: RUNNING 🟢")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.reset_button.setEnabled(False)
            self.reset_traffic_button.setEnabled(True) 
            for k, w in self.widgets.items(): w.setEnabled(False)
        else:
            self.status_label.setText("Status: STOPPED 🔴")
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.reset_button.setEnabled(True)
            self.reset_traffic_button.setEnabled(False) 
            self.traffic_timer.stop()
            for k, w in self.widgets.items(): w.setEnabled(True)
            self._update_runtime_display() # Ensure status is updated when stopped
            
    # --- MODIFIED: append_log 确保非流量日志能够显示 ---
    def append_log(self, text, level='normal'):
        match = self.traffic_log_pattern.search(text)
        
        if match:
            direction = match.group(1)
            try:
                bytes_added = int(match.group(2))
                
                if direction == 'RECEIVED': 
                    self.total_server_received_bytes += bytes_added
                    return 
                elif direction == 'SENT': 
                    self.total_server_sent_bytes += bytes_added
                    return 
                
            except ValueError: 
                text = f"[TRAFFIC PARSING ERROR] Malformed traffic log found: {text}"
                level = 'error'
        
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
        # ... (unchanged) ...
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
        # ... (unchanged) ...
        if self.server_process.state() == QProcess.Running:
            reply = QMessageBox.question(self, 'Confirm Exit', "Server is running. Stop before exit?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_server()
                QApplication.processEvents()
            elif reply == QMessageBox.No: 
                self.stop_server()
            else:
                event.ignore()
                return
        self._save_settings()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ServerGUI()
    gui.show()
    sys.exit(app.exec_())