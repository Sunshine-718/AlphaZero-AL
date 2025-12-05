import sys
import os
import torch
import signal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy,
    QMessageBox, QFrame, QScrollArea # 导入 QScrollArea
)
from PyQt5.QtCore import QProcess, QSettings, Qt, pyqtSignal, QProcessEnvironment

# 使用一个极大的数来标记参数在实际应用中没有上限
UNBOUNDED_INT = 2000000000

# --- Server Process Management ---

class ServerProcess(QProcess):
    """
    A simple QProcess wrapper for the single server instance.
    """
    log_signal = pyqtSignal(str, str)
    
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
        # Read and decode all standard error data
        data = self.readAllStandardError().data().decode('utf-8', errors='ignore').strip()
        if data and self.log_signal:
            self.log_signal.emit(f"[SERVER][STDERR] {data}", 'error')
            
    def kill_server(self):
        """Attempts to terminate the server process."""
        if self.state() == QProcess.Running:
            if self.terminate():
                return True
            else:
                # Force kill if termination fails
                self.kill()
                return False

# --- GUI Application ---

class ServerGUI(QMainWindow):
    # Signal for logging from the QProcess thread to the main thread
    log_signal = pyqtSignal(str, str) # text, level

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero Training Server Manager")
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
            pass
        # --- Determine Default Device END ---

        # 优化后的参数分组
        self.params_groups = [
            ("🔌 1. 连接 & 核心配置 (Connection & Core Setup)", {
                '--host': {'label': 'Host IP', 'type': QLineEdit, 'default': '0.0.0.0'},
                '--port': {'label': 'Port', 'type': QSpinBox, 'default': 7718, 'range': (1024, 65535)},
                '-d': {'label': 'Device (cuda/cpu)', 'type': QLineEdit, 'default': DEFAULT_DEVICE},
                '-e': {'label': 'Environment', 'type': QLineEdit, 'default': 'Connect4'},
                '-m': {'label': 'Model Type (CNN)', 'type': QLineEdit, 'default': 'CNN'},
                '--name': {'label': 'Pipeline Name', 'type': QLineEdit, 'default': 'AZ'},
            }),
            ("🧠 2. MCTS & 训练核心参数 (MCTS & Training Core)", {
                '-n': {'label': 'MCTS Simulations/Action', 'type': QSpinBox, 'default': 100, 'range': (1, UNBOUNDED_INT)}, 
                '-b': {'label': 'Batch Size', 'type': QSpinBox, 'default': 512, 'range': (1, UNBOUNDED_INT)}, 
                '--buf': {'label': 'Buffer Size', 'type': QSpinBox, 'default': 5000, 'range': (1, UNBOUNDED_INT)}, 
                '--lr': {'label': 'Learning Rate', 'type': QDoubleSpinBox, 'default': 1e-3, 'range': (1e-6, 1e-1), 'decimals': 6, 'single_step': 1e-4},
                '-c': {'label': 'C_puct Init', 'type': QDoubleSpinBox, 'default': 1.25, 'range': (0.1, 10.0), 'decimals': 3, 'single_step': 0.01},
                '-a': {'label': 'Dirichlet Alpha', 'type': QDoubleSpinBox, 'default': 0.7, 'range': (0.0, 1.0), 'decimals': 3, 'single_step': 0.01},
                '-t': {'label': 'Softmax Temperature', 'type': QDoubleSpinBox, 'default': 1, 'range': (0.0, UNBOUNDED_INT), 'decimals': 3, 'single_step': 0.1}, 
                '--discount': {'label': 'Discount Factor', 'type': QDoubleSpinBox, 'default': 0.99, 'range': (0.0, 1.0), 'decimals': 3, 'single_step': 0.01},
            }),
            ("🔥 3. 评估 & 流程控制 (Evaluation & Flow Control)", {
                '--n_play': {'label': 'Games per Update (n_play)', 'type': QSpinBox, 'default': 1, 'range': (1, UNBOUNDED_INT)},
                '--mcts_n': {'label': 'Pure MCTS Playouts (mcts_n)', 'type': QSpinBox, 'default': 1000, 'range': (1, UNBOUNDED_INT)}, 
                '--num_eval': {'label': 'Evaluation Games (num_eval)', 'type': QSpinBox, 'default': 50, 'range': (1, UNBOUNDED_INT)},
                '--interval': {'label': 'Eval Interval', 'type': QSpinBox, 'default': 10, 'range': (1, UNBOUNDED_INT)},
                '--thres': {'label': 'Win Rate Threshold', 'type': QDoubleSpinBox, 'default': 0.65, 'range': (0.5, 1.0), 'decimals': 3, 'single_step': 0.01},
                '--pause': {'label': 'Start Paused (Training)', 'type': QCheckBox, 'default': False, 'flag': True},
            }),
            ("💾 4. 缓存设置 (Caching Settings)", {
                '--cache_size': {'label': 'Transposition Table Size', 'type': QSpinBox, 'default': 5000, 'range': (0, UNBOUNDED_INT)},
                '--no-cache': {'label': 'Disable Transposition Table', 'type': QCheckBox, 'default': False, 'flag': True},
            })
        ]
        
        # 将分组参数扁平化，以方便其他方法访问
        self.params = {k: v for _, group in self.params_groups for k, v in group.items()}
        self.widgets = {}
        # 存储置换表默认大小，用于同步时恢复
        self.cache_default_size = self.params['--cache_size']['default'] 

    def _setup_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        # --- Left Side: Controls and Parameters ---
        left_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setMinimumWidth(350) # 允许水平调整大小

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

        # 3. Parameter Configuration (Wrapped in ScrollArea)
        
        # 3a. 创建参数容器 (用于放置在 QScrollArea 中)
        param_container = QWidget()
        param_layout = QGridLayout(param_container)
        
        row = 0
        for group_title, params_dict in self.params_groups:
            # 添加分组标题和分隔线
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
                
                # --- 智能显示范围提示 ---
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

                # --- 参数设置逻辑 ---
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
                # --- 参数设置逻辑结束 ---
                
                param_layout.addWidget(widget, row, 1)
                self.widgets[key] = widget
                row += 1
        
        param_container.setLayout(param_layout)
        
        # 3b. 将参数容器包裹在 QScrollArea 中
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(param_container) 
        
        # 3c. 将 ScrollArea 放入 QGroupBox
        param_group_box = QGroupBox("Server Parameter Configuration")
        # 确保 QGroupBox 占据剩余的垂直空间
        param_group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        param_group_box_layout = QVBoxLayout(param_group_box)
        param_group_box_layout.addWidget(scroll_area)
        
        # 将参数区添加到左侧布局
        left_layout.addWidget(param_group_box)
        # 移除 left_layout.addStretch(1)，确保参数组框填满剩余空间

        # --- 置换表同步逻辑初始化 ---
        self.cache_size_spinbox = self.widgets['--cache_size']
        self.no_cache_checkbox = self.widgets['--no-cache']

        self.cache_size_spinbox.valueChanged.connect(self._sync_size_to_check)
        self.no_cache_checkbox.stateChanged.connect(self._sync_check_to_size)
        
        if self.cache_size_spinbox.value() == 0:
             self.no_cache_checkbox.setChecked(True)
        # --- 置换表同步逻辑初始化结束 ---


        # --- Right Side: Log Area ---
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

    # --- 同步方法 ---
    def _sync_size_to_check(self, value):
        """将置换表大小同步到 'Disable Cache' 复选框。"""
        checkbox = self.no_cache_checkbox
        
        checkbox.blockSignals(True)
        
        if value == 0:
            if not checkbox.isChecked():
                checkbox.setChecked(True)
        elif value > 0:
            if checkbox.isChecked():
                checkbox.setChecked(False)
        
        checkbox.blockSignals(False)


    def _sync_check_to_size(self, state):
        """将 'Disable Cache' 复选框状态同步到置换表大小。"""
        spinbox = self.cache_size_spinbox
        
        spinbox.blockSignals(True)
        
        if state == Qt.Checked:
            spinbox.setValue(0)
            
        elif state == Qt.Unchecked:
            if spinbox.value() == 0:
                spinbox.setValue(self.cache_default_size)
                
        spinbox.blockSignals(False)
    # --- 同步方法结束 ---
    
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
                    # QSettings saves floats as strings, need conversion
                    try:
                        widget.setValue(float(value))
                    except ValueError:
                        widget.setValue(config['default']) # Fallback
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
                    args.extend([key, f"{value:.6g}"]) 
                else:
                    args.extend([key, str(value)])
            elif isinstance(widget, QLineEdit):
                value = widget.text().strip()
                if value:
                    args.extend([key, value])
            elif isinstance(widget, QCheckBox):
                # The server code uses '--no-cache' for *disabling* the cache (action='store_false', dest='cache')
                # The GUI should pass the flag *only* if the checkbox is checked.
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
        
        # 确保 server.py 存在
        if not os.path.exists("server.py"):
             self.append_log("Error: server.py not found. Ensure it is in the same directory.", 'error')
             return

        python_executable = sys.executable 

        self.append_log(f"--- Starting Server ---", 'info')
        
        # Use '-u' flag to force unbuffered output for real-time logging
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
    try:
        # 尝试导入 PyTorch 以检查设备可用性
        if 'torch' not in sys.modules:
            import torch
    except ImportError:
        pass 

    app = QApplication(sys.argv)
    gui = ServerGUI()
    gui.show()
    sys.exit(app.exec_())