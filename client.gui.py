import sys
import os
import signal
import torch 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy,
    QMessageBox, QScrollArea, QFrame 
)
from PyQt5.QtCore import QProcess, QSettings, Qt, pyqtSignal

# 使用一个极大的数来标记参数在实际应用中没有上限
UNBOUNDED_INT = 2000000000

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
            pass
        # --- Determine Default Device END ---

        # 优化后的参数分组
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
        
        self.start_button = QPushButton("🚀 Start All Clients")
        self.start_button.clicked.connect(self.start_clients)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("🛑 Stop All Clients")
        self.stop_button.clicked.connect(self.stop_all_clients)
        control_layout.addWidget(self.stop_button)
        
        # --- Reset Button ---
        self.reset_button = QPushButton("🔄 Reset Parameters")
        self.reset_button.clicked.connect(self.reset_parameters)
        control_layout.addWidget(self.reset_button)
        # -------------------------
        
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
                    
                    if key in ('--c_init', '--alpha', '--tempD'):
                        widget.setSingleStep(0.01)
                    elif key == '--temp':
                        widget.setSingleStep(0.1)

                elif widget_type == QCheckBox:
                    widget.setChecked(config['default'])
                elif widget_type == QLineEdit:
                    widget.setText(config['default'])
                # --- 参数设置逻辑结束 ---
                
                param_layout.addWidget(widget, row, 1)
                self.widgets[key] = widget
                
                # --- Log signal connections ---
                if widget_type in (QSpinBox, QDoubleSpinBox):
                    # SpinBoxes pass the new value
                    widget.valueChanged.connect(lambda val, k=key: self._log_parameter_change(k, val))
                elif widget_type == QCheckBox:
                    # CheckBoxes pass the new state
                    widget.stateChanged.connect(lambda state, k=key: self._log_parameter_change(k, state))
                elif widget_type == QLineEdit:
                    # LineEdits should log on focus loss (editingFinished)
                    widget.editingFinished.connect(lambda w=widget, k=key: self._log_parameter_change(k, w.text()))
                # --- Log signal connections END ---
                
                row += 1
            
        param_container.setLayout(param_layout) 
        
        # 3b. 将参数容器包裹在 QScrollArea 中
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) 
        scroll_area.setWidget(param_container) 

        # 3c. 将 ScrollArea 放入 QGroupBox (使其具有边框和标题)
        param_group_box = QGroupBox("Client Parameter Configuration")
        # 确保 QGroupBox 占据剩余的垂直空间
        param_group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        param_group_box_layout = QVBoxLayout(param_group_box)
        param_group_box_layout.addWidget(scroll_area)
        
        # 将参数区添加到左侧布局
        left_layout.addWidget(param_group_box)

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
        """Logs the change of a parameter."""
        widget = self.widgets[key]
        
        display_value = str(value)
        
        if isinstance(widget, QCheckBox):
            # QCheckBox passes state (2 or 0)
            display_value = "Enabled" if value == Qt.Checked else "Disabled"
            
        elif isinstance(widget, QDoubleSpinBox):
            # QDoubleSpinBox passes float value
            display_value = f"{value:.4g}"
        
        # For QLineEdit, 'value' is the text (string)
        # For QSpinBox, 'value' is the int
        
        label = self.params[key]['label']
        # The synchronization methods block signals, so the automatic sync changes will not trigger this log.
        self.append_log(f"[PARAM CHANGE] '{label}' ({key}) set to: {display_value}", 'info')
    
    # --- 重置参数方法 ---
    def reset_parameters(self):
        """Resets all parameter widgets to their default values."""
        
        reply = QMessageBox.question(self, 'Confirm Reset',
            "Are you sure you want to reset all parameters to their default values? Unsaved changes will be lost.", 
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.No:
            return
            
        # 临时阻断信号，防止同步逻辑在重置过程中多次触发
        self.cache_size_spinbox.blockSignals(True)
        self.no_cache_checkbox.blockSignals(True)
        
        # 记录重置操作
        self.append_log("--- Starting parameter reset to defaults ---", 'warning')
        
        for key, config in self.params.items():
            widget = self.widgets[key]
            default_value = config['default']
            
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                if widget.value() != default_value:
                    widget.setValue(default_value)
            elif isinstance(widget, QCheckBox):
                if widget.isChecked() != default_value:
                    widget.setChecked(default_value)
            elif isinstance(widget, QLineEdit):
                if widget.text() != default_value:
                    widget.setText(default_value)

        # 重新启用信号
        self.cache_size_spinbox.blockSignals(False)
        self.no_cache_checkbox.blockSignals(False)
        
        # 确保缓存复选框与默认的缓存大小保持同步
        if self.cache_default_size == 0:
            self.no_cache_checkbox.setChecked(True)
        else:
            self.no_cache_checkbox.setChecked(False) 
        
        self.append_log("All parameters have been reset to default values.", 'info')

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
            # 临时阻断信号，防止加载设置时触发日志
            widget.blockSignals(True) 
            
            value = self.settings.value(key, config.get('default'))
            if value is not None:
                if isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QDoubleSpinBox):
                    try:
                        widget.setValue(float(value))
                    except ValueError:
                        widget.setValue(config['default']) 
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(value == 'true' or value is True)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                    
            widget.blockSignals(False) # 重新启用信号
            
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
        self.reset_button.setEnabled(not is_running) # 禁用重置按钮

        for key in self.widgets:
            self.widgets[key].setEnabled(not is_running)

    def start_clients(self):
        """Starts all client processes."""
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
            instance_id = i
            process = ClientProcess(instance_id, self, self.log_signal, self.process_finished_signal)
            
            # 确保加上 '-u' 标志，强制非缓冲输出
            command_args = ['-u', 'client.py'] + args 
            
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
        
        # 1. 检查当前是否处于深色模式 (通过检查背景色亮度)
        bg_color = self.log_text_edit.palette().color(self.log_text_edit.backgroundRole())
        # 简单的亮度检查：R+G+B < 382.5 判定为深色背景
        is_dark_mode = (bg_color.red() + bg_color.green() + bg_color.blue()) < 382
        
        # 2. 定义深色和浅色模式下的颜色映射
        if is_dark_mode:
            # 深色模式下的颜色 (背景暗，字体亮)
            color_map = {
                'normal': 'white',      # 正常输出 (浅色/白色)
                'error': '#FF6B6B',     # 错误 (亮红)
                'warning': '#FFCC66',   # 警告 (亮橙/黄)
                'success': '#8BC34A',   # 成功 (亮绿)
                'info': '#66A5FF'       # 信息 (亮蓝)
            }
            fallback_color = 'white'
        else:
            # 浅色模式下的颜色 (背景亮，字体暗)
            color_map = {
                'normal': 'black',
                'error': 'red',
                'warning': 'orange',
                'success': 'green',
                'info': 'blue'
            }
            fallback_color = 'black'
            
        # 3. 应用颜色
        color = color_map.get(level, fallback_color)
        
        html = f'<span style="color:{color};">{text}</span><br>'
        self.log_text_edit.insertHtml(html)
        self.log_text_edit.ensureCursorVisible()

    def closeEvent(self, event):
        """Stops all processes and saves settings before closing the window."""
        if any(p.state() == QProcess.Running for p in self.processes.values()):
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