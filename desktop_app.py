import sys
import os
import time
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QSystemTrayIcon, QMenu,
    QAction, QMessageBox, QDialog, QVBoxLayout, QProgressBar, QLabel
)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
class LoadingDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于Prototype的网络入侵检测系统")
        self.setFixedSize(400, 150)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setStyleSheet("QDialog { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; } QLabel { color: #ffffff; font-size: 14px; font-weight: 500; } QProgressBar { border: none; border-radius: 6px; background-color: rgba(255, 255, 255, 0.1); height: 8px; } QProgressBar::chunk { background: linear-gradient(90deg, #00d4ff, #7b2cbf); border-radius: 6px; }")
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        title_label = QLabel("基于Prototype的网络入侵检测系统")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("正在启动服务...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        self.setLayout(layout)
        self.setWindowIcon(QIcon(os.path.join(BASE_DIR, "frontend", "icon.ico")))
        self.centerOnScreen()
    def centerOnScreen(self):
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    def updateProgress(self, value, status_text):
        self.progress.setValue(value)
        self.status_label.setText(status_text)
        QApplication.processEvents()
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.backend_thread = None
        self.initUI()
        self.initTrayIcon()
    def initUI(self):
        self.setWindowTitle("基于Prototype的网络入侵检测系统")
        self.setGeometry(100, 100, 1200, 800)
        self.web_view = QWebEngineView()
        settings = self.web_view.settings()
        settings.setAttribute(settings.JavascriptEnabled, True)
        settings.setAttribute(settings.PluginsEnabled, True)
        settings.setAttribute(settings.LocalStorageEnabled, True)
        settings.setAttribute(settings.JavascriptCanAccessClipboard, True)
        settings.setAttribute(settings.AllowRunningInsecureContent, True)
        settings.setAttribute(settings.AllowGeolocationOnInsecureOrigins, True)
        self.setCentralWidget(self.web_view)
        self.showMaximized()
    def initTrayIcon(self):
        self.tray_icon = QSystemTrayIcon(self)
        icon_path = os.path.join(BASE_DIR, "frontend", "icon.ico")
        if os.path.exists(icon_path):
            self.tray_icon.setIcon(QIcon(icon_path))
            self.setWindowIcon(QIcon(icon_path))
        else:
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.blue)
            self.tray_icon.setIcon(QIcon(pixmap))
            self.setWindowIcon(QIcon(pixmap))
        tray_menu = QMenu()
        show_action = QAction("显示窗口", self)
        show_action.triggered.connect(self.showNormal)
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.quitApplication)
        tray_menu.addAction(show_action)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.onTrayActivated)
        self.tray_icon.show()
    def onTrayActivated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self.showNormal()
    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "基于Prototype的网络入侵检测系统",
            "系统已最小化到托盘",
            QSystemTrayIcon.Information,
            2000
        )
    def quitApplication(self):
        self.tray_icon.hide()
        if self.backend_thread and self.backend_thread.is_alive():
            pass
        QApplication.quit()
    def startBackend(self):
        def run_backend():
            try:
                from api.server import start_server
                from config.config import load_config
                config = load_config("config/config.yaml")
                start_server(config)
            except Exception as e:
                print(f"Backend error: {e}")
        self.backend_thread = threading.Thread(target=run_backend, daemon=True)
        self.backend_thread.start()
    def loadWebPage(self):
        import requests
        for i in range(30):
            try:
                response = requests.get('http://localhost:8000')
                if response.status_code == 200:
                    print("Backend is ready, loading web page...")
                    self.web_view.load(QUrl("http://localhost:8000"))
                    return True
            except Exception as e:
                print(f"Waiting for backend... ({i+1}/30)")
            time.sleep(1)
        return False
    def loadWebPageDirect(self):
        self.web_view.load(QUrl("http://localhost:8000"))
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("基于Prototype的网络入侵检测系统")
    app.setQuitOnLastWindowClosed(False)
    loading_dialog = LoadingDialog()
    loading_dialog.show()
    loading_dialog.updateProgress(10, "正在初始化应用...")
    window = MainWindow()
    loading_dialog.updateProgress(20, "正在启动后端服务...")
    window.startBackend()
    loading_dialog.updateProgress(30, "等待服务就绪...")
    import requests
    success = False
    for i in range(30):
        try:
            response = requests.get('http://localhost:8000')
            if response.status_code == 200:
                loading_dialog.updateProgress(90, "服务已就绪，加载页面...")
                window.loadWebPageDirect()
                success = True
                break
        except:
            pass
        progress = 30 + int(i * 60 / 30)
        loading_dialog.updateProgress(min(progress, 85), f"正在连接服务... ({i+1}/30)")
        time.sleep(1)
    if success:
        loading_dialog.updateProgress(100, "启动完成")
        time.sleep(0.5)
        loading_dialog.close()
        window.show()
    else:
        loading_dialog.close()
        QMessageBox.critical(None, "启动失败", "无法连接到后端服务，请检查端口是否被占用")
        sys.exit(1)
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()
