import os
import win32com.client
import pythoncom
def create_desktop_shortcut():
    pythoncom.CoInitialize()
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    shortcut_path = os.path.join(desktop_path, "基于Prototype的网络入侵检测系统.lnk")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_path = "C:\\ProgramData\\anaconda3\\pythonw.exe"
    script_path = os.path.join(current_dir, "desktop_app.py")
    icon_path = os.path.join(current_dir, "frontend", "icon.ico")
    if os.path.exists(shortcut_path):
        os.remove(shortcut_path)
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.TargetPath = python_path
    shortcut.Arguments = f'"{script_path}"'
    shortcut.WorkingDirectory = current_dir
    if os.path.exists(icon_path):
        shortcut.IconLocation = f"{icon_path},0"
    shortcut.Description = "基于Prototype的网络入侵检测系统"
    shortcut.WindowStyle = 1
    shortcut.Save()
    print(f"快捷方式已创建: {shortcut_path}")
    print(f"目标: {python_path}")
    print(f"参数: {script_path}")
    print(f"工作目录: {current_dir}")
if __name__ == "__main__":
    create_desktop_shortcut()
    print("快捷方式创建完成！")
