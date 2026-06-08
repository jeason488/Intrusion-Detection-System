import os
from win32com.client import Dispatch

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
path = os.path.join(desktop, "基于Prototype的网络入侵检测系统.lnk")
target = r"D:\Desktop\项目\PyCharm\me\network_security_system\启动网络入侵检测系统.bat"
wDir = r"D:\Desktop\项目\PyCharm\me\network_security_system"
icon = r"D:\Desktop\项目\PyCharm\me\network_security_system\frontend\icon.ico"

shell = Dispatch('WScript.Shell')
shortcut = shell.CreateShortCut(path)
shortcut.TargetPath = target
shortcut.WorkingDirectory = wDir
shortcut.IconLocation = icon
shortcut.save()

print("快捷方式已更新！")