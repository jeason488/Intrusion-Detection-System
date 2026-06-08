import subprocess
import os

os.chdir(r"d:\Desktop\项目\PyCharm\me\network_security_system")

cmd = [
    "pyinstaller",
    "--onefile",
    "--windowed",
    "--exclude-module=tkinter",
    "--exclude-module=matplotlib",
    "--exclude-module=sklearn",
    "--exclude-module=skimage",
    "desktop_app.py"
]

print("Starting packaging...")
print("Command:", " ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print("\nSTDOUT:")
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("\nSTDERR:")
print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
print("\nReturn code:", result.returncode)