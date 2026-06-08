import os
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QPainter, QColor
def create_icon(output_path):
    app = QApplication(sys.argv)
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor(255, 255, 255))
    painter = QPainter(pixmap)
    painter.setBrush(QColor(26, 115, 232))
    painter.setPen(QColor(26, 115, 232))
    painter.drawEllipse(4, 4, 24, 24)
    painter.setBrush(QColor(52, 168, 83))
    painter.setPen(QColor(52, 168, 83))
    painter.drawEllipse(10, 10, 12, 12)
    painter.setBrush(QColor(255, 255, 255))
    painter.setPen(QColor(255, 255, 255))
    painter.drawEllipse(14, 14, 4, 4)
    painter.end()
    pixmap.save(output_path, 'ICO')
    print(f"图标已创建: {output_path}")
if __name__ == "__main__":
    icon_path = os.path.join(os.path.dirname(__file__), "frontend", "icon.ico")
    create_icon(icon_path)
