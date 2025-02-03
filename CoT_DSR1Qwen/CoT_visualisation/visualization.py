from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem
from PyQt5.QtGui import QPen, QPainter, QFontMetrics, QFont
from PyQt5.QtCore import Qt
import sys

class vector2D:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class FlowchartView(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)  # Enable smooth rendering

        self.setSceneRect(0,0,600,400)  # Set canvas size
        self.showMaximized()
        self.windowWidth = self.size().width()
        self.windowHeight = self.size().height()

        self.setDragMode(QGraphicsView.ScrollHandDrag)  # Enable panning
        
        self.nodes = []  # Store nodes for correct ordering
        self.lines = []  # Store lines

    def convertToScreenSpace(self, x, y):
        self.windowWidth = self.size().width()
        self.windowHeight = self.size().height()
        return vector2D((x + 1) * self.windowWidth/2, (y + 1) * self.windowHeight/2)

    def addNode(self, pos, text, width=120, padding=10, fontSize=9):
        font = QFont("Arial", fontSize)

        # Create a QGraphicsTextItem and set its width for proper wrapping
        label = QGraphicsTextItem(text)
        label.setFont(font)
        label.setTextWidth(width - 2 * padding)  # Ensure wrapping
        label.setDefaultTextColor(Qt.black)

        # Use boundingRect() to get the real wrapped text height
        text_rect = label.boundingRect()
        height = text_rect.height() + 2 * padding

        # Create rectangle that perfectly fits the text
        rect = QGraphicsRectItem(pos.x-width/2, pos.y-height/2, width, height)
        rect.setBrush(Qt.white)
        rect.setPen(QPen(Qt.black, 2))
        rect.setZValue(1)  # Ensure rectangles are above lines
        self.scene.addItem(rect)

        # Position text inside rectangle
        label.setParentItem(rect)
        label.setPos(pos.x- width/2+padding, pos.y- height/2+padding)  # Position inside rectangle

        return rect

    def addConnection(self, node1, node2):
        # Get center positions of nodes
        center1 = node1.sceneBoundingRect().center()
        center2 = node2.sceneBoundingRect().center()

        x1, y1 = center1.x(), center1.y()
        x2, y2 = center2.x(), center2.y()

        # Draw line between nodes
        line = QGraphicsLineItem(x1, y1, x2, y2)
        line.setPen(QPen(Qt.black, 2))
        line.setZValue(0)  # Ensure lines are drawn below rectangles
        self.scene.addItem(line)
        self.lines.append(line)  # Store line