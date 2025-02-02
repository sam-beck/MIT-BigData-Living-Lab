from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem
from PyQt5.QtGui import QPen, QPainter
from PyQt5.QtCore import Qt
import sys

class FlowchartView(QGraphicsView):
    """ A QGraphicsView to display a flowchart or tree diagram """

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)  # Enable smooth rendering

        self.setSceneRect(0, 0, 800, 600)  # Set canvas size
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # Enable panning
        self.scale(1.2, 1.2)  # Initial zoom

        self.nodes = []  # Store nodes for correct ordering
        self.lines = []  # Store lines

    def add_node(self, x, y, text):
        """ Add a node (rectangle with text) at (x, y) """
        rect = QGraphicsRectItem(x, y, 100, 50)
        rect.setBrush(Qt.white)  # Fill color
        rect.setPen(QPen(Qt.black, 2))
        rect.setZValue(1)  # Ensure rectangles are above lines
        self.scene.addItem(rect)

        label = QGraphicsTextItem(text, parent=rect)
        label.setDefaultTextColor(Qt.black)
        label.setPos(x + 10, y + 15)  # Center text
        self.scene.addItem(label)

        self.nodes.append(rect)  # Store node
        return rect

    def add_connection(self, node1, node2):
        """ Connect two nodes with a line """

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

# Create Application
app = QApplication(sys.argv)
view = FlowchartView()
view.setWindowTitle("PyQt5 Flowchart Generator")
view.resize(800, 600)
view.show()

# Example Flowchart Nodes (Added First)
nodeA = view.add_node(300, 100, "Start")
nodeB = view.add_node(200, 250, "Process 1")
nodeC = view.add_node(400, 250, "Process 2")
nodeD = view.add_node(300, 400, "End")

# Connect Nodes (Added After Nodes)
view.add_connection(nodeA, nodeB)
view.add_connection(nodeA, nodeC)
view.add_connection(nodeB, nodeD)
view.add_connection(nodeC, nodeD)

sys.exit(app.exec_())