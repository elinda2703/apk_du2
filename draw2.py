from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtGui import QMouseEvent, QPaintEvent
from PyQt6.QtWidgets import *
import geopandas as gpd


class Draw(QWidget):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.building = QPolygonF()
        self.building_simp = QPolygonF()
        # Method of creating CH
        self.__methodCH = True
        # Check shapefile loding
        self.shp_loaded = False
        # List of buildings
        self.shp_buildings = []
        

    def openFile(self):
        """
        Open file and read file
        """
        file_name, _ = QFileDialog.getOpenFileName(None, "Open File", "", "Shapefile (*.shp)")

        if file_name:
            
            try:
                # Load shp
                self.shp = gpd.read_file(file_name)
                
                # Shp geometry
                self.geomShapefile()
                                
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Could not open file: {str(e)}")


    def geomShapefile(self):
        """
        Geometry for drawing
        """
        self.shp_buildings.clear()
        
        # Chatgpt + own from line 47 to 71
        if self.shp is None or self.shp.empty:
            QMessageBox.critical(None, "Error", "No geometry in shapefile")
            return

        # Find min/max to normalize coordinates
        min_x, min_y, max_x, max_y = self.shp.total_bounds

        width = self.width()
        height = self.height()
        
        for geom in self.shp.geometry:
            if geom.geom_type == "Polygon":                
                pol = QPolygonF([QPointF((float(x) - min_x) / (max_x - min_x) * width,
                    height - (float(y) - min_y) / (max_y - min_y) * height
                    )
                    for x, y in geom.exterior.coords
                ])
                
                self.shp_buildings.append(pol)
                        
            else:
                print(f"Geometry: {str(geom.geom_type)}")
                QMessageBox.critical(None, "Error", f"Shapefile contains: {str(geom.geom_type)}")
                break

        self.shp_loaded = True
        self.repaint()


    def exit(self):
        """
        Exit GUI
        """
        QApplication.instance().quit()


    def clearAll(self):
        '''
        Clear all data
        '''
        self.shp_buildings.clear()
        self.building.clear()
        self.building_simp.clear()
        self.shp_loaded = False
        
        #Repaint screen
        self.repaint()
        
        
    def clearRes(self):
        '''
        Clear result (simplify buildings)
        '''
        self.building_simp.clear()
        
        #Repaint screen
        self.repaint()


    def mousePressEvent(self, e:QMouseEvent):
        #Get coordinates x,y
        x = e.position().x()
        y = e.position().y()
        

        #Create temporary point
        p = QPointF(x, y)
    
        #Add p to polygon
        self.building.append(p)
        
        #Repaint screen
        self.repaint()
 
 
    def paintEvent(self, e: QPaintEvent):
        '''
        Draw situation
        '''
        #Create new graphic object
        qp = QPainter(self)
        
        #Start drawing
        qp.begin(self)

        if self.shp_loaded:
            qp.setPen(Qt.GlobalColor.black)
            qp.setBrush(Qt.GlobalColor.darkGray)
            # Draw buildings
            for b in self.shp_buildings:
                qp.drawPolygon(b)
        else:
            #Set graphic attributes: buillding
            qp.setPen(Qt.GlobalColor.black)
            qp.setBrush(Qt.GlobalColor.yellow)
            #Draw building
            qp.drawPolygon(self.building)
        
        #Set graphic attributes: buillding_simplify
        qp.setBrush(QColor(255,0,0,50))
        #Draw building simplify
        for simp in self.building_simp:
            qp.drawPolygon(simp)
        
        #End drawing
        qp.end()
    
    
    def switchMethodCH(self):
        self.__methodCH = not (self.__methodCH)
            
    
    def getBuilding(self):
        #Return list of analyze buildings
        if self.shp_loaded:
            return self.shp_buildings
        else:
            return [self.building]
    
    
    def setSimplifBuilding(self, buildings_simp_):
        self.building_simp = buildings_simp_
        
        
    def getMethodCH(self):
        if self.__methodCH:
            return 1
        else:
            return 2