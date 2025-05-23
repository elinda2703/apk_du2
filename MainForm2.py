# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt6 UI code generator 6.8.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets
from draw2 import Draw
from algorithms2 import *



class Ui_MainForm(object):
    def setupUi(self, MainForm):
        MainForm.setObjectName("MainForm")
        MainForm.resize(1123, 856)
        self.centralwidget = QtWidgets.QWidget(parent=MainForm)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Canvas = Draw(parent=self.centralwidget)
        self.Canvas.setObjectName("Canvas")
        self.horizontalLayout.addWidget(self.Canvas)
        MainForm.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainForm)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1123, 26))
        self.menubar.setObjectName("menubar")
        self.menuOpen = QtWidgets.QMenu(parent=self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.menuSimplify = QtWidgets.QMenu(parent=self.menubar)
        self.menuSimplify.setObjectName("menuSimplify")
        self.menuView = QtWidgets.QMenu(parent=self.menubar)
        self.menuView.setObjectName("menuView")
        MainForm.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainForm)
        self.statusbar.setObjectName("statusbar")
        MainForm.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(parent=MainForm)
        self.toolBar.setObjectName("toolBar")
        MainForm.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolBar)
        self.actionOpen = QtGui.QAction(parent=MainForm)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Images/Icons/open_file.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionOpen.setIcon(icon)
        self.actionOpen.setObjectName("actionOpen")
        self.actionExit = QtGui.QAction(parent=MainForm)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("Images/Icons/exit.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionExit.setIcon(icon1)
        self.actionExit.setObjectName("actionExit")
        self.actionMBR = QtGui.QAction(parent=MainForm)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("Images/Icons/maer.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionMBR.setIcon(icon2)
        self.actionMBR.setObjectName("actionMBR")
        self.actionPCA = QtGui.QAction(parent=MainForm)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("Images/Icons/pca.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionPCA.setIcon(icon3)
        self.actionPCA.setObjectName("actionPCA")
        self.actionClear_results = QtGui.QAction(parent=MainForm)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("Images/Icons/clear_ch.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionClear_results.setIcon(icon4)
        self.actionClear_results.setObjectName("actionClear_results")
        self.actionClear_all = QtGui.QAction(parent=MainForm)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("Images/Icons/clear_er.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionClear_all.setIcon(icon5)
        self.actionClear_all.setObjectName("actionClear_all")
        self.actionLongest_Edge = QtGui.QAction(parent=MainForm)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("Images/Icons/longestedge.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionLongest_Edge.setIcon(icon6)
        self.actionLongest_Edge.setObjectName("actionLongest_Edge")
        self.actionWeighted_Bisector = QtGui.QAction(parent=MainForm)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("Images/Icons/weightedbisector.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionWeighted_Bisector.setIcon(icon7)
        self.actionWeighted_Bisector.setObjectName("actionWeighted_Bisector")
        self.actionWall_Average = QtGui.QAction(parent=MainForm)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("Images/Icons/wa.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionWall_Average.setIcon(icon8)
        self.actionWall_Average.setObjectName("actionWall_Average")
        self.actionJarvis_Graham_Scan = QtGui.QAction(parent=MainForm)
        self.actionJarvis_Graham_Scan.setCheckable(True)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("Images/Icons/ch.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionJarvis_Graham_Scan.setIcon(icon9)
        self.actionJarvis_Graham_Scan.setObjectName("actionJarvis_Graham_Scan")
        self.menuOpen.addAction(self.actionOpen)
        self.menuOpen.addSeparator()
        self.menuOpen.addAction(self.actionExit)
        self.menuSimplify.addAction(self.actionMBR)
        self.menuSimplify.addAction(self.actionPCA)
        self.menuSimplify.addAction(self.actionLongest_Edge)
        self.menuSimplify.addAction(self.actionWeighted_Bisector)
        self.menuSimplify.addAction(self.actionWall_Average)
        self.menuSimplify.addSeparator()
        self.menuSimplify.addAction(self.actionJarvis_Graham_Scan)
        self.menuView.addAction(self.actionClear_results)
        self.menuView.addAction(self.actionClear_all)
        self.menubar.addAction(self.menuOpen.menuAction())
        self.menubar.addAction(self.menuSimplify.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionJarvis_Graham_Scan)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionMBR)
        self.toolBar.addAction(self.actionPCA)
        self.toolBar.addAction(self.actionLongest_Edge)
        self.toolBar.addAction(self.actionWeighted_Bisector)
        self.toolBar.addAction(self.actionWall_Average)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionClear_results)
        self.toolBar.addAction(self.actionClear_all)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionExit)

        self.retranslateUi(MainForm)
        QtCore.QMetaObject.connectSlotsByName(MainForm)
        
        # Connect menu and functions
        self.actionMBR.triggered.connect(self.simplifyBuildingMBR)
        self.actionPCA.triggered.connect(self.simplifyBuildingBRPCA)
        self.actionLongest_Edge.triggered.connect(self.simplifyBuildingLE)
        self.actionWeighted_Bisector.triggered.connect(self.simplifyBuildingWB)
        self.actionWall_Average.triggered.connect(self.simplifyBuildingWA)
        self.actionJarvis_Graham_Scan.triggered.connect(self.switchClick)
        self.actionOpen.triggered.connect(self.openClick)
        self.actionExit.triggered.connect(self.exitClick)
        self.actionClear_results.triggered.connect(self.clearClick)
        self.actionClear_all.triggered.connect(self.clearAllClick)
        

    def retranslateUi(self, MainForm):
        _translate = QtCore.QCoreApplication.translate
        MainForm.setWindowTitle(_translate("MainForm", "Building Simplify"))
        self.menuOpen.setTitle(_translate("MainForm", "File"))
        self.menuSimplify.setTitle(_translate("MainForm", "Simplify"))
        self.menuView.setTitle(_translate("MainForm", "View"))
        self.toolBar.setWindowTitle(_translate("MainForm", "toolBar"))
        self.actionOpen.setText(_translate("MainForm", "Open"))
        self.actionOpen.setToolTip(_translate("MainForm", "Open file"))
        self.actionExit.setText(_translate("MainForm", "Exit"))
        self.actionExit.setToolTip(_translate("MainForm", "Exit aplication"))
        self.actionMBR.setText(_translate("MainForm", "MBR"))
        self.actionMBR.setToolTip(_translate("MainForm", "Simplify building using MBR"))
        self.actionPCA.setText(_translate("MainForm", "PCA"))
        self.actionPCA.setToolTip(_translate("MainForm", "Simplify building use PCA"))
        self.actionClear_results.setText(_translate("MainForm", "Clear results"))
        self.actionClear_all.setText(_translate("MainForm", "Clear all"))
        self.actionLongest_Edge.setText(_translate("MainForm", "Longest Edge"))
        self.actionLongest_Edge.setToolTip(_translate("MainForm", "Simplify building using Longest Edge"))
        self.actionWeighted_Bisector.setText(_translate("MainForm", "Weighted Bisector"))
        self.actionWeighted_Bisector.setToolTip(_translate("MainForm", "Simplify building using Weighted Bisector"))
        self.actionWall_Average.setText(_translate("MainForm", "Wall Average"))
        self.actionWall_Average.setToolTip(_translate("MainForm", "Simplify building using Wall Average"))
        self.actionJarvis_Graham_Scan.setText(_translate("MainForm", "Jarvis/Graham Scan"))
        self.actionJarvis_Graham_Scan.setToolTip(_translate("MainForm", "Use Jarvis/Graham Scan"))


    def simplifyBuildingCore(self, method: str):
        buildings = ui.Canvas.getBuilding()
        buildings_simp = []
        a = Algorithms()
        correct_building_counter = 0
        for b in buildings:
            if method == "MBR":
                ch_method = ui.Canvas.getMethodCH()
                simplified, sig = a.createMBR(b, ch_method)
            elif method == "BRPCA":
                simplified, sig = a.createBRPCA(b)
            elif method == "LE":
                simplified, sig = a.longestEdge(b)
            elif method == "WB":
                simplified, sig = a.weightedBisector(b)
            elif method == "WA":
                simplified, sig = a.wallAverage(b)
            else:
                raise ValueError(f"Unknown simplification method: {method}")

            buildings_simp.append(simplified)
            
            if sig is not None:
                building_dif=a.evaluate(sig, b)
                if abs(building_dif) < 10:
                    correct_building_counter+=1
            
            final_accuracy = 100*correct_building_counter/len(buildings)
                

        ui.Canvas.setSimplifBuilding(buildings_simp)
        self.Canvas.repaint()
        
        dialog = QtWidgets.QMessageBox()
        dialog.setWindowTitle('Simplification accuracy')
        
        dialog.setText(f"The accuracy of simplification is {round(final_accuracy,2)} %")

        # Show dialog
        dialog.exec()
        
        
    def simplifyBuildingMBR(self):
        self.simplifyBuildingCore("MBR")

    def simplifyBuildingBRPCA(self):
        self.simplifyBuildingCore("BRPCA")

    def simplifyBuildingLE(self):
        self.simplifyBuildingCore("LE")

    def simplifyBuildingWB(self):
        self.simplifyBuildingCore("WB")

    def simplifyBuildingWA(self):
        self.simplifyBuildingCore("WA")


        
    def switchClick(self):
        '''
        Switch method of creating CH
        '''
        ui.Canvas.switchMethodCH()
        
        
    def openClick(self):
        self.Canvas.openFile()
        
    def exitClick(self):
        self.Canvas.exit()
        
    def clearAllClick(self):
        self.Canvas.clearAll()
        
    def clearClick(self):
        self.Canvas.clearRes()
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainForm = QtWidgets.QMainWindow()
    ui = Ui_MainForm()
    ui.setupUi(MainForm)
    MainForm.show()
    sys.exit(app.exec())
