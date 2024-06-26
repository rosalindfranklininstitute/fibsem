# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemCryoDepositionWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(416, 221)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label_duration = QtWidgets.QLabel(Dialog)
        self.label_duration.setObjectName("label_duration")
        self.gridLayout.addWidget(self.label_duration, 5, 0, 1, 1)
        self.comboBox_stage_position = QtWidgets.QComboBox(Dialog)
        self.comboBox_stage_position.setObjectName("comboBox_stage_position")
        self.gridLayout.addWidget(self.comboBox_stage_position, 1, 1, 1, 1)
        self.label_title = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 2)
        self.label_stage_position = QtWidgets.QLabel(Dialog)
        self.label_stage_position.setObjectName("label_stage_position")
        self.gridLayout.addWidget(self.label_stage_position, 1, 0, 1, 1)
        self.lineEdit_gas = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_gas.setObjectName("lineEdit_gas")
        self.gridLayout.addWidget(self.lineEdit_gas, 4, 1, 1, 1)
        self.doubleSpinBox_duration = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBox_duration.setMaximum(1000.0)
        self.doubleSpinBox_duration.setProperty("value", 30.0)
        self.doubleSpinBox_duration.setObjectName("doubleSpinBox_duration")
        self.gridLayout.addWidget(self.doubleSpinBox_duration, 5, 1, 1, 1)
        self.label_port = QtWidgets.QLabel(Dialog)
        self.label_port.setObjectName("label_port")
        self.gridLayout.addWidget(self.label_port, 3, 0, 1, 1)
        self.pushButton_run_sputter = QtWidgets.QPushButton(Dialog)
        self.pushButton_run_sputter.setObjectName("pushButton_run_sputter")
        self.gridLayout.addWidget(self.pushButton_run_sputter, 7, 0, 1, 2)
        self.label_gas = QtWidgets.QLabel(Dialog)
        self.label_gas.setObjectName("label_gas")
        self.gridLayout.addWidget(self.label_gas, 4, 0, 1, 1)
        self.label_insert_position = QtWidgets.QLabel(Dialog)
        self.label_insert_position.setObjectName("label_insert_position")
        self.gridLayout.addWidget(self.label_insert_position, 6, 0, 1, 1)
        self.lineEdit_insert_position = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_insert_position.setObjectName("lineEdit_insert_position")
        self.gridLayout.addWidget(self.lineEdit_insert_position, 6, 1, 1, 1)
        self.comboBox_port = QtWidgets.QComboBox(Dialog)
        self.comboBox_port.setObjectName("comboBox_port")
        self.gridLayout.addWidget(self.comboBox_port, 3, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_duration.setText(_translate("Dialog", "Duration (s)"))
        self.label_title.setText(_translate("Dialog", "Cryo Deposition"))
        self.label_stage_position.setText(_translate("Dialog", "Stage Position"))
        self.lineEdit_gas.setText(_translate("Dialog", "Pt cryo"))
        self.label_port.setText(_translate("Dialog", "Port"))
        self.pushButton_run_sputter.setText(_translate("Dialog", "Run Cryo Deposition"))
        self.label_gas.setText(_translate("Dialog", "Gas"))
        self.label_insert_position.setText(_translate("Dialog", "Insert Position"))
        self.lineEdit_insert_position.setText(_translate("Dialog", "cryo"))
