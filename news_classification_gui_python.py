# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Users/alias/PycharmProjects/newsClassification/news_classification_gui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(918, 779)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.dataset_tab = QtWidgets.QWidget()
        self.dataset_tab.setObjectName("dataset_tab")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.dataset_tab)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_4.addItem(spacerItem)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        spacerItem1 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_13.addItem(spacerItem1)
        self.datasetLabel = QtWidgets.QLabel(self.dataset_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.datasetLabel.setFont(font)
        self.datasetLabel.setObjectName("datasetLabel")
        self.horizontalLayout_13.addWidget(self.datasetLabel)
        self.datasetDatasetBox = QtWidgets.QComboBox(self.dataset_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.datasetDatasetBox.setFont(font)
        self.datasetDatasetBox.setObjectName("datasetDatasetBox")
        self.datasetDatasetBox.addItem("")
        self.datasetDatasetBox.addItem("")
        self.horizontalLayout_13.addWidget(self.datasetDatasetBox)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_13.addItem(spacerItem2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        spacerItem3 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem3)
        self.datasetCategories = QtWidgets.QLabel(self.dataset_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.datasetCategories.setFont(font)
        self.datasetCategories.setObjectName("datasetCategories")
        self.horizontalLayout_14.addWidget(self.datasetCategories)
        self.verticalLayout_4.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        spacerItem4 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem4)
        self.datasetlabel_9 = QtWidgets.QLabel(self.dataset_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.datasetlabel_9.setFont(font)
        self.datasetlabel_9.setObjectName("datasetlabel_9")
        self.horizontalLayout_15.addWidget(self.datasetlabel_9)
        self.datasetModelBox = QtWidgets.QComboBox(self.dataset_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.datasetModelBox.setFont(font)
        self.datasetModelBox.setObjectName("datasetModelBox")
        self.datasetModelBox.addItem("")
        self.datasetModelBox.addItem("")
        self.datasetModelBox.addItem("")
        self.datasetModelBox.addItem("")
        self.horizontalLayout_15.addWidget(self.datasetModelBox)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem5)
        self.verticalLayout_4.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem6)
        self.datasetPushButton = QtWidgets.QPushButton(self.dataset_tab)
        self.datasetPushButton.setObjectName("datasetPushButton")
        self.horizontalLayout_16.addWidget(self.datasetPushButton)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem7)
        self.verticalLayout_4.addLayout(self.horizontalLayout_16)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_20.addItem(spacerItem8)
        self.datasetReportLabel = QtWidgets.QLabel(self.dataset_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.datasetReportLabel.setFont(font)
        self.datasetReportLabel.setObjectName("datasetReportLabel")
        self.horizontalLayout_20.addWidget(self.datasetReportLabel)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_20.addItem(spacerItem9)
        self.verticalLayout_4.addLayout(self.horizontalLayout_20)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_17.addItem(spacerItem10)
        self.datasetReportText = QtWidgets.QTextBrowser(self.dataset_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.datasetReportText.sizePolicy().hasHeightForWidth())
        self.datasetReportText.setSizePolicy(sizePolicy)
        self.datasetReportText.setMinimumSize(QtCore.QSize(350, 240))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.datasetReportText.setFont(font)
        self.datasetReportText.setObjectName("datasetReportText")
        self.horizontalLayout_17.addWidget(self.datasetReportText)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_17.addItem(spacerItem11)
        self.verticalLayout_4.addLayout(self.horizontalLayout_17)
        spacerItem12 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_4.addItem(spacerItem12)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem13)
        self.label = QtWidgets.QLabel(self.dataset_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem14 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem14)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem15)
        self.datasetNewsText = QtWidgets.QTextBrowser(self.dataset_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.datasetNewsText.setFont(font)
        self.datasetNewsText.setObjectName("datasetNewsText")
        self.horizontalLayout_18.addWidget(self.datasetNewsText)
        spacerItem16 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem16)
        self.verticalLayout_4.addLayout(self.horizontalLayout_18)
        spacerItem17 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_4.addItem(spacerItem17)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)
        self.tabWidget.addTab(self.dataset_tab, "")
        self.sentence_tab = QtWidgets.QWidget()
        self.sentence_tab.setObjectName("sentence_tab")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout(self.sentence_tab)
        self.horizontalLayout_19.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        spacerItem18 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_3.addItem(spacerItem18)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        spacerItem19 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem19)
        self.sentenceDatasetLabel = QtWidgets.QLabel(self.sentence_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sentenceDatasetLabel.setFont(font)
        self.sentenceDatasetLabel.setObjectName("sentenceDatasetLabel")
        self.horizontalLayout_9.addWidget(self.sentenceDatasetLabel)
        self.sentenceDatasetBox = QtWidgets.QComboBox(self.sentence_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sentenceDatasetBox.setFont(font)
        self.sentenceDatasetBox.setObjectName("sentenceDatasetBox")
        self.sentenceDatasetBox.addItem("")
        self.sentenceDatasetBox.addItem("")
        self.horizontalLayout_9.addWidget(self.sentenceDatasetBox)
        spacerItem20 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem20)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        spacerItem21 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem21)
        self.sentenceCategories = QtWidgets.QLabel(self.sentence_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sentenceCategories.setFont(font)
        self.sentenceCategories.setObjectName("sentenceCategories")
        self.horizontalLayout_10.addWidget(self.sentenceCategories)
        self.verticalLayout_3.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        spacerItem22 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem22)
        self.sentenceClearButton = QtWidgets.QPushButton(self.sentence_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sentenceClearButton.setFont(font)
        self.sentenceClearButton.setObjectName("sentenceClearButton")
        self.horizontalLayout_11.addWidget(self.sentenceClearButton)
        spacerItem23 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem23)
        self.verticalLayout_3.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem24 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem24)
        self.sentenceTextEdit = QtWidgets.QTextEdit(self.sentence_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sentenceTextEdit.sizePolicy().hasHeightForWidth())
        self.sentenceTextEdit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sentenceTextEdit.setFont(font)
        self.sentenceTextEdit.setObjectName("sentenceTextEdit")
        self.horizontalLayout_4.addWidget(self.sentenceTextEdit)
        spacerItem25 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem25)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        spacerItem26 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_3.addItem(spacerItem26)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        spacerItem27 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem27)
        self.sentenceFindButton = QtWidgets.QPushButton(self.sentence_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sentenceFindButton.setFont(font)
        self.sentenceFindButton.setObjectName("sentenceFindButton")
        self.horizontalLayout_12.addWidget(self.sentenceFindButton)
        spacerItem28 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem28)
        self.verticalLayout_3.addLayout(self.horizontalLayout_12)
        spacerItem29 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.verticalLayout_3.addItem(spacerItem29)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem30 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem30)
        self.sentenceCategoryTable = QtWidgets.QTableWidget(self.sentence_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sentenceCategoryTable.sizePolicy().hasHeightForWidth())
        self.sentenceCategoryTable.setSizePolicy(sizePolicy)
        self.sentenceCategoryTable.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sentenceCategoryTable.setFont(font)
        self.sentenceCategoryTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.sentenceCategoryTable.setObjectName("sentenceCategoryTable")
        self.sentenceCategoryTable.setColumnCount(4)
        self.sentenceCategoryTable.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.sentenceCategoryTable.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.sentenceCategoryTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.sentenceCategoryTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.sentenceCategoryTable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.sentenceCategoryTable.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.sentenceCategoryTable.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.sentenceCategoryTable.setItem(0, 1, item)
        self.sentenceCategoryTable.horizontalHeader().setCascadingSectionResizes(False)
        self.sentenceCategoryTable.verticalHeader().setStretchLastSection(True)
        self.horizontalLayout_3.addWidget(self.sentenceCategoryTable)
        spacerItem31 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem31)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        spacerItem32 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem32)
        self.horizontalLayout_19.addLayout(self.verticalLayout_3)
        self.tabWidget.addTab(self.sentence_tab, "")
        self.verticalLayout_6.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 918, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.sentenceFindButton.clicked.connect(MainWindow.predict)
        self.sentenceClearButton.clicked.connect(MainWindow.clear_text_edit)
        self.datasetPushButton.clicked.connect(MainWindow.classification_report)
        self.tabWidget.tabBarClicked['int'].connect(MainWindow.change_window_size)
        self.sentenceDatasetBox.currentIndexChanged['int'].connect(MainWindow.change_category_list_sentence)
        self.datasetDatasetBox.currentIndexChanged['int'].connect(MainWindow.change_category_list_dataset)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "News Classification"))
        self.datasetLabel.setText(_translate("MainWindow", "Dataset: "))
        self.datasetDatasetBox.setItemText(0, _translate("MainWindow", "News Article Dataset"))
        self.datasetDatasetBox.setItemText(1, _translate("MainWindow", "BBC News Dataset"))
        self.datasetCategories.setText(_translate("MainWindow", "Categories: Sport -  World - US - Business - Health - Entertainment - Science & Technology"))
        self.datasetlabel_9.setText(_translate("MainWindow", "Model:    "))
        self.datasetModelBox.setItemText(0, _translate("MainWindow", "Multinomial Naive Bayes"))
        self.datasetModelBox.setItemText(1, _translate("MainWindow", "Support Vector Machines"))
        self.datasetModelBox.setItemText(2, _translate("MainWindow", "Logistic Regression"))
        self.datasetModelBox.setItemText(3, _translate("MainWindow", "Gradient Boosting Classifier"))
        self.datasetPushButton.setText(_translate("MainWindow", "Report"))
        self.datasetReportLabel.setText(_translate("MainWindow", "Classification Report"))
        self.label.setText(_translate("MainWindow", "News - Actual Classes & Predicted Classes"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.dataset_tab), _translate("MainWindow", "Dataset"))
        self.sentenceDatasetLabel.setText(_translate("MainWindow", "Dataset: "))
        self.sentenceDatasetBox.setItemText(0, _translate("MainWindow", "News Article Dataset"))
        self.sentenceDatasetBox.setItemText(1, _translate("MainWindow", "BBC News Dataset"))
        self.sentenceCategories.setText(_translate("MainWindow", "Categories: Sport -  World - US - Business - Health - Entertainment - Science & Technology"))
        self.sentenceClearButton.setText(_translate("MainWindow", "Clear"))
        self.sentenceFindButton.setText(_translate("MainWindow", "Find The Category"))
        item = self.sentenceCategoryTable.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Category"))
        item = self.sentenceCategoryTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Multinomial Naive Bayes"))
        item = self.sentenceCategoryTable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Support Vector Machines"))
        item = self.sentenceCategoryTable.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Logistic Regression"))
        item = self.sentenceCategoryTable.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Gradient Boosting Classifier"))
        __sortingEnabled = self.sentenceCategoryTable.isSortingEnabled()
        self.sentenceCategoryTable.setSortingEnabled(False)
        self.sentenceCategoryTable.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.sentence_tab), _translate("MainWindow", "Text"))

