from PyQt5.QtWidgets import *
from news_classification_gui_python import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

class newsClassification(QMainWindow):
    data_news_article = []
    na_tfidf = na_vec = na_model_nb = na_model_svm = na_model_logistic_reg = na_model_gradient_boosting = None
    data_bbc = []
    bbc_tfidf = bbc_vec = bbc_model_nb = bbc_model_svm = bbc_model_logistic_reg = bbc_model_gradient_boosting = None

    def __init__(self):
        global data_news_article, na_tfidf, na_vec, na_model_nb, na_model_svm, na_model_logistic_reg, na_model_gradient_boosting
        global training_data, na_X_train, na_X_test, na_y_train, na_y_test
        global data_bbc, bbc_tfidf, bbc_vec, bbc_model_nb, bbc_model_svm, bbc_model_logistic_reg, bbc_model_gradient_boosting
        global bbc_training_data, bbc_X_train, bbc_X_test, bbc_y_train, bbc_y_test

        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.sentenceCategoryTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.ui.sentenceCategoryTable.resizeColumnsToContents()

        # Load Model for News Article
        na_vec = CountVectorizer(vocabulary=pickle.load(open("news_article/na_count_vector.pkl", "rb")))
        na_tfidf = pickle.load(open("news_article/na_tfidf.pkl", "rb"))
        na_model_nb = pickle.load(open("news_article/na_nb_model.pkl", "rb"))
        na_model_svm = pickle.load(open("news_article/na_svm.pkl", "rb"))
        na_model_logistic_reg = pickle.load(open("news_article/na_logistic_reg.pkl", "rb"))
        na_model_gradient_boosting = pickle.load(open("news_article/na_gradient_boosting.pkl", "rb"))

        # Load the News Article Dataset
        training_data = pd.read_csv("news_article/news_article.csv", sep=',', encoding='utf-8')
        # Get Vector Count
        na_X_train_counts = na_vec.fit_transform(training_data.data)
        # Transforming word vectors to tf–idf
        na_X_train_tfidf = na_tfidf.fit_transform(na_X_train_counts)
        # Split the data for training and test set
        na_X_train, na_X_test, na_y_train, na_y_test = train_test_split(na_X_train_tfidf, training_data.flag, test_size=0.30, random_state=42)

        # Load Model for BBC
        bbc_vec = CountVectorizer(vocabulary=pickle.load(open("bbc/bbc_count_vector.pkl", "rb")))
        bbc_tfidf = pickle.load(open("bbc/bbc_tfidf.pkl", "rb"))
        bbc_model_nb = pickle.load(open("bbc/bbc_nb_model.pkl", "rb"))
        bbc_model_svm = pickle.load(open("bbc/bbc_svm.pkl", "rb"))
        bbc_model_logistic_reg = pickle.load(open("bbc/bbc_logistic_reg.pkl", "rb"))
        bbc_model_gradient_boosting = pickle.load(open("bbc/bbc_gradient_boosting.pkl", "rb"))

        # Load the BBC News Dataset
        bbc_training_data = pd.read_csv("bbc/bbc.csv")
        # Tuning on the dataset
        bbc_training_data['category_id'] = bbc_training_data['type'].factorize()[0]
        colslist = ['Index', 'news', 'type', 'category_id']
        bbc_training_data.columns = colslist
        # Get Vector Count
        bbc_X_train_counts = bbc_vec.fit_transform(bbc_training_data.news)
        # Transforming word vectors to tf–idf
        bbc_X_train_tfidf = bbc_tfidf.fit_transform(bbc_X_train_counts)
        # Split the data for training and test set
        bbc_X_train, bbc_X_test, bbc_y_train, bbc_y_test = train_test_split(bbc_X_train_tfidf, bbc_training_data.type, test_size=0.25, random_state=42)

    def predict(self):
        category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
        dataset = self.ui.sentenceDatasetBox.currentIndex()
        text = self.ui.sentenceTextEdit.toPlainText()
        text = [text]

        if dataset == 0:
            # Get vector counts and transforming word vectors to tf–idf for News Article dataset
            X_new_counts = na_vec.transform(text)
            X_new_tfidf = na_tfidf.transform(X_new_counts)

            # Predictions for the News Article dataset
            predicted1 = na_model_nb.predict(X_new_tfidf)
            predicted2 = na_model_svm.predict(X_new_tfidf)
            predicted3 = na_model_logistic_reg.predict(X_new_tfidf)
            predicted4 = na_model_gradient_boosting.predict(X_new_tfidf)

            item1 = QtWidgets.QTableWidgetItem(category_list[predicted1[0]])
            item1.setTextAlignment(Qt.AlignCenter)
            self.ui.sentenceCategoryTable.setItem(0, 0, item1)
            item2 = QtWidgets.QTableWidgetItem(category_list[predicted2[0]])
            item2.setTextAlignment(Qt.AlignCenter)
            self.ui.sentenceCategoryTable.setItem(0, 1, item2)
            item3 = QtWidgets.QTableWidgetItem(category_list[predicted3[0]])
            item3.setTextAlignment(Qt.AlignCenter)
            self.ui.sentenceCategoryTable.setItem(0, 2, item3)
            item4 = QtWidgets.QTableWidgetItem(category_list[predicted4[0]])
            item4.setTextAlignment(Qt.AlignCenter)
            self.ui.sentenceCategoryTable.setItem(0, 3, item4)

        else:
            # Get vector counts and transforming word vectors to tf–idf for BBC News dataset
            X_new_counts = bbc_vec.transform(text)
            X_new_tfidf = bbc_tfidf.transform(X_new_counts)

            # Predictions for the BBC News dataset
            predicted_nb = bbc_model_nb.predict(X_new_tfidf)
            predicted_svm = bbc_model_svm.predict(X_new_tfidf)
            predicted_logistic_reg = bbc_model_logistic_reg.predict(X_new_tfidf)
            predicted_gradient_boosting = bbc_model_gradient_boosting.predict(X_new_tfidf)

            item1 = QtWidgets.QTableWidgetItem(predicted_nb[0])
            item1.setTextAlignment(Qt.AlignCenter)
            self.ui.sentenceCategoryTable.setItem(0, 0, item1)
            item2 = QtWidgets.QTableWidgetItem(predicted_svm[0])
            item2.setTextAlignment(Qt.AlignCenter)
            self.ui.sentenceCategoryTable.setItem(0, 1, item2)
            item3 = QtWidgets.QTableWidgetItem(predicted_logistic_reg[0])
            item3.setTextAlignment(Qt.AlignCenter)
            self.ui.sentenceCategoryTable.setItem(0, 2, item3)
            item4 = QtWidgets.QTableWidgetItem(predicted_gradient_boosting[0])
            item4.setTextAlignment(Qt.AlignCenter)
            self.ui.sentenceCategoryTable.setItem(0, 3, item4)

        self.ui.statusbar.showMessage("Prediction is complete.", 3000)

    def clear_text_edit(self):
        self.ui.sentenceTextEdit.clear()
        item1 = QtWidgets.QTableWidgetItem("")
        self.ui.sentenceCategoryTable.setItem(0, 0, item1)
        item2 = QtWidgets.QTableWidgetItem("")
        self.ui.sentenceCategoryTable.setItem(0, 1, item2)
        item3 = QtWidgets.QTableWidgetItem("")
        self.ui.sentenceCategoryTable.setItem(0, 2, item3)
        item4 = QtWidgets.QTableWidgetItem("")
        self.ui.sentenceCategoryTable.setItem(0, 3, item4)

    def change_window_size(self):
        # Changes window size on tab changes
        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.datasetNewsText.clear()
            self.ui.datasetReportText.clear()
            self.ui.datasetReportLabel.setText("Classification Report")
            self.resize(801, 547)
        else:
            self.resize(849, 780)

    def change_category_list_dataset(self):
        # Existing categories are shown on dataset changes in dataset tab
        if self.ui.datasetDatasetBox.currentIndex() == 1:
            self.ui.datasetCategories.setText("Categories: Business - Entertainment - Politics - Sport - Technology")
        else:
            self.ui.datasetCategories.setText("Categories: Sport -  World - US - Business - Health - Entertainment - Science & Technology")

    def change_category_list_sentence(self):
        # Existing categories are shown on dataset changes in sentence tab
        if self.ui.sentenceDatasetBox.currentIndex() == 1:
            self.ui.sentenceCategories.setText("Categories: Business - Entertainment - Politics - Sport - Technology")
        else:
            self.ui.sentenceCategories.setText("Categories: Sport -  World - US - Business - Health - Entertainment - Science & Technology")

    def classification_report(self):
        #A report is presented on the classification performance of the models according to the data set.
        #Each news in the data set is shown with its actual class and the predicted class.

        dataset_category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
        dataset_target_names = ["sport", "world", "us", "business", "health", "entmt", "sci_tech"]

        bbc_category_list = ["business", "entertainment", "politics", "sport", "tech"]
        bbc_target_names = ["business", "entmt", "politics", "sport", "tech"]

        dataset_dataset = self.ui.datasetDatasetBox.currentIndex()
        dataset_model = self.ui.datasetModelBox.currentIndex()

        # Classification on News Article Dataset
        if dataset_dataset == 0:
            if dataset_model == 0:
                self.ui.datasetReportLabel.setText("Classification Report: News Article Dataset - Multinomial Naive Bayes")
                predicted = na_model_nb.predict(na_X_test)
                self.ui.datasetReportText.setText(classification_report(na_y_test, predicted, target_names=dataset_target_names))

                i = 1
                self.ui.datasetNewsText.clear()
                for predicted_item, result in zip(predicted, na_y_test):
                    a = str(i) + ") " + training_data.data[i-1] + "\n{ " + dataset_category_list[result] + ' - ' + dataset_category_list[predicted_item] + " }\n"
                    self.ui.datasetNewsText.append(a)
                    i += 1

            elif dataset_model == 1:
                self.ui.datasetReportLabel.setText("Classification Report: News Article Dataset - Support Vector Machines")
                predicted = na_model_svm.predict(na_X_test)
                self.ui.datasetReportText.setText(classification_report(na_y_test, predicted, target_names=dataset_target_names))

                i = 1
                self.ui.datasetNewsText.clear()
                for predicted_item, result in zip(predicted, na_y_test):
                    a = str(i) + ") " + training_data.data[i-1] + "\n{ " + dataset_category_list[result] + ' - ' + dataset_category_list[predicted_item] + " }\n"
                    self.ui.datasetNewsText.append(a)
                    i += 1
            elif dataset_model == 2:
                self.ui.datasetReportLabel.setText("Classification Report: News Article Dataset - Logistic Regression")
                predicted = na_model_logistic_reg.predict(na_X_test)
                self.ui.datasetReportText.setText(classification_report(na_y_test, predicted, target_names=dataset_target_names))

                i = 1
                self.ui.datasetNewsText.clear()
                for predicted_item, result in zip(predicted, na_y_test):
                    a = str(i) + ") " + training_data.data[i-1] + "\n{ " + dataset_category_list[result] + ' - ' + dataset_category_list[predicted_item] + " }\n"
                    self.ui.datasetNewsText.append(a)
                    i += 1
            else:
                self.ui.datasetReportLabel.setText("Classification Report: News Article Dataset - Gradient Boosting Classifier")
                predicted = na_model_gradient_boosting.predict(na_X_test)
                self.ui.datasetReportText.setText(classification_report(na_y_test, predicted, target_names=dataset_target_names))

                i = 1
                self.ui.datasetNewsText.clear()
                for predicted_item, result in zip(predicted, na_y_test):
                    a = str(i) + ") " + training_data.data[i-1] + "\n{ " + dataset_category_list[result] + ' - ' + dataset_category_list[predicted_item] + " }\n"
                    self.ui.datasetNewsText.append(a)
                    i += 1
        # Classification on BBC News Dataset
        else:
            if dataset_model == 0:
                self.ui.datasetReportLabel.setText("Classification Report: BBC News Dataset - Multinomial Naive Bayes")
                predicted = bbc_model_nb.predict(bbc_X_test)
                self.ui.datasetReportText.setText(classification_report(bbc_y_test, predicted, target_names=bbc_target_names))
                i = 1
                self.ui.datasetNewsText.clear()
                for predicted_item, result in zip(predicted, bbc_y_test):
                    a = str(i) + ") " + training_data.data[i-1] + "\n{ " + result + ' - ' + predicted_item + " }\n"
                    self.ui.datasetNewsText.append(a)
                    i += 1

            elif dataset_model == 1:
                self.ui.datasetReportLabel.setText("Classification Report: BBC News Dataset - Support Vector Machines")
                predicted = bbc_model_svm.predict(bbc_X_test)
                self.ui.datasetReportText.setText(classification_report(bbc_y_test, predicted, target_names=bbc_target_names))
                i = 1
                self.ui.datasetNewsText.clear()
                for predicted_item, result in zip(predicted, bbc_y_test):
                    a = str(i) + ") " + training_data.data[i-1] + "\n{ " + result + ' - ' + predicted_item + " }\n"
                    self.ui.datasetNewsText.append(a)
                    i += 1

            elif dataset_model == 2:
                self.ui.datasetReportLabel.setText("Classification Report: BBC News Dataset - Logistic Regression")
                predicted = bbc_model_logistic_reg.predict(bbc_X_test)
                self.ui.datasetReportText.setText(classification_report(bbc_y_test, predicted, target_names=bbc_target_names))
                i = 1
                self.ui.datasetNewsText.clear()
                for predicted_item, result in zip(predicted, bbc_y_test):
                    a = str(i) + ") " + training_data.data[i-1] + "\n{ " + result + ' - ' + predicted_item + " }\n"
                    self.ui.datasetNewsText.append(a)
                    i += 1

            else:
                self.ui.datasetReportLabel.setText("Classification Report: BBC News Dataset - Gradient Boosting Classifier")
                predicted = bbc_model_gradient_boosting.predict(bbc_X_test)
                self.ui.datasetReportText.setText(classification_report(bbc_y_test, predicted, target_names=bbc_target_names))
                i = 1
                self.ui.datasetNewsText.clear()
                for predicted_item, result in zip(predicted, bbc_y_test):
                    a = str(i) + ") " + training_data.data[i-1] + "\n{ " + result + ' - ' + predicted_item + " }\n"
                    self.ui.datasetNewsText.append(a)
                    i += 1

        self.ui.statusbar.showMessage("Classification is complete.", 3000)


app = QApplication([])
window = newsClassification()
window.show()
app.exec_()
