
# News Classification

The purpose of this project is to predict the categories of the news in the data sets and also to predict the category of the news entered by the user as an example and show it to the user.

In the project, BBC News Dataset and News Article Dataset were used as dataset. The details of these datasets are mentioned in the "Datasets" section.

Among the classification models, Multinomial Naive Bayes, Support Vector Machines, Logistic Regression and Gradient Boosting Classifier models were used in the project.

The classification results of the models are shown to the user in detail, both for each category in the data sets and for the entire data set, in the classification report section.

PyQt5 library was used for interface design in the project. The sklearn library was used for classification models.


## Datasets

- `BBC News Dataset` : The dataset consists of a total of 2225 news, as news and related news categories, respectively. The categories of news in the dataset are: business, entertainment, politics, sport, technology.
- `News Article Dataset` : The dataset consists of a total of 32603 news, as news and related news categories, respectively. The categories of news in the dataset are: sport, world, us, business, health, entertainment, science & technology.


##  File Hierarchy

- `/bbc` : There are files belonging to the BBC News Dataset. These files consist of related data set and model files.
- `/news_article` : There are files belonging to the News Article Dataset. These files consist of related data set and model files.


## Usage
When the application is run, two tabs appear that we can view. The first of these is the "Dataset" tab. In this tab, the user can classify news on two data sets in the program with one of the four classification models in the program. The classification report is presented to the user when the user clicks the "Report" button after selecting the data set and classification model. Also, at the bottom, the news in the data set is shown to the user, including the actual classes and the classes predicted by the model.

In the "Text" tab, which is another tab, the user can write a sample news text and find out which category this news text belongs to. The user chooses the data set on which he/she wants to use the classification models trained on the data set. Then, the user writes the text of the news that he wants to know the category of as an example in the text box and clicks the "Find The Category" button. As a result, the result of each classification model for the news text entered in the table below is presented to the user.