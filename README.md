# Fake_News_Detection
<h1> Objective: </h1>
<p> The objective of this project is to develop an intelligent system that can automatically classify news articles as real or fake using machine learning techniques. By leveraging Natural Language Processing (NLP) and classification algorithms, this project aims to: </p>
<li>1. Detect and classify misleading news articles with high accuracy.</li>
<li>2. Improve trustworthiness of online content by flagging fake news.</li>   
<li>3. Enhance feature extraction using advanced NLP techniques such as TF-IDF, Word2Vec, and N-grams.</li>
<li>4. Optimize model performance through hyperparameter tuning and comparative analysis of different classifiers.</li>
<li>5. Provide a user-friendly interface for real-time fake news detection.</li>
<h2> Prerequisites: </h2>
<p>To run this project, ensure you have the following installed:</p>
<h2>1. Python 3.6+ </h2>
<h2>2. Alternative:</h2> Anaconda 
Install from Anaconda Official Site
Use the Anaconda Prompt to run the project
<h2>3. Required Libraries</h2>
Run the following commands to install dependencies:
<ul>
  <li>For Python (pip) Users:</li>
  
    pip install -U scikit-learn numpy scipy pandas nltk
    
  <li>For Anaconda Users:</li>
      
    conda install -c scikit-learn
    conda install -c anaconda numpy scipy pandas nltk
</ul>
<h2> File descriptions </h2>

#### DataPrep.py
This file contains all the pre processing functions needed to process all input documents and texts. First we read the train, test and validation data files then performed some pre processing like tokenizing, stemming etc. There are some exploratory data analysis is performed like response variable distribution and data quality checks like null or missing values etc.

#### FeatureSelection.py
In this file we have performed feature extraction and selection methods from sci-kit learn python libraries. For feature selection, we have used methods like simple bag-of-words and n-grams and then term frequency like tf-tdf weighting. we have also used word2vec and POS tagging to extract the features, though POS tagging and word2vec has not been used at this point in the project.

#### classifier.py
Here we have build all the classifiers for predicting the fake news detection. The extracted features are fed into different classifiers. We have used Naive-bayes, Logistic Regression, Linear SVM, Stochastic gradient descent and Random forest classifiers from sklearn. Each of the extracted features were used in all of the classifiers. Once fitting the model, we compared the f1 score and checked the confusion matrix. After fitting all the classifiers, 2 best performing models were selected as candidate models for fake news classification. We have performed parameter tuning by implementing GridSearchCV methods on these candidate models and chosen best performing parameters for these classifier. Finally selected model was used for fake news detection with the probability of truth. In Addition to this, We have also extracted the top 50 features from our term-frequency tfidf vectorizer to see what words are most and important in each of the classes. We have also used Precision-Recall and learning curves to see how training and test set performs when we increase the amount of data in our classifiers.

#### prediction.py
Our finally selected and best performing classifier was ```Logistic Regression``` which was then saved on disk with name ```final_model.sav```. Once you close this repository, this model will be copied to user's machine and will be used by prediction.py file to classify the fake news. It takes an news article as input from user then model is used for final classification output that is shown to user along with probability of truth.

  
