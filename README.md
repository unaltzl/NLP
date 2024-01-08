#####################
#FOR NLP1
#####################
Sentiment Analysis for Amazon Reviews
Business Problem
Kozmos aims to boost sales by analyzing and improving product features based on customer sentiment in Amazon reviews.

Dataset
The dataset includes reviews, titles, star ratings, and helpful votes.

Review: Customer feedback.
Title: Short review title.
Helpful: Number of users finding the review helpful.
Star: Product rating.
Tasks
Text Preprocessing:

Convert to lowercase, remove punctuation.
Eliminate numerical expressions.
Remove stopwords and low-frequency words.
Apply lemmatization.
Text Visualization:

Barplot: Visualize word frequencies.
WordCloud: Generate a word cloud.
Sentiment Analysis:

Use SentimentIntensityAnalyzer for polarity scores.
Label sentiment based on compound scores.
Machine Learning Preparation:

Train-test split.
Convert text to numerical form using TfidfVectorizer.
Modeling (Random Forest):

Build and fit a Random Forest Classifier.
Cross-validation for mean accuracy.


###############
FOR NLP2 
###############
Wikipedia Data Cleaning and Text Analysis
Introduction
This project aims to clean and analyze Wikipedia data. The text in the dataset is processed by converting letters to lowercase, removing punctuation, eliminating numerical expressions, removing English stop words, handling low-frequency words, and applying lemmatization. Additionally, the analysis includes visualizations such as barplots for word frequencies and a WordCloud.

Data
The dataset is sourced from the "wiki_data.csv" file, containing text from Wikipedia pages.

Text Cleaning Operations

Convert letters to lowercase.
Remove punctuation marks.
Eliminate numerical expressions.
Remove English stop words.
Remove low-frequency words (less than 1500 occurrences).
Apply lemmatization.

Text Analysis and Visualization
Create a barplot for word frequencies.
Generate a WordCloud for visual representation.

Functionality
A function named data_prep has been created to encapsulate the data cleaning and analysis processes.
This function performs text cleaning, word frequency analysis, and provides optional visualizations such as barplots and WordCloud.
