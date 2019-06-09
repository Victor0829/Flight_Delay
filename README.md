# Flight_Delay
2018 Flight Data is fetched on https://www.transtats.bts.gov/DL_SelectFields.asp?DB_Short_Name=On-Time&Table_ID=236.

Weather data in different airports exisiting above for 365 days in 2018 is crawled on https://www.worldweatheronline.com/.

The project analyzes and predicts the flight delays in the United States by several prevalent supervised machine learning methods. All domestic flight data and weather data among those airports are fetched on BTS and World Weather Online respectively. 10-fold Cross Validation is performed for various classifiers, including SVM, Random Forest, Naive Bayes, K-NN, Neural Network and Logistic Regression. The overall accuracy for most algorithms is more than 80%. Meanwhile, F1-score and AUC are checked for each algorithm and perform well. 
