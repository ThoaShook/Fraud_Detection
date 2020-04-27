##### Fraud_Detection
1. Objective:
To identify fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
To find new fraud patterns

2. About The Data:
The datasets contains transactions made by credit cards in September 2013.These transactions occured in two days, and there are 429 frauds out of 284,807 transactions.
PCA (Principal Component Analysis)- an unsupervised, non-parametric statistical technique is used for dimensionality reduction in machine learning. Features from v1 - v28 are the results of this PCA
Other features (except 'Time' and 'Amount') are anonymized to protect the privacy of the customers.
Feature 'Time' is the time elapsed between each transaction and the first transaction
Feature 'Amount' depicts the amount of each transaction
Feature 'Class' is the dependent variable/target variable. 1 means fraudulent, and 0 means genuine.

3. Solutions:
 * Labeled data are based on historical experience, and it is hard to find unseen fraud patterns by using labeled data. Since unsupervised learning has no past knowledge, we can use unsupervised learning methods to find new fraud patterns
 * If an upcoming credit card transaction is not accepted by the trained HMM with high probability, it would be considered as fraud. Supervised and unsupervised machine learning techniques were combined to detect credit frauds. The results showed that the hybrid technique is efficient and could improve the accuracy of detection
system for anomaly detection should NOT be a supervised ML algorithm as it will (maybe) learn only anomalies it has seen during training. The true magic lies in being able to identify an anomaly never seen before
 * As the data is very skewed - there are only 0.17% fraudulent transactions in the 280k samples - accuracy is not a good metric: any "model" predicting ALL are normal transactions will have a 99.83% accuracy.
Use Recall, Precision and their prodigy (harmonic mean) - the F1 score. Try to optimize each model's hyperparameters for the best F1.
 * The models below do not take into account the time sequences, (while still having the time as a separate feature).The time series nature of the anomaly detection should be dealt with RNN or LSMT or etc. - maybe another notebook.
The training set does NOT include any Fraud, so when the model is exposed to one in Test, it will stand out from the normal transactions. Try dividing the Fraud half into a Validation subset and half in Test - F1 score being lower.

I. Feature Distributions:

![](images/histogram1.png)
![](images/histogram2.png)

* Annotations:
    * There are two columns in the 'Amount' feature. We can assume that the tall - dominant column is associated with the valid transactions whereas the very low-barely notice column is corresponding with the fraudulent transactions
    * There should be two columns in the 'Class' feature. The number of frauds isn't showed due to its significantly small compare to the number of valid transactions
    * Time is measured in seconds. The parabolas shape indicates many transactions occur at the same time
Some of the Vs - PCA components have some what skew and bell shape. The x variable expands in both negative and postive regions

II. Correlation Heatmap

![](images/heatmap.png)

* Annotations:
    * None of the V1 to V28 has any correlation to each other
    * 'Class' don't have any correlation with 'Amount' or 'Time', but it does have some form of positive and negative (mostly low negative) correlations with the 'V' components
    * 'Amount' does have some high positive correlations with V7 and V20, and high negative correlations with the V2 and V5 components
    * We can safely focus on the three main features: Class, Amount, and Time

III. Fraud vs Not-Fraud Distribution

![](images/classDistribution.png)

* Annotations:
    * There are 17 fraud transactions out of 8000 transactions during 2 consecutive days in September.
    * This highly unbalance with respect to target variable 'Class' needs to be treated before any further data processing.
    
IV. Fraudulent Transactions Trend

![](images/fraudulentTrend.png)

V. Valid Transactions Trend

![](images/validTrend.png)


