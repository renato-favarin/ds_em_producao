This project was inspired by the "Rossmann Store Sales" challenge published on kaggle (https://www.kaggle.com/c/rossmann-store-sales)

Here is the business scenario: the sales director of the Rossmann stores wants to estimate the sales forecast for the next 6 weeks on its different units spread across Europe 

The resolution of the challenge was carried out following the CRISP (Cross-industry standard process for data mining) methodology, which is a cyclical approach that streamlines the delivery of value.

![crisp_cycle](https://user-images.githubusercontent.com/64495168/129498233-5b3c6cb0-39ce-4187-a96a-c298c7edc95b.png)

The first step was to understand the data; soon after, the cleaning of the database and treatment of missing values began.

The next step was to perform exploratory data analysis (EDA). But right before that, a mind map of hypotheses was made in order to guide EDA, generate insights and understand a little more about the database and the most important attributes. 

![MindMapHypothesis](https://user-images.githubusercontent.com/64495168/129498190-589e0307-4c29-4dd6-b966-52e2b8f8b874.png)

With the feature diagram above, several hypotheses were generated; the ones that were judged to be most relevant were selected (listed below) and then the EDA actually started.

1. Stores with a larger assortment should sell more
2. Stores with closer competitors should sell less
3. Stores with longer-standing competitors should sell more
4. Stores where products cost less for longer (active promotions) should sell more
5. Stores with more promotion days should sell more
6. Stores with more extended promotions should sell more
7 Stores open during Christmas holiday should sell more
8. Stores should sell more over the years
9. Stores should sell more in the second half of the year
10. Stores should sell more after the 10th day of each month
11. Stores should sell less on weekends
12. Stores should sell less during school holidays
13. Stores that open on Sundays should sell more

The discussion of each hypothesis to validate or refute it is in the notebook.
Below are the summary of the analysis of hypotheses 1, 10 and 13.

1. Stores with a larger assortment should sell more: **true**

By plotting the moving average of the last 4 weeks over the entire period available in the database, it is noted the better performance, in sales, of stores that have 'extra' assortment, compared to the 'extended', which in turn is better than the 'basic'.

![h1](https://user-images.githubusercontent.com/64495168/129499183-ac3e5744-0800-49e4-92ef-4ee691a69e99.png)

It is also noted that there seems to be a tendency to accentuate the difference of stores with extra assortment as time goes on.
<br>
<br>

10. Stores should sell more after the 10th day of each month: **false**

After analyzing the composition of the 4 graphs related to this hypothesis, it is noted that before the 10th, on average, stores sell more. 

![h10](https://user-images.githubusercontent.com/64495168/129499532-abffc7db-1f3f-455a-974f-8d2ea11dbefc.png)

The confusion matrix reveals a tendency for the number of the day to be inversely correlated with the sales result.
<br>
<br>

13. Stores that open on Sundays should sell more: **true**

After identifying the different stores that open on Sundays (which most stores remain closed), the average sales performance between these stores that never open on Sundays vs stores that always open on Sundays was compared; to be a fair comparison, the Sunday sales of these last stores were excluded, so only the performance of Monday to Saturday sales was compared.

![h13](https://user-images.githubusercontent.com/64495168/129499835-85cefe88-7648-45c1-94b7-593717869ffa.png)

Perhaps this analysis is an insight and investigation is recommended if such superior performance is due to the location of the store or if the fact that they are open on Sundays makes them gain extra confidence in availability, which makes customers look for such stores regardless of the day of the week.
It may be for yet another factor to be identified.

After EDA, data preparation was performed, where reascaling and encondings were applied.
As a highlight, there is the transformation of cyclical data (such as day and week) using the sine and cosine functions to leave such variables correctly spaced according to the calendar, for example approximating the end and beginning of the month (image below) or the end and the beginning of the year.
 
![sin_cos](https://user-images.githubusercontent.com/64495168/129500567-88f18fe1-d361-4c8d-b070-229b10848abd.png)

The next step was to identify the most relevant features for training machine learning models.
For this, in addition to the knowledge acquired during EDA, the Python implementations of the Boruta R package (https://github.com/scikit-learn-contrib/boruta_py) was used.

Four different models (linear regression, regularized linear regression - lasso, random forest and XGBoost ) were evaluated with the cross-validation technique schematically represented below:

![ts_cross_validation](https://user-images.githubusercontent.com/64495168/129501073-58f20c0c-543d-4d0f-899b-10da4eac3011.png)

The results in terms of Mean Absolute Error (MAE), Mean absolute percentage error (MAPE) and Root Mean Square Error (RMSE) were:

![performance_models](https://user-images.githubusercontent.com/64495168/129501252-1b1e4e47-d7e1-41a1-b307-15efd9495c06.png)

Although the random forest model was the best, the model chosen to go ahead with the tuning of the hyperparameters was XGBoost because, in addition to not having such a significant difference in performance, it is considerably **lighter** to operate in production, an extremely important requirement for this project.
