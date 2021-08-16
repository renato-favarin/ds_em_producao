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

1. Stores with a larger assortment should sell more:

By plotting the moving average of the last 4 weeks over the entire period available in the database, it is noted the better performance, in sales, of stores that have 'extra' assortment, compared to the 'extended', which in turn is better than the 'basic'.

![h1](https://user-images.githubusercontent.com/64495168/129499183-ac3e5744-0800-49e4-92ef-4ee691a69e99.png)

It is also noted that there seems to be a tendency to accentuate the difference of stores with extra assortment as time goes on.
