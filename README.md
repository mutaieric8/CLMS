IFRS 17 Insurance Contracts Classification Tool.
1. **Introduction**

IFRS 17 is an International Financial Reporting Standard that was issued by the International Accounting Standards Board in May 2017. It will replace IFRS 4 on accounting for insurance contracts and has an effective date of 1 January 2023 [Wikipedia defination](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjBycbxksf-AhXgTEEAHdvQDtUQFnoECA0QAw&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FIFRS_17&usg=AOvVaw1EwBMcb05JafukK914fuee).
This standard requires that insurance companies classify their contracts as either profit making(Non-Onerous) or loss making(Onerous) from the time of inception. 
This necessitates insurance companies to come up with actuarial models that will enable them make this classification. 

Having been part of the implementation team for the new standard, it challenged me to find ways of coming up with a tool that will enable the organization I work for to make this classification by a click of a button.

I then embarked on exploring ways to use machine learning knowledge I gained from ADS, CRISP-DM framework, to train a model using our historical premiums and claims data to calculate the loss ratios for each policy, and depending on the value of the loss ratio, mark the policy as either onerous or non-onerous.

2. **Data Collection**

I used my well versed SQL skills to pull the data in the format I needed. Concious of the data protection laws, I ensured I did not extract any personally Identifiable data. I picked the features I thought were relevant to train a model such as member age, height, weight, claims incurred and the premium paid.

3. **Exploratory data Analysis**

Upon extracting my data, I did a little analysis to check on the column data types and also to check on any occurences of null data. I used pandas methods such as head and tail, as well as info() and describe, and dtypes. Given my data was custom data that I built the query for and pulled manually, there was no missing values. Most columns had the relevant datatypes as well. 

I also utilized seaborn's relplot to check on the comparative distribution of paired features. This is where I noticed the possible inaccuracy of the data in our system given the relationships between the different features did not support the popular believe in the industry. For example, comparison of policy holders average age and the claims did not show any pattern in relationship, as this is normally expected to have some linear relationship. This was the case with other parameters such as average height and weight of the family members against the claims. This pointed to possible incompletion of data during capturing in our system.

4. **Data Preparation**

On data preparation, I used seaborns boxplots to check for any outliers in my data. Outliers were detected on Claims amounts, Average weight and Basic Premium amount. I then applied a formula to calculate the upper and lower limits, using mean and standard deviation, and multiplying by a factor, and applied on my data to remove the outliers. This is demonstrated on the jupiter notebook.
I also did scaling of my data using sklearns MinMax() scaling function. This transformation was however not utilized as on evaluating my model, the performance seemed slightly better without the scaling than when I scaled my data, albeit by a small margin.
I also did log tranform of the claims amounts column, just to have normal distributon of data, as it was skewed upon testing using plotly histogram. This however was an inconsequential transformation as the claims amounts variable could not be used during training as I had already used this value to feature engineer the target column.

On encoding, I encoded my Target column using label encoding by first converting it from object datatype to category datatype. After encoding, Non-Onerous (profit making) was encoded as value 0 and Onerous (loss making) as value 1. 

5. **Model Training and Selection**

I used a correlation matrix method and a heatmap to test for correlation between the feature variables and the target variable. There was a negative correlation between Average Age and the Target column, which was expected as the claims are expected to go up as age advances as the member is more likely to get ill with old age hence the claims will go up, increasing the possibility of the business being onerous, loss making. There was also a negative correlation between Basic Premium and the target column, which is also expected as a lower premium means lower benefits limits hence high likelihood of the member exhausting their benefits.

There was a positive but insignificant correlation between Average weight and the target column as well as between average weight and the target column, which implies that as the Weight or Height increases, the business will tend to be Non-Onerous (0), which is contrary to general believe. This means height and weight is not captured well in the business system. 

The Onerous (1) and Non-Onerous (0) were encoded using label encoding.

On modeling, given I was handling a classification problem, I chose to use sklearn's ensemble model RandomForestClassifier. For better results, I used a function to test for the best estimator to use, as the choice of parameter, which showed 150 as the better estimator value. I also used TrainTestSplit function to split my data into the training and testing sets, on a ratio 0f 7:3.

I then trained my model using the training set. After training, I used the feature test set to do a prediction. On evaluation, the results were not that impressive as it returned a score of 65.93% on test data and 96.96% on training set.

I further evaluated the results using sklearns metrics, classification_report, and the values of Precision, recall and f1 score were 67, 70 and 68 respectively for target value 0 AND 65, 61 and 63 for target value 1.

These evaluations pointed to one fact that our data is not accurate. Most of the KYCs may not be sitting well, as most likely the weights and heights are not actual measurements. With this knowledge, I will advise the business accordingly. 
Given I am working with existing data, I decided to proceed with it the way it is as I needed to complete my project nonetheless.

I then saved my model as a pickle file ready for use by applications

6. **Deployment and Monitoring**

On deployment, I did the UI part using python, generating a form that enables users to key in the values of all the feature variables that were used to train the model. The form has a button that once clicked, depending on the values keyed in, will determine whether the business the company is about to onboard will be Onerous or Non-Onerous at the end. This will aid in meeting the IFRS 17 requirement of having to classify a contract as either Onerous or Non Onerous at inception, for tracking to the contract end.

I uploaded the trained model to reside in the same location as the application file(UI) to enable loading directly without the need to host as an API service. The application is hosted as a streamlit app, and I have added the URL link as the website address on the Github public repository.
