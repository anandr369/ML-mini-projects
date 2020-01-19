Dataset consists of 1338 records. Each record contains the below data for specific person.

age – Age of the person
sex – Sex of the person
bmi – Body Mass Index(BMI) of the person
children – Number of children for the person
smoker – Smoking status of the person
region – Region of the person in US
charges – Medical Insurance costs per year for the person

insurancepriceMLR.py description :
We have chosen the following attributes for regression, 
age,bmi,children,smoker_no,region_northeast
based on the statistical significance of them by finding p-values through gretl application and applied multiple linear regression.

insurancepriceRFR.py description :
We have chosen the following attributes for regression, 
age,bmi,children,smoker_no,region_northeast
based on the statistical significance of them by finding p-values through gretl application and applied random forest regressor.

all-inMLR.py description :
We have opted for the all-in model building technique so included all the attributes to predict the dependent variable using random forest regressor.

