# English Premier League Match Results Prediction Model
## Introduction
> This project leverages machine learning algorithms and data analysis techniques to predict EPL match results (wins, losses, and draws) based on historical match data, team stats, and other factors. This code is intended for sports analysts, EPL fans seeking to gain insights into match outcomes and team performance, and other data scientists exploring the application of machine learning in sports analytics and predictive modeling. 

 Please note: This model is not intended to influence betting behaviors. Any use of the model for such purposes is entirely at the user's discretion and risk.

## Project Team
 - Abdul Dawson
 - Casey Clayton
 - Cassandra Griffin
 - Cody Schneggenburger
 - Dylan Ross
 - Shantesh Dalal

## Key Features
 - **Machine Learning Models:** Utilizes Random Forest and Gradient Boosting to predict match outcomes accurately.

 - **Feature Engineering:** The Project incorporates feature engineering techniques, including missing value imputation, datetime features, encoding, and feature interactions to enhance the predictive effectiveness of the models. 

 - **Evaluation Metrics:** The models' performance is assessed using evaluation metrics such as accuracy, precision, recall, and F1-score.

 - **Data Preprocessing:** The data set undergoes several preprocessing methodologies to clean, normalize, and transform it in preparation for modeling.

 - **External Factors:** External factors, including weather conditions, player injuries, and other team dynamics, were not used in the current models but are planned for future iterations.

## Data Collection
 The data was pulled from "https://fbref.com/en/comps/9/Premier-League_Stats" using Python's BeautifulSoup web scraper. The scraped data is stored in the [epl_matches.csv](Presentation/Resources/epl_matches.csv) file and references the Scores and Fixtures table from 2018 to the present.

### Features

The most important features were identified and grouped in order of importance.

![image](https://github.com/cschneggenburger/group-2-project-2/assets/152223124/ebd7a16c-4ca1-4b96-bd1d-3dd4838623ac)


## Installation

The project begins with the [scraper.ipynb](Presentation/scraper.ipynb) file using Python's BeautifulSoup library. 

The scraper file requires the following dependencies:

```
import requests
import pandas as pd
from bs4 import BeautifulSoup

```
The resulting [epl_matches.csv](Presentation/Resources/epl_matches.csv) file is stored in the [Resources](Presentation/Resources) folder.

The [matches_starter_code.ipynb](Presentation/matches_starter_code.ipynb) file contains the code to structure the DataFrame. 

The starter code file requires the following dependencies:

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

```

The end result of this code is the [epl_matches_combined.csv](Presentation/Resources/epl_matches_combined.csv) file also located in the [Resources](Presentation/Resources) folder.

The [rf_model_rolling_3_new_data_structure.ipynb](Presentation/rf_model_rolling_3_new_data_structure.ipynb) file contains the Random Forest Classifier and the Gradient Boosting predictive models.

The predictive model file requires the following dependencies:

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

```

## Model Code Snippets

### Random Forest Classifier

```
predictors = ['venue_code', 'opp_code', 'hour', 'day_code','team_code','last_3_results','last_3_results_2', 'xg', 'xga', 'xg_2', 'xga_2' ]
X = matches[predictors]
y = matches["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, random_state=1)
rf_model = rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
train_accuracy = rf_model.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)
test_accuracy = rf_model.score(X_test, y_test)
print("Testing Accuracy:", test_accuracy)

```
**This model produced a Training Accuracy of 85.99% and a 
Testing Accuracy of 73.58%**

### Gradient Boosting Classifier

```
gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=1)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)
gb_preds_train = gb_model.predict(X_train)
gb_preds_train = gb_preds_train.reshape(-1, 1)
gb_preds = gb_preds.reshape(-1, 1)
boost_train_accuracy = gb_model.score(X_train, y_train)
print("Training Accuracy:", boost_train_accuracy)
boost_test_accuracy = gb_model.score(X_test, y_test)
print("Testing Accuracy:", boost_test_accuracy)

```

**The Gradient Boosting model produced a Training Accuracy of 76.21% and a
Testing Accuracy of 74.96%**

## Next Steps

Future iterations of this model will explore additional factors such as weather conditions, player injuries, red cards, and team dynamics. An interactive user input component is also in production.






