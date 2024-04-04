#Dependencies
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def preprocess_match_data(matches):
    """
    Written for the matches data; performs data cleanup, encoding and
    engineering in order to assist the model to perform with the best accuracy.
    """

    #convert date to date-time
    matches["date"] = pd.to_datetime(matches["date"])

    # setting the venue code
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    # 0 = away 
    # 1 = home

    # Create numeric codes for each unique 'opponent' value and store them in a new column 'opp_code'.
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes

    # Create numeric codes for each unique 'team' value and store them in a new column 'team_code'.
    matches["team_code"] = matches["team"].astype("category").cat.codes

    # Extract the hour component from the 'time' column and store it as integers in a new column named 'hour'.
    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

    # Add a new column 'day_code' to the 'matches' DataFrame, containing the day of the week from the 'date' column.
    matches["day_code"] = matches["date"].dt.dayofweek

    #target will be if team won lost or drawn
    conditions = [
        (matches["result"] == "W"),  # Win condition
        (matches["result"] == "L"),  # Lose condition
        (matches["result"] == "D")   # Draw condition
    ]

    values = [1, -1, 0,]  # 1 for win, -1 for lose, 0 for draw
    matches["target"] = np.select(conditions, values, default=np.nan)
    # Apply np.select to create the 'target' column based on the defined conditions and values

    #convert poss from an int to a percentage represented as a float
    matches["poss"] = matches["poss"]/100

    return matches

def calculate_rolling_stats(matches, window):
    """
    This function calculates the rolling stats for Team, and Opponent, and the 
    comparison stats between the team and the opponent.The function takes two 
    arguments: the 'matches' DataFrame and the 'window' size.
    The function returns the 'matches' DataFrame with the calculated stats.
    to change the window size,(number of previous matches to include) change 
    the value of the 'window' argument in the function call.
    """


    # Sort the DataFrame by team and date
    matches = matches.sort_values(['team', 'date'])

    # Calculate rolling statistics for the team
    matches['last_{}_results'.format(window)] = matches.groupby('team')['target'].rolling(window).sum().reset_index(0, drop=True)
    matches['last_{}_gf'.format(window)] = matches.groupby('team')['gf'].rolling(window).sum().reset_index(0, drop=True)
    matches['last_{}_ga'.format(window)] = matches.groupby('team')['ga'].rolling(window).sum().reset_index(0, drop=True)
    matches['last_{}_avg_poss'.format(window)] = matches.groupby('team')['poss'].rolling(window).mean().reset_index(0, drop=True)
    matches['last_{}_avg_sot'.format(window)] = matches.groupby('team')['sot'].rolling(window).mean().reset_index(0, drop=True)
    
    # Sort the DataFrame by opponent and date
    matches = matches.sort_values(['opponent', 'date'])

    # Calculate rolling statistics for the opponent
    matches['opp_last_{}_results'.format(window)] = matches.groupby('opponent')['target'].rolling(window).mean().reset_index(0, drop=True)
    matches['opp_last_{}_gf'.format(window)] = matches.groupby('opponent')['gf'].rolling(window).mean().reset_index(0, drop=True)
    matches['opp_last_{}_ga'.format(window)] = matches.groupby('opponent')['ga'].rolling(window).mean().reset_index(0, drop=True)
    matches['opp_last_{}_avg_poss'.format(window)] = matches.groupby('opponent')['poss'].rolling(window).mean().reset_index(0, drop=True)
    matches['opp_last_{}_avg_sot'.format(window)] = matches.groupby('opponent')['sot'].rolling(window).mean().reset_index(0, drop=True)

    # Calculate the difference between team and opponent stats
    matches['last_{}_gd'.format(window)] = matches['last_{}_gf'.format(window)] - matches['last_{}_ga'.format(window)]
    matches['opp_last_{}_gd'.format(window)] = matches['opp_last_{}_gf'.format(window)] - matches['opp_last_{}_ga'.format(window)]
    matches['last_{}_gd_diff'.format(window)] = matches['last_{}_gd'.format(window)] - matches['opp_last_{}_gd'.format(window)]
    matches['last_{}_avg_poss_diff'.format(window)] = matches['last_{}_avg_poss'.format(window)] - matches['opp_last_{}_avg_poss'.format(window)]
    matches['last_{}_avg_sot_diff'.format(window)] = matches['last_{}_avg_sot'.format(window)] - matches['opp_last_{}_avg_sot'.format(window)]

    # This function will fill the NaN values in the dataframe columns with the median value of those columns

    columns_to_fill = [
        'last_3_results', 'opp_last_3_results',
        'last_3_gf', 'opp_last_3_ga',
        'last_3_gd', 'opp_last_3_gd',
        'last_3_gd_diff', 'last_3_ga',
        'opp_last_3_gf', 'opp_last_3_avg_poss',
        'last_3_avg_poss', 'last_3_avg_poss_diff',
        'opp_last_3_avg_sot', 'last_3_avg_sot',
        'last_3_avg_sot_diff','last_3_gd',
    ]

    for column in columns_to_fill:
        median_value = matches[column].median()
        matches[column].fillna(median_value, inplace=True)

    return matches

def create_rf_model(matches):
    """
    This function is used with the matches data to define X and y and then 
    split the data into training and testing data.
    """

    # Define the list of features (predictors) including venue code, opponent code, hour, and day code.
    predictors = ["venue_code", "opp_code", "hour", "day_code", 'team_code',
    'last_3_results',
    'last_3_gf',
    'last_3_ga',
    'last_3_avg_poss',
    'opp_last_3_results',
    'opp_last_3_gf',
    'opp_last_3_ga',
    'opp_last_3_avg_poss',
    'last_3_gd',
    'opp_last_3_gd',
    'last_3_gd_diff',
    'last_3_avg_poss_diff',]

    # Define the features (predictors) and the target variable
    X = matches[predictors]
    y = matches["target"]

    # Define a variable that stores the feature names
    feature_names = X.columns

    # Split the data into training and testing sets with a ratio of 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    # Initialize a Random Forest classifier with 50 trees, minimum samples split of 10, and a fixed random state.
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, random_state=1)

    # Train the Random Forest classifier on the training data using specified predictors.
    rf_model = rf_model.fit(X_train, y_train)

    # Save model for use in other files
    joblib.dump(rf_model, "rf_model.joblib")

    # Generate predictions using the trained Random Forest classifier on the test data using specified predictors.
    # Train the Random Forest classifier on the training data using specified predictors.
    test_predictions = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    train_predictions = rf_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    # Create a contingency table showing the counts of actual versus predicted labels
    contingency_table = pd.DataFrame({"actual": y_test, "prediction": test_predictions})
    contingency_table = pd.crosstab(index=contingency_table["actual"], columns=contingency_table["prediction"])

    # Generate classification report
    class_report = classification_report(y_test, test_predictions)

    return (rf_model, test_accuracy, train_accuracy, contingency_table, class_report, feature_names)

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")