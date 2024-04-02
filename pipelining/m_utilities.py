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

def user_inputs_to_df(venue_code, opp_code, hour, day_code, team_code):
    user_input = {"venue_code": [venue_code],
                  "opp_code": [opp_code],
                  "hour": [hour],
                  "day_code": [day_code],
                  "team_code": [team_code]}
    user_input_df = pd.DataFrame(user_input)
    print("Here is your input:")
    print(user_input_df)
    confirm = input("Are you happy with your selections? (y/n): ").lower()

    if confirm == "y":
        print("User input saved.")
        return user_input_df
    else:
        print("User input was not saved.")

def user_input_prediction(user_inputs):
    """
    This function takes in the user input dataframe and adds in the rest of the team stats and features to pass
    to the model for predictions.
    """

    #load in dataset for current team data
    matches = pd.read_csv("epl_matches.csv", index_col=0) 
    
    team_1_columns = ['last_3_results',
                      'last_3_gf',
                      'last_3_ga',
                      'last_3_avg_poss']


    team_2_columns= ['last_3_results',
                     'last_3_gf',
                     'last_3_ga',
                     'last_3_avg_poss']

    opponent_columns_heading = ['opp_last_3_results',
                                'opp_last_3_gf',
                                'opp_last_3_ga',
                                'opp_last_3_avg_poss']
    
    # Defining the teams list and their corresponding codes
    teams_list = ['Arsenal', 'Aston Villa',
                  'Bournemouth', 'Brentford', 'Brighton and Hove Albion', 'Burnley',
                  'Cardiff City', 'Chelsea', 'Crystal Palace',
                  'Everton',
                  'Fulham',
                  'Huddersfield Town',
                  'Leeds United', 'Leicester City', 'Liverpool', 'Luton Town',
                  'Manchester City', 'Manchester United',
                  'Newcastle United', 'Norwich City', 'Nottingham Forest',
                  'Sheffield United', 'Southampton',
                  'Tottenham Hotspur',
                  'Watford', 'West Bromwich Albion', 'West Ham United', 'Wolverhampton Wanderers'
                  ]
    
    # Assuming is your DataFrame, 'team_name' is the name of the team, and 'team' is the column with team names
    # subtract 1 from the team code in the User_input dataframe to get the team name from the teams_list and store as a variable
    team_1_name = teams_list[user_inputs["team_code"].values[0] - 1]
    team_1_data = matches.loc[matches['team'] == team_1_name, team_1_columns]
    # subtract 1 from the opp_code in the User_input dataframe to get the team name from the teams_list and store as a variable
    team_2_name = teams_list[user_inputs["opp_code"].values[0] - 1]
    team_2_data = matches.loc[matches['team'] == team_2_name, team_2_columns]
    # Get the last row
    last_values_1 = team_1_data.iloc[-1]
    last_values_2 = team_2_data.iloc[-1]

    #create a new DataFrame with the last values 1 using the 'team_1_columns' as column names
    team_1_last_values = pd.DataFrame(last_values_1.values.reshape(1, -1), columns=team_1_columns)
    #display(team_1_last_values)
    #create a new DataFrame with the last values 2 using the 'opponent_column_headings' as column names
    team_2_last_values = pd.DataFrame(last_values_2.values.reshape(1, -1), columns=opponent_columns_heading)
    #display(team_2_last_values)
    #create a new dataframe with the user input using the 'user_input_columns' as column names
    user_input_df = pd.DataFrame(np.array(user_inputs).reshape(1, -1), columns=user_inputs.columns)
    #display(user_input_df)
    
    #concatenate the 'user_input_df''team_1_last_values', and 'team_2_last_values', DataFrames along the columns
    combined_df = pd.concat([user_input_df, team_1_last_values, team_2_last_values], axis=1)
    
    combined_df = calculate_differentials(combined_df, 3)
    
    return combined_df

#Calculate the comparison stats and add them to the dataframecomparison Stats
def calculate_differentials(combined_df, window):
    # Calculate the stat difference between the team and the opponent
    combined_df['last_{}_gd'.format(window)] = combined_df['last_{}_gf'.format(window)] - combined_df['last_{}_ga'.format(window)]
    combined_df['opp_last_{}_gd'.format(window)] = combined_df['opp_last_{}_gf'.format(window)] - combined_df['opp_last_{}_ga'.format(window)]
    combined_df['last_{}_gd_diff'.format(window)] = combined_df['last_{}_gd'.format(window)] - combined_df['opp_last_{}_gd'.format(window)]
    combined_df['last_{}_avg_poss_diff'.format(window)] = combined_df['last_{}_avg_poss'.format(window)] - combined_df['opp_last_{}_avg_poss'.format(window)]

    return combined_df

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")