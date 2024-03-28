# Defining the teams list and their corresponding codes
teams_list = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton and Hove Albion', 'Burnley', 'Cardiff City', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Huddersfield Town', 'Leeds United', 'Leicester City', 'Liverpool', 'Luton Town', 'Manchester City', 'Manchester United', 'Newcastle United', 'Norwich City', 'Nottingham Forest', 'Sheffield United', 'Southampton', 'Tottenham Hotspur', 'Watford', 'West Bromwich Albion', 'West Ham United', 'Wolverhampton Wanderers']

teams_code = list(range(1, len(teams_list) + 1))

# Mapping teams to their codes
team_mapping = dict(zip(teams_list, teams_code))

# Reverse mapping (codes to teams)
reverse_team_mapping = dict(zip(teams_code, teams_list))


# create a mapped list to encode hours from noon to 8 PM
hour_mapping = {
    "12:00": 12,
    "1:00": 13,
    "2:00": 14,
    "3:00": 15,
    "4:00": 16,
    "5:00": 17,
    "6:00": 18,
    "7:00": 19,
    "8:00": 20
}

# Define days of the week and their corresponding codes
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
days_code = list(range(7))


# print the teams list with corresponding codes ask the user to input the code of the team they want to predict
# the match for
# print the list of teams

print('Select a team from the list below:')
for i, team in enumerate(teams_list):
    print(f'{i+1}. {team}')

# ask the user to input the code of the team they want to predict the match for   
     
while True:
    team_code = input('Enter the code of the team you want to predict the match for: ')
    if team_code.isdigit() and 1 <= int(team_code) <= len(teams_list):
        team_index = int(team_code) - 1
        break
    else:
        print("Invalid input! Please enter a valid team code.")

selected_team = teams_list[team_index]
print(f"You've selected {selected_team} with code {team_code}.")



# ask the user to select home or away
while True:
    venue_code = input('Is the match at home or away? (0 for away, 1 for home): ')
    if venue_code.isdigit() and int(venue_code) in [0, 1]:
        venue_code = int(venue_code)
        break
    else:
        print("Invalid input! Please enter either 0 for away or 1 for home.")

if venue_code == 0:
    venue = "Away"
else:
    venue = "Home"

print(f"Selected venue: {venue}")



# ask the user to input the code of the opponent
print('Select an opponent from the list below:')
while True:
    print("Available opponents:")
    for i, team in enumerate(teams_list):
        print(f'{i+1}. {team}')
    opp_code = input('Enter the code of the opponent: ')

    if opp_code.isdigit() and 1 <= int(opp_code) <= len(teams_list):
        opp_index = int(opp_code) - 1
        break
    else:
        print("Invalid input! Please enter a valid opponent code.")

selected_opponent = teams_list[opp_index]
print(f"Selected opponent: {selected_opponent} (Code: {opp_code})")



# ask the user to input the hour of day
print('Select the hour of the match (in 12-hour format between 12:00 and 8:00):')
while True:
    print("Available hours:")
    for hour in hour_mapping:
        print(hour)
        
    selected_hour = input('What time is the match at? (Enter the hour in "HH:MM" format): ')

    if selected_hour in hour_mapping:
        hour = hour_mapping[selected_hour]
        break
    else:
        print("Invalid input! Please enter a valid hour in the format 'HH:MM'.")

print(f"Selected hour: {selected_hour} (Encoded: {hour})")



# ask the user to input the day of the match
print('Select the day of the match:')
while True:
    print("Available days:")
    for i, day in enumerate(days):
        print(f'{i+1}. {day}')
    selected_day_index = input('Enter the number corresponding to the day of the match: ')

    if selected_day_index.isdigit() and int(selected_day_index) in range(1, len(days) + 1):
        selected_day_index = int(selected_day_index)
        day_code = days_code[selected_day_index - 1]
        break
    else:
        print("Invalid input! Please enter a valid day number.")

print(f"Selected day: {days[selected_day_index - 1]} (Code: {day_code})")




user_input = [venue_code, opp_code, hour, day_code, team_code]
print(user_input)