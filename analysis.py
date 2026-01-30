
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PART 1: Data Loading & Exploration
# =========================

data = pd.read_csv('can_stats.csv')

print("First rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

# =========================
# PART 2: Descriptive Statistics
# =========================

print("\nAverage goals scored:")
print(data['Goals_Scored'].mean())

print("\nTotal matches played:")
print(data['Matches_Played'].sum())

top_scorer = data.loc[data['Goals_Scored'].idxmax()]
print("\nTop scoring team:")
print(top_scorer[['Team', 'Goals_Scored']])

best_defense = data.loc[data['Goals_Conceded'].idxmin()]
print("\nBest defense team:")
print(best_defense[['Team', 'Goals_Conceded']])

data['Goal_Difference'] = data['Goals_Scored'] - data['Goals_Conceded']
print("\nGoal difference:")
print(data[['Team', 'Goal_Difference']])

# =========================
# PART 3: Ranking & Comparison
# =========================

print("\nRanking by wins:")
data_sorted = data.sort_values(by='Wins', ascending=False)
print(data_sorted[['Team', 'Wins']])

print("\nRanking by goal difference:")
data_sorted_gd = data.sort_values(by='Goal_Difference', ascending=False)
print(data_sorted_gd[['Team', 'Goal_Difference']])

undefeated_teams = data[data['Losses'] == 0]
print("\nUndefeated teams:")
print(undefeated_teams[['Team', 'Wins', 'Draws']])

early_eliminated_teams = data[data['Matches_Played'] < 5]
print("\nEarly eliminated teams:")
print(early_eliminated_teams[['Team', 'Matches_Played']])

# =========================
# PART 4: Correlation Analysis
# =========================

correlation_possession = data['Possession'].corr(data['Goals_Scored'])
correlation_shots = data['Shots_On_Target'].corr(data['Goals_Scored'])

print("\nCorrelation Possession vs Goals:", correlation_possession)
print("Correlation Shots on Target vs Goals:", correlation_shots)

if abs(correlation_possession) > abs(correlation_shots):
    print("Possession has a stronger influence on goals.")
else:
    print("Shots on target has a stronger influence on goals.")

# =========================
# PART 5: Visualizations
# =========================

# Bar chart
plt.figure(figsize=(10,6))
plt.bar(data['Team'], data['Goals_Scored'])
plt.xlabel('Team')
plt.ylabel('Goals Scored')
plt.title('Goals Scored by Each Team')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
plt.boxplot(data['Possession'], vert=False)
plt.xlabel('Possession (%)')
plt.title('Distribution of Possession')
plt.show()

# Heatmap
plt.figure(figsize=(10,8))
correlation_matrix = data.select_dtypes(include='number').corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# =========================
# PART 6: Interpretation
# =========================

average_possession_wins = data[data['Wins'] > 0]['Possession'].mean()
average_possession_losses = data[data['Losses'] > 0]['Possession'].mean()

print("\nAverage possession (winning teams):", average_possession_wins)
print("Average possession (losing teams):", average_possession_losses)

if average_possession_wins > average_possession_losses:
    print("Higher possession tends to be associated with winning.")
else:
    print("Possession does not guarantee victory.")

# Efficiency
data['Efficiency'] = data['Goals_Scored'] / data['Shots_On_Target']
most_efficient_team = data.loc[data['Efficiency'].idxmax()]

print("\nMost efficient team:")
print(most_efficient_team[['Team', 'Efficiency']])

print("\nFinal Conclusion:")
print("The analysis shows that while possession and shots on target influence performance, "
      "the most important factor is efficiency in converting opportunities into goals.")
