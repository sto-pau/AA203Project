import pandas as pd
import random
import numpy as np

# Read the CSV file into a DataFrame
name = 'data_short'
file_name = name + '.csv'
df = pd.read_csv(file_name)

# Add the header row
header = ['x0', 'x1', 'y0', 'y1', 'v', 'a', 'r']
df.columns = header

added_columns = ['x0p', 'x1p', 'y0p', 'y1p', 'vp']

for next_state_element in added_columns:
    df[next_state_element] = np.nan

# Iterate over the rows in the DataFrame
for row_index in range(len(df) - 1):
    # Append 5 numbers to the end of a specific row
    next_row_first_5 = df.iloc[row_index+1, :5].values #next state to be appended above

    df.loc[row_index, 'x0p':'vp'] = next_row_first_5

    # # Get the first 5 numbers from the next row
    # next_row_first_5 = df.iloc[i+1, :5].values
    
    # # # Append the first 5 numbers to the current row
    # # df.iloc[i, :] = df.iloc[i, :].append(pd.Series(next_row_first_5))
    # # Append the first 5 numbers to the current row
    # # df.iloc[i, :] = pd.concat([df.iloc[i, :], pd.Series(next_row_first_5)], ignore_index=True)
    # # Append the first 5 numbers to the current row
    # df.loc[i, 'x0':'r'] = pd.Series(next_row_first_5, index=df.loc[i, 'x0p':'rp'].index)

# Remove the last row from the DataFrame since it doesn't have a next row
df = df.iloc[:-1, :]

# Shuffle the rows
shuffled_df = df.sample(frac=1, random_state=random.seed())

# Reset the index of the shuffled DataFrame
shuffled_df.reset_index(drop=True, inplace=True)

#write to new file to be used in training
# Write the DataFrame to a temporary file with the header
new_name = name + '_train.csv'
shuffled_df.to_csv(new_name, index=False)

