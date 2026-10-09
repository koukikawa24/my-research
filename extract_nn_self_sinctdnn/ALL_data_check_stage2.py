import pandas as pd

# Read the input file
with open('/home/s246245/s226059/workspace/self_sinctdnn/ALL_DATA_CHECK.txt', 'r') as infile:
    lines = infile.readlines()

# Initialize a dictionary to store the counts
counts = {
    'male': {age: {'AgeVoxCeleb': 0, 'Rakuten': 0, 'ALAGIN': 0, 'EARS': 0} for age in range(101)},
    'female': {age: {'AgeVoxCeleb': 0, 'Rakuten': 0, 'ALAGIN': 0, 'EARS': 0} for age in range(101)}
}

# Process the data
for line in lines:
    parts = line.split()
    dataset = parts[0]
    age = int(parts[1])
    gender = parts[2]
    
    counts[gender][age][dataset] += 1

# Create a DataFrame for each gender
male_df = pd.DataFrame.from_dict(counts['male'], orient='index')
female_df = pd.DataFrame.from_dict(counts['female'], orient='index')

# Write the DataFrames to an Excel file
with pd.ExcelWriter('/home/s246245/s226059/workspace/self_sinctdnn/ALL_DATA_CHECK.xlsx') as writer:
    male_df.to_excel(writer, sheet_name='Male')
    female_df.to_excel(writer, sheet_name='Female')

print("The data has been processed and saved to output.xlsx.")