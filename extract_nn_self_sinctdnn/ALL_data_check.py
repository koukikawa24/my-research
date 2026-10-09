def process_line(line):
    parts = line.split()
    path = parts[0]
    age_gender = parts[1]
    
    # Determine the new first element based on the path
    if path.startswith("id"):
        new_first_element = "AgeVoxCeleb"
    elif path.startswith("rakuten"):
        new_first_element = "Rakuten"
    elif path.startswith("ALAGINChildVoice16kHz"):
        new_first_element = "ALAGIN"
    elif path.startswith("ears_train"):
        new_first_element = "EARS"
    else:
        new_first_element = path.split('/')[0]
    
    # Extract age and gender
    age = age_gender[:-1]
    gender = age_gender[-1]
    
    # Convert gender to full form
    if gender == 'm':
        gender_full = "male"
    elif gender == 'f':
        gender_full = "female"
    
    return f"{new_first_element} {age} {gender_full}"

# Read the input file
with open('/home/s246245/s226059/workspace/data/age_training/wav/utt2aslab_ken3.train', 'r') as infile:
    lines = infile.readlines()

# Process each line and collect the results
processed_lines = [process_line(line) for line in lines]

# Write the processed lines to the output file
with open('/home/s246245/ALL_DATA_CHECK.txt', 'w') as outfile:
    outfile.write("\n".join(processed_lines))

print("The file has been processed and saved as output.txt.")