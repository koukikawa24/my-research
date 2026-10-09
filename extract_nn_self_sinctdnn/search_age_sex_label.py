def get_age_gender_variations(txt_file):
    age_gender_combinations = {}
    age_gender_list = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) == 2:
                age_gender = parts[1]
                if age_gender in age_gender_combinations:
                    age_gender_combinations[age_gender].append(line.strip())
                else:
                    age_gender_combinations[age_gender] = [line.strip()]
                if age_gender not in age_gender_list:
                    age_gender_list.append(age_gender)
    return age_gender_combinations, age_gender_list

txt_file = '/home/s246245/s226059/workspace/data/age_training/wav/utt2aslab_ken2.train'
age_gender_variations, age_gender_list = get_age_gender_variations(txt_file)

print("Age and gender combinations found in the TXT file:")
for age_gender, lines in age_gender_variations.items():
    print(f"{age_gender}: {len(lines)} instances")

print("\nList of age and gender combinations:")
print(age_gender_list)
print(len(age_gender_list))
