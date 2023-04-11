from typing import Optional
import pandas as pd
import re
from word2number import w2n
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def main():
    # Read the CSV file into a DataFrame object
    df = read_csv_file()
    #  Extract the information from the DataFrame object and add the information to the DataFrame object
    df = extract_information(df)
    # Write the DataFrame object to a CSV file
    write_csv_file(df)


def read_csv_file() -> pd.DataFrame:
    # Use pandas to read the CSV file into a DataFrame object
    df = pd.read_csv('mtsamples.csv', encoding='utf-8')
    # Return the DataFrame object
    return df


def write_csv_file(df: pd.DataFrame) -> None:
    # write the DataFrame to a CSV file
    df.to_csv('mtsamples_with_more_info.csv', index=False)


def extract_information(df: pd.DataFrame) -> pd.DataFrame:
    genders, ages, treatments = [], [], []
    # Select the description and transcription columns
    selected_columns = df[['description', 'transcription']]
    # Loop through each row in the selected columns
    for _, row in selected_columns.iterrows():
        # Concatenate the description and transcription columns
        # because both columns may contain information about the patient
        description = row['description']
        transcription = row['transcription']
        text = f'{description} {transcription}'
        # extract gender and age information from the text
        gender = extract_gender(text)
        genders.append(gender)
        # extract age information from the text
        age = extract_age(text)
        ages.append(age)

        treatment = extract_treatment(transcription)
        treatments.append(treatment)

    # add the gender and age columns to the DataFrame
    df.insert(1, 'gender', genders)
    df.insert(2, 'age', ages)
    df.insert(3, 'treatment', treatments)

    # Return the DataFrame object
    return df


# extract_gender extracts the gender of a patient from a given text using a set of keywords.
# extract_gender returns 'male' if male-related keywords are found in the text, 'female' if female-related keywords are found,
# and None if no keywords are found.
def extract_gender(text: str) -> Optional[str]:
    # Define a dictionary of gender-related keywords, with 'male' and 'female' as keys and a list of keywords as values.
    gender_keywords = {
        'male': ['male', 'man', 'boy', 'gentleman', 'guy', 'lad', 'dude', 'chap', 'fellow', 'bro', 'gent'],
        'female': ['female', 'woman', 'girl', 'lady', 'gal', 'lass', 'dame', 'miss', 'maiden', 'sister']
    }
    # Loop through each gender and their corresponding keywords.
    for gender, keywords in gender_keywords.items():
        # Loop through each keyword and create a regular expression pattern.
        for keyword in keywords:
            # Use the re.search() function to check if the keyword exists in the text.
            pattern = r'\b{}\b'.format(keyword)
            if re.search(pattern, text.lower()):
                return gender

    # If no keywords are found, return None.
    return None


# extract_age extracts the age of a patient from a given text using a regular expression pattern.
# extract_age returns a string with the format 'age_value-age_unit-old' if age-related keywords are found in the text,
# and None if no keywords are found.


def extract_age(text: str) -> Optional[str]:
    # Define a regular expression pattern to extract age information.
    age_pattern = re.compile(
        r'((\d{1,3})|([a-zA-Z]+))[-\s]?(years?|months?)[-\s]?old', re.IGNORECASE)
    # Use the re.search() function to check if the keyword exists in the text.
    match = age_pattern.search(text)
    # If a match is found, extract the age number and unit.
    if match:
        age_number = match.group(1)
        age_unit = match.group(4).lower()

        try:
            # Convert the age number to an integer.
            age_value = int(age_number)
        except ValueError:
            # If the age number is not an integer, use the word2number library to convert it to an integer.
            try:
                # Convert the age number to an integer.
                age_value = w2n.word_to_num(age_number)
            except Exception as e:
                # If the age number is not a number, return None.
                print(f"Error: {e}\nAge number: {age_number}")
                return None
        # If the age unit is plural, remove the last letter.
        if age_unit.endswith('s'):
            age_unit = age_unit[:-1]

        # Return the age value and unit.
        # Assume the it needs a string with the format: 'age_value-age_unit-old' rather than only the age value.
        return f'{age_value}-{age_unit}-old'

    # If no match is found, return None.
    return None


def extract_treatment(transcription):
    treatments = []
    try:
        sentences = sent_tokenize(transcription)
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)

            for i, (token, pos) in enumerate(pos_tags):
                if token.lower() in ["try", "use", "given", "prescription", "prescribe", "administer", "recommend",
                                     "apply", "inject", "perform", "initiate", "start", "continue", "increase",
                                     "decrease", "switch", "discontinue", "monitor", "evaluate", "adjust", "follow-up"]:
                    treatment = " ".join([t for t, _ in pos_tags[i:]])
                    treatments.append(treatment)
                    break
    except Exception as e:
        print(f"Error: {e}\nTranscription: {transcription}")
    print(treatments)
    return ' '.join(treatments)


# Run the main function
if __name__ == '__main__':
    main()
