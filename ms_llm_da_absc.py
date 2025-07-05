
# This file contains all the methods and steps for MS-LLM-DA-ABSC augmentations.
# It should be separately uploaded to the Colab as the first step for these DAs.
# After the code below is ran, one will obtain the augmented data fiels in the format that is suitable for the main code. However, these files will only contain train data and thus, one must merge it with test data based on the year.
# After the code below is ran and the data obtained is merged with test data into new joint files (with an appropriate naming), one can start running the TorchBERT.py code for embeddings and then the prepare_bert.py code for the main model.
# The data files obtained after this code is finished (and test data is added to them) have to be uploaded manually into the raw_data folder such taht they can be located when the prepare_bert.py is being executed. 
# Note: make sure that the file names are in accordance with the names defined in config

## MS-LLM-DA-ABSC

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import pickle
import json
import torch
from typing import List, Tuple
from tqdm import tqdm
import time
import nltk
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine as cosine_distance

"""##### Below there are genearal methods as well as specific ones for all the augmentation techniques

##### Processing data per category is the same for each method, so the structure below up to the augmentation part can be rerun for multiple augmentation methods.
"""

# Path to the input file, HARDCODED so change when necessary
input_file = "ABSA-15_Restaurants_Train_Final.xml"

# Process XML file and save all the opinions with no NULL targets to the dataframe
def process_xml_file(filepath):

  tree = ET.parse(filepath)
  root = tree.getroot()

  all_opinions_data = []

  for review in root.findall('Review'):
      # Go through each sentence
      for sentences in review.findall('sentences'):
          for sentence in sentences.findall('sentence'):

              sentence_text_node = sentence.find('text')
              sentence_text = sentence_text_node.text.strip()

              # Get opinions for this sentence
              opinions = sentence.find('Opinions')
              if opinions is not None:
                  for opinion in opinions.findall('Opinion'):
                      target = opinion.get('target')
                      category = opinion.get('category')
                      polarity = opinion.get('polarity')

                      # Skip target if it is NULL
                      if target == "NULL":
                        continue

                      # The review is saved with all the necessary details
                      all_opinions_data.append({
                          'Category': category,
                          'Sentence': sentence_text,
                          'Target': target,
                          'Polarity': polarity
                      })

  # To check that no reviews were lost and they are correctly saved to the output file
  print(f"Saving {len(all_opinions_data)} entries to the dataframe.")
  # Create the dataframe with all the data entries
  df = pd.DataFrame(all_opinions_data)

  return df

# Here, we split the main dataframe by category so we have dtaframes for each category and they are saved in CSV and Pickle formats
def save_dataframes_by_category(main_df, csv_output_dir, pkl_output_dir, base_output_filename):

  # Check whether the main dataframe is empty
  if main_df.empty:
    print("The dataframe is empty.")

  # Get unique categories from the main dataframe
  unique_categories = main_df['Category'].unique()

  # For debugging:
  # print(f"Unique categories: {unique_categories}")

  for category in unique_categories:
      category_df = main_df[main_df['Category'] == category].copy()
      # Rename the category dataframes to make it simpler
      safe_category_name = category.replace('#', '').replace('_', '')

      # Save as CSV
      category_csv_filename = f"{base_output_filename}_{safe_category_name}_category.csv"
      category_csv_filepath = os.path.join(csv_output_dir, category_csv_filename)
      category_df.to_csv(category_csv_filepath, index=False, encoding='utf-8')
      print(f"  Saved {category} CSV to {category_csv_filepath}")

      # Save as Pickle
      category_pkl_filename = f"{base_output_filename}_{safe_category_name}_category.pkl"
      category_pkl_filepath = os.path.join(pkl_output_dir, category_pkl_filename)
      category_df.to_pickle(category_pkl_filepath)
      print(f"  Saved {category} PKL to {category_pkl_filepath}")

# Make sure you are in the correct directory
current_directory = os.getcwd()
# Get the path of the input file
selected_filepath = os.path.join(current_directory, input_file)

# For debugging:
# Check if the path exists:
# if not os.path.exists(selected_filepath):
#   print(f"Error: I cannot find the input file called '{input_file}' in the directory you are in called '{current_directory}'.")

# Process the xml file and save to dataframe
df = process_xml_file(selected_filepath)

# Get the name of the input file but only the base without the .xml part
base_output_filename = os.path.splitext(input_file)[0]

# Set up th main output directory to keep the data organised
main_output_dir_name = f"{base_output_filename}_output"
main_output_dir_path = os.path.join(current_directory, main_output_dir_name)
os.makedirs(main_output_dir_path, exist_ok=True)
print(f"Main output directory: {main_output_dir_path}")

# Create inside main output directory CSV and PKL subdirectories where the csv and pickle category data will be stored
csv_dir_path = os.path.join(main_output_dir_path, "csv")
pkl_dir_path = os.path.join(main_output_dir_path, "pkl")
os.makedirs(csv_dir_path, exist_ok=True)
os.makedirs(pkl_dir_path, exist_ok=True)

# Save main _processed.csv to csv/ subfolder
main_csv_filename = f"{base_output_filename}_processed.csv"
main_csv_filepath = os.path.join(csv_dir_path, main_csv_filename)
df.to_csv(main_csv_filepath, index=False, encoding='utf-8')
print(f"Main dataframe CSV saved to {main_csv_filepath}")

# Save main _processed.pkl to pkl/ subfolder
main_pkl_filename = f"{base_output_filename}_processed.pkl"
main_pkl_filepath = os.path.join(pkl_dir_path, main_pkl_filename)
df.to_pickle(main_pkl_filepath)
print(f"Main dataframe PKL saved to {main_pkl_filepath}")

# Save data per category to both CSV and PCKL directories
save_dataframes_by_category(df, csv_dir_path, pkl_dir_path, base_output_filename)

"""#### Generate SimCSE embeddings"""

input_folder = main_output_dir_name
print(f"Currently working with {input_folder}")

# Again get the path to the folder
input_folder_path = os.path.join(current_directory, input_folder)

# Get the path to the pkl directory
pkl_dir_path = os.path.join(input_folder_path, "pkl")

# Load the SimCSe model and also the tokenizer
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

# Method to generate embeddings
def generate_embeddings_simcse(model, tokenizer, sentences):
  inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

  with torch.no_grad():
      embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

  return embeddings

# Generate embeddinsg for each category file

print("Started scanning category files")

# Get all the pkl files for all the categories
pkl_files = [f for f in os.listdir(pkl_dir_path) if f.endswith("_category.pkl")]

# For debugging:
# Check if the pckl files can be found
# if not pkl_files:
#   print(f"I could not find pkl files for categories in the {pkl_dir_path} directory")

# Process each file
for file in pkl_files:
  print(f"Processing file {file}")
  df = pd.read_pickle(os.path.join(pkl_dir_path, file))

  # Get the sentences
  sentences = df['Sentence'].dropna().tolist()

  # Get the embeddings
  embeddings_values = generate_embeddings_simcse(model, tokenizer, sentences)

  output_name = file.replace("_category.pkl", "_embeddings.txt")
  output_path = os.path.join(input_folder_path, output_name)

  # Save embeddings in txt format
  with open(output_path, "w", encoding="utf-8") as f:
    for i in range(embedding_values.shape[0]):
      embedding_values = embedding_values[i].cpu().numpy().tolist()
      embedding_in_str = [str(val) for val in embedding_values]
      f.write(" ".join(embedding_in_str) + "\n")

"""### Select k diverse examples"""

# HARDCODED path so adjust when necessary
input_folder_name = "ABSA-15_Restaurants_Train_Final_output"
# Number of diverse examples passed to the LLM
num_diverse_examples = 5

input_folder_path = os.path.join(current_directory, input_folder_name)
pkl_folder_path = os.path.join(input_folder_path, "pkl")

# Get the text files for embeddings
txt_files = [f for f in os.listdir(input_folder_name) if f.endswith("_embeddings.txt")]

# Here, we select diverse examples per category to pass to LLM based on cosine similarity
# This method returns the list of indices of selected sentences
def select_diverse_sentences(all_embeddings, all_sentences, num_to_select):
    # Get the total number of available sentences
    num_available_sentences = len(all_sentences)

    # If no sentences availble then an empty list will be returned
    if num_available_sentences == 0:
        return []

    # If the number of availbel sentences in the category is smaller then the number of sentences requested then we select all these sentences as examples
    if num_available_sentences <= num_to_select:
        return list(range(num_available_sentences))

    # List of selected indices
    selected_indices = []
    # List of corresponding to selected indices embeddings
    selected_embeddings_list = []

    # The first sentences from all the sentences is added to start the selection process, it acts like an anchor
    selected_indices.append(0)
    selected_embeddings_list.append(all_embeddings[0])

    for _ in range(1, num_to_select):
        next_diverse_sentence_index = -1
        lowest_max_similarity_score = float('inf')

        for candidate_idx in range(num_available_sentences):
            # If the sentence is already in the list, skip
            if candidate_idx in selected_indices:
                continue

            current_candidate_embedding = all_embeddings[candidate_idx]
            max_similarity_of_candidate_to_selected_set = -float('inf')

            # Compute similarity between the current candidate and all the previous selcted sentences
            for selected_emb in selected_embeddings_list:
                similarity = 1 - cosine_distance(current_candidate_embedding, selected_emb)
                if similarity > max_similarity_of_candidate_to_selected_set:
                    max_similarity_of_candidate_to_selected_set = similarity

            # If the current candidate is the least similar so far we select this candidate to be the next sentence joining the list
            if max_similarity_of_candidate_to_selected_set < lowest_max_similarity_score:
                lowest_max_similarity_score = max_similarity_of_candidate_to_selected_set
                next_diverse_sentence_index = candidate_idx

        if next_diverse_sentence_index != -1:
            selected_indices.append(next_diverse_sentence_index)
            selected_embeddings_list.append(all_embeddings[next_diverse_sentence_index])
        else:
            break

    return selected_indices

# Go over each txt embeddings file
for file_emb in txt_files:
  print(f"Started woring with the file : {file_emb}")

  embeddings_filepath = os.path.join(input_folder_path, file_emb)
  first_name_part = file_emb.replace("_embeddings.txt", "")
  pkl_filename = first_name_part + "_category.pkl"
  pkl_filepath = os.path.join(input_folder_path, "pkl", pkl_filename)

  # Load embeddings from txt
  embeddings = []

  with open(embeddings_filepath, 'r', encoding='utf-8') as f:
    for line in f:
      embeddings.append(np.array([float(val) for val in line.strip().split()]))

  embeddings_arr= np.array(embeddings)

  # Extrcat the dataframe for the category
  df_category = pd.read_pickle(pkl_filepath)

  # Get the list of sentences
  sentences_list = df_category['Sentence'].tolist()
  # get the name of the category to then put in the sentence obj
  category_display_name = df_category['Category'].iloc[0] if not df_category.empty else first_name_part

  # Get the indices of sentences that will be the diverse examples for that category
  selected_indices = select_diverse_sentences(embeddings_arr, sentences_list, num_diverse_examples)

  # If indices succesfully selected
  if selected_indices:
    for_json_file = []
    for idx in selected_indices:

      sentence_text = df_category['Sentence'].iloc[idx]
      target = df_category['Target'].iloc[idx]
      polarity = df_category['Polarity'].iloc[idx]

      # Creeate sentence object for json file
      sentence_obj = {
          "sentence": sentence_text,
          "target_word": target,
          "polarity": polarity
      }

      # add sentence object to the new json file
      for_json_file.append(sentence_obj)

    # Merge examples and the category name
    output_data_for_category = {
        "category": category_display_name,
        "sentences": for_json_file
    }

    # For debugging:
    # print(""Finished selecting diverse examples for category: {category_display_name}")

    json_output_filename = f"{first_name_part}_diverse_analysis.json"
    json_output_filepath = os.path.join(input_folder_path, json_output_filename)

    with open(json_output_filepath, 'w', encoding='utf-8') as f_json:
      json.dump(output_data_for_category, f_json, indent=4)

"""### Call GPT-4o model for Balanced Augmentation"""

# HRDCODED folder that should conatin all the files with diverse examples produced by the previous steps
input_folder_name = "ABSA-15_Restaurants_Train_Final_output"

# OpenAI model that will be called
model_name = "gpt-4o"

# Retry strategy
max_retries = 2
backoff_seconds = 2

# Batch size of how many sentences is the model asked to generate
batch_size = 18

# In order to perform this augmentation you have to extract your own API kep to access the OpenAI and store it in Secrets in your own Collab
from google.colab import userdata
client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

# This method gets the list of all the file names with the diverse examples stored
def list_diverse_json_files(general_folder_path):
    result = []

    for filename in os.listdir(general_folder_path):
        if filename.endswith("_diverse_analysis.json"):
            full_path = os.path.join(general_folder_path, filename)
            result.append(full_path)

    return result

# In this method the system prompt is built
def build_system_prompt(needed):

    base  = needed // 3
    extra = needed % 3

    # This method is for balanced augmentation so if the base numebr is not divisble by 3 then we add one more positive sentences (and if extra is 2, one more negative sentence)
    pos = base
    if extra > 0:
      pos += 1

    if extra > 1:
      neg += 1

    neu = base

    prompt = f"""

    Your task is to generate {needed} brand-new restaurant review sentences about the same category that will be used as training examples for Aspect-Based Sentiment Classification.

    Input: ou will always receive exactly one JSON object with the structure specified below.
    {{
      "category": "<string>",
      "sentences": [ 5 examples]
    }}

    Augmentation method:
    1. Each generated sentence must be assigned one polarity label: positive, negative, or neutral.
    2. Produce exactly {pos} positive, {neg} negative, {neu} neutral sentences (counts may differ by at most 1 when the total is not divisible by 3).

    Constraints:
    1. Output must contain exactly {needed} objects in "generated_sentences".
    2. Each object must have keys: "sentence", "target_word", "polarity".
    3. "target_word" must be ONE word present in its sentence.
    4. Don't repeat any example sentence or target_word.
    5. Sentences must be realistic, unique, and clearly express the stated polarity.
    6. Return only the JSON object.
    7. The new sentences have to be restaurant reviews.

    Output:
    {{
      "category": "<string>",
      "generated_sentences": [ {{ … }} ]
    }}

"""
    return prompt

# This method is a support method and it checks whether the response form the LLM is in the correct format (if the response is a valid dictionary (the dictionary is called response) with the correct structure and the number of sentences(called needed)).
# It is essential to check the output of LLM when it is asked to generate significant amount of new sentences with little supervison in order to prevent hallucination.
def validate_response(response, needed):

    # Check if the response is indeed a dictionary and not a string
    if not isinstance(response, dict):
        return False

    # Check if the dictionary contains the newly generated sentences ("generated_sentences") and category
    if "generated_sentences" not in response or "category" not in response:
        return False

    sentences = response["generated_sentences"]
    # Check that the list of generated sentence is a list and has the length as expected
    if not isinstance(sentences, list) or len(sentences) != needed:
        return False

    # Ensure that teh generated sentence contains target and polarity
    for item in sentences:
        if "sentence" not in item or "target_word" not in item or "polarity" not in item:
            return False

    return True

# In this method we call the LLM with system and user prompts, get the response, and return responsse in the json file
def call_chat_completion(system_prompt, user_prompt, batch_n):

    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system",
             "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user",
             "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        text={"format": {"type": "json_object"}},
        temperature=1, top_p=1, max_output_tokens=2048, store=False,
    )

    result = None

    if getattr(resp, "output", None):
        out = resp.output
        if isinstance(out, list) and hasattr(out[0], "content") and out[0].content:
            inner = out[0].content[0]
            if hasattr(inner, "text"):
                result = inner.text
        elif isinstance(out, (str, dict)):
            result = out

    elif getattr(resp, "text", None):
        result = resp.text

    if result is None:
        raise ValueError("No usable result found in response")

    if isinstance(result, dict):
        return json.dumps(result)
    else:
        return str(result)

input_folder = os.path.join(current_directory, input_folder_name)

# GEt all the json files with the diverse examples to be passed to the LLM
json_files = list_diverse_json_files(input_folder)

# tqdm is used to track the progress of the process of processing each file
for file_path in tqdm(json_files, desc="Categories", unit="cat"):

    with open(file_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # Load pkl file to find out how many sentences we need for augmentation
    key_part = os.path.basename(file_path).replace("_diverse_analysis.json", "")
    pkl_path = os.path.join(input_folder, "pkl", f"{key_part}_category.pkl")

    df = pd.read_pickle(pkl_path)
    # Needed is the number of sentences we need for augmentation
    needed = len(df.index)

    remaining = needed
    aggregated = []

    while remaining > 0:

        if remaining > batch_size:
            batch_n = batch_size
        else:
            batch_n = remaining

        system_prompt = build_system_prompt(batch_n)

        for attempt in range(max_retries + 1):
            try:
                # Call the model and get the response
                response_model = call_chat_completion(system_prompt, json.dumps(original_data), batch_n)
                batch_data = json.loads(response_model)

                #If the generated response did not pass the validation checks it breaks
                if validate_response(batch_data, batch_n):
                    break
            except:
                time.sleep(backoff_seconds)

        # Add genearted sentences to the list
        aggregated.extend(batch_data["generated_sentences"])
        # Decrease the count
        remaining -= batch_n

    # All the charcters that the model hallucinates sometimes
    map = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00e9": "e",
    }

    # Here we filter out all the duplicate sentences that have been generated by the model
    seen_sentences = set()
    unique_sentences = []

    for item in aggregated:
        sent_norm = item["sentence"]

        # replace the hallucintion with teh correct characters
        for bad, good in map.items():
            sent_norm = sent_norm.replace(bad, good)

        if item["sentence"] not in seen_sentences:
            item["sentence"] = sent_norm
            seen_sentences.add(sent_norm)
            unique_sentences.append(item)

    # Generate new sentences in case the model generated duplicats
    if len(unique_sentences) < needed:
        print( "Generated duplicated sentences, so running extra batch to fill in the deleted sentences with new ones")

        # The model will rerun the augmentation sentence by sentence until it reaches the correct number of sentences
        filler_needed = needed - len(unique_sentences)

        while filler_needed > 0:
            system_prompt_single = build_system_prompt(1)
            single_data = json.loads(call_chat_completion(system_prompt_single, json.dumps(original_data), 1))
            sentence_one = single_data["generated_sentences"][0]

            if sentence_one["sentence"] not in seen_sentences:
                unique_sentences.append(sentence_one)
                seen_sentences.add(sentence_one["sentence"])
                filler_needed -= 1

    final_output = {
        "category": original_data.get("category", ""),
        "generated_sentences": unique_sentences[:needed],
    }

    output_path = file_path.replace("_diverse_analysis.json", "_generated.json")

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(final_output, f_out, indent=4)

print("All sentences are generated and saved")

"""### Save the augmented sentences and merged them with the training data (below both general methods for MS-LLM-DA-ABSC as well as GPT-4o_balanced augmentation specific methods are present)"""

# This method transforms polarity word to the number so it is ready for the main model
def polarity_to_number(polarity):

    # Lowercase the polarity word in case the LLM haluccinated and produced polarity with the uppercase
    polarity_lower = polarity.lower()

    if polarity_lower == 'positive':
        return '1'
    elif polarity_lower == 'negative':
        return '-1'
    elif polarity_lower == 'neutral':
        return '0'
    else:
        return '0'  # Default is set to neutral

# This method replaces the target word in the sentence with $T$ to match the format of the data for the main code
def replace_target_in_sentence(sentence, target):

    # Here we tokeniz the sentences and target words using nltk tokenizer like in dataReader2016.py
    sentence_tokens = nltk.word_tokenize(sentence)
    target_tokens = nltk.word_tokenize(target)

    # Convert every word to lowercase
    sentence_lower = ' '.join(token.lower() for token in sentence_tokens)
    target_lower = ' '.join(token.lower() for token in target_tokens)

    # Replace tareget with the $T$ like in the original method
    result = sentence_lower.replace(target_lower, '$T$')

    return result

# This method helps to read the xml file with the original data
def process_xml_for_augmentation(filepath):

    tree = ET.parse(filepath)
    root = tree.getroot()
    resulted_file = []

    # Set the counter varaibles that are used below
    total_sentences = 0
    sentences_filtered = 0

    for review in root.findall('Review'):

        for sentences_node in review.findall('sentences'):

            for sentence_node in sentences_node.findall('sentence'):
                sentence_text_node = sentence_node.find('text')

                if sentence_text_node is None or sentence_text_node.text is None:
                    continue

                sentence_text = sentence_text_node.text.strip()
                if not sentence_text:
                    continue

                total_sentences += 1

                opinions = sentence_node.find('Opinions')

                # Skip Opinion if it is none
                if opinions is None:
                    continue

                for opinion in opinions.findall('Opinion'):
                    target = opinion.get('target')
                    category = opinion.get('category')
                    polarity = opinion.get('polarity')

                    # Only take into account sentences that do NOT have NULL targets
                    if target.upper() != 'NULL':
                        # Replace target in sentence with $T$
                        sentence_with_Ts = replace_target_in_sentence(sentence_text, target)
                        # Replace polarity word with the number
                        polarity_number = polarity_to_number(polarity)

                        resulted_file.append((sentence_with_Ts, target, polarity_number))

                        # Increase the count of filtered sentences that are actually used in the data
                        sentences_filtered += 1

    # For debugging:
    # print(f"Total sentences in the raw XML file: {total_sentences}")
    # print(f"Sentences with no NULL targets: {sentences_filtered}")

    return resulted_file

# This method finds all the generated files
def find_generated_json_files(base_path):
  generated_files = []

  for file_name in os.listdir(base_path):
    if file_name.endswith("_generated.json"):
      full_path = os.path.join(base_path, file_name)
      generated_files.append(full_path)

  return generated_files

# In this method we process the generated json files that contain augmented data generated by LLMs
# This method returns list of (sentence_with_target_replaced, target, polarity_number)
def process_generated_files(json_files: List[str]) -> List[Tuple[str, str, str]]:
    """
    Process generated JSON files and return list of (sentence_with_target_replaced, target, polarity_number).
    """
    augmentation_data = []
    # COunter for all the sentnces in the generated method
    total_processed = 0

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            category = data.get('category', 'Unknown')
            generated_sentences = data

            for it in generated_sentences:
                sentence = it.get('sentence')
                target = it.get('target_word')
                polarity = it.get('polarity')

                if sentence and target and polarity:
                    # Replace target in sentence with $T$
                    sentence_with_Ts = replace_target_in_sentence(sentence, target)
                    # Convert polarity to number
                    polarity_number = polarity_to_number(polarity)

                    augmentation_data.append((sentence_with_Ts, target, polarity_number))
                    total_processed += 1

        except Exception as e:
            print(f"  Error processing {json_path}: {e}")

    return augmentation_data

# This method takes the original data and the generated data and combines them into one txt file
def save_augmented_data(original_data, generated_data, output_filepath):

    with open(output_filepath, 'w', encoding='utf-8') as f:
        # First process the original file
        for sentence, target, polarity in original_data:
            f.write(f"{sentence}\n")
            f.write(f"{target}\n")
            f.write(f"{polarity}\n")

        # Next process the generated file
        for sentence, target, polarity in generated_data:
            f.write(f"{sentence}\n")
            f.write(f"{target}\n")
            f.write(f"{polarity}\n")

nltk.download('punkt')
nltk.download('punkt_tab')

# Get the current directory
current_directory = os.getcwd()
# HARDCODED FILENAMES, so change when necessary
input_xml_file = "ABSA-16_Restaurants_Train_Final.xml"
folder_name_inp = "2016_gpt_balanced/"

# Load data
xml_filepath = os.path.join(current_directory, input_xml_file)

# If file not found will raise an eroro - for debugging
if not os.path.exists(xml_filepath):
    print(f"Error: Input XML file '{input_xml_file}' not found in '{current_directory}'.")

# Process the original xml file in here
original_data = process_xml_for_augmentation(xml_filepath)
print("Completed processing original file")

# Process generated JSON files
input_folder = os.path.join(current_directory, folder_name_inp)
json_files = find_generated_json_files(input_folder)
generated_data = process_generated_files(json_files)
print("Completed processing generated files")

# Save combined data
base_name = os.path.splitext(input_xml_file)[0]
output_filename = f"{base_name}_augmented_using_GPT_4o_balanced.txt"
output_filepath = os.path.join(current_directory, output_filename)
save_augmented_data(original_data, generated_data, output_filepath)

"""### Call GPT-4o model for Proportional Augmentation"""

# HRDCODED folder that should conatin all the files with diverse examples produced by the previous steps
input_folder_name = "ABSA-15_Restaurants_Train_Final_output"

# OpenAI model that will be called
model_name = "gpt-4o"

# Retry strategy
max_retries = 2
backoff_seconds = 2

# Batch size of how many sentences is the model asked to generate
batch_size = 18

# In order to perform this augmentation you have to extract your own API kep to access the OpenAI and store it in Secrets in your own Collab
from google.colab import userdata
client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

# This method gets the list of all the file names with the diverse examples stored
def list_diverse_json_files(general_folder_path):
    result = []

    for filename in os.listdir(general_folder_path):
        if filename.endswith("_diverse_analysis.json"):
            full_path = os.path.join(general_folder_path, filename)
            result.append(full_path)

    return result

# In this method the system prompt is built
def build_system_prompt_proportional(needed, pos_count, neg_count, neu_count):
    prompt = f"""

    Your task is to generate {needed} brand-new restaurant review sentences about the same category that will be used as training examples for Aspect-Based Sentiment Classification.

    Input: ou will always receive exactly one JSON object with the structure specified below.
    {{
      "category": "<string>",
      "sentences": [ 5 examples]
    }}

    Augmentation method:
    1. Each generated sentence must be assigned one polarity label: positive, negative, or neutral.
    2. Produce exactly {pos_count} positive, {neg_count} negative, {neu_count} neutral sentences.

    Constraints:
    1. Output must contain exactly {needed} objects in "generated_sentences".
    2. Each object must have keys: "sentence", "target_word", "polarity".
    3. "target_word" must be one word present in its sentence.
    4. Don't repeat any example sentence or target_word.
    5. Sentences must be realistic, unique, and clearly express the stated polarity.
    6. Return only the JSON object.
    7. The new sentences have to be restaurant reviews.

    Output:
    {{
      "category": "<string>",
      "generated_sentences": [ {{ … }} ]
    }}

"""
    return prompt

# This method is a support method and it checks whether the response form the LLM is in the correct format (if the response is a valid dictionary (the dictionary is called response) with the correct structure and the number of sentences(called needed)).
# It is essential to check the output of LLM when it is asked to generate significant amount of new sentences with little supervison in order to prevent hallucination.
def validate_response(response, needed):

    # Check if the response is indeed a dictionary and not a string
    if not isinstance(response, dict):
        return False

    # Check if the dictionary contains the newly generated sentences ("generated_sentences") and category
    if "generated_sentences" not in response or "category" not in response:
        return False

    sentences = response["generated_sentences"]
    # Check that the list of generated sentence is a list and has the length as expected
    if not isinstance(sentences, list) or len(sentences) != needed:
        return False

    # Ensure that teh generated sentence contains target and polarity
    for item in sentences:
        if "sentence" not in item or "target_word" not in item or "polarity" not in item:
            return False

    return True

# In this method we call the LLM with system and user prompts, get the response, and return responsse in the json file
def call_chat_completion(system_prompt, user_prompt, batch_n):

    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system",
             "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user",
             "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        text={"format": {"type": "json_object"}},
        temperature=1, top_p=1, max_output_tokens=2048, store=False,
    )

    result = None

    if getattr(resp, "output", None):
        out = resp.output
        if isinstance(out, list) and hasattr(out[0], "content") and out[0].content:
            inner = out[0].content[0]
            if hasattr(inner, "text"):
                result = inner.text
        elif isinstance(out, (str, dict)):
            result = out

    elif getattr(resp, "text", None):
        result = resp.text

    if result is None:
        raise ValueError("No usable result found in response")

    if isinstance(result, dict):
        return json.dumps(result)
    else:
        return str(result)

# This method counts how many sentences of all the polarities are in the original dataset and returns the dictionary of those counts
def analyze_polarity_distribution(df):

    polarity_counts = df['Polarity'].value_counts()

    # Extract all the counts of polarities
    counts = {}
    for polarity, count in polarity_counts.items():
        polarity_lower = polarity.lower()
        counts[polarity_lower] = count

    return {
        'positive': counts.get('positive', 0),
        'negative': counts.get('negative', 0),
        'neutral': counts.get('neutral', 0)
    }

# This method determines how many sentences of each polarity to include in the batch
def calculate_batch_polarities(batch_size, remaining_pos, remaining_neg, remaining_neu):

    total_remaining = remaining_pos + remaining_neg + remaining_neu
    actual_batch_size = min(batch_size, total_remaining)

    # We simply take what we need, up to batch size

    batch_pos = min(remaining_pos, actual_batch_size)
    remaining_for_batch = actual_batch_size - batch_pos

    batch_neg = min(remaining_neg, remaining_for_batch)
    remaining_for_batch -= batch_neg

    batch_neu = min(remaining_neu, remaining_for_batch)

    batch_total = batch_pos + batch_neg + batch_neu

    return batch_pos, batch_neg, batch_neu, batch_total

current_directory = os.getcwd()
input_folder = os.path.join(current_directory, input_folder_name)

# GEt all the json files with the diverse examples to be passed to the LLM
json_files = list_diverse_json_files(input_folder)

# tqdm is used to track the progress of the process of processing each file
for file_path in tqdm(json_files, desc="Categories", unit="cat"):

    with open(file_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # Load pkl file to find out how many sentences we need for augmentation
    key_part = os.path.basename(file_path).replace("_diverse_analysis.json", "")
    pkl_path = os.path.join(input_folder, "pkl", f"{key_part}_category.pkl")

    df = pd.read_pickle(pkl_path)
    # Needed is the number of sentences we need for augmentation
    needed = len(df.index)
    # Analyze polarity distribution
    polarity_dist = analyze_polarity_distribution(df)

    remaining_total = needed
    remaining_pos = polarity_dist['positive']
    remaining_neg = polarity_dist['negative']
    remaining_neu = polarity_dist['neutral']
    aggregated = []

    while remaining_total > 0:

        # Calculate proportional batch
        batch_pos, batch_neg, batch_neu, batch_total = calculate_batch_polarities(batch_size, remaining_pos, remaining_neg, remaining_neu)

        if batch_total == 0:
          break

        system_prompt = build_system_prompt_proportional(batch_total, batch_pos, batch_neg, batch_neu)

        for attempt in range(max_retries + 1):
            try:
                # Call the model and get the response
                response_model = call_chat_completion(system_prompt, json.dumps(original_data), batch_total)
                batch_data = json.loads(response_model)

                #If the generated response did not pass the validation checks it breaks
                if validate_response(batch_data, batch_n):
                    break
            except:
                time.sleep(backoff_seconds)

        # Add genearted sentences to the list
        aggregated.extend(batch_data["generated_sentences"])
        # Decrease the count
        remaining_total -= batch_total
        remaining_pos -= batch_pos
        remaining_neg -= batch_neg
        remaining_neu -= batch_neu

    # All the charcters that the model hallucinates sometimes
    map = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00e9": "e",
    }

    # Here we filter out all the duplicate sentences that have been generated by the model
    seen_sentences = set()
    unique_sentences = []

    for item in aggregated:
        sent_norm = item["sentence"]

        # replace the hallucintion with teh correct characters
        for bad, good in map.items():
            sent_norm = sent_norm.replace(bad, good)

        if item["sentence"] not in seen_sentences:
            item["sentence"] = sent_norm
            seen_sentences.add(sent_norm)
            unique_sentences.append(item)

    # Here we count the polarity distribution in generated sentences
    generated_polarity_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for item in unique_sentences:
        polarity = item.get('polarity', '').lower()
        if polarity in generated_polarity_counts:
            generated_polarity_counts[polarity] += 1

    # For debugging
    # print(f"Generated polarity distribution: Positive: {generated_polarity_counts['positive']}, Negative: {generated_polarity_counts['negative']}, Neutral: {generated_polarity_counts['neutral']}")

    # Generate new sentences in case the model generated duplicats
    if len(unique_sentences) < needed:
        print( "Generated duplicated sentences, so running extra batch to fill in the deleted sentences with new ones")

        # First, calculate what excatly is missing for each polarity
        missing_pos = needed - generated_polarity_counts['positive']
        missing_neg = needed - generated_polarity_counts['negative']
        missing_neu = needed - generated_polarity_counts['neutral']

        # For debugging
        # print(f"Missing this number of sentences: Positive: {missing_pos}, Negative: {missing_neg}, Neutral: {missing_neu}")

        for polarity_type, missing_count in [('positive', missing_pos), ('negative', missing_neg), ('neutral', missing_neu)]:
            while missing_count > 0:

                 # Generate one sentence of the needed polarity

                 if polarity_type == 'positive':
                    pos_count =1
                 else:
                    pos_count = 0

                 if polarity_type == 'negative':
                    neg_count =1
                 else:
                    neg_count = 0

                 if polarity_type == 'neutral':
                    neu_count =1
                 else:
                    neu_count = 0

                 system_prompt_single = build_system_prompt_proportional(1, pos_count, neg_count, neu_count)
                 single_data = json.loads(call_chat_completion(system_prompt_single, json.dumps(original_data), 1))
                 sentence_one = single_data["generated_sentences"][0]

                if sentence_one["sentence"] not in seen_sentences and sentence_one.get('polarity', '').lower() == polarity_type:
                    unique_sentences.append(sentence_one)
                    seen_sentences.add(sentence_one["sentence"])
                    missing_count -= 1
                    generated_polarity_counts[polarity_type] += 1

    final_output = {
        "category": original_data.get("category", ""),
        "generated_sentences": unique_sentences[:needed],
    }

    # Create proportional output directory
    proportional_dir = os.path.join(input_folder, "proportional")
    os.makedirs(proportional_dir, exist_ok=True)

    # Save to proportional directory
    out_filename = os.path.basename(file_path).replace("_diverse_analysis.json", "_generated.json")
    out_path = os.path.join(proportional_dir, out_filename)
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(final_output, f_out, indent=4)

print("All sentences are generated and saved")

"""##### Merge GPT-4o Proportional Augmentation data with the original training data"""

# Get the current directory
current_directory = os.getcwd()
# HARDCODED FILENAMES, so change when necessary
input_xml_file = "ABSA-16_Restaurants_Train_Final.xml"
folder_name_inp = "2016_gpt_proportional/"

# Load data
xml_filepath = os.path.join(current_directory, input_xml_file)

# If file not found will raise an eroro - for debugging
if not os.path.exists(xml_filepath):
    print(f"Error: Input XML file '{input_xml_file}' not found in '{current_directory}'.")

# Process the original xml file in here
original_data = process_xml_for_augmentation(xml_filepath)
print("Completed processing original file")

# Process generated JSON files
input_folder = os.path.join(current_directory, folder_name_inp)
json_files = find_generated_json_files(input_folder)
generated_data = process_generated_files(json_files)
print("Completed processing generated files")

# Save combined data
base_name = os.path.splitext(input_xml_file)[0]
output_filename = f"{base_name}_augmented_using_GPT_4o_proportional.txt"
output_filepath = os.path.join(current_directory, output_filename)
save_augmented_data(original_data, generated_data, output_filepath)

"""## Llama-3-70B Instruct Balanced Augmentation"""

# HRDCODED folder that should conatin all the files with diverse examples produced by the previous steps
input_folder_name = "ABSA-15_Restaurants_Train_Final_output"

# DeepInfra Llama model that will be called
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

# Retry strategy
max_retries = 2
backoff_seconds = 2

# Batch size of how many sentences is the model asked to generate
batch_size = 18

# In order to perform this augmentation you have to extract your own API kep to access the DeepInfra Platform and store it in Secrets in your own Collab
# You can get your DeepInfra API token from https://deepinfra.com/
from google.colab import userdata
client = OpenAI(api_key=userdata.get('DEEPINFRA_API_KEY'), base_url="https://api.deepinfra.com/v1/openai")

# This method gets the list of all the file names with the diverse examples stored
def list_diverse_json_files(general_folder_path):
    result = []

    for filename in os.listdir(general_folder_path):
        if filename.endswith("_diverse_analysis.json"):
            full_path = os.path.join(general_folder_path, filename)
            result.append(full_path)

    return result

# In this method the system prompt is built
def build_system_prompt(needed):

    base  = needed // 3
    extra = needed % 3

    # This method is for balanced augmentation so if the base numebr is not divisble by 3 then we add one more positive sentences (and if extra is 2, one more negative sentence)
    pos = base
    if extra > 0:
      pos += 1

    if extra > 1:
      neg += 1

    neu = base

    prompt = f"""

    Your task is to generate {needed} brand-new restaurant review sentences about the same category that will be used as training examples for Aspect-Based Sentiment Classification.

    Input: ou will always receive exactly one JSON object with the structure specified below.
    {{
      "category": "<string>",
      "sentences": [ 5 examples]
    }}

    Augmentation method:
    1. Each generated sentence must be assigned one polarity label: positive, negative, or neutral.
    2. Produce exactly {pos} positive, {neg} negative, {neu} neutral sentences (counts may differ by at most 1 when the total is not divisible by 3).

    Constraints:
    1. Output must contain exactly {needed} objects in "generated_sentences".
    2. Each object must have keys: "sentence", "target_word", "polarity".
    3. "target_word" must be ONE word present in its sentence.
    4. Don't repeat any example sentence or target_word.
    5. Sentences must be realistic, unique, and clearly express the stated polarity.
    6. Return only the JSON object.
    7. The new sentences have to be restaurant reviews.

    Output:
    {{
      "category": "<string>",
      "generated_sentences": [ {{ … }} ]
    }}

"""
    return prompt

# This method is a support method and it checks whether the response form the LLM is in the correct format (if the response is a valid dictionary (the dictionary is called response) with the correct structure and the number of sentences(called needed)).
# It is essential to check the output of LLM when it is asked to generate significant amount of new sentences with little supervison in order to prevent hallucination.
def validate_response(response, needed):

    # Check if the response is indeed a dictionary and not a string
    if not isinstance(response, dict):
        return False

    # Check if the dictionary contains the newly generated sentences ("generated_sentences") and category
    if "generated_sentences" not in response or "category" not in response:
        return False

    sentences = response["generated_sentences"]
    # Check that the list of generated sentence is a list and has the length as expected
    if not isinstance(sentences, list) or len(sentences) != needed:
        return False

    # Ensure that teh generated sentence contains target and polarity
    for item in sentences:
        if "sentence" not in item or "target_word" not in item or "polarity" not in item:
            return False

    return True

# In this method we call the LLM with system and user prompts, get the response, and return responsse in the json file
def call_chat_completion(system_prompt, user_prompt, batch_n):

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",
             "content":  system_prompt},
            {"role": "user",
             "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=1, top_p=1, max_output_tokens=2048, store=False,
    )

    # Get the content of the response
    if response.choices and len(response.choices) > 0:
      content = response.choices[0].message.content
      if content:
        return content
    else:
      print("No response found")
      return None

input_folder = os.path.join(current_directory, input_folder_name)

# GEt all the json files with the diverse examples to be passed to the LLM
json_files = list_diverse_json_files(input_folder)

# tqdm is used to track the progress of the process of processing each file
for file_path in tqdm(json_files, desc="Categories", unit="cat"):

    with open(file_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # Load pkl file to find out how many sentences we need for augmentation
    key_part = os.path.basename(file_path).replace("_diverse_analysis.json", "")
    pkl_path = os.path.join(input_folder, "pkl", f"{key_part}_category.pkl")

    df = pd.read_pickle(pkl_path)
    # Needed is the number of sentences we need for augmentation
    needed = len(df.index)

    remaining = needed
    aggregated = []

    while remaining > 0:

        if remaining > batch_size:
            batch_n = batch_size
        else:
            batch_n = remaining

        system_prompt = build_system_prompt(batch_n)

        for attempt in range(max_retries + 1):
            try:
                # Call the model and get the response
                response_model = call_chat_completion(system_prompt, json.dumps(original_data), batch_n)
                batch_data = json.loads(response_model)

                #If the generated response did not pass the validation checks it breaks
                if validate_response(batch_data, batch_n):
                    break
            except:
                time.sleep(backoff_seconds)

        # Add genearted sentences to the list
        aggregated.extend(batch_data["generated_sentences"])
        # Decrease the count
        remaining -= batch_n

    # All the charcters that the model hallucinates sometimes
    map = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00e9": "e",
    }

    # Here we filter out all the duplicate sentences that have been generated by the model
    seen_sentences = set()
    unique_sentences = []

    for item in aggregated:
        sent_norm = item["sentence"]

        # replace the hallucintion with teh correct characters
        for bad, good in map.items():
            sent_norm = sent_norm.replace(bad, good)

        if item["sentence"] not in seen_sentences:
            item["sentence"] = sent_norm
            seen_sentences.add(sent_norm)
            unique_sentences.append(item)

    # Generate new sentences in case the model generated duplicats
    if len(unique_sentences) < needed:
        print( "Generated duplicated sentences, so running extra batch to fill in the deleted sentences with new ones")

        # The model will rerun the augmentation sentence by sentence until it reaches the correct number of sentences
        filler_needed = needed - len(unique_sentences)

        while filler_needed > 0:
            system_prompt_single = build_system_prompt(1)
            single_data = json.loads(call_chat_completion(system_prompt_single, json.dumps(original_data), 1))
            sentence_one = single_data["generated_sentences"][0]

            if sentence_one["sentence"] not in seen_sentences:
                unique_sentences.append(sentence_one)
                seen_sentences.add(sentence_one["sentence"])
                filler_needed -= 1

    final_output = {
        "category": original_data.get("category", ""),
        "generated_sentences": unique_sentences[:needed],
    }

    output_path = file_path.replace("_diverse_analysis.json", "_generatedllama.json")

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(final_output, f_out, indent=4)

print("All sentences are generated and saved")

"""##### Merge Llama-3-70B Instruct Balanced Augmentation data with the original training data"""

# Get the current directory
current_directory = os.getcwd()
# HARDCODED FILENAMES, so change when necessary
input_xml_file = "ABSA-16_Restaurants_Train_Final.xml"
folder_name_inp = "2016_llama3_balanced/"

# Load data
xml_filepath = os.path.join(current_directory, input_xml_file)

# If file not found will raise an eroro - for debugging
if not os.path.exists(xml_filepath):
    print(f"Error: Input XML file '{input_xml_file}' not found in '{current_directory}'.")

# Process the original xml file in here
original_data = process_xml_for_augmentation(xml_filepath)
print("Completed processing original file")

# Process generated JSON files
input_folder = os.path.join(current_directory, folder_name_inp)
json_files = find_generated_json_files(input_folder)
generated_data = process_generated_files(json_files)
print("Completed processing generated files")

# Save combined data
base_name = os.path.splitext(input_xml_file)[0]
output_filename = f"{base_name}_augmented_using_Llama3_70B_Instruct_balanced.txt"
output_filepath = os.path.join(current_directory, output_filename)
save_augmented_data(original_data, generated_data, output_filepath)

"""#### Call Llama-3-70B Instruct model for Proportional Augmentation"""

# HRDCODED folder that should conatin all the files with diverse examples produced by the previous steps
input_folder_name = "ABSA-15_Restaurants_Train_Final_output"

# DeepInfra Llama model that will be called
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

# Retry strategy
max_retries = 2
backoff_seconds = 2

# Batch size of how many sentences is the model asked to generate
batch_size = 18

# In order to perform this augmentation you have to extract your own API kep to access the DeepInfra Platform and store it in Secrets in your own Collab
# You can get your DeepInfra API token from https://deepinfra.com/
from google.colab import userdata
client = OpenAI(api_key=userdata.get('DEEPINFRA_API_KEY'), base_url="https://api.deepinfra.com/v1/openai")

# This method gets the list of all the file names with the diverse examples stored
def list_diverse_json_files(general_folder_path):
    result = []

    for filename in os.listdir(general_folder_path):
        if filename.endswith("_diverse_analysis.json"):
            full_path = os.path.join(general_folder_path, filename)
            result.append(full_path)

    return result

# In this method the system prompt is built
def build_system_prompt_proportional(needed, pos_count, neg_count, neu_count):
    prompt = f"""

    Your task is to generate {needed} brand-new restaurant review sentences about the same category that will be used as training examples for Aspect-Based Sentiment Classification.

    Input: ou will always receive exactly one JSON object with the structure specified below.
    {{
      "category": "<string>",
      "sentences": [ 5 examples]
    }}

    Augmentation method:
    1. Each generated sentence must be assigned one polarity label: positive, negative, or neutral.
    2. Produce exactly {pos_count} positive, {neg_count} negative, {neu_count} neutral sentences.

    Constraints:
    1. Output must contain exactly {needed} objects in "generated_sentences".
    2. Each object must have keys: "sentence", "target_word", "polarity".
    3. "target_word" must be one word present in its sentence.
    4. Don't repeat any example sentence or target_word.
    5. Sentences must be realistic, unique, and clearly express the stated polarity.
    6. Return only the JSON object.
    7. The new sentences have to be restaurant reviews.

    Output:
    {{
      "category": "<string>",
      "generated_sentences": [ {{ … }} ]
    }}

"""
    return prompt

# This method is a support method and it checks whether the response form the LLM is in the correct format (if the response is a valid dictionary (the dictionary is called response) with the correct structure and the number of sentences(called needed)).
# It is essential to check the output of LLM when it is asked to generate significant amount of new sentences with little supervison in order to prevent hallucination.
def validate_response(response, needed):

    # Check if the response is indeed a dictionary and not a string
    if not isinstance(response, dict):
        return False

    # Check if the dictionary contains the newly generated sentences ("generated_sentences") and category
    if "generated_sentences" not in response or "category" not in response:
        return False

    sentences = response["generated_sentences"]
    # Check that the list of generated sentence is a list and has the length as expected
    if not isinstance(sentences, list) or len(sentences) != needed:
        return False

    # Ensure that teh generated sentence contains target and polarity
    for item in sentences:
        if "sentence" not in item or "target_word" not in item or "polarity" not in item:
            return False

    return True

# In this method we call the LLM with system and user prompts, get the response, and return responsse in the json file
def call_chat_completion(system_prompt, user_prompt, batch_n):

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",
             "content":  system_prompt},
            {"role": "user",
             "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=1, top_p=1, max_output_tokens=2048, store=False,
    )

    # Get the content of the response
    if response.choices and len(response.choices) > 0:
      content = response.choices[0].message.content
      if content:
        return content
    else:
      print("No response found")
      return None

# This method counts how many sentences of all the polarities are in the original dataset and returns the dictionary of those counts
def analyze_polarity_distribution(df):

    polarity_counts = df['Polarity'].value_counts()

    # Extract all the counts of polarities
    counts = {}
    for polarity, count in polarity_counts.items():
        polarity_lower = polarity.lower()
        counts[polarity_lower] = count

    return {
        'positive': counts.get('positive', 0),
        'negative': counts.get('negative', 0),
        'neutral': counts.get('neutral', 0)
    }

# This method determines how many sentences of each polarity to include in the batch
def calculate_batch_polarities(batch_size, remaining_pos, remaining_neg, remaining_neu):

    total_remaining = remaining_pos + remaining_neg + remaining_neu
    actual_batch_size = min(batch_size, total_remaining)

    # We simply take what we need, up to batch size

    batch_pos = min(remaining_pos, actual_batch_size)
    remaining_for_batch = actual_batch_size - batch_pos

    batch_neg = min(remaining_neg, remaining_for_batch)
    remaining_for_batch -= batch_neg

    batch_neu = min(remaining_neu, remaining_for_batch)

    batch_total = batch_pos + batch_neg + batch_neu

    return batch_pos, batch_neg, batch_neu, batch_total

current_directory = os.getcwd()
input_folder = os.path.join(current_directory, input_folder_name)

# GEt all the json files with the diverse examples to be passed to the LLM
json_files = list_diverse_json_files(input_folder)

# tqdm is used to track the progress of the process of processing each file
for file_path in tqdm(json_files, desc="Categories", unit="cat"):

    with open(file_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # Load pkl file to find out how many sentences we need for augmentation
    key_part = os.path.basename(file_path).replace("_diverse_analysis.json", "")
    pkl_path = os.path.join(input_folder, "pkl", f"{key_part}_category.pkl")

    df = pd.read_pickle(pkl_path)
    # Needed is the number of sentences we need for augmentation
    needed = len(df.index)
    # Analyze polarity distribution
    polarity_dist = analyze_polarity_distribution(df)

    remaining_total = needed
    remaining_pos = polarity_dist['positive']
    remaining_neg = polarity_dist['negative']
    remaining_neu = polarity_dist['neutral']
    aggregated = []

    while remaining_total > 0:

        # Calculate proportional batch
        batch_pos, batch_neg, batch_neu, batch_total = calculate_batch_polarities(batch_size, remaining_pos, remaining_neg, remaining_neu)

        if batch_total == 0:
          break

        system_prompt = build_system_prompt_proportional(batch_total, batch_pos, batch_neg, batch_neu)

        for attempt in range(max_retries + 1):
            try:
                # Call the model and get the response
                response_model = call_chat_completion(system_prompt, json.dumps(original_data), batch_total)
                batch_data = json.loads(response_model)

                #If the generated response did not pass the validation checks it breaks
                if validate_response(batch_data, batch_n):
                    break
            except:
                time.sleep(backoff_seconds)

        # Add genearted sentences to the list
        aggregated.extend(batch_data["generated_sentences"])
        # Decrease the count
        remaining_total -= batch_total
        remaining_pos -= batch_pos
        remaining_neg -= batch_neg
        remaining_neu -= batch_neu

    # All the charcters that the model hallucinates sometimes
    map = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00e9": "e",
    }

    # Here we filter out all the duplicate sentences that have been generated by the model
    seen_sentences = set()
    unique_sentences = []

    for item in aggregated:
        sent_norm = item["sentence"]

        # replace the hallucintion with teh correct characters
        for bad, good in map.items():
            sent_norm = sent_norm.replace(bad, good)

        if item["sentence"] not in seen_sentences:
            item["sentence"] = sent_norm
            seen_sentences.add(sent_norm)
            unique_sentences.append(item)

    # Here we count the polarity distribution in generated sentences
    generated_polarity_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for item in unique_sentences:
        polarity = item.get('polarity', '').lower()
        if polarity in generated_polarity_counts:
            generated_polarity_counts[polarity] += 1

    # For debugging
    # print(f"Generated polarity distribution: Positive: {generated_polarity_counts['positive']}, Negative: {generated_polarity_counts['negative']}, Neutral: {generated_polarity_counts['neutral']}")

    # Generate new sentences in case the model generated duplicats
    if len(unique_sentences) < needed:
        print( "Generated duplicated sentences, so running extra batch to fill in the deleted sentences with new ones")

        # First, calculate what excatly is missing for each polarity
        missing_pos = needed - generated_polarity_counts['positive']
        missing_neg = needed - generated_polarity_counts['negative']
        missing_neu = needed - generated_polarity_counts['neutral']

        # For debugging
        # print(f"Missing this number of sentences: Positive: {missing_pos}, Negative: {missing_neg}, Neutral: {missing_neu}")

        for polarity_type, missing_count in [('positive', missing_pos), ('negative', missing_neg), ('neutral', missing_neu)]:
            while missing_count > 0:

                 # Generate one sentence of the needed polarity

                 if polarity_type == 'positive':
                    pos_count =1
                 else:
                    pos_count = 0

                 if polarity_type == 'negative':
                    neg_count =1
                 else:
                    neg_count = 0

                 if polarity_type == 'neutral':
                    neu_count =1
                 else:
                    neu_count = 0

                 system_prompt_single = build_system_prompt_proportional(1, pos_count, neg_count, neu_count)
                 single_data = json.loads(call_chat_completion(system_prompt_single, json.dumps(original_data), 1))
                 sentence_one = single_data["generated_sentences"][0]

                if sentence_one["sentence"] not in seen_sentences and sentence_one.get('polarity', '').lower() == polarity_type:
                    unique_sentences.append(sentence_one)
                    seen_sentences.add(sentence_one["sentence"])
                    missing_count -= 1
                    generated_polarity_counts[polarity_type] += 1

    final_output = {
        "category": original_data.get("category", ""),
        "generated_sentences": unique_sentences[:needed],
    }

    # Create proportional llama output directory
    proportional_llama_dir = os.path.join(input_folder, "proportional_llama")
    os.makedirs(proportional_llama_dir, exist_ok=True)

    # Save to proportional directory
    out_filename = os.path.basename(file_path).replace("_diverse_analysis.json", "_generated.json")
    out_path = os.path.join(proportional_llama_dir, out_filename)
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(final_output, f_out, indent=4)

print("All sentences are generated and saved")

"""##### Merge Llama-3-70B Instruct Proportional Augmentation data with the original training data"""

# Get the current directory
current_directory = os.getcwd()
# HARDCODED FILENAMES, so change when necessary
input_xml_file = "ABSA-16_Restaurants_Train_Final.xml"
folder_name_inp = "2016_llama3_balanced/"

# Load data
xml_filepath = os.path.join(current_directory, input_xml_file)

# If file not found will raise an eroro - for debugging
if not os.path.exists(xml_filepath):
    print(f"Error: Input XML file '{input_xml_file}' not found in '{current_directory}'.")

# Process the original xml file in here
original_data = process_xml_for_augmentation(xml_filepath)
print("Completed processing original file")

# Process generated JSON files
input_folder = os.path.join(current_directory, folder_name_inp)
json_files = find_generated_json_files(input_folder)
generated_data = process_generated_files(json_files)
print("Completed processing generated files")

# Save combined data
base_name = os.path.splitext(input_xml_file)[0]
output_filename = f"{base_name}_augmented_using_Llama3_70B_Instruct_proportional.txt"
output_filepath = os.path.join(current_directory, output_filename)
save_augmented_data(original_data, generated_data, output_filepath)