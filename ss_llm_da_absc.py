# This file contains all the methods and steps for SS-LLM-DA-ABSC Contextual and Category augmentations.
# It should be separately uploaded to the Colab as the first step for these DAs.
# After the code below is ran, one will obtain the augmented data fiels in the format that is suitable for the main code. However, these files will only contain train data and thus, one must merge it with test data based on the year.
# After the code below is ran and the data obtained is merged with test data into new joint files (with an appropriate naming), one can start running the TorchBERT.py code for embeddings and then the prepare_bert.py code for the main model.
# The data files obtained after this code is finished (and test data is added to them) have to be uploaded manually into the raw_data folder such taht they can be located when the prepare_bert.py is being executed. 
# Note: make sure that the file names are in accordance with the names defined in config

# Import all the necessary packages
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import json as json
import os
import pickle
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import time
from typing import List, Tuple
from openai import OpenAI
from tqdm import tqdm
import nltk
import glob
import random

# This script performs everything related to SS-LLM-DA-ABSC for Contextual and Category augmentations, and starts with Contextual one.

# The path to files is hardcoded and thus requires adjustments when necessary
# The original xml file
xml_file = "ABSA-16_Restaurants_Train_Final.xml"
# File where the data will be saved in the new format
output_file = "data_for_LLM_DA_ABSC_2016.json"

print(f"Reading XML file: {xml_file}")

tree = ET.parse(xml_file)
root = tree.getroot()

data = []

# Go over each review
for review in root.findall('Review'):

    # Go through each sentence
    sentences = review.find('sentences')
    for sentence in sentences.findall('sentence'):
        sentence_id = sentence.get('id')
        sentence_text = sentence.find('text').text

        # Get opinions for this sentence
        opinions = sentence.find('Opinions')
        if opinions is not None:
            for opinion in opinions.findall('Opinion'):
                target = opinion.get('target')
                polarity = opinion.get('polarity')

                # Skip target if it is NULL
                if target == 'NULL':
                    continue

                # The same review is now in the new format (JSON) that is more readable by LLMs than the xml file
                new_review = {
                    'sentence': sentence_text,
                    'target': target,
                    'polarity': polarity
                }
                data.append(new_review)

# To check that no reviews were lost and they are correctly saved to the output file
print(f"Saving {len(data)} entries to: {output_file}")

# Here, we trasfer all the data to the output file
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

# Perfrom SimCSE on every sentence

# Import the model
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

# Read data from the JSON file
with open('data_for_LLM_DA_ABSC_2016.json', 'r') as f:
    data = json.load(f)

# Load sentences in the list to pass to the SimCSE model
sentences_for_embeddings = [entry['sentence'] for entry in data]

# Generate embeddings
inputs = tokenizer(sentences_for_embeddings, padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

# For debudding
# print(f"Generated embeddings for {len(sentences_for_embeddings)} sentences.")
# print(f"The sahpe of the embeddings is {embeddings.shape}")
# print(f"The first sentence is {sentences_for_embeddings[0]}")

# Perfrom the Contextual augmentation

# In this method we specify the system prompt for Contextual augmentation
def system_prompt():
    prompt = """
      Your only task is to create one new restaurant review sentence that will be used as a training example for Aspect-Based Sentiment Classification.

      Input:
      You will always receive content in JSON file format, with specification provided below
      {
        "original_sentence": "<string>",
        "target": "<string>",
        "polarity": "<positive|negative|neutral>",
        "similar_sentences": ["<string>", "<string>", "<string>", "<string>", "<string>"]
      }

      Your task is to pick one or more augmentation methods provided below:
      1. Make sentence shorter.
      2. Make sentence longer.
      3. Add advance words to the sentence
      4. Add adverbs to the sentence.
      5. Add more adjectives to the sentence.
      6. Add more prepositions to the sentence.
      7. Add more conjuctions to the sentence.
      8. Incorporate subordinate clauses.
      9. Change the presentation style of the sentence to news style sentence.
      10. Change the presentation style of the sentence to spoken language style sentence.
      11. Change the presentation style of the sentence to magazines style sentence.
      12. Change the presentation style of the sentence to fictions style sentence.
      13. Change the presentation style of the sentence to Wikipedia style sentence.
      14. Change the presentation style of the sentence to movie reviews style sentence.

      Constraints:
      1. Keep the target token identical (this applies to spelling, casing, spacing).
      2. Preserve the original polarity.
      3. Do not introduce conflicting sentiment cues.
      4. The new sentence must be written in English.
      5. Output no metadata or explanations.
      6. The new sentence has to be a restaurant review.

      Output:
      Return exactly one line in the way presented below

      {"augmented_sentence": "<your new sentence here>"}

      Example of the augmentation that is provided for reference only:
      Given
      {
        "original_sentence": "The chips and salsa are so yummy.",
        "target":            "chips and salsa",
        "polarity":          "positive",
        "similar_sentences": [
          "Risotto was very tasty.",
          "The steak is delicious.",
          "The food was one of the best I have ever eaten.",
          "The chef did not lie about how delicious the fish is.",
          "We tried mushroom soup that did not disappoint."
        ]
      }

      Possible Output
      {"augmented_sentence": "The chips and salsa were easily the highlight of the evening."}
          """
    return prompt

# In this method we specify the user prompt for Contextual augmentation
# Similar sentences are sentences that are obtained as similar examples after SimCSE is performed
def user_prompt(input_sentence, target, polarity, similar_sentences):

  prompt = f"""
{{
    "original_sentence": "{input_sentence}",
    "target":            "{target}",
    "polarity":          "{polarity}",
    "
    similar_sentences": [
  """
  for i, sentence in enumerate(similar_sentences):
    prompt += f'"{sentence}"'
    if i < len(similar_sentences) - 1:
      prompt += ",\n"

  prompt += """

  ]
}}
  """
  return prompt

# In this method, we find the 5 most similar sentnces based on cosin similarity measure and generated embeddings and return the list of these sentences
def top_5_examples(target_index, embeddings, all_sentences):

  target_embedding = embeddings[target_index]

  examples = []
  similarity_scores_indices = []
  seen_sentences = set()

  for i in range(len(all_sentences)):
    # If we have reached the index of the target sentence that we want to find examples for, we skip this sentence
    if i == target_index:
      continue
    if all_sentences[i] in seen_sentences:
      continue
    seen_sentences.add(all_sentences[i])

    cosine_sim = 1 - cosine(target_embedding, embeddings[i])
    similarity_scores_indices.append((cosine_sim, i))

  # Sort based on the similarity scores which is the first item in the tuple inside similarity_scores_indices
  # Also sort in descending order so top 5 examples are easily extractable
  similarities_sorted = sorted(similarity_scores_indices, key=lambda item: item[0], reverse=True)

  # Only get the indices of the top 5 sentnces from the list of all indices
  most_similar_indices = [item[1] for item in similarities_sorted[:5]]

  # Extract these sentences from the list of all the sentnces to then pass to the LLM
  examples = [all_sentences[i] for i in most_similar_indices]

  return examples

model_name = "gpt-4o"
# In order to perform this augmentation you have to extract your own API kep to access the OpenAI and store it in Secrets in your own Collab
from google.colab import userdata
client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

# Here we call gpt-4o model

# Get the system prompt
system_prompt_for_gpt = system_prompt()
results = []
all_the_sentences_for_augmentation = sentences_for_embeddings

# For debudding
print(f"Augmentation for {len(all_the_sentences_for_augmentation)} sentences started.")


# i here is the index of the target sentnce for augmentation
for i in range(len(all_the_sentences_for_augmentation)):

  # For debugging
  # print(f"\nProcessing sentence index: {i}")

  top_5_sentences = top_5_examples(i, embeddings, all_the_sentences_for_augmentation)

  user_prompt_for_gpt = user_prompt(all_the_sentences_for_augmentation[i], data[i]['target'], data[i]['polarity'], top_5_sentences)
  # For debugging:
  # print(f"Prompt for ChatGPT: {user_prompt_for_gpt}")

  resp = client.responses.create(
        model= model_name,
        input=[
            {"role": "system",
             "content": [{"type": "input_text", "text": system_prompt_for_gpt}]},
            {"role": "user",
             "content": [{"type": "input_text", "text": user_prompt_for_gpt}]},
        ],
        text={"format": {"type": "json_object"}},
        temperature=1, top_p=1, max_output_tokens=2048,
    )

  # For debugging:
  # print(f"Full response object:\n{resp}")

  try:
      # Here we extract JSON from the full response object
      response_text = resp.output[0].content[0].text

      # Add this response to dictionary
      response_dict = json.loads(response_text)

      # Extract augmented sentence
      augmented_sentence = response_dict.get("augmented_sentence", "ERROR")

  except Exception as e:
      print(f"Error happens at response at index {i}: {e}")
      augmented_sentence = "ERROR"

  # For debugging:
  # print(f"New Augmented sentence: {augmented_sentence}")
  # print(f"Target in the sentence: {data[i]['target']}")
  # print(f"Polarity of the sentence: {data[i]['polarity']}")
  # print(f"Original sentence: {data[i]['sentence']}")

  # Add augmented sentence to results with respected target and polarity
  results.append({
      "sentence": augmented_sentence,
      "target": data[i]['target'],
      "polarity": data[i]['polarity']
  })

# Save to JSON file
# The file name is HARDCODED so change when necessary
with open("augmented_sentences_2015.json", "w", encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Below, we merge the augmenated data with the original data and save it to txt file

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

# In this method we process the generated json files that contain augmented data generated by LLMs
# This method returns list of (sentence_with_target_replaced, target, polarity_number)
def process_generated_files(json_file):

    augmentation_data = []
    # COunter for all the sentnces in the generated method
    total_sentences = 0

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        generated_sentences = data

        for it in generated_sentences:
            sentence = it.get('sentence')
            target = it.get('target')
            polarity = it.get('polarity')

            if sentence and target and polarity:
                # Replace target in sentence with $T$
                sentence_with_Ts = replace_target_in_sentence(sentence, target)
                # Convert polarity to number
                polarity_number = polarity_to_number(polarity)

                augmentation_data.append((sentence_with_Ts, target, polarity_number))
                total_sentences += 1

    except Exception as e:
        print(f"Error in processing {json_file}: {e}")

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

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# The code below merges the original data with the generated data.
# Thus, make sure that when the script is executed for different datasets, the hardcoded paths are adjusted accordingly.

# Get the current directory
current_dir = os.getcwd()
# Load data
xml_filepath = os.path.join(current_dir, xml_file)

# If file not found will raise an eroro - for debugging
if not os.path.exists(xml_filepath):
    print(f"XML file not found: '{xml_file}' in '{current_dir}'")

# Process the original xml file in here
original_data = process_xml_for_augmentation(xml_file)
print("Completed processing original file")

json_file = "augmented_sentences.json"
generated_data = []
generated_data = process_generated_files(json_file)
print("Completed processing generated files")

# Create a file that will be an output filename
output_file = os.path.join(current_dir, xml_file.replace(".xml", "_augmented_using_LLM-DA-ABSC_contextual_2016.txt"))
save_augmented_data(original_data, generated_data, output_file)
# The Contextual augmentation is finished and the full dataset ready for the embeddings and prepare_bert.py is created
print(f"Saved combined data to: {output_file}")

# Here starts the Category augmentation.

# The path to files is hardcoded and thus requires adjustments when necessary
# The original xml file
xml_file = "ABSA-15_Restaurants_Train_Final.xml"
# File where the data will be saved in the new format
output_file = "data_for_LLM_DA_ABSC_2015_c.json"

print(f"Reading XML file: {xml_file}")
tree = ET.parse(xml_file)
root = tree.getroot()

data = []

# Go over each review
for review in root.findall('Review'):

    # Go through each sentence
    sentences = review.find('sentences')
    for sentence in sentences.findall('sentence'):
        sentence_id = sentence.get('id')
        sentence_text = sentence.find('text').text

        # Get opinions for this sentence
        opinions = sentence.find('Opinions')
        if opinions is not None:
            for opinion in opinions.findall('Opinion'):
                target = opinion.get('target')
                category = opinion.get('category')
                polarity = opinion.get('polarity')

                # Skip target if it is NULL
                if target == 'NULL':
                    continue

                # The same review is now in the new format (JSON) that is more readable by LLMs than the xml file
                # Compared to the COntextual augmentation with each review we also pass the category of the reveview sentence
                new_review = {
                    'sentence': sentence_text,
                    'target': target,
                    'category': category,
                    'polarity': polarity
                }
                data.append(new_review)

# To check that no reviews were lost and they are correctly saved to the output file
print(f"Saving {len(data)} entries to: {output_file}")

# Here, we trasfer all the data to the output file
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

# Classify the data into different categories and create new json files for that categories

# Load the full data
with open("data_for_LLM_DA_ABSC_2015_c.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Store unique categories inside
category_entries = {}

# Each entry in the json file classify by category
for entry in data:
    category = entry.get("category")

    if category not in category_entries:
        category_entries[category] = []

    category_entries[category].append(entry)

# Here we crate a new json file per category
for category, entries in category_entries.items():
    filename = f"{category}_category.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

# In this method we specify the system prompt for Category augmentation
def system_prompt_for_category():
    prompt = """
      Your only task is to create one new restaurant review sentence that will be used as a training example for Aspect-Based Sentiment Classification.

      Input:
      You will always receive content in JSON file format, with specification provided below

      {
        "original_sentence": "<string>",
        "target": "<string>",
        "polarity": "<positive|negative|neutral>",
        "category": "<string>",
        "similar_targets": ["<string>", "<string>", "<string>", "<string>", "<string>"]
      }

      Augmentation method:
      1. Pick one of the two augmentation techniques with 50% chance:
         a. Swap out the target in the given sentences with one of the given examples of targets.
         b. Swap out the target in the given sentence with one of the targets generated by you that matches the sentence's category.

      Constraints:
      1. Keep everything except for the target token identical (this applies to spelling, casing, spacing).
      2. Preserve the original polarity.
      3. Do not change the language of the original sentence.
      4. Output no metadata or explanations.
      5. The new sentence has to be a restaurant review.

      Output:
      JSON file in the way presented below

      {"augmented_sentence": "<your new sentence here>",
       "target": "<your new target here>",
       "polarity": "<polarity here>"
      }

      Example of the augmentation that is provided for reference only:
      Given
      {
        "original_sentence": "the chips and salsa are so yummy , and the prices are fabulous.",
        "target":            "chips and salsa",
        "polarity":          "positive",
        "category":          "FOOD#GENERAL",
        "similar_targets":   [ "risotto and pasta", "burger and fries", "sushi", "appetizers", "chief special and lasagna"]
      }

      Possible Output with strategy a:
      {"augmented_sentence": "the burger and fries are so yummy , and the prices are fabulous.",
       "target" : "burger and fries",
       "polarity": "positive"
       }

      Possible Output with strategy b:
      {"augmented_sentence": "the burritos are so yummy , and the prices are fabulous.",
       "target" : "burritos",
       "polarity": "positive"
      }
          """
    return prompt

# In this method we specify the user prompt for Category augmentation
# Similar targets are tergets that are randoly selected form the unique targets list per category
def user_prompt_for_category(input_sentence, target, polarity, similar_targets, category):

  prompt = f"""
{{
    "original_sentence": "{input_sentence}",
    "target":          "{target}",
    "polarity":        "{polarity}",
    "category":        "{category}",
    "
    similar_targets": [
  """
  for i, target in enumerate(similar_targets):
    prompt += f'"{target}"'
    if i < len(similar_targets) - 1:
      prompt += ",\n"

  prompt += """

  ]
}}
  """
  return prompt

model_name = "gpt-4o"

from google.colab import userdata
client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

system_prompt_for_category_augmentation = system_prompt_for_category()

# Below, we perform the Contextual augmentation but before that we also create a list of unique targets per category

# Dictionary to store the unique target lists
all_unique_targets = {}

# Loop over all the *_category.json files
for filepath in glob.glob("*_category.json"):

    # Extract the category name from the filename
    category_name = filepath.split("_category.json")[0]
    list_name = f"unique_targets_{category_name}"
    results = []

    # For debugging:
    # print(f"Currently working with : {category_name}")

    # Load the JSON file
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect unique targets
    unique_targets = set()
    for entry in data:
        target = entry.get("target")
        if target:
            unique_targets.add(target)

    # Convert set to list
    all_unique_targets[list_name] = list(unique_targets)

    # Loop through each entry in the file
    for entry in data:
        sentence = entry.get("sentence")
        target = entry.get("target")
        polarity = entry.get("polarity")
        category = entry.get("category")

        if not sentence:
          continue

        # Select 5 random targets from the unique list
        selected_targets = random.sample(list(unique_targets), min(5, len(unique_targets)))
        # For debugging:
        # print(f"Selected targets for {category}: {selected_targets}")

        # Get the user prompt from the method
        user_prompt_for_category_augmentation = user_prompt_for_category(sentence, target,polarity, selected_targets, category)
        # For debugging:
        # print(f"Prompt for {sentence}: {user_prompt_for_category_augmentation}")

        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system",
                  "content": [{"type": "input_text", "text": system_prompt_for_category_augmentation}]},
                {"role": "user",
                  "content": [{"type": "input_text", "text": user_prompt_for_category_augmentation}]},
                  ],
            text={"format": {"type": "json_object"}},
            temperature=1, top_p=1, max_output_tokens=2048,
        )

        # For debugging:
        # print(f"Full response object:\n{resp}")

        try:
            # Here we extract JSON from the full response object
            response_text = resp.output[0].content[0].text

            # Add this response to dictionary
            response_dict = json.loads(response_text)

            # Extract augmented sentence
            augmented_sentence = response_dict.get("augmented_sentence", "")
            new_target = response_dict.get("target", "")
            new_polarity = response_dict.get("polarity", "")

            # Add augmented sentence to results with respected target and polarity
            results.append({
                "sentence": augmented_sentence,
                "target": new_target,
                "polarity": new_polarity
            })

        except Exception as e:
            print(f"Error happened at reposnse: {e}")

        # For debugging:
        # print(f"New augmented sentence: {augmented_sentence}")
        # print(f"New TARGET in the augmented sentence: {new_target}")

    # Save results for this category to a JSON file
    with open(f"{category_name}_augmented.json", "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)

# Merge all the Category augmeneted files to create one dataset with all the augmented sentences

# List to place all merged data
merged_data = []

# Loop through all files ending with _augmented.json
for filepath in glob.glob("*_augmented.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Immediately add data to the list
        merged_data.extend(data)

# Here we put all the merged data into one file
# HARDCODED path, adjust the name if necessary
with open("merged_augmented.json", "w", encoding="utf-8") as out_file:
    json.dump(merged_data, out_file, indent=4, ensure_ascii=False)

# HARDCODED file path, adjust if necessary
xml_file = "ABSA-15_Restaurants_Train_Final.xml"

# Get the current directory
current_dir = os.getcwd()
# Load data
xml_filepath = os.path.join(current_dir, xml_file)

# If file not found will raise an eroro - for debugging
if not os.path.exists(xml_filepath):
    print(f"XML file not found: '{xml_file}' in '{current_dir}'")

# Process the original xml file in here
original_data = process_xml_for_augmentation(xml_file)
print("Completed processing original file")

json_file = "merged_augmented.json"
generated_data = []
generated_data = process_generated_files(json_file)
print("Completed processing generated files")

# Create a file that will be an output filename
output_file = os.path.join(current_dir, xml_file.replace(".xml", "_augmented_using_LLM-DA-ABSC_category_2015.txt"))
save_augmented_data(original_data, generated_data, output_file)
# The Contextual augmentation is finished and the full dataset ready for the embeddings and prepare_bert.py is created
print(f"Saved combined data to: {output_file}")