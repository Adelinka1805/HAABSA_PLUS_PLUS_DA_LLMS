'''
This Program makes the BERT embedding matrix and test-/traindata, using the tokenisation for BERT
First 'TorchBert.py' should be used to compile the subfiles containing embeddings. After the embeddings matrix is obtained, the embeddings file shoul dbe uploaded into the programGeneratedData folder BEFORE RUNING THIS PROGRAM (The embeddings files names should be in corrspondance with the required name stated in config)
The new test-/traindata files contain original data, with every word unique and corresponding to vector in emb_matrix (This program will purput the train and test files that will be then used during hyperparameter optimization and model tarining -> Example: BERT768traindata2015_BERT_prepend.txt)
'''
from config import *
from transformers import BertTokenizer

def tokenize_sentence(sentence, tokenizer, word_counts):
    words = "[CLS] " + sentence + " [SEP]"
    tokenized_words = tokenizer.tokenize(words)
    tokens = []
    targetbool = True
    for word in tokenized_words[1:-1]:
      if word == '$':
        if targetbool:
          tokens.append("$T$")
          targetbool = False
      elif word == 't':
        pass
      else:
        count = word_counts.get(word, -1) + 1
        word_counts[word] = count
        token = f'{word}_{count}'
        tokens.append(token)
    return ' '.join(tokens)


#def main():
    '''
    Adds BERT embedding values to sentences in the original test and train datasets. Then
    saves these as separate test and train files, which can be used as an input for a classification
    algorithm.
    '''

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
word_counts = {}

with open('data/programGeneratedData/temp/unique2016_BERT_Data_All.txt', 'w') as output_f:
    lines = open(FLAGS.complete_data_file, errors='replace').readlines()

    print("Succesfully took complete dat file located in: ", FLAGS.complete_data_file)

    for i in range(0, len(lines) - 1, 3):
        sentence = lines[i].strip()
        target = lines[i + 1].strip()
        sentiment = lines[i + 2].strip()

        # Tokenize sentence and target
        tokenized_sentence = tokenize_sentence(sentence, tokenizer, word_counts)
        tokenized_target = tokenize_sentence(target, tokenizer, word_counts)

        # Write to output file
        output_f.write(f'{tokenized_sentence}\n')
        output_f.write(f'{tokenized_target}\n')
        output_f.write(f'{sentiment}\n')
    print('Text processing complete. Saved to temporary file')

# NOTE: this path is HARDCODED and serves a s temporary storage for data when we run this program
linesAllData = open('data/programGeneratedData/temp/unique2016_BERT_Data_All.txt').readlines()

with open(FLAGS.train_path,'w') as outTrain, \
        open(FLAGS.test_path,'w') as outTest:
    
    # 2015: 3837 for no augmentation, 7674 BERT-models, 15336 EDA-adjusted, 19185 EDA-original, for all LLM-based DAs the number is 7674 as they do one-to-one augmentations
    # 2016: 5640 for no augmentation, 11280 BERT-models, 22560 EDA-adjusted, 28200 EDA-original, for all LLM-based DAs the numebr is 11280 as they do one-to-one augmentations
    # The numbers above should be pasted intead of the current bound present in the range below
    # The numbers above show the split of the merged data file that has train data (and augmentaed data if DA is TRUE) and test data present located raw_data folder.
    for j in range(0, 7674):
        outTrain.write(linesAllData[j])
    for k in range(7674, len(linesAllData)):
        outTest.write(linesAllData[k])
print('Wrote embedding data to train and test files')