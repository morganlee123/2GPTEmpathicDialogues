# CAUTION: OLD SCRIPT. I HIGHLY RECOMMEND USING THE 2GPT STRUCTURE SCRIPT IN THIS DIRECTORY

# Author: Morgan Sandler (sandle20@msu.edu)
# Purpose: To generate a empathic dialogue conversation given only the
# prompt and the situation. Uses the empathicdialogues dataset from FB as a
# seed/prompt for the ChatGPT (4) model

import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import openai
import tqdm
import time
import sys

nltk.download('punkt')
nltk.download('stopwords')

import os

openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key is None:
    raise Exception('OPENAI_API_KEY not found in the environment variables.')


openai.api_key = openai_api_key


start_from = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # Get start index from command line arguments


train_df = pd.read_csv('empatheticdialogues/train.csv', sep=',', header=0, on_bad_lines='warn')

print(len(train_df['conv_id'].unique()), 'samples loaded from train csv')

# Assume df is your dataframe
grouped = train_df.groupby('conv_id')

if start_from > 0:
  df_new = pd.read_csv('wocontext_Saves_gpt4/'+str(start_from)+'_gpt_responses.csv')
else:
  df_new = pd.DataFrame(columns=['conv_id', 'context', 'prompt', 'gptgen'])

gg = list(grouped.groups.keys())

try:
  print('starting from line', start_from, 'with entry\n', gg[start_from])
  for idx, i in tqdm.tqdm(enumerate(gg[start_from:])):
    # Let's say we are interested in the first conversation.
    first_conv_id = i
    first_conversation = grouped.get_group(first_conv_id).sort_values('utterance_idx')
    #print(first_conversation)

    # Form the prompt string
    # ORIGINAL
    #prompt_string = f'Provided a label and a situation, generate an empathic conversation between a speaker and a listener (the empathic one). Generate 3-4 back-and-forth dialogues\nEmotion/Context: {first_conversation.iloc[0].context},  Situation: {first_conversation.iloc[0].prompt}\n'
    # W/O Context
    prompt_string = f'Provided a label and a situation, generate an empathic conversation between a speaker and a listener (the empathic one). Generate 3-4 back-and-forth dialogues\nEmotion/Context: {first_conversation.iloc[0].context},  Situation: {first_conversation.iloc[0].prompt}\n\nThe listener must infer using their emotional intelligence the emotion/context and situation information\n'

    curr_chat = [
        {"role": "assistant", "content":prompt_string},
    ]
    while True:
        try:
         #print(prompt_string)
          response = openai.ChatCompletion.create(
            model = "gpt-4",
            messages=curr_chat,
            temperature=0.5,
            max_tokens=500
          )
          break  # If the request was successful, exit the loop

        except openai.error.OpenAIError as e:
          print(f"API error ({e}): Pausing for a minute before retrying.")
          time.sleep(60)  # Wait for 60 seconds before retrying

    gptgen = response['choices'][0]['message']['content']

    # Add the new row to the dataframe
    df_new = df_new.append({
            'conv_id': first_conv_id,
            'context': first_conversation.iloc[0].context,  # Last context in the conversation
            'prompt': first_conversation.iloc[0].prompt,  # Last prompt in the conversation
            'gptgen': gptgen
        }, ignore_index=True)

    # Save the dataframe every 2 responses
    if (idx + 1) % 2 == 0:
      df_new.to_csv('wocontext_Saves_gpt4/'+str(start_from+idx + 1)+'_gpt_responses.csv', index=False)
      print('Saved to wocontext_Saves_gpt4/'+str(start_from+idx + 1)+'_gpt_responses.csv')
except KeyboardInterrupt:
  print('Keyboard interrupt. Stopping early and saving')
finally:
  # Write the new dataframe to a CSV file
  df_new.to_csv('wocontext_Saves_gpt4/gpt_responses.csv', index=False)