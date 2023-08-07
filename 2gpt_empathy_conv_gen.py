# Author: Morgan Sandler (sandle20@msu.edu)
# Purpose: To generate a empathic dialogue conversation given only the
# prompt and the situation. This version assumes no mutual memory and has two separate
# ChatGPT instances communicate with each other -- similar to how two humans may interact.
# Uses the empathicdialogues dataset from FB as a seed/prompt for the ChatGPT (4) model

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

start_from = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # Get start index from command line arguments


train_df = pd.read_csv('empatheticdialogues/train.csv', sep=',', header=0, on_bad_lines='warn')

print(len(train_df['conv_id'].unique()), 'samples loaded from train csv')

import os

openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key is None:
    raise Exception('OPENAI_API_KEY not found in the environment variables.')


openai.api_key = openai_api_key


# Assume df is your dataframe
grouped = train_df.groupby('conv_id')


def converse(prompt1, prompt2, max_tokens=250, num_rounds=2):
    conversation = []

    message1 = {"role": "system", "content": prompt1}
    message2 = {"role": "system", "content": prompt2}

    curr_chat1 = [message1]
    curr_chat2 = [message2]

    while True:
      try:
          for round in range(num_rounds):
              response1 = openai.ChatCompletion.create(
                model="gpt-4",
                messages=curr_chat1,
                temperature=0.5,
                max_tokens=max_tokens
              )

              assistant1_message = response1.choices[0].message['content']
              conversation.append({"role": "assistant1", "content": assistant1_message})

              message2 = {"role": "user", "content": assistant1_message}
              curr_chat2.append(message2)

              response2 = openai.ChatCompletion.create(
                  model="gpt-4",
                  messages=curr_chat2,
                  temperature=0.5,
                  max_tokens=max_tokens
              )

              assistant2_message = response2.choices[0].message['content']
              conversation.append({"role": "assistant2", "content": assistant2_message})

              message1 = {"role": "user", "content": assistant2_message}
              curr_chat1.append(message1)

          #for msg in conversation:
          #    print(f'{msg["role"]}: {msg["content"]}\n')

          return conversation
      except openai.error.OpenAIError as e:
          print(f"API error ({e}): Pausing for a minute before retrying.")
          time.sleep(60)  # Wait for 60 seconds before retrying

if start_from > 0:
  df_new = pd.read_csv('2gpt_gpt4saves/'+str(start_from)+'_gpt_responses.csv')
else:
  df_new = pd.DataFrame(columns=['conv_id', 'context', 'prompt', 'gptgen'])

gg = list(grouped.groups.keys())

try:
  print('starting from line', start_from, 'with entry\n', gg[start_from])
  for idx, i in tqdm.tqdm(enumerate(gg[start_from:])):

    first_conv_id = i
    first_conversation = grouped.get_group(first_conv_id).sort_values('utterance_idx')

    #prompt1 = f'Provided a label and a situation, pretend you are the one in the situation and interact with your friend (the listener). Followup messages will include the listener\'s responses.\nEmotion/Context: {first_conversation.iloc[0].context},  Situation: {first_conversation.iloc[0].prompt}. Do not generate the listener response. You will begin the conversation.'
    #prompt2 = f'You are an empathic listener having a conversation with a friend. Followup messages will include the speaker\'s responses.\nUse your emotional intelligence to infer the emotion/context and situation information. Do not generate the speaker response.'
    #prompt1 = f"You are in a certain situation and feeling a particular emotion. Your role is to express these emotions to a friend in a conversation. You should not predict or generate the friend's response; that's their role. Here's the situation:\n\nEmotion/Context: {first_conversation.iloc[0].context}\nSituation: {first_conversation.iloc[0].prompt}\n\nBegin your conversation."
    #prompt2 = f"You are an empathic friend. Your role is to listen, understand, and respond to a friend who's expressing their feelings and situation. It's not your role to express their feelings for them. Your responses should follow what they have expressed. Be ready for their next message."
    #prompt1 = f'You are in a situation where {first_conversation.iloc[0].context}, and {first_conversation.iloc[0].prompt}. The conversation will continue based on responses you receive. Do not generate responses for the other party.'
    #prompt2 = f'You are an empathic listener in a conversation. Use your emotional intelligence to respond to the situation described. Do not generate statements for the other party.'
    prompt1 = f'You are in a conversation where you {first_conversation.iloc[0].context}, and {first_conversation.iloc[0].prompt}. You are the speaker in the conversation and your responses will be based on the listener\'s replies. Do not generate responses for the listener.'

    prompt2 = f'You are a listener in a conversation. As an empathic listener, use your emotional intelligence to respond to the speaker\'s statements. Do not generate statements for the speaker.'


    # Call the function
    conv = converse(prompt1, prompt2)


    # Format the conversation into a string
    conv_string = "\n".join([f'{c["role"]}: {c["content"]}' for c in conv])

    # Append the conversation to the DataFrame
    df_new = df_new.append({
        'conv_id': first_conv_id,
        'context': first_conversation.iloc[0].context,
        'prompt': first_conversation.iloc[0].prompt,
        'gptgen': conv_string
    }, ignore_index=True)



    # Save the dataframe every 2 responses
    if (idx + 1) % 2 == 0:
      df_new.to_csv('2gpt_gpt4saves/'+str(start_from+idx + 1)+'_gpt_responses.csv', index=False)
      print('Saved to 2gpt_gpt4saves/'+str(start_from+idx + 1)+'_gpt_responses.csv')
except KeyboardInterrupt:
  print('Keyboard interrupt. Stopping early and saving')
finally:
  # Write the new dataframe to a CSV file
  df_new.to_csv('2gpt_gpt4saves/gpt_responses.csv', index=False)

