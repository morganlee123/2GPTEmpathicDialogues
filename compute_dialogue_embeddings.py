# Purpose: Compute the embeddings for all dialogues in the all the datasets (Rashkin et al 2019 and 2GPT-EmpathicDialogues)
# and then visualize them in a 2-dimensional space (reduced by UMAP/t-SNE) colored by emotion.
# Runtime is about 1 hr with the OpenAI API
# Author: Morgan Sandler (sandle20@msu.edu)

import openai
import umap
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

import nltk
from transformers import GPT2Tokenizer

from transformers import GPT2Tokenizer


import os

openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key is None:
    raise Exception('OPENAI_API_KEY not found in the environment variables.')


openai.api_key = openai_api_key


def get_embeddings(conv_data, df_new):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    embeddings = []
    for i, conv in tqdm(conv_data.iterrows()):

        conv_id = conv['conv_id']
        dialogue = conv['processed'] # where the dialogue is stored. For human it's 'conversation' for gpt its 'processed'

        try:
            tokens = tokenizer.encode(dialogue)
            truncated_tokens = tokens[:8000]
            if len(tokens) > 8000:
                print(f"Truncating conversation {conv_id} from {len(tokens)} tokens to {len(truncated_tokens)} tokens")
            dialogue_str_truncated = tokenizer.decode(truncated_tokens)

            response = openai.Embedding.create(
                input=dialogue_str_truncated,
                model="text-embedding-ada-002"
            )
            embeddings.append(response['data'][0]['embedding'])

            # Append the conversation to the DataFrame
            df_new = df_new.append({
                'conv_id': conv_id,
                'context': conv['context'],
                'embedding': np.array(response['data'][0]['embedding'], dtype=float)
            }, ignore_index=True)

        except KeyboardInterrupt:
            print("KeyboardInterrupt occurred. Saving data and exiting.")
            df_new.to_pickle('gpt_embeddings.pkl')  # Change here
            return np.array(embeddings), df_new
        except Exception as e:
            print(f"An error occurred: {e}")
            df_new.to_pickle('gpt_embeddings.pkl')  # And here
            return np.array(embeddings), df_new

    df_new.to_pickle('gpt_embeddings.pkl')  # And here
    return np.array(embeddings), df_new



def visualize_embeddings(embeddings):
    print(embeddings.shape)
    print(np.isnan(embeddings).any())
    print(np.isinf(embeddings).any())

    reducer = umap.UMAP()
    embeddings_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.show()
    plt.savefig('gpt_embedding_viz.png')

# Assume dialogues is your dataset
conv_data = pd.read_csv('2gpt_responses.csv', sep=',', header=0, on_bad_lines='warn')
# Assume df is your dataframe
df_new = pd.DataFrame(columns=['conv_id', 'context', 'prompt', 'embedding'])

embeddings, df_new = get_embeddings(conv_data,df_new)
visualize_embeddings(embeddings)
