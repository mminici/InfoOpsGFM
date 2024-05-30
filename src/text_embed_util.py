import concurrent.futures
import pickle
import torch
import pandas as pd

from transformers import AutoTokenizer, XLMRobertaModel
from data_loader import filename_dict, IO_FILE_IDX, CONTROL_FILE_IDX
from tqdm import tqdm


def _split_list_into_buckets(lst, num_buckets):
    if not lst:
        return [[] for _ in range(num_buckets)]

    avg_length = len(lst) // num_buckets
    remainder = len(lst) % num_buckets

    buckets = []
    start = 0

    for i in range(num_buckets):
        if i < remainder:
            end = start + avg_length + 1
        else:
            end = start + avg_length

        buckets.append(lst[start:end])
        start = end

    return buckets


def get_tweet_embed(base_dir, dataset_name, noderemapping, noderemapping_rev,
                    num_cores, num_tweet_to_sample, aggr_type, device):
    processed_data_dir = base_dir / 'data' / 'processed' / dataset_name
    # Save the obtained network on disk
    valid_user_ids = list(noderemapping.keys())
    # Read raw data
    data_dir = base_dir / 'data' / 'raw'
    treated = pd.read_csv(data_dir / dataset_name / filename_dict[dataset_name][IO_FILE_IDX], sep=",")
    control = pd.read_json(data_dir / dataset_name / filename_dict[dataset_name][CONTROL_FILE_IDX], lines=True)
    control['userid'] = control['user'].apply(lambda x: str(dict(x)['id']))  # Retrieve user ID
    # filter data by considering only user in the similarity network
    treated = treated[treated['userid'].astype(str).isin(valid_user_ids)]
    control = control[control['userid'].astype(str).isin(valid_user_ids)]
    # Clean data
    control, treated = _clean_data_for_text_analysis(control, treated)
    # Join the two dataframes
    control['text'] = control['full_text_cleaned']
    treated['text'] = treated['tweet_text_cleaned']
    control = control[['userid', 'text']]
    treated = treated[['userid', 'text']]
    stacked_df = pd.concat([control, treated], ignore_index=True)
    # Divide user population in @num_cores buckets
    user_ids = list(range(len(valid_user_ids)))
    user_ids_buckets = _split_list_into_buckets(user_ids, num_cores)
    user_ids_buckets_raw_fmt = _convert_inner_lists(user_ids_buckets, noderemapping_rev)
    get_embeddings_from_user_tweets_fn = lambda bucket_idx: _get_tweet_embeddings(
        stacked_df[stacked_df['userid'].astype(str).isin(user_ids_buckets_raw_fmt[bucket_idx])],
        user_ids_buckets_raw_fmt[bucket_idx],
        num_tweet_to_sample, aggr_type, device)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the function to the data list using multiple threads
        embeds = list(executor.map(get_embeddings_from_user_tweets_fn, range(len(user_ids_buckets))))
        embeds = torch.vstack(embeds)
    return embeds


def _clean_data_for_text_analysis(control, treated):
    control['tweet_length'] = control['full_text'].apply(lambda x: len(x))
    treated['tweet_length'] = treated['tweet_text'].apply(lambda x: len(x))
    import re
    import emoji
    def _cleaner(tweet):
        tweet = re.sub("@[A-Za-z0-9]+", "", tweet)  # Remove @ sign
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)  # Remove http links
        tweet = " ".join(tweet.split())
        tweet = emoji.replace_emoji(tweet, '')  # Remove Emojis
        tweet = tweet.replace("#", "").replace("_", " ")  # Remove hashtag sign but keep the text
        return tweet

    control['full_text_cleaned'] = control['full_text'].apply(lambda x: _cleaner(x))
    treated['tweet_text_cleaned'] = treated['tweet_text'].apply(lambda x: _cleaner(x))
    # Remove tweets with length < 20
    control['tweet_length_cleaned'] = control['full_text_cleaned'].apply(lambda x: len(x))
    treated['tweet_length_cleaned'] = treated['tweet_text_cleaned'].apply(lambda x: len(x))
    control = control[control['tweet_length_cleaned'] > 20]
    treated = treated[treated['tweet_length_cleaned'] > 20]
    return control, treated


def _convert_inner_lists(list_of_lists, id_to_id2_dict):
    return [list(map(lambda x: id_to_id2_dict.get(x, x), inner_list)) for inner_list in list_of_lists]


def _get_tweet_embeddings(users_df, users_list, num_tweet_to_sample, aggr_type, device):
    # Create list that will host all embeddings
    user_embeddings_list = []
    # Create embedding model
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    model = XLMRobertaModel.from_pretrained("xlm-roberta-large")
    model.to(device)

    # Function to get [CLS] token embedding for a given text
    def _get_cls_embedding(text, tokenizer, model, device):
        # Tokenize input text
        sentence = torch.tensor([tokenizer.encode(text)]).to(device)
        # Get model output
        with torch.no_grad():
            output = model(sentence)
        # Extract last layer token embeddings
        hidden_states = output[-1]
        # Extract [CLS] token embeddings
        return hidden_states[0]

    for userid_i in tqdm(users_list, 'Get user embeds'):
        user_tweets_sample = users_df[users_df.userid.astype(str) == userid_i]
        if num_tweet_to_sample is not None:
            user_tweets_sample = user_tweets_sample.sample(min([num_tweet_to_sample, len(user_tweets_sample)]))
        user_tweets_embeddings = []
        for tweet_text in user_tweets_sample['text']:
            user_tweets_embeddings.append(_get_cls_embedding(tweet_text, tokenizer, model, device))
        if aggr_type == 'mean':
            user_embeddings_list.append(torch.vstack(user_tweets_embeddings).mean(dim=0))
        elif aggr_type == 'max':
            user_embeddings_list.append(torch.vstack(user_tweets_embeddings).max(dim=0).values)
        else:
            raise Exception(f'{aggr_type} is not supported.')
    return torch.vstack(user_embeddings_list)
