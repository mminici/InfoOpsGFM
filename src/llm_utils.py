import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def average_embedding(user_embeddings):
    """
    Calculate the average embedding across all users.

    Parameters:
    user_embeddings (dict): A dictionary where keys are user IDs and values are torch tensors.

    Returns:
    torch.Tensor: The average embedding.
    """
    # Ensure there are embeddings to average
    if not user_embeddings:
        raise ValueError("The user_embeddings dictionary is empty.")

    # Initialize the sum tensor
    total_embedding = None
    num_users = 0

    for user_id, embedding in user_embeddings.items():
        if total_embedding is None:
            # Initialize the total_embedding tensor with the shape of the first embedding
            total_embedding = torch.zeros_like(embedding)
        total_embedding += embedding
        num_users += 1

    # Compute the average
    average_embedding = total_embedding / num_users

    return average_embedding


# Custom Dataset
class TweetDataset(Dataset):
    def __init__(self, dataframe, userids, labels, mask, device, pathToUserEmbed=None):
        self.userids = [userids[i] for i in range(len(userids)) if mask[i]]
        self.dataframe = dataframe[dataframe['userid'].isin(self.userids)]
        self.labels = labels[mask].float()
        self.encoder = SentenceTransformer('stsb-xlm-r-multilingual')
        self.encoder = self.encoder.to(device)
        self.mask = mask
        if pathToUserEmbed is None:
            self.user_embeddings = self.compute_user_embeddings(device)
        else:
            self.user_embeddings = load_user_embeddings(pathToUserEmbed)

    def compute_user_embeddings(self, device):
        user_embeddings = {}
        user_to_be_excluded = list()
        for i in tqdm(range(len(self.userids))):
            userid = self.userids[i]
            user_tweets = self.dataframe[self.dataframe['userid'] == userid]['clean_tweet']
            tweet_list = user_tweets.tolist()
            if len(tweet_list) > 0:
                embeddings = self.encoder.encode(user_tweets.tolist())
                avg_embedding = torch.tensor(embeddings.mean(axis=0)).to(device)
                user_embeddings[userid] = avg_embedding
            else:
                # remove the user from the dataset
                user_to_be_excluded.append(i)
        # Compute the average embedding
        population_embedding_vec = average_embedding(user_embeddings)
        for i in user_to_be_excluded:
            userid = self.userids[i]
            user_embeddings[userid] = population_embedding_vec

        # remove the user from the dataset
        # mask = torch.tensor([True] * len(self.userids)).to(device)
        # mask[user_to_be_excluded] = False
        # self.labels = self.labels[mask]
        # self.userids = [i for j, i in enumerate(self.userids) if j not in user_to_be_excluded]
        return user_embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        userid = self.userids[idx]
        label = self.labels[idx]
        embedding = self.user_embeddings[userid]
        return embedding, label
