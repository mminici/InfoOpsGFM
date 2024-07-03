import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Custom Dataset
class TweetDataset(Dataset):
    def __init__(self, dataframe, userids, labels, mask, device):
        self.userids = [userids[i] for i in range(len(userids)) if mask[i]]
        self.dataframe = dataframe[dataframe['userid'].isin(self.userids)]
        self.labels = labels[mask].float()
        self.encoder = SentenceTransformer('stsb-xlm-r-multilingual')
        self.encoder = self.encoder.to(device)
        self.mask = mask
        self.user_embeddings = self.compute_user_embeddings(device)

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
        # remove the user from the dataset
        mask = torch.tensor([True] * len(self.userids)).to(device)
        mask[user_to_be_excluded] = False
        self.labels = self.labels[mask]
        self.userids = [i for j, i in enumerate(self.userids) if j not in user_to_be_excluded]
        return user_embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        userid = self.userids[idx]
        label = self.labels[idx]
        embedding = self.user_embeddings[userid]
        return embedding, label
