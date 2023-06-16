from torch.utils.data import Dataset

import pandas as pd
from sklearn.model_selection import train_test_split

# Load JSON file into pandas DataFrame
df = pd.read_json('intents.json')

# Split data into features (X) and labels (y)
X = df['intents']
y = df['patterns']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class ChatDataset(Dataset):
    def __init__(self, data, tags):
        self.data = data
        self.tags = tags
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        words, label = self.data[idx]
        
        # Convert words to bag-of-words representation
        bow = [0]*len(self.tags)
        for w in words:
            for i, t in enumerate(self.tags):
                if t == w:
                    bow[i] = 1
                    
        return torch.tensor(bow, dtype=torch.float32), label
