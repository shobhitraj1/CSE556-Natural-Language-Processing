import torch
from torch.utils.data import Dataset, DataLoader
from task1 import WordPieceTokenizer
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Word2VecDataset(Dataset):
    def __init__(self, corpus, vocab_size, context_window=2, wpt=None):
        self.context_window = context_window

        self.wpt = wpt

        self.oov_token = self.wpt.oov
        self.id_token_map = {idx: token for idx, token in enumerate(self.wpt.vocabulary)} # initilize the vocabulary indices
        self.token_id_map = {token: idx for idx, token in self.id_token_map.items()}
        self.vocab_len = len(self.token_id_map)

        self.pad = self.wpt.pad

        self.preprocess_data(corpus)

    def preprocess_data(self, corpus):
        self.data = []

        if type(corpus) == str:
            corpus = corpus.split("\n")

        for sentence in corpus:
            sentence_tokens = self.wpt.tokenize(sentence) # tokenize the sentence
            for idx, token in enumerate(sentence_tokens):
                if token == self.oov_token:
                    continue

                window = []

                for i in range(-self.context_window, self.context_window + 1): # create the context window for the target word with pad tokens for out of bounds
                    if i == 0 or idx + i < 0 or idx + i >= len(sentence_tokens):
                        if i != 0:
                            window.append(self.token_id_map[self.pad])
                        continue
                    
                    window.append(self.token_id_map[sentence_tokens[idx + i]])

                self.data.append((window, self.token_id_map[token]))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        window, target = self.data[idx]
        target = torch.tensor(target)
        window = torch.tensor(window)
        return window, target
                    

class Word2VecModel(torch.nn.Module):
    def __init__(self, vocab_size=5000, embedding_dim=100, token_id_map=None, pad_token=None):
        super(Word2VecModel, self).__init__()
        self.token_id_map = token_id_map
        if pad_token is not None:
            self.pad_token = pad_token
            self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=self.token_id_map[pad_token]) # embedding layer for the model
        else:
            self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size) # linear layer for the model

    def forward(self, context_window):
        embedded = self.embeddings(context_window)
        # print(embedded.shape)
        return self.linear(embedded.mean(dim=1))
    

def train(corpus, window_size=2, epochs=10, batch_size=64, embedding_dim=100, corpus_size=5000):
    # train test split the corpus
    corpus_lines = corpus.split("\n")
    np.random.shuffle(corpus_lines)
    train_size = int(0.8 * len(corpus_lines)) # train test split
    train_corpus = "\n".join(corpus_lines[:train_size])
    test_corpus = "\n".join(corpus_lines[train_size:])

    tokenizer = WordPieceTokenizer()
    tokenizer.construct_vocabulary(train_corpus, corpus_size) # construct the vocabulary

    train_dataset = Word2VecDataset(train_corpus, corpus_size, window_size, wpt=tokenizer)
    test_dataset = Word2VecDataset(test_corpus, corpus_size, window_size, wpt=tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # dataloaders for the datasets
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = Word2VecModel(train_dataset.vocab_len, embedding_dim, train_dataset.token_id_map, train_dataset.pad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs): # train loop
        train_loss = 0
        for window, target in train_dataloader:
            window = window.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(window)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(window)

        train_losses.append(train_loss / len(train_dataset))

        with torch.no_grad():
            val_loss = 0
            for window, target in test_dataloader:
                window = window.to(device)
                target = target.to(device)
                output = model(window)
                loss = criterion(output, target)
                val_loss += loss.item() * len(window)

            test_losses.append(val_loss / len(test_dataset))

        print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}")

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task2_loss.png")
    plt.show()

    with open("trained_model.pth", "wb") as f: # save the model
        torch.save(model.state_dict(), f)

    return model, tokenizer

def get_cosine_similarity(model, tokenizer, word1, word2):
    word1_tokens = tokenizer.tokenize(word1)
    word2_tokens = tokenizer.tokenize(word2)

    word1_ids = [model.token_id_map[token] for token in word1_tokens]
    word2_ids = [model.token_id_map[token] for token in word2_tokens]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        word1_embedding = torch.stack([model.embeddings(torch.tensor(word1_id).to(device)) for word1_id in word1_ids]).mean(dim=0) # vector representation of the word
        word2_embedding = torch.stack([model.embeddings(torch.tensor(word2_id).to(device)) for word2_id in word2_ids]).mean(dim=0)

    return torch.dot(word1_embedding, word2_embedding) / (torch.norm(word1_embedding) * torch.norm(word2_embedding)) # cosine similarity

if __name__ == "__main__":
    with open('corpus.txt') as f:
        text = f.read()

    model, tokenizer = train(text, epochs=10, embedding_dim=750, window_size=10, corpus_size=10000)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open("token_id_map.pkl", "wb") as f:
        pickle.dump(model.token_id_map, f)

    print(get_cosine_similarity(model, tokenizer, "bike", "wheels"))

# load model
# with open("trained_model.pth", "rb") as f:
#     model = Word2VecModel()
#     model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))
