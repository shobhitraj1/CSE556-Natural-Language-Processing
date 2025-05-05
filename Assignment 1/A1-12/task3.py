from task1 import WordPieceTokenizer
from task2 import Word2VecDataset, Word2VecModel
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

with open("tokenizer.pkl", "rb") as f: # load previous results
    tokenizer = pickle.load(f)

with open("token_id_map.pkl", "rb") as f:
    token_id_map = pickle.load(f)

with open("corpus.txt") as f:
    corpus = f.read()
    corpus = [line.strip() for line in corpus.split("\n") if line.strip()]

class NeuralLMDataset(Dataset): # neural lm dataset similar to word2vec dataset
    def __init__(self, corpus, vocab_size, context_window=2, wpt=None):
        self.context_window = context_window

        self.wpt = wpt

        self.oov_token = self.wpt.oov
        self.id_token_map = {idx: token for idx, token in enumerate(self.wpt.vocabulary)}
        self.token_id_map = {token: idx for idx, token in self.id_token_map.items()}
        self.vocab_len = len(self.token_id_map)

        self.pad = self.wpt.pad

        self.preprocess_data(corpus)

    def preprocess_data(self, corpus):
        self.idx2token = {idx: word for idx, word in enumerate(self.wpt.vocabulary)}
        self.data = []

        if type(corpus) == str:
            corpus = corpus.split("\n")

        for sentence in corpus:
            sentence_tokens = self.wpt.tokenize(sentence)
            for idx, token in enumerate(sentence_tokens):
                if token == self.oov_token:
                    continue

                window = []

                for i in range(-self.context_window, 0):
                    if i == 0 or idx + i < 0 or idx + i >= len(sentence_tokens):
                        if i != 0:
                            window.append(self.token_id_map[self.pad])
                        continue
                    
                    window.append(self.token_id_map[sentence_tokens[idx + i]])

                self.data.append((window, self.token_id_map[token]))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target, window = self.data[idx]
        target = torch.tensor(target)
        window = torch.tensor(window)
        return target, window

def get_word_vector(word): # get word vector for a word
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_tokens = tokenizer.tokenize(word)
    word_ids = [token_id_map[token] for token in word_tokens]

    with torch.no_grad():
        embedding = torch.stack([model.embeddings(torch.tensor(word1_id).to(device)) for word1_id in word_ids]).mean(dim=0)

    return embedding

class NeuralLM1(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(NeuralLM1, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.inp_layer = torch.nn.Linear(embedding_dim * context_size, hidden_dim)
        self.ac1 = torch.nn.Tanh()
        self.out_layer = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.inp_layer(x)
        x = self.ac1(x)
        return self.out_layer(x)

class NeuralLM2(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(NeuralLM2, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.inp_layer = torch.nn.Linear(embedding_dim * context_size, hidden_dim)
        self.ac1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.ac2 = torch.nn.Tanh()
        self.out_layer = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.inp_layer(x)
        x = self.ac1(x)
        res = x
        x = self.fc1(x)
        x = self.ac2(x)
        x = x + res
        return self.out_layer(x)

class NeuralLM3(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(NeuralLM3, self).__init__()
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.flatten = torch.nn.Flatten()
        self.inp_layer = torch.nn.Linear(hidden_dim * context_size, hidden_dim)
        self.ac1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim*2)
        self.ac2 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.ac3 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.ac4 = torch.nn.Tanh()
        self.out_layer = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.lstm(x)[0]
        x = self.flatten(x)
        x = self.inp_layer(x)
        x = self.ac1(x)
        res = x
        x = self.fc1(x)
        x = self.ac2(x)
        x = self.fc2(x)
        x = self.ac3(x)
        x = x + res
        x = self.fc3(x)
        x = self.ac4(x)
        return self.out_layer(x)
    
def train(train_loader, test_loader, epochs, hidden_dim, model):

    lm1 = NeuralLM1(vocab_size, embedding_dim, context_size, hidden_dim)
    lm2 = NeuralLM2(vocab_size, embedding_dim, context_size, hidden_dim)
    lm3 = NeuralLM3(vocab_size, embedding_dim, context_size, hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm1.to(device)
    lm2.to(device)
    lm3.to(device)
    optimizer1 = torch.optim.Adam(lm1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(lm2.parameters(), lr=0.001)
    optimizer3 = torch.optim.Adam(lm3.parameters(), lr=0.001)

    model_optim = [(lm1, optimizer1, torch.nn.CrossEntropyLoss()), (lm2, optimizer2, torch.nn.CrossEntropyLoss()), (lm3, optimizer3, torch.nn.CrossEntropyLoss())] # indivisual losses for each model

    train_losses = [list(), list(), list()]
    test_losses = [list(), list(), list()]

    for epoch in range(epochs): # train loop
        train_loss = [0, 0, 0]
        train_correct = [0, 0, 0]
        for i, (window, target) in tqdm(enumerate(train_loader)):
            window = torch.stack([model.embeddings(window[:, i]) for i in range(context_size)], dim=1)
            window = window.to(device)
            target = target.to(device)
            
            idx = 0
            for lm, optimizer, criterion in model_optim:
                optimizer.zero_grad()
                output = lm(window)
                loss = criterion(output, target)
                loss.backward(retain_graph=True)
                optimizer.step()

                predicted = torch.argmax(output, dim=1)
                train_correct[idx] += (predicted == target).sum().item()

                train_loss[idx] += loss.item()
                idx += 1
            
        for i in range(3):
            model_optim[i][0].eval()
            train_losses[i].append(train_loss[i] / len(train_loader))

        test_correct = [0, 0, 0]
        test_loss = [0, 0, 0]
        for i, (window, target) in enumerate(test_loader):
            window = torch.stack([model.embeddings(window[:, i]) for i in range(context_size)], dim=1)
            window = window.to(device)
            target = target.to(device)

            idx = 0
            for lm, optim, _ in model_optim:
                output = lm(window)
                loss = criterion(output, target)
                test_loss[idx] += loss.item()
                predicted = torch.argmax(output, dim=1)
                test_correct[idx] += (predicted == target).sum().item()
                idx += 1

        for i in range(3):
            model_optim[i][0].train()
            test_losses[i].append(test_loss[i] / (len(test_loader)))

        for i in range(3):
            print(f"Epoch: {epoch}, Model: NeuralLM{i+1}, Train Loss Model {i+1}: {train_losses[i][-1]}, Test Loss Model {i+1}: {test_losses[i][-1]}")

    print("Training Complete")
    for i in range(3):
        print(f"Model {i+1}: Train Accuracy: {train_correct[i] / len(train_loader)}; Train Perplexity: {np.exp(train_losses[i][-1])}; Test Accuracy: {test_correct[i] / len(test_loader)}; Test Perplexity: {np.exp(test_losses[i][-1])}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    for i in range(3):
        axs[i].plot(train_losses[i], label=f"Train Loss Model {i+1}")
        axs[i].plot(test_losses[i], label=f"Test Loss Model {i+1}")
        axs[i].set_title(f"Model {i+1} Loss")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()

    plt.tight_layout()
    plt.savefig("task3_loss.png")
    plt.show()

    for i in range(3):
        with open(f"trained_model_{i+1}.pth", "wb") as f: # save the weights
            torch.save(model_optim[i][0].state_dict(), f)

    return model_optim

def generate_text(model_optim, context_size, text_file, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(len(model_optim)):
        print(f"Output of Model {i+1}")
        lm, optimizer, _ = model_optim[i]
        lm.eval()
        with open(text_file) as f:
            test_sentences = f.read().split("\n")
            for sentence in test_sentences: # form the sentence context
                sentence_tokens = tokenizer.tokenize(sentence)
                while len(sentence_tokens) < context_size:
                    sentence_tokens = [tokenizer.pad] + sentence_tokens
                sentence_ids = [token_id_map[token] for token in sentence_tokens]
                sentence_ids = torch.tensor(sentence_ids).to(device)

                with torch.no_grad():
                    model = model.to(device)
                    for new_token in range(3):
                        window = torch.stack([model.embeddings(sentence_ids[i]) for i in range(-context_size, 0)], dim=0) # find next 3 vectors
                        window = window.unsqueeze(0)
                        window = window.to(device)
                        output = lm(window)
                        next_token_id = torch.argmax(output, dim=1)
                        for token in token_id_map:
                            if token_id_map[token] == next_token_id:
                                next_token = token
                                break
                        sentence_tokens.append(next_token)
                        sentence_ids = torch.cat([sentence_ids[1:], next_token_id])

                for token in sentence_tokens:
                    if token[0] != "#":
                        print(f" {token}", end="")
                    else:
                        print(token[2:], end="")

                print()

if __name__ == "__main__":
    vocab_size = 10000
    embedding_dim = 750
    context_size = 15
    batch_size = 128

    model = Word2VecModel(vocab_size=vocab_size, embedding_dim=embedding_dim, token_id_map=token_id_map, pad_token=tokenizer.pad)

    with open("trained_model.pth", "rb") as f:
        model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

    train_corpus, test_corpus = train_test_split(corpus, test_size=0.2, random_state=42)

    train_dataset = NeuralLMDataset(train_corpus, vocab_size, wpt=tokenizer, context_window=context_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = NeuralLMDataset(test_corpus, vocab_size, wpt=tokenizer, context_window=context_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    epochs = 5
    hidden_dim = 256
    model_optim = train(train_loader, test_loader, epochs, hidden_dim, model)
    generate_text(model_optim, context_size, "test.txt", model)
    
    # lm1 = NeuralLM1(vocab_size, embedding_dim, context_size, hidden_dim)
    # lm2 = NeuralLM2(vocab_size, embedding_dim, context_size, hidden_dim)
    # lm3 = NeuralLM3(vocab_size, embedding_dim, context_size, hidden_dim)

    # with open("trained_model_1.pth", "rb") as f:
    #     lm1.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

    # with open("trained_model_2.pth", "rb") as f:
    #     lm2.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

    # with open("trained_model_3.pth", "rb") as f:
    #     lm3.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

    # model_optim = [(lm1, None, None), (lm2, None, None), (lm3, None, None)]

    # generate_text(model_optim, context_size, "test.txt", model)
