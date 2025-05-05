from task2 import Word2VecModel
import torch
import pickle

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("token_id_map.pkl", "rb") as f:
    token_id_map = pickle.load(f)

with open("corpus.txt") as f:
    corpus = f.read()
    corpus = [line.strip() for line in corpus.split("\n") if line.strip()]

words = set([word for line in corpus for word in line.split()])

vocab_size = 10000
embedding_dim = 750
context_size = 15

model = Word2VecModel(vocab_size=vocab_size, embedding_dim=embedding_dim, token_id_map=token_id_map, pad_token=tokenizer.pad)

with open("trained_model.pth", "rb") as f:
    model.load_state_dict(torch.load(f))

def get_cosine_similarity(model, tokenizer, word1, word2):
    word1_tokens = tokenizer.tokenize(word1)
    word2_tokens = tokenizer.tokenize(word2)

    word1_ids = [model.token_id_map[token] for token in word1_tokens]
    word2_ids = [model.token_id_map[token] for token in word2_tokens]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        word1_embedding = torch.stack([model.embeddings(torch.tensor(word1_id).to(device)) for word1_id in word1_ids]).mean(dim=0)
        word2_embedding = torch.stack([model.embeddings(torch.tensor(word2_id).to(device)) for word2_id in word2_ids]).mean(dim=0)

    return torch.dot(word1_embedding, word2_embedding) / (torch.norm(word1_embedding) * torch.norm(word2_embedding))

model.eval()

def get_scores(word):
    scores = {}
    for w in words:
        scores[w] = get_cosine_similarity(model, tokenizer, word, w).item()

    for w, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(w, s)

def get_opposite(word):
    scores = {}
    for w in words:
        scores[w] = get_cosine_similarity(model, tokenizer, word, w).item()

    for w, s in sorted(scores.items(), key=lambda x: x[1])[:10]:
        print(w, s)

def get_dissimilar(word):
    scores = {}
    for w in words:
        scores[w] = abs(get_cosine_similarity(model, tokenizer, word, w).item())

    for w, s in sorted(scores.items(), key=lambda x: x[1])[:10]:
        print(w, s)

tiplets = [("see", "sense", "illusion"), ("bike", "wheels", "fly")]

for tiplet in tiplets:
    print(tiplet)
    for word1_idx in range(len(tiplet)):
        for word2_idx in list(range(len(tiplet)))[word1_idx+1:]:
            word1, word2 = tiplet[word1_idx], tiplet[word2_idx]
            similarity = get_cosine_similarity(model, tokenizer, word1, word2).item()
            print(f"Cosine similarity between {word1} and {word2}: {similarity:.3f}")