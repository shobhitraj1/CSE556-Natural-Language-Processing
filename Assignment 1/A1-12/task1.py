import json
from tqdm import tqdm

class WordPieceTokenizer:
    def __init__(self):
        self.oov = "[UNK]"
        self.pad = "[PAD]"
        self.vocabulary = set([self.oov, self.pad]) # initial set of vocabulary

    def preprocess_data(self, text):
        text = text.lower() # normalization

        for char in text:
            if not char.isalnum():
                text = text.replace(char, ' ') # getting rid of any non-alphanumeric characters

        text = text.split() # tokenization based on space and new line
        while '' in text:
            text.remove('') # removing empty strings
        return text
    
    def get_pair_score(self, word_tokens, word_counter):
        vocab_counter = {} # individual token count
        bigram_counter = {} # bigram count

        for word in word_counter: # iterate through each word and count the tokens
            cur_tokens = word_tokens[word]
            for token in cur_tokens:
                if token not in vocab_counter:
                    vocab_counter[token] = 0
                vocab_counter[token] += word_counter[word]

        for word in word_counter: # iterate through each word and count the bigrams
            cur_tokens = word_tokens[word]

            for i in range(len(cur_tokens) - 1):
                bigram = (cur_tokens[i], cur_tokens[i + 1])

                if bigram not in bigram_counter:
                    bigram_counter[bigram] = 0
                bigram_counter[bigram] += word_counter[word]

        scores = {bigram: count / (vocab_counter[bigram[0]] * vocab_counter[bigram[1]]) for bigram, count in bigram_counter.items()}

        return scores, vocab_counter, bigram_counter
    
    def get_best_pair(self, scores):
        if not scores:
            return None
        return max(scores, key=scores.get) # return token with highest value
    
    def merge_tokens(self, token1, token2, word_token):
        new_token = token1 + token2[2:] # merge the tokens
        
        for word in word_token: # iterate through each word and merge the tokens
            cur_tokens = word_token[word]
            if len(cur_tokens) == 1:
                continue

            to_pop = []

            i = 0

            while i < len(cur_tokens) - 1:
                if cur_tokens[i] == token1 and cur_tokens[i + 1] == token2: # merge tokens and remove the old tokens
                    cur_tokens[i] = new_token
                    to_pop.append(i + 1)
                    i += 1
                i += 1

            for i in to_pop[::-1]:
                cur_tokens.pop(i)

            word_token[word] = cur_tokens

        return word_token
    
    def construct_vocabulary(self, text, size):
        words = self.preprocess_data(text) # preprocess the text

        word_counter = {}
        for word in words:
            if word not in word_counter:
                word_counter[word] = 0
            word_counter[word] += 1 # count the occurrences of each word

        word_tokens = {}

        vocab_counter = {}
        for word in word_counter:
            first = True
            for char in word:
                if first:
                    first = False
                else:
                    char = '##' + char # for the middle tokens

                if char not in vocab_counter:
                    vocab_counter[char] = 0

                vocab_counter[char] += word_counter[word] # count the occurrences of each token in each unique word
                self.vocabulary.add(char)

                if word not in word_tokens:
                    word_tokens[word] = []
                word_tokens[word].append(char)

        scores, vocab_counter, bigram_counter = self.get_pair_score(word_tokens, word_counter)

        with tqdm(total=size) as pbar:
            pbar.set_description("Constructing vocabulary")
            pbar.update(len(self.vocabulary))
            while len(self.vocabulary) < size:
                best_pair = self.get_best_pair(scores) # get the best pair

                if best_pair is None:
                    print("No more pairs to add")
                    break

                token1, token2 = best_pair
                word_tokens = self.merge_tokens(token1, token2, word_tokens) # merge the tokens
                self.vocabulary.add(token1 + token2[2:])

                scores, vocab_counter, bigram_counter = self.get_pair_score(word_tokens, word_counter) # update scores
            
                pbar.update(1)

        with open("vocabulary_12.txt", "w") as f:
            for token in self.vocabulary:
                f.write(token + "\n") # write to a file

    def tokenize_word(self, word): # tokenize a word
        tokens = []
        i = 0

        while i < len(word):
            end = len(word)
            prefix = "##" if i != 0 else ""
            while (prefix + word[i:end]) not in self.vocabulary and end >= i + 1: # find longest prefix which is in the vocabulary
                end -= 1
            if end == i: # if no word is found then return out of vocabulary token
                tokens.append(self.oov)
                i += 1
                return [self.oov]

            tokens.append(prefix + word[i:end])

            i = end

        return tokens
    
    def tokenize(self, sentence):
        words = self.preprocess_data(sentence)
        tokens = []
        for word in words:
            tokens.extend(self.tokenize_word(word)) # process each word in the sentence using the tokenize_word function
        return tokens
    
    def read_vocab_from_text_file(self, text_file):
        with open(text_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.vocabulary.add(line)


if __name__ == "__main__":
    with open('corpus.txt') as f:
        text = f.read()

    wpt = WordPieceTokenizer()
    wpt.construct_vocabulary(text, 5000)

    with open("test.json") as f:
        test = json.load(f)

    test_result = "tokenized_12.json"
    results = []

    for sentence in test:
        results.append({})
        results[-1][sentence["id"]] = wpt.tokenize(sentence["sentence"]) # tokenize each sentence

    with open(test_result, "w") as f:
        json.dump(results, f, indent=4) # write the tokenized sentences to a file