from collections import Counter

with open("data/text.txt", "r") as f:
    text = f.read()

# Splits each word into subslist of single chars (symbols)
def words_to_symbols(text: str) -> list[list[str]]:
    symbol_words = [list(word) for word in text.split()]    # splits each word into a sublist of characters (symbols)
    return symbol_words

# Testing
symbol_words = words_to_symbols(text)

# Creates list of pairs with a count, from symbol words (text)
def get_pair_stats(symbol_words: list[list[str]]) -> Counter[tuple[str, str]]:
    pairs = []
    for word in symbol_words:           # creates a list of pairs
        for i in range(len(word) - 1):
            pairs.append((word[i], word[i+1]))
    return Counter(pairs)               # counts pairs

# Testing --- Returns counter object
stats = get_pair_stats(words_to_symbols(text))

# Sorts stats and returns most common pair
def best_pair(stats) -> tuple[str,str]:
    return stats.most_common()[0][0]    # returns the first pair from a sorted stats list

# merges pair of symbols to create new token and replaces both with new token, returns a new list
def merge_pairs(symbol_words: list[list[str]], pair: tuple[str, str]) -> list[list[str]]:
    new_tok = pair[0] + pair[1]         # creates a new token from the pair tuple
    new_words = []

    for word in symbol_words:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and pair[0] == word[i] and pair[1] == word[i+1]: # checks if pair matches the letters in word
                new_word.append(new_tok)    # if so, adds new token to new word and jumps next symbol
                i += 2
            else:                           # else, adds the symbol to new word
                new_word.append(word[i])
                i += 1
        new_words.append(new_word)          # appends word to new words list
    return new_words

# Testing - after single merge
new_words = merge_pairs(symbol_words, best_pair(stats))

# runs a merge (training) loop a certain number of times
def train_bpe(text: str, num_merges: int) -> tuple[list[tuple[str, str]], list[list[str]]]:
    symbol_words = words_to_symbols(text)
    merges = []
    for step in range(num_merges):
        stats = get_pair_stats(symbol_words)
        if not stats: break
        pair = best_pair(stats)
        merges.append(pair)
        symbol_words = merge_pairs(symbol_words, pair)
        # print(f"{step:02d} merge {pair} count ={stats[pair]}")

    return (merges, symbol_words)

# Testing for 10 merges
merges, final_words = train_bpe(text, num_merges=10)

# --- Encode/Decode ---
# applies merge to single word
def apply_merges(symbols: list[str], merges: list[tuple[str, str]]) -> list[str]:
    new_word = []
    new_word.append(symbols)
    for pair in merges:
        new_word = merge_pairs(new_word, pair)
    return new_word[0]

# Encode - apply merges to full text
def encode(text: str, merges: list[tuple[str, str]]) -> list[list[str]]:
    output = []
    symbol_words = words_to_symbols(text)
    for word in symbol_words:
        output.append(apply_merges(word, merges))

    return output

# Decode - returns original text
def decode(tokens: list[list[str]]) -> str:
    decoded_list = []
    for word in tokens:
        decoded = "".join(word)
        decoded_list.append(decoded)
    return " ".join(decoded_list)



# print(apply_merges(list("wider"), merges))
# print(apply_merges(list("lowest"), merges))
# print(apply_merges(list("low"), merges))
tokens = encode(text, merges)
decoded = decode(tokens)
print(decoded)