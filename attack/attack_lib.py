from nltk.tokenize import RegexpTokenizer
from english import ENGLISH_FILTER_WORDS
import random
import sys
from word2vec.word2vec_embed import Word2VecSubstitute
import shutil
import numpy as np
from compute_img_sim import compute_img_sim
from tqdm import tqdm

class Error(Exception):
    """Base class for other exceptions"""
    pass

class WordNotInDictionaryException(Error):
    """Raised when the input value is too small"""
    pass

class attack():
    def __init__(self, tar_sent):
        
        self.count = 0

        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(tar_sent.lower())

        #filter unimportant word
        target_word_ls = []
        for token in tokens:
            if token.lower() in ENGLISH_FILTER_WORDS:
                continue
            target_word_ls.append(token)
        self.target_sent_tokens = target_word_ls
        print("tar_sent_tokens: ", self.target_sent_tokens)

        self.Word2vec = Word2VecSubstitute(tar_tokens=self.target_sent_tokens)

        print("initialize attack class.")

    
    def selectBug(self, original_word, sent_token, word_pos, sub_word_pos):
        target_num = random.randint(0, 6)
        bug_choice = self.generateBugs(target_num, original_word, sent_token, word_pos, sub_word_pos)
        return bug_choice
    

    def replaceWithBug(self, x_prime, word_idx, bug):
        return x_prime[:word_idx] + [bug] + x_prime[word_idx + 1:]

    def generateBugs(self, target_num, word, sent_token, word_pos, sub_word_pos):
        
        if len(word) <= 2:
            return word

        if target_num == 0:  
            bugs = self.bug_sub_W(word, sent_token, word_pos, sub_word_pos)
        elif target_num == 1:
            bugs = self.bug_insert(word)
        elif target_num == 2:
            bugs = self.bug_delete(word)
        elif target_num == 3:
            bugs = self.bug_sub_tar_W(word, sent_token, word_pos, sub_word_pos)
        elif target_num == 4:
            bugs = self.bug_swap(word)
        elif target_num == 5:
            bugs = self.bug_sub_C(word)
        elif target_num == 6:
            bugs = self.bug_convert_to_leet(word)

        return bugs

    def bug_sub_tar_W(self, word, sent_token, word_pos, sub_word_pos):
        word_index = random.randint(0, len(self.target_sent_tokens) - 1)
        tar_word = self.target_sent_tokens[word_index]
        res = self.Word2vec.substitute(tar_word, sent_token, word_pos, sub_word_pos)
        if len(res) == 0:
            return word
        return res[0][0]

    def bug_sub_W(self, word, sent_token, word_pos, sub_word_pos):
        try:
            res = self.Word2vec.substitute(word, sent_token, word_pos, sub_word_pos)
            if len(res) == 0:
                return word
            return res[0][0]
        except WordNotInDictionaryException:
            return word

    def bug_insert(self, word):
        if len(word) >= 6:
            return word
        res = word
        point = random.randint(1, len(word) - 1)
        #insert _ instread " "
        res = res[0:point] + "_" + res[point:]
        return res

    def bug_delete(self, word):
        res = word
        point = random.randint(1, len(word) - 2)
        res = res[0:point] + res[point + 1:]
        return res

    def bug_swap(self, word):
        if len(word) <= 4:
            return word
        res = word
        points = random.sample(range(1, len(word) - 1), 2)
        a = points[0]
        b = points[1]

        res = list(res)
        w = res[a]
        res[a] = res[b]
        res[b] = w
        res = ''.join(res)
        return res

    def bug_random_sub(self, word):
        res = word
        point = random.randint(0, len(word)-1)

        choices = "qwertyuiopasdfghjklzxcvbnm"
        
        subbed_choice = choices[random.randint(0, len(list(choices))-1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)
        return res
    
    def bug_convert_to_leet(self, word):
        # Dictionary that maps each letter to its leet speak equivalent.
        leet_dict = {
            'a': '4',
            'b': '8',
            'e': '3',
            'g': '6',
            'l': '1',
            'o': '0',
            's': '5',
            't': '7'
        }
        
        # Replace each letter in the text with its leet speak equivalent.
        # res = ''.join(leet_dict.get(c.lower(), c) for c in word)

        leet_idx = []
        for i, c in enumerate(word):
            if c in leet_dict.keys():
                leet_idx.append(i)
        if len(leet_idx) < 1:
            return word
        rnd_idx = np.random.choice(leet_idx)

        res = word[:rnd_idx] + leet_dict.get(word[rnd_idx].lower(), word[rnd_idx]) + word[rnd_idx + 1:]
        
        return res


    def bug_sub_C(self, word):
        res = word
        key_neighbors = self.get_key_neighbors()
        point = random.randint(0, len(word) - 1)

        if word[point] not in key_neighbors:
            return word
        choices = key_neighbors[word[point]]
        subbed_choice = choices[random.randint(0, len(choices) - 1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)

        return res

    def get_key_neighbors(self):
        ## TODO: support other language here
        # By keyboard proximity
        neighbors = {
            "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
            "i": "uojkl", "o": "ipkl", "p": "ol",
            "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
            "j": "yuihknm", "k": "uiojlm", "l": "opk",
            "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
        }
        # By visual proximity
        neighbors['i'] += '1'
        neighbors['l'] += '1'
        neighbors['z'] += '2'
        neighbors['e'] += '3'
        neighbors['a'] += '4'
        neighbors['s'] += '5'
        neighbors['g'] += '6'
        neighbors['b'] += '8'
        neighbors['g'] += '9'
        neighbors['q'] += '9'
        neighbors['o'] += '0'

        return neighbors
