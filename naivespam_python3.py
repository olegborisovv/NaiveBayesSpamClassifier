import os
import email
import time
import collections
import math
import codecs


LONGWORD_CHAR_NUM = 12

def get_paths(directory):
    path = []
    for subdir, dirs, files in os.walk(directory):
        for fil in files:
            filepath = subdir + os.sep + fil
            path.append(filepath)
    return path


def tokenize_email(email_path: str):
    with codecs.open(email_path, encoding='utf-8', errors='ignore') as f:
        m = email.message_from_file(f)
        l = email.iterators.body_line_iterator(m)
        tokenized_email = []
        for el in l:
            el = el.lower()
            for char in "!?,:;":
                el = el.replace(char, ' ')

            el = el.split()
            if el:
                tokenized_email += el
    return tokenized_email


def log_probs(email_paths, smoothing):
    texts_joined = []
    for mail in email_paths:
        texts_joined = texts_joined + tokenize_email(mail)

    bigrams = []
    for i in range(1, len(texts_joined)):
        bigrams.append(texts_joined[i-1] + " " + texts_joined[i])

    vocab = set(bigrams+texts_joined)
    frequency = collections.Counter(bigrams+texts_joined)

    len_text = len(bigrams+texts_joined)
    len_vocab = len(vocab)

    prob_dict = {}
    long_word_count = 0
    for el in vocab:
        # remember vocabulary consists of unigrams and bigrams!
        count = frequency[el]
        get_words = el.split()

        # if we are dealing with unigram,  which is longer than 12 tokens
        if len(get_words) == 1 and len(get_words[0]) > LONGWORD_CHAR_NUM:
            long_word_count += 1

        prob_dict[el] = math.log((count + smoothing)/(len_text + smoothing*(len_vocab+1)))
    prob_dict["<UNK>"] = math.log(smoothing/(len_text + smoothing*(len_vocab+1)))

    prob_dict["<LONG_W>"] = math.log((long_word_count + smoothing)/(len_text + smoothing*(len_vocab+1)))
    return prob_dict

class SpamFilter(object):

    def __init__(self, spam_dir: str, ham_dir: str):
        smoothing = 1e-8

        paths_spam = get_paths(spam_dir)
        paths_ham = get_paths(ham_dir)

        self.spam = log_probs(paths_spam, smoothing)
        self.ham = log_probs(paths_ham, smoothing)

        num_of_docs = len(paths_spam)+len(paths_ham)

        self.prob_of_sp = math.log(float(len(paths_spam))/num_of_docs)
        self.prob_of_not_sp = math.log(float(len(paths_ham))/num_of_docs)

    # given email decide if it's a spam or not
    def is_spam(self, email_path: str):
        cur_mail = tokenize_email(email_path)
        bigram = []

        for i in range(1, len(cur_mail)):
            bigram.append(cur_mail[i-1] + " " + cur_mail[i])

        vocab = set(bigram+cur_mail)
        count_w = collections.Counter(bigram+cur_mail)

        P_spam_doc = self.prob_of_sp
        P_ham_doc = self.prob_of_not_sp
        no_of_longWords = 0

        for el in vocab:
            try:
                P_word_given_spam = self.spam[el]
            except KeyError:
                P_word_given_spam = self.spam["<UNK>"]
            try:
                P_word_given_ham = self.ham[el]
            except KeyError:
                P_word_given_ham = self.ham["<UNK>"]

            get_word = el.split()
            
            if len(get_word) == 1 and len(get_word[0]) > LONGWORD_CHAR_NUM:
                no_of_longWords += 1

            P_spam_doc += P_word_given_spam*count_w[el]
            P_ham_doc += P_word_given_ham*count_w[el]

        # don't forget about long words!
        P_spam_doc = P_spam_doc + self.spam["<LONG_W>"]*no_of_longWords
        P_ham_doc = P_ham_doc + self.ham["<LONG_W>"]*no_of_longWords
        return P_spam_doc > P_ham_doc


if __name__ == "__main__":
    start = time.time()
    sf = SpamFilter("data/train/spam", "data/train/ham")

    print(f"done with initialization! \t\t {time.time()-start:.2f} s")

    # Performance with Spam
    spam_directory = "data/dev/spam"
    spam_paths = get_paths(spam_directory)

    spam_errors = 0
    for mail in spam_paths:
       val = sf.is_spam(mail)
       if not val:
           spam_errors += 1

    # Performance with Ham
    ham_directory = "data/dev/ham"
    ham_paths = get_paths(ham_directory)

    ham_errors = 0
    for mail in ham_paths:
       val = sf.is_spam(mail)
       if val:
           ham_errors += 1

    print("spam errors:", spam_errors, "\nham errors:", ham_errors)
    print("correct identified:", 1 - float(spam_errors + ham_errors) / (len(ham_paths) + len(spam_paths)))
