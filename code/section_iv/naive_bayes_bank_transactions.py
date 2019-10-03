import math
import re

import pandas as pd
from collections import defaultdict


class BankTransaction:
    def __init__(self, date, amount, memo, category):
        self.date = date
        self.amount = amount
        self.memo = memo
        self.category = category


bank_transactions = [(BankTransaction(row[0], row[1], row[2], row[3])) for index, row in
                     pd.read_csv("https://tinyurl.com/yy38e9jj").iterrows()]

categories = set(bt.category for bt in bank_transactions)


# Helper function to break up words from a string
def break_up_words(str):
    return re.sub(r'[^\w\s]', '', str.lower()).split()


def probability_for_category(memo, category):
    positive_word_count = sum(1 for transaction in bank_transactions if transaction.category == category)
    negative_word_count = sum(1 for transaction in bank_transactions if transaction.category != category)

    # Get count of words that occurred in this category
    positive_word_counts = defaultdict(int)

    for transaction in bank_transactions:
        if transaction.category == category:
            for word in break_up_words(transaction.memo):
                positive_word_counts[word] += 1

    # Get count of words that did not occur in this category
    negative_word_counts = defaultdict(int)
    for transaction in bank_transactions:
        if transaction.category != category:
            for word in break_up_words(transaction.memo):
                negative_word_counts[word] += 1

    # get count of words in all transactions
    all_word_counts = defaultdict(int)

    for transaction in bank_transactions:
        for word in break_up_words(transaction.memo):
            all_word_counts[word] += 1

    # Create functions to calculate probability of word occurring in category or not
    # add a little .1 and .2 to numerator/denominator respectively to prevent 0 division
    def prob_word_appears_in_category(w):
        return (.1 + positive_word_counts.get(w, 0)) / (.2 + positive_word_count)

    def prob_word_not_appears_in_category(w):
        return (.1 + negative_word_counts.get(w, 0)) / (.2 + negative_word_count)

    # Here we go! Naive Bayes happens here
    def category_score_for_memo(memo):
        message_words = break_up_words(memo)

        total_positive_probability = 0.0
        for w in all_word_counts:
            if w in message_words:
                total_positive_probability += math.log(prob_word_appears_in_category(w))
            else:
                total_positive_probability += math.log(1.0 - prob_word_not_appears_in_category(w))

        total_negative_probability = 0.0
        for w in all_word_counts:
            if w in message_words:
                total_negative_probability += math.log(prob_word_not_appears_in_category(w))
            else:
                total_negative_probability += math.log(1.0 - prob_word_not_appears_in_category(w))

        return math.exp(total_positive_probability) / (
                math.exp(total_positive_probability) + math.exp(total_negative_probability))

    return category_score_for_memo(memo)


while True:
    memo = input("Input a bank memo to predict its category:")

    best_category = None
    best_probability = 0.0

    for category in categories:
        category_probability = probability_for_category(memo, category)
        if category_probability > .50 and category_probability > best_probability:
            best_probability = category_probability
            best_category = category

    print("{0}, Probability: {1}\r\n%".format(best_category, round(best_probability * 100.0, 6)))
