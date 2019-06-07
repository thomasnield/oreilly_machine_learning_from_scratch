import re


class Email:
    def __init__(self, message, is_spam):
        self.message = message
        self.is_spam = is_spam


emails = [
    Email("Hey there! I thought you might find this interesting. Click here.", True),
    Email("Get viagra for a discount as much as 90%", True),
    Email("Viagra prescription for less", True),
    Email("Even better than Viagra, try this new prescription drug", True),

    Email("Hey, I left my phone at home. Email me if you need anything. I'll be in a meeting for the afternoon.",False),
    Email("Please see attachment for notes on today's meeting. Interesting findings on your market research.", False),
    Email("An item on your Amazon wish list received a discount", False),
    Email("Your prescription drug order is ready", False),
    Email("Your Amazon account password has been reset", False),
    Email("Your Amazon order", False)
]

spam_email_count = sum(1 for email in emails if email.is_spam)
non_spam_email_count = sum(1 for email in emails if not email.is_spam)

# Helper function to break up words from a string
def break_up_words(str):
    return word in re.sub(r'[^\w\s]', '', str.lower()).split()


# get count of words for spam emails
spam_count_by_word = dict()

for email in emails:
    if email.is_spam:
        for word in break_up_words(email.message):
            spam_count_by_word[word] = spam_count_by_word.get(word, 0) + 1

# get count of words for non-spam emails
non_spam_count_by_word = dict()

for email in emails:
    if not email.is_spam:
        for word in break_up_words(email.message):
            non_spam_count_by_word[word] = non_spam_count_by_word.get(word, 0) + 1

# get count of words in all emails
all_emails_count_by_word = dict()

for email in emails:
    for word in break_up_words(email.message):
        all_emails_count_by_word[word] = all_emails_count_by_word.get(word, 0) + 1


# Create functions to calculate probability of word occuring in spam or not spam
# add a little .1 and .2 to numerator/denominator respectively to prevent 0 division
def prob_word_appears_in_spam(word):
    (.1 + spam_count_by_word(word)) / (.2 + spam_email_count)


def prob_word_appears_in_non_spam(word):
    (.1 + non_spam_count_by_word(word)) / (.2 + non_spam_email_count)

# Here we go! Naive Bayes happens here
def spam_score_for_message(message):
    message_words = [break_up_words(message)]

    probability_of_spam = 0.0
#    for word in message_words:
#       if (word in )

# Test a new message
message1 = "discount viagra wholesale, hurry while this offer lasts"


