# Will randomly select a specified number of classes from LRW and zip this data up, ready for transmission to the CSF
import os
import random
import shutil

dataset_path = "D:/USB/LRW/lipread_mp4/"
words = [
    dir
    for dir in os.listdir(path=dataset_path)
    if os.path.isdir(os.path.join(dataset_path, dir))
]

num_words = 2
exceptions = [
    "ABOUT",
    "BELIEVE",
    "CHANCE",
    "FAMILY",
    "INFLATION",
    "TALKING",
    "PAYING",
    "POWERS",
    "FRANCE",
    "LEADERS",
    "THROUGH",
    "STREET",
    "IMPACT",
    "OFFICIALS",
    "GIVING",
    "MINUTES",
    "NEEDS",
    "NOTHING",
    "ARRESTED",
    "THIRD",
    "BANKS",
    "POSSIBLE",
    "SITUATION",
    "ELECTION",
    "FIGURES",
    "AHEAD",
    "WATER",
    "MIGHT",
    "AGAIN",
    "LATER",
    "RIGHTS",
    "BUSINESS",
    "FOREIGN",
    "DESPITE",
]
selected_words = []

i = 0
while i < num_words:
    ran_index = random.randint(a=0, b=(len(words) - 1))
    if (words[ran_index] not in selected_words) and (
        words[ran_index] not in exceptions
    ):
        selected_words.append(words[ran_index])
        i += 1

for dir in selected_words:
    path = os.path.join(dataset_path, dir)
    shutil.make_archive(path, "zip", path)

print(f"selected words: {selected_words}")
# ['INFLATION', 'TALKING', 'PAYING', 'POWERS', 'FRANCE', 'LEADERS', 'THROUGH', 'STREET', 'IMPACT', 'OFFICIALS', 'GIVING', 'MINUTES', 'NEEDS', 'NOTHING', 'ARRESTED', 'THIRD', 'BANKS', 'POSSIBLE', 'SITUATION', 'ELECTION', 'FIGURES', 'AHEAD', 'WATER', 'MIGHT', 'AGAIN', 'LATER', 'RIGHTS', 'BUSINESS', 'FOREIGN', 'DESPITE']
