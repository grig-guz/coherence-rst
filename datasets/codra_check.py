from nltk import tokenize
import numpy as np
import os
import re
from itertools import permutations

# analystic constants
punc_list = [",", ".", ")", "_", "-", "/", '"', "'", "}", "]", "|", "?", "!"]
char_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
num_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
total_list = char_list + [x.upper() for x in char_list] + num_list

# dynamic variables
foldernames = ["train_set_permutes/7/"]

print_specifics = False

# boolean operation tags, in the same order as the input folders
error_clean_up = [False]
key_clean_up = [False]
empty_clean_up = [False]
error_fix = [False]
key_fix = [False]
empty_fix = [False]

total_error = 0
total_interupt = 0
total_empty = 0
total_files = 0

for filename in os.listdir(foldernames[0]):
    if filename.endswith("_permuted"):
        foldernames.append(foldernames[0] + filename + "/")
        error_clean_up.append(False)
        key_clean_up.append(False)
        empty_clean_up.append(False)
        error_fix.append(False)
        key_fix.append(False)
        empty_fix.append(False)

print("Folders being checked: ", foldernames)
print("Number of folders: ", len(foldernames))

for i, foldername in enumerate(foldernames):

    error_name_list = []
    key_interupt_list = []
    empty_name_list = []

    print("Evaluating folder " + foldername)

    for filename in os.listdir(foldername):
        if filename.endswith("_tree.txt"):
            total_files += 1
            file = open(foldername + filename, "r")
            file_text = file.read()
            if "Key" in file_text:
                key_interupt_list.append(filename)
                if print_specifics:
                    print("Found key problem in " + filename)
            elif "Error" in file_text:
                error_name_list.append(filename)
                if print_specifics:
                    print("Found error in " + filename)
            elif file_text == "":
                empty_name_list.append(filename)
                if print_specifics:
                    print("Empty file in " + filename)

    print("Error count: ", len(error_name_list))
    print("Key count: ", len(key_interupt_list))
    print("Empty count: ", len(empty_name_list))

    total_error += len(error_name_list)
    total_interupt += len(key_interupt_list)
    total_empty += len(empty_name_list)

    if error_clean_up[i]:
        for filename in error_name_list:
            original_file_name = filename.replace("_tree.txt", "")
            if (original_file_name + ".txt.edu") in os.listdir(foldername):
                os.remove(foldername + original_file_name + ".txt.edu")
                if print_specifics:
                    print("Removed file " + foldername + original_file_name + ".txt.edu")
            if (original_file_name + "_tree.txt") in os.listdir(foldername):
                os.remove(foldername + original_file_name + "_tree.txt")
                if print_specifics:
                    print("Removed file " + foldername + original_file_name + "_tree.txt")
        print("Cleaned up " + str(len(error_name_list)) + " error files.")

    if key_clean_up[i]:
        for filename in key_interupt_list:
            original_file_name = filename.replace("_tree.txt", "")
            if (original_file_name + ".txt.edu") in os.listdir(foldername):
                os.remove(foldername + original_file_name + ".txt.edu")
                if print_specifics:
                    print("Removed file " + foldername + original_file_name + ".txt.edu")
            if (original_file_name + "_tree.txt") in os.listdir(foldername):
                os.remove(foldername + original_file_name + "_tree.txt")
                if print_specifics:
                    print("Removed file " + foldername + original_file_name + "_tree.txt")
        print("Cleaned up " + str(len(key_interupt_list)) + " key interupted files.")

    if empty_clean_up[i]:
        for filename in empty_name_list:
            original_file_name = filename.replace("_tree.txt", "")
            if (original_file_name + ".txt.edu") in os.listdir(foldername):
                os.remove(foldername + original_file_name + ".txt.edu")
                if print_specifics:
                    print("Removed file " + foldername + original_file_name + ".txt.edu")
            if (original_file_name + "_tree.txt") in os.listdir(foldername):
                os.remove(foldername + original_file_name + "_tree.txt")
                if print_specifics:
                    print("Removed file " + foldername + original_file_name + "_tree.txt")
        print("Cleaned up " + str(len(empty_name_list)) + " empty files.")

    if error_fix[i]:
        for filename in error_name_list:
            fix_punc(foldername + filename.replace("_tree.txt", "") + ".txt")
            if print_specifics:
                print("Fixed file " + filename.replace("_tree.txt", "") + ".txt")
        print("Fixed up data for selected error files.")

    if key_fix[i]:
        for filename in key_interupt_list:
            fix_punc(foldername + filename.replace("_tree.txt", "") + ".txt")
            if print_specifics:
                print("Fixed file " + filename.replace("_tree.txt", "") + ".txt")
        print("Fixed up data for selected key files.")

    if empty_fix[i]:
        for filename in empty_name_list:
            fix_punc(foldername + filename.replace("_tree.txt", "") + ".txt")
            if print_specifics:
                print("Fixed file " + filename.replace("_tree.txt", "") + ".txt")
        print("Fixed up data for selected empty files.")

print("Final results:")
print("Total error: ", total_error)
print("Total interupt: ", total_interupt)
print("Total empty: ", total_empty)
print("Total files: ", total_files)

def fix_punc(filename):
    """
        filename is the name of the file to be permuted
        this function creates the permutations and then saves them in files names using the {filename}_permuted_number format
    """

    file = open(filename, "r")
    file_text = file.read()
    for c1 in total_list:
        for c2 in total_list:
            file_text = file_text.replace(c1 + c2 + "\n", c1 + c2 + ".\n")

    if file_text[-1:] in total_list:
        if file_text[-2:] != "\n":
            file_text += "."

    file_write = open(filename, "w")
    file_write.write(file_text)
