import pandas as pd
import os
import re


def parse_gcdc(source_path, dest_path):
    datasets_folder = os.path.join(dest_path, "gcdc_dataset")
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
    datasets = ["Clinton_train", "Clinton_test", "Enron_train", "Enron_test",
                    "Yahoo_train", "Yahoo_test",  "Yelp_train", "Yelp_test"]
    total_count = 0
    for dataset in datasets:
        curr_dataset_folder = os.path.join(datasets_folder, dataset)
        if not os.path.exists(curr_dataset_folder):
            os.makedirs(curr_dataset_folder)
        dataset_df = pd.read_csv(os.path.join(source_path, dataset + ".csv"))
        for idx, row in dataset_df.iterrows():
            file_id = row["text_id"]
            file_text = row["text"]
            new_file_path = os.path.join(curr_dataset_folder, "F" + str(file_id) + ".txt")
            with open(new_file_path, "w") as text_file:
                cleaned_text = re.sub("[\.]+\)", ")", file_text)
                cleaned_text = re.sub("\)[\.]+", ")", cleaned_text)
                text_file.write(cleaned_text)
    print(total_count)
parse_gcdc(os.path.join(os.getcwd(), "GCDC_rerelease"), os.getcwd())
