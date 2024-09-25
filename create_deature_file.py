import numpy as np
import pandas as pd
import os
import pickle

pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    feature_files_dir = "/home/asif/PycharmProjects/sba/save"
    files = os.listdir(feature_files_dir)
    df = pd.DataFrame(columns=["id", "time", "feature", "label"])
    counter = 0
    for file in files:
        print(file)
        uid = file.split(".")[0].split("_")
        # print(uid[0],uid[3])
        with open(os.path.join(feature_files_dir,file),"rb") as f:
            dictionary = pickle.load(f)
            # print(len(dictionary))
            for key, value in dictionary.items():
                # print(key,value)
                value = [round(i,2) for i in value]
                value = np.asarray(value)
                unique_id = uid[0]+"_"+uid[3]+"_"+key
                new_row = {'id':unique_id, "time":key, "feature":value, "label":uid[3]}
                df.loc[counter] = new_row
                counter+=1
    print(df)
    save_ = feature_files_dir + "/features_combined"
    df.to_pickle(save_)
    df = df.sample(frac=1).reset_index(drop=True)
    print(df)
    save_ = feature_files_dir + "/features_shuffle"
    df.to_pickle(save_)