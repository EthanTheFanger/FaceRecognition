"""
** This file was made using Claude, by prompting it to turn my jupyter notebook into a runnable python file

Siamese preprocessing script

Builds anchor/positive/negative triplets from the LFW dataset for training
a siamese network with triplet loss.

Approach: pull every identity with more than one image, generate anchor/positive
pairs from within that identity, and pair each with a random negative drawn from
identities that only have a single image. Each individual photo is capped at
4 uses, and each identity is capped at 10 triplets in the final output.
"""

import argparse
import random
import pandas as pd


def get_lfw_zeros(number):
    """
    simple helper function to return enough zeros to have 4 digits in the file,
    works only on positive numbers, it works for the purposes of this project
    """
    count = 0
    while number // (10 ** count) != 0:
        count += 1
    return (4 - count) * '0'


def get_lfw_file_names(arr):
    """
    helper function for create_triplets. changes ['Aaron_Peirsol', 4] to
    ['Aaron_Peirsol_0004.jpg','Aaron_Peirsol_0003.jpg','Aaron_Peirsol_0002.jpg','Aaron_Peirsol_0001.jpg']

    args:
        arr (list): the person, ie. ['Aaron_Peirsol', 4]

    returns:
        res (list): the list with the multiple people
        ['Aaron_Peirsol_0004.jpg','Aaron_Peirsol_0003.jpg','Aaron_Peirsol_0002.jpg','Aaron_Peirsol_0001.jpg']
    """
    person, number = arr
    zeros = get_lfw_zeros(number)
    res = []
    while number != 0:
        res.append(person + "_" + str(zeros) + str(number) + ".jpg")
        number -= 1
    return res


def create_triplets(multiple, solo, counts=4):
    """
    build triplets by choosing people from a list of multiple people, set two images of the same people to be the anchor and the positive,
    and then choose a third one from another list of people that only have one picture.

    args:
        multiple (pd.dataFrame): list of people that have multiple people
        solo (pd.dataFrame): list of people that have only one picture
        counts (int): maximum amount of times a single photo can be reused across triplets

    return:
        res: a list of triplets
    """
    # flatten both dfs into lists so its easier to work with
    multiple = multiple.to_numpy().tolist()
    solo = solo.to_numpy().tolist()

    # convert all of them into their respective file names
    # in the multiple and solo list, in the respective lists like ['Aaron_Peirsol', 4], change it to
    # ['Aaron_Peirsol_0004.jpg','Aaron_Peirsol_0003.jpg','Aaron_Peirsol_0002.jpg','Aaron_Peirsol_0001.jpg']
    # changes the solo into a dict early
    multiple = [get_lfw_file_names(a) for a in multiple]
    solo = dict.fromkeys([get_lfw_file_names(b)[0] for b in solo], 0)
    res = []

    # turn the nested list into a list of dicts like, with the count of times it was used as the value:
    # {'Aaron_Peirsol_0004.jpg':0, ....... 'Aaron_Peirsol_0001.jpg':0}
    # do this for the solo dict, as we want the counts of individual photos too
    for i in multiple:
        i = dict.fromkeys(i, 0)
        while len(i) > 1 and len(solo) > 0:
            # choose two random elements inside i
            anchor, positive = random.sample(list(i.keys()), 2)

            # choose one random element inside solo
            negative = random.choice(list(solo.keys()))

            # append into res
            res.append([anchor, positive, negative])

            # pop if any element reaches `counts` uses
            for d, key in [(i, anchor), (i, positive), (solo, negative)]:
                d[key] += 1
                if d[key] == counts:
                    d.pop(key)
    return res


def triplets_to_df(arr, max=10):
    """
    changes the nested list type that create_triplets make and change them to df,
    limiting each person to a maximum of max rows each

    args:
        arr (list): the nested list from create_triplets
        max (int): maximum amount of rows that each person is limited to

    return:
        res (pd.DataFrame): the resulting df from the filtering
    """
    res = pd.DataFrame(arr, columns=['Anchor', 'Positive', 'Negative'])

    # make a row that gets rid of the last _000X.jpg portion so that only the name is present
    res['name'] = res['Anchor'].apply(lambda x: x[0:-9])
    res = res.groupby('name').head(max).reset_index(drop=True)
    return res


def main(counts, max_rows, allnames_path):
    # load the full LFW identity list (name, number of images per identity)
    all_names = pd.read_csv(allnames_path)

    # split into identities with multiple images vs. only one image
    multiple = all_names[all_names['images'] > 1]
    solo = all_names[all_names['images'] == 1]

    triplets_raw = create_triplets(multiple=multiple, solo=solo, counts=counts)
    triplets = triplets_to_df(triplets_raw, max=max_rows)

    print(f"counts={counts}, max={max_rows} -> {len(triplets)} triplets generated")
    print(triplets.head())

    return triplets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LFW triplets for siamese network training")
    parser.add_argument(
        "--counts", type=int, default=4,
        help="max number of times a single photo can be reused across triplets (default: 4)"
    )
    parser.add_argument(
        "--max", type=int, default=10, dest="max_rows",
        help="max number of triplet rows allowed per identity (default: 10)"
    )
    parser.add_argument(
        "--allnames-path", type=str, default="../data/lfw_dataset/lfw_allnames.csv",
        help="path to lfw_allnames.csv"
    )
    args = parser.parse_args()

    main(counts=args.counts, max_rows=args.max_rows, allnames_path=args.allnames_path)