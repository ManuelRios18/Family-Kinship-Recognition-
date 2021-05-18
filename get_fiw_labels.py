import os
import cv2
import random
import numpy as np
import pandas as pd


def get_parent_child_images(parent_id, child_id, dataset):
    parent_images = os.listdir(os.path.join(root_path, f"{dataset}-faces", parent_id))
    child_images = os.listdir(os.path.join(root_path, f"{dataset}-faces", child_id))

    parent_image_name = os.path.join(parent_id, random.choice(parent_images))
    child_image_name = os.path.join(child_id, random.choice(child_images))

    return parent_image_name, child_image_name


def get_no_child(all_children_list, parent_id, dataset):
    parent_family = parent_id.split("/")[0]
    possible_children = [child for child in all_children_list if parent_family not in child]
    no_child_id = random.choice(possible_children)
    parent_image_name, no_child_image_name = get_parent_child_images(parent_id, no_child_id, dataset)
    return parent_image_name, no_child_image_name


def visualize(final_result, dataset):
    for i, row in final_result.iterrows():
        parent_img = cv2.imread(os.path.join(root_path, f"{dataset}-faces", row.parent_image))
        child_img = cv2.imread(os.path.join(root_path, f"{dataset}-faces", row.child_image))
        parent_img = cv2.resize(parent_img, (124, 124), interpolation=cv2.INTER_AREA)
        child_img = cv2.resize(child_img, (124, 124), interpolation=cv2.INTER_AREA)
        final_image = np.hstack([parent_img, child_img])
        print("kin", row.pair_type, row.kin)
        cv2.imshow("parent", final_image)
        cv2.waitKey()


set_name = "val" #train
root_path = "/media/manuel/New Volume/Computer Vision/fiw"
result = list()

for pair_type in ["fs", "fd", "ms", "md"]:
    data = pd.read_csv(f"/media/manuel/New Volume/Computer Vision/fiw/{set_name}-pairs-updated.csv")
    data = data[data.ptype == pair_type]

    all_parents = list(data.p1.unique())
    all_children = list(data.p2.unique())

    pair_result = pd.DataFrame(columns=["parent_image", "child_image", "pair_type", "kin"])
    for i, row in data.iterrows():
        parent_image, child_image = get_parent_child_images(row.p1, row.p2, set_name)
        pair_result.loc[len(pair_result)] = [parent_image, child_image, pair_type, 1]
        parent_image, child_image = get_no_child(all_children, row.p1, set_name)
        pair_result.loc[len(pair_result)] = [parent_image, child_image, pair_type, 0]
    result.append(pair_result)

result = pd.concat(result)
result.to_csv(os.path.join("data", f"fiw_{set_name}_pairs.csv"), index=False)
#visualize(result, set_name)



