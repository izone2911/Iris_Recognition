# ảnh vành khăn từ json để train y
import os

import labelme
import numpy as np
import json
from PIL import Image



def json_to_ring_mask(json_file, output_path):
    if not os.path.isfile(json_file):
        print(f"File not found: {json_file}")
        return

    # Mở file JSON và đọc nội dung
    with open(json_file) as f:
        data = json.load(f)

    # Lấy kích thước ảnh
    img_shape = (data['imageHeight'], data['imageWidth'], 3)

    # Tạo bản ánh xạ từ tên nhãn đến giá trị
    label_name_to_value = {'_background_': 0}
    for shape in data['shapes']:
        label_name_to_value[shape['label']] = len(label_name_to_value)

    # đặt c2 lên trước 1 trong json vì thao tác gán label 0 1 2 cho lbl có thể ghi đè -> có thể mất vùng c1 trong c2
    reordered_shapes = []
    for shape in data['shapes']:
        if shape['label'] == 'c2':
            reordered_shapes.insert(0, shape)  # Đặt c2 ở đầu danh sách
        elif shape['label'] == 'c1':
            reordered_shapes.append(shape)  # Đặt c1 ở cuối danh sách
    data['shapes'] = reordered_shapes

    # Tạo mask từ các anotate
    lbl, _ = labelme.utils.shapes_to_label(img_shape, data['shapes'], label_name_to_value)

    # Kiểm tra xem nhãn c1 và c2 có tồn tại không
    if 'c1' not in label_name_to_value or 'c2' not in label_name_to_value:
        raise ValueError("Both 'c1' and 'c2' labels must be present in the JSON file.")

    # Lấy giá trị tương ứng với các nhãn c1 và c2
    c1_val = label_name_to_value['c1']
    c2_val = label_name_to_value['c2']

    # # Tạo mask cho c2 (vòng lớn) và c1 (vòng nhỏ)
    # mask_c2 = np.zeros_like(lbl, dtype=np.uint8)
    # mask_c2[lbl == c2_val] = 255  # Vùng của c2 (vòng lớn)
    #
    # mask_c1 = np.zeros_like(lbl, dtype=np.uint8)
    # mask_c1[lbl == c1_val] = 255  # Vùng của c1 (vòng nhỏ)
    #
    # # Tạo vành khăn: trừ mask của c1 khỏi mask của c2
    # ring_mask = mask_c2 - mask_c1

    mask_c1 = np.zeros_like(lbl, dtype=np.uint8)
    mask_c1[lbl == c1_val] = 255  # Vùng của c1 (vòng nhỏ)
    ring_mask = mask_c1

    # Đảm bảo giá trị trong ring_mask nằm trong khoảng từ 0 đến 255
    ring_mask = np.clip(ring_mask, 0, 255)

    # Lưu ảnh mask cho vành khăn
    Image.fromarray(ring_mask).save(output_path)

def create(mlist, postfix):
    if postfix == "L":
        range_j = list(range(1, 6))
    else:
        range_j = list(range(6, 11))
    for i in mlist:
        for j in range_j:
            json_file = "mask_json/" + f"{i:03}" + f"{j:02}_" + postfix + ".json"
            output_path = "./train/segmentation/pupil_segmentation/mask_truth/" + f"{i:03}" + f"{j:02}_" + postfix + ".png"
            try:
                json_to_ring_mask(json_file, output_path)
                print(f"Create mask {i:02}{j:02} successful")
            except Exception as e:
                print(f"Error processing {json_file}: {e}")


mask_left =  list(range(1, 33)) + [37, 38, 76, 93, 113, 118, 123]
mask_right = [15, 26, 76, 93, 113, 118, 123]

create(mask_left, "L")
create(mask_right, "R")



