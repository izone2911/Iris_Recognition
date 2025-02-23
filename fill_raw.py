# chuyển ảnh vào data train

import os
import shutil


def copy_for_train(mlist, postfix):
    if postfix == "L":
        range_j = list(range(1, 6))
    else:
        range_j = list(range(6, 11))
    destination_path = './train/segmentation/pupil_segmentation/mask_truth/'
    for i in mlist:
        source_path_1 = '../IITD Database/' + f"{i:03}" + "/"
        for j in range_j:
            source_path_2 = source_path_1 + f"{j:02}_" + postfix + ".bmp"
            new_file_name = f"{i:03}{j:02}_" + postfix + ".bmp"
            new_file_path = os.path.join(destination_path, new_file_name)

            try:
                shutil.copy(source_path_2, new_file_path)
                print(f"Sao chép thành công {source_path_2} tới {new_file_path}")
            except FileNotFoundError:
                print(f"Lỗi: Không tìm thấy tệp {source_path_2}")
            except Exception as e:
                print(f"Lỗi khi sao chép {source_path_2} tới {new_file_path}: {e}")

def copy_for_test(imgs_index):
    destination_path = './train/img-test/'
    for i in imgs_index:
        source_path_1 = '../IITD Database/' + f"{i:03}" + "/"
        for j in range(1, 11):
            for suffix in ['L', 'R']:  # Kiểm tra cả hai hậu tố _L và _R
                source_path_2 = source_path_1 + f"{j:02}_{suffix}.bmp"
                new_file_name = f"{i:03}{j:02}_{suffix}.bmp"
                new_file_path = os.path.join(destination_path, new_file_name)

                try:
                    shutil.copy(source_path_2, new_file_path)
                    print(f"Sao chép thành công {source_path_2} tới {new_file_path}")
                except FileNotFoundError:
                    print(f"Lỗi: Không tìm thấy tệp {source_path_2}")
                except Exception as e:
                    print(f"Lỗi khi sao chép {source_path_2} tới {new_file_path}: {e}")


mask_left =  list(range(1, 33)) + [37, 38, 76, 93, 113, 118, 123]
mask_right = [15, 26, 76, 93, 113, 118, 123]
copy_for_train(mask_left, "L")
copy_for_train(mask_right, "R")
# copy_for_test(list)