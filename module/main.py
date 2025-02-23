import cv2
import numpy as np
from normalization import Normalization

def main():
    count_successfull_img = 0
    normalization = Normalization()
    for i in range(1,100):
        source_path_1 = "../train/img-test/" + f"{i:03}"
        for j in range(1,11):
            for suffix in ['L', 'R']:
                source_path_2 = source_path_1 + f"{j:02}_" + f"{suffix}" + ".bmp"
                image = cv2.imread(source_path_2, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Không thể đọc ảnh từ: {source_path_2}")
                    continue
                normalization_img, segmentation_img = normalization.normalize(image, return_segmentation=True)
                rectangle_img_np = np.array(normalization_img)
                # cv2.imshow("normalization", rectangle_img_np)
                # cv2.waitKey(-1)
                # cv2.destroyAllWindows()
                output_path = "./normalization_img/" + f"{i:03}" + f"{j:02}_" + f"{suffix}" + ".png"
                output_segmentation_path = "./segmentation_img/" + f"{i:03}" + f"{j:02}_" + f"{suffix}" + "_segmentation.png"
                cv2.imwrite(output_path, rectangle_img_np)
                cv2.imwrite(output_segmentation_path, segmentation_img)
                print(f"Phân đoạn thành công: {source_path_2}")
                count_successfull_img+=1
    print(f"Đã phân đoạn {count_successfull_img} ảnh.")

def tmp(i, j, suffix):
    path = "../train/img-test/" + f"{i:03}" + f"{j:02}_" + f"{suffix}" + ".bmp"
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Không thể đọc ảnh từ: {path}")
        return
    normalization_img, seg_img = Normalization().normalize(image, return_segmentation=True)
    rectangle_img_np = np.array(normalization_img)
    cv2.imshow("normalization", seg_img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
    # output_path = "./segmentation_img/" + f"{i:03}" + f"{j:02}_" + f"{suffix}" + ".png"
    # cv2.imwrite(output_path, rectangle_img_np)  # Sử dụng cv2.imwrite để lưu ảnh

if __name__ == "__main__":
    main()
    # tmp(14, 6, "R")d
