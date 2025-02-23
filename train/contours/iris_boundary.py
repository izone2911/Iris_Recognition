
import numpy as np
import cv2
from numpy.ma.core import shape


def find_circle_by_three_point(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    A = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
        [x3, y3, 1]
    ])
    B = np.array([
        [x1**2 + y1**2, y1, 1],
        [x2**2 + y2**2, y2, 1],
        [x3**2 + y3**2, y3, 1]
    ])
    C = np.array([
        [x1**2 + y1**2, x1, 1],
        [x2**2 + y2**2, x2, 1],
        [x3**2 + y3**2, x3, 1]
    ])
    D = np.array([
        [x1**2 + y1**2, x1, y1],
        [x2**2 + y2**2, x2, y2],
        [x3**2 + y3**2, x3, y3]
    ])
    a = np.linalg.det(A)
    b = -np.linalg.det(B)
    c = np.linalg.det(C)
    d = -np.linalg.det(D)
    h = -b / (2 * a)
    k = -c / (2 * a)
    r = np.sqrt((b**2 + c**2 - 4 * a * d) / (4 * a**2))
    return (h, k), r

def find_circle(iris_mask):
    contours, _ = cv2.findContours(iris_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_iris_contour = max(contours, key=cv2.contourArea)

    # IRIS
    (x_i, y_i), radius_i = cv2.minEnclosingCircle(largest_iris_contour)
    x_i = int(x_i)
    y_i = int(y_i)
    radius_i = int(radius_i)

    y_coords, x_coords = np.where(iris_mask > 0 )
    mask_points = list(zip(x_coords, y_coords))

    # Lấy các điểm giao với đường thẳng y = y_i
    vertical_left_points = []
    vertical_right_points = []
    for point in mask_points:
        x, y = int(point[0]), int(point[1])
        if abs(y - y_i) < 1:
            if x < x_i:
                vertical_left_points.append((x, y))
            elif x > x_i:
                vertical_right_points.append((x, y))
    left_point = min(vertical_left_points, key=lambda p: abs(p[0] - x_i))
    right_point = min(vertical_right_points, key=lambda p: abs(p[0] - x_i))

    # Lấy điểm bên dưới giao với đường dọc x = x_i
    horizontal_bottom_points = []
    for point in mask_points:
        x, y = int(point[0]), int(point[1])
        if abs(x - x_i) < 1:
            if y > y_i:
                horizontal_bottom_points.append((x, y))
    bottom_point = min(horizontal_bottom_points, key=lambda p: abs(p[1] - y_i))

    (x_p, y_p), radius_p = find_circle_by_three_point(left_point, right_point, bottom_point)
    x_p = int(x_p)
    y_p = int(y_p)
    radius_p = int(radius_p)

    return (x_i, y_i, radius_i, x_p, y_p, radius_p)


def main():
    for i in range(1,100):
        source_path_1 = "../../segmentation/iris_segmentation/segmentation_img/" + f"{i:03}"
        for j in range(1,11):
            for suffix in ['L', 'R']:
                source_path_2 = source_path_1 + f"{j:02}_" + f"{suffix}" + "_binary-mask.png"
                image = cv2.imread(source_path_2, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Không thể đọc ảnh {source_path_2}")
                    continue

                circle = find_circle(image)
                output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.circle(output_image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(output_image, (circle[0], circle[1]), 5, (0, 255, 0), 2)
                cv2.circle(output_image, (circle[3], circle[4]), circle[5], (255, 0, 0), 2)
                cv2.circle(output_image, (circle[3], circle[4]), 5, (255, 0, 0), 2)

                output_path = "./segmentation_img/" + f"{i:03}" + f"{j:02}_" + f"{suffix}" + ".png"
                # output_mask_path = "./segmentation_img/" + f"{i:03}" + f"{j:02}_" + f"{suffix}" + "_mask.png"
                cv2.imwrite(output_path, output_image)  # Sử dụng cv2.imwrite để lưu ảnh
                # cv2.imwrite(output_mask_path, circle[6])  # Sử dụng cv2.imwrite để lưu ảnh
                print("find " + f"{i:03}" + f"{j:02}_" + f"{suffix}" + " successful")



if __name__ == "__main__":
    main()
