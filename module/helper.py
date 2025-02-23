
import numpy as np
import cv2


class Helper:
    def find_circle_by_three_point(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        A = np.array([
            [x1, y1, 1],
            [x2, y2, 1],
            [x3, y3, 1]
        ])
        B = np.array([
            [x1 ** 2 + y1 ** 2, y1, 1],
            [x2 ** 2 + y2 ** 2, y2, 1],
            [x3 ** 2 + y3 ** 2, y3, 1]
        ])
        C = np.array([
            [x1 ** 2 + y1 ** 2, x1, 1],
            [x2 ** 2 + y2 ** 2, x2, 1],
            [x3 ** 2 + y3 ** 2, x3, 1]
        ])
        D = np.array([
            [x1 ** 2 + y1 ** 2, x1, y1],
            [x2 ** 2 + y2 ** 2, x2, y2],
            [x3 ** 2 + y3 ** 2, x3, y3]
        ])
        a = np.linalg.det(A)
        b = -np.linalg.det(B)
        c = np.linalg.det(C)
        d = -np.linalg.det(D)
        h = -b / (2 * a)
        k = -c / (2 * a)
        r = np.sqrt((b ** 2 + c ** 2 - 4 * a * d) / (4 * a ** 2))
        return (h, k), r

    def find_circle(self, iris_mask):
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

        (x_p, y_p), radius_p = self.find_circle_by_three_point(left_point, right_point, bottom_point)
        x_p = int(x_p)
        y_p = int(y_p)
        radius_p = int(radius_p)

        return (x_i, y_i, radius_i, x_p, y_p, radius_p)

