import numpy as np
import cv2
from PIL import Image
from helper import Helper
from segmentation import Segmentation


class Normalization:
    def __init__(self):
        self.segmentation = Segmentation()

    def polar_to_cartesian(self, phi, ratio, r_phi, R_phi, inner_center):
        R_point = (1-ratio) * r_phi + ratio * R_phi
        x = inner_center[0] + R_point * np.cos(phi)
        y = inner_center[1] + R_point * np.sin(phi)
        return x, y

    def normalize(self, raw_img, num_angles=432, num_point_per_angles=48, return_segmentation=False):
        predicted_mask = self.segmentation.predict(raw_img)
        binary_mask = (predicted_mask > 0.9).astype(np.uint8) * 255
        segmentation_img = cv2.bitwise_and(binary_mask, raw_img)

        # cv2.imshow("binary-mask", binary_mask)
        # cv2.waitKey(-1)
        # cv2.destroyAllWindows()

        centers = Helper().find_circle(binary_mask)

        # output_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        # cv2.circle(output_image, (centers[0], centers[1]), centers[2], (0, 255, 0), 2)
        # cv2.circle(output_image, (centers[0], centers[1]), 5, (0, 255, 0), 2)
        # cv2.circle(output_image, (centers[3], centers[4]), centers[5], (255, 0, 0), 2)
        # cv2.circle(output_image, (centers[3], centers[4]), 5, (255, 0, 0), 2)
        # cv2.imshow("circle", output_image)
        # cv2.waitKey(-1)
        # cv2.destroyAllWindows()

        rectangle_img = Image.new('L', (num_angles, num_point_per_angles))

        test_nor_image = cv2.cvtColor(segmentation_img, cv2.COLOR_GRAY2BGR)
        cv2.circle(test_nor_image, (centers[0], centers[1]), centers[2], (0, 255, 0), 2)
        cv2.circle(test_nor_image, (centers[0], centers[1]), 2, (0, 255, 0), 2)
        cv2.circle(test_nor_image, (centers[3], centers[4]), centers[5], (255, 0, 0), 2)
        cv2.circle(test_nor_image, (centers[3], centers[4]), 2, (255, 0, 0), 2)

        outer_center = np.array(centers[0:2])
        outer_radius = centers[2]
        inner_center = np.array(centers[3:5])
        inner_radius = centers[5]
        delta_r = np.linalg.norm(outer_center - inner_center)
        # goc cua vector IP voi Ox
        if delta_r == 0:
            theta = 0
        else:
            theta = np.arccos((inner_center[0] - outer_center[0]) / delta_r)

        for i in range(num_angles):
            phi = 2 * np.pi * i / num_angles

            R_phi = delta_r * np.cos(np.pi - theta - phi) + np.sqrt(
                outer_radius ** 2 - delta_r ** 2 + (delta_r * np.cos(np.pi - theta - phi)) ** 2)
            r_phi = inner_radius

            for j in range(num_point_per_angles):
                ratio = j / num_point_per_angles
                x, y = self.polar_to_cartesian(phi, ratio, r_phi, R_phi, inner_center)

                height, width = segmentation_img.shape
                if 0 <= x < width and 0 <= y < height:
                    pixel_value = segmentation_img[int(y), int(x)]
                    rectangle_img.putpixel((i, j), int(pixel_value))

                if j==num_point_per_angles-1 :
                    cv2.circle(test_nor_image, (int(x), int(y)), 2, (0, 0, 255), 1)

        # cv2.imshow("test_nor_image", test_nor_image)
        # cv2.waitKey(-1)
        # cv2.destroyAllWindows()

        if return_segmentation:
            return rectangle_img, test_nor_image
        return rectangle_img






