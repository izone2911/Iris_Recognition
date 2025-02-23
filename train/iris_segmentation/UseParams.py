import os.path

import torch
import cv2
import numpy as np
from src.train.segmentation.UNet import UNet

# Đường dẫn tới mô hình đã lưu
model_path = 'unet_iris_segmentation_14-11.pth'

# Tạo lại mô hình với cùng kiến trúc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(1, 1).to(device)  # Số kênh đầu ra là 2

# Tải trọng số mô hình từ file .pth
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

# Chuyển mô hình sang chế độ eval (inference)
model.eval()

# Hàm để xử lý một ảnh đầu vào và dự đoán mask
def predict(image_path, model):
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Tiền xử lý: chuẩn hóa và thêm chiều kênh
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Thêm chiều kênh
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch

    # Chuyển đổi sang tensor
    image_tensor = torch.from_numpy(image).to(device)

    # Dự đoán
    with torch.no_grad():
        output = model(image_tensor)
        # print("Output shape:", output.shape)

    # Chuyển đổi kết quả dự đoán từ tensor sang numpy
    predicted_mask = torch.sigmoid(output).cpu().numpy()
    # print("Predicted mask shape:", predicted_mask.shape)

    # Loại bỏ các chiều không cần thiết
    predicted_mask = np.squeeze(predicted_mask)  # Xoá chiều không cần thiết
    # print("Squeezed mask shape:", predicted_mask.shape)

    return predicted_mask


for i in range (1, 100):
    for img_index in range (1, 11):
        for suffix in ['L', 'R']:  # Kiểm tra cả hai hậu tố _L và _R
            try:
                # Đường dẫn tới ảnh cần dự đoán
                image_path = f'../img-test/{i:03}{img_index:02}_{suffix}.bmp'
                raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if raw_img is None:
                    print(f"Không thể đọc ảnh: {image_path}")
                    continue

                # Thực hiện dự đoán
                predicted_mask = predict(image_path, model)

                # Chuyển đổi mặt nạ sang định dạng nhị phân (0, 255)
                binary_mask = (predicted_mask > 0.9).astype(np.uint8) * 255  # Ngưỡng

                # Lưu các kết quả
                # cv2.imwrite(f"./segmentation_img/{i:03}{img_index:02}_{suffix}.bmp", raw_img)
                # cv2.imwrite(f"./segmentation_img/{i:03}{img_index:02}_{suffix}_predict-mask.bmp", predicted_mask * 255)
                cv2.imwrite(f"./img_result/{i:03}{img_index:02}_{suffix}_binary-mask.png", binary_mask)
                # cv2.imwrite(f"./segmentation_img/{i:03}{img_index:02}_{suffix}_segmentation.png", cv2.bitwise_and(binary_mask, raw_img))

                print(f"Dự đoán {i:03}{img_index:02}_{suffix}.bmp thành công")

            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {i:02} với hậu tố {suffix}: {e}")
