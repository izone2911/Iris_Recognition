import torch
import numpy as np
from unet import UNet

class Segmentation:
    def __init__(self):
        self.model_path = 'unet_iris_segmentation.pth'
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(1, 1).to(device)
        # Tải trọng số mô hình từ file .pth
        self.model.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
        # Chuyển mô hình sang chế độ eval (inference)
        self.model.eval()

    def predict(self, image):

        # Tiền xử lý: chuẩn hóa và thêm chiều kênh
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Thêm chiều kênh
        image = np.expand_dims(image, axis=0)  # Thêm chiều batch

        # Chuyển đổi sang tensor
        image_tensor = torch.from_numpy(image).to(self.device)

        # Dự đoán
        with torch.no_grad():
            output = self.model(image_tensor)

        # Chuyển đổi kết quả dự đoán từ tensor sang numpy
        predicted_mask = torch.sigmoid(output).cpu().numpy()

        # Loại bỏ các chiều không cần thiết
        predicted_mask = np.squeeze(predicted_mask)  # Xoá chiều không cần thiết

        return predicted_mask

