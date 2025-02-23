import os
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from UNet import UNet

class IrisDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.bmp')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.bmp', '.png')

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Kiểm tra sự tồn tại của file
        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Image or mask file not found: {img_path} or {mask_path}")

        # Đọc ảnh và mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Kiểm tra việc đọc ảnh
        if image is None or mask is None:
            raise ValueError(f"Failed to read image or mask: {img_path} or {mask_path}")

        # Tiền xử lý ảnh
        image = image.astype(np.float32) / 255.0  # Chia cho 255 để giá trị nằm trong khoảng [0, 1]
        mask = mask.astype(np.float32) / 255.0    # Nếu mask cũng cần kiểu float

        # Chuyển đổi mặt nạ về dạng nhị phân (0 và 1)
        mask_binary = np.where(mask > 0.5, 1.0, 0.0)  # Giả sử giá trị lớn hơn 0.5 là đối tượng

        # Chuyển đổi sang tensor
        image = np.expand_dims(image, axis=0)  # Thêm chiều kênh
        mask = np.expand_dims(mask, axis=0)  # Thêm chiều kênh
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask

# Đường dẫn tới dữ liệu
image_dir = '../img-raw/'
mask_dir = 'mask_truth/'
# image_dir = 'TangCuong/img-raw'
# mask_dir = 'TangCuong/mask_truth'

# Tạo Dataset và DataLoader
dataset = IrisDataset(image_dir, mask_dir)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Tạo mô hình U-Net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")  # Hiển thị CPU hoặc GPU
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load('unet_iris_segmentation_14-11.pth', map_location=device, weights_only=True))

# Đặt loss function và optimizer
criterion = nn.BCEWithLogitsLoss()  # Sử dụng BCEWithLogitsLoss đã có sẵn sigmoid
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 100
train_losses = []  # Lưu trữ loss trong quá trình huấn luyện
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, masks)
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

# Vẽ biểu đồ Loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Time')
plt.show()

# Lưu mô hình
torch.save(model.state_dict(), 'unet_iris_segmentation_21-11.pth')
