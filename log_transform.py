import cv2
import numpy as np
import math

# Đọc ảnh xám
image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Lỗi: Không thể mở ảnh Lenna.png")
    exit()

cv2.imshow('Original image', image)

# Chuyển đổi sang float để tránh tràn số khi log
image_float = image.astype(np.float32)

# Tính hằng số c (log transform: s = c * log(1 + r))
c = 255 / np.log(1 + np.max(image_float))

# Áp dụng phép biến đổi log
log_image = c * np.log(1 + image_float)

# Chuyển về kiểu uint8
log_image = np.uint8(np.clip(log_image, 0, 255))

# Hiển thị ảnh kết quả
cv2.imshow('Log transformed image', log_image)
cv2.waitKey(0)
cv2.destroyAllWindows()