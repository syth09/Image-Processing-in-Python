import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh màu tự nhiên (Thay đường dẫn bằng ảnh của bạn)
# Lưu ý: OpenCV đọc ảnh theo thứ tự BGR (Blue, Green, Red)
img_color = cv2.imread('Warframe0001.jpg')
h, w, c = img_color.shape

# =========================================================================
# 1. Vẽ biểu đồ Histogram của ảnh màu theo từng kênh (trên cùng 1 biểu đồ)
# =========================================================================
hist_B = [0] * 256
hist_G = [0] * 256
hist_R = [0] * 256

for y in range(h):
    for x in range(w):
        b = img_color[y, x, 0]
        g = img_color[y, x, 1]
        r = img_color[y, x, 2]
        hist_B[b] += 1
        hist_G[g] += 1
        hist_R[r] += 1

plt.figure(figsize=(10, 5))
plt.title("Histogram - 3 Kênh Màu (RGB)")
plt.plot(hist_R, color='red', label='Red')
plt.plot(hist_G, color='green', label='Green')
plt.plot(hist_B, color='blue', label='Blue')
plt.legend()
plt.xlim([0, 255])
plt.show()

# --> CÂU 2: Nhìn vào biểu đồ plt vừa vẽ để đưa ra nhận xét (như hướng dẫn ở phần I).

# =========================================================================
# 3. Chuyển đổi ảnh màu thành ảnh xám bằng trung bình cộng 3 kênh
# =========================================================================
img_gray = np.zeros((h, w), dtype=np.uint8)

for y in range(h):
    for x in range(w):
        b = img_color[y, x, 0]
        g = img_color[y, x, 1]
        r = img_color[y, x, 2]
        # Trung bình cộng của 3 kênh màu
        gray_val = int((int(b) + int(g) + int(r)) / 3)
        img_gray[y, x] = gray_val

cv2.imshow("Anh Xam", img_gray)

# =========================================================================
# 4. Vẽ biểu đồ Histogram của ảnh xám
# =========================================================================
hist_gray = [0] * 256
for y in range(h):
    for x in range(w):
        val = img_gray[y, x]
        hist_gray[val] += 1

plt.figure(figsize=(10, 4))
plt.title("Histogram - Ảnh Xám")
plt.bar(range(256), hist_gray, color='gray')
plt.xlim([0, 255])
plt.show()

# =========================================================================
# 5. Cân bằng Histogram cho ảnh xám (Thuật toán thủ công)
# =========================================================================
total_pixels = h * w

# Bước 1 & 2: Tính CDF (Tích luỹ xác suất)
cdf = [0] * 256
sum_hist = 0
for i in range(256):
    sum_hist += hist_gray[i]
    cdf[i] = sum_hist / total_pixels

# Bước 3: Tạo bảng ánh xạ giá trị mới
mapping_gray = [0] * 256
for i in range(256):
    mapping_gray[i] = round(cdf[i] * 255)

# Bước 4: Áp dụng lên ma trận ảnh xám
img_gray_equalized = np.zeros((h, w), dtype=np.uint8)
hist_gray_eq = [0] * 256

for y in range(h):
    for x in range(w):
        old_val = img_gray[y, x]
        new_val = mapping_gray[old_val]
        img_gray_equalized[y, x] = new_val
        hist_gray_eq[new_val] += 1

# Hiển thị kết quả câu 5
cv2.imshow("Anh Xam Da Can Bang", img_gray_equalized)
plt.figure(figsize=(10, 4))
plt.title("Histogram - Ảnh Xám Sau Cân Bằng")
plt.bar(range(256), hist_gray_eq, color='black')
plt.xlim([0, 255])
plt.show()


# =========================================================================
# 6. Cân bằng Histogram cho ảnh màu (Trong không gian RGB)
# Ý tưởng: Cân bằng độc lập trên từng kênh R, G, B rồi gộp lại.
# =========================================================================
def equalize_channel(channel_matrix, channel_hist, total_p):
    cdf_c = [0] * 256
    sum_h = 0
    for i in range(256):
        sum_h += channel_hist[i]
        cdf_c[i] = sum_h / total_p

    mapping_c = [round(cdf_c[i] * 255) for i in range(256)]

    out_matrix = np.zeros_like(channel_matrix)
    for y in range(channel_matrix.shape[0]):
        for x in range(channel_matrix.shape[1]):
            out_matrix[y, x] = mapping_c[channel_matrix[y, x]]
    return out_matrix


# Tách và cân bằng từng kênh sử dụng hàm tự định nghĩa bằng vòng lặp bên trên
img_color_equalized = np.zeros_like(img_color)
img_color_equalized[:, :, 0] = equalize_channel(img_color[:, :, 0], hist_B, total_pixels)
img_color_equalized[:, :, 1] = equalize_channel(img_color[:, :, 1], hist_G, total_pixels)
img_color_equalized[:, :, 2] = equalize_channel(img_color[:, :, 2], hist_R, total_pixels)

cv2.imshow("Anh Mau Da Can Bang (RGB)", img_color_equalized)

cv2.waitKey(0)
cv2.destroyAllWindows()