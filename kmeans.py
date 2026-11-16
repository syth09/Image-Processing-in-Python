import cv2
import numpy as np

# Đọc ảnh gốc
image = cv2.imread("anhnauy.jpg")
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# ĐIỀU CHỈNH THAM SỐ: Tăng k lên 5 để tách biệt màu áo tốt hơn
k, max_iter = 5, 100

# 1. Khởi tạo ngẫu nhiên k tâm cụm từ dữ liệu
np.random.seed(42)
random_indices = np.random.choice(len(pixel_values), k, replace=False)
centers = pixel_values[random_indices]

for i in range(max_iter):
    # 2. Gán nhãn: tìm tâm gần nhất cho từng pixel
    distances = np.linalg.norm(pixel_values[:, None] - centers[None, :], axis=2)
    labels = np.argmin(distances, axis=1)

    # 3. Tính lại tâm cụm
    new_centers = np.array(
        [pixel_values[labels == j].mean(axis=0) if np.any(labels == j) else centers[j] for j in range(k)])

    # 4. Kiểm tra hội tụ
    if np.allclose(centers, new_centers, atol=1e-4):
        print(f"Thuật toán hội tụ ở vòng lặp thứ: {i}")
        break
    centers = new_centers

# Chuyển tâm cụm về giá trị 8-bit (dạng BGR)
centers = np.uint8(centers)

print("\n--- KẾT QUẢ PHÂN TÍCH MÀU ÁO CỦA ĐỘI TUYỂN ---")

# Duyệt qua từng cụm để kiểm tra điều kiện
for idx, color in enumerate(centers):
    b, g, r = int(color[0]), int(color[1]), int(color[2])

    # In thông số cơ bản của cụm
    print(f"Cụm {idx} (B={b}, G={g}, R={r}):", end=" ")

    # Điều kiện 1: Áo Na Uy (R rất cao, B và G thấp)
    # Ta xét R > 150 và vượt trội hẳn so với B, G (ví dụ: hơn ít nhất 40 đơn vị)
    if r > 150 and (r - b > 40) and (r - g > 40):
        print("-> Đây chính là cụm Áo Na Uy (Đỏ).")

    # Điều kiện 2: Áo Anh và quần trắng (B, G, R đều cao tương đương và > 200)
    elif b > 200 and g > 200 and r > 200 and abs(b - r) < 30 and abs(g - r) < 30:
        print("-> Đây là cụm Áo Anh (và quần trắng).")

    else:
        print("-> Cụm màu nền, da người hoặc các chi tiết khác.")

# Thay mỗi điểm ảnh bằng tâm cụm tương ứng
segmented_image = centers[labels]
segmented_image = segmented_image.reshape(image.shape)

# Hiển thị kết quả
cv2.imshow("Anh goc", image)
cv2.imshow("Phan cum K-means (K=5)", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()