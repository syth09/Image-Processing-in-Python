import cv2
import numpy as np

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """Hàm sinh nhiễu muối tiêu"""
    noisy_image = image.copy()
    # Số lượng pixel bị nhiễu muối
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[tuple(coords)] = 255

    # Số lượng pixel bị nhiễu tiêu
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(coords)] = 0
    return noisy_image

def add_gaussian_noise(image, mean=0, sigma=25):
    """Hàm sinh nhiễu Gauss"""
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image.astype(np.float32) + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# 1. Đọc ảnh gốc dưới dạng ảnh xám (Grayscale)
# Thay "Lena.png" bằng đường dẫn ảnh của em
img_path = "Lena.png"
original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if original_img is None:
    print(f"❌ Không tìm thấy ảnh '{img_path}'. Tự động tạo ảnh mẫu để chạy...")
    # Tạo ảnh mẫu nếu không có sẵn file ảnh để tránh lỗi
    original_img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(original_img, (50, 50), (250, 250), 180, -1)
    cv2.circle(original_img, (150, 150), 60, 80, -1)

# 2. Tạo các ảnh bị nhiễu
sp_noisy_img = add_salt_and_pepper_noise(original_img, 0.04, 0.04)
gaussian_noisy_img = add_gaussian_noise(original_img, mean=0, sigma=25)

# 3. Áp dụng bộ lọc TỐT NHẤT tương ứng cho từng loại nhiễu
# Lọc Trung vị (Median Filter) kích thước 3x3 cho nhiễu Muối tiêu
sp_filtered = cv2.medianBlur(sp_noisy_img, 3)

# Lọc Trung bình (Mean Filter) kích thước 3x3 cho nhiễu Gauss
gaussian_filtered = cv2.blur(gaussian_noisy_img, (3, 3))

# 4. Lưu lại các ảnh minh chứng ra thư mục
cv2.imwrite("1_nhieu_muoi_tieu.png", sp_noisy_img)
cv2.imwrite("2_loc_trung_vi_tot_nhat.png", sp_filtered)
cv2.imwrite("3_nhieu_gauss.png", gaussian_noisy_img)
cv2.imwrite("4_loc_trung_binh_tot_nhat.png", gaussian_filtered)

print("✅ Đã sinh và lọc ảnh thành công! Hãy kiểm tra các file ảnh trong thư mục:")
print("- 1_nhieu_muoi_tieu.png (Ảnh nhiễu muối tiêu)")
print("- 2_loc_trung_vi_tot_nhat.png (Ảnh kết quả lọc tốt nhất cho muối tiêu)")
print("- 3_nhieu_gauss.png (Ảnh nhiễu Gauss)")
print("- 4_loc_trung_binh_tot_nhat.png (Ảnh kết quả lọc tốt nhất cho Gauss)")