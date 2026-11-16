import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import random

# Biến toàn cục để lưu trữ ảnh đang xử lý
current_image = None
image_path = ""


# =====================================================================
# 1. CÁC HÀM XỬ LÝ HIỆU ỨNG ẢNH
# =====================================================================

# Hàm đọc ảnh từ máy tính
def open_file():
    global current_image, image_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
    )
    if file_path:
        image_path = file_path
        # Đọc ảnh ở dạng ảnh xám theo logic bài tập ban đầu
        current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if current_image is not None:
            # Reset thanh trượt độ sáng về mức 0 khi mở ảnh mới
            brightness_slider.set(0)
            cv2.imshow("Original Image", current_image)
            cv2.waitKey(1)
            messagebox.showinfo("Thành công", "Đã tải ảnh lên thành công!")
        else:
            messagebox.showerror("Lỗi", "Không thể đọc file ảnh này.")


# Hàm tăng/giảm độ sáng (Được gọi tự động khi kéo Slider)
def change_brightness(val):
    global current_image
    if current_image is None:
        return

    # Lấy giá trị từ thanh trượt (-100 đến 100)
    brightness = int(val)

    # Sử dụng hàm cv2.convertScaleAbs để tăng/giảm độ sáng an toàn (chống tràn pixel quá 255 hoặc dưới 0)
    # alpha=1 (giữ nguyên độ tương phản), beta=brightness (cộng/trừ độ sáng)
    adjusted = cv2.convertScaleAbs(current_image, alpha=1, beta=brightness)

    cv2.imshow("Result - Brightness Adjusted", adjusted)
    cv2.waitKey(1)


# Hàm tạo hiệu ứng mưa rơi phức hợp
def trigger_rain():
    global current_image
    if current_image is None:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn và tải ảnh lên trước!")
        return

    height, width = current_image.shape[:2]
    intensity = 0.4

    # Tạo lớp nhiễu mưa rơi
    noise = np.random.randn(height, width) > 1.8
    rain_layer = np.zeros((height, width), dtype=np.uint8)
    rain_layer[noise] = 255

    # Bộ lọc chuyển động nghiêng
    kernel_size = 25
    kernel = np.zeros((kernel_size, kernel_size))
    np.fill_diagonal(kernel, 1)
    kernel /= kernel.sum()

    rain_layer = cv2.filter2D(rain_layer, -1, kernel)
    rain_img = cv2.addWeighted(current_image, 1 - intensity, rain_layer, intensity, 0)

    cv2.imshow("Result - Rain Effect", rain_img)
    cv2.waitKey(1)


# Hàm tạo hiệu ứng tuyết rơi
def trigger_snow():
    global current_image
    if current_image is None:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn và tải ảnh lên trước!")
        return

    height, width = current_image.shape[:2]
    res = current_image.copy()
    num_snowflakes = 400

    for _ in range(num_snowflakes):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(1, 3)
        snow_color = random.randint(170, 255)
        cv2.circle(res, (x, y), radius, snow_color, -1)

    cv2.imshow("Result - Snow Effect", res)
    cv2.waitKey(1)


# Hàm tạo hiệu ứng nhiễu hạt (Salt & Pepper Noise)
def trigger_noise():
    global current_image
    if current_image is None:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn và tải ảnh lên trước!")
        return

    height, width = current_image.shape[:2]
    res = current_image.copy()

    # Tỷ lệ nhiễu xuất hiện trên ảnh (0.02 = 2% số pixel bị nhiễu)
    noise_ratio = 0.02
    num_noise_pixels = int(height * width * noise_ratio)

    for _ in range(num_noise_pixels // 2):
        # Tạo điểm nhiễu trắng (Salt)
        y_salt = random.randint(0, height - 1)
        x_salt = random.randint(0, width - 1)
        res[y_salt, x_salt] = 255

        # Tạo điểm nhiễu đen (Pepper)
        y_pepper = random.randint(0, height - 1)
        x_pepper = random.randint(0, width - 1)
        res[y_pepper, x_pepper] = 0

    cv2.imshow("Result - Noise Effect", res)
    cv2.waitKey(1)


# =====================================================================
# 2. THIẾT KẾ GUI VÀ MENU CHỨC NĂNG (TKINTER)
# =====================================================================
root = tk.Tk()
root.title("Advanced Image Processor GUI")
root.geometry("500x520")
root.resizable(False, False)

# --- THIẾT LẬP THANH MENU PHÍA TRÊN ---
menu_bar = tk.Menu(root)

# Menu Hệ thống (File)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Đọc ảnh từ máy...", command=open_file)
file_menu.add_separator()
file_menu.add_command(label="Thoát ứng dụng", command=lambda: [cv2.destroyAllWindows(), root.quit()])
menu_bar.add_cascade(label="Hệ thống", menu=file_menu)

# Menu Hiệu ứng nhanh
effects_menu = tk.Menu(menu_bar, tearoff=0)
effects_menu.add_command(label="Hiệu ứng Mưa rơi", command=trigger_rain)
effects_menu.add_command(label="Hiệu ứng Tuyết rơi", command=trigger_snow)
effects_menu.add_command(label="Hiệu ứng Nhiễu hạt", command=trigger_noise)
menu_bar.add_cascade(label="Hiệu ứng", menu=effects_menu)

root.config(menu=menu_bar)

# --- GIAO DIỆN NÚT BẤM VÀ ĐIỀU KHIỂN ---
label_title = tk.Label(root, text="BỘ XỬ LÝ ẢNH ĐA CHỨC NĂNG", font=("Arial", 14, "bold"), fg="#1E3A8A", pady=15)
label_title.pack()

frame_main = tk.Frame(root, padx=25, pady=5)
frame_main.pack(fill=tk.BOTH, expand=True)

# 1. Nút Đọc ảnh chính trên giao diện
btn_open = tk.Button(
    frame_main, text="📂 ĐỌC FILE ẢNH (CHỌN TỪ MÁY TÍNH)", command=open_file,
    font=("Arial", 10, "bold"), bg="#4B5563", fg="white", height=2, bd=0, cursor="hand2"
)
btn_open.pack(fill=tk.X, pady=8)

# Phân tách vùng bằng nhãn
label_section1 = tk.Label(frame_main, text="— THANH TRƯỢT ĐIỀU CHỈNH ĐỘ SÁNG —", font=("Arial", 9, "bold"),
                          fg="#6B7280", pady=5)
label_section1.pack()

# 2. Thanh trượt Slider Tăng/Giảm độ sáng
brightness_slider = tk.Scale(
    frame_main, from_=-100, to=100, orient=tk.HORIZONTAL,
    command=change_brightness, font=("Arial", 9),
    tickinterval=50, bg="#F3F4F6", fg="#374151", activebackground="#3B82F6"
)
brightness_slider.set(0)  # Mặc định ở giữa (bằng 0)
brightness_slider.pack(fill=tk.X, pady=5)

label_section2 = tk.Label(frame_main, text="— DANH SÁCH HIỆU ỨNG THỜI TIẾT & TÍN HIỆU —", font=("Arial", 9, "bold"),
                          fg="#6B7280", pady=5)
label_section2.pack()

# Các cấu hình chung cho nhóm button hiệu ứng
effect_btn_style = {
    "font": ("Arial", 10, "bold"), "fg": "white", "height": 2, "bd": 0, "cursor": "hand2"
}

# 3. Nút Mưa rơi
btn_rain = tk.Button(frame_main, text="🌧️ Hiệu Ứng Mưa Rơi (Rain)", command=trigger_rain, bg="#0284C7",
                     activebackground="#0369A1", **effect_btn_style)
btn_rain.pack(fill=tk.X, pady=4)

# 4. Nút Tuyết rơi
btn_snow = tk.Button(frame_main, text="❄️ Hiệu Ứng Tuyết Rơi (Snow)", command=trigger_snow, bg="#10B981",
                     activebackground="#059669", **effect_btn_style)
btn_snow.pack(fill=tk.X, pady=4)

# 5. Nút Nhiễu hạt
btn_noise = tk.Button(frame_main, text="📺 Hiệu Ứng Nhiễu Muối Tiêu (Noise)", command=trigger_noise, bg="#D97706",
                      activebackground="#B45309", **effect_btn_style)
btn_noise.pack(fill=tk.X, pady=4)

# Nút Thoát chương trình nhanh
btn_exit = tk.Button(
    frame_main, text="Thoát phần mềm", command=lambda: [cv2.destroyAllWindows(), root.quit()],
    font=("Arial", 10, "bold"), bg="#EF4444", fg="white", activebackground="#DC2626", height=2, bd=0, cursor="hand2"
)
btn_exit.pack(fill=tk.X, pady=15)

root.mainloop()