import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import random

# Biến toàn cục lưu trữ dữ liệu ảnh
current_image = None
image_path = ""


# =====================================================================
# 1. CÁC HÀM XỬ LÝ ĐỒ HỌA & HIỆU ỨNG (TỐI ƯU BẰNG NUMPY)
# =====================================================================

# Chức năng: Đọc ảnh động từ máy tính
def open_file():
    global current_image, image_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
    )
    if file_path:
        image_path = file_path
        current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if current_image is not None:
            # Khởi tạo lại các thanh trượt điều khiển
            brightness_slider.set(0)
            contrast_slider.set(100)
            gamma_slider.set(10)  # Mức 10 tương đương gamma = 1.0 (giữ nguyên gốc)
            cv2.imshow("Original Image", current_image)
            cv2.waitKey(1)
            messagebox.showinfo("Thành công", "Đã tải dữ liệu ảnh lên hệ thống!")
        else:
            messagebox.showerror("Lỗi", "Định dạng file không hỗ trợ hoặc bị lỗi.")


# Chức năng: Điều chỉnh kết hợp Độ sáng & Độ tương phản
def update_brightness_contrast(*args):
    global current_image
    if current_image is None:
        return

    brightness = int(brightness_slider.get())
    contrast_scale = float(contrast_slider.get()) / 100.0

    adjusted = cv2.convertScaleAbs(current_image, alpha=contrast_scale, beta=brightness)
    cv2.imshow("Result - Brightness & Contrast", adjusted)
    cv2.waitKey(1)


# Chức năng: Biến đổi toán học Logarit (s = c * log(1 + r))
def trigger_log_transform():
    global current_image
    if current_image is None:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn và tải ảnh lên hệ thống trước!")
        return

    image_float = current_image.astype(np.float32)
    c = 255 / np.log(1 + np.max(image_float))
    log_image = c * np.log(1 + image_float)
    log_image = np.uint8(np.clip(log_image, 0, 255))

    cv2.imshow('Log transformed image', log_image)
    cv2.waitKey(1)


# Chức năng: Biến đổi Gamma thực thi qua LUT (Look-Up Table) bằng Numpy và OpenCV
def update_gamma(*args):
    global current_image
    if current_image is None:
        return

    # Lấy giá trị từ slider (chia cho 10 để có dải từ 0.1 đến 3.0)
    gamma = float(gamma_slider.get()) / 10.0
    if gamma == 0:
        gamma = 0.1  # Tránh lỗi chia/mũ với số 0

    # Tạo bảng tra cứu (Look-Up Table - LUT) bằng Numpy để tối ưu tốc độ tính toán
    # Thay vì tính lũy thừa cho từng pixel, ta chỉ cần tính trước cho 256 giá trị màu từ 0->255
    inv_gamma = 1.0 / gamma

    # Công thức chuẩn hóa dải pixel về [0, 1], tính lũy thừa, rồi đưa ngược lại về [0, 255]
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")

    # Áp dụng bảng màu LUT lên ảnh gốc bằng OpenCV (Tốc độ xử lý cực nhanh theo thời gian thực)
    gamma_image = cv2.LUT(current_image, table)

    cv2.imshow(f"Result - Gamma Transform (g={gamma})", gamma_image)
    cv2.waitKey(1)


# Chức năng: Tạo hiệu ứng mưa rơi phức hợp
def trigger_rain():
    global current_image
    if current_image is None:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
        return

    height, width = current_image.shape[:2]
    intensity = 0.4

    noise = np.random.randn(height, width) > 1.8
    rain_layer = np.zeros((height, width), dtype=np.uint8)
    rain_layer[noise] = 255

    kernel_size = 25
    kernel = np.zeros((kernel_size, kernel_size))
    np.fill_diagonal(kernel, 1)
    kernel /= kernel.sum()

    rain_layer = cv2.filter2D(rain_layer, -1, kernel)
    rain_img = cv2.addWeighted(current_image, 1 - intensity, rain_layer, intensity, 0)

    cv2.imshow("Result - Rain Effect", rain_img)
    cv2.waitKey(1)


# Chức năng: Tạo hiệu ứng tuyết rơi mềm mại
def trigger_snow():
    global current_image
    if current_image is None:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
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


# Chức năng: Tạo nhiễu hạt muối tiêu (Salt & Pepper Noise)
def trigger_noise():
    global current_image
    if current_image is None:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
        return

    res = current_image.copy()
    prob = 0.02
    rd_matrix = np.random.rand(*res.shape)

    res[rd_matrix < (prob / 2)] = 0
    res[(rd_matrix > (prob / 2)) & (rd_matrix < prob)] = 255

    cv2.imshow("Result - Noise Effect", res)
    cv2.waitKey(1)


# =====================================================================
# 2. XÂY DỰNG KHUNG GIAO DIỆN GUI VÀ THANH MENU (TKINTER)
# =====================================================================
root = tk.Tk()
root.title("Numpy & OpenCV Advanced Image Processor v3.0")
root.geometry("540x680")
root.resizable(False, False)

# --- THIẾT LẬP THANH MENU ---
menu_bar = tk.Menu(root)

file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Đọc ảnh đầu vào...", command=open_file)
file_menu.add_separator()
file_menu.add_command(label="Thoát chương trình", command=lambda: [cv2.destroyAllWindows(), root.quit()])
menu_bar.add_cascade(label="Hệ thống", menu=file_menu)

effects_menu = tk.Menu(menu_bar, tearoff=0)
effects_menu.add_command(label="Toán học Logarit", command=trigger_log_transform)
effects_menu.add_command(label="Hiệu ứng Mưa rơi", command=trigger_rain)
effects_menu.add_command(label="Hiệu ứng Tuyết rơi", command=trigger_snow)
effects_menu.add_command(label="Hiệu ứng Nhiễu hạt", command=trigger_noise)
menu_bar.add_cascade(label="Chức năng nâng cao", menu=effects_menu)

root.config(menu=menu_bar)

# --- THIẾT LẬP BẢNG ĐIỀU KHIỂN CHÍNH ---
label_title = tk.Label(root, text="BỘ CÔNG CỤ XỬ LÝ ẢNH CHUYÊN SÂU", font=("Arial", 13, "bold"), fg="#1E3A8A", pady=12)
label_title.pack()

frame_main = tk.Frame(root, padx=25, pady=5)
frame_main.pack(fill=tk.BOTH, expand=True)

# Nút Đọc file ảnh chính
btn_open = tk.Button(
    frame_main, text="📂 ĐỌC FILE ẢNH ĐẦU VÀO", command=open_file,
    font=("Arial", 10, "bold"), bg="#4B5563", fg="white", height=2, bd=0, cursor="hand2"
)
btn_open.pack(fill=tk.X, pady=6)

# PHÂN ĐOẠN ĐIỀU CHỈNH CHỈ SỐ MA TRẬN (SLIDERS)
label_sec1 = tk.Label(frame_main, text="— ĐIỀU CHỈNH THÔNG SỐ ĐỒ HỌA SONG SONG —", font=("Arial", 9, "bold"),
                      fg="#6B7280", pady=4)
label_sec1.pack()

# Slider 1: Độ sáng (Brightness)
label_b = tk.Label(frame_main, text="Độ sáng (Brightness):", font=("Arial", 9))
label_b.pack(anchor=tk.W)
brightness_slider = tk.Scale(
    frame_main, from_=-100, to=100, orient=tk.HORIZONTAL,
    command=update_brightness_contrast, bg="#F3F4F6", activebackground="#3B82F6"
)
brightness_slider.set(0)
brightness_slider.pack(fill=tk.X, pady=2)

# Slider 2: Độ tương phản (Contrast)
label_c = tk.Label(frame_main, text="Độ tương phản (Contrast %):", font=("Arial", 9))
label_c.pack(anchor=tk.W)
contrast_slider = tk.Scale(
    frame_main, from_=50, to=200, orient=tk.HORIZONTAL,
    command=update_brightness_contrast, bg="#F3F4F6", activebackground="#3B82F6"
)
contrast_slider.set(100)
contrast_slider.pack(fill=tk.X, pady=2)

# Slider 3: Biến đổi Gamma (Dải thanh trượt từ 1 đến 30 tương đương gamma từ 0.1 -> 3.0)
label_g = tk.Label(frame_main, text="Biến đổi số mũ Gamma (0.1 - 3.0):", font=("Arial", 9))
label_g.pack(anchor=tk.W)
gamma_slider = tk.Scale(
    frame_main, from_=1, to=30, orient=tk.HORIZONTAL,
    command=update_gamma, bg="#F3F4F6", activebackground="#7C3AED"
)
gamma_slider.set(10)  # Mặc định là 10 (tức là hệ số gamma = 1.0)
gamma_slider.pack(fill=tk.X, pady=2)

# PHÂN ĐOẠN CÁC HIỆU ỨNG VÀ PHÉP TOÁN
label_sec2 = tk.Label(frame_main, text="— TOÁN HỌC HÌNH ẢNH & THỜI TIẾT PHỨC HỢP —", font=("Arial", 9, "bold"),
                      fg="#6B7280", pady=6)
label_sec2.pack()

style_btn = {"font": ("Arial", 10, "bold"), "fg": "white", "height": 2, "bd": 0, "cursor": "hand2"}

# Nút: Tính toán Logarit
btn_log = tk.Button(frame_main, text="🧮 Biến Đổi Toán Học Logarit (Log Transform)", command=trigger_log_transform,
                    bg="#7C3AED", activebackground="#6D28D9", **style_btn)
btn_log.pack(fill=tk.X, pady=3)

# Nút: Hiệu ứng mưa
btn_rain = tk.Button(frame_main, text="🌧️ Tạo Hiệu Ứng Mưa Rơi (Rain Effect)", command=trigger_rain, bg="#0284C7",
                     activebackground="#0369A1", **style_btn)
btn_rain.pack(fill=tk.X, pady=3)

# Nút: Hiệu ứng tuyết
btn_snow = tk.Button(frame_main, text="❄️ Tạo Hiệu Ứng Tuyết Rơi (Snow Effect)", command=trigger_snow, bg="#10B981",
                     activebackground="#059669", **style_btn)
btn_snow.pack(fill=tk.X, pady=3)

# Nút: Tạo nhiễu hạt
btn_noise = tk.Button(frame_main, text="📺 Tạo Hiệu Ứng Nhiễu Hạt (Salt & Pepper)", command=trigger_noise, bg="#D97706",
                      activebackground="#B45309", **style_btn)
btn_noise.pack(fill=tk.X, pady=3)

# Nút: Thoát ứng dụng
btn_exit = tk.Button(
    frame_main, text="Thoát ứng dụng", command=lambda: [cv2.destroyAllWindows(), root.quit()],
    font=("Arial", 10, "bold"), bg="#EF4444", fg="white", activebackground="#DC2626", height=2, bd=0, cursor="hand2"
)
btn_exit.pack(fill=tk.X, pady=12)

root.mainloop()