import cv2
# Đọc ảnh xám
image = cv2.imread('Lenna.png',cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Lỗi: Không thể mở ảnh Lenna.png")
else:
    cv2.imshow('Lenna image', image)
# Lấy kích thước ảnh
height, width = image.shape
# Thay đổi giá trị các pixel
for x in range(width):
    for y in range(height):
        image[x,y]=255-image[x,y]


cv2.imshow('Lenna image change', image)
cv2.waitKey(0)
cv2.destroyAllWindows()