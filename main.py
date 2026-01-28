import matplotlib.pyplot as plt
import numpy as np
import cv2 

#Black and white colormap

#mang pixel values to 0-255 range (8 bits)
M = 1024 #1024x768
N = 768
K = 100
# ảnh có MxN pixels 
img = np.zeros((M,N), dtype=np.uint8)
for i in range(M):
    img[i,0:100] = i % 256
# plt.imshow(img, cmap='gray')
# plt.show()
cv2.imshow('Gray Map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Color colormap
img_color = np.zeros((M,N,3), dtype=np.uint8) #Red, Green, Blue

for i in range(M):
    img_color[i,0:64,0] = i % 256  #Red channel
    img_color[i,100:150,1] = i % 256  #Green channel
    img_color[i,200:250,2] = i % 256  #Blue channel
# plt.imshow(img_color)
# plt.show()
# print(img_color)
cv2.imshow('Color Map', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Tạo một bức hình có kích thước 1024x768 với các giá trị pixel ngẫu nhiên
img_random = np.random.randint(0, 256, (M, N, 3), dtype=np.uint8)

# Vẽ một đường chéo màu đỏ
for i in range(min(M, N)):
    img_random[i, i] = [0, 0, 255]  # Màu đỏ (BGR format in OpenCV)

cv2.imshow('Random with Diagonal', img_random)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Vẽ các chữ số La Mã I đến XII tương tự như trên một chiếc đồng hồ
import time
from datetime import datetime

roman_numbers = [
    "XII", "I", "II", "III", "IV", "V",
    "VI", "VII", "VIII", "IX", "X", "XI"
]

# Tạo figure và axis
plt.ion()  # Interactive mode
fig, ax = plt.subplots(figsize=(8, 8))

try:
    while True:
        ax.clear()
        
        # Vẽ vòng tròn
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax.add_artist(circle)
        
        # Vẽ số La Mã
        for i, roman in enumerate(roman_numbers):
            angle = np.pi/2 - i * (2 * np.pi / 12)
            x = 0.85 * np.cos(angle)
            y = 0.85 * np.sin(angle)
            ax.text(
                x, y, roman,
                ha='center', va='center',
                fontsize=14, fontweight='bold'
            )
        
        # Lấy thời gian hiện tại
        now = datetime.now()
        hours = now.hour % 12
        minutes = now.minute
        seconds = now.second
        
        # Tính góc cho kim phút (360 độ = 60 phút)
        minute_angle = np.pi / 2 - (minutes + seconds / 60) * (2 * np.pi / 60)
        ax.plot([0, 0.7*np.cos(minute_angle)],
                [0, 0.7*np.sin(minute_angle)], linewidth=2, color='blue', label='Phút')
        
        # Tính góc cho kim giờ (360 độ = 12 giờ)
        hour_angle = np.pi / 2 - (hours + minutes / 60) * (2 * np.pi / 12)
        ax.plot([0, 0.5*np.cos(hour_angle)],
                [0, 0.5*np.sin(hour_angle)], linewidth=4, color='red', label='Giờ')
        
        # Tính góc cho kim giây (360 độ = 60 giây)
        second_angle = np.pi / 2 - seconds * (2 * np.pi / 60)
        ax.plot([0, 0.8*np.cos(second_angle)],
                [0, 0.8*np.sin(second_angle)], linewidth=1, color='green', label='Giây')
        
        # Set khung nhìn
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title(f"Dong ho so La Ma - {now.strftime('%H:%M:%S')}")
        plt.pause(0.1)  # Cập nhật mỗi 0.1 giây
        
except KeyboardInterrupt:
    print("Đã dừng đồng hồ")
    plt.close()

