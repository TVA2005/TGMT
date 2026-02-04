# ==============================================
# TẤT CẢ CÁC THƯ VIỆN CẦN THIẾT
# ==============================================
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import time
from datetime import datetime
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PHẦN 1: TẠO HÌNH ẢNH MÀU XÁM VÀ MÀU
# ==============================================

print("=" * 60)
print("PHẦN 1: TẠO HÌNH ẢNH MÀU XÁM VÀ MÀU")
print("=" * 60)

# Black and white colormap
M = 1024  # Chiều rộng
N = 768   # Chiều cao

# Ảnh đen trắng với gradient
img_gray = np.zeros((M, N), dtype=np.uint8)
for i in range(M):
    img_gray[i, 0:100] = i % 256
    
# Hiển thị bằng matplotlib
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Gray Gradient (Matplotlib)')
plt.axis('off')

# Hiển thị bằng OpenCV
cv2.imshow('Gray Gradient (OpenCV)', img_gray)
cv2.waitKey(1000)  # Hiển thị 1 giây
cv2.destroyAllWindows()

# Color colormap
img_color = np.zeros((M, N, 3), dtype=np.uint8)  # Red, Green, Blue

for i in range(M):
    img_color[i, 0:64, 0] = i % 256    # Red channel
    img_color[i, 100:164, 1] = i % 256  # Green channel
    img_color[i, 200:264, 2] = i % 256  # Blue channel

# Hiển thị bằng matplotlib
plt.subplot(2, 2, 2)
plt.imshow(img_color)
plt.title('Color Gradient (Matplotlib)')
plt.axis('off')

# Hiển thị bằng OpenCV (chuyển RGB sang BGR)
img_color_bgr = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
cv2.imshow('Color Gradient (OpenCV)', img_color_bgr)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# ==============================================
# PHẦN 2: TẠO HÌNH ẢNH NGẪU NHIÊN
# ==============================================

print("\n" + "=" * 60)
print("PHẦN 2: TẠO HÌNH ẢNH NGẪU NHIÊN")
print("=" * 60)

# Tạo hình ảnh ngẫu nhiên
img_random = np.random.randint(0, 256, (M, N, 3), dtype=np.uint8)

# Vẽ đường chéo màu đỏ (OpenCV dùng BGR format)
for i in range(min(M, N)):
    img_random[i, i] = [0, 0, 255]  # Màu đỏ (BGR)

# Hiển thị
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(img_random, cv2.COLOR_BGR2RGB))
plt.title('Random Image with Red Diagonal')
plt.axis('off')

cv2.imshow('Random Image with Diagonal', img_random)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# ==============================================
# PHẦN 3: HÀM TẢI HÌNH CON MÈO
# ==============================================

print("\n" + "=" * 60)
print("PHẦN 3: TẢI HÌNH CON MÈO")
print("=" * 60)

def load_cat_image(use_internet=True):
    """
    Tải hình ảnh con mèo từ internet hoặc tạo hình giả
    
    Parameters:
    - use_internet: bool, có sử dụng internet để tải ảnh không
    """
    
    if use_internet:
        try:
            print("Đang tải hình ảnh con mèo từ internet...")
            # URL hình ảnh con mèo
            url = "https://cdn.pixabay.com/photo/2017/02/20/18/03/cat-2083492_1280.jpg"
            
            # Tải ảnh
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            img_data = BytesIO(response.content)
            cat_img = mpimg.imread(img_data)
            
            print("✓ Đã tải thành công hình ảnh con mèo từ internet!")
            return cat_img
            
        except Exception as e:
            print(f"✗ Không thể tải ảnh từ internet: {e}")
            print("Chuyển sang tạo hình ảnh con mèo giả...")
            return create_fake_cat_image()
    else:
        print("Đang tạo hình ảnh con mèo giả...")
        return create_fake_cat_image()

def create_fake_cat_image():
    """Tạo hình ảnh con mèo giả"""
    
    # Kích thước ảnh
    height, width = 600, 800
    cat_img = np.ones((height, width, 3))
    
    # Màu nền (màu be nhạt)
    cat_img[:, :, 0] = 0.96  # R
    cat_img[:, :, 1] = 0.93  # G
    cat_img[:, :, 2] = 0.88  # B
    
    # Tọa độ trung tâm
    center_x, center_y = width // 2, height // 2
    
    # Vẽ đầu mèo (hình tròn)
    head_radius = min(width, height) // 4
    
    for y in range(height):
        for x in range(width):
            # Tính khoảng cách đến tâm
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Vẽ đầu mèo
            if dist <= head_radius:
                # Màu lông mèo (màu cam nhạt)
                cat_img[y, x, 0] = 0.95  # R
                cat_img[y, x, 1] = 0.75  # G
                cat_img[y, x, 2] = 0.55  # B
                
                # Vẽ tai trái
                ear_left_dist = np.sqrt((x - (center_x - head_radius*0.6))**2 + 
                                        (y - (center_y - head_radius*0.7))**2)
                if ear_left_dist <= head_radius * 0.3:
                    cat_img[y, x, :] = [0.9, 0.65, 0.5]
                
                # Vẽ tai phải
                ear_right_dist = np.sqrt((x - (center_x + head_radius*0.6))**2 + 
                                         (y - (center_y - head_radius*0.7))**2)
                if ear_right_dist <= head_radius * 0.3:
                    cat_img[y, x, :] = [0.9, 0.65, 0.5]
                
                # Vẽ mắt trái
                eye_left_dist = np.sqrt((x - (center_x - head_radius*0.3))**2 + 
                                        (y - (center_y - head_radius*0.1))**2)
                if eye_left_dist <= head_radius * 0.15:
                    cat_img[y, x, :] = [0, 0, 0]  # Mắt đen
                    # Đồng tử
                    if eye_left_dist <= head_radius * 0.05:
                        cat_img[y, x, :] = [1, 1, 1]  # Đồng tử trắng
                
                # Vẽ mắt phải
                eye_right_dist = np.sqrt((x - (center_x + head_radius*0.3))**2 + 
                                         (y - (center_y - head_radius*0.1))**2)
                if eye_right_dist <= head_radius * 0.15:
                    cat_img[y, x, :] = [0, 0, 0]  # Mắt đen
                    # Đồng tử
                    if eye_right_dist <= head_radius * 0.05:
                        cat_img[y, x, :] = [1, 1, 1]  # Đồng tử trắng
                
                # Vẽ mũi
                nose_dist = np.sqrt((x - center_x)**2 + 
                                   (y - (center_y + head_radius*0.1))**2)
                if nose_dist <= head_radius * 0.1:
                    cat_img[y, x, :] = [0.9, 0.3, 0.3]  # Mũi hồng
                
                # Vẽ miệng
                if abs(y - (center_y + head_radius*0.25)) < 2 and abs(x - center_x) < head_radius * 0.2:
                    cat_img[y, x, :] = [0, 0, 0]
                
                # Vẽ ria mép
                for k in range(3):
                    ria_y = center_y + head_radius*(0.1 + k*0.05)
                    if abs(y - ria_y) < 2:
                        # Ria trái
                        if abs(x - (center_x - head_radius*0.5)) < 3:
                            cat_img[y, x-3:x+3, :] = [0, 0, 0]
                        # Ria phải
                        if abs(x - (center_x + head_radius*0.5)) < 3:
                            cat_img[y, x-3:x+3, :] = [0, 0, 0]
    
    # Thêm chữ "MEOW"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_img_cv = (cat_img * 255).astype(np.uint8)
    cat_img_cv = cv2.cvtColor(cat_img_cv, cv2.COLOR_RGB2BGR)
    cv2.putText(cat_img_cv, 'MEOW!', (width//2 - 100, 50), 
                font, 2, (100, 50, 200), 3, cv2.LINE_AA)
    cat_img = cv2.cvtColor(cat_img_cv, cv2.COLOR_BGR2RGB) / 255.0
    
    print("✓ Đã tạo thành công hình ảnh con mèo giả!")
    return cat_img

# Tải hình ảnh con mèo
try:
    cat_img = load_cat_image(use_internet=True)
except:
    cat_img = create_fake_cat_image()

# Hiển thị hình ảnh con mèo
plt.subplot(2, 2, 4)
plt.imshow(cat_img)
plt.title('Hình nền con mèo')
plt.axis('off')

plt.tight_layout()
plt.show()

# ==============================================
# PHẦN 4: TẠO ĐỒNG HỒ LA MÃ VỚI HÌNH NỀN CON MÈO
# ==============================================

print("\n" + "=" * 60)
print("PHẦN 4: ĐỒNG HỒ LA MÃ VỚI HÌNH NỀN CON MÈO")
print("=" * 60)
print("Đang khởi tạo đồng hồ...")
print("Nhấn Ctrl+C để dừng chương trình")
print("=" * 60)

# Các số La Mã
roman_numbers = [
    "XII", "I", "II", "III", "IV", "V",
    "VI", "VII", "VIII", "IX", "X", "XI"
]

# Tạo figure và axis
plt.ion()  # Chế độ interactive
fig = plt.figure(figsize=(12, 10))
fig.canvas.manager.set_window_title('Đồng hồ La Mã với hình nền con mèo')

ax = plt.gca()

# Thêm hình nền con mèo
imagebox = OffsetImage(cat_img, zoom=0.8)
ab = AnnotationBbox(imagebox, (0, 0), frameon=False, 
                    pad=0, box_alignment=(0.5, 0.5))
ax.add_artist(ab)

# Thông tin cập nhật
update_count = 0
start_time = time.time()

try:
    while True:
        # Xóa các thành phần cũ của đồng hồ (giữ lại hình nền)
        for artist in ax.artists:
            if artist != ab:  # Giữ lại hình nền con mèo
                artist.remove()
        
        for line in ax.lines:
            line.remove()
        
        for text in ax.texts:
            text.remove()
        
        # Vẽ vòng tròn đồng hồ
        circle_outer = plt.Circle((0, 0), 1.1, fill=False, 
                                 color='darkblue', linewidth=4, alpha=0.7)
        circle_inner = plt.Circle((0, 0), 0.95, fill=False, 
                                 color='gold', linewidth=2, alpha=0.7)
        ax.add_artist(circle_outer)
        ax.add_artist(circle_inner)
        
        # Vẽ các số La Mã
        for i, roman in enumerate(roman_numbers):
            angle = np.pi/2 - i * (2 * np.pi / 12)
            x = 0.8 * np.cos(angle)
            y = 0.8 * np.sin(angle)
            
            # Màu sắc cho các số
            color_idx = i / len(roman_numbers)
            text_color = plt.cm.hsv(color_idx)
            
            # Vẽ nền cho số
            circle_bg = plt.Circle((x, y), 0.08, color='white', 
                                  alpha=0.8, zorder=2)
            ax.add_artist(circle_bg)
            
            # Vẽ số
            ax.text(
                x, y, roman,
                ha='center', va='center',
                fontsize=20, fontweight='bold',
                color=text_color,
                zorder=3,
                fontfamily='serif'
            )
        
        # Lấy thời gian hiện tại
        now = datetime.now()
        hours = now.hour % 12
        minutes = now.minute
        seconds = now.second
        milliseconds = now.microsecond // 10000
        
        # Tính góc cho các kim đồng hồ
        # Kim giờ
        hour_angle = np.pi/2 - (hours + minutes/60 + seconds/3600) * (2*np.pi/12)
        hour_x = 0.5 * np.cos(hour_angle)
        hour_y = 0.5 * np.sin(hour_angle)
        
        # Kim phút
        minute_angle = np.pi/2 - (minutes + seconds/60) * (2*np.pi/60)
        minute_x = 0.7 * np.cos(minute_angle)
        minute_y = 0.7 * np.sin(minute_angle)
        
        # Kim giây
        second_angle = np.pi/2 - (seconds + milliseconds/100) * (2*np.pi/60)
        second_x = 0.8 * np.cos(second_angle)
        second_y = 0.8 * np.sin(second_angle)
        
        # Vẽ kim giờ
        hour_line, = ax.plot([0, hour_x], [0, hour_y], 
                           linewidth=8, color='darkred', 
                           alpha=0.9, zorder=4, 
                           solid_capstyle='round')
        
        # Vẽ kim phút
        minute_line, = ax.plot([0, minute_x], [0, minute_y], 
                             linewidth=5, color='darkblue', 
                             alpha=0.8, zorder=4,
                             solid_capstyle='round')
        
        # Vẽ kim giây
        second_line, = ax.plot([0, second_x], [0, second_y], 
                             linewidth=2, color='green', 
                             alpha=0.7, zorder=5,
                             solid_capstyle='round')
        
        # Vẽ tâm đồng hồ
        ax.plot(0, 0, 'o', markersize=15, 
                color='gold', alpha=0.9, zorder=6)
        ax.plot(0, 0, 'o', markersize=8, 
                color='red', alpha=0.9, zorder=6)
        
        # Vẽ các vạch chia phút
        for minute in range(0, 60, 5):
            angle = np.pi/2 - minute * (2*np.pi/60)
            inner_radius = 0.9
            outer_radius = 1.0 if minute % 15 == 0 else 0.95
            
            start_x = inner_radius * np.cos(angle)
            start_y = inner_radius * np.sin(angle)
            end_x = outer_radius * np.cos(angle)
            end_y = outer_radius * np.sin(angle)
            
            linewidth = 3 if minute % 15 == 0 else 1
            color = 'red' if minute % 15 == 0 else 'darkgray'
            
            ax.plot([start_x, end_x], [start_y, end_y], 
                   linewidth=linewidth, color=color, alpha=0.6)
        
        # Thông tin thời gian
        time_str = now.strftime('%H:%M:%S')
        date_str = now.strftime('%d/%m/%Y')
        day_str = now.strftime('%A')
        
        # Hiển thị thông tin
        ax.text(0, -1.3, f"THỜI GIAN: {time_str}", 
               ha='center', fontsize=16, 
               fontweight='bold', color='darkblue',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='lightyellow', 
                        alpha=0.8, edgecolor='gold'))
        
        ax.text(0, -1.45, f"{date_str} - {day_str}", 
               ha='center', fontsize=14, 
               color='darkred', style='italic')
        
        # Tiêu đề
        title_text = "ĐỒNG HỒ LA MÃ VỚI HÌNH NỀN CON MÈO"
        ax.set_title(title_text, fontsize=22, fontweight='bold', 
                    color='darkblue', pad=25)
        
        # Chú thích
        ax.text(1.1, 0.8, "CHÚ THÍCH:", 
               fontsize=12, fontweight='bold', color='darkgreen')
        ax.text(1.1, 0.7, "• Kim giờ: Đỏ", 
               fontsize=10, color='darkred')
        ax.text(1.1, 0.6, "• Kim phút: Xanh dương", 
               fontsize=10, color='darkblue')
        ax.text(1.1, 0.5, "• Kim giây: Xanh lá", 
               fontsize=10, color='green')
        
        # Thống kê
        update_count += 1
        elapsed_time = time.time() - start_time
        ax.text(-1.1, -1.4, f"Cập nhật: {update_count} lần", 
               fontsize=10, color='purple')
        ax.text(-1.1, -1.5, f"Thời gian chạy: {elapsed_time:.1f}s", 
               fontsize=10, color='purple')
        
        # Cài đặt khung nhìn
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.6, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Cập nhật đồ họa
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Chờ 0.05 giây (cập nhật 20 lần/giây)
        plt.pause(0.05)
        
except KeyboardInterrupt:
    print("\n" + "=" * 60)
    print("CHƯƠNG TRÌNH ĐÃ DỪNG")
    print(f"Tổng thời gian chạy: {time.time() - start_time:.2f} giây")
    print(f"Số lần cập nhật: {update_count}")
    print("=" * 60)
    
except Exception as e:
    print(f"\nCó lỗi xảy ra: {e}")
    
finally:
    plt.ioff()
    plt.close('all')
    cv2.destroyAllWindows()
    print("\nĐã đóng tất cả cửa sổ.")

# ==============================================
# PHẦN 5: LƯU HÌNH ẢNH ĐỒNG HỒ
# ==============================================

print("\n" + "=" * 60)
print("PHẦN 5: LƯU HÌNH ẢNH ĐỒNG HỒ")
print("=" * 60)

# Tạo một phiên bản tĩnh của đồng hồ để lưu
fig_static, ax_static = plt.subplots(figsize=(10, 10))

# Thêm hình nền
imagebox_static = OffsetImage(cat_img, zoom=0.8)
ab_static = AnnotationBbox(imagebox_static, (0, 0), frameon=False, 
                          pad=0, box_alignment=(0.5, 0.5))
ax_static.add_artist(ab_static)

# Lấy thời gian hiện tại
now = datetime.now()
hours = now.hour % 12
minutes = now.minute
seconds = now.second

# Vẽ đồng hồ (tương tự như trên)
circle_outer = plt.Circle((0, 0), 1.1, fill=False, 
                         color='darkblue', linewidth=4, alpha=0.7)
circle_inner = plt.Circle((0, 0), 0.95, fill=False, 
                         color='gold', linewidth=2, alpha=0.7)
ax_static.add_artist(circle_outer)
ax_static.add_artist(circle_inner)

# Vẽ số La Mã
for i, roman in enumerate(roman_numbers):
    angle = np.pi/2 - i * (2 * np.pi / 12)
    x = 0.8 * np.cos(angle)
    y = 0.8 * np.sin(angle)
    
    color_idx = i / len(roman_numbers)
    text_color = plt.cm.hsv(color_idx)
    
    circle_bg = plt.Circle((x, y), 0.08, color='white', 
                          alpha=0.8, zorder=2)
    ax_static.add_artist(circle_bg)
    
    ax_static.text(
        x, y, roman,
        ha='center', va='center',
        fontsize=20, fontweight='bold',
        color=text_color,
        zorder=3,
        fontfamily='serif'
    )

# Tính góc và vẽ kim đồng hồ
hour_angle = np.pi/2 - (hours + minutes/60) * (2*np.pi/12)
minute_angle = np.pi/2 - (minutes + seconds/60) * (2*np.pi/60)
second_angle = np.pi/2 - seconds * (2*np.pi/60)

# Vẽ các kim
ax_static.plot([0, 0.5*np.cos(hour_angle)], [0, 0.5*np.sin(hour_angle)], 
              linewidth=8, color='darkred', alpha=0.9, zorder=4)
ax_static.plot([0, 0.7*np.cos(minute_angle)], [0, 0.7*np.sin(minute_angle)], 
              linewidth=5, color='darkblue', alpha=0.8, zorder=4)
ax_static.plot([0, 0.8*np.cos(second_angle)], [0, 0.8*np.sin(second_angle)], 
              linewidth=2, color='green', alpha=0.7, zorder=5)

# Tâm đồng hồ
ax_static.plot(0, 0, 'o', markersize=15, color='gold', alpha=0.9, zorder=6)

# Thông tin
time_str = now.strftime('%H:%M:%S')
date_str = now.strftime('%d/%m/%Y')
ax_static.text(0, -1.3, f"THỜI GIAN: {time_str}", 
              ha='center', fontsize=16, fontweight='bold', color='darkblue')
ax_static.text(0, -1.45, date_str, ha='center', fontsize=14, color='darkred')

# Tiêu đề
ax_static.set_title("ĐỒNG HỒ LA MÃ - ẢNH TĨNH", 
                   fontsize=22, fontweight='bold', color='darkblue', pad=25)

# Cài đặt
ax_static.set_xlim(-1.5, 1.5)
ax_static.set_ylim(-1.6, 1.2)
ax_static.set_aspect('equal')
ax_static.axis('off')

# Lưu ảnh
output_filename = f"dong_ho_la_ma_{now.strftime('%Y%m%d_%H%M%S')}.png"
plt.tight_layout()
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"✓ Đã lưu hình ảnh đồng hồ vào file: {output_filename}")

# Hiển thị ảnh đã lưu
plt.show()

# ==============================================
# PHẦN 6: HIỂN THỊ TẤT CẢ HÌNH ẢNH BẰNG OPENCV
# ==============================================

print("\n" + "=" * 60)
print("PHẦN 6: HIỂN THỊ TẤT CẢ HÌNH ẢNH")
print("=" * 60)

# Tạo một bảng tổng hợp tất cả hình ảnh
print("Đang tạo bảng tổng hợp hình ảnh...")

# Đọc ảnh đã lưu bằng OpenCV
saved_img = cv2.imread(output_filename)
if saved_img is not None:
    # Resize ảnh
    saved_img = cv2.resize(saved_img, (600, 600))
    
    # Tạo ảnh tổng hợp
    collage_height = 800
    collage_width = 1200
    collage = np.ones((collage_height, collage_width, 3), dtype=np.uint8) * 255
    
    # Thêm tiêu đề
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(collage, "TONG HOP HINH ANH DONG HO", (50, 50), 
                font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(collage, f"Thoi gian: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}", 
                (50, 100), font, 0.7, (0, 100, 0), 2, cv2.LINE_AA)
    
    # Thêm các ảnh vào bảng
    # Ảnh gradient xám
    gray_resized = cv2.resize(img_gray, (300, 300))
    gray_colored = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    collage[150:450, 50:350] = gray_colored
    cv2.putText(collage, "1. Gradient Xam", (100, 140), 
                font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Ảnh gradient màu
    color_resized = cv2.resize(img_color_bgr, (300, 300))
    collage[150:450, 400:700] = color_resized
    cv2.putText(collage, "2. Gradient Mau", (450, 140), 
                font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Ảnh ngẫu nhiên
    random_resized = cv2.resize(img_random, (300, 300))
    collage[150:450, 750:1050] = random_resized
    cv2.putText(collage, "3. Anh Ngau Nhien", (800, 140), 
                font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Ảnh đồng hồ đã lưu
    collage[500:800, 200:800] = cv2.resize(saved_img, (600, 300))
    cv2.putText(collage, "4. Dong Ho La Ma", (450, 490), 
                font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Hiển thị
    cv2.imshow('TONG HOP HINH ANH DONG HO', collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Lưu bảng tổng hợp
    cv2.imwrite("tong_hop_dong_ho.png", collage)
    print("✓ Đã lưu bảng tổng hợp vào file: tong_hop_dong_ho.png")
else:
    print("✗ Không thể đọc ảnh đã lưu")

print("\n" + "=" * 60)
print("CHUONG TRINH KET THUC")
print("=" * 60)
print("Cam on ban da su dung chuong trinh!")
print("Cac file da duoc tao:")
print("1. Hinh anh dong ho: dong_ho_la_ma_*.png")
print("2. Bang tong hop: tong_hop_dong_ho.png")
print("=" * 60)