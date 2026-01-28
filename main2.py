import matplotlib.pyplot as plt
import numpy as np
import cv2
from datetime import datetime
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PHẦN 1: TẠO HÌNH TRÁI ĐẤT 3D
# ==============================================

print("=" * 60)
print("TRÁI ĐẤT 3D XOAY VÒNG - ĐỒNG HỒ LA MÃ")
print("=" * 60)

def create_earth_3d(rotation_angle=0):
    """Tạo hình ảnh Trái Đất 3D"""
    
    # Tạo dữ liệu cho hình cầu
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 25)
    
    phi, theta = np.meshgrid(phi, theta)
    
    # Tọa độ hình cầu
    r = 1.0
    x = r * np.sin(theta) * np.cos(phi + rotation_angle)
    y = r * np.sin(theta) * np.sin(phi + rotation_angle)
    z = r * np.cos(theta)
    
    # Tạo texture cho Trái Đất
    colors = np.zeros((*theta.shape, 3))
    
    # Màu xanh nước biển
    colors[:, :, 0] = 0.2  # R
    colors[:, :, 1] = 0.4  # G
    colors[:, :, 2] = 0.8  # B
    
    # Thêm lục địa (màu xanh lá)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            # Tạo hình dạng lục địa
            lon = phi[i, j]
            lat = theta[i, j] - np.pi/2
            
            # Châu Á
            if (lon > 1.5 and lon < 3.0) and (lat > 0.2 and lat < 1.0):
                colors[i, j] = [0.3, 0.7, 0.3]
            
            # Châu Phi
            if (lon > 0 and lon < 1.5) and (lat > -0.5 and lat < 0.5):
                colors[i, j] = [0.3, 0.7, 0.3]
            
            # Châu Mỹ
            if (lon > 4.0 or lon < 0.5) and (lat > -0.8 and lat < 0.8):
                colors[i, j] = [0.3, 0.7, 0.3]
            
            # Nam Cực (màu trắng)
            if lat < -1.2:
                colors[i, j] = [0.9, 0.9, 0.9]
            
            # Bắc Cực (màu trắng)
            if lat > 1.2:
                colors[i, j] = [0.9, 0.9, 0.9]
            
            # Thêm mây (màu trắng loang lổ)
            if np.random.random() < 0.2:
                cloud_intensity = np.random.random() * 0.3 + 0.7
                colors[i, j] = [cloud_intensity, cloud_intensity, cloud_intensity]
    
    return x, y, z, colors

# Tạo hình ảnh Trái Đất đầu tiên
x, y, z, colors = create_earth_3d()

# Hiển thị Trái Đất 3D
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Vẽ Trái Đất
earth_surface = ax_3d.plot_surface(x, y, z, facecolors=colors, 
                                  rstride=1, cstride=1, alpha=0.9)

# Cài đặt 3D view
ax_3d.view_init(elev=20, azim=45)
ax_3d.set_xlim([-1.5, 1.5])
ax_3d.set_ylim([-1.5, 1.5])
ax_3d.set_zlim([-1.5, 1.5])
ax_3d.axis('off')
ax_3d.set_title('TRÁI ĐẤT 3D', fontsize=16, fontweight='bold')

# Thêm ánh sáng
ax_3d.set_facecolor('black')
fig_3d.patch.set_facecolor('black')

plt.tight_layout()
plt.show()

# ==============================================
# PHẦN 2: HÀM TẠO HÌNH 2D TỪ TRÁI ĐẤT 3D
# ==============================================

def create_earth_2d(rotation_angle=0, size=(400, 400)):
    """Tạo hình 2D của Trái Đất từ góc nhìn cố định"""
    height, width = size
    
    # Tạo canvas
    earth_2d = np.zeros((height, width, 3), dtype=np.float32)
    
    # Tâm ảnh
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 2 - 10
    
    # Tạo hiệu ứng Trái Đất
    for y in range(height):
        for x in range(width):
            # Tính khoảng cách đến tâm
            dx = x - center_x
            dy = y - center_y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist <= radius:
                # Tính góc để tạo texture
                angle = np.arctan2(dy, dx) + rotation_angle
                norm_dist = dist / radius
                
                # Tọa độ trên bề mặt hình cầu
                phi = angle
                theta = np.pi * (0.5 - norm_dist * 0.8)
                
                # Màu cơ bản (xanh nước biển)
                r, g, b = 0.2, 0.4, 0.8
                
                # Thêm lục địa dựa trên góc
                # Châu Á
                if (phi > 1.5 and phi < 3.0) and (theta > 0.2 and theta < 1.0):
                    r, g, b = 0.3, 0.7, 0.3
                
                # Châu Phi
                if (phi > 0 and phi < 1.5) and (theta > -0.5 and theta < 0.5):
                    r, g, b = 0.3, 0.7, 0.3
                
                # Châu Mỹ
                if (phi > 4.0 or phi < 0.5) and (theta > -0.8 and theta < 0.8):
                    r, g, b = 0.3, 0.7, 0.3
                
                # Vùng cực (trắng)
                if abs(theta) > 1.2:
                    intensity = 0.9
                    r, g, b = intensity, intensity, intensity
                
                # Hiệu ứng chiếu sáng
                light = 0.7 + 0.3 * np.sin(phi * 2)
                r *= light
                g *= light
                b *= light
                
                # Áp dụng màu
                earth_2d[y, x] = [r, g, b]
    
    return earth_2d

# Tạo hình Trái Đất 2D
earth_2d = create_earth_2d()

# Hiển thị Trái Đất 2D
plt.figure(figsize=(6, 6))
plt.imshow(earth_2d)
plt.title('TRÁI ĐẤT 2D', fontsize=14)
plt.axis('off')
plt.show()

# ==============================================
# PHẦN 3: ĐỒNG HỒ LA MÃ VỚI TRÁI ĐẤT XOAY
# ==============================================

print("\n" + "=" * 60)
print("KHỞI TẠO ĐỒNG HỒ LA MÃ VỚI TRÁI ĐẤT 3D")
print("Nhấn Ctrl+C để dừng chương trình")
print("=" * 60)

# Các số La Mã
roman_numbers = [
    "XII", "I", "II", "III", "IV", "V",
    "VI", "VII", "VIII", "IX", "X", "XI"
]

# Tạo figure và axis
plt.ion()
fig, (ax_clock, ax_earth) = plt.subplots(1, 2, figsize=(16, 8))
fig.canvas.manager.set_window_title('Đồng hồ La Mã với Trái Đất 3D')

# Biến xoay
earth_rotation = 0

try:
    while True:
        # ========== PHẦN TRÁI: ĐỒNG HỒ LA MÃ ==========
        ax_clock.clear()
        
        # Đặt nền đen cho đồng hồ
        ax_clock.set_facecolor('black')
        
        # Vẽ vòng tròn đồng hồ
        circle_outer = plt.Circle((0, 0), 0.95, fill=False, 
                                 color='cyan', linewidth=4, alpha=0.8)
        circle_inner = plt.Circle((0, 0), 0.85, fill=False, 
                                 color='white', linewidth=2, alpha=0.6)
        ax_clock.add_artist(circle_outer)
        ax_clock.add_artist(circle_inner)
        
        # Vẽ các số La Mã
        for i, roman in enumerate(roman_numbers):
            angle = np.pi/2 - i * (2 * np.pi / 12)
            x = 0.75 * np.cos(angle)
            y = 0.75 * np.sin(angle)
            
            # Màu sắc theo vị trí
            color = plt.cm.hsv(i / 12)
            
            ax_clock.text(
                x, y, roman,
                ha='center', va='center',
                fontsize=18, fontweight='bold',
                color=color,
                fontfamily='serif',
                bbox=dict(boxstyle='circle,pad=0.3', 
                         facecolor='black', 
                         edgecolor=color, alpha=0.8)
            )
        
        # Lấy thời gian hiện tại
        now = datetime.now()
        hour = now.hour % 12
        minute = now.minute
        second = now.second
        millisecond = now.microsecond // 1000
        
        # Tính góc cho các kim
        hour_angle = np.pi/2 - (hour + minute/60) * (2*np.pi/12)
        minute_angle = np.pi/2 - (minute + second/60) * (2*np.pi/60)
        second_angle = np.pi/2 - (second + millisecond/1000) * (2*np.pi/60)
        
        # Vẽ kim giờ
        ax_clock.plot([0, 0.5*np.cos(hour_angle)], 
                     [0, 0.5*np.sin(hour_angle)], 
                     linewidth=8, color='yellow', alpha=0.9,
                     solid_capstyle='round')
        
        # Vẽ kim phút
        ax_clock.plot([0, 0.7*np.cos(minute_angle)], 
                     [0, 0.7*np.sin(minute_angle)], 
                     linewidth=5, color='magenta', alpha=0.8,
                     solid_capstyle='round')
        
        # Vẽ kim giây
        ax_clock.plot([0, 0.8*np.cos(second_angle)], 
                     [0, 0.8*np.sin(second_angle)], 
                     linewidth=2, color='cyan', alpha=0.7,
                     solid_capstyle='round')
        
        # Vẽ tâm đồng hồ
        ax_clock.plot(0, 0, 'o', markersize=15, 
                     color='white', alpha=0.9, zorder=10)
        ax_clock.plot(0, 0, 'o', markersize=8, 
                     color='red', alpha=0.9, zorder=11)
        
        # Vẽ các vạch phút
        for minute_mark in range(0, 60):
            angle = np.pi/2 - minute_mark * (2*np.pi/60)
            length = 0.9 if minute_mark % 5 == 0 else 0.93
            width = 3 if minute_mark % 5 == 0 else 1
            color = 'yellow' if minute_mark % 15 == 0 else 'white'
            
            x_start = 0.85 * np.cos(angle)
            y_start = 0.85 * np.sin(angle)
            x_end = length * np.cos(angle)
            y_end = length * np.sin(angle)
            
            ax_clock.plot([x_start, x_end], [y_start, y_end],
                         linewidth=width, color=color, alpha=0.6)
        
        # Hiển thị thời gian
        time_str = now.strftime('%H:%M:%S')
        date_str = now.strftime('%d/%m/%Y')
        
        ax_clock.text(0, -1.15, f"⏰ {time_str}", 
                     ha='center', fontsize=20, fontweight='bold',
                     color='yellow',
                     bbox=dict(boxstyle='round,pad=0.5', 
                              facecolor='darkblue', alpha=0.8))
        
        ax_clock.text(0, -1.3, f"📅 {date_str}", 
                     ha='center', fontsize=14,
                     color='white', style='italic')
        
        # Cài đặt đồng hồ
        ax_clock.set_xlim(-1.4, 1.4)
        ax_clock.set_ylim(-1.4, 1.4)
        ax_clock.set_aspect('equal')
        ax_clock.axis('off')
        ax_clock.set_title('ĐỒNG HỒ LA MÃ', fontsize=20, 
                          fontweight='bold', color='cyan', pad=20)
        
        # ========== PHẦN PHẢI: TRÁI ĐẤT 3D XOAY ==========
        ax_earth.clear()
        
        # Tạo dữ liệu Trái Đất mới với góc xoay mới
        earth_rotation += 0.02  # Tốc độ xoay
        x, y, z, colors = create_earth_3d(earth_rotation)
        
        # Vẽ Trái Đất 3D
        ax_earth = fig.add_subplot(122, projection='3d')
        earth_surface = ax_earth.plot_surface(x, y, z, facecolors=colors, 
                                            rstride=1, cstride=1, 
                                            alpha=0.95, antialiased=True)
        
        # Thêm quỹ đạo
        theta_orbit = np.linspace(0, 2*np.pi, 100)
        orbit_radius = 1.5
        x_orbit = orbit_radius * np.cos(theta_orbit)
        y_orbit = orbit_radius * np.sin(theta_orbit)
        z_orbit = np.zeros_like(x_orbit)
        
        ax_earth.plot(x_orbit, y_orbit, z_orbit, '--', 
                     color='yellow', alpha=0.3, linewidth=1)
        
        # Thêm các ngôi sao
        n_stars = 100
        stars_x = np.random.uniform(-3, 3, n_stars)
        stars_y = np.random.uniform(-3, 3, n_stars)
        stars_z = np.random.uniform(-3, 3, n_stars)
        stars_size = np.random.uniform(10, 50, n_stars)
        
        ax_earth.scatter(stars_x, stars_y, stars_z, 
                        s=stars_size, color='white', alpha=0.6)
        
        # Cài đặt view 3D
        ax_earth.view_init(elev=20, azim=earth_rotation * 20)
        ax_earth.set_xlim([-2, 2])
        ax_earth.set_ylim([-2, 2])
        ax_earth.set_zlim([-2, 2])
        ax_earth.axis('off')
        
        # Đặt nền đen cho không gian
        ax_earth.set_facecolor('black')
        ax_earth.xaxis.pane.fill = False
        ax_earth.yaxis.pane.fill = False
        ax_earth.zaxis.pane.fill = False
        
        # Thêm ánh sáng mặt trời
        ax_earth.scatter([3], [0], [0], s=500, 
                        color='yellow', alpha=0.7, marker='o')
        
        # Tiêu đề
        ax_earth.set_title('TRÁI ĐẤT 3D XOAY', fontsize=20, 
                          fontweight='bold', color='yellow', pad=20)
        
        # Thêm thông tin góc xoay
        ax_earth.text2D(0.05, 0.95, f"Góc xoay: {earth_rotation:.2f} rad", 
                       transform=ax_earth.transAxes,
                       color='white', fontsize=10)
        
        # ========== CẬP NHẬT ==========
        plt.suptitle('ĐỒNG HỒ LA MÃ VỚI TRÁI ĐẤT 3D XOAY VÒNG', 
                    fontsize=24, fontweight='bold', color='white')
        
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        
        # Hiển thị FPS
        current_time = time.time()
        if 'last_time' not in locals():
            last_time = current_time
            fps = 0
        else:
            fps = 1.0 / (current_time - last_time)
            last_time = current_time
        
        fig.text(0.02, 0.02, f"FPS: {fps:.1f}", 
                fontsize=10, color='white')
        
        # Cập nhật đồ họa
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Delay
        plt.pause(0.03)  # ~30 FPS
        
except KeyboardInterrupt:
    print("\n" + "=" * 60)
    print("ĐÃ DỪNG CHƯƠNG TRÌNH")
    print("=" * 60)
    
except Exception as e:
    print(f"\nCó lỗi xảy ra: {e}")
    
finally:
    plt.ioff()
    plt.close('all')
    cv2.destroyAllWindows()

# ==============================================
# PHẦN 4: TẠO HOẠT HÌNH TRÁI ĐẤT XOAY
# ==============================================

print("\n" + "=" * 60)
print("TẠO HOẠT HÌNH TRÁI ĐẤT XOAY")
print("=" * 60)

# Tạo figure mới cho animation
fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

# Hàm cập nhật animation
def update_anim(frame):
    ax_anim.clear()
    
    # Tạo Trái Đất với góc xoay mới
    rotation = frame * 0.1
    x, y, z, colors = create_earth_3d(rotation)
    
    # Vẽ Trái Đất
    ax_anim.plot_surface(x, y, z, facecolors=colors, 
                        rstride=1, cstride=1, alpha=0.9)
    
    # Cài đặt view
    ax_anim.view_init(elev=25, azim=frame * 2)
    ax_anim.set_xlim([-1.5, 1.5])
    ax_anim.set_ylim([-1.5, 1.5])
    ax_anim.set_zlim([-1.5, 1.5])
    ax_anim.axis('off')
    
    ax_anim.set_facecolor('black')
    fig_anim.patch.set_facecolor('black')
    
    ax_anim.set_title(f'TRÁI ĐẤT 3D XOAY - Frame {frame}', 
                     fontsize=16, fontweight='bold', color='white', pad=20)
    
    return ax_anim,

# Tạo animation (ngắn gọn)
print("Đang tạo animation... (có thể mất vài giây)")

try:
    anim = FuncAnimation(fig_anim, update_anim, frames=36, 
                        interval=50, blit=False, repeat=True)
    
    # Lưu animation dưới dạng GIF
    anim.save('trai_dat_xoay.gif', writer='pillow', fps=20)
    print("✓ Đã lưu animation: trai_dat_xoay.gif")
    
    # Hiển thị
    plt.show()
    
except Exception as e:
    print(f"Không thể tạo animation: {e}")

# ==============================================
# PHẦN 5: LƯU HÌNH ẢNH
# ==============================================

print("\n" + "=" * 60)
print("LƯU HÌNH ẢNH")
print("=" * 60)

# Tạo và lưu hình Trái Đất 2D
earth_2d_final = create_earth_2d(earth_rotation, size=(600, 600))
earth_2d_uint8 = (earth_2d_final * 255).astype(np.uint8)
earth_2d_bgr = cv2.cvtColor(earth_2d_uint8, cv2.COLOR_RGB2BGR)
cv2.imwrite('trai_dat_2d.png', earth_2d_bgr)
print("✓ Đã lưu hình Trái Đất 2D: trai_dat_2d.png")

# Tạo hình đồng hồ tĩnh
fig_static, ax_static = plt.subplots(figsize=(8, 8))
ax_static.set_facecolor('black')

# Vẽ đồng hồ tĩnh
circle = plt.Circle((0, 0), 0.9, fill=False, color='cyan', linewidth=4, alpha=0.8)
ax_static.add_artist(circle)

# Vẽ số La Mã
for i, roman in enumerate(roman_numbers):
    angle = np.pi/2 - i * (2 * np.pi / 12)
    x = 0.75 * np.cos(angle)
    y = 0.75 * np.sin(angle)
    ax_static.text(x, y, roman,
                  ha='center', va='center',
                  fontsize=16, fontweight='bold',
                  color='white')

# Lấy thời gian hiện tại
now = datetime.now()
hour = now.hour % 12
minute = now.minute
second = now.second

# Vẽ kim đồng hồ
hour_angle = np.pi/2 - (hour + minute/60) * (2*np.pi/12)
minute_angle = np.pi/2 - (minute + second/60) * (2*np.pi/60)
second_angle = np.pi/2 - second * (2*np.pi/60)

ax_static.plot([0, 0.5*np.cos(hour_angle)], [0, 0.5*np.sin(hour_angle)], 
              linewidth=6, color='yellow', alpha=0.9)
ax_static.plot([0, 0.7*np.cos(minute_angle)], [0, 0.7*np.sin(minute_angle)], 
              linewidth=4, color='magenta', alpha=0.8)
ax_static.plot([0, 0.8*np.cos(second_angle)], [0, 0.8*np.sin(second_angle)], 
              linewidth=2, color='cyan', alpha=0.7)

ax_static.set_xlim(-1.2, 1.2)
ax_static.set_ylim(-1.2, 1.2)
ax_static.set_aspect('equal')
ax_static.axis('off')
ax_static.set_title(f'ĐỒNG HỒ LA MÃ\n{now.strftime("%H:%M:%S")}', 
                   fontsize=18, fontweight='bold', color='white', pad=20)

plt.tight_layout()
plt.savefig('dong_ho_la_ma_static.png', dpi=150, facecolor='black', 
           bbox_inches='tight')
print("✓ Đã lưu hình đồng hồ: dong_ho_la_ma_static.png")

plt.show()

print("\n" + "=" * 60)
print("CHƯƠNG TRÌNH KẾT THÚC")
print("=" * 60)
print("Các file đã được tạo:")
print("1. trai_dat_2d.png - Hình Trái Đất 2D")
print("2. trai_dat_xoay.gif - Animation Trái Đất xoay")
print("3. dong_ho_la_ma_static.png - Hình đồng hồ tĩnh")
print("=" * 60)