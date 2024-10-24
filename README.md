import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, filedialog, Frame
from PIL import Image, ImageTk

# Hiển thị ảnh với giao diện đẹp mắt sử dụng Matplotlib
def show_image(title, image):
    """Hiển thị ảnh với tiêu đề (Matplotlib)."""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')  # Ẩn trục tọa độ để tập trung vào ảnh
    plt.show()

# Chuyển ảnh sang âm tính
def negative_image(image):
    """Chuyển ảnh sang âm tính."""
    return 255 - image

# Tăng độ tương phản sử dụng phương pháp CLAHE
def contrast_enhancement(image):
    """Tăng độ tương phản của ảnh bằng phương pháp CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Chuyển sang không gian màu LAB
    l, a, b = cv2.split(lab)  # Tách kênh màu

    # Tạo đối tượng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)  # Áp dụng CLAHE lên kênh sáng (L)

    # Gộp lại các kênh sau khi tăng độ tương phản
    lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # Chuyển lại sang BGR
    return enhanced_image

# Biến đổi logarit cho ảnh
def log_transform(image):
    """Thực hiện phép biến đổi log trên ảnh."""
    c = 255 / np.log(1 + np.max(image))  # Hằng số cho biến đổi log
    log_image = c * np.log(1 + image)  # Áp dụng công thức biến đổi log
    return np.array(log_image, dtype=np.uint8)

# Cân bằng Histogram cho ảnh xám
def histogram_equalization(image):
    """Cân bằng Histogram cho ảnh."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
    equalized = cv2.equalizeHist(gray)  # Áp dụng cân bằng Histogram
    return equalized

# Hàm để chọn ảnh từ hệ thống
def load_image():
    """Mở hộp thoại chọn ảnh và hiển thị ảnh đã chọn."""
    global img, img_display
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        img = cv2.imread(file_path)  # Đọc ảnh từ file
        if img is not None:
            display_image(img)  # Hiển thị ảnh lên giao diện
        else:
            print("Không thể đọc ảnh từ đường dẫn đã cung cấp.")
    else:
        print("Bạn chưa chọn file ảnh.")

# Hiển thị ảnh trong giao diện Tkinter
def display_image(image):
    """Hiển thị ảnh trên giao diện Tkinter."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)  # Chuyển từ OpenCV sang PIL
    img_tk = ImageTk.PhotoImage(image=img_pil)  # Chuyển sang định dạng hiển thị Tkinter

    img_display.config(image=img_tk)  # Cập nhật hình ảnh trên giao diện
    img_display.image = img_tk  # Để tránh bị xóa bởi garbage collection

# Hàm áp dụng một trong các hiệu ứng và hiển thị
def apply_effect(effect_function):
    """Áp dụng hiệu ứng lên ảnh đã chọn và hiển thị kết quả."""
    if img is not None:
        result = effect_function(img)  # Gọi hàm xử lý ảnh
        display_image(result)  # Hiển thị ảnh kết quả

# Tạo giao diện người dùng bằng Tkinter
def create_gui():
    """Tạo giao diện người dùng với các nút chức năng."""
    global img_display

    # Tạo cửa sổ chính
    root = Tk()
    root.title("Công cụ xử lý ảnh cao cấp")
    root.geometry("900x600")
    root.configure(bg="#f0f0f0")  # Màu nền cho giao diện

    # Tạo khung chứa các nút chức năng
    frame = Frame(root, bg="#e0e0e0", padx=20, pady=20)
    frame.pack(side="right", fill="y")

    # Nút chọn ảnh
    Button(frame, text="Chọn ảnh", command=load_image, bg="#007bff", fg="white", padx=10, pady=5).pack(pady=10)

    # Các nút chức năng xử lý ảnh
    Button(frame, text="Ảnh âm tính", command=lambda: apply_effect(negative_image), bg="#28a745", fg="white", padx=10, pady=5).pack(pady=10)
    Button(frame, text="Tăng độ tương phản", command=lambda: apply_effect(contrast_enhancement), bg="#ffc107", fg="white", padx=10, pady=5).pack(pady=10)
    Button(frame, text="Biến đổi log", command=lambda: apply_effect(log_transform), bg="#17a2b8", fg="white", padx=10, pady=5).pack(pady=10)
    Button(frame, text="Cân bằng Histogram", command=lambda: apply_effect(histogram_equalization), bg="#dc3545", fg="white", padx=10, pady=5).pack(pady=10)

    # Nơi hiển thị ảnh
    img_display = Label(root, bg="#d3d3d3", width=600, height=400)
    img_display.pack(side="left", padx=20, pady=20)

    # Khởi chạy giao diện
    root.mainloop()

# Chạy ứng dụng
if __name__ == "__main__":
    img = None  # Biến toàn cục chứa ảnh đã chọn
    create_gui()  # Tạo giao diện và xử lý ảnh
