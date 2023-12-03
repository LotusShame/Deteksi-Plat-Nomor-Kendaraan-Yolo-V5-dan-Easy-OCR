import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import easyocr

# Fungsi untuk memeriksa resolusi gambar
def check_image_resolution(image, min_width=12, min_height=72):
    width, height = image.size
    return width >= min_width and height >= min_height

# Fungsi untuk deteksi helm menggunakan YOLOv5
def detect_helmet(image):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best_helm.pt')
    result = model(image)
    return result

# Fungsi untuk deteksi plat nomor menggunakan YOLOv5
def detect_license_plate(image):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best_Plate.pt')
    result = model(image)
    return result

# Fungsi untuk melakukan OCR pada plat nomor
def recognize_license_plate(plate_image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(plate_image)

    if results:
        plate_text = results[0][1]
    else:
        plate_text = "Plat nomor tidak ada"

    return plate_text

# Fungsi untuk mencari indeks plat nomor terdekat dari pengendara "tanpa_helm"
def find_nearest_license_plate(helmet_result, without_helmet_index, license_plate_results):
    without_helmet_x1, without_helmet_y1, without_helmet_x2, without_helmet_y2, _, _ = helmet_result.xyxy[0][without_helmet_index].cpu().numpy()
    without_helmet_x_center = (without_helmet_x1 + without_helmet_x2) / 2
    without_helmet_y_center = (without_helmet_y1 + without_helmet_y2) / 2
    distances = []

    for i, plate_result in enumerate(license_plate_results.pred[0]):
        x1, y1, x2, y2, _, _ = plate_result.cpu().numpy()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        distance = np.sqrt((x_center - without_helmet_x_center) ** 2 + (y_center - without_helmet_y_center) ** 2)
        distances.append((i, distance))

    distances.sort(key=lambda x: x[1])

    if distances:
        return distances[0][0]
    else:
        return None

# Fungsi untuk merotasi gambar
def rotate_to_horizontal(image, angle_threshold=10):
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Deteksi tepi pada gambar
    edges = cv2.Canny(gray, 100, 200)

    # Temukan kontur plat nomor
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Pilih kontur plat nomor dengan area terbesar
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            # Dapatkan sudut rotasi
            rect = cv2.minAreaRect(max_contour)
            angle = rect[-1]

            if angle > 45:
                angle += 270

            if abs(angle) > angle_threshold:
                # Rotasi gambar untuk membuatnya horizontal
                center = tuple(np.array(image.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return image, angle

# Streamlit App
st.title("🏍 Pengenalan Plat Nomor Kendaraan Bagi Yang Tidak Menggunakan Helm")

uploaded_image = st.file_uploader("Upload gambar pengendara sepeda motor (resolusi minimal 1920x1080)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    if check_image_resolution(image):
        image_np = np.array(image)

        st.image(image_np, caption="Gambar pengendara", use_column_width=True)

        # Deteksi helm
        helmet_result = detect_helmet(image_np)
        st.image(helmet_result.render()[0], caption="Deteksi Helm", use_column_width=True)

        without_helmet_label = "tanpa_helm"

        # Find the indices of detected "tanpa_helm"
        without_helmet_indices = [i for i in range(len(helmet_result.pred[0])) if helmet_result.names[int(helmet_result.pred[0][i][-1])] == without_helmet_label]

        if without_helmet_indices:
            st.write("Pengendara tidak menggunakan helm, melanjutkan dengan deteksi plat nomor.")
            license_plate_results = detect_license_plate(image_np)
            for without_helmet_index in without_helmet_indices:
                nearest_license_plate_index = find_nearest_license_plate(helmet_result, without_helmet_index, license_plate_results)

                if nearest_license_plate_index is not None:
                    x1, y1, x2, y2, _, _ = license_plate_results.pred[0][nearest_license_plate_index].cpu().numpy()
                    plate_image = image_np[int(y1):int(y2), int(x1):int(x2)]

                    st.image(plate_image, caption="Plat Nomor", use_column_width=True)

                    # Rotasi plat nomor ke posisi horizontal
                    st.write("### Rotasi Plat Nomor ke Posisi Horizontal")
                    plate_image, angle = rotate_to_horizontal(plate_image)
                    st.image(plate_image, caption="Plat Nomor Rotated", use_column_width=True)

                    plate_text = recognize_license_plate(plate_image)
                    if plate_text == "Plat nomor tidak ada":
                        st.write("Plat nomor tidak ditemukan!")
                    else:
                        st.write("Plat Nomor:", plate_text)
                else:
                    st.write("Plat nomor tidak ditemukan.")
        else:
            st.write("Tidak ada pengendara yang terdeteksi tidak menggunakan helm.")
    else:
        st.write("Resolusi gambar tidak mencukupi (minimal 1920x1080). Silakan unggah gambar dengan resolusi yang sesuai.")
else:
    st.write("Silakan upload gambar pengendara sepeda motor.")
