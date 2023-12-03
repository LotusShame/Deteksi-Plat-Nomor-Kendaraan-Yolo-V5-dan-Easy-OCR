import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import easyocr

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

# Fungsi untuk merotasi gambar
def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# Fungsi untuk mencari indeks plat nomor terdekat dari pengendara "tanpa_helm"
def find_nearest_license_plate(helmet_result, without_helmet_index, license_plate_results):
    without_helmet_x1, without_helmet_y1, without_helmet_x2, without_helmet_y2, _, _ = helmet_result.xyxy[0][without_helmet_index].cpu().numpy()
    without_helmet_x_center = (without_helmet_x1 + without_helmet_x2) / 2
    without_helmet_y_center = (without_helmet_y1 + without_helmet_y2) / 2
    distances = []

    if len(license_plate_results.pred[0]) > 0:  # Periksa apakah license_plate_results.pred[0] tidak kosong
        for i, plate_result in enumerate(license_plate_results.pred[0]):
            x1, y1, x2, y2, _, _ = plate_result.cpu().numpy()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            distance = np.sqrt((x_center - without_helmet_x_center) ** 2 + (y_center - without_helmet_y_center) ** 2)
            distances.append((i, distance))

        distances.sort(key=lambda x: x[1])
        return distances[0][0]
    else:
        # Tangani kasus ketika license_plate_results.pred[0] kosong
        st.write("Tidak ada plat nomor yang terdeteksi.")
        return None

# Streamlit App
st.title("🏍 Pengenalan Plat Nomor Kendaraan Bagi Yang Tidak Menggunakan Helm")

uploaded_image = st.file_uploader("Upload gambar pengendara sepeda motor", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    st.image(image_np, caption="Gambar pengendara", use_column_width=True)

    # Deteksi helm
    helmet_result = detect_helmet(image_np)
    st.image(helmet_result.render()[0], caption="Deteksi Helm", use_column_width=True)

    without_helmet_label = "tanpa_helm"

    # Temukan indeks yang terdeteksi "tanpa_helm"
    without_helmet_indices = [i for i in range(len(helmet_result.pred[0])) if helmet_result.names[int(helmet_result.pred[0][i][-1])] == without_helmet_label]

    if without_helmet_indices:
        st.write("Pengendara tidak menggunakan helm, melanjutkan dengan deteksi plat nomor.")
        license_plate_results = detect_license_plate(image_np)
        for without_helmet_index in without_helmet_indices:
            nearest_license_plate_index = find_nearest_license_plate(helmet_result, without_helmet_index, license_plate_results)

            if nearest_license_plate_index is not None:
                x1, y1, x2, y2, _, _ = license_plate_results.pred[0][nearest_license_plate_index].cpu().numpy()
                plate_image = image_np[int(y1):int(y2), int(x1):int(x2)]

                # Transformasi Perspektif
                gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_plate, 50, 150, apertureSize=3)
                lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

                if lines is not None:
                    for line in lines:
                        rho, theta = line[0]
                        if np.degrees(theta) > 45:
                            angle = np.degrees(theta) - 90
                            rotated_plate = rotate_image(plate_image, angle)
                            st.image(rotated_plate, caption="Plat Nomor yang Sudah Terputar", use_column_width=True)

                            plate_text = recognize_license_plate(rotated_plate)
                            st.write("Plat Nomor:", plate_text)
                            break
                else:
                    st.image(plate_image, caption="Plat Nomor", use_column_width=True)
                    plate_text = recognize_license_plate(plate_image)
                    st.write("Plat Nomor:", plate_text)
    else:
        st.write("Tidak ada pengendara yang terdeteksi tidak menggunakan helm.")
else:
    st.write("Silakan upload gambar pengendara sepeda motor.")
