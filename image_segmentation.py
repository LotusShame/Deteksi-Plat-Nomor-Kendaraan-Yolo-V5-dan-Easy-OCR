import os
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

# Fungsi erosi
def erode_image(image, kernel_size=3):
    erode_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, erode_kernel, iterations=1)
    return eroded_image

# Fungsi dilasi
def dilate_image(image, kernel_size=3):
    dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(image, dilate_kernel, iterations=1)
    return dilated_image

# Fungsi untuk segmentasi karakter setelah erosi
def segment_characters(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    isolated_characters = image.copy()
    segmented_characters = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(isolated_characters, (x, y), (x + w, y + h), (0, 255, 0), 2)
        character = image[y:y + h, x:x + w]

        # Resize karakter ke ukuran yang seragam
        character = cv2.resize(character, (50, 50))  # Sesuaikan ukuran sesuai kebutuhan

        # Simpan karakter sebagai file gambar terpisah
        character_filepath = f"character_{i}.png"
        cv2.imwrite(character_filepath, character)

        # Tambahkan karakter tersegmentasi ke dalam daftar
        segmented_characters.append(character_filepath)

    # Tampilkan gambar yang diisolasi
    st.image(isolated_characters, caption="Gambar yang diisolasi", use_column_width=True)

    return segmented_characters

# Fungsi untuk menggabungkan karakter menjadi satu teks
def combine_characters(characters):
    combined_text = ""
    reader = easyocr.Reader(['en'])

    for character_filepath in characters:
        # Tambahkan try-except block untuk menangani kesalahan FileNotFoundError
        try:
            character_filepath = os.path.abspath(character_filepath)

            results = reader.readtext(character_filepath)

            if results:
                combined_text += results[0][1]
        except FileNotFoundError:
            st.warning(f"File not found: {character_filepath}")

    return combined_text


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
                    rotated_plate, angle = rotate_to_horizontal(plate_image)

                    st.image(rotated_plate, caption="Plat Nomor setelah dirotasi", use_column_width=True)

                    # Grayscale
                    gray_plate = cv2.cvtColor(rotated_plate, cv2.COLOR_RGB2GRAY)
                    st.image(gray_plate, caption="Plat Nomor Grayscale", use_column_width=True)

                    # Proses Blur
                    blur_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)
                    st.image(blur_plate, caption="Plat Nomor setelah Blur", use_column_width=True)

                    # Konversi ke citra biner menggunakan threshold
                    _, binary_plate = cv2.threshold(blur_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    st.image(binary_plate, caption="Plat Nomor setelah Konversi ke Biner", use_column_width=True)

                    # Tambahkan slider untuk mengatur nilai erosi dan dilasi
                    erode_kernel_size = st.slider("Ukuran Kernel Erosi", min_value=1, max_value=10, value=3)

                    # Erosi pada plat nomor
                    eroded_plate = erode_image(binary_plate, erode_kernel_size)
                    st.image(eroded_plate, caption="Plat Nomor setelah Erosi", use_column_width=True)

                    # dilate_kernel_size = st.slider("Ukuran Kernel Dilasi", min_value=1, max_value=10, value=3)

                    # # Dilasi pada plat nomor
                    # dilated_plate = dilate_image(binary_plate, dilate_kernel_size)
                    # st.image(dilated_plate, caption="Plat Nomor setelah Dilasi", use_column_width=True)

                    # Segmentasi karakter setelah erosi
                    segmented_characters = segment_characters(eroded_plate)

                    # Tampilkan setiap karakter tersegmentasi
                    for character_filepath in segmented_characters:
                        st.image(character_filepath, caption="Karakter Tersegmentasi", use_column_width=True)

                    # Gabungkan karakter menjadi satu teks
                    combined_text = combine_characters(segmented_characters)
                    st.write("Teks gabungan dari karakter:", combined_text)
                    

                    plate_text = recognize_license_plate(combined_text)
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
