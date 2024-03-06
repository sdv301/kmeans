import cv2
import numpy as np
from sklearn.cluster import KMeans

def process_frame(frame, n_clusters):
    # Преобразование изображения в формат RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Переформатирование изображения в одномерный массив пикселей
    pixels = frame.reshape((-1, 3))

    # Применение K-Means для кластеризации цветов
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)

    # Получение цветов центров кластеров
    centers = kmeans.cluster_centers_.astype(int)

    # Замена цветов пикселей
    for i in range(len(pixels)):
        pixels[i] = centers[kmeans.labels_[i]]

    # Возвращение изображения к исходным размерам
    processed_frame = pixels.reshape(frame.shape)
    return processed_frame

def process_video(input_path, output_path, n_clusters):
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for _ in range(frame_count):
        ret, frame = cap.read()
        if ret:
            processed_frame = process_frame(frame, n_clusters)
            out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_file = "one.mp4"
    output_file = "output_one_video.mp4"
    n_clusters = 16  # Число базовых оттенков

    process_video(input_file, output_file, n_clusters)
