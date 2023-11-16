import shutil
import cv2
from ultralytics import YOLO
import os
from glob import glob
import pandas as pd
import numpy as np

# Загрузка модели YOLOv8
model = YOLO(r"D:\cabbageDopBig_yolov8\best.pt")  # Путь к файлу весов модели



def compute_fill_percentage(polygon, image_dims):
    contour = cv2.contourArea(cv2.convexHull(np.array(polygon, dtype='int32')))
    image_area = image_dims[0] * image_dims[1]
    fill_percentage = (contour / image_area) * 100
    return fill_percentage

def classify_distance(fill_percentage, close_threshold, far_threshold):
    if fill_percentage >= close_threshold:
        return "close"
    elif fill_percentage <= far_threshold:
        return "far"
    else:
        return "medium"

def aggregate_results(classifications):
    if not classifications:
        return "No objects detected"
    return max(set(classifications), key=classifications.count)

def process_image(image_path, close_threshold, far_threshold, show_image=False):
    image = cv2.imread(image_path)
    image_dims = image.shape[:2]
    detections = model(image)[0]
    classifications = []

    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < 0.4:  # Порог уверенности
            continue
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        polygon = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        fill_percentage = compute_fill_percentage(polygon, image_dims)
        classification = classify_distance(fill_percentage, close_threshold, far_threshold)
        classifications.append(classification)
        cv2.polylines(image, [np.array(polygon, dtype='int32')], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, classification, polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    final_classification = aggregate_results(classifications)
    if show_image:
        cv2.putText(image, f'Final Classification: {final_classification}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_classification

def main():
    image_folder = r"D:\Learning\UII\DataSets\DataSet_2.1_Train_Valid\train\images"
    image_paths = glob(os.path.join(image_folder, '*.jpg')) + glob(os.path.join(image_folder, '*.png'))
    close_threshold = 11
    far_threshold = 4
    report_data = []

    for image_path in image_paths:
        final_classification = process_image(image_path, close_threshold, far_threshold, show_image=False)
        # Копирование изображений в соответствующие папки
        destination_folder = os.path.join("path_to_destination_folder", final_classification)
        os.makedirs(destination_folder, exist_ok=True)
        shutil.copy(image_path, destination_folder)

        report_data.append({'Image Path': image_path, 'Classification': final_classification})

    report_df = pd.DataFrame(report_data)
    report_df.to_csv('classification_report.csv', index=False)

if __name__ == '__main__':
    main()



