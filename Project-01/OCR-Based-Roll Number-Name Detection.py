import os
import cv2
import numpy as np
import easyocr
from math import atan2, degrees
import csv
import natsort  # Import natsort library for natural sorting of filenames

def calculate_rectangle_dimensions(start, width, height, scale_factor):
    scaled_width = int(width * scale_factor[0])
    scaled_height = int(height * scale_factor[1])
    return (start[0] + scaled_width, start[1] + scaled_height)

def calculate_angle_of_rotation(top_left, top_right):
    delta_x = top_right[0] - top_left[0]
    delta_y = top_right[1] - top_left[1]
    angle_rad = atan2(delta_y, delta_x)
    angle_deg = degrees(angle_rad)
    return angle_deg

def rotate_box(corners, angle, center):
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_corners = []
    for corner in corners:
        rotated_corner = np.dot(rotation_matrix, np.array([corner[0], corner[1], 1]))
        rotated_corners.append((int(rotated_corner[0]), int(rotated_corner[1])))
    return rotated_corners

def annotate_image(image, position, box_dimensions, scaling, box_color=(0, 255, 0), rotation_angle=0):
    rect_end = calculate_rectangle_dimensions(position, box_dimensions[0], box_dimensions[1], scaling)
    start_point = tuple(map(int, position))
    end_point = tuple(map(int, rect_end))

    # Calculate the four corners of the rectangle
    top_left = start_point
    top_right = (end_point[0], start_point[1])
    bottom_left = (start_point[0], end_point[1])
    bottom_right = end_point

    # Apply rotation if needed
    if rotation_angle != 0:
        center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
        rotated_corners = rotate_box([top_left, top_right, bottom_right, bottom_left], rotation_angle, center)
        top_left, top_right, bottom_right, bottom_left = rotated_corners

    # Draw the rectangle
    cv2.line(image, top_left, top_right, box_color, thickness=4)
    cv2.line(image, top_right, bottom_right, box_color, thickness=4)
    cv2.line(image, bottom_right, bottom_left, box_color, thickness=4)
    cv2.line(image, bottom_left, top_left, box_color, thickness=4)

    return top_left, top_right, bottom_right, bottom_left

def process_images(folder_path, reference_box, ocr_engine, confidence_limit=0.25, output_folder="results", csv_file="boxes.csv"):
    os.makedirs(output_folder, exist_ok=True)
    
    results = []

    for image_file in natsort.natsorted(os.listdir(folder_path)):  # Use natural sorting
        if not image_file.lower().endswith(('.jpg', '.png', '.jpeg')): 
            continue

        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        detections = ocr_engine.readtext(image)

        ref_width = reference_box[2] - reference_box[0]
        ref_height = (reference_box[3] - reference_box[1]) + 15

        for bbox_data in detections:
            corners, detected_text, detection_score = bbox_data

            if detection_score > confidence_limit and "2023" in detected_text:
                top_left, top_right, bottom_right, bottom_left = (
                    np.array(corners[0]),
                    np.array(corners[1]),
                    np.array(corners[2]),
                    np.array(corners[3]),
                )

                rotation_angle = calculate_angle_of_rotation(top_left, top_right)

                actual_width = top_right[0] - top_left[0]
                actual_height = bottom_left[1] - top_left[1]
                scale_w = actual_width / ref_width
                scale_h = actual_height / ref_height

                # Annotate "Roll Number" box (around the student's roll number)
                roll_shift = np.array([-95, 130])
                roll_box_width = 220
                roll_corners = annotate_image(
                    image, top_left + roll_shift, (roll_box_width, ref_height), (1, 1), box_color=(0, 255, 0), rotation_angle=rotation_angle
                )

                # Annotate "Name" box (around the student's name)
                name_shift = np.array([290, 120])
                name_box_width = 280
                name_corners = annotate_image(
                    image, top_left + name_shift, (name_box_width, ref_height), (1, 1), box_color=(0, 255, 0), rotation_angle=rotation_angle  # Changed to green
                )

                # Store results for writing later
                results.append([
                    image_file,
                    roll_corners[0][1], roll_corners[0][0],
                    roll_corners[2][1], roll_corners[2][0],
                    name_corners[0][1], name_corners[0][0],
                    name_corners[2][1], name_corners[2][0],
                ])

        # Save annotated image
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)
    
    # Write sorted results to CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        for row in results:
            writer.writerow(row)

    print(f"Processed images saved in '{output_folder}' and coordinates saved in '{csv_file}'.")

# Define template bounding box coordinates (x1, y1, x2, y2)
reference_box_coords = [366, 119, 641, 172]

# Initialize EasyOCR Reader
ocr_reader = easyocr.Reader(['en'], gpu=True)

# Process images in the target folder
process_images("../top_halves", reference_box_coords, ocr_reader)
