import cv2
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
import matplotlib.pyplot as plt

class BuildingDamageDetector:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    # Function to load pre-trained model from API
    def load_model(model_api):
        response = requests.get(model_api)
        model_bytes = response.content
        model = cv2.dnn.readNetFromONNX(model_bytes)
        return model

    # Function to load satellite images
    def load_images(pre_earthquake_path, post_earthquake_path):
        pre_earthquake_image = Image.open(pre_earthquake_path)
        post_earthquake_image = Image.open(post_earthquake_path)
        return pre_earthquake_image, post_earthquake_image

    # Function to preprocess images
    def preprocess_images(pre_earthquake_image, post_earthquake_image, input_size):
        pre_earthquake_image = pre_earthquake_image.resize(input_size)
        pre_earthquake_image = np.asarray(pre_earthquake_image) / 255.0
        post_earthquake_image = post_earthquake_image.resize(input_size)
        post_earthquake_image = np.asarray(post_earthquake_image) / 255.0
        return pre_earthquake_image, post_earthquake_image

    # Function to apply pre-trained model to images
    def detect_buildings(model, pre_earthquake_image, post_earthquake_image, threshold):
        blob_pre = cv2.dnn.blobFromImage(pre_earthquake_image, scalefactor=1/255.0)
        model.setInput(blob_pre)
        detections_pre = model.forward()
        mask_pre = (detections_pre > threshold).astype('uint8')

        blob_post = cv2.dnn.blobFromImage(post_earthquake_image, scalefactor=1/255.0)
        model.setInput(blob_post)
        detections_post = model.forward()
        mask_post = (detections_post > threshold).astype('uint8')

        _, contours_pre, _ = cv2.findContours(mask_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_post, _ = cv2.findContours(mask_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        buildings_pre = []
        for contour in contours_pre:
            area = cv2.contourArea(contour)
            if area > 50:
                buildings_pre.append(contour)

        buildings_post = []
        for contour in contours_post:
            area = cv2.contourArea(contour)
            if area > 50:
                buildings_post.append(contour)

        return buildings_pre, buildings_post

    # Function to estimate the damage level of each building
    def estimate_damage_levels(pre_buildings, post_buildings, threshold_low, threshold_high):
        damage_levels = []
        for pre_building, post_building in zip(pre_buildings, post_buildings):
            pre_area = cv2.contourArea(pre_building)
            post_area = cv2.contourArea(post_building)
            damage = (post_area - pre_area) / pre_area
            if damage < threshold_low:
                damage_level = 'low'
            elif damage < threshold_high:
                damage_level = 'medium'
            else:
                damage_level = 'high'
            damage_levels.append(damage_level)
        return damage_levels


    def visualize_results(self, pre_earthquake_image, post_earthquake_image):
        # Detect buildings in pre- and post-earthquake images
        buildings_pre = self.detect_buildings(pre_earthquake_image)
        buildings_post = self.detect_buildings(post_earthquake_image)

        # Estimate damage levels for each building
        damage_levels = self.estimate_damage_levels(buildings_pre, buildings_post)

        # Convert images to OpenCV format
        pre_earthquake_image_cv = np.array(pre_earthquake_image)
        post_earthquake_image_cv = np.array(post_earthquake_image)

        # Draw contours of detected buildings
        cv2.drawContours(pre_earthquake_image_cv, buildings_pre, -1, (0, 255, 0), 2)
        cv2.drawContours(post_earthquake_image_cv, buildings_post, -1, (0, 255, 0), 2)

        # Add text labels for damage levels
        for i in range(len(buildings_pre)):
            color = (0, 0, 0)
            if damage_levels[i] == 'low':
                color = (0, 255, 0)
            elif damage_levels[i] == 'medium':
                color = (0, 255, 255)
            elif damage_levels[i] == 'high':
                color = (0, 0, 255)
            cv2.putText(pre_earthquake_image_cv, damage_levels[i], tuple(buildings_pre[i][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(post_earthquake_image_cv, damage_levels[i], tuple(buildings_post[i][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Concatenate images horizontally
        result_image = np.concatenate((pre_earthquake_image_cv, post_earthquake_image_cv), axis=1)

        # Show legend for damage levels
        legend_image = np.zeros((100, 300, 3), dtype=np.uint8)
        cv2.putText(legend_image, 'Low damage', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(legend_image, 'Medium damage', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(legend_image, 'High damage', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Concatenate result and legend images vertically
        final_image = np.concatenate((result_image, legend_image), axis=0)

        # Display the final image
        cv2.imshow('Results', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #

# Load pre- and post-earthquake images
pre_earthquake_image = cv2.imread('pre_earthquake.jpg')
post_earthquake_image = cv2.imread('post_earthquake.jpg')
visualize_results(pre_earthquake_image, post_earthquake_image)