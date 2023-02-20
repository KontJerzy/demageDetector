import cv2
import numpy as np
import requests
from PIL import Image

class EarthquakeDamageEstimator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def load_images(self, pre_earthquake_image_path, post_earthquake_image_path):
        # Load images
        pre_earthquake_image = cv2.imread(pre_earthquake_image_path)
        post_earthquake_image = cv2.imread(post_earthquake_image_path)

        # Check if images were loaded successfully
        if pre_earthquake_image is None:
            raise ValueError(f"Could not load image at {pre_earthquake_image_path}")
        if post_earthquake_image is None:
            raise ValueError(f"Could not load image at {post_earthquake_image_path}")

        return pre_earthquake_image, post_earthquake_image

    def preprocess_images(self, pre_earthquake_image, post_earthquake_image):
        # Resize images
        height, width = pre_earthquake_image.shape[:2]
        if height > 1000 or width > 1000:
            scale_percent = 50
            pre_earthquake_image = cv2.resize(pre_earthquake_image, (int(width * scale_percent / 100), int(height * scale_percent / 100)))
            post_earthquake_image = cv2.resize(post_earthquake_image, (int(width * scale_percent / 100), int(height * scale_percent / 100)))

        # Convert images to grayscale
        pre_earthquake_image = cv2.cvtColor(pre_earthquake_image, cv2.COLOR_BGR2GRAY)
        post_earthquake_image = cv2.cvtColor(post_earthquake_image, cv2.COLOR_BGR2GRAY)

        # Normalize pixel values
        pre_earthquake_image = pre_earthquake_image.astype(np.float32) / 255.0
        post_earthquake_image = post_earthquake_image.astype(np.float32) / 255.0

        return pre_earthquake_image, post_earthquake_image


    def detect_buildings(self, pre_earthquake_image, post_earthquake_image):
        # Call Megadetector API and retrieve results
        url = "https://detect.roboflow.com/object-detection/<your_model_id>"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer <your_api_key>",
        }
        data = {
            "image": pre_earthquake_image.tolist(),
            "resize": {"width": 416, "height": 416},
        }
        response = requests.post(url, headers=headers, json=data)
        pre_results = response.json()

        data = {
            "image": post_earthquake_image.tolist(),
            "resize": {"width": 416, "height": 416},
        }
        response = requests.post(url, headers=headers, json=data)
        post_results = response.json()

        # Extract building contours
        buildings_pre = []
        buildings_post = []
        for result in pre_results["predictions"]:
            if result["class_name"] == "building" and result["confidence"] >= self.threshold:
                building_pre = np.array(result["segmentation"])
                buildings_pre.append(building_pre.reshape(-1, 2))

        for result in post_results["predictions"]:
            if result["class_name"] == "building" and result["confidence"] >= self.threshold:
                building_post = np.array(result["segmentation"])
                buildings_post.append(building_post.reshape(-1, 2))

        return buildings_pre, buildings_post

    def estimate_damage_levels(self, buildings_pre, buildings_post):
        # Estimate damage levels for each building based on the ratio of post-earthquake area to pre-earthquake area
        damage_levels = []
        for i in range(len(buildings_pre)):
            pre_area = cv2.contourArea(buildings_pre[i])
            post_area = cv2.contourArea(buildings_post[i])
            area_ratio = post_area / pre_area
            if area_ratio >= 0.9:
                damage_levels.append('low')
            elif area_ratio >= 0.7:
                damage_levels.append('medium')
            else:
                damage_levels.append('high')
        return damage_levels

    def visualize_results(self, pre_earthquake_image, post_earthquake_image):
        # Detect buildings in pre- and post-earthquake images
        buildings_pre, buildings_post = self.detect_buildings(pre_earthquake_image, post_earthquake_image)

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

if __name__ == '__main__':
    estimator = EarthquakeDamageEstimator()
    pre_earthquake_image, post_earthquake_image = estimator.load_images('./test/pre_earthquake.png', './test/post_earthquake.png')
    estimator.visualize_results(pre_earthquake_image, post_earthquake_image)




