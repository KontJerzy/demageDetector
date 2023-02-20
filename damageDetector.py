import numpy as np
import cv2
import requests
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import os
from keras.models import load_model, model_from_json
import keras
import numpy as np
import json
from mrcnn.config import Config
import json
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def create_new_model_from_config(config):
    # Create a new model using the configuration object
    model = Sequential()
    model.add(Dense(units=config.num_units, activation=config.activation))
    model.add(Dense(units=config.num_classes, activation='softmax'))
    model.compile(loss=config.loss, optimizer=config.optimizer, metrics=config.metrics)
    return model


class SpaceNetConfig(Config):
    def __init__(self, num_units=None, num_classes=None, activation=None, loss=None, optimizer=None, metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.num_classes = num_classes
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class ModelLoader:
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path

    def return_dict(self, config):

        config_dict = {}
        for key in dir(config):
            if not key.startswith("__") and key.isupper():
                value = getattr(config, key)
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                config_dict[key] = value

        # Add the class name to the dictionary
        config_dict = {'class_name': 'SpaceNetConfig', 'config': config_dict}

        return config_dict

    def save_config(self):
        # Create a new configuration object and convert it to a dictionary
        config = SpaceNetConfig(
            num_units=32,
            num_classes=10,
            activation='relu',
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        # Convert the config object to a dictionary
        config_dict = self.return_dict(config)

        # Save the configuration as a JSON file
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f)

    def load_config(self):
        # Load the configuration from the config.json file
        with open(self.config_path, 'r') as f:
            config_dict = json.load(f)
        config_class_name = config_dict.pop('class_name')
        config_class = globals()[config_class_name]
        config = config_class(**config_dict)
        return config

    def load_trained_model(self):
        # Load the configuration from the config.json file
        with open(self.config_path, 'r') as f:
            config_dict = json.load(f)
        config = SpaceNetConfig()
        config.from_dict(config_dict)
        # Load the weights into the model
        # Load the model using the updated configuration object
        model = create_new_model_from_config(config)
        model.load_weights(self.model_path)

        return model
        """_
        # Load the model
        model = load_model(self.model_path)

        # Compile the model
        model.compile(
            optimizer=Adam(lr=config.LEARNING_RATE),
            loss={
                'rpn_class_loss': None,
                'rpn_bbox_loss': None,
                'mrcnn_class_loss': None,
                'mrcnn_bbox_loss': None,
                'mrcnn_mask_loss': None
            }
        )

        return model
        """

class EarthquakeDamageEstimator:
    def __init__(self, model_path, config_path, threshold=0.5, input_size=(512, 512)):
        self.model_path = model_path
        self.config_path = config_path
        self.threshold = threshold
        self.input_size = input_size
        #self.model, self.config = self.load_trained_model()
        loader = ModelLoader('mask_rcnn_spacenet_0151.h5', 'config.json')
        loader.save_config()
        model = loader.load_trained_model()
        self.model = model


    # Function to load satellite images
    def load_images(self, pre_earthquake_path, post_earthquake_path):
        pre_earthquake_image = Image.open(pre_earthquake_path)
        post_earthquake_image = Image.open(post_earthquake_path)
        return pre_earthquake_image, post_earthquake_image

    # Function to preprocess images
    def preprocess_images(self, pre_earthquake_image, post_earthquake_image):
        pre_earthquake_image = pre_earthquake_image.resize(self.input_size)
        pre_earthquake_image = np.asarray(pre_earthquake_image) / 255.0
        post_earthquake_image = post_earthquake_image.resize(self.input_size)
        post_earthquake_image = np.asarray(post_earthquake_image) / 255.0
        return pre_earthquake_image, post_earthquake_image

    # Function to apply pre-trained model to images
    def detect_buildings(self, image):
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0)
        self.model.setInput(blob)
        detections = self.model.forward()
        mask = (detections > self.threshold).astype('uint8')
        contours, hierarchy = cv2.findContours(mask[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # Function to estimate damage levels for detected buildings
    def estimate_damage_levels(self, buildings_pre, buildings_post):
        damage_levels = []
        for i in range(len(buildings_pre)):
            if i >= len(buildings_post):
                damage_levels.append('destroyed')
            else:
                pre_area = cv2.contourArea(buildings_pre[i])
                post_area = cv2.contourArea(buildings_post[i])
                area_ratio = post_area / pre_area
                if area_ratio > 0.9:
                    damage_levels.append('low')
                elif area_ratio > 0.7:
                    damage_levels.append('medium')
                else:
                    damage_levels.append('high')
        return damage_levels

    def visualize_results(self, pre_earthquake_image, post_earthquake_image, save_path=None):
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

        # Show or save the final image
        if save_path is not None:
            cv2.imwrite(save_path, final_image)
        else:
            cv2.imshow('Results', final_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # Make model run 
    model_path = './mask_rcnn_spacenet_0151.h5'
    config_path = './config.json'
    model_path = os.path.abspath(model_path)
    config_path = os.path.abspath(config_path)
    estimator = EarthquakeDamageEstimator(model_path, config_path)
    pre_earthquake_image, post_earthquake_image = estimator.load_images('./test/pre_earthquake.png', './test/post_earthquake.png')
    estimator.visualize_results(pre_earthquake_image, post_earthquake_image)


