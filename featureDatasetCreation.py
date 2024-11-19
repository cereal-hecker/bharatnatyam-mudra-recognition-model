import cv2 # type: ignore
import mediapipe as mp # type: ignore
import os
import numpy as np # type: ignore

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

dataset_path = 'bharatnatyam_mudras' 
output_path = 'output_images'

if not os.path.exists(output_path):
    os.makedirs(output_path)


def process_images(image_folder, output_folder):
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    
    for mudra in os.listdir(image_folder):
        mudra_folder = os.path.join(image_folder, mudra)
        save_folder = os.path.join(output_folder, mudra)
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for img_name in os.listdir(mudra_folder):
            img_path = os.path.join(mudra_folder, img_name)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                img_height, img_width, _ = img.shape
                blank_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(blank_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                save_img_path = os.path.join(save_folder, img_name)
                cv2.imwrite(save_img_path, blank_image)
    
    hands.close()

process_images(dataset_path, output_path)
