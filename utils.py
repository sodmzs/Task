from configparser import ConfigParser
from roboflow import Roboflow
import cv2
from PIL import Image
import io
import numpy as np
import requests
import base64

config = ConfigParser()

def replacing_gun_with_musical_instrument(file_name, mask_file, instrument_file):
    print (">> replacing_gun_with_musical_instrument) executing..")
    image = cv2.imread(file_name)

    mask = cv2.imread(mask_file, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    gun_roi = image[y:y+h, x:x+w]

    tuba_image = cv2.imread(instrument_file, cv2.IMREAD_UNCHANGED)

    resized_tuba = cv2.resize(tuba_image, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha_channel = resized_tuba[:, :, 3]

    resized_alpha = cv2.resize(alpha_channel, (w, h), interpolation=cv2.INTER_LINEAR)

    resized_alpha = cv2.cvtColor(resized_alpha, cv2.COLOR_GRAY2BGR)
    alpha_mask = resized_alpha / 255.0
    merged_roi = (gun_roi * (1 - alpha_mask)).astype(np.uint8)
    merged_tuba = (resized_tuba[:, :, :3] * alpha_mask).astype(np.uint8)

    modified_roi = cv2.add(merged_roi, merged_tuba)

    image[y:y+h, x:x+w] = modified_roi

    return image



def smile_expression(file_name):
    print (">> smile_expression() executing..")
    print (config.read('config.ini'))
    
    files = {
        'image_target': open(file_name, 'rb'),
        'service_choice': (None, '""'),
    }

    response = requests.post(config.get('ailabapi','url'), files=files, headers = { 'ailabapi-api-key': config.get('ailabapi','api_key')})

    response = response.json()
    if (response):
        image = Image.open(io.BytesIO(base64.b64decode(response['data']['image'])))
        image.save("Temp/smile.png")
        return "smile.png","smile.png successfully saved in the temp file for the reference."
    else:
        return response['error_msg']


def inpainting_detected_gun(file_name,masked_file):
    print (">> inpainting_detected_gun() executing..")
    print (config.read('config.ini'))
    
    image_file_object = open(file_name,'rb')
    mask_file_object = open(masked_file,'rb')
    r = requests.post('https://clipdrop-api.co/cleanup/v1',
                      files = {
                          'image_file': (file_name, image_file_object, 'image/jpeg'),
                          'mask_file': (masked_file, mask_file_object, 'image/jpeg')},
                      headers = { 'x-api-key': config.get('clipdrop','api_key')})

    if (r.ok):
        image = Image.open(io.BytesIO(r.content))
        image.save("Temp/inpainted_gun.png")
        return "inpainted_gun.png","inpainted_gun.png successfully saved in the temp file for the reference."
    else:
        return r.raise_for_status()

def detect_gun(file_name):
    print (">> detect_gun() executing..")
    print (config.read('config.ini'))

    rf = Roboflow(api_key=config.get('roboflow','api_key'))
    project = rf.workspace().project(config.get('roboflow','project_name'))
    model = project.version(config.get('roboflow','version')).model

    predictions = model.predict(file_name).json()

    model.predict(file_name).save("Temp/detect.png")

    img = cv2.imread("Temp/detect.png")

    for bounding_box in predictions["predictions"]:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
    
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(img, start_point, end_point, (0,255,0), -1)
   

    #for masking

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([52, 0, 55])
    upper = np.array([104, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(img,img, mask= mask)

    return res
