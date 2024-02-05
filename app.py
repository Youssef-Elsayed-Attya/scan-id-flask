from flask import  Flask , request , render_template,url_for,jsonify
from tensorflow.keras.models import load_model
from  PIL import Image
import numpy as np
import pytesseract
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import re


###################################################################################
def preprocess_image(image, alpha=1.1, beta=35):
    h, w = image.shape[:2]
    new_width = 1024
    ratio = new_width / float(w)
    new_height = int(h * ratio)

    # تغيير حجم الصورة
    resized_image = cv2.resize(image, (new_width, new_height))

    # تحويل الصورة إلى رمادية وتعزيزها
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
    return enhanced_image


def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# معالجة الصورة المقصوصة واستخراج النص
# processed_image = preprocess_image(cropped_image)
# extracted_text = extract_text_from_image(processed_image)

# عرض الصورة المقصوصة والنص المستخرج
# show_image(processed_image, title='Processed Cropped Image')
# print("Extracted Text:\n", extracted_text)

###################################################################################

def find_and_format_id_in_text(text):
    pattern = r"JE (\d+)"
    match = re.search(pattern, text)
    if match:
        return f"JE {match.group(1)}"
    else:
        return 'ID not found'


def crop(img):
    yolo = YOLO('mus_weight t.pt')
    #img = cv2.imread(image_path)
    results = yolo(img)
    results = results[0]
    box = results.boxes
    current_coordinates = [round(x) for x in box.xyxy.flatten().tolist()]
    print(current_coordinates)
    (x1, y1, x2, y2) = current_coordinates
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img

#cropped_image = crop('photo_2023-11-10_22-20-08.jpg')

# إذا كنت تريد عرض الصورة المقصوصة يمكنك استخدام الكود التالي:
#plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
#plt.title('Cropped Image')
#plt.show()
######################################################################################



app= Flask(__name__)

# def preproessing(image):
#     image=Image.open(image)
#     image=image.resize((150,150))
#     image_arr=np.array(image.convert('RGB'))
#     image_arr.shape=(1,150,150,3)
#     return image_arr
#
#
# classes =['Building','Forest','Glacier','Mountain','Sea','Street']
# model=load_model("Intel_Image_Classification.h5")


#
# @app.route('/predict',methods=['GET','POST'])
# def predict():
#     print('run code')
#     if request.method=='POST':
#         print('waiting to load image...')
#         image=request.files['fileup']
#         image_arr=preproessing(image)
#         result=model.predict(image_arr)
#         index=np.argmax(result)
#         prediction=classes[index]

@app.route('/predictApi',methods=['POST'])
def api():
    try:
        if 'fileup' not in request.files:
            return 'please try again, no image found'
        image = request.files.get('fileup')
        image=cv2.imread(image)
        print('there is image')
        cropped_image = crop(image)
        print('cropped image')
        pre_image= preprocess_image(cropped_image)
        print('done preprocess')
        result= extract_text_from_image(pre_image)
        print(result)
        result=find_and_format_id_in_text(result)
        return jsonify({'prediction': result})
    except:
        return jsonify({'error':'error try again'})

if __name__=='__main__':
    # print('cropped image')
    # pre_image = preprocess_image(cropped_image)
    # print('done preprocess')
    # result = extract_text_from_image(pre_image)
    # print(result)
    # result = find_and_format_id_in_text(result)
    # print(result)
    app.run(debug=True)

