# 根据task参数选择进行表情识别或者性别识别
import cv2
import numpy as np
from keras.models import load_model # 加载模块
# 报错！！！
from src.utils.grad_cam import compile_gradient_function

from src.utils.grad_cam import compile_saliency_function
from src.utils.grad_cam import register_gradient
from src.utils.grad_cam import modify_backprop
from src.utils.grad_cam import calculate_guided_gradient_CAM

from src.utils.inference import detect_faces
from src.utils.inference import apply_offsets
from src.utils.inference import load_detection_model
from src.utils.preprocessor import preprocess_input
from src.utils.inference import draw_bounding_box
from src.utils.datasets import get_class_to_arg # label转换

# 根据task参数选择进行表情识别或者性别识别 gender / emotion
task = 'gender' #  性别识别
if task == 'gender':
    model_filename = '../trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5'
    class_to_arg = get_class_to_arg('imdb') # {'woman': 0, 'man': 1}
    predicted_class = 0
    offsets = (0,0)
elif task == 'emotion':
    model_filename = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    class_to_arg = get_class_to_arg('fer2013') # 表签转换
    # {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4,'surprise': 5, 'neutral': 6}
    predicted_class = 0
    offsets = (0,0)

# load_model:
model = load_model(model_filename,compile=False)
gradient_function = compile_gradient_function(model,predicted_class,'conv2d_7')
register_gradient()

guided_model = modify_backprop(model,'GuidedBackProp',task)
saliency_function = compile_saliency_function(guided_model,'conv2d_7')

# 加载图片和数据的相关参数
# 人脸检测分类器
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
# 检测到的人脸
face_detection = load_detection_model(detection_model_path)

# 设定人脸框的颜色
color = (0,255,0) # green
# 得到输入模型的shape
target_size = model.input_shape[1:3]

#
emotion_window = []
cv2.namedWindow('face_window')
# 摄像头
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)  # 灰度图
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 彩色图
    # 人脸
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        guided_gradCAM = calculate_guided_gradient_CAM(gray_face,
                            gradient_function,saliency_function)
        guided_gradCAM = cv2.resize(guided_gradCAM,(x2-x1,y2-y1))

        try:
            rgb_guided_gradCAM = cv2.repeat(guided_gradCAM[:,:,np.newaxis],3,axis=2)
            rgb_image[y1:y2,x1:x2,:] = rgb_guided_gradCAM
        except:
            continue
        draw_bounding_box(x1,y1,x2-x1,y2-y1, rgb_image, color)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    try:
        cv2.imshow('window_face', bgr_image)
    except:
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break














