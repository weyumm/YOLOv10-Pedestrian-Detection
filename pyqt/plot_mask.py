import cv2
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

# 浅紫色 [148, 24, 110]
# 深紫色 [244, 71, 165]
# 橘黄色 [22, 60, 202]
# 中蓝色 [176, 75, 0]
# 浅绿色 [102, 156, 50]
# 浅红色 [114, 114, 255]
# 蓝色 [255, 0, 0]
# 红色 [0, 0, 255]
# 紫色 [255, 0, 255]
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        # x1, y1, x2, y2 = box.astype(int)
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


# model = YOLO("ultralytics/yolov8s.pt")

# from PIL
# im1 = Image.open("ultralytics/assets/zidane.jpg")
# im1 = cv2.imread('ultralytics/assets/zidane.jpg')
# results = model.predict(source=im1, save=True, imgsz=[640, 640])  # save plotted images
# print(type(results))                    # list        
# print(results[0].names)                 # {0: '', 1: '', ...}
# print(results[0].speed)                 # {'preprocess': 4.937 'inference': 56.6473, 'postprocess': 1.7004}
# print(results[0].orig_shape)            # (720, 1280)
# print(results[0].boxes.conf.shape[0])   # 3
# print(results[0].boxes.conf)            # tensor([0.8891, 0.8845, 0.7178], device='cuda:0')
# print(results[0].boxes.cls)             # tensor([ 0.,  0., 27.], device='cuda:0')
# # tensor([[ 747.3132,   41.4733, 1140.3921,  712.9236],
# #         [ 144.8749,  200.0329, 1107.1971,  712.7000],
# #         [ 437.3798,  434.4805,  529.9605,  717.0511]], device='cuda:0')
# print(results[0].boxes.xyxy) 
# # tensor([[943.8527, 377.1985, 393.0789, 671.4503],
# #         [626.0360, 456.3665, 962.3223, 512.6671],
# #         [483.6702, 575.7658,  92.5807, 282.5707]], device='cuda:0')
# print(results[0].boxes.xywh) 

# print(results[0].boxes.conf.tolist())
# print(results[0].boxes.cls.tolist())
# print(results[0].boxes.xyxy.tolist())

# conf_list = results[0].boxes.conf.tolist()
# cls_list_int = [int(i) for i in results[0].boxes.cls.tolist()]
# xyxy_list_int = [[round(num) for num in sublist] for sublist in results[0].boxes.xyxy.tolist()]

# combined_image = draw_detections(im1, xyxy_list_int, conf_list, cls_list_int, 0.4)
# print(combined_image.shape)
# cv2.imwrite('detected_bus.jpg', combined_image)

# for result in results:
#     # Detection
#     # result.boxes.xyxy   # box with xyxy format, (N, 4)
#     # result.boxes.xywh   # box with xywh format, (N, 4)
#     # result.boxes.conf   # confidence score, (N, 1)
#     # result.boxes.cls    # cls, (N, 1)
#     print(result.boxes.xyxy)
#     print(result.boxes.conf)
#     print(result.boxes.cls)
#     bbox = result.boxes.xyxy
#     conf =result.boxes.conf
#     cls = result.boxes.cls
#     concatenated_tensor = [torch.cat((bbox, conf.unsqueeze(1), cls.unsqueeze(1)), dim=1)]
#     print(concatenated_tensor)

# res_ploted = results[0].plot()
# cv2.imshow('result', res_ploted)
# cv2.waitKey(0)

# # # Each result is composed of torch.Tensor by default, 
# # # in which you can easily use following functionality:
# # result = result.cuda()
# # result = result.cpu()
# # result = result.to("cpu")
# # result = result.numpy()