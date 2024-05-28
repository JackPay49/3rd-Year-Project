# Will grab a random frame, make a face map and then export 4 images: Normal full-frame, frame with landmarks drawn on, frame with a bounding box drawn on and a crop of the lips
import cv2
import numpy as np
from PIL import Image
import random
import sys
from copy import deepcopy

sys.path.append("../classes")

from FaceMapper import FaceMapper
from utility.Dataclasses import Region

FM = FaceMapper()
file_dst = f"D:/USB/LRW/lipread_mp4/FAMILY/train/FAMILY_00002_Crop_Example"


file = f"D:/USB/LRW/lipread_mp4/FAMILY/train/FAMILY_00002"
file_src = f"{file}.mp4"

cap = cv2.VideoCapture(file_src)
frame_index = random.randint(1, 28)
for i in range(frame_index):
    _, frame = cap.read()

FM.make_face_map(img=frame)

bbox_img = deepcopy(frame)
for bbox in FM.lip_bboxs:
    bbox_img = cv2.rectangle(
        img=bbox_img,
        pt1=bbox.get_min(),
        pt2=bbox.get_max(),
        color=(0, 255, 0),
        thickness=2,
    )

ldmk_img = deepcopy(frame)
width, height = Image.fromarray(ldmk_img).size
for ldmk_list in FM.lip_landmarks_condensed:
    for ldmk in ldmk_list:
        ldmk_img = cv2.circle(
            img=ldmk_img,
            center=(int(ldmk[0] * width), int(ldmk[1] * height)),
            radius=1,
            thickness=1,
            color=(255, 0, 0),
        )


lip_crop = deepcopy(FM.crop_to_face_region(region_type=Region.LIPS)[0])


Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(f"{file_dst}.png")
Image.fromarray(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)).save(f"{file_dst}_bbox.png")
Image.fromarray(cv2.cvtColor(lip_crop, cv2.COLOR_BGR2RGB)).save(f"{file_dst}_Crop.png")
Image.fromarray(cv2.cvtColor(ldmk_img, cv2.COLOR_BGR2RGB)).save(f"{file_dst}_ldmk.png")
