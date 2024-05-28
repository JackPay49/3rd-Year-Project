# Will create an image showing the person, then the person with their lip landmarks highlighted and then the lip crop
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
file_dst = "MediaPipe Example.png"

combined = []

for i in range(1, 4):
    file = f"D:/USB/LRW/lipread_mp4/FAMILY/train/FAMILY_0000{i}"
    file_src = f"{file}.mp4"

    cap = cv2.VideoCapture(file_src)
    frame_index = random.randint(1, 28)
    for i in range(frame_index):
        _, frame = cap.read()

    new_frame = deepcopy(frame)
    width, height = Image.fromarray(new_frame).size
    FM.make_face_map(img=frame)
    for ldmk_list in FM.lip_landmarks_condensed:
        for ldmk in ldmk_list:
            new_frame = cv2.circle(
                img=new_frame,
                center=(int(ldmk[0] * width), int(ldmk[1] * height)),
                radius=1,
                thickness=1,
                color=(255, 0, 0),
            )

    lip_crop = deepcopy(FM.crop_to_face_region(region_type=Region.LIPS)[0])
    lip_crop = Image.fromarray(np.array(lip_crop))
    pad_width = int((Image.fromarray(frame).width - lip_crop.width) / 2)
    pad_height = int((Image.fromarray(frame).height - lip_crop.height) / 2)

    left = pad_width
    bottom = pad_height
    if int((pad_width * 2) + lip_crop.width) != Image.fromarray(frame).width:
        right = pad_width + 1
    else:
        right = pad_width
    if int((pad_height * 2) + lip_crop.height) != Image.fromarray(frame).height:
        top = pad_height + 1
    else:
        top = pad_height

    lip_crop = cv2.copyMakeBorder(
        src=np.array(lip_crop),
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    combined_img = np.concatenate((frame, new_frame, lip_crop), axis=1)
    combined.append(combined_img)

output = combined[0]
for frame in combined[1:]:
    output = np.concatenate((output, frame), axis=0)
img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
img.save(file_dst)
