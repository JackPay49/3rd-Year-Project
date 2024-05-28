# Will display a single landmark on an image, outputting the raw coordinate and then the normalised landmark
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
file_dst = f"D:/USB/LRW/lipread_mp4/FAMILY/train/FAMILY_00001_Landmark_Example"


file = f"D:/USB/LRW/lipread_mp4/FAMILY/train/FAMILY_00001"
file_src = f"{file}.mp4"

cap = cv2.VideoCapture(file_src)
frame_index = random.randint(1, 28)
for i in range(frame_index):
    _, frame = cap.read()

new_frame = deepcopy(frame)
width, height = Image.fromarray(new_frame).size
FM.make_face_map(img=frame)

ldmk_index = random.randint(0, len(FM.lip_landmarks_condensed[0]))
print(ldmk_index)

raw_ldmk = FM.lip_landmarks_condensed[0][ldmk_index]
new_frame = cv2.circle(
    img=new_frame,
    center=(int(raw_ldmk[0] * width), int(raw_ldmk[1] * height)),
    radius=1,
    thickness=1,
    color=(255, 0, 0),
)

norm_ldmk = FM.normalised_lip_landmarks[0][0][ldmk_index]
lip_crop = deepcopy(FM.crop_to_face_region(region_type=Region.LIPS)[0])
width, height = Image.fromarray(lip_crop).size
lip_crop = np.array(lip_crop)
lip_crop = cv2.circle(
    img=lip_crop,
    center=(int(norm_ldmk[0] * width), int(norm_ldmk[1] * height)),
    radius=1,
    thickness=1,
    color=(255, 0, 0),
)

with open(f"{file_dst}.txt", "w+") as file:
    file.write(f"Raw landmarks: {raw_ldmk}")
    file.write(f"Normalised landmarks: {norm_ldmk}")

img = Image.fromarray(cv2.cvtColor(lip_crop, cv2.COLOR_BGR2RGB))
img.save(f"{file_dst}_Crop.png")

img = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
img.save(f"{file_dst}.png")
