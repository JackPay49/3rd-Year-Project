import cv2
import numpy as np
from PIL import Image

words = ["ABOUT", "BELIEVE", "CHANCE", "FAMILY"]

stacked = True

for word in words:
    for i in range(1, 4):
        file = f"D:/USB/LRW/lipread_mp4/{word}/train/{word}_0000{i}"
        file_src = file + ".mp4"
        file_dst = file + " (stacked).png"

        cap = cv2.VideoCapture(file_src)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break

        if stacked:
            stack = []
            for i in range(3):
                output = frames[(i * 7)]
                for frame in frames[((i * 7) + 1) : ((i + 1) * 7)]:
                    output = np.concatenate((output, frame), axis=1)
                stack.append(output)
            output = np.concatenate((stack[0], stack[1], stack[2]), axis=0)
        else:
            output = frames[0]
            for frame in frames[1:]:
                output = np.concatenate((output, frame), axis=1)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(output)
        img.save(file_dst)
