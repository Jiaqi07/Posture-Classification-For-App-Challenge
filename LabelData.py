import os
import cv2
import numpy as np

# session_dir = "/home/alan/.ros/"
# filename_prefix = "color_Frame #"
#
# session_number = 38  # You may change this number as needed
# session_file = session_dir + "session_" + str(session_number) + "/"

frame_list_filename = "C:/Users/ac913/PycharmProjects/appChallenge/tmp.txt"
keyframes_filename = "C:/Users/ac913/PycharmProjects/unlabeled_data/Frame_"

# frame_list_file = open(frame_list_filename, "w")
# frame_index = 0
# while True:
#     filename = session_file + filename_prefix + str(frame_index) + ".png"
#     if not os.path.exists(filename):
#         break
#     frame_list_file.write(f"{frame_index}\n")
#     frame_index += 1
# frame_list_file.close()
#
#
# # Once you have assigned the keyframe labels, create the keyframes file
frame_list_file = open(frame_list_filename, "r")
keyframes_file = open(keyframes_filename, "w")
fes = frame_list_file.read()
#
cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Image", 640, 480)

labels = []
line = 0
while line < len(fes) - 1:
    filename = keyframes_filename + str(line) + ".jpg"
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    if image is not None:
        cv2.imshow("Color Image", image)

        key = cv2.waitKey(0)
        if key == ord('1') or key == ord('0'):
            # Write the filename with label to keyframes.txt
            label = int(chr(key))  # Convert the ASCII value of key to an integer
            labels.append(label)
            print(str(line) + " " + str(label))
            line += 1
        elif key == ord('q'):
            break
        elif key == ord('z'):
            if line > 0:
                labels.pop()  # Remove the last label (Undo)
                line -= 1
                print("UNDOED " + str(line))
    else:
        break

print(np.asarray(labels))
np.savetxt(keyframes_filename, np.asarray(labels), fmt='%d')
# keyframes_file.write(np)

frame_list_file.close()
keyframes_file.close()

cv2.destroyAllWindows()
