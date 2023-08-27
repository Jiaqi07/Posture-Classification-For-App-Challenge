import os
import cv2
import numpy as np

m = 0
data_dir = "C:/Users/ac913/PycharmProjects/appChallenge/unlabeled_data/folder_" + str(m) + "/"
keyframes_filename = "C:/Users/ac913/PycharmProjects/appChallenge/labeled_data/" + str(m) + ".txt"


keyframes_file = open(keyframes_filename, "w")
fes = len([entry for entry in os.listdir("C:/Users/ac913/PycharmProjects/appChallenge/unlabeled_data/folder_" + str(m) + "/") if os.path.isfile(os.path.join("C:/Users/ac913/PycharmProjects/appChallenge/unlabeled_data/folder_" + str(m) + "/", entry))])

cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Image", 640, 480)

labels = []
line = 0
while line < fes-1:
    filename = data_dir + "Frame_" + str(line) + ".jpg"
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    if image is not None:
        cv2.imshow("Color Image", image)

        key = cv2.waitKey(0)
        if key == ord('1') or key == ord('0'):
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

keyframes_file.close()

cv2.destroyAllWindows()
