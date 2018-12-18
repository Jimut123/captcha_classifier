import cv2
import numpy as np
import sys
from itertools import combinations
from PIL import Image 
samples = np.loadtxt('samples.data',np.float32)
characters = np.loadtxt('characters.data',np.float32)

model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE, characters)


def extract_characters(image_file):
    image = cv2.imread(image_file)
    
    upper = np.array([255,255,255],dtype=np.uint8)
    lower = np.array([200,200,200],dtype=np.uint8)

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)

    output_g = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    
    ret,thresh = cv2.threshold(output_g,127,255,0)
    im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    colors = [(0,255,0),(255,0,0),(0,0,255)]
    coordinates=[]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        coordinates.append((x,y,w,h))
    coordinates.sort(key=lambda x:x[0])
    print(coordinates)
    c_copy = coordinates[:]

    comb = combinations(list(range(len(coordinates))),2)
    for com in comb:
        curr = c_copy[com[0]]
        curr2 = c_copy[com[1]]
        if curr not in coordinates or curr2 not in coordinates:
            continue
        print("Comparing: "+str(curr) + " with "+str(curr2))
        if (curr2[0] >= curr[0] and curr2[1] >= curr[1] and curr2[0]+curr2[2] <= curr[0]+curr[2] and curr2[1]+curr2[3] <= curr[1] + curr[3]):
                print("Removing 1:" + str(curr2))
                coordinates.remove(curr2)
        elif (curr2[0] >= curr[0] and curr2[0]+curr2[2] <= curr[0]+curr[2] and curr2[1] <= curr[1] and curr2[1]+curr2[3] <= curr[1] + curr[3]):
                print("Removing 2:" + str(curr2))
                ind = coordinates.index(curr)
                coordinates[ind] = (curr[0],curr2[1],curr[2],(curr[1]+curr[3])-curr2[1])
                coordinates.remove(curr2)
                


    digits = []
    for coord in coordinates:
        crop_img = output_g[coord[1]:coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
        digits.append(crop_img)
    return image, digits, coordinates

filename = sys.argv[1]
image, characters, coordinates = extract_characters(filename)
if len(characters) != 6:
    print("Error")
    raise Exception
    
print(characters)
for index, character in enumerate(characters):
    character = cv2.resize(character,(10,10))
    character = character.reshape((1,100))
    character = np.float32(character)
    
    retval, results, neigh_resp, dists = model.findNearest(character, k = 1)
    string = chr(int((results[0][0])))
    coord = coordinates[index]
    cv2.rectangle(image, (coord[0],coord[1]),(coord[0]+coord[2],coord[1]+coord[3]),(0,255,0),1)
    cv2.putText(image, string, (coord[0],coord[1]+coord[3]), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2,cv2.LINE_AA)
    print(string, end='')
    
cv2.imshow("image",image)
cv2.waitKey(0)
