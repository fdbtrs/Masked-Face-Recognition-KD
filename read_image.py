import cv2

f=open("fadi/detect.txt","r")
for l in f.readlines():
    splits=l.split(" ")
    img=cv2.imread(splits[0])
    color = (255, 0, 0)
    print(int(float(splits[1])))
    #img=cv2.rectangle(img, (int(float(splits[1]())),int(float(splits[2]))),(int(float(splits[1]+splits[3])),int(float(splits[2]+splits[4]))),color=color)
    img = cv2.rectangle(img, (int(float(splits[1])),int(float(splits[2]))),(int(float(splits[3])),int(float(splits[4]))), color=color)
    #left eye
    center_coordinates = (int(float(splits[5])),int(float(splits[6])))

    # Radius of circle
    radius = 5

    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    #right eye
    img = cv2.circle(img, center_coordinates, radius, color, thickness)

    center_coordinates = (int(float(splits[7])),int(float(splits[8])))

    # Radius of circle
    radius = 5

    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    img = cv2.circle(img, center_coordinates, radius, color, thickness)


    #mouse
    center_coordinates = (int(float(splits[9])),int(float(splits[10])))

    # Radius of circle
    radius = 5

    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    img = cv2.circle(img, center_coordinates, radius, color, thickness)


    center_coordinates = (int(float(splits[11])),int(float(splits[12])))

    # Radius of circle
    radius = 5

    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    img = cv2.circle(img, center_coordinates, radius, color, thickness)



    center_coordinates = (int(float(splits[13])),int(float(splits[14])))

    # Radius of circle
    radius = 5

    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    img = cv2.circle(img, center_coordinates, radius, color, thickness)
    cv2.imwrite(splits[0].replace(".jpg","_bbx.jpg"),img)
    cv2.imshow("bbox",img)
    cv2.waitKey()