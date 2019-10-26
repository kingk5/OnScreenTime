import cv2
import face_recognition

#get the vedio
input_movie = cv2.VideoCapture("H:/work/opencv/screenontime/Kabir Singh_Trim.mp4") #write the actual local address of the vedio

#getting the fps of the vedio to get the screen on time
fps = int(input_movie.get(cv2.CAP_PROP_FPS))
print(fps)

#initiallizing all cast's screen time to zero
amitabh=0
mac=0
paidi=0
pinchoo=0
ramlaghoo=0
saira=0
sulakshana=0
vinod=0
yunus=0

#getting images of all cast
#replace the address of load_image_file with actual local address of the particular actors image(just one image is sufficient of the same era and front image)

image = face_recognition.load_image_file("H:/work/opencv/screenontime/actual/amitabh.jpg")
face_locations_amitabh = face_recognition.face_locations(image, model="hog")
face_encoding1 = face_recognition.face_encodings(image, face_locations_amitabh)[0]

image = face_recognition.load_image_file("H:/work/opencv/screenontime/actual/Mac.jpg")
face_locations_mac = face_recognition.face_locations(image, model="hog")
face_encoding2 = face_recognition.face_encodings(image, face_locations_mac)[0]

image = face_recognition.load_image_file("H:/work/opencv/screenontime/actual/Paidi.jpg")
face_locations_paidi = face_recognition.face_locations(image, model="hog")
face_encoding3 = face_recognition.face_encodings(image, face_locations_paidi)[0]

image = face_recognition.load_image_file("H:/work/opencv/screenontime/actual/Pinchoo.jpg")
face_locations_pinchoo = face_recognition.face_locations(image, model="hog")
face_encoding4 = face_recognition.face_encodings(image, face_locations_pinchoo)[0]

image = face_recognition.load_image_file("H:/work/opencv/screenontime/actual/ramlaghoo.jpg")
face_locations_ramlaghoo = face_recognition.face_locations(image, model="hog")
face_encoding5 = face_recognition.face_encodings(image, face_locations_ramlaghoo)[0]

image1 = face_recognition.load_image_file("H:/work/opencv/screenontime/actual/saira.png")
face_locations_saira = face_recognition.face_locations(image1, model="hog")
face_encoding6 = face_recognition.face_encodings(image1, face_locations_saira)[0]

image1 = face_recognition.load_image_file("H:/work/opencv/screenontime/actual/sulakshana.jpg")
face_locations_sulakshana = face_recognition.face_locations(image1, model="hog")
face_encoding7 = face_recognition.face_encodings(image1, face_locations_sulakshana)[0]

image1 = face_recognition.load_image_file("H:/work/opencv/screenontime/actual/vinod.jpg")
face_locations_vinod = face_recognition.face_locations(image1, model="hog")
face_encoding8 = face_recognition.face_encodings(image1, face_locations_vinod)[0]

image1 = face_recognition.load_image_file("H:/work/opencv/screenontime/actual/Yunus.jpg")
face_locations_yunus = face_recognition.face_locations(image1, model="hog")
face_encoding9 = face_recognition.face_encodings(image1, face_locations_yunus)[0]

#list of known faces from dataset
known_faces = [
face_encoding1, face_encoding2, face_encoding3, face_encoding4, face_encoding5, face_encoding6, face_encoding7, face_encoding8, face_encoding9,
]

face_locations = []
face_encodings = []

while True:
    ret, frame = input_movie.read()

    if not ret:
        break

    # Convert the image from BGR color to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known faces
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.60)

        #matching the output
        if match[0]:
            amitabh+=1
        if match[1]:
            mac+=1
        if match[2]:
            paidi+=1
        if match[3]:
            pinchoo+=1
        if match[4]:
            ramlaghoo+=1
        if match[5]:
            saira+=1
        if match[6]:
            sulakshana+=1
        if match[7]:
            vinod+=1
        if match[8]:
            yunus+=1

#calculating onscreen time using frame count and fps
amitabh=amitabh/fps
mac=mac/fps
paidi=paidi/fps
pinchoo=pinchoo/fps
ramlaghoo=ramlaghoo/fps
saira=saira/fps
sulakshana=sulakshana/fps
vinod=vinod/fps
yunus=yunus/fps

#Print the time
print("Approx time of {}: {} Second".format('Amitabh',amitabh))
print("Approx time of {}: {} Second".format('Vinod Khanna',vinod))
print("Approx time of {}: {} Second".format('Mac Mohan',mac))
print("Approx time of {}: {} Second".format('Paidi Jairaj',paidi))
print("Approx time of {}: {} Second".format('ShreeRam Laghoo',ramlaghoo))
print("Approx time of {}: {} Second".format('Saira Banu',saira))
print("Approx time of {}: {} Second".format('Pinchoo Kapoor',pinchoo))
print("Approx time of {}: {} Second".format('Sulakshana Pandit',sulakshana))
print("Approx time of {}: {} Second".format('Yunus Parvez',yunus))

#releasing all resources
input_movie.release()
cv2.destroyAllWindows()