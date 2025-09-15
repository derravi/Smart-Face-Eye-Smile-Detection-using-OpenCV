import cv2 

face_cascade = cv2.CascadeClassifier("E:\Python\Python All Projects\Smart Face, Eye & Smile Detection using OpenCV\haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("E:\Python\Python All Projects\Smart Face, Eye & Smile Detection using OpenCV\haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier("E:\Python\Python All Projects\Smart Face, Eye & Smile Detection using OpenCV\haarcascade_eye.xml")

cap = cv2.VideoCapture(0) 

while True:

    ret,frame = cap.read()

    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_image,1.1,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,"Face_detected",(x,y-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        print("Face_Detected..........")
    
    roi_gray = frame[y:y+h,x:x+w]
    roi_color = gray_image[y:y+h,x:x+w]

    eye = eye_cascade.detectMultiScale(roi_gray,1.1,10)
    if len(eye) > 0:
        cv2.putText(frame,"Eye_detected",(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        print("Eye_detected")
    
    smile = smile_cascade.detectMultiScale(roi_gray,1.7,20)
    if len(smile) > 0:
        cv2.putText(frame,"Smil_detected",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        print("Smile_detected")

    cv2.imshow("Smart Face Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quitinggg.............")
        break

cap.release()
cv2.destroyAllWindows()