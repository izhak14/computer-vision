import collections
import numpy as np
from PIL import Image
import matplotlib.pyplot as pyplot



class DigitalImaging:

    def convert_to_gs(image_path):

        img_obj = Image.open(image_path)
        gray_img = img_obj.convert('L')
        print(gray_img.mode)
        return gray_img

    def color_at(img_arr,row_num,column_num):
        if img_arr.flags.writeable and DigitalImaging.validate_coordinates(img_arr,row_num,column_num):
            r,g,b=img_arr[row_num][column_num]
            return r,g,b

    def validate_coordinates(img_arr,row_num,column_num):
        if isinstance(img_arr,np.ndarray):
            num_of_rows, num_of_columns, channels = img_arr.shape # validate bounds
            if isinstance(row_num,int) and isinstance(column_num,int):
                if 0 <= row_num < num_of_rows:
                    if 0 <= column_num < num_of_columns:
                        return True
                else:
                    raise ValueError("row number not valid")
            else:
                raise ValueError("row/column not of int type")
        else:
            raise ValueError("img_arr is not a numpy array")

    def reduce_to(image_path, char):
        img_obj = Image.open(image_path)
        img_arr = np.array(img_obj)
        if char.upper()=='R':
            img_arr[:, :, (1, 2)] = 0
            return  Image.fromarray(img_arr)
        if char.upper()=='G':
            img_arr[:, :, (0, 2)] = 0
            return  Image.fromarray(img_arr)
        if char.upper()=='B':
            img_arr[:, :, (0, 1)] = 0
            return  Image.fromarray(img_arr)

    def make_collage(images_arr):
        images_np=[np.array(img) for img in images_arr]
        count=1
        char='r'
        for img in images_np:
            if count>3:
                count=1
                if char=='r':
                    char='g'
                elif char=='g':
                    char='b'
                elif char=='b':
                    char='r'
            if char=='r':
                img[:, :, (1, 2)] = 0
            if char == 'g':
                img[:, :, (0, 2)] = 0
            if char == 'b':
                img[:, :, (0, 1)] = 0
            count+=1

        gallery = np.hstack(images_np)
        Image.fromarray(gallery).show()

    def shapes_dict(images_arr):
        d={}
        for img in images_arr:
            d[np.array(img).shape[0]]=np.array(img).shape
        d2=dict(collections.OrderedDict(sorted(d.items())))
        return d2

    def show_image(name):
        while True:
            cv2.imshow('image', name)
            key_pressed = cv2.waitKey(0)
            # if key_pressed & 27: # by default
            if key_pressed & ord('q'):  # q character is pressed
                break;
        # cv2.destroyWindow('image') # release image window resources
        cv2.destroyAllWindows()

    def detect_obj(image_path,name):
        if name.lower()=='face':
            face_classifier = \
                cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_frontalface_default.xml')

        elif name.lower()=='eyes':
            face_classifier = \
                cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_eye.xml')
        else:
            print("not Valid")
        face = face_classifier.detectMultiScale(image_path)

        for (_x, _y, _w, _h) in face:
            cv2.rectangle(image_path,
                          (_x, _y),  # upper-left corner
                          (_x + _w, _y + _h),  # lower-right corner
                          (0, 255, 0),
                          2)

        DigitalImaging.show_image(image_path)

    def detect_obj_adv(image_path,face,eyes):

        if face:
            face_classifier = \
                cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_frontalface_default.xml')
            face = face_classifier.detectMultiScale(image_path)

            for (_x, _y, _w, _h) in face:
                cv2.rectangle(image_path,
                              (_x, _y),  # upper-left corner
                              (_x + _w, _y + _h),  # lower-right corner
                              (0, 255, 0),
                              2)

        if eyes:
            face_classifier = \
                cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_eye.xml')
            face = face_classifier.detectMultiScale(image_path)
            for (_x, _y, _w, _h) in face:
                cv2.rectangle(image_path,
                          (_x, _y),  # upper-left corner
                          (_x + _w, _y + _h),  # lower-right corner
                          (0, 255, 0),
                          2)



        DigitalImaging.show_image(image_path)

    def detect_face_in_vid(cap):
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            face_classifier = \
                cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_frontalface_default.xml')
            face = face_classifier.detectMultiScale(frame)

            for (_x, _y, _w, _h) in face:
                cv2.rectangle(frame,
                              (_x, _y),  # upper-left corner
                              (_x + _w, _y + _h),  # lower-right corner
                              (0, 255, 0),
                              2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    img =Image.open('image.jpg')
    img_arr=np.array(img)
    img_grey =DigitalImaging.convert_to_gs('image.jpg')
    # img.show()
    # img_grey.show()

    print(DigitalImaging.color_at(img_arr,0,0))

    img_color= DigitalImaging.reduce_to('image.jpg','b')
    # img_color.show()
    img2=Image.open('image2.jpg')
    img3=Image.open('image3.jpg')
    images =[img,img,img,img,img,img,img,img,img,img,img]
    images2= [img2,img,img3]
    # DigitalImaging.make_collage(images)

    DigitalImaging.shapes_dict(images2)

    cast = cv2.imread('face.jpg')
    # DigitalImaging.detect_obj(cast,'face')
    # DigitalImaging.detect_obj_adv(cast,True,True)

    cap = cv2.VideoCapture('vtest.mp4')

    DigitalImaging.detect_face_in_vid(cap)
