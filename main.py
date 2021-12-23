from flask import Flask
from flask import render_template
# from flask import url_for
from flask import request
from flask import Response
import time
#********
import cv2 as cv
import mediapipe as mp
import numpy as np
import imutils


SRC = 0

camera = cv.VideoCapture(SRC)
app = Flask(__name__)

#*******************UTILITIES***********************
def show(image):
    cv.imshow('frame',image)
    cv.waitKey(0)
def rescale(image,factor=.3):
    height,width,_ = image.shape
    height = int( height * factor )
    width = int( width * factor )
    rescaled = cv.resize(image,(width,height))
    return rescaled    
def distance_between_points(a,b):
    distance = np.sqrt( (b[0]-a[0])**2 + (b[1]-a[1])**2 )
    return int(distance)
def get_bb(result,image_shape):
    ih,iw,ic = image_shape
    faces = []
    extend = 30
    if result.detections:
        for detection in result.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int( bbox.xmin * iw) - extend + 10
            y = int( bbox.ymin * ih) - extend #+ 10
            w = int( bbox.width * iw) + extend
            h = int( bbox.height * ih) + extend
            faces.append([x,y,w,h])
    return np.array(faces)
#******************PREBUILD UTILITIES*************************
face_detection_model_name = 'haarcascade_frontalface_alt2.xml'
landmark_detection_model_name = 'lbfmodel.yaml'

landmark_detector = cv.face.createFacemarkLBF()
landmark_detector.loadModel(landmark_detection_model_name)

face_detector = cv.CascadeClassifier(face_detection_model_name)

mpFace = mp.solutions.face_detection
facemp = mpFace.FaceDetection()
#******************FILTER IMAGES******************************
#****thug life******
t_glass = cv.imread('filters/thug_glass2.jpg')
t_glass = rescale(t_glass,.6)
joint = cv.imread('filters/joint.jpg')
joint = rescale(joint,.5)
#****party******
whistle = cv.imread('filters/wisle.png')
hat = cv.imread('filters/hat.jpg')
glass = cv.imread('filters/glass.jpg')
#*****swag*****
s_glass = cv.imread('filters/swag1.jpg')
s_glass = rescale(s_glass,.5)
moust = cv.imread('filters/moust.jpg')
moust = rescale(moust,.5)
#******************FILTER FUNCTIONS****************************
def thug_life(image):
    grey_img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    rgb_img = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    result = facemp.process(rgb_img)
    faces = get_bb(result,image.shape)
    for face in faces:
        landmarks = landmark_detector.fit(grey_img,face.reshape(1,4))
        #****************Calculating Face Orientation************
        nose_line_pt1 = landmarks[1][0][0][27]
        nose_line_pt2 = landmarks[1][0][0][30]
        delta_y = nose_line_pt2[1] - nose_line_pt1[1]
        delta_x = nose_line_pt2[0] - nose_line_pt1[0]
        orientation = np.arctan2(delta_y,delta_x) * 180 / np.pi
        if orientation < 0:
            orientation = orientation + 90
        else:
            orientation = orientation - 90
        ################# G L A S S ####################################
        #*************resizing filter according to face****************
        left_eye_brw = np.mean(landmarks[1][0][0][17:19][:,0])-18,np.mean(landmarks[1][0][0][17:19][:,1])
        right_eye_brw = landmarks[1][0][0][26]
        eye_width = int(distance_between_points(left_eye_brw,right_eye_brw)*1.1) 
        glass_resized = imutils.resize(t_glass,width=eye_width)
       #glass_resized = imutils.rotate_bound(glass_resized,orientation) #<--- manipulate here
        glass_height = glass_resized.shape[0]
        glass_width = glass_resized.shape[1]
        #*********************finding region of interest***************
        glass_roi = image[int(left_eye_brw[1])+3:int(left_eye_brw[1])+glass_height+3,int(left_eye_brw[0])+3:int(left_eye_brw[0])+glass_width+3]

        #**********************Creating Mask for filter****************
        glass_grey = cv.cvtColor(glass_resized,cv.COLOR_BGR2GRAY)
        _,glass_mask = cv.threshold(glass_grey,200,255,cv.THRESH_BINARY_INV)
        glass_mask_inv = cv.bitwise_not(glass_mask)
        #**************************creating roi fg and bg*************
        glass_bg = cv.bitwise_and(glass_roi,glass_roi,mask=glass_mask_inv)
        glass_fg = cv.bitwise_and(glass_resized,glass_resized,mask=glass_mask)
        glass_filled = cv.add(glass_bg,glass_fg)
        #######################J O I N T###############################
        left_lip = landmarks[1][0][0][48]
        right_lip = landmarks[1][0][0][54]
        lip_dist = distance_between_points(left_lip,right_lip)
        joint_resized = imutils.resize(joint,width = lip_dist)
        #****roi****
        j_width = joint_resized.shape[1]
        j_height = joint_resized.shape[0]
        j_roi = image[int(left_lip[1]):int(left_lip[1])+j_height,int(left_lip[0])-j_width+20:int(left_lip[0])+20]

        #****masking*******
        j_grey = cv.cvtColor(joint_resized,cv.COLOR_BGR2GRAY)
        _,j_mask = cv.threshold(j_grey,1,255,cv.THRESH_BINARY)
        j_mask_inv = cv.bitwise_not(j_mask)
        #******bg and fg
        j_bg = cv.bitwise_and(j_roi,j_roi,mask=j_mask_inv)
        j_filled = cv.add(j_bg,joint_resized)
        #**********************Adding Filter*******************
        image[int(left_eye_brw[1])+3:int(left_eye_brw[1])+glass_height+3,int(left_eye_brw[0])+3:int(left_eye_brw[0])+glass_width+3] = glass_filled
        image[int(left_lip[1]):int(left_lip[1])+j_height,int(left_lip[0])-j_width+20:int(left_lip[0])+20] = j_filled
    return image
def glass_filter(image):
    grey_img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    rgb_img = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    result = facemp.process(rgb_img)
    faces = get_bb(result,image.shape)
    for face in faces:
        landmarks = landmark_detector.fit(grey_img,face.reshape(1,4))
        #****************Calculating Face Orientation************
        nose_line_pt1 = landmarks[1][0][0][27]
        nose_line_pt2 = landmarks[1][0][0][30]
        delta_y = nose_line_pt2[1] - nose_line_pt1[1]
        delta_x = nose_line_pt2[0] - nose_line_pt1[0]
        orientation = np.arctan2(delta_y,delta_x) * 180 / np.pi
        if orientation < 0:
            orientation = orientation + 90
        else:
            orientation = orientation - 90
        ################# H A T ###############################################
        #*************resizing filter according to face****************
        left_pt = landmarks[1][0][0][19]
        right_pt = landmarks[1][0][0][24]
        hat_width = distance_between_points(left_pt,right_pt) 
        hat_resized = imutils.resize(hat,width=hat_width)
        #*****************Finding Roi***************************************
        hat_height = hat_resized.shape[0]
        hat_roi = image[int(left_pt[1])-hat_height-60:int(left_pt[1])-60,int(left_pt[0]):int(left_pt[0])+hat_width]
        #*****************Creating Mak*****************************
        hat_grey = cv.cvtColor(hat_resized,cv.COLOR_BGR2GRAY)
        _,hat_mask = cv.threshold(hat_grey,10,255,cv.THRESH_BINARY)
        hat_mask_inv = cv.bitwise_not(hat_mask)
        #******************creating fg and bg******************
        hat_bg = cv.bitwise_and(hat_roi,hat_roi,mask=hat_mask_inv)
        hat_filled = cv.add(hat_bg,hat_resized)
        ################# G L A S S ##############################################
        #*************resizing filter according to face****************
        left_eye_brw = np.mean(landmarks[1][0][0][17:19][:,0])-4,np.mean(landmarks[1][0][0][17:19][:,1])
        right_eye_brw = landmarks[1][0][0][26]
        eye_width = distance_between_points(left_eye_brw,right_eye_brw) 
        glass_resized = imutils.resize(glass,width=eye_width)
    #         glass_resized = imutils.rotate_bound(glass_resized,orientation) #<--- manipulate here
        glass_height = glass_resized.shape[0]
        glass_width = glass_resized.shape[1]
        #*********************finding region of interest***************
        glass_roi = image[int(left_eye_brw[1])-6:int(left_eye_brw[1])+glass_height-6,int(left_eye_brw[0])-4:int(left_eye_brw[0])+glass_width-4]

        #**********************Creating Mask for filter****************
        glass_grey = cv.cvtColor(glass_resized,cv.COLOR_BGR2GRAY)
        _,glass_mask = cv.threshold(glass_grey,1,255,cv.THRESH_BINARY_INV)
        _,glass_mask_inv = cv.threshold(glass_grey,1,255,cv.THRESH_BINARY)
        #**************************creating roi fg and bg*************
        glass_bg = cv.bitwise_and(glass_roi,glass_roi,mask=glass_mask)
        glass_fg = cv.bitwise_and(glass_resized,glass_resized,mask=glass_mask_inv)
        t_glass_filled = cv.add(glass_bg,glass_fg)
        ##################### W H I S T L E ########################################
        left_lip = landmarks[1][0][0][48]
        right_lip = landmarks[1][0][0][54]
        lip_dist = distance_between_points(left_lip,right_lip)
        whistle_resized = imutils.resize(whistle,width = lip_dist)
        #****roi****
        w_width = whistle_resized.shape[1]
        w_height = whistle_resized.shape[0]
        w_roi = image[int(left_lip[1])-6:int(left_lip[1])+w_height-6,int(left_lip[0])-w_width+20:int(left_lip[0])+20]
        
        #****masking*******
        w_grey = cv.cvtColor(whistle_resized,cv.COLOR_BGR2GRAY)
        _,w_mask = cv.threshold(w_grey,1,255,cv.THRESH_BINARY)
        w_mask_inv = cv.bitwise_not(w_mask)
        #******bg and fg
        try:
            w_bg = cv.bitwise_and(w_roi,w_roi,mask=w_mask_inv)
            w_fg = cv.bitwise_and(whistle_resized,whistle_resized,mask=w_mask)
            w_filled = cv.add(w_bg,w_fg)
            #*********************** ADDING FILTERS **********************************************
            image[int(left_eye_brw[1])-6:int(left_eye_brw[1])+glass_height-6,int(left_eye_brw[0])-4:int(left_eye_brw[0])+glass_width-4] = t_glass_filled    
            image[int(left_lip[1])-6:int(left_lip[1])+w_height-6,int(left_lip[0])-w_width+20:int(left_lip[0])+20] = w_filled
            image[int(left_pt[1])-hat_height-60:int(left_pt[1])-60,int(left_pt[0]):int(left_pt[0])+hat_width] = hat_filled
        except:
            pass
    return image
def swag(image):    
    grey_img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    rgb_img = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    result = facemp.process(rgb_img)
    faces = get_bb(result,image.shape)
    for face in faces:
        landmarks = landmark_detector.fit(grey_img,face.reshape(1,4))
        #****************Calculating Face Orientation************
        nose_line_pt1 = landmarks[1][0][0][27]
        nose_line_pt2 = landmarks[1][0][0][30]
        delta_y = nose_line_pt2[1] - nose_line_pt1[1]
        delta_x = nose_line_pt2[0] - nose_line_pt1[0]
        orientation = np.arctan2(delta_y,delta_x) * 180 / np.pi
        if orientation < 0:
            orientation = orientation + 90
        else:
            orientation = orientation - 90
         ################# G L A S S ####################################
        #*************resizing filter according to face****************
        left_eye_brw = np.mean(landmarks[1][0][0][17:19][:,0])-18,np.mean(landmarks[1][0][0][17:19][:,1])
        right_eye_brw = landmarks[1][0][0][26]
        eye_width = int(distance_between_points(left_eye_brw,right_eye_brw)*1.1) 
        glass_resized = imutils.resize(s_glass,width=eye_width)
        glass_height = glass_resized.shape[0]
        glass_width = glass_resized.shape[1]
        #*********************finding region of interest***************
        glass_roi = image[int(left_eye_brw[1])-5:int(left_eye_brw[1])+glass_height-5,int(left_eye_brw[0])+3:int(left_eye_brw[0])+glass_width+3]
        #**********************Creating Mask for filter****************
        glass_grey = cv.cvtColor(glass_resized,cv.COLOR_BGR2GRAY)
        _,glass_mask = cv.threshold(glass_grey,10,255,cv.THRESH_BINARY)
        glass_mask_inv = cv.bitwise_not(glass_mask)
        #************Adding some color*******************************
        canny = cv.Canny(glass_grey,20,170)
        contours,_ = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
        color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        skipper = np.random.randint(4,10)
        cont = np.array(contours)[::skipper]
        glass_resized = cv.drawContours(glass_resized,cont,-1,color,2)
        #**************************creating roi fg and bg*************
        glass_bg = cv.bitwise_and(glass_roi,glass_roi,mask=glass_mask_inv)
        glass_fg = cv.bitwise_and(glass_resized,glass_resized,mask=glass_mask)
        glass_filled = cv.add(glass_bg,glass_fg)
        ##################### Meesha ##################
        #******* resizing the filter*********
        left_lip = landmarks[1][0][0][48]
        right_lip = landmarks[1][0][0][54]
        lip_dist = distance_between_points(left_lip,right_lip)
        moust_resized = imutils.resize(moust,width = lip_dist)
        #*********roi*******
        m_width = moust_resized.shape[1]
        m_height = moust_resized.shape[0]
        m_roi = image[int(left_lip[1])-m_height:int(left_lip[1]),int(left_lip[0]):int(left_lip[0])+m_width]
        #****masking*******
        m_grey = cv.cvtColor(moust_resized,cv.COLOR_BGR2GRAY)
        _,m_mask = cv.threshold(m_grey,10,255,cv.THRESH_BINARY)
        m_mask_inv = cv.bitwise_not(m_mask)
        #******bg and fg
        m_bg = cv.bitwise_and(m_roi,m_roi,mask=m_mask_inv)
        m_filled = cv.add(m_bg,moust_resized)
        ############ADDING FILTERS##################################3
        image[int(left_eye_brw[1])-5:int(left_eye_brw[1])+glass_height-5,int(left_eye_brw[0])+3:int(left_eye_brw[0])+glass_width+3]=glass_filled
        image[int(left_lip[1])-m_height:int(left_lip[1]),int(left_lip[0]):int(left_lip[0])+m_width] = m_filled
    return image    
#**************************************************************
filters = [thug_life,glass_filter,swag]
#**************************************************************
@app.route('/')
def homepage():
    return render_template('homepage.html')
@app.route('/requests',methods=['GET','POST'])
def filter():
    global filter_no
    if request.form.get('camera_source1') == 'c1':
        SRC = 0
    elif request.form.get('camera_source2') == 'c2':
        SRC = 1    
    if request.form.get('Filter 1') == 'Thug Life':
        filter_no = 0
    elif request.form.get('Filter 2') == 'The Party':
        filter_no = 1
    elif request.form.get('Filter 3') == 'Be Swag':
        filter_no = 2
    return render_template('f1.html')

def generate_frames():
    ptime = 0
    while True:
        success,frame = camera.read()
        if success:
            frame = cv.flip(frame,1)
            frame = filters[filter_no](frame)
            ctime = time.time()
            fps = str( int(1/(ctime-ptime)) )
            frame = cv.putText(frame,'fps:'+fps,(10,40),cv.FONT_HERSHEY_COMPLEX,1,(0,233,0),1)
            ptime = ctime
            ret,buffer = cv.imencode('.jpg',frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
        else:
            pass

@app.route('/camera')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug = True)