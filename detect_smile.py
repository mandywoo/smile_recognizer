from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

NO_TEETH_SMILE_RATIO = 0.38
TEETH_SMILE_RATIO = 0.5
UPPER_EH = .55

IMAGE_TAKEN = False

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--shape-predictor', required=True, help='path to facial landmark predictor')
ap.add_argument('-i', '--image-path', help='path you want image to be saved in')
args = vars(ap.parse_args())

def rotateImage(image, angle):
    row,col,_ = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
	"""
	@brief      Overlays a transparant PNG onto another image using CV2
	
	@param      background_img    The background image
	@param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
	@param      x                 x location to place the top-left corner of our overlay
	@param      y                 y location to place the top-left corner of our overlay
	@param      overlay_size      The size to scale our overlay to (tuple), no scaling if None
	
	@return     Background image with overlay on top
	"""
	
	bg_img = background_img.copy()
	
	if overlay_size is not None:
		img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

	# Extract the alpha mask of the RGBA image, convert to RGB 
	b,g,r,a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b,g,r))
	
	# Apply some simple filtering to remove edge noise
	mask = cv2.medianBlur(a,1)

	h, w, _ = overlay_color.shape
	roi = bg_img[y:y+h, x:x+w]

	# Black-out the area behind the logo in our original ROI
	img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
	
	# Mask out the logo from the logo image.
	img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	# Update the original image with our new ROI
	bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

	return bg_img

# calculate mouth aspect ratio to determine smile
def smile_ratio(mouth):
    L = dist.euclidean(mouth[2], mouth[10]) + dist.euclidean(mouth[3], mouth[9]) + dist.euclidean(mouth[4], mouth[8])
    D = dist.euclidean(mouth[0], mouth[6])
    mar = L/(3*D)
    return mar

def check_aligned(right_eye_pts, left_eye_pts) -> bool:
    left_eye_center = left_eye_pts.mean(axis=0).astype('int')
    right_eye_center = right_eye_pts.mean(axis=0).astype('int')

    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]

    angle = np.degrees(np.arctan2(dY, dX)) - 180

    if angle == 0:
        return True
    return False

# initialize dlibs frontal face detector
print('[INFO] loading facial landmark preditor...')
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(args['shape_predictor'])

# grab eye points
(start_r_eye, end_r_eye) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(start_l_eye, end_l_eye) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']

# grab mouth points 49-68
(start_m, end_m) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

# start video stream
vs = VideoStream(src=0).start()
time.sleep(1.0)



while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in grayscale
    rects = detector(gray, 0)

    # loop over facial detections
    for rect in rects:
        # get facial landmarks
        shape = shape_predictor(gray, rect)
        # convert (x, y) coordinates to np array
        shape = face_utils.shape_to_np(shape)

        # get mouth coordinates
        mouth = shape[start_m:end_m]
        # get first coordinate of right eye
        right_eye = shape[start_r_eye:end_r_eye]
        right_eye_37 = right_eye[0]
        left_eye = shape[start_l_eye:end_l_eye]
        left_eye_coord_45 = left_eye[2]
        left_eye_coord_46 = left_eye[3]
        left_eye_coord_47 = left_eye[4]

        # cv2.circle(frame, tuple(left_eye_coord_47), 3, (0, 255, 0), -1)
        # print(right_eye_coord[0], right_eye_coord[1])

        # calculate mouth aspect ratio
        mar = smile_ratio(mouth)

        # compute convex hull for mouth
        # convex hull: smallest shape that contains all the (x, y)-coordinate points of the mouth in it
        mouth_shape = cv2.convexHull(mouth)

        # draw mouth contour
        cv2.drawContours(frame, [mouth_shape], -1, (0, 255, 0), 1)

        if mar < NO_TEETH_SMILE_RATIO:
            smile = 'no teeth smile'
        elif mar > TEETH_SMILE_RATIO and mar < UPPER_EH:
            smile = 'teeth smile'
        else:
            smile = 'ehh'



        # overlay image if there's a smile
        if 'smile' in smile:
            img_cigarette = cv2.imread('smile-detection/thug_life_cigarette.png', -1)
            img_cigarette_length = int(dist.euclidean(mouth[0], mouth[6]))
            img_cigarette = imutils.resize(img_cigarette, width=img_cigarette_length)

            img_glasses = cv2.imread('smile-detection/thug_life_glasses.png', -1)
            img_glasses_length = int(dist.euclidean(right_eye_37, left_eye_coord_46) + 80)

            img_glasses = imutils.resize(img_glasses, width=img_glasses_length)
            height, width, channels = img_glasses.shape 
        
            # frame = overlay_transparent(frame, img_glasses, right_eye_37[0] - 37, right_eye_37[1] - 50)
            frame = overlay_transparent(frame, img_cigarette, mouth[15][0] - 5, mouth[15][1])
           
            # capture image
            if check_aligned(right_eye, left_eye) and not IMAGE_TAKEN:
                while True:
                    frame = overlay_transparent(frame, img_glasses, right_eye_37[0] - 37, right_eye_37[1] - 40)
                    cv2.imshow('Captured Image', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('y'):
                        # cv2.imwrite('smile-detection/img_1.png', frame)
                        if args['image_path']:
                            cv2.imwrite(args['image_path'], frame)
                        IMAGE_TAKEN = True
                        cv2.destroyAllWindows()
                        break  
                    elif key == ord('n'):
                        cv2.destroyAllWindows()
                        break  

        # write text
        # mar
        cv2.putText(frame, 'MAR: {}'.format(str(mar)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # smile
        cv2.putText(frame, smile, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # cv2.imshow('Frame',overlay_transparent(frame, overlay_t, 0, 0, (200,200)))
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or IMAGE_TAKEN:
        break

cv2.destroyAllWindows()
vs.stop()





