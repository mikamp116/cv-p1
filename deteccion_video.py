import cv2
import deteccion_orb as orbdet
import deteccion_haar as haardet

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html


def haar_detection(video_source, scale_factor, min_neighbors):
    """Devuelve una lista con los frames con los frontales y matriculas detectados en coches mediante haar"""
    coches_cascade = cv2.CascadeClassifier('haar_opencv_4.1-4.2/coches.xml')
    matricula_cascade = cv2.CascadeClassifier('haar_opencv_4.1-4.2/matriculas.xml')

    cap = cv2.VideoCapture(video_source)

    gray_frames = []
    color_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            color_frames.append(frame)
        else:
            break

    frames = haardet.detect(gray_frames, color_frames, coches_cascade, matricula_cascade, scale_factor, min_neighbors)

    cap.release()
    return frames


def orb_detection(video_source, frame_rate, num_keypoints, scale_factor, pyramid_levels, knn_matches,
                  gaussian_kernel_sigma, debug):
    """Devuelve una lista con los frames con los centros de frontales detectados en coches mediante orb"""
    cap = cv2.VideoCapture(video_source)

    all_frames = []
    color_frames = []
    gray_frames = []
    output_frames = []
    count = -1

    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if ret == True:
            all_frames.append(frame)
            if count % frame_rate == 0:
                color_frames.append(frame)
                gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            break

    train_images = orbdet.load()
    orb = cv2.ORB_create(nfeatures=num_keypoints, scaleFactor=scale_factor, nlevels=pyramid_levels)
    match_table, flann = orbdet.train(train_images, orb)
    detected_points = orbdet.detect(gray_frames, orb, match_table, flann, knn_matches, gaussian_kernel_sigma, debug)

    count = -1
    point = [0, 0]
    for i in range(len(all_frames)):
        count += 1
        if count % frame_rate == 0:
            t = detected_points[int(i / frame_rate)]
            point = [t[0], t[1]]
            cv2.circle(color_frames[int(i / frame_rate)], (point[0], point[1]), 5, (255, 0, 0), thickness=4,
                       lineType=8, shift=0)

            output_frames.append(color_frames[int(i / frame_rate)])
        else:
            cv2.circle(all_frames[i], (point[0], point[1]), 5, (255, 0, 0), thickness=4, lineType=8, shift=0)
            output_frames.append(all_frames[i])

    cap.release()
    return output_frames


def show(video_source, frames, fps):
    # https: // www.geeksforgeeks.org/python-play-a-video-using-opencv/
    for frame in frames:
        cv2.imshow('output_' + video_source[:video_source.find('.')], frame)
        if cv2.waitKey(fps) & 0xFF == ord('q'):
            break


def main(video_source, frame_rate, num_keypoints, scale_factor, pyramid_levels, knn_matches, gaussian_kernel_sigma,
         min_neighbors, debug=0):
    fps = int(cv2.VideoCapture(video_source).get(5)) * 2
    haar_frames = haar_detection(video_source, scale_factor, min_neighbors)
    orb_frames = orb_detection(video_source, frame_rate, num_keypoints, scale_factor, pyramid_levels, knn_matches,
                               gaussian_kernel_sigma, debug)
    show(video_source, haar_frames, fps)
    show(video_source, orb_frames, fps)


if __name__ == "__main__":
    source = 'video2.wmv'
    NUM_KEYPOINTS = 100
    SCALE_FACTOR = 1.3
    PYRAMID_LEVELS = 4
    KNN_MATCHES = 6
    GAUSSIAN_KERNEL_SIGMA = 2
    FRAME_RATE = 10
    MIN_NEIGHBORS = 5

    main(source, FRAME_RATE, NUM_KEYPOINTS, SCALE_FACTOR, PYRAMID_LEVELS, KNN_MATCHES, GAUSSIAN_KERNEL_SIGMA,
         MIN_NEIGHBORS)
