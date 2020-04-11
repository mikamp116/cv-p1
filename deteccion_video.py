import cv2
import deteccion_orb as orbdet
import deteccion_haar as haardet


def haar_detection(video_source, scale_factor, min_neighbors):
    coches_cascade = cv2.CascadeClassifier('haar_opencv_4.1-4.2/coches.xml')
    matricula_cascade = cv2.CascadeClassifier('haar_opencv_4.1-4.2/matriculas.xml')

    cap = cv2.VideoCapture(video_source)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter('output_haar_' + video_source[:6] + '.avi', fourcc, 20.0, (width, height))

    gray_frames = []
    color_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            color_frames.append(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    images = haardet.detect(gray_frames, color_frames, coches_cascade, matricula_cascade, scale_factor, min_neighbors)
    for frame in images:
        out.write(frame)

    cap.release()
    out.release()


def orb_detection(video_source, frame_ratio, num_keypoints, scale_factor, pyramid_levels, knn_matches,
                  gaussian_kernel_sigma, debug):
    cap = cv2.VideoCapture(video_source)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(3))
    height = int(cap.get(4))
    cap = cv2.VideoCapture(video_source)
    out = cv2.VideoWriter('output_orb_' + video_source[:6] + '.avi', fourcc, 20.0, (width, height))

    all_frames = []
    color_frames = []
    gray_frames = []
    count = -1

    while (cap.isOpened()):
        count += 1
        ret, frame = cap.read()
        if ret:
            all_frames.append(frame)
            if count % frame_ratio == 0:
                color_frames.append(frame)
                gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
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
        if count % frame_ratio == 0:
            t = detected_points[int(i / frame_ratio)]
            point = [t[0], t[1]]
            cv2.circle(color_frames[int(i / frame_ratio)], (point[0], point[1]), 5, (255, 0, 0), thickness=4,
                       lineType=8, shift=0)
            out.write(color_frames[int(i / frame_ratio)])
            # Display the resulting frame
            cv2.imshow(video_source, color_frames[int(i / frame_ratio)])

            # Press Q on keyboard to  exit
            if cv2.waitKey(int(cap.get(5))) & 0xFF == ord('q'):
                break
        else:
            cv2.circle(all_frames[i], (point[0], point[1]), 5, (255, 0, 0), thickness=4, lineType=8, shift=0)
            out.write(all_frames[i])
            # Display the resulting frame
            cv2.imshow(video_source, all_frames[i])

            # Press Q on keyboard to  exit
            if cv2.waitKey(int(cap.get(5))) & 0xFF == ord('q'):
                break

    # Release everything if job is finished
    cap.release()
    out.release()


def main(video_source, frame_ratio, num_keypoints, scale_factor, pyramid_levels, knn_matches, gaussian_kernel_sigma,
         min_neighbors, debug=0):
    haar_detection(video_source, scale_factor, min_neighbors)
    orb_detection(video_source, frame_ratio, num_keypoints, scale_factor, pyramid_levels, knn_matches,
                  gaussian_kernel_sigma, debug)


if __name__ == "__main__":
    source = 'video2.wmv'
    NUM_KEYPOINTS = 100
    SCALE_FACTOR = 1.3
    PYRAMID_LEVELS = 4
    KNN_MATCHES = 6
    GAUSSIAN_KERNEL_SIGMA = 2
    FRAME_RATIO = 10
    MIN_NEIGHBORS = 5

    main(source, FRAME_RATIO, NUM_KEYPOINTS, SCALE_FACTOR, PYRAMID_LEVELS, KNN_MATCHES, GAUSSIAN_KERNEL_SIGMA, MIN_NEIGHBORS)