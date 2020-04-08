import cv2
import deteccion_orb as orbdet


def main(video_source, frame_ratio, NUM_KEYPOINTS, SCALE_FACTOR, PYRAMID_LEVELS, KNN_MATCHES, GAUSSIAN_KERNEL_SIGMA, DEBUG=0):
    coches_cascade = cv2.CascadeClassifier('haar_opencv_4.1-4.2/coches.xml')
    matricula_cascade = cv2.CascadeClassifier('haar_opencv_4.1-4.2/matriculas.xml')

    cap = cv2.VideoCapture(video_source)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_haar_' + video_source[:6] + '.avi', fourcc, 20.0, (480, 270))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            coche = coches_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in coche:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                matricula = matricula_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in matricula:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 4)

            matricula = matricula_cascade.detectMultiScale(gray)
            for (ex, ey, ew, eh) in matricula:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

            # write the flipped frame
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()

    cap = cv2.VideoCapture(video_source)
    out = cv2.VideoWriter('output_orb_' + source[:6] + '.avi', fourcc, 20.0, (480, 270))

    all_frames = []
    frames = []
    grays = []
    count = -1

    while (cap.isOpened()):
        count += 1
        ret, frame = cap.read()
        if ret == True:
            all_frames.append(frame)
            if count % frame_ratio == 0:
                frames.append(frame)
                grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    train_images = orbdet.load()
    orb = cv2.ORB_create(nfeatures=NUM_KEYPOINTS, scaleFactor=SCALE_FACTOR, nlevels=PYRAMID_LEVELS)
    match_table, flann = orbdet.train(train_images, orb)
    detected_points = orbdet.detect(grays, orb, match_table, flann, KNN_MATCHES, GAUSSIAN_KERNEL_SIGMA, DEBUG)

    count = -1
    point = [0, 0]
    for i in range(len(all_frames)):
        count += 1
        if count % frame_ratio == 0:
            t = detected_points[int(i/frame_ratio)]
            point = [t[0], t[1]]
            cv2.circle(frames[int(i/frame_ratio)], (point[0], point[1]), 5, (255, 0, 0), thickness=4, lineType=8, shift=0)
            out.write(frames[int(i/frame_ratio)])
        else:
            cv2.circle(all_frames[i], (point[0], point[1]), 5, (255, 0, 0), thickness=4, lineType=8, shift=0)
            out.write(all_frames[i])

    # Release everything if job is finished
    cap.release()
    out.release()

if __name__ == "__main__":
    source = 'video1.wmv'
    NUM_KEYPOINTS = 100
    SCALE_FACTOR = 1.3
    PYRAMID_LEVELS = 4
    KNN_MATCHES = 6
    GAUSSIAN_KERNEL_SIGMA = 2
    FRAME_RATIO = 10

    main(source, FRAME_RATIO, NUM_KEYPOINTS, SCALE_FACTOR, PYRAMID_LEVELS, KNN_MATCHES, GAUSSIAN_KERNEL_SIGMA)