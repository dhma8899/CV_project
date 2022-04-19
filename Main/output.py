import cv2


def create_video(frames, size):
    output_video = cv2.VideoWriter('video_with_structures.avi', cv2.VideoWriter_fourcc(*'DIVX'), 16.0, size)

    for i in range(len(frames)):
        output_video.write(frames[i])
    output_video.release()
