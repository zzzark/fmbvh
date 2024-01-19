import cv2 as cv
import os
import re


def images_to_video(image_folder, output_video_file, fps=60) -> None:
    """
    :param image_folder: a folder that contains images indexed as: 0000.png, 0001.png, ...
    :param output_video_file: 
    :param fps: 
    :return: None
    """
    if not os.path.isdir(image_folder):
        images = []
    else:
        images = sorted([img for img in os.listdir(image_folder) if re.match(r'\d{4}\.png', img)],
                        key=lambda x: int(x.split('.')[0]))

    if not images:
        raise ValueError("No images found in the folder.")

    # Retrieve the dimensions of the first image
    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or 'MJPG'
    video = cv.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv.imread(img_path)
        video.write(frame)  # Write out frame to video

    video.release()  # Release the video writer


def main():
    base_folder = r'D:\projects\SMPLX_UnityProject_20210617\SMPLX-Unity\Output\SavedImages'
    session = "vid_lpmd_M003692_1"

    image_folder_path = os.path.join(base_folder, session)
    output_video_path = f'{session}.mp4'
    images_to_video(image_folder_path, output_video_path, fps=60)


if __name__ == "__main__":
    main()
