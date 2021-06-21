import cv2
import os
import shutil
from colorizor.cnn_colorizor import colorize as deep_colorize

def colorize(vid_path, output_path = 'tmp', ckpt_path = './colorizor/model/model_imagenet.ckpt', dvp_path = './colorizor/deep-video-prior'):
    if (not os.path.isdir(output_path)):
        os.mkdir(output_path)

    processed_path = "{}/processed".format(output_path)
    original_path = "{}/original".format(output_path)
    result_path = "{}/result".format(output_path)

    if (os.path.isdir(processed_path)):
        shutil.rmtree(processed_path)

    if (os.path.isdir(original_path)):
        shutil.rmtree(original_path)

    if (os.path.isdir(result_path)):
        shutil.rmtree(result_path)
    
    os.mkdir(processed_path)
    os.mkdir(original_path)
    os.mkdir(result_path)
    
    vid = cv2.VideoCapture(vid_path)
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    vid_h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    rval = vid.isOpened()
    cnt = 0
    while (rval):
        rval, frame = vid.read()
        if (rval):
            cv2.imwrite("{}/frame{:04d}.png".format(original_path, cnt), frame)
            colorized_img = deep_colorize(frame, ckpt_path, './colorizor/model/pts_in_hull.npy')
            cv2.imwrite("{}/frame{:04d}.png".format(processed_path, cnt), colorized_img)
        cnt += 1
    
    pwd = os.getcwd()
    os.chdir(dvp_path)
    
    os.system('python dvp_video_consistency.py --max_epoch 50 --input {} --processed {} --task colorization --with_IRT 1 --IRT_initialization 1 --output {}'
                .format(os.path.join(pwd, original_path),
                os.path.join(pwd, processed_path),
                os.path.join(pwd, result_path)))
    
    os.chdir(pwd)

    cnt = 485
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread("{}/0050/out_main_{:05d}.png".format(result_path, 0))
    
    vid_w = frame.shape[1]
    vid_h = frame.shape[0]

    videowriter = cv2.VideoWriter('output.mp4', fourcc, round(vid_fps), (round(vid_w), round(vid_h)))

    for i in range(cnt):
        frame = cv2.imread("{}/0050/out_main_{:05d}.png".format(result_path, i))
        videowriter.write(frame)
        
    videowriter.release()

    shutil.rmtree(original_path)
    shutil.rmtree(processed_path)
    shutil.rmtree(result_path)
    os.remove('./colorizor/deep-video-prior/result/colorization/commandline_args.txt')

