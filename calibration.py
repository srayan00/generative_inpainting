import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import torch
import os
from torchvision import transforms
from PIL import Image
import utils_csi


from inpaint_model import InpaintCAModel

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='output_test', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='model_logs', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--image_dir', default='/nfs/turbo/lsa-tewaria/places2/test_256', type=str,
                    help='The directory of image directory')
parser.add_argument(
    '--image_height', default=256, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--image_width', default=256, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument('--return_image', default=False, type=bool,
                    help='Whether to return the image or not')
parser.add_argument('--n_calibration', default=10, type=int,
                    help='number of calibration points')
parser.add_argument('--alpha', default=0.05, type=int,
                    help='alpha value for calibration')
parser.add_argument('--calib_file', default='output_test/calibration_output.txt', type=str,
                    help='caiibration info file')


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    args, unknown = parser.parse_known_args()

    # Getting the inception model
    inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    print(inception.eval())

    # Preprocessing model for the inception model
    preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Starting the session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    # Loading the model
    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width*2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    # Extracting all the calibration and test images
    dirs = os.listdir(args.image_dir)

    # Picking random 10000 images for calibration
    np.random.seed(0)
    n_calib = args.n_calibration
    print((n_calib + 1)*(1 - args.alpha)/n_calib)

    calibration_dirs = np.random.choice(dirs, n_calib, replace=False)
    calibration_scores = np.zeros(n_calib)
    for dir in calibration_dirs:
        img_path = os.path.join(args.image_dir, dir)
        image = cv2.imread(img_path)
        mask = utils_csi.random_bbox()
        unmasked_image = image 
        image = unmasked_image * ((256 - mask)/256)
        actual_image = Image.fromarray(np.uint8(unmasked_image))

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        # Getting Calibration scores
        n_gen_samples = 10
        original_inception_score = utils_csi.get_inception_score(actual_image, preprocess, inception)
        y_hats, inception_scores = utils_csi.get_multiple_samples_and_inception(sess, input_image, preprocess, 
                                                                                inception, n_gen_samples, output, input_image_ph)
        calibration_scores[calibration_dirs.tolist().index(dir)] = utils_csi.calibration_score_v2(inception_scores, original_inception_score)


        # Write the image to the output directory
        if args.return_image:
            result = sess.run(output, feed_dict={input_image_ph: input_image})
            path = os.path.join(args.output_dir, dir[:-4]+ "_output.png")
            cv2.imwrite(path, result[0][:, :, ::-1])

    # Calculating the score quantile
    score_quantile = np.quantile(calibration_scores, (n_calib + 1)*(1 - args.alpha)/n_calib)

    # Writing the calibration output to a file
    with open(args.calib_file) as f:
        f.write("N calibration: {}".format(n_calib))
        f.write("\n")
        f.write("\n".join(calibration_dirs.tolist()))
        f.write("\n")
        # f.write("Calibration scores: {}\n".format(calibration_scores))
        f.write("Score quantile: {}\n".format(score_quantile))
        f.write("Alpha: {}\n".format(args.alpha))
    f.close()
    print("Score quantile: {}".format(score_quantile))



