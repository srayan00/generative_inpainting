import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import torch
import os
from torchvision import transforms
from PIL import Image

from inpaint_model import InpaintCAModel
import utils_csi

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='output_test', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='model_logs', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--image_dir', default='/nfs/turbo/lsa-tewaria/places2/test_256', 
                    type=str, help='The directory of image directory')
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
parser.add_argument('--calib_file', default='output_test/calibration_output.txt', type=str,
                    help='caiibration info file')

parser.add_argument('--n_test', default = 1, type = int,
                    help = 'number of samples to generate')


def get_diffused_trajs(input_image, T, conformal_quantile, N, session, n_samples, preprocess, inception, output, input_image_ph, eta = 0.01):
        
    # Get reference y_hats and corresponding inception scores for score computation
    ref_y_hats , ref_inception_scores = utils_csi.get_multiple_samples_and_inception(session, input_image, 
                                                                                     preprocess, inception, n_samples,
                                                                                     output, input_image_ph)
    
        
    # Initialize y_hats and inception scores
    y_hats , inception_scores = utils_csi.get_multiple_samples_and_inception(session, input_image, 
                                                                                    preprocess, inception, N,
                                                                                    output, input_image_ph)
    trajectories = []

    print("Total trajectories: {} \n Shape of inception scores: {} \n Shape of image {}".format(inception_scores.shape,
                                                                                                 inception_scores[0].shape, 
                                                                                                 y_hats[0].shape))
    for _ in range(T):
        proposed_y_hat = inception_scores.copy() + eta * np.random.randn(inception_scores.shape[0], inception_scores.shape[1]).astype(np.float32)
        # np.mean((np.repeat(proposed_y_hat, n_samples, axis = 0) - np.tile(ref_inception_scores, (N, 1)))**2).reshape((N, n_samples))
        calib_scores = np.apply_along_axis(lambda x,y: utils_csi.calibration_score_v2(y, x), axis = 1, arr = proposed_y_hat, y = ref_inception_scores)
        
        # proposed_probs = self.encoder.log_prob(proposed_y_hat, test_x_tiled).detach().cpu().exp().numpy()

        in_region = calib_scores < conformal_quantile
        inception_scores[in_region, :] = proposed_y_hat[in_region, :]
        trajectories.append(inception_scores.copy())
    return np.array(trajectories)

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

    # Getting the calibration files and score quantile
    calib_files = []
    with open(args.calib_file, 'r') as f:
        n_calib = f.readline().strip()
        n_calib = int(n_calib.replace('N calibration: ', ''))

        for i in range(n_calib):
            calib_files.append(f.readline().strip())
        score_quantile = float(f.readline().strip().replace('Score quantile: ', ''))
    print("Score quantile: {}".format(score_quantile))

    # Getting the test files
    np.random.seed(0)
    total_test_files = np.delete(dirs, np.where(np.isin(calib_files, dirs)))
    test_files = np.random.choice(total_test_files, size=args.n_test, replace=False)

    # Diffusion process
    for dir in test_files:
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
        traj = get_diffused_trajs(input_image = input_image, T = 1000, conformal_quantile=score_quantile, N =10,
                                  session = sess, n_samples=10, preprocess=preprocess, inception=inception, output=output, input_image_ph=input_image_ph)
        print("Trajectory shape: {}".format(traj.shape))
    


    
