import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import torch
import os
from torchvision import transforms
from PIL import Image

def random_bbox(row_size = 128, col_size = 128):
    mask = np.zeros((256, 256, 3), dtype=np.uint8)
    row_start = np.random.randint(0, 256 - row_size)
    col_start = np.random.randint(0, 256 - col_size)
    mask[row_start:(row_start + row_size), col_start:(col_start + col_size), :] = 255
    return mask

def penultimate_layer(model, x):
    # N x 3 x 299 x 299
    x = model.Conv2d_1a_3x3(x)
    # N x 32 x 149 x 149
    x = model.Conv2d_2a_3x3(x)
    # N x 32 x 147 x 147
    x = model.Conv2d_2b_3x3(x)
    # N x 64 x 147 x 147
    x = model.maxpool1(x)
    # N x 64 x 73 x 73
    x = model.Conv2d_3b_1x1(x)
    # N x 80 x 73 x 73
    x = model.Conv2d_4a_3x3(x)
    # N x 192 x 71 x 71
    x = model.maxpool2(x)
    # N x 192 x 35 x 35
    x = model.Mixed_5b(x)
    # N x 256 x 35 x 35
    x = model.Mixed_5c(x)
    # N x 288 x 35 x 35
    x = model.Mixed_5d(x)
    # N x 288 x 35 x 35
    x = model.Mixed_6a(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6b(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6c(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6d(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6e(x)
    # N x 768 x 17 x 17
    aux: Optional[Tensor] = None
    if model.AuxLogits is not None:
        if model.training:
            aux = model.AuxLogits(x)
    # N x 768 x 17 x 17
    x = model.Mixed_7a(x)
    # N x 1280 x 8 x 8
    x = model.Mixed_7b(x)
    # N x 2048 x 8 x 8
    x = model.Mixed_7c(x)
    # N x 2048 x 8 x 8
    # Adaptive average pooling
    x = model.avgpool(x)
    # N x 2048 x 1 x 1
    x = model.dropout(x)
    # N x 2048 x 1 x 1
    x = torch.flatten(x, 1)
    return x


def get_inception_score(output, preprocess, inception):
    output_image = preprocess(output)
    output_image = torch.unsqueeze(output_image, 0)
    if torch.cuda.is_available():
        output_image = output_image.to('cuda')
        inception.to('cuda')

    with torch.no_grad():
        feat_ouput = penultimate_layer(inception, output_image)
    return feat_ouput

def calibration_score(session, input_image, original, preprocess, inception, n_samples, output, input_image_ph):
    temp_scores = np.zeros(n_samples)
    for i in range(n_samples):
        # Running the model
        result = session.run(output, feed_dict={input_image_ph: input_image})

        # Calculating the inception score for each generated image
        output_image = Image.fromarray(np.uint8(result[0][:, :, ::-1]))
        feat_actual = get_inception_score(original, preprocess, inception)
        feat_output = get_inception_score(output_image, preprocess, inception)
        calib_score = torch.nn.functional.mse_loss(feat_output, feat_actual)
        temp_scores[i] = calib_score.item()
    return np.min(temp_scores)

def get_multiple_samples_and_inception(session, input_image, preprocess, inception, samples, output, input_image_ph):
    output_images = []
    inception_scores = []
    for i in range(samples):
        # Running the model
        result = session.run(output, feed_dict={input_image_ph: input_image})
        # Calculating the inception score for each generated image
        output_image = Image.fromarray(np.uint8(result[0][:, :, ::-1]))
        output_images.append(result[0][:, :, ::-1])
        inception_scores.append(get_inception_score(output_image, preprocess, inception))
    return output_images, torch.cat(inception_scores, dim = 0).detach().cpu().numpy()

def calibration_score_v2(inception_scores, score_in_q):
    temp_scores = np.zeros(inception_scores.shape[0])
    if torch.is_tensor(score_in_q):
        score_in_q = score_in_q.numpy()
    for i in range(inception_scores.shape[0]):
        # calib_score = torch.nn.functional.mse_loss(inception_scores[i], score_in_q)
        temp_scores[i] = np.mean((inception_scores[i] - score_in_q)**2)
    return np.min(temp_scores)

