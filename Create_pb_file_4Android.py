# coding: utf-8
from __future__ import print_function

import os
import time

import cv2
import numpy as np
import tensorflow as tf

import model
from preprocessing import vgg_preprocessing

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                                                   'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 400, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "./models/fast-style-model.ckpt", "")
tf.app.flags.DEFINE_string("image_file", "./img/test1.jpg", "")

FLAGS = tf.app.flags.FLAGS


def main(_):
    img_read_o = cv2.imread(FLAGS.image_file)

    dim = (FLAGS.image_size, FLAGS.image_size)
    img_read_o = cv2.resize(img_read_o, dim, interpolation=cv2.INTER_AREA)
    shape = img_read_o.shape
    height = shape[0]
    width = shape[1]
    img_read = np.float32(img_read_o)
    preprocessing_image = None
    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            X = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input')
            cast = tf.cast(X, tf.uint8)
            image0 = vgg_preprocessing.preprocess_image(cast, height, width, is_training=False)
            expand_dims_input = tf.expand_dims(image0, 0, name='expand_dims_input')
            preprocessing_image = sess.run(expand_dims_input, feed_dict={X: img_read})
    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            X = tf.placeholder(tf.float32, shape=[1, FLAGS.image_size, FLAGS.image_size, 3], name='input')
            # cast = tf.cast(X, tf.uint8)
            #
            # # Read image data.
            # image0 = vgg_preprocessing.preprocess_image(cast, height, width, is_training=False)
            # print('image0')
            # print(image0.shape)
            # # Add batch dimension
            # expand_dims_input = tf.expand_dims(image0, 0, name='expand_dims_input')
            # image = tf.expand_dims(image, 0)

            generated = model.net(X, training=False)
            # generated = tf.cast(generated, tf.uint8)
            # generated = tf.cast(generated, tf.float32)

            # Remove batch dimension----
            # generated = tf.squeeze(generated, [0], name="output_new")

            # generated1 = tf.cast(generated, tf.uint8)
            # print('generated1')
            # print(generated1.shape)
            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            # Make sure 'generated' directory exists.
            generated_file = 'generated/res4A.jpg'
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # Generate and write image data to file.
            with open(generated_file, 'wb') as img:
                start_time = time.time()

                # output_tensors = tf.io.encode_jpeg(generated1)
                # print('output_tensors')
                # print(output_tensors.shape)
                output = sess.run(generated, feed_dict={X: preprocessing_image})
                # img.write(output)
                end_time = time.time()
                # Generated the protobuf file.
                # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output_new'])
                # with tf.gfile.FastGFile('models/saved_model.pb', mode = 'wb') as f:
                #     f.write(output_graph_def.SerializeToString())

                # Convert the model.
                converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [X], [generated])
                tflite_model = converter.convert()

                # Save the model.
                # with open('model.tflite', 'wb') as f:
                #     f.write(tflite_model)
                builder = tf.saved_model.Builder('saved-model-builder-%d' % FLAGS.image_size)

                sig_def = tf.saved_model.predict_signature_def(
                    inputs={'input': X},
                    outputs={'output': generated})

                builder.add_meta_graph_and_variables(
                    sess, tags=["serve"], signature_def_map={
                        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: sig_def
                    })
                builder.save()

                tf.logging.info('Convert done!')
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
