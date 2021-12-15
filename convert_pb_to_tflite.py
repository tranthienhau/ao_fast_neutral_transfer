from __future__ import print_function

import tensorflow as tf

intput_size = 400
converter = tf.lite.TFLiteConverter.from_saved_model('./saved-model-builder-%d' % intput_size)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
tflite_quantized_model = converter.convert()
open("converted_model_%d.tflite" % intput_size, "wb").write(tflite_quantized_model)