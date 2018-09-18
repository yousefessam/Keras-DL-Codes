# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:32:13 2018

@author: Youssef
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))