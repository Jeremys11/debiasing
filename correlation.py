import tensorflow as tf
import numpy as np
import time


#Souce for pearson correlation code
#https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras


#r = sum(xi - xbar)(yi - ybar) / sqrt [sum(xi - xbar)**2 * sum(yi - ybar)**2]
#xm = xi - xbar 
#r = r_num / r_den
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    #tf.print(x)
    #tf.print(y)
    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_sum(tf.multiply(xm,ym))
    xm_square = tf.square(xm)
    ym_square = tf.square(ym)
    xm_sum = tf.math.reduce_sum(xm_square)
    ym_sum = tf.math.reduce_sum(ym_square)
    mulXY = tf.multiply(xm_sum,ym_sum)
    r_den = tf.sqrt(mulXY)
    r = r_num / r_den

    r = tf.maximum(tf.minimum(r, 1.0), -1.0)
    return 1 - tf.square(r)