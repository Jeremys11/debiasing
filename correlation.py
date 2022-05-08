import tensorflow as tf
from keras import backend as K
import numpy as np

#Souce for pearson correlation code
#https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras


#r = sum(xi - xbar)(yi - ybar) / sqrt [sum(xi - xbar)**2 * sum(yi - ybar)**2]
#xm = xi - xbar 
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

x = [1,2,3,4,5]
y = [2,3,4,5,6]
x = K.constant(x)
y = K.constant(y)

print(correlation_coefficient_loss(x,y))