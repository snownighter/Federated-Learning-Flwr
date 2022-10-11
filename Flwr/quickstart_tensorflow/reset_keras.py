from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import tensorflow as tf
import gc

clear = True # use
# Reset Keras Session
def reset_keras():
    if clear:
        sess = get_session()
        clear_session()
        sess.close()
        sess = get_session()
        try:
            del classifier # this is from global space - change this as you need
        except:
            pass
        #print(gc.collect()) # if it does something you should see a number as output
        # use the same config as you used to create the session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        set_session(tf.compat.v1.Session(config=config))
    else:
        return