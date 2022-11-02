from tensorflow.python.keras.backend import clear_session, set_session, get_session
import tensorflow as tf

clear = True # use
# Reset Keras Session
def reset_keras():
    if clear:
        sess = get_session()
        clear_session()
        sess.close()
        get_session()
        #try:
        #    del classifier # this is from global space - change this as you need
        #except:
        #    pass
        # use the same config as you used to create the session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        set_session(tf.compat.v1.Session(config=config))

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        #return
    else:
        return