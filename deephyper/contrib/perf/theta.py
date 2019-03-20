import os
import tensorflow as tfA
from tensorflow.keras import backend as K

def get_session_conf():
    """Set env variables for better performance on Theta.

    Return:
        A tf.ConfigProto object with specific settings.
    """
    os.environ['KMP_BLOCKTIME'] = 0
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    os.environ['OMP_NUM_THREADS'] = 62

    session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS'])
            )

    return session_conf

def set_perf_settings_for_keras():
/projects/datascience/regele/deephyper/deephyper/contrib/perf
theta_tools.py
    """
    session_conf = get_session_conf()
    session = tf.Session(config=session_conf)
    K.set_session(sess)


