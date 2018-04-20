from .ctpn.cfg import Config
from .ctpn.other import resize_im
from .lib.networks.factory import get_network
from .lib.fast_rcnn.config import cfg
from .lib.fast_rcnn.test import test_ctpn
import tensorflow as tf
import sys
sys.path.append('ctpn')


def load_tf_model():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('ctpn/models/')
    saver.restore(sess, ckpt.model_checkpoint_path)
    return sess, saver, net


def ctpn(img, sess, net):
    """
    text box detect
    """
    scale, max_scale = Config.SCALE, Config.MAX_SCALE
    img, f = resize_im(img, scale=scale, max_scale=max_scale)
    scores, boxes = test_ctpn(sess, net, img)
    return scores, boxes, img
