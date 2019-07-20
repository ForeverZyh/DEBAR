import tensorflow as tf
import numpy as np
import math

def choose_max_batch(x_train, y_train, batch_size, grads, evaled, sess, feed_dict, additional):
    ret_x = None
    ret_y = None
    min_value = 1e20
    ix = None
    for i in range(0, len(x_train), batch_size):
        eval_grads = sess.run(grads, feed_dict={**additional, feed_dict[0]:x_train[i:i+batch_size], feed_dict[1]:y_train[i:i+batch_size]})
        value = 0
        for k in range(len(eval_grads)):
            value += np.sum(np.array(eval_grads[k]) * evaled[k])
        if value < min_value:
            ret_x = x_train[i:i+batch_size]
            ret_y = y_train[i:i+batch_size]
            min_value = value
            ix = i
    return ret_x, ret_y, min_value, ix

def rank_delta(x_train, y_train, batch_size, delta, sess, feed_dict, whole=None, additional={}):
    if whole is None:
        whole = range(len(x_train))
    good_points = []
    for i in range(0, len(whole), batch_size):
        eval_delta = sess.run(delta, feed_dict={**additional, feed_dict[0]:x_train[whole[i:min(i+batch_size, len(x_train))]], feed_dict[1]:y_train[whole[i:min(i+batch_size, len(x_train))]]})
        for j in range(len(eval_delta)):
            good_points.append((eval_delta[j], whole[j + i]))
    good_points.sort(key=lambda x:-x[0])
    return np.array(list(map(lambda x:x[1], good_points)))

def random_choose(size, batch_size):
    return np.random.choice(np.arange(size), batch_size)


def iterative_find(x_new_train, y_new_train, x_train, y_train, chosen_idx, grad_logits_delta, grad_loss, feed_dict, batch_size, sess, weights, grad_image):
#     for i in range(len(x_train)):
    modify_everytime = 10
    for i in range(10):
        best = chosen_idx[:1]

        eval_grads = [np.zeros(weight.shape) for weight in weights]
        eval_grads_tmp = sess.run(grad_logits_delta, feed_dict={feed_dict[0]:x_new_train[best], feed_dict[1]:y_new_train[best]})
        for j in range(len(eval_grads)):
            eval_grads[j] += eval_grads_tmp[j]

#         for times in range(100):
        random_idx = random_choose(len(x_train), batch_size * modify_everytime)
        x_batch, y_batch, value, ix = choose_max_batch(x_train[random_idx], y_train[random_idx], batch_size, grad_loss, eval_grads, sess, feed_dict)
#             if value < 0:
#                 break
        
        ret = sess.run(grad_image, feed_dict={feed_dict[0]:x_new_train[best], feed_dict[1]:y_new_train[best]})[0]
        x_new_train[best]=np.clip(x_new_train[best] + ret * 1e-2, -1, 1)
        
        if value < 0 or i == 9:
            return x_batch, y_batch
        
def to_img(x, des):
    if len(x.shape) == 1:
        n = int(math.sqrt(x.shape[0]))
        x =np.reshape(x, (n, x.shape[0]//n))
    v_min = np.min(x)
    v_max = np.max(x)
    x = np.uint8(np.squeeze((x-v_min)/(v_max-v_min)) * 255)
    from PIL import Image
    im = Image.fromarray(np.squeeze(x))
    im.save(des)
        