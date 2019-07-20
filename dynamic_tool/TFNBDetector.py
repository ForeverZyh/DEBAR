import tensorflow as tf
import math
import TFNBUtils as utils
import numpy as np
import copy

class TFNBDetector:
    def __init__(self, sess, x, y, x_train, y_train, x_test, y_test, train_op, loss, batch_size, max_epochs, **kwargs):
        """
        :param sess: the TF sess
        :param x: the input tensor x, e.g. the images
        :param y: the input tensor y, e.g. the labels
        :param x_train, y_train, x_test, y_test: the training and testing data
        :param train_op: the training operator
        :param loss: the loss tensor
        :param batch_size: the batch size
        :param max_epochs: the max training epochs
        :param large_batch_size: a large batch size for inference/testing, not used for training. default: 1000
        :param clip_min, clip_max: the range of the input data
        """
        self.sess = sess
        self.x = x
        self.y = y
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_op = train_op
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.loss = loss
        self.weight = None
        if "large_batch_size" in kwargs:
            self.large_batch_size = kwargs["large_batch_size"]
        else:
            self.large_batch_size = 1000
        if "clip_min" in kwargs:
            self.clip_min = kwargs["clip_min"]
        else:
            self.clip_min = min(np.min(x_train), np.min(x_test))
        if "clip_max" in kwargs:
            self.clip_max = kwargs["clip_max"]
        else:
            self.clip_max = max(np.max(x_train), np.max(x_test))
        if "train_feed_dict" in kwargs:
            self.train_feed_dict = kwargs["train_feed_dict"]
        else:
            self.train_feed_dict = {}
        if "test_feed_dict" in kwargs:
            self.test_feed_dict = kwargs["test_feed_dict"]
        else:
            self.test_feed_dict = {}
        
    def trigger(self, suspect_input, suspect, trigger_type, flag_NNweights, flag_inputs, eps = 0.3, fix_epoch = 1000, rand_pool_size = 20):
        """
        :param suspect_input: the input of the suspect node
        :param suspect: the suspect node which may lead to NAN or INF
        :param trigger_type: trigger method type: "max", "max_difference"
        :param flag_NNweights: if set up, the method will iteratively find training batches to guide the triggering process
        :param flag_inputs: if set up, the method will modify some training inputs to guide the triggering process
            if neither the above two flags are set up, the trigger procedure is a normal training process
        :param eps: linf norm of the edit on the input, larger eps makes more edits on the input
        :param fix_epoch: If set up flag_NNweights, before the fix epoch is reached, the process will find the best triggering input. Larger fix_epoch makes the process runs slower but reduces the number of total epochs.
        :param rand_pool_size: larger rand_pool_size will make the process more effective but reduce the randomness of the training batch
        :return: True if NAN of INF is found, False otherwise
        """
        self.sess.run(tf.global_variables_initializer())
        if trigger_type == "max":
            pass
        elif trigger_type == "max_difference":
            suspect_input = tf.reduce_max(suspect_input, axis=-1) - tf.reduce_min(suspect_input, axis=-1)
        elif trigger_type == "min_abs":
            suspect_input = -tf.reduce_min(tf.abs(suspect_input), axis=-1)
        else:
            raise NotImplemented("Error unknow trigger type: %s" % trigger_type)
        
        if flag_inputs:
            grad_input = tf.gradients(suspect_input, self.x)
        if flag_NNweights:
            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            grad_loss = tf.gradients(self.loss, weights)
            grad_weights = tf.gradients(suspect_input, weights)
            self.weights = []
            self.grad_loss = []
            self.grad_weights = []
            for (grad_l, grad_w, weight) in zip(grad_loss, grad_weights, weights):
                if grad_l is not None and grad_w is not None:
                    self.weights.append(weight)
                    self.grad_loss.append(grad_l)
                    self.grad_weights.append(grad_w)
            weights = self.weights
            grad_weights = self.grad_weights
        
        target = None
        input_id = [np.random.randint(0, len(self.x_train))]
        while target is not None and np.argmax(self.y_train[input_id[0]]) != target:
            input_id = [np.random.randint(0, len(self.x_train))]
        whole = range(len(self.x_train))
        x_new_train = copy.copy(self.x_train)
        for i in range(self.max_epochs):
            # generate the trigger input
            if flag_inputs:
                ret = self.sess.run(grad_input, feed_dict={**self.test_feed_dict, self.x:x_new_train[input_id], self.y:self.y_train[input_id]})[0]
#                 ord = 2
#                 delta = np.clip(x_new_train[input_id] + ret , self.clip_min, self.clip_max) - self.x_train[input_id]
#                 scale_eps = (self.clip_max - self.clip_min) * eps
#                 scale = np.sqrt(np.sum(delta * delta)) / scale_eps
#                 x_new_train[input_id]=np.clip(self.x_train[input_id] + delta / scale, self.clip_min, self.clip_max)

#                 ord = np.inf
#                 delta = np.clip(x_new_train[input_id] + ret, self.clip_min, self.clip_max) - self.x_train[input_id] + 1e-8
#                 scale_eps = (self.clip_max - self.clip_min) * eps
#                 scale = np.max(np.abs(delta)) / scale_eps
#                 x_new_train[input_id]=np.clip(self.x_train[input_id] + delta / scale, self.clip_min, self.clip_max)
                
#                 ord = np.inf iterative
                scale = (self.clip_max - self.clip_min) * eps / np.max(np.abs(ret) + 1e-8)
                x_new_train[input_id]=np.clip(x_new_train[input_id] + ret * scale, self.clip_min, self.clip_max)
                trigger_input = x_new_train[input_id], self.y_train[input_id]
            if flag_NNweights:
                whole = utils.rank_delta(x_new_train, self.y_train, self.large_batch_size, suspect_input, self.sess, [self.x, self.y], whole, self.test_feed_dict)
                if i < fix_epoch:
                    whole = whole[:-(len(x_new_train) // (fix_epoch + 1))]
                input_id = [whole[0]]
                trigger_input = x_new_train[input_id], self.y_train[input_id]
            
            # generate the training batch
            if flag_NNweights and i >= fix_epoch:
                eval_grads = self.sess.run(grad_weights, feed_dict={**self.test_feed_dict, self.x:x_new_train[input_id], self.y:self.y_train[input_id]})
                random_idx = utils.random_choose(len(self.x_train), self.batch_size * rand_pool_size)
                x_batch, y_batch, _, _ = utils.choose_max_batch(self.x_train[random_idx], self.y_train[random_idx], self.batch_size, self.grad_loss, eval_grads, self.sess, [self.x, self.y], self.test_feed_dict)
            else:
                random_idx = utils.random_choose(len(self.x_train), self.batch_size)
                x_batch, y_batch = self.x_train[random_idx], self.y_train[random_idx]
            
            # if none of the flag is set, we use a random batch as the trigger inputs
            if flag_inputs or flag_NNweights:
                suspect_val = self.sess.run(suspect, feed_dict={**self.test_feed_dict, self.x:trigger_input[0], self.y:trigger_input[1]})
            else:
                suspect_val = self.sess.run(suspect, feed_dict={**self.test_feed_dict, self.x:x_batch, self.y:y_batch})
            
            if i % 2000 == 0 and (flag_inputs or flag_NNweights):
                print(np.max(self.sess.run(suspect_input, feed_dict={**self.test_feed_dict, self.x:trigger_input[0], self.y:trigger_input[1]})))

            if i % 2000 == 0 and not (flag_inputs or flag_NNweights):
                print(np.max(self.sess.run(suspect_input, feed_dict={**self.test_feed_dict, self.x:x_batch, self.y:y_batch})))
                
            if len(np.where(np.isnan(suspect_val))[0]) > 0:
                if flag_inputs:
                    utils.to_img(self.x_train[input_id[0]], "./norm.png")
                    utils.to_img(trigger_input[0][0], "./nan.png")
                print("NAN found in %d-th epoch" % i)
                return True
            elif len(np.where(np.isinf(suspect_val))[0]) > 0:
                if flag_inputs:
                    utils.to_img(self.x_train[input_id[0]], "./norm.png")
                    utils.to_img(trigger_input[0][0], "./nan.png")
                print("INF found in %d-th epoch" % i)
                return True
            
            self.sess.run(self.train_op, feed_dict={**self.train_feed_dict, self.x:x_batch, self.y:y_batch})
            
        return False
