class SpecifiedRanges:
    # a dictionary with key = "filename", and value with another dictionary {"variable_name" -> ranges}
    specified_ranges = {
        "ssd_mobile_net_v1": {"IteratorV2": [[0,None],[-1,1],[None,None],[1,None],[None,None],[0,299],[0,1],[0,1],[None,None],[False,True],[0,1],[1,100]]},
    # HASH_KEY ; image pixels from [-1, 1]; unknown; real shape of image; unknown; the corner of boxes; one hot values; one hot values; unknown; boolean value; weights; number of boxes
        "ssd_inception_v2": {"IteratorV2": [[0,None],[-1,1],[None,None],[1,None],[None,None],[0,299],[0,1],[0,1],[None,None],[False,True],[0,1],[1,100]]}, 
    # the same reason above
        "ssd_mobile_net_v2": {"IteratorV2": [[0,None],[-1,1],[None,None],[1,None],[None,None],[0,299],[0,1],[0,1],[None,None],[False,True],[0,1],[1,100]]},
    # the same reason above
        "faster_rcnn_resnet_50": {"IteratorV2": [[0,None],[-1,1],[None,None],[1,None],[None,None],[0,299],[0,1],[0,1],[None,None],[False,True],[0,1],[1,100]]},
    # the same reason above
        "deep_speech": {"IteratorV2": [[-1,1],[0,None],[0,None],[0,None]]},
        # spectrogram features; input length; label length; number of classes
        "deeplab": {},
        # no input related
        "vae": {"Placeholder": [-1,1]},
        # mnist image pixel
        "attention_ocr": {"CharAccuracy/mean/count": [1, None], "SequenceAccuracy/mean/count": [1, None]},
        # count; count
        "textsum": {"targets": [0,None], "loss_weights": [1e-10, 1]},
        # token_id; loss_weights
        "shake_shake_32": {"model/accuracy/count": [1, None], "model_1/accuracy/count": [1, None]},
        # count; count
        "shake_shake_96": {"model/accuracy/count": [1, None], "model_1/accuracy/count": [1, None]},
        # the same reason above
        "shake_shake_112": {"model/accuracy/count": [1, None], "model_1/accuracy/count": [1, None]},
        # the same reason above
        "pyramid_net": {"model/accuracy/count": [1, None], "model_1/accuracy/count": [1, None]},
        # the same reason above
        "sbn": {"Variable": [-1,1], "Placeholder": [1,None], "Placeholder_1": [-1,1]},
        # weights; number of examples ; image pixels
        "sbnrebar": {"Variable": [-1,1], "Placeholder": [1,None], "Placeholder_1": [-1,1]},
        # the same reason above
        "sbndynamicrebar": {"Variable": [-1,1], "Placeholder": [1,None], "Placeholder_1": [-1,1]},
        # the same reason above
        "sbngumbel": {"Variable": [-1,1], "Placeholder": [1,None], "Placeholder_1": [-1,1]},
        # the same reason above
        "audioset": {"vggish/input_features": [-1,1]},
        # vggish input_features
        "learning_to_remember_rare_events": {"Placeholder": [-1,1], "recent_idx": [0,None], "Placeholder_1": [0,9], "memvals": [None,None]},
        # mnist pixel, index, mnist label; unknown
        "neural_gpu1": {"global_step": [1,None], "inp": [None,None], "length": [1,None], "tgt": [None,None], "do_training": [0.1,1]},
        # global training step; unknown inp, length, unknown tgt, dropout rate
        "neural_gpu2": {"global_step": [1,None], "inp": [None,None], "length": [1,None], "tgt": [None,None], "do_training": [0.1,1]},
        # the same reason above
        "ptn": {},
        # no input related
        "namignizer": {"model/Placeholder_2": [1e-10,1]},
        # weights
        "feelvos": {},
        # no input related
        "fivo_srnn": {"OneShotIterator": [[0,1],[0,1],[1,None]]},
        # one hot values, one hot values, len
        "fivo_vrnn": {"OneShotIterator": [[0,1],[0,1],[1,None]]},
        # the same reason above
        "deep_contextual_bandits_var_bnn": {"Placeholder_1": [None,None], "Placeholder":[1,None]},
        # rewards, size of data
        "deep_contextual_bandits_neural_ban": {"global_step": [1,None]},
        # global training step
        "deep_contextual_bandits_bb_alpha_nn": {"data_size": [1,None], "w": [0,1], "y": [None,None], "x": [None,None]},
        # data size; weights for actions, rewards, rewards
        "deep_contextual_bandits_rms_bnn": {"global_step": [1,None]},
        # global training step
        "adversarial_crypto": {},
        # no input related
        "sentiment_analysis": {"input_1": [0,None], "batch_normalization_v1/keras_learning_phase": [False,True], "batch_normalization_v1/moving_variance": [0,None]},
        # token_id; train or test; variance
        "next_frame_prediction": {"shuffle_batch/random_shuffle_queue": [-1,1]},
        # video feature
        "minigo": {"pos_tensor": [-1,1]},
        # pos feature 
        "compression_entropy_coder": {"padding_fifo_queue": [-1,1]},
        # image pixel
        "lfads": {"LFADS/keep_prob": [0.1,1], "LFADS/data": [-1,1]},
        # dropout prob; embedding
        "lm_1b": {},
        # no related input
    }

    # the following dict is called by the parse_format_text.py 
    ranges_looking_up = {}