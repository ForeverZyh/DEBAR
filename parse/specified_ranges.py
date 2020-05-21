class SpecifiedRanges:
    models = ["Github-IPS-1",
              "Github-IPS-6",
              "Github-IPS-9",
              "StackOverflow-IPS-1",
              "StackOverflow-IPS-2",
              "StackOverflow-IPS-6",
              "StackOverflow-IPS-7",
              "StackOverflow-IPS-14",
              "TensorFuzz",
              "ssd_mobile_net_v1", 
              "ssd_inception_v2", 
              "ssd_mobile_net_v2", 
              "faster_rcnn_resnet_50", 
              "deep_speech", 
              "deeplab", 
              "autoencoder_mnae", 
              "autoencoder_vae", 
              "attention_ocr", 
              "textsum", 
              "shake_shake_32", 
              "shake_shake_96", 
              "shake_shake_112", 
              "pyramid_net", 
              "sbn", 
              "sbnrebar", 
              "sbndynamicrebar", 
              "sbngumbel", 
              "audioset", 
              "learning_to_remember_rare_events", 
              "neural_gpu1", 
              "neural_gpu2", 
              "ptn", 
              "namignizer", 
              "feelvos", 
              "fivo_srnn", 
              "fivo_vrnn",
              "fivo_ghmm",
              "deep_contextual_bandits_var_bnn", 
              "deep_contextual_bandits_neural_ban", 
              "deep_contextual_bandits_bb_alpha_nn", 
              "deep_contextual_bandits_rms_bnn", 
              "adversarial_crypto", 
              "sentiment_analysis", 
              "next_frame_prediction", 
              "minigo", 
              "compression_entropy_coder", 
              "lfads", 
              "lm_1b", 
              "swivel", 
              "skip_thought", 
              "video_prediction",
              "gan_mnist",
              "gan_cifar",
              "gan_image_compression",
              "vid2depth",
              "domain_adaptation",
              "delf",]
    
    # a dictionary with key = "filename", and value with another dictionary {"variable_name" -> ranges}
    specified_ranges = {
        "Github-IPS-1": {"Placeholder_2": [0.5,1], "Placeholder": [-1,1]},
        # keep prob; mnist image pixel
        "Github-IPS-6": {"x-input": [-1,1]},
        # mnist image pixel
        "Github-IPS-9": {"Placeholder": [-1,1]},
        # mnist image pixel
        "StackOverflow-IPS-1": {"Placeholder_2": [0.5,1], "Placeholder": [-1,1]},
        # keep prob; mnist image pixel
        "StackOverflow-IPS-2": {"Placeholder_2": [0.5,1], "Placeholder": [-1,1]},
        # keep prob; mnist image pixel
        "StackOverflow-IPS-6": {"Placeholder_2": [0.5,1], "Placeholder": [-1,1]},
        # keep prob; mnist image pixel
        "StackOverflow-IPS-7": {"Placeholder": [0.5,1]},
        # mnist image pixel
        "StackOverflow-IPS-14": {"Placeholder": [0.5,1]},
        # mnist image pixel
        "TensorFuzz": {"OneShotIterator": [[-1,1],[0,9]]},
        # mnist image pixel; labels
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
        "autoencoder_vae": {"Placeholder": [-1,1]},
        # mnist image pixel
        "autoencoder_mnae": {"Placeholder": [0.5,1]},
        # keep prob
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
        "fivo_ghmm": {"OneShotIterator": [[-1,1], [-1,1]]},
        # inputs range
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
        "next_frame_prediction": {"shuffle_batch/random_shuffle_queue": [[-1,1]]},
        # video feature
        "minigo": {"pos_tensor": [-1,1]},
        # pos feature 
        "compression_entropy_coder": {"padding_fifo_queue": [[-1,1]]},
        # image pixel
        "lfads": {"LFADS/keep_prob": [0.95,1], "LFADS/data": [-1,1], "LFADS/z/ic_enc_2_post_g0/logvar/b": [-0.0625, 0.0625], "LFADS/z/ic_enc_2_post_g0/logvar/W": [-0.0625, 0.0625]},
        # dropout prob; embedding; ranged initialize; ranged initialize
        "lm_1b": {},
        # no related input
        "swivel": {"input_producer": [[None,None]]},
        # unknown
        "skip_thought": {"random_input_queue": [[0, None],[0, None],[0, None],[0, None],[0, None],[0, None],[0, None],[0, None],[0, None],[0, None],[0, None]], "beta2_power": [0.1,0.9], "beta1_power": [0.1,0.9]},
        # token_id & one hot values; optimizer beta powers; optimizer beta powers
        "video_prediction": {"model/Placeholder": [1,10000], "model/batch/fifo_queue": [[-1,1],[None,None],[None,None]], "val_model/Placeholder": [1,10000], "val_model/batch/fifo_queue": [[-1,1],[None,None],[None,None]]},
        # iter num; inputs video, unknown, unknown; iter num; inputs video, unknown, unknown
        "gan_mnist": {"inputs/batch/fifo_queue": [[-1,1],[None,None]]},
        # mnist image pixel; unknown
        "gan_cifar": {},
        # no related input
        "gan_image_compression": {},
        # no related input
        "vid2depth": {"data_loading/batching/shuffle_batch/random_shuffle_queue": [[0,1],[1,100],[1,100]]},
        # inputs video, unknown, unknown;
        "domain_adaptation": {"batch_1/fifo_queue": [[-1,1],[None,None]], "batch/fifo_queue": [[-1,1],[None,None]]},
        # mnist pixel, unknown; cifar pixel, unknown;
        "real_nvp": {"model/shuffle_batch/random_shuffle_queue": [[0,1]]},
        #
        "delf": {"input_scales": [0.1,1], "input_image": [0,255], "input_abs_thres": [0,None], "input_max_feature_num": [0,None]}
        # scale; input pixel in 0-255; abs value; feature num
    }

    # the following dict is called by the parse_format_text.py 
    ranges_looking_up = {}