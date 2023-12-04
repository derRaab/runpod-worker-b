INPUT_SCHEMA = {
    'zip_url': {
        'type': str,
        'required': True
    },

    'instance_name': {
        'type': str,
        'required': True
    },
    'class_name': {
        'type': str,
        'required': True
    },

    # https://github.com/bmaltais/kohya_ss/wiki/LoRA-training-parameters#unet-learning-rate
    'unet_lr': {
        'type': float,
        'required': False,
        'default': 0.0001
    },

    # https://github.com/bmaltais/kohya_ss/wiki/LoRA-training-parameters#network-rank-dimension
    # https://github.com/bmaltais/kohya_ss/blob/master/train_network_README.md#options-for-learning-lora
    'network_dim': {
        'type': int,
        'required': False,
        'default': 2
    },

    'lr_scheduler_num_cycles': {
        'type': int,
        'required': False,
        'default': 1
    },


    'learning_rate': {
        'type': float,
        'required': False,
        'default': 0.0000004
    },
    'lr_scheduler': {
        'type': str,
        'required': False,
        'default': 'constant'
    },
    'lr_warmup_steps': {
        'type': int,
        'required': False,
        'default': 280
    },
    'train_batch_size': {
        'type': int,
        'required': False,
        'default': 1
    },
    'max_train_steps': {
        'type': int,
        'required': False,
        'default': 1
    },
    'mixed_precision': {
        'type': str,
        'required': False,
        'default': 'fp16'
    },
    'save_precision': {
        'type': str,
        'required': False,
        'default': 'fp16'
    },
    'optimizer_type': {
        'type': str,
        'required': False,
        'default': 'Adafactor'
    },
    'max_data_loader_num_workers': {
        'type': int,
        'required': False,
        'default': 0
    },
    'steps': {
        'type': int,
        'required': False,
        'default': 1
    },
    'num_cpu_threads_per_process': {
        'type': int,
        'required': False,
        'default': 0
    },
}
