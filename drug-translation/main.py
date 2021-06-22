# %%
def initialize_setting():

    # GPU setting
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2048)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main():
    initialize_setting()    # set the strategy of GPU allocation.
    
if __name__ == "__main__":
    main()