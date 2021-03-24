# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        # tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 16000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
# %%
## 모든 split에 대해서 훈련
# for split_num in range(len(X_train_split_list)):
for split_num in range(1):
    # train 데이터에 대해서 input, output, target 데이터로 다시 나눠줘야 함
    tfdata_protein_input = tf.data.Dataset.from_tensor_slices(X_train_split_list[split_num])
    tfdata_compound_output = tf.data.Dataset.from_tensor_slices(y_train_split_list[split_num][:, 0:-1])
    tfdata_compound_target = tf.data.Dataset.from_tensor_slices(y_train_split_list[split_num][:, 1:])

tfdata_train_input = np.array(list(tfdata_protein_input))
tfdata_train_output = np.array(list(tfdata_compound_output))
tfdata_train_target = np.array(list(tfdata_compound_target))

# BATCH_SIZE = 128    # 배치크기
# tfdata_train_input = np.array(list(tfdata_protein_input.batch(BATCH_SIZE)))
# tfdata_train_output = np.array(list(tfdata_compound_output.batch(BATCH_SIZE)))
# tfdata_train_target = np.array(list(tfdata_compound_target.batch(BATCH_SIZE)))


#  --- Memo ---
# 어차피 나중에 model.fit()으로, epoch, batch, target 데이터로 loss를 계산하는 것까지 한방에 구현됨.
# 고로, 5-fold cross-validation을 어떻게 model.fit() 과 공존시킬지만 생각하자.
# 단, model을 개발하는 과정에서는 

# %%
## 각 split을 훈련시키는 함수 
num_epoch = 300
def train_each_split(num_epoch):
    for epoch in range(num_epoch):
                