# %%
## 모든 split에 대해서 훈련
# for split_num in range(len(X_train_split_list)):
for split_num in range(1):
    tfdata_protein_train = tf.data.Dataset.from_tensor_slices(X_train_split_list[split_num])
    tfdata_compount_train = tf.data.Dataset.from_tensor_slices(y_train_split_list[split_num])

BATCH_SIZE = 256    # 배치크기
tfdata_protein_train = tfdata_protein_train.batch(BATCH_SIZE)

# %%
## 각 split을 훈련시키는 함수 
num_epoch = 300
def train_each_split(num_epoch):
    for epoch in range(num_epoch):
                