import numpy as np
train_inputs1 = np.load('FINAL_SORT/Data48/data_NSFW.npy')
train_inputs2 = np.load('FINAL_SORT/Data48/data_SFW.npy')
train_inputs3 = np.load('FINAL_SORT/Data48/data_unknown.npy')
train_merge1 = np.append(train_inputs1,train_inputs2,axis=0)
train_merge2 = np.append(train_merge1,train_inputs3,axis=0)
np.random.shuffle(train_merge2)
X_train_split = train_merge2[1000:,:2304]
y_train_split = train_merge2[1000:,2304:]
X_test_split = train_merge2[:1000,:2304]
y_test_split = train_merge2[:1000,2304:]
np.save('FINAL_SORT/Data48/X_train.npy', X_train_split)
np.save('FINAL_SORT/Data48/y_train.npy', y_train_split)
np.save('FINAL_SORT/Data48/X_test.npy',X_test_split)
np.save('FINAL_SORT/Data48/y_test.npy',y_test_split)
print(X_train_split.shape)
print(y_train_split.shape)
print(X_test_split.shape)
print(y_test_split.shape)
