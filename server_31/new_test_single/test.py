import tensorflow as tf
from improve_single import CreateGetTrainBatchFile, CreateSaveModelFile

if CreateSaveModelFile().run():
    print("model create success")
else:
    print("model create failed")
    exit(1)

graph = tf.Graph()
with graph.as_default():
    if CreateGetTrainBatchFile().run():
        print("data create success")
    else:
        print("data create failed")
        exit(1)
