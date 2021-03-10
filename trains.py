"""
    date:       2021/3/9 4:04 下午
    written by: neonleexiang
"""
from model import SRCNN
from data_preprocessing import SRCNNLoader
import tensorflow as tf

# parameters
num_epochs = 5
batch_size = 5
learning_rate = 0.001

model = SRCNN()
data_loader = SRCNNLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    print(X)
    print(model(X))
    print(y)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.mean_squared_error(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print('training ended')
