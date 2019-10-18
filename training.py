import tensorflow as tf
import gpflow
import numpy as np

from tensorflow.contrib.opt import ScipyOptimizerInterface as ScipyOpt


def initialize_model(model, objective, session, learning_rate):

    model.initialize(session=session, force=False)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    model_vars = model.trainable_tensors
    model_task_vars = [var for var in model_vars if "H" in var.name]
    train_step = optimizer.minimize(objective, var_list=model_vars)
    if model.name == "MLSVGP":
        task_infer_step = optimizer.minimize(objective, var_list=model_task_vars)
    else:
        task_infer_step = None

    session.run(tf.variables_initializer(optimizer.variables()))

    return train_step, task_infer_step, optimizer

#######################################################
def initialize_training(model, objective,learning_rate ):
    # Create session and initialize model vars
    session = tf.Session()
    model.initialize(session=session, force=False)

    # Create training step
    optimizer = tf.train.AdamOptimizer(learning_rate)
    model_vars = model.trainable_tensors
    model_task_vars = [var for var in model_vars if "H" in var.name]
    train_step = optimizer.minimize(objective, var_list=model_vars)

    model_variables = tf.trainable_variables()
    optimizer_slots = [
        optimizer.get_slot(var, name)
        for name in optimizer.get_slot_names()
        for var in model_variables]
    if isinstance(optimizer, tf.train.AdamOptimizer):
        optimizer_slots.extend([
            a for a in optimizer._get_beta_accumulators()
        ])
    optimizer_slots = [var for var in optimizer_slots if var is not None]
    session.run([tf.initialize_variables(optimizer_slots)])

    saver = tf.train.Saver()

    return session, optimizer, train_step, saver

def train(model, objective, data, learning_rate, n_train_tasks, batch_size, train_steps ):

    session, optimizer, train_step, saver =\
        initialize_training(model, objective, learning_rate)

    num_batches = int(n_train_tasks / batch_size)
    seq = np.arange( n_train_tasks)
    for epoch in range(train_steps):
        all_obj = []
        np.random.shuffle(seq)
        for b in range(int(num_batches)):
            si = b * ARGS.batch_size
            ei = si + ARGS.batch_size
            n_data= data.shape[0]
            X_b=data[:,0].reshape(-1,1)
            Y_b=data[:,0].reshape(-1,1)
            data_scale = n_data / X_b.shape[0]
            ids_b= np.zeros((X_b.shape[0], ), dtype=int)
            task_scale= 1

            _, obj = session.run(
                [train_step, objective],
                feed_dict={
                    model.X_ph: X_b,
                    model.Y_ph: Y_b,
                    model.H_ids_ph: ids_b,
                    model.num_steps: num_steps,
                    model.data_scale: data_scale,
                    model.task_scale: task_scale})

            all_obj.append(obj)

        mobj = np.mean(all_obj)
        print("Epoch {} :: {:.2f}".format(epoch+1, mobj))

    saver.save(session, model_path)

    return session, saver





