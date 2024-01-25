# coding=utf-8
import glob
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import random
import shutil
import time
import tensorflow as tf
from utils.model_exporter import model_best_exporter
from utils.json_reader import load_json
from utils import schema_utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name", 'train', "One of 'train', 'test'.")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of epochs.")
tf.app.flags.DEFINE_integer("batch_size", 1024, "Number of batch size.")
tf.app.flags.DEFINE_integer("log_steps", 100, "Save summary every steps.")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate.")
tf.app.flags.DEFINE_float("l2_reg", 0, "L2 regularization.")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "Optimizer type {SGD, Adam, Adagrad, Momentum}.")
tf.app.flags.DEFINE_string("deep_layers", '32,16', "Deep layers.")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5', "Dropout rate.")
tf.app.flags.DEFINE_string('train_data_path', '../data/train/', 'Training set path.')
tf.app.flags.DEFINE_string('test_data_path', '../data/test/', 'Test set path.')
cur_time = int(time.time())
tf.app.flags.DEFINE_string("model_dir", '../model/swan_%d' % cur_time, "Check point dir.")
tf.app.flags.DEFINE_string("servable_model_dir", '../model/swan_%d' % cur_time,
                           "Export servable code for TensorFlow Serving.")
tf.app.flags.DEFINE_string("task_type", 'train', "Task type {train, infer, eval, export}.")
tf.app.flags.DEFINE_boolean("clear_existing_model", True, "Clear existing model or not.")
tf.app.flags.DEFINE_string("pos_weights", "1,1", "Positive sample weight.")
tf.app.flags.DEFINE_integer("adaptive_experts_num", 10, "Num of experts in the AEG.")
tf.app.flags.DEFINE_integer("shared_experts_num", 10, "Num of experts in the SEG.")
tf.app.flags.DEFINE_float("tau", 0.001, "Temperature coefficient tau.")

tf.app.flags.DEFINE_boolean("multi_task", False, "multi-task or single-task.")
tf.app.flags.DEFINE_integer("task_num", 1, "Num of tasks, corr to multi-task.")

tf.app.flags.DEFINE_string("loss_weights", '1.0,1.0', "Loss weights.")
tf.app.flags.DEFINE_float("cosine_loss_weight", 0.001, "Cosine similarity loss weight.")
tf.app.flags.DEFINE_float("var_loss_weight", 0.001, "Var loss weight.")

tf.app.flags.DEFINE_string("tower_units", '32,16', "Tower units.")
tf.app.flags.DEFINE_string("tower_dropouts", '0.5,0.5', "Tower dropout ratio.")

tf.app.flags.DEFINE_string("config_file_path", '../config/SwAN.json', "Config json path.")

# log setting
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(asctime)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

config = load_json(json_file_path=FLAGS.config_file_path)
features_schemas = schema_utils.get_feature_schema(config)
feature_transforms = schema_utils.get_feature_transform(config)
label_schemas = schema_utils.get_label_schema(config)
fc = tf.feature_column


def dics(vec, threshold, tau):
    threshold = tf.concat([threshold for _ in range(vec.shape[1])], axis=-1)
    x = (vec - threshold) / tau
    return tf.expand_dims(tf.sigmoid(x), axis=-1)


def san(similar_emb, query_emb, user_emb):
    l2_reg = tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)
    embedding_size = similar_emb.get_shape().as_list()[2]
    san_all = tf.concat([user_emb, similar_emb, query_emb, similar_emb - query_emb, query_emb * similar_emb], axis=-1)
    san_all = tf.layers.dense(inputs=san_all, units=16, activation=None,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=l2_reg
                              )
    score_weight = tf.layers.dense(inputs=san_all, units=1)
    score_weight = tf.nn.softmax(logits=score_weight, axis=1)
    san_att_out = tf.multiply(score_weight, similar_emb)
    san_att_out = tf.reduce_sum(san_att_out, axis=1)
    san_att_out = tf.reshape(san_att_out, shape=[-1, embedding_size])
    return san_att_out, score_weight


def input_fn(filenames, batch_size=32, num_epochs=None, perform_shuffle=False):
    def _parse_fn(record):
        features = tf.io.parse_single_example(record, features_schemas)
        labels = tf.io.parse_single_example(record, label_schemas)
        return features, labels

    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=10).prefetch(1000000)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def model_fn(features, labels, mode, params):
    print("Start a new epoch~")
    print(params)
    """build Estimator model"""

    # L2 loss config
    l2_reg = tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)

    # ---------features input---------
    base_features = fc.input_layer(
        features=features,
        feature_columns=feature_transforms['base_features']['columns']
    )

    user_features = fc.input_layer(
        features=features,
        feature_columns=feature_transforms['user_features']['columns']
    )
    scene_features = fc.input_layer(
        features=features,
        feature_columns=feature_transforms['scene_features']['columns']
    )

    # SRG
    mod_base = 1000
    srg_dict = tf.get_variable(name='scene_emb', shape=[mod_base, 4],
                               dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    curr_scene = tf.cast(fc.input_layer(
        features=features,
        feature_columns=feature_transforms['curr_scene']['columns']
    ), dtype=tf.int64)
    curr_scene_emb = tf.nn.embedding_lookup(srg_dict, tf.mod(curr_scene, mod_base))
    curr_scene_emb = tf.squeeze(curr_scene_emb, axis=1)

    # SAN
    similar_scene = tf.cast(fc.input_layer(
        features=features,
        feature_columns=feature_transforms['similar_scene']['columns']
    ), dtype=tf.int64)
    similar_scene_emb = tf.nn.embedding_lookup(srg_dict, tf.mod(similar_scene, mod_base))
    query_emb = tf.concat([tf.expand_dims(curr_scene_emb, axis=1) for _ in range(similar_scene_emb.shape[1])], axis=1)
    user_emb = tf.concat([tf.expand_dims(user_features, axis=1) for _ in range(similar_scene_emb.shape[1])], axis=1)
    similar_scene_emb, score_weights = san(similar_scene_emb, query_emb, user_emb)

    # CFR
    is_similar_scene = tf.cast(fc.input_layer(
        features=features,
        feature_columns=feature_transforms['is_similar_scene']['columns']
    ), dtype=tf.int64)
    total_scene_group = [1, 2, 3]  # old scene group
    for i, scene_group_i in enumerate(total_scene_group):
        cfr_emb = fc.input_layer(
            features=features,
            feature_columns=feature_transforms['base_features']['columns']
        )
        scene_curr_i = tf.expand_dims(tf.cast(is_similar_scene[:, i], tf.float32), axis=-1)
        scene_curr_i = tf.concat([scene_curr_i for _ in range(cfr_emb.shape[1])], axis=-1)
        base_features += tf.where(
            tf.equal(scene_curr_i, tf.ones_like(scene_curr_i)),
            tf.multiply(cfr_emb, scene_curr_i),
            tf.zeros_like(scene_curr_i)
        )

    # emb concatenation
    input_emb = tf.concat([base_features, user_features, scene_features, curr_scene_emb, similar_scene_emb],
                          axis=-1)
    input_emb = tf.layers.batch_normalization(input_emb)

    # ------Model implement------
    experts = []
    with tf.variable_scope("experts-part"):
        for j in range(FLAGS.adaptive_experts_num + FLAGS.shared_experts_num):
            expert_layer = input_emb
            dnn_layer_nodes = list(map(int, FLAGS.deep_layers.split(',')))
            dropout = list(map(float, FLAGS.dropout.split(',')))
            for i in range(len(dnn_layer_nodes)):
                expert_layer = tf.layers.dense(inputs=expert_layer, units=dnn_layer_nodes[i], activation=tf.nn.relu,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                               bias_initializer=tf.zeros_initializer(), name='expert_%d_%d' % (j, i))
                if mode == tf.estimator.ModeKeys.TRAIN:
                    expert_layer = tf.nn.dropout(expert_layer, keep_prob=dropout[i])
            experts.append(tf.expand_dims(expert_layer, axis=1))

    # cosine similarity loss
    expert_cosine_loss = 0
    for expert_i in range(FLAGS.adaptive_experts_num):
        for contrast_i in range(FLAGS.adaptive_experts_num):
            if expert_i != contrast_i:
                e_1, e_2 = experts[expert_i], experts[contrast_i]
                e_1 = e_1 / tf.sqrt(tf.reduce_sum(tf.square(e_1)))
                e_2 = e_2 / tf.sqrt(tf.reduce_sum(tf.square(e_2)))
                expert_cosine_loss += tf.abs(1 - tf.losses.cosine_distance(e_1, e_2, axis=-1))

    experts = tf.concat(experts, axis=1)

    final_outputs = []
    gate_networks_adaptive, gate_networks_shared = [], []
    task_out_adaptive, task_out_shared = [], []
    adaptive_networks = []

    # AEG
    for i in range(FLAGS.task_num):
        gate_network_adaptive = tf.contrib.layers.fully_connected(
            inputs=input_emb,
            num_outputs=FLAGS.adaptive_experts_num,
            activation_fn=tf.nn.relu,
            weights_regularizer=l2_reg)
        gate_network_shape = gate_network_adaptive.get_shape().as_list()
        gate_network_adaptive = tf.nn.softmax(gate_network_adaptive, axis=1)
        gate_network_adaptive = tf.reshape(gate_network_adaptive, shape=[-1, gate_network_shape[1], 1])
        gate_networks_adaptive.append(gate_network_adaptive)

        AEG_threshold = tf.layers.dense(inputs=similar_scene_emb,
                                        units=1,
                                        activation=tf.nn.sigmoid,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer())
        adaptive_network = tf.layers.dense(inputs=tf.concat([similar_scene_emb, user_features], axis=-1),
                                           units=FLAGS.adaptive_experts_num,
                                           activation=tf.nn.sigmoid,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           bias_initializer=tf.zeros_initializer(),
                                           name='adaptive_%d' % i)
        adaptive_networks.append(dics(adaptive_network, AEG_threshold, FLAGS.tau))

    for gate_network, adaptive_network in zip(gate_networks_adaptive, adaptive_networks):
        combine_network = tf.multiply(gate_network, adaptive_network)
        combine_network /= tf.reduce_sum(combine_network, axis=1, keepdims=True)
        task_out_ = tf.multiply(experts[:, 0: FLAGS.adaptive_experts_num, :], combine_network)
        task_out_adaptive.append(tf.reduce_sum(task_out_, axis=1))

    # SEG
    for i in range(FLAGS.task_num):
        gate_network_shared = tf.contrib.layers.fully_connected(
            inputs=input_emb,
            num_outputs=FLAGS.shared_experts_num,
            activation_fn=tf.nn.relu,
            weights_regularizer=l2_reg)
        gate_network_shape = gate_network_shared.get_shape().as_list()
        gate_network_shared = tf.nn.softmax(gate_network_shared, axis=1)
        gate_network_shared = tf.reshape(gate_network_shared, shape=[-1, gate_network_shape[1], 1])  # None * Nums * 1
        gate_networks_shared.append(gate_network_shared)

    for gate_network in gate_networks_shared:
        task_out_ = tf.multiply(experts[:, -FLAGS.shared_experts_num:, :], gate_network)
        task_out_shared.append(tf.reduce_sum(task_out_, axis=1))  # None * Nums * E

    for i in range(FLAGS.task_num):
        final_outputs.append(tf.concat([task_out_adaptive[i], task_out_shared[i]], axis=1))

    def tower(x, units_info):
        units = list(map(int, units_info.strip().split(',')))
        tower_layer = x
        for unit in units:
            tower_layer = tf.layers.dense(inputs=tower_layer, units=unit, activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.zeros_initializer())
        return tower_layer

    predictions = {}
    y_task_1 = final_outputs[0]
    y_task_1 = tower(y_task_1, units_info=FLAGS.tower_units)
    y_task_1_ = tf.contrib.layers.fully_connected(inputs=y_task_1,
                                                  num_outputs=1,
                                                  activation_fn=None,
                                                  weights_regularizer=l2_reg,
                                                  scope='task_1')
    y_task_1_ = tf.reshape(y_task_1_, [-1, ])
    y_task_1 = tf.sigmoid(y_task_1_)
    predictions.update({'y_click': y_task_1})

    y_task_2_, y_task_2 = None, None
    if FLAGS.multi_task:
        y_task_2 = final_outputs[1]
        y_task_2 = tower(y_task_2, units_info=FLAGS.tower_units)
        y_task_2_ = tf.contrib.layers.fully_connected(inputs=y_task_2,
                                                      num_outputs=1,
                                                      activation_fn=None,
                                                      weights_regularizer=l2_reg,
                                                      scope='task_2')
        y_task_2_ = tf.reshape(y_task_2_, [-1, ])
        y_task_2 = tf.sigmoid(y_task_2_)
        predictions.update({'y_order': y_task_2})

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}
    # Estimator predict mod
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # var loss
    variance_loss = 0.
    for i in range(FLAGS.task_num):
        f_vec = tf.squeeze(adaptive_networks[i], axis=-1)
        f_mean = tf.reduce_mean(f_vec, axis=-1, keepdims=True)
        top = tf.reduce_sum(tf.square(f_vec - f_mean), axis=-1)
        variance_loss -= tf.reduce_mean(top / FLAGS.adaptive_experts_num, axis=0)

    # loss
    eval_metric_ops = {}
    label_task_1 = tf.cast(tf.reshape(labels['is_click'], shape=[-1, ]), dtype=tf.float32)
    label_task_2 = tf.cast(tf.reshape(labels['is_order'], shape=[-1, ]), dtype=tf.float32)
    loss_weights = list(map(float, FLAGS.loss_weights.strip().split(',')))
    pos_weights = list(map(float, FLAGS.pos_weights.strip().split(',')))
    if FLAGS.multi_task:
        loss = loss_weights[0] * tf.losses.sigmoid_cross_entropy(multi_class_labels=label_task_1,
                                                                 logits=y_task_1_,
                                                                 weights=tf.add(
                                                                     label_task_1 * (pos_weights[0] - 1.0),
                                                                     tf.ones_like(label_task_1))) + \
               loss_weights[1] * tf.losses.sigmoid_cross_entropy(multi_class_labels=label_task_2,
                                                                 logits=y_task_2_,
                                                                 weights=tf.add(label_task_2 * (pos_weights[1] - 1.0),
                                                                                tf.ones_like(label_task_2))) + \
               FLAGS.cosine_loss_weight * expert_cosine_loss + \
               FLAGS.var_loss_weight * variance_loss
    else:
        loss = loss_weights[0] * tf.losses.sigmoid_cross_entropy(multi_class_labels=label_task_1,
                                                                 logits=y_task_1_,
                                                                 weights=tf.add(
                                                                     label_task_1 * (pos_weights[0] - 1.0),
                                                                     tf.ones_like(label_task_1))) + \
               FLAGS.cosine_loss_weight * expert_cosine_loss + \
               FLAGS.var_loss_weight * variance_loss

    eval_metric_ops.update({"auc_task_1": tf.metrics.auc(label_task_1, y_task_1)})
    if FLAGS.multi_task:
        eval_metric_ops.update({"auc_task_2": tf.metrics.auc(label_task_2, y_task_2)})

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ------build optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate, initial_accumulator_value=1e-6)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def main(_):
    tr_files = glob.glob('%s/*' % FLAGS.train_data_path)
    random.shuffle(tr_files)
    va_files = glob.glob('%s/*' % FLAGS.test_data_path)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing code cleaned at %s" % FLAGS.model_dir)

    model_params = {
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
    }

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    config_proto = tf.ConfigProto(allow_soft_placement=True,
                                  intra_op_parallelism_threads=0,
                                  inter_op_parallelism_threads=0,
                                  log_device_placement=False,
                                  )
    run_config = tf.estimator.RunConfig(train_distribute=strategy,
                                        eval_distribute=strategy,
                                        session_config=config_proto,
                                        log_step_count_steps=FLAGS.log_steps,
                                        save_checkpoints_steps=FLAGS.log_steps * 5,
                                        save_summary_steps=FLAGS.log_steps * 5,
                                        tf_random_seed=114514)

    SwAN = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params,
                                  config=run_config)

    serving_input_receiver_fn = schema_utils.build_raw_serving_input_receiver_fn(config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs,
                                      batch_size=FLAGS.batch_size))

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=2048),
            steps=None,
            exporters=[model_best_exporter(FLAGS.job_name, serving_input_receiver_fn, exports_to_keep=1,
                                           metric_key='auc_task_1', big_better=False)],
            start_delay_secs=10, throttle_secs=10
        )
        tf.estimator.train_and_evaluate(SwAN, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        SwAN.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'export':
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            serving_input_receiver_fn)
        SwAN.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
