# coding=utf-8
import tensorflow as tf

fc = tf.feature_column


def _generate_bucketized_embedding_columns(features, dimension, default_value):
    for key in features:
        boundaries = features[key]
        for i in range(len(boundaries) - 1):
            if boundaries[i] >= boundaries[i + 1]:
                raise ValueError('%s boundaries must be a sorted list.' % key)
    bucket = [
        fc.bucketized_column(fc.numeric_column(key, default_value=default_value), boundaries=features[key])
        for key in features
    ]
    fcs = [
        fc.embedding_column(column, dimension=dimension)
        for column in bucket
    ]
    return {'columns': fcs}


def _generate_categorical_column_with_hash_bucket(features, dimension, hash_bucket_size):
    bucket = [
        fc.categorical_column_with_hash_bucket(key, hash_bucket_size, tf.int64)
        for key in features
    ]
    fcs = fc.shared_embedding_columns(categorical_columns=bucket, dimension=dimension)
    return {'columns': fcs}


def _generate_embedding_columns(features, dimension, default_value):
    categorical_columns = [
        fc.categorical_column_with_vocabulary_list(key, features[key], num_oov_buckets=1, default_value=default_value)
        for key in features
    ]
    fcs = [
        fc.embedding_column(column, dimension=dimension)
        for column in categorical_columns
    ]
    res = dict()
    res["dimension"] = dimension
    res["columns"] = fcs
    return res


def _generate_raw_numeric_columns(features, default_value):
    fcs = [
        fc.numeric_column(key, default_value=default_value)
        for key in features
    ]
    return {'columns': fcs}


def get_feature_transform(js):
    if not isinstance(js, dict):
        assert TypeError("js is not dict!")
    feature_schema = {}
    for group_name, v in js['data_schema']['feature_group'].items():
        default_value = v['default_value']
        operator = v['operator']
        reshape = v['reshape']
        dimension = v['dimension']
        features = v['features']
        if operator == 'bucket':
            feature_schema[group_name] = _generate_bucketized_embedding_columns(features, dimension, default_value)
            feature_schema[group_name]["reshape"] = reshape
            feature_schema[group_name]["embedding_size"] = dimension
        elif operator == 'hash_bucket':
            feature_schema[group_name] = _generate_categorical_column_with_hash_bucket(features, dimension,
                                                                                       v['hash_bucket_size'])
            feature_schema[group_name]["reshape"] = reshape
            feature_schema[group_name]["embedding_size"] = dimension
        elif operator == 'embedding':
            feature_schema[group_name] = _generate_embedding_columns(features, dimension, default_value)
            feature_schema[group_name]["reshape"] = reshape
            feature_schema[group_name]["embedding_size"] = dimension
        elif operator == 'raw_numeric':
            feature_schema[group_name] = _generate_raw_numeric_columns(features, default_value)
        elif operator == 'autodis_numeric':
            feature_schema[group_name] = _generate_raw_numeric_columns(features, default_value)
            feature_schema[group_name]["embedding_size"] = dimension
    return feature_schema


def get_feature_schema(js):
    if not isinstance(js, dict):
        assert TypeError("js is not dict!")
    feature_schema = {}
    for group_name, v in js['data_schema']['feature_group'].items():
        dtype = str(v['dtype'])
        default_value = v['default_value']
        features = v['features']
        for key in features:
            feature_schema.update(
                {key: tf.io.FixedLenFeature((1,), eval("tf.{}".format(dtype)), default_value=default_value)}
            )
    return feature_schema


def get_label_schema(js):
    if not isinstance(js, dict):
        assert TypeError("js is not dict!")
    labels = js['data_schema']['labels']
    label_schema = {}
    label_schema.update({
        name: tf.io.FixedLenFeature((1,), eval("tf.{}".format(config["dtype"])), default_value=config["default_value"])
        for name, config in labels.items()
    })
    return label_schema


def build_raw_serving_input_receiver_fn(js, exclude_feats=None):
    if not isinstance(js, dict):
        assert TypeError("feature_schemas is not dict!")
    if not isinstance(exclude_feats, list):
        assert TypeError("exclude_feats is not list!")
    features_placeholders = {}

    for group_name, v in js['data_schema']['feature_group'].items():
        if exclude_feats is not None and group_name in exclude_feats:
            continue
        dtype = str(v['dtype'])
        features = v['features']
        for key in features:
            features_placeholders.update(
                {key: tf.placeholder(dtype=eval("tf.{}".format(dtype)), shape=[None, ], name=key)}
            )
    print(features_placeholders)
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features_placeholders)
