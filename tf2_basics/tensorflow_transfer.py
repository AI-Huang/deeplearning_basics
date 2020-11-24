#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-09-20 21:58
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @RefLink : https://blog.csdn.net/loveliuzz/article/details/81661875
# @RefLink : https://stackoverflow.com/questions/39137597/how-to-restore-variables-using-checkpointreader-in-tensorflow
# @RefLink : https://blog.csdn.net/lujian1989/article/details/104334685
"""tensorflow_transfer
transfer using tf.train module
"""
import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


# 谷歌的模型
MODEL_DIR = "D:\\DeepLearningData\\semi-conductor-image-classification-first\\efficientnet-b0"

# 自己的模型
MY_MODEL_DIR = "D:\\DeepLearningData\\semi-conductor-image-classification-first\\my-efficientnet-b0"


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError(
            'No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError(
            'There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)     # 通过checkpoint文件找到模型文件名
    if ckpt and ckpt.model_checkpoint_path:
        # ckpt.model_checkpoint_path表示模型存储的位置，不需要提供模型的名字，它回去查看checkpoint文件
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def useless():
    with tf.Session() as sess:
        meta_file, ckpt_file = get_model_filenames(model_dir=MODEL_DIR)
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver_src = tf.train.import_meta_graph(
            os.path.join(MODEL_DIR, "model.ckpt.meta"))
        saver_src.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

        src_reader = pywrap_tensorflow.NewCheckpointReader(
            os.path.join(MODEL_DIR, ckpt_file))
        # print(len(src_reader.variables))
        # input("yeyeye")
        var_to_shape_map = src_reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor_name: ", key)
        input("aaaa")
        """
        for v in tf.trainable_variables():
            tensor_name = v.name.split(':')[0]
            print(tensor_name)
            if src_reader.has_tensor(tensor_name):
                print('has tensor: ', tensor_name)
                print("shape:", v.get_shape())
        """
        # global 参数是不需要的，只要 trainable_variables
        var_list = tf.trainable_variables()
        # print(len(var_list))  # 213

        # 目标 graph saver_dst
        saver_dst = tf.train.import_meta_graph(
            os.path.join(MY_MODEL_DIR, "model.ckpt-120.meta"))
        # saver_dst.restore(sess, tf.train.latest_checkpoint(MY_MODEL_DIR))
        saver_dst.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

        graph_restore = tf.get_default_graph()

        print("这样")
        saved_path = saver_dst.save(sess, MY_MODEL_DIR)

        dst_reader = pywrap_tensorflow.NewCheckpointReader(
            os.path.join(MY_MODEL_DIR, ckpt_file))
        # global 参数是不需要的，只要 trainable_variables
        dst_var_list = tf.trainable_variables()
        print("len: %d" % len(var_list))  # 213
        print("len: %d" % len(dst_var_list))  # 426
        for v in var_list:  # tf.trainable_variables()
            print(v.name)
            tensor_name = v.name.split(':')[0]
            print(tensor_name)
            if dst_reader.has_tensor(tensor_name):
                print('has tensor: ', tensor_name)
                print("shape:", v.get_shape())
        input("heheh")
        for i, v in enumerate(dst_var_list):  # tf.trainable_variables()
            print(i, v.name)
            tensor_name = v.name.split(':')[0]
            print(tensor_name)
            if dst_reader.has_tensor(tensor_name):
                print('has tensor: ', tensor_name)
                print("shape:", v.get_shape())
        input("heheh")
        for i, v in enumerate(dst_var_list):
            print("assigning layer %d" % i)
            tf.assign(v, var_list[i])

    """
    with tf.Session() as dst_sess:
        # 目标 graph saver_dst
        meta_file, ckpt_file = get_model_filenames(model_dir=MY_MODEL_DIR)
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver_dst = tf.train.import_meta_graph(
            os.path.join(MY_MODEL_DIR, "model.ckpt-0.meta"))
        saver_dst.restore(dst_sess, tf.train.latest_checkpoint(MY_MODEL_DIR))
        # global 参数是不需要的，只要 trainable_variables
        dst_var_list = tf.trainable_variables()
        print("len: %d" % len(dst_var_list))  # 426
        print("len: %d" % len(dst_var_list))  # 426
        for i, v in enumerate(dst_var_list):
            print("assigning layer %d" % i)
            tf.assign(v, var_list[i])
        saved_path = saver_dst.save(dst_sess, MY_MODEL_DIR)
    """


def main():
    # 保存待修改变量名与修改后的变量值
    transfer_variables = {}
    i = 0
    for var_name, var_shape in tf.train.list_variables(MODEL_DIR):
        # var_name 没有:0
        if var_name.find('dense') > -1:  # for last dense layer
            print(i, "dense layer", var_name)
        elif var_name.find("step") > -1:
            print(i, "step", var_name)
            input("find step")
        else:
            var = tf.train.load_variable(MODEL_DIR, var_name)
            transfer_variables[var_name] = var  # 可以在这里修改
        i += 1
    # input("step")

    # 实际变量赋值操作
    with tf.Session() as sess:
        # 目标 graph saver_dst
        saver_dst = tf.train.import_meta_graph(
            os.path.join(MY_MODEL_DIR, "model.ckpt-120.meta"))
        # 当前网络结构区分直接恢复预训练值的变量与修改变量
        # for model.ckpt-120
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        # print("len list: %d" % len(variables_to_restore))  # 623
        # restore_variable_list = []
        transfer_op = {}
        for i, var_to_restore in enumerate(variables_to_restore):
            var_name = var_to_restore.name.split(':')[0]
            if var_name in transfer_variables:
                print("src_var", var_name)
                new_var = tf.convert_to_tensor(transfer_variables[var_name])
                op_assign_var = tf.assign(
                    var_to_restore, new_var)  # 给当前网络中的特定变量赋新值操作
                print(var_to_restore.name)  #
                print(var_to_restore)  # tensor
                transfer_op[var_name] = op_assign_var
                # restore_variable_list.append(var_to_restore)
            else:
                print("dst_var", var_name)
                origin_var = tf.train.load_variable(
                    MY_MODEL_DIR, var_name)  # 不需要 restore latest_checkpoint
                origin_var = tf.convert_to_tensor(origin_var)
                print(i, "var_name", var_name)
                # global_step:0
                print("var_to_restore.name", var_to_restore.name)
                # <tf.Variable 'global_step:0' shape=() dtype=int64>
                print("var_to_restore", var_to_restore)
                print("Print origin_var: ", origin_var)
                op_assign_var = tf.assign(
                    var_to_restore, origin_var)  # 不需要迁移的变量
                transfer_op[var_name] = op_assign_var

        saver_dst = tf.train.Saver(var_list=variables_to_restore)
        i = 0
        for var_name, op in transfer_op.items():
            # if var_name != "global_step":
            # continue
            print("assigning:", i, var_name, op)
            sess.run(op)
            i += 1
        print("Done!")
        saved_path = saver_dst.save(sess, os.path.join(
            MY_MODEL_DIR, "transfered-efficientnet-b0"))


if __name__ == "__main__":
    main()
