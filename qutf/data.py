# coding: utf-8

"""
Helpers for handling the input pd_data.
"""

import os
from random import shuffle
import shutil

import numpy as np
import six

import pyarrow
import pandas as pd
import tensorflow as tf
from glob import glob
import json

from . import util, specs

import hyperparameters as hp
from IPython import embed

# Parameter in def data_split
train_percentage = 0.7
validation_percentage=0.2
SHUFFLE_BUFFER=123456


def _check_pd_dataset_file(kind, index):
    # validate the kind
    kinds = ["train", "valid", "test"]
    if kind not in kinds:
        raise ValueError("unknown pd_dataset kind '{}', must be one of {}".format(
            kind, ",".join(kinds)))

    # validate the index
    if not (0 <= index < specs.n_files[kind]):
        raise ValueError("pd_dataset '{}' has no file index {}".format(kind, index))

    return "{}_{}.npz".format(kind, index)


def provide_file(kind, index):
    """
    Returns the absolute file path of a file of an input pd_dataset given by *kind* ("train", "valid",
    "test", or "challenge") and its *index*. When the file is not locally accessible, it is
    downloaded from the public CERNBox directory.
    """
    # get the name of the file
    file_name = _check_pd_dataset_file(kind, index)

    # make sure the user pd_data directory exists
    #Specs wegmachen oder auch umschreiben?
    if not os.path.exists(specs.pd_data_dir):
        os.makedirs(specs.pd_data_dir)

    # when it's already in the user pd_data directy, return the full path
    file_path = "/nfs/dust/cms/user/wittmath/root_to_parquet/SKIM_ggF_BulkGraviton_m300/output_*.parquet"
    #os.path.join(specs.pd_data_dir, file_name)
    if os.path.exists(file_path):
        return file_path

def load_pd_data_to_pandas(kind, start_file=0, stop_file=-1, variable_config = "variables.json"):
    """
    Loads a certain *kind* of pd_dataset ("train", "valid", or "test") and returns the
    four-vectors of the jet constituents, the true jet four-vector and the truth label in a 3-tuple.
    When *kind* is *challenge*, the two latter elements of the returned tuple are *None*.
    Internally, each pd_dataset consists of multiple files whose arrays are automatically concatenated.
    For faster prototyping and testing, *start_file* (included) and *stop_file* (first file that is
    *not* included) let you define the range of files to load.
    """
    # fix the file range if necessary
    if stop_file < 0:
        stop_file = specs.n_files[kind] - 1

    # get all local file paths via download or public access
    # file_paths = [provide_file(kind, i) for i in range(start_file, stop_file)]
    file_paths = glob("/nfs/dust/cms/user/wittmath/root_to_parquet/SKIM_ggF_BulkGraviton_m300/output_*.parquet")
    file_paths += glob("/nfs/dust/cms/user/wittmath/root_to_parquet/SKIM_ggF_BulkGraviton_m400/output_*.parquet")
    file_paths += glob("/nfs/dust/cms/user/wittmath/root_to_parquet/SKIM_ggF_BulkGraviton_m500/output_*.parquet")
    # instead of loading all files, storing their contents and concatenating in the end, which can
    # have a peak memory consumption of twice the inputs, define output arrays with the correct
    # dimensions right away and fill it while iterating through files
    #n_events = len(file_paths) * specs.n_events_per_file
    #load_truth = kind != "challenge"

    with open(variable_config) as conf:
        variable_conf = json.load(conf)
    
    input_features = variable_conf.get("training_features")
    gen_features = variable_conf.get("gen_features")

    pd_data = pd.concat([
        pd.read_parquet(file_path, columns=input_features+gen_features)
        for file_path in file_paths
    ])
    return pd_data

    #from IPython import embed; embed()

    #DeltaR-Matching
def apply_delta_r_selection(pd_data, var1, var2):
    deltaR_mask = (pd_data[var1] <= 0.4) & (pd_data[var2] <= 0.4)
    pd_data = pd_data[~deltaR_mask]
    deltaR_mask = (pd_data[var1] > 0.4) & (pd_data[var2] > 0.4)
    pd_data = pd_data[~deltaR_mask]
    return pd_data

def data_preparation(pd_data, variable_config = "variables.json", shuffle_random_state=None):
    
    # first, shuffle the input data
    if shuffle_random_state == None:
        pd_data = pd_data.sample(frac=1).reset_index(drop=True)
    else:
        pd_data = (pd_data.sample(frac=1, random_state=shuffle_random_state)
                    .reset_index(drop=True))
    
    # load input features
    with open(variable_config) as conf:
        variable_conf = json.load(conf)
    
    input_features = variable_conf.get("training_features")
    gen_features = variable_conf.get("gen_features")

    # filtere falsche pT-Werte heraus (sind auf -1 gesetzt)
    pd_data_mask = (pd_data["bjet1_pt"] >= 0) & (pd_data["bjet2_pt"] >= 0)
    pd_data = pd_data[pd_data_mask]

    # DeltaR-Columns werden eingefügt
    pd_data["deltaR_det1_gen1"] = util.deltaR(dataset=pd_data, eta1="bjet1_eta", phi1="bjet1_phi", eta2 = "m_genB1_eta", phi2 = "m_genB1_phi")
    pd_data["deltaR_det1_gen2"] = util.deltaR(dataset=pd_data, eta1="bjet1_eta", phi1="bjet1_phi", eta2 = "m_genB2_eta", phi2 = "m_genB2_phi")
    pd_data["deltaR_det2_gen1"] = util.deltaR(dataset=pd_data, eta1="bjet2_eta", phi1="bjet2_phi", eta2 = "m_genB1_eta", phi2 = "m_genB1_phi")
    pd_data["deltaR_det2_gen2"] = util.deltaR(dataset=pd_data, eta1="bjet2_eta", phi1="bjet2_phi", eta2 = "m_genB2_eta", phi2 = "m_genB2_phi")
    
    # Filter Daten mit Maske raus
    pd_data = apply_delta_r_selection(pd_data=pd_data, var1="deltaR_det1_gen1", var2="deltaR_det1_gen2")
    pd_data = apply_delta_r_selection(pd_data=pd_data, var1="deltaR_det2_gen1", var2="deltaR_det2_gen2")
    
    # zuordnung zu bjet1 oder bjet2
    pd_data["true_bjet1_pt"] = np.where(pd_data["deltaR_det1_gen1"] < 0.4, pd_data["m_genB1_pt"], pd_data["m_genB2_pt"])
    pd_data["true_bjet2_pt"] = np.where(pd_data["deltaR_det2_gen1"] < 0.4, pd_data["m_genB1_pt"], pd_data["m_genB2_pt"])
    

    # extrahiere gen_features aus pd_framework
    gen_data = pd.concat([pd_data.pop(x) for x in gen_features], axis=1)
    
    # extrahiere target aus pd_framework
    target_data = pd.concat([pd_data.pop(f"true_bjet{i}_pt") for i in range(1,3)], axis=1)
        
        #gen_data["matched_gen_bjet1_pt"] = target_data["true_bjet1_pt"].copy()
        #try:
        #     pd_data["matched_gen_bjet1_pt"] = target_data["true_bjet1_pt"].copy()
        # except:
        #     from IPython import embed; embed()
        # normalize to detector-level pt

        # target_data["true_bjet1_pt"] /= pd_data["bjet1_pt"]
    # pd_data["real_truth"] = target_data["true_bjet1_pt"].copy()
    # test_input_features = pd.DataFrame()
    # test_input_features["real_truth"] = target_data["true_bjet1_pt"].copy()
    

    return pd_data, gen_data, target_data

def load_model_inputs(BATCH_SIZE=128, shuffle_random_state=1):
    input_features, gen_info, labels = load_data(shuffle_random_state)
    labels, mean, std = transform_labels(labels)
    # mean, std = 0, 1
    # numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(input_features), labels))
    tf_input_features = tf.data.Dataset.from_tensor_slices(dict(input_features))
    # tf_labels = tf.data.Dataset.from_tensor_slices(dict(labels))
    tf_labels = tf.data.Dataset.from_tensor_slices(labels.to_numpy())
    numeric_dict_ds = tf.data.Dataset.zip((tf_input_features, tf_labels))
    # embed()
    train_data, validation_data, test_data = split_dataset(numeric_dict_ds, SHUFFLE_BUFFER=len(input_features), BATCH_SIZE=BATCH_SIZE)

    return input_features, mean, std, train_data, validation_data, test_data

def labels_to_onehot(labels):
    labels = labels.astype(np.int32)
    onehot = np.zeros((labels.shape[0], labels.max() + 1), dtype=np.float32)
    onehot[np.arange(labels.shape[0]), labels] = 1
    return onehot

def preprocess_input_features(input_features):
    inputs = {}
    for name, column in input_features.items():
        inputs[name] = tf.keras.Input(
            shape=(1,), name=name, dtype=tf.float32)
    return inputs

def load_data(shuffle_random_state=None):
    pd_data = load_pd_data_to_pandas("train", stop_file=4)
    input_features, gen_info, labels = data_preparation(
        pd_data,
        shuffle_random_state=shuffle_random_state
    )
    columns = [
        "deltaR_det1_gen1",
        "deltaR_det1_gen2",
        "deltaR_det2_gen1",
        "deltaR_det2_gen2",
    ]
    input_features = input_features.drop(columns=columns)
    return input_features, gen_info, labels

def split_dataset(
    numeric_dict_ds, 
    train_percentage=train_percentage,
    validation_percentage=validation_percentage,
    SHUFFLE_BUFFER=SHUFFLE_BUFFER,
    BATCH_SIZE=128
):
    numeric_dict_batches = numeric_dict_ds
    nevents = len(list(numeric_dict_batches))
    train_size = np.floor(train_percentage*nevents)  
    valid_size = np.floor(validation_percentage*nevents)
    # test_size =  np.floor((1-train_percentage-validation_percentage)*nevents)

    print(f"have {nevents} events")
    print(f"train size: {train_size}")
    print(f"validation size: {valid_size}")
    # print(f"test size: {test_size}")

    train_data = numeric_dict_batches.take(train_size)
    testval_data = numeric_dict_batches.skip(train_size)
    val_data = testval_data.take(valid_size)
    rest = testval_data.skip(valid_size)
    test_data = rest.take(-1)

    # from IPython import embed; embed()
    # ntrain = len(list(train_data))
    # train_validation_split = np.floor((1-validation_percentage)*ntrain)
    # final_train_data = train_data.take(train_validation_split)
    # validation = test_data.skip(train_validation_split)

    return train_data.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE), val_data.batch(BATCH_SIZE), test_data.batch(BATCH_SIZE)

def transform_labels(labels):
    label_mean = np.mean(labels)
    label_std = np.std(labels)

    # Nur ein Teil, aber funktioniert alles nicht
    # label_mean["deltaR_det1_gen1"]=np.mean(labels["deltaR_det1_gen1"])
    # label_mean["deltaR_det1_gen2"]=np.mean(labels["deltaR_det1_gen2"])   
    # label_mean["deltaR_det2_gen1"]=np.mean(labels["deltaR_det2_gen1"]) 
    # label_mean["deltaR_det2_gen2"]=np.mean(labels["deltaR_det2_gen2"])

    # label_std["deltaR_det1_gen1"]=np.std(labels["deltaR_det1_gen1"])
    # label_std["deltaR_det1_gen2"]=np.std(labels["deltaR_det1_gen2"])   
    # label_std["deltaR_det2_gen1"]=np.std(labels["deltaR_det2_gen1"]) 
    # label_std["deltaR_det2_gen2"]=np.std(labels["deltaR_det2_gen2"])

    for c in labels.columns:
        labels[c] = (labels[c] - label_mean[c]) / label_std[c]

    return labels, label_mean, label_std

