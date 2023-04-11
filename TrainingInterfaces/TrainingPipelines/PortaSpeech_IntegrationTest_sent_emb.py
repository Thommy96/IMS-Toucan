"""
This is basically an integration test
"""

import time

import torch
import wandb
import tensorflow as tf

from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeech import PortaSpeech
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.portaspeech_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"1,{gpu_id}"
        device = torch.device(f"cuda:1")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    name = "05_PortaSpeech_IntegrationTest_a07"
    """
    a01: integrate before encoder
    a02: integrate before encoder and decoder
    a03: integrate before encoder and decoder and postnet
    a04: integrate before each encoder layer
    a05: integrate before each encoder and decoder layer
    a06: integrate before each encoder and decoder layer and postnet
    a07: concatenate with style embedding and apply projection
    a08: concatenate with style embedding
    loss: additionally use sentence style loss
    """

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    train_set = prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_integration_test(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "IntegrationTest"),
                                          lang="en",
                                          save_imgs=True)

    sent_embed_dim = 192

    if "a01" in name:
        if "loss" in name:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True)
    if "a02" in name:
        if "loss" in name:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True)
    if "a03" in name:
        if "loss" in name:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_postnet=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_postnet=True)
    if "a04" in name:
        if "loss" in name:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_each=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_each=True)
    if "a05" in name:
        if "loss" in name:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_each=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_each=True)
    if "a06" in name:
        if "loss" in name:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_each=True, sent_embed_postnet=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_each=True, sent_embed_postnet=True)
    if "a07" in name:
        if "loss" in name:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, concat_sent_style=True, use_concat_projection=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, concat_sent_style=True, use_concat_projection=True)
    if "a08" in name:
        if "loss" in name:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, concat_sent_style=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(sent_embed_dim=sent_embed_dim, concat_sent_style=True)

    if use_wandb:
        wandb.init(
            name=f"{name}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               datasets=[train_set],
               device=device,
               save_directory=save_dir,
               batch_size=8,
               eval_lang="en",
               warmup_steps=500,
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=finetune,
               resume=resume,
               phase_1_steps=1000,
               phase_2_steps=500,
               postnet_start_steps=600,
               use_wandb=use_wandb,
               use_sent_emb=True)
    if use_wandb:
        wandb.finish()
