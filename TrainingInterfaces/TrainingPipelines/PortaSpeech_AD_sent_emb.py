import time

import torch
import wandb

from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeech import PortaSpeech
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.portaspeech_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR
from Preprocessing.sentence_embeddings.STSentenceEmbeddingExtractor import STSentenceEmbeddingExtractor


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    name = "03_PortaSpeech_AD_sent_emb_a07_loss"
    """
    a01: integrate before encoder
    a02: integrate before encoder and decoder
    a03: integrate before encoder and decoder and postnet
    a04: integrate before each encoder layer
    a05: integrate before each encoder and decoder layer
    a06: integrate before each encoder and decoder layer and postnet
    a07: concatenate with style embedding and apply projection
    loss: additionally use sentence style loss
    """

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    sentence_embedding_extractor = STSentenceEmbeddingExtractor(model='camembert')
    sentence_embedding_extractor_name = 'STCamembert'

    train_set = prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023ad"),
                                          lang="fr",
                                          save_imgs=False,
                                          sentence_embedding_extractor=sentence_embedding_extractor,
                                          sentence_embedding_extractor_name=sentence_embedding_extractor_name)
    
    del sentence_embedding_extractor

    if "a01" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=768, sent_embed_encoder=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=768, sent_embed_encoder=True)
    if "a02" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=768, sent_embed_encoder=True, sent_embed_decoder=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=768, sent_embed_encoder=True, sent_embed_decoder=True)
    if "a03" in name:
        #TODO
        pass
    if "a04" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=768, sent_embed_encoder=True, sent_embed_each=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=768, sent_embed_encoder=True, sent_embed_each=True)
    if "a05" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=768, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_each=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=768, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_each=True)
    if "a06" in name:
        #TODO
        pass
    if "a07" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=768, concat_sent_style=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=768, concat_sent_style=True)

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
               eval_lang="fr",
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb,
               postnet_start_steps=16000,
               phase_2_steps=0)
    if use_wandb:
        wandb.finish()
