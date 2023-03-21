import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeech import PortaSpeech
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.portaspeech_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR
from Preprocessing.sentence_embeddings.STSentenceEmbeddingExtractor import STSentenceEmbeddingExtractor
from Preprocessing.sentence_embeddings.LEALLASentenceEmbeddingExtractor import LEALLASentenceEmbeddingExtractor


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

    name = "04_PortaSpeech_French_sent_emb_a07"
    """
    a01: integrate before encoder
    a02: integrate before encoder and decoder
    a03: integrate before encoder and decoder and postnet
    a04: integrate before each encoder layer
    a05: integrate before each encoder and decoder layer
    a06: integrate before each encoder and decoder layer and postnet
    a07: concatenate with style embedding and apply projection
    a08: concatenate with style embedding
    a09: integrate before each encoder layer and concatenate with style embedding and apply projection
    loss: additionally use sentence style loss
    """

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    #sentence_embedding_extractor = STSentenceEmbeddingExtractor(model='camembert')
    #sentence_embedding_extractor_name = 'STCamembert'

    #sentence_embedding_extractor = LEALLASentenceEmbeddingExtractor()
    sentence_embedding_extractor = "placeholder because tensorflow won't release memory after delete"
    sentence_embedding_extractor_name = 'LEALLA'

    french_datasets = list()

    french_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_siwis_subset(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "siwis"),
                                                     lang="fr",
                                                     sentence_embedding_extractor=sentence_embedding_extractor,
                                                     sentence_embedding_extractor_name=sentence_embedding_extractor_name))

    french_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023ad"),
                                                     lang="fr",
                                                     sentence_embedding_extractor=sentence_embedding_extractor,
                                                     sentence_embedding_extractor_name=sentence_embedding_extractor_name))

    french_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023neb"),
                                                     lang="fr",
                                                     sentence_embedding_extractor=sentence_embedding_extractor,
                                                     sentence_embedding_extractor_name=sentence_embedding_extractor_name))
    
    del sentence_embedding_extractor

    sent_embed_dim = 768
    if sentence_embedding_extractor_name == 'LEALLA':
        sent_embed_dim = 192

    if "a01" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True)
    if "a02" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True)
    if "a03" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_postnet=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_postnet=True)
    if "a04" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_each=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_each=True)
    if "a05" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_each=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_each=True)
    if "a06" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_each=True, sent_embed_postnet=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, sent_embed_encoder=True, sent_embed_decoder=True, sent_embed_each=True, sent_embed_postnet=True)
    if "a07" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, concat_sent_style=True, use_concat_projection=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, concat_sent_style=True, use_concat_projection=True)
    if "a08" in name:
        if "loss" in name:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, concat_sent_style=True, use_sent_style_loss=True)
        else:
            model = PortaSpeech(lang_embs=None, sent_embed_dim=sent_embed_dim, concat_sent_style=True)

    if use_wandb:
        wandb.init(
            name=f"{name}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               datasets=ConcatDataset(french_datasets),
               device=device,
               save_directory=save_dir,
               batch_size=8,
               eval_lang="fr",
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb,
               warmup_steps=8000,
               postnet_start_steps=9000,
               phase_1_steps=80000,
               phase_2_steps=0)
    if use_wandb:
        wandb.finish()
