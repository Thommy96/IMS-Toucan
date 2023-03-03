import time

import torch
import wandb

from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeech import PortaSpeech
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.portaspeech_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR
#from Preprocessing.sentence_embeddings.STSentenceEmbeddingExtractor import STSentenceEmbeddingExtractor
from Preprocessing.sentence_embeddings.CAMEMBERTSentenceEmbeddingExtractor import CAMEMBERTSentenceEmbeddingExtractor


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

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "PortaSpeech_NEB_encoder_Camembert")
    os.makedirs(save_dir, exist_ok=True)

    #sentence_embedding_extractor = STSentenceEmbeddingExtractor(model='camembert')
    sentence_embedding_extractor = CAMEMBERTSentenceEmbeddingExtractor(device=device)

    train_set = prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023neb_Camembert"),
                                          lang="fr",
                                          save_imgs=False,
                                          sentence_embedding_extractor=sentence_embedding_extractor)
    
    del sentence_embedding_extractor

    model = PortaSpeech(lang_embs=None, sent_embed_dim=768)
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
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
               sent_emb_integration='encoder',
               warmup_steps=100,
               phase_1_steps=200)
    if use_wandb:
        wandb.finish()
