"""
Taken from ESPNet, modified by Florian Lux
"""

import os
from abc import ABC

import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
from matplotlib.lines import Line2D

import Layers.ConditionalLayerNorm
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id


def float2pcm(sig, dtype='int16'):
    """
    https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def make_estimated_durations_usable_for_inference(xs, offset=1.0):
    return torch.clamp(torch.round(xs.exp() - offset), min=0).long()


def cut_to_multiple_of_n(x, n=4, return_diff=False, seq_dim=1):
    max_frames = x.shape[seq_dim] // n * n
    if return_diff:
        return x[:, :max_frames], x.shape[seq_dim] - max_frames
    return x[:, :max_frames]


def pad_to_multiple_of_n(x, n=4, seq_dim=1, pad_value=0):
    max_frames = ((x.shape[seq_dim] // n) + 1) * n
    diff = max_frames - x.shape[seq_dim]
    return torch.nn.functional.pad(x, [0, 0, 0, diff, 0, 0], mode="constant", value=pad_value)


def kl_beta(step_counter, kl_cycle_steps):
    return min([(1 / (kl_cycle_steps / ((step_counter % kl_cycle_steps) + 1))), 1.0]) * 0.01


@torch.inference_mode()
def plot_progress_spec(net,
                       device,
                       save_dir,
                       step,
                       lang,
                       default_emb,
                       before_and_after_postnet=False,
                       run_postflow=True):
    tf = ArticulatoryCombinedTextFrontend(language=lang)
    sentence = tf.get_example_sentence(lang=lang)
    if sentence is None:
        return None
    phoneme_vector = tf.string_to_tensor(sentence).squeeze(0).to(device)
    if run_postflow:
        spec, durations, pitch, energy = net.inference(text=phoneme_vector,
                                                       return_duration_pitch_energy=True,
                                                       utterance_embedding=default_emb,
                                                       lang_id=get_language_id(lang).to(device),
                                                       run_postflow=run_postflow)
    else:
        spec, durations, pitch, energy = net.inference(text=phoneme_vector,
                                                       return_duration_pitch_energy=True,
                                                       utterance_embedding=default_emb,
                                                       lang_id=get_language_id(lang).to(device))

    if before_and_after_postnet:
        # ToucanTTS case, because there it's more interesting
        spec_before, spec_after = spec

        spec = spec_before.transpose(0, 1).to("cpu").numpy()
        duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
        os.makedirs(os.path.join(save_dir, "spec_before"), exist_ok=True)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        lbd.specshow(spec,
                     ax=ax,
                     sr=16000,
                     cmap='GnBu',
                     y_axis='mel',
                     x_axis=None,
                     hop_length=256)
        ax.yaxis.set_visible(False)
        ax.set_xticks(duration_splits, minor=True)
        ax.xaxis.grid(True, which='minor')
        ax.set_xticks(label_positions, minor=False)
        phones = tf.get_phone_string(sentence, for_plot_labels=True)
        ax.set_xticklabels(phones)
        word_boundaries = list()
        for label_index, word_boundary in enumerate(phones):
            if word_boundary == "|":
                word_boundaries.append(label_positions[label_index])
        ax.vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.5)
        ax.vlines(x=word_boundaries, colors="orange", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.5)
        pitch_array = pitch.cpu().numpy()
        for pitch_index, xrange in enumerate(zip(duration_splits[:-1], duration_splits[1:])):
            if pitch_array[pitch_index] > 0.001:
                ax.hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="red",
                          linestyles="solid",
                          linewidth=1.0)
        ax.set_title(sentence)
        plt.savefig(os.path.join(os.path.join(save_dir, "spec_before"), f"{step}.png"))
        plt.clf()
        plt.close()

        spec = spec_after.transpose(0, 1).to("cpu").numpy()
        duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
        os.makedirs(os.path.join(save_dir, "spec_after"), exist_ok=True)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        lbd.specshow(spec,
                     ax=ax,
                     sr=16000,
                     cmap='GnBu',
                     y_axis='mel',
                     x_axis=None,
                     hop_length=256)
        ax.yaxis.set_visible(False)
        ax.set_xticks(duration_splits, minor=True)
        ax.xaxis.grid(True, which='minor')
        ax.set_xticks(label_positions, minor=False)
        phones = tf.get_phone_string(sentence, for_plot_labels=True)
        ax.set_xticklabels(phones)
        word_boundaries = list()
        for label_index, word_boundary in enumerate(phones):
            if word_boundary == "|":
                word_boundaries.append(label_positions[label_index])
        ax.vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
        ax.vlines(x=word_boundaries, colors="orange", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
        pitch_array = pitch.cpu().numpy()
        for pitch_index, xrange in enumerate(zip(duration_splits[:-1], duration_splits[1:])):
            if pitch_array[pitch_index] > 0.001:
                ax.hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="blue",
                          linestyles="solid",
                          linewidth=0.5)
        ax.set_title(sentence)
        plt.savefig(os.path.join(os.path.join(save_dir, "spec_after"), f"{step}.png"))
        plt.clf()
        plt.close()
        return os.path.join(os.path.join(save_dir, "spec_before"), f"{step}.png"), os.path.join(
            os.path.join(save_dir, "spec_after"), f"{step}.png")

    else:
        # FastSpeech case, standard
        spec = spec.transpose(0, 1).to("cpu").numpy()
        duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
        if not os.path.exists(os.path.join(save_dir, "spec")):
            os.makedirs(os.path.join(save_dir, "spec"))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        lbd.specshow(spec,
                     ax=ax,
                     sr=16000,
                     cmap='GnBu',
                     y_axis='mel',
                     x_axis=None,
                     hop_length=256)
        ax.yaxis.set_visible(False)
        ax.set_xticks(duration_splits, minor=True)
        ax.xaxis.grid(True, which='minor')
        ax.set_xticks(label_positions, minor=False)
        phones = tf.get_phone_string(sentence, for_plot_labels=True)
        ax.set_xticklabels(phones)
        word_boundaries = list()
        for label_index, word_boundary in enumerate(phones):
            if word_boundary == "|":
                word_boundaries.append(label_positions[label_index])
        ax.vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
        ax.vlines(x=word_boundaries, colors="orange", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
        pitch_array = pitch.cpu().numpy()
        for pitch_index, xrange in enumerate(zip(duration_splits[:-1], duration_splits[1:])):
            if pitch_array[pitch_index] > 0.001:
                ax.hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="blue",
                          linestyles="solid",
                          linewidth=0.5)
        ax.set_title(sentence)
        plt.savefig(os.path.join(os.path.join(save_dir, "spec"), f"{step}.png"))
        plt.clf()
        plt.close()
        return os.path.join(os.path.join(save_dir, "spec"), f"{step}.png")


@torch.inference_mode()
def plot_progress_spec_toucantts(net,
                                 device,
                                 save_dir,
                                 step,
                                 lang,
                                 default_emb,
                                 static_speaker_embed=False,
                                 sent_embs=None,
                                 word_embedding_extractor=None,
                                 run_postflow=True):
    tf = ArticulatoryCombinedTextFrontend(language=lang)
    sentence = tf.get_example_sentence(lang=lang)
    if sentence is None:
        return None
    if sent_embs is not None:
        try:
            sentence_embedding = sent_embs[sentence]
        except KeyError:
            sentence_embedding = sent_embs["neutral"][0]
    else:
        sentence_embedding = None
    if static_speaker_embed:
        speaker_id = torch.LongTensor([0])
    else:
        speaker_id = None
    if word_embedding_extractor is not None:
        word_embedding, _ = word_embedding_extractor.encode([sentence])
        word_embedding = word_embedding.squeeze()
    else:
        word_embedding = None
    phoneme_vector = tf.string_to_tensor(sentence).squeeze(0).to(device)
    if run_postflow:
        spec_before, spec_after, durations, pitch, energy = net.inference(text=phoneme_vector,
                                                                          return_duration_pitch_energy=True,
                                                                          utterance_embedding=default_emb,
                                                                          speaker_id=speaker_id,
                                                                          sentence_embedding=sentence_embedding,
                                                                          word_embedding=word_embedding,
                                                                          lang_id=get_language_id(lang).to(device),
                                                                          run_postflow=run_postflow)
    else:
        spec_before, spec_after, durations, pitch, energy = net.inference(text=phoneme_vector,
                                                                          return_duration_pitch_energy=True,
                                                                          utterance_embedding=default_emb,
                                                                          speaker_id=speaker_id,
                                                                          sentence_embedding=sentence_embedding,
                                                                          word_embedding=word_embedding,
                                                                          lang_id=get_language_id(lang).to(device),
                                                                          run_postflow=False)
    spec = spec_before.transpose(0, 1).to("cpu").numpy()
    duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
    os.makedirs(os.path.join(save_dir, "spec_before"), exist_ok=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    lbd.specshow(spec,
                 ax=ax,
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax.yaxis.set_visible(False)
    ax.set_xticks(duration_splits, minor=True)
    ax.xaxis.grid(True, which='minor')
    ax.set_xticks(label_positions, minor=False)
    phones = tf.get_phone_string(sentence, for_plot_labels=True)
    ax.set_xticklabels(phones)
    word_boundaries = list()
    for label_index, word_boundary in enumerate(phones):
        if word_boundary == "|":
            word_boundaries.append(label_positions[label_index])
    ax.vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
    ax.vlines(x=word_boundaries, colors="orange", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
    pitch_array = pitch.cpu().numpy()
    for pitch_index, xrange in enumerate(zip(duration_splits[:-1], duration_splits[1:])):
        if pitch_array[pitch_index] > 0.001:
            ax.hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="magenta",
                      linestyles="solid",
                      linewidth=1.0)
    ax.set_title(sentence)
    plt.savefig(os.path.join(os.path.join(save_dir, "spec_before"), f"{step}.png"))
    plt.clf()
    plt.close()

    spec = spec_after.transpose(0, 1).to("cpu").numpy()
    duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
    os.makedirs(os.path.join(save_dir, "spec_after"), exist_ok=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    lbd.specshow(spec,
                 ax=ax,
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax.yaxis.set_visible(False)
    ax.set_xticks(duration_splits, minor=True)
    ax.xaxis.grid(True, which='minor')
    ax.set_xticks(label_positions, minor=False)
    phones = tf.get_phone_string(sentence, for_plot_labels=True)
    ax.set_xticklabels(phones)
    word_boundaries = list()
    for label_index, word_boundary in enumerate(phones):
        if word_boundary == "|":
            word_boundaries.append(label_positions[label_index])
    ax.vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
    ax.vlines(x=word_boundaries, colors="orange", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
    pitch_array = pitch.cpu().numpy()
    for pitch_index, xrange in enumerate(zip(duration_splits[:-1], duration_splits[1:])):
        if pitch_array[pitch_index] > 0.001:
            ax.hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="magenta",
                      linestyles="solid",
                      linewidth=1.0)
    ax.set_title(sentence)
    plt.savefig(os.path.join(os.path.join(save_dir, "spec_after"), f"{step}.png"))
    plt.clf()
    plt.close()
    return os.path.join(os.path.join(save_dir, "spec_before"), f"{step}.png"), os.path.join(
        os.path.join(save_dir, "spec_after"), f"{step}.png")


def cumsum_durations(durations):
    out = [0]
    for duration in durations:
        out.append(duration + out[-1])
    centers = list()
    for index, _ in enumerate(out):
        if index + 1 < len(out):
            centers.append((out[index] + out[index + 1]) // 2)
    return out, centers


def delete_old_checkpoints(checkpoint_dir, keep=5):
    checkpoint_list = list()
    for el in os.listdir(checkpoint_dir):
        if el.endswith(".pt"):
            try:
                checkpoint_list.append(int(el.replace("checkpoint_", "").replace(".pt", "")))
            except ValueError:
                pass
    if len(checkpoint_list) <= keep:
        return
    else:
        checkpoint_list.sort(reverse=False)
        checkpoints_to_delete = [os.path.join(checkpoint_dir, "checkpoint_{}.pt".format(step)) for step in
                                 checkpoint_list[:-keep]]
        for old_checkpoint in checkpoints_to_delete:
            os.remove(os.path.join(old_checkpoint))


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function after loss.backwards() and unscaling as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


def get_most_recent_checkpoint(checkpoint_dir, verbose=True):
    checkpoint_list = list()
    for el in os.listdir(checkpoint_dir):
        if el.endswith(".pt") and el != "best.pt" and el != "embedding_function.pt":
            try:
                checkpoint_list.append(int(el.split(".")[0].split("_")[1]))
            except ValueError:
                pass
    if len(checkpoint_list) == 0:
        print("No previous checkpoints found, cannot reload.")
        return None
    checkpoint_list.sort(reverse=True)
    if verbose:
        print("Reloading checkpoint_{}.pt".format(checkpoint_list[0]))
    return os.path.join(checkpoint_dir, "checkpoint_{}.pt".format(checkpoint_list[0]))


def make_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """
    Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    if device is not None:
        seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=device)
    else:
        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """
    Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    """
    return ~make_pad_mask(lengths, xs, length_dim, device=device)


def initialize(model, init):
    """
    Initialize weights of a neural network module.

    Parameters are initialized using the given method or distribution.

    Args:
        model: Target.
        init: Method of initialization.
    """

    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init)
    # bias init
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()

    # reset some modules with default init
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding,
                          torch.nn.LayerNorm,
                          Layers.ConditionalLayerNorm.ConditionalLayerNorm,
                          Layers.ConditionalLayerNorm.SequentialWrappableConditionalLayerNorm
                          )):
            m.reset_parameters()


def pad_list(xs, pad_value):
    """
    Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def subsequent_mask(size, device="cpu", dtype=torch.bool):
    """
    Create mask for subsequent steps (size, size).

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)


class ScorerInterface:
    """
    Scorer interface for beam search.

    The scorer performs scoring of the all tokens in vocabulary.

    Examples:
        * Search heuristics
            * :class:`espnet.nets.scorers.length_bonus.LengthBonus`
        * Decoder networks of the sequence-to-sequence models
            * :class:`espnet.nets.pytorch_backend.nets.transformer.decoder.Decoder`
            * :class:`espnet.nets.pytorch_backend.nets.rnn.decoders.Decoder`
        * Neural language models
            * :class:`espnet.nets.pytorch_backend.lm.transformer.TransformerLM`
            * :class:`espnet.nets.pytorch_backend.lm.default.DefaultRNNLM`
            * :class:`espnet.nets.pytorch_backend.lm.seq_rnn.SequentialRNNLM`

    """

    def init_state(self, x):
        """
        Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return None

    def select_state(self, state, i, new_id=None):
        """
        Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label index to select a state if necessary

        Returns:
            state: pruned state

        """
        return None if state is None else state[i]

    def score(self, y, state, x):
        """
        Score new token (required).

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys

        """
        raise NotImplementedError

    def final_score(self, state):
        """
        Score eos (optional).

        Args:
            state: Scorer state for prefix tokens

        Returns:
            float: final score

        """
        return 0.0


class BatchScorerInterface(ScorerInterface, ABC):

    def batch_init_state(self, x):
        """
        Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return self.init_state(x)

    def batch_score(self, ys, states, xs):
        """
        Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        scores = list()
        outstates = list()
        for i, (y, state, x) in enumerate(zip(ys, states, xs)):
            score, outstate = self.score(y, state, x)
            outstates.append(outstate)
            scores.append(score)
        scores = torch.cat(scores, 0).view(ys.shape[0], -1)
        return scores, outstates


def to_device(m, x):
    """Send tensor into the device of the module.
    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.
    Returns:
        Tensor: Torch tensor located in the same place as torch module.
    """
    if isinstance(m, torch.nn.Module):
        device = next(m.parameters()).device
    elif isinstance(m, torch.Tensor):
        device = m.device
    else:
        raise TypeError(
            "Expected torch.nn.Module or torch.tensor, " f"bot got: {type(m)}"
        )
    return x.to(device)


def curve_smoother(curve):
    if len(curve) < 3:
        return curve
    new_curve = list()
    for index in range(len(curve)):
        if curve[index] != 0:
            current_value = curve[index]
            if index != len(curve) - 1:
                if curve[index + 1] != 0:
                    next_value = curve[index + 1]
                else:
                    next_value = curve[index]
            if index != 0:
                if curve[index - 1] != 0:
                    prev_value = curve[index - 1]
                else:
                    prev_value = curve[index]
            else:
                prev_value = curve[index]
            smooth_value = (current_value * 3 + prev_value + next_value) / 5
            new_curve.append(smooth_value)
        else:
            new_curve.append(0)
    return new_curve

def get_speakerid_from_path(path):
    speaker_id = None
    if "EmoVDB" in path:
        if "bea" in path:
            speaker_id = 0
        if "jenie" in path:
            speaker_id = 1
        if "josh" in path:
            speaker_id = 2
        if "sam" in path:
            speaker_id = 3
    if "Emotional_Speech_Dataset_Singapore" in path:
        speaker = os.path.split(os.path.split(os.path.dirname(path))[0])[1]
        if speaker == "0011":
            speaker_id = 0
        if speaker == "0012":
            speaker_id = 1
        if speaker == "0013":
            speaker_id = 2
        if speaker == "0014":
            speaker_id = 3
        if speaker == "0015":
            speaker_id = 4
        if speaker == "0016":
            speaker_id = 5
        if speaker == "0017":
            speaker_id = 6
        if speaker == "0018":
            speaker_id = 7
        if speaker == "0019":
            speaker_id = 8
        if speaker == "0020":
            speaker_id = 9
    if "CREMA_D" in path:
        speaker = os.path.basename(path).split('_')[0]
        for i, sp_id in enumerate(range(1001, 1092)):
            if int(speaker) == sp_id:
                speaker_id = i
    if "RAVDESS" in path:
        speaker = os.path.split(os.path.dirname(path))[1].split('_')[1]
        speaker_id = int(speaker) - 1
    
    if speaker_id is None:
        raise TypeError('speaker id could not be extracted from filename')

    return speaker_id

def get_emotion_from_path(path):
    emotion = None
    if "EmoV_DB" in path or "EmoVDB" in path or "EmoVDB_Sam" in path:
        emotion = os.path.splitext(os.path.basename(path))[0].split("-16bit")[0].split("_")[0].lower()
        if emotion == "amused":
            emotion = "joy"
        if emotion == "sleepiness":
            raise NameError("emotion sleepiness should not be included")
    if "CREMA_D" in path:
        emotion = os.path.splitext(os.path.basename(path))[0].split('_')[2]
        if emotion == "ANG":
            emotion = "anger"
        if emotion == "DIS":
            emotion = "disgust"
        if emotion == "FEA":
            emotion = "fear"
        if emotion == "HAP":
            emotion = "joy"
        if emotion == "NEU":
            emotion = "neutral"
        if emotion == "SAD":
            emotion = "sadness"
    if "Emotional_Speech_Dataset_Singapore" in path:
        emotion = os.path.basename(os.path.dirname(path)).lower()
        if emotion == "angry":
            emotion = "anger"
        if emotion == "happy":
            emotion = "joy"
        if emotion == "sad":
            emotion = "sadness"
    if "RAVDESS" in path:
        emotion = os.path.splitext(os.path.basename(path))[0].split('-')[2]
        if emotion == "01":
            emotion = "neutral"
        if emotion == "02":
            raise NameError("emotion calm should not be included")
        if emotion == "03":
            emotion = "joy"
        if emotion == "04":
            emotion = "sadness"
        if emotion == "05":
            emotion = "anger"
        if emotion == "06":
            emotion = "fear"
        if emotion == "07":
            emotion = "disgust"
        if emotion == "08":
            emotion = "surprise"
    if "TESS" in path:
        emotion = os.path.split(os.path.dirname(path))[1].split('_')[1].lower()
        if emotion == "angry":
            emotion = "anger"
        if emotion == "happy":
            emotion = "joy"
        if emotion == "sad":
            emotion = "sadness"
    if "LJSpeech" in path:
        emotion = "neutral"

    if emotion is None:
        raise TypeError('emotion could not be extracted from filename')
    
    return emotion

def get_speakerid_from_path(path, libri_speakers):
    speaker_id = None
    if "LJSpeech" in path:
        speaker_id = 0
    if "EmoVDB" in path:
        if "bea" in path:
            speaker_id = 0
        if "jenie" in path:
            speaker_id = 1
        if "josh" in path:
            speaker_id = 2
        if "sam" in path:
            speaker_id = 3
    if "Emotional_Speech_Dataset_Singapore" in path:
        speaker = os.path.split(os.path.split(os.path.dirname(path))[0])[1]
        if speaker == "0011":
            speaker_id = 0
        if speaker == "0012":
            speaker_id = 1
        if speaker == "0013":
            speaker_id = 2
        if speaker == "0014":
            speaker_id = 3
        if speaker == "0015":
            speaker_id = 4
        if speaker == "0016":
            speaker_id = 5
        if speaker == "0017":
            speaker_id = 6
        if speaker == "0018":
            speaker_id = 7
        if speaker == "0019":
            speaker_id = 8
        if speaker == "0020":
            speaker_id = 9
    if "CREMA_D" in path:
        speaker = os.path.basename(path).split('_')[0]
        for i, sp_id in enumerate(range(1001, 1092)):
            if int(speaker) == sp_id:
                speaker_id = i
    if "RAVDESS" in path:
        speaker = os.path.split(os.path.dirname(path))[1].split('_')[1]
        speaker_id = int(speaker) - 1
    if "LibriTTS_R" in path:
        speaker = os.path.split(os.path.split(os.path.dirname(path))[0])[1]
        speaker_id = libri_speakers.index(int(speaker))
    if "TESS" in path:
        speaker = os.path.split(os.path.dirname(path))[1].split('_')[0]
        if speaker == "OAF":
            speaker_id = 0
        if speaker == "YAF":
            speaker_id = 1
    
    if speaker_id is None:
        raise TypeError('speaker id could not be extracted from filename')

    return int(speaker_id)

def get_speakerid_from_path_all(path, libri_speakers):
    speaker_id = None
    if "LJSpeech" in path: # 1 speaker
        # 0
        speaker_id = 0
    if "CREMA_D" in path: # 91 speakers
        # 1 - 91
        speaker = os.path.basename(path).split('_')[0]
        speaker_id = int(speaker) - 1001 + 1
    if "RAVDESS" in path: # 24 speakers
        # 92 - 115
        speaker = os.path.split(os.path.dirname(path))[1].split('_')[1]
        speaker_id = int(speaker) - 1 + 1 + 91
    if "EmoVDB" in path: # 4 speakers
        # 116 - 119
        if "bea" in path:
            speaker_id = 0 + 1 + 91 + 24
        if "jenie" in path:
            speaker_id = 1 + 1 + 91 + 24
        if "josh" in path:
            speaker_id = 2 + 1 + 91 + 24
        if "sam" in path:
            speaker_id = 3 + 1 + 91 + 24
    if "Emotional_Speech_Dataset_Singapore" in path: # 10 speakers
        # 120 - 129
        speaker = os.path.split(os.path.split(os.path.dirname(path))[0])[1]
        if speaker == "0011":
            speaker_id = 0 + 1 + 91 + 24 + 4
        if speaker == "0012":
            speaker_id = 1 + 1 + 91 + 24 + 4
        if speaker == "0013":
            speaker_id = 2 + 1 + 91 + 24 + 4
        if speaker == "0014":
            speaker_id = 3 + 1 + 91 + 24 + 4
        if speaker == "0015":
            speaker_id = 4 + 1 + 91 + 24 + 4
        if speaker == "0016":
            speaker_id = 5 + 1 + 91 + 24 + 4
        if speaker == "0017":
            speaker_id = 6 + 1 + 91 + 24 + 4
        if speaker == "0018":
            speaker_id = 7 + 1 + 91 + 24 + 4
        if speaker == "0019":
            speaker_id = 8 + 1 + 91 + 24 + 4
        if speaker == "0020":
            speaker_id = 9 + 1 + 91 + 24 + 4
    if "LibriTTS_R" in path: # 1230 speakers
        # 130 - 1359
        speaker = os.path.split(os.path.split(os.path.dirname(path))[0])[1]
        speaker_id = libri_speakers.index(int(speaker)) + 1 + 91 + 24 + 4 + 10
    if "TESS" in path: # 2 speakers
        # 1360 - 1361
        speaker = os.path.split(os.path.dirname(path))[1].split('_')[0]
        if speaker == "OAF":
            speaker_id = 0 + 1 + 91 + 24 + 4 + 10 + 1230
        if speaker == "YAF":
            speaker_id = 1 + 1 + 91 + 24 + 4 + 10 + 1230
    
    if speaker_id is None:
        raise TypeError('speaker id could not be extracted from filename')

    return speaker_id

def get_speakerid_from_path_all2(path, libri_speakers):
    speaker_id = None
    if "LJSpeech" in path: # 1 speaker
        # 0
        speaker_id = 0
    if "RAVDESS" in path: # 24 speakers
        # 1 - 24
        speaker = os.path.split(os.path.dirname(path))[1].split('_')[1]
        speaker_id = int(speaker) - 1 + 1
    if "Emotional_Speech_Dataset_Singapore" in path: # 10 speakers
        # 25 - 34
        speaker = os.path.split(os.path.split(os.path.dirname(path))[0])[1]
        if speaker == "0011":
            speaker_id = 0 + 1 + 24
        if speaker == "0012":
            speaker_id = 1 + 1 + 24
        if speaker == "0013":
            speaker_id = 2 + 1 + 24
        if speaker == "0014":
            speaker_id = 3 + 1 + 24
        if speaker == "0015":
            speaker_id = 4 + 1 + 24
        if speaker == "0016":
            speaker_id = 5 + 1 + 24
        if speaker == "0017":
            speaker_id = 6 + 1 + 24
        if speaker == "0018":
            speaker_id = 7 + 1 + 24
        if speaker == "0019":
            speaker_id = 8 + 1 + 24
        if speaker == "0020":
            speaker_id = 9 + 1 + 24
    if "LibriTTS_R" in path: # 1230 speakers
        # 35 - 1264
        speaker = os.path.split(os.path.split(os.path.dirname(path))[0])[1]
        speaker_id = libri_speakers.index(int(speaker)) + 1 + 24 + 10
    if "TESS" in path: # 2 speakers
        # 1265, 1266
        speaker = os.path.split(os.path.dirname(path))[1].split('_')[0]
        if speaker == "OAF":
            speaker_id = 0 + 1 + 24 + 10 + 1230
        if speaker == "YAF":
            speaker_id = 1 + 1 + 24 + 10 + 1230
    
    if speaker_id is None:
        raise TypeError('speaker id could not be extracted from filename')

    return speaker_id


if __name__ == '__main__':
    data = np.random.randn(50)
    plt.plot(data, color="b")
    smooth = curve_smoother(data)
    plt.plot(smooth, color="g")
    plt.show()
