"""Deterministic occurrence/history randomness, independent of loader workers."""
import random


def history_rng(seed, epoch, window_id, occurrence):
    return random.Random(f'decoder-history:{seed}:{epoch}:{window_id}:{occurrence}')


def history_dropped(config, training, epoch, row):
    return training and history_rng(config['decoder_train_seed'], epoch,
        row['window_id'], row['occurrence']).random() < config['decoder_history_dropout']
