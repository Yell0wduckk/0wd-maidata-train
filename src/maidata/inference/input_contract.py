"""Input compatibility for inference using one shared audio grid."""


def check_shared_grid_config(teacher, decoder):
    """Compare input dimensions only; Encoder architectures remain independent."""
    differences = [f'{key}: teacher={teacher[key]}, M={decoder[key]}'
                   for key in ('samples_per_beat', 'mel_bins')
                   if teacher[key] != decoder[key]]
    if differences:
        raise ValueError('teacher and M cannot share one audio grid ('
                         + '; '.join(differences)
                         + '); use checkpoints with matching samples_per_beat and mel_bins')
