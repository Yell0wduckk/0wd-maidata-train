IDENTITY = {'schema': 'maimai.row_field_ar.v1', 'width': 256,
            'stages': ['kind', 'position_flags', 'route', 'duration'],
            'training': 'teacher-forced-predecessors-only', 'decoder_passes': 1}

def parameter_count(vocabulary=None):
    sizes=(8,8,35,10) if vocabulary is None else tuple(len(vocabulary.fields[n].values) for n in ('event','kind','pad','start_pad'))
    return sum(sizes)*256+256*6+70*256+4*(512+2*(256*256+256))
