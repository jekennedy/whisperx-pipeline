INPUT_VALIDATIONS = {
    'audio_file': {'type': str,  'required': True},
    'language':   {'type': str,  'required': False, 'default': None},
    'language_detection_min_prob': {'type': float, 'required': False, 'default': 0},
    'language_detection_max_tries': {'type': int,   'required': False, 'default': 5},
    'initial_prompt': {'type': str, 'required': False, 'default': None},
    'batch_size':     {'type': int, 'required': False, 'default': 64},
    'beam_size':      {'type': int, 'required': False, 'default': None},
    'temperature':    {'type': float,'required': False, 'default': 0.0},
    'patience':       {'type': float,'required': False, 'default': None},
    'length_penalty': {'type': float,'required': False, 'default': None},
    'no_speech_threshold':        {'type': float,'required': False, 'default': None},
    'log_prob_threshold':         {'type': float,'required': False, 'default': None},
    'compression_ratio_threshold':{'type': float,'required': False, 'default': None},
    'vad_onset':      {'type': float,'required': False, 'default': 0.500},
    'vad_offset':     {'type': float,'required': False, 'default': 0.363},
    'align_output':   {'type': bool, 'required': False, 'default': False},
    'diarization':    {'type': bool, 'required': False, 'default': False},
    'huggingface_access_token': {'type': str, 'required': False, 'default': None},
    'min_speakers':   {'type': int,  'required': False, 'default': None},
    'max_speakers':   {'type': int,  'required': False, 'default': None},
    'debug':          {'type': bool, 'required': False, 'default': False},
    'speaker_verification': {'type': bool, 'required': False, 'default': False},
    'speaker_samples': {'type': list, 'required': False, 'default': []},

    # add this so your payload can specify the Whisper model explicitly
    'model': {'type': str, 'required': False, 'default': 'large-v3'},

    # optional extras if your worker supports them
    # 'return_subtitles': {'type': bool, 'required': False, 'default': True},
    # 'return_srt':       {'type': bool, 'required': False, 'default': True},
    # 'return_vtt':       {'type': bool, 'required': False, 'default': True},
}
