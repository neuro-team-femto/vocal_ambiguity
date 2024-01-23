# This script uses praat to move between the formants of two words only differing by a single vowel
# it then puts these files through whisper to give log probabilities that they belong to either of the
# target words
# in this way we create a word with a vowel that is ambiguous and is close to equal log probabilities
# between the two words

# set up whisper
import numpy as np
import sys

import json 

# we load from a local whisper as modifications were made to extract log probs of specific words
sys.path.insert(1, './whisper')

import torch
import whisper

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

# medium is multilingual and in both French and English is the smallest model where we do not
#see a signifigant drop in performance in French
model = whisper.load_model('medium')

print(
  f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
  f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

#set up praat and parselmouth
import parselmouth
from parselmouth.praat import call

MAX_FORMANT  = 4000
MAX_NUM_FORMANTS = 4
WINDOW_LENGTH = 0.04
STEP_SIZE = 10
LANGUAGE = 'french'
SOUND1 = {'pull': './sounds/tts/pull_flat.wav'}
SOUND2 = {'poule': './sounds/tts/poule_flat.wav'}
KEYS = {'poule': [' P','oule','.'], 'pull' : [' P','ule','.']}
RESULT_FILE_NAME = "french_results"

def resynthesize(sound, formant_object):
  """rebuild a sound from the source and formants

  Parameters
  ----------
  sound : praat sound object
    a full sound object to extract the source

  formant_object : praat formant object
    an extracted formant object

  Returns
  -------
  rebuilt_sound : praat sound object
    a resynthesized sound with the input formants and source of the input sound
  
  """
  #extract source from origin sound
  lpc_object = call(sound, 'To LPC (burg)', 16, 0.04, 0.005, 50)
  source = call([sound, lpc_object], 'Filter (inverse)')

  rebuilt_sound = call([source, formant_object], 'Filter')

  return rebuilt_sound

def get_formants(formant_object):
  """get the mean of the first 4 formants

  Parameters
  ----------
  formant_object : praat formant object
    an extracted formant object

  Returns
  -------
  formants : List[float]
    list of the means of the first 4 formants
  
  """
  f1 = call(formant_object, 'Get mean', 1, 0, 0, 'Hertz')
  f2 = call(formant_object, 'Get mean', 2, 0, 0, 'Hertz')
  f3 = call(formant_object, 'Get mean', 3, 0, 0, 'Hertz')
  f4 = call(formant_object, 'Get mean', 4, 0, 0, 'Hertz')

  formants = [f1,f2,f3,f4]

  return formants

def check_asr_prob(audiofile: str, keys: list[str], language: str, ground_truth: str):
  """Get the log probability of specified words from Whisper for a single word audio input

  Parameters
  ----------
  audiofile : str
    The path to the audio file

  keys : List[str]
    a list of the words you would like the log probabilites for

  language : str
    the language to encode and decode the audio

  Returns
  -------
  prediction : str
    the predicted word, that with the highest log probability

  key_results : Dict{[str:float]}
    a dictionary containing the provided key words and their resulting log probabilities
  """
  # load audio and pad/trim it to fit 30 seconds
  audio = whisper.load_audio(audiofile)
  audio = whisper.pad_or_trim(audio)
  
  # make log-Mel spectrogram and move to the same device as the model
  mel = whisper.log_mel_spectrogram(audio).to(model.device)
  
  # decode the audio
  # predict without timestamps for short-form transcription
  options = whisper.DecodingOptions(
    language=language,
    without_timestamps=True,
    keys = keys,
    sample_len = 3,
    ground_truth = ground_truth
  )
  result = whisper.decode(model, mel, options)
  
  #extract the key strings and resulting log probabilities
  key_results = {}
  for key in result.keylogprobs:
    key_results[key] = result.keylogprobs[key]['sum_log_probs'][0].item()

  return key_results

def modify_formant(
  sound,
  formant_object,
  formant: dict[float],
  language: str,
  ground_truth: str,
  keys: dict[list[str]],
  current_minimum: dict[float],
  log_prob_grid: dict[str, list]
):
  """move formants, resynthesize and get ars prediction on the new audio

  Parameters
  ----------
  sound : praat sound object
    The origin audio

  formant_object : praat formant object
    the input formants

  formant : dict[float]
    list of current formant values to set

  language : str
    the language to encode and decode the audio

  ground_truth : str
    the original word in the input audio

  current_minimum : dict[float]
    current minimum log prob difference file name and value of the difference

  Returns
  -------
  current_minimum : dict[float]
    the current minimum log prob difference that may have been updated

  """

  audio_file = f"./sounds/tts/formant_tests/{ground_truth}_f1_{formant['f1']}_f2_{formant['f2']}_f3_{formant['f3']}_f4_{formant['f4']}.wav"

  #move F1
  call(
    formant_object,
    'Formula (frequencies)',
    f"if row = 1 then {formant['f1']} else self fi"
  )
  #move F2
  call(
    formant_object,
    'Formula (frequencies)',
    f"if row = 2 then {formant['f2']} else self fi"
  )

  #build and save audio file with new formants
  resynthesize(sound, formant_object).save(audio_file,'WAV')

  sound1 = parselmouth.Sound(audio_file)
  resampled_sound1 = sound1.resample(new_frequency = 10000)
  formant_object1 = resampled_sound1.to_formant_burg(
    maximum_formant = 4000,
    max_number_of_formants = 4,
    window_length = 0.04
  )

  hold_formants = get_formants(formant_object1)

  #check new file with ars
  result = check_asr_prob(audio_file, keys, language, ground_truth)
  log_prob_diff = abs(list(result.values())[0] - list(result.values())[1])

  #update current minimum difference in log probs
  if log_prob_diff < list(current_minimum.values())[0]:
    current_minimum = {audio_file : log_prob_diff}

  log_prob_grid[ground_truth]['f1'].append(hold_formants[0])
  log_prob_grid[ground_truth]['f2'].append(hold_formants[1])
  log_prob_grid[ground_truth]['log_prob_diff'].append(log_prob_diff)

  return formant_object, current_minimum, log_prob_grid


###### check intial ARS prediction to get basline log prob differences ######
#############################################################################

result = check_asr_prob(list(SOUND1.values())[0], KEYS, LANGUAGE, list(SOUND1.keys())[0])
log_prob_diff = abs(list(result.values())[0] - list(result.values())[1])

current_minimum = {list(SOUND1.values())[0] : log_prob_diff}
result = check_asr_prob(list(SOUND2.values())[0], KEYS, LANGUAGE, list(SOUND2.keys())[0])
log_prob_diff = abs(list(result.values())[0] - list(result.values())[1])

if log_prob_diff < list(current_minimum.values())[0]:
  current_minimum = {list(SOUND2.values())[0] : log_prob_diff}

################ extract the formants and build search grid ################
############################################################################

sound1 = parselmouth.Sound(list(SOUND1.values())[0])
resampled_sound1 = sound1.resample(new_frequency = 10000)

sound2 = parselmouth.Sound(list(SOUND2.values())[0])
resampled_sound2 = sound2.resample(new_frequency = 10000)

# max formant 4600 female, 4000 male 
formant_object1 = resampled_sound1.to_formant_burg(
  maximum_formant = MAX_FORMANT,
  max_number_of_formants = MAX_NUM_FORMANTS,
  window_length = WINDOW_LENGTH
)

formant_object2 = resampled_sound2.to_formant_burg(
  maximum_formant = MAX_FORMANT,
  max_number_of_formants = MAX_NUM_FORMANTS,
  window_length = WINDOW_LENGTH
)

formants1 = get_formants(formant_object1)
formants2 = get_formants(formant_object2)

# define search matrix
min_f1 = min(formants1[0], formants2[0])
max_f1 = max(formants1[0], formants2[0])
min_f2 = min(formants1[1], formants2[1])
max_f2 = max(formants1[1], formants2[1])

f1_array = np.arange(min_f1, max_f1, STEP_SIZE)
f1_array = np.append(f1_array, max_f1)
f2_array = np.arange(min_f2, max_f2, STEP_SIZE)
f2_array = np.append(f2_array, max_f2)

#check the direction of interation for the first sound, assume the second is opposite
direction = []
for i,f in enumerate(formants1):
  if formants1[i] > formants2[i]:
    direction.append('down')
  else:
    direction.append('up')

# store as dicts for easy access in loop
formants1 = {'f1' : formants1[0], 'f2' : formants1[1], 'f3' : formants1[2], 'f4' : formants1[3]}
formants2 = {'f1' : formants2[0], 'f2' : formants2[1], 'f3' : formants2[2], 'f4' : formants2[3]}

########################### conduct search ################################
###########################################################################

#save all differences in the grid
log_prob_grid = {
  list(SOUND1.keys())[0]: {'f1' : [], 'f2' : [], 'log_prob_diff' : []}, 
  list(SOUND2.keys())[0]: {'f1' : [], 'f2' : [], 'log_prob_diff' : []}
}

for i,formant1 in enumerate(f1_array):
  #move in the correct direction
  if direction == 'up':
    formants1['f1'] = formant1
    formants2['f1'] = f1_array[-i-1]
  else:
    formants1['f1'] = f1_array[-i-1]
    formants2['f1'] = formant1

  for k, formant2 in enumerate(f2_array):
    if direction == 'up':
      formants1['f2'] = formant2
      formants2['f2'] = f2_array[-k-1]
    else:
      formants1['f2'] = f2_array[-k-1]
      formants2['f2'] = formant2

    formant_object1, current_minimum, log_prob_grid = modify_formant(
      resampled_sound1,
      formant_object1,
      formants1,
      LANGUAGE,
      list(SOUND1.keys())[0],
      KEYS,
      current_minimum,
      log_prob_grid
    )
    formant_object2, current_minimum, log_prob_grid = modify_formant(
      resampled_sound2,
      formant_object2,
      formants2,
      LANGUAGE,
      list(SOUND2.keys())[0],
      KEYS,
      current_minimum,
      log_prob_grid
    )

# Convert and write JSON object to file
with open(f"{RESULT_FILE_NAME}.json", "w") as outfile:
    json.dump(log_prob_grid, outfile)
