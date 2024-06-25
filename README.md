[[Paper]](https://arxiv.org/pdf/2406.05515)
[[Website]](https://rosielab.github.io/vocal_ambiguity/)

Here lies the code for the Interpeech 2024 publication: **Mmm whatcha say? Uncovering distal and proximal context effects in first and
second-language word perception using psychophysical reverse correlation**

Here we used psychophysical reverse correlation to determine what changes in duration and pitch control the perception of difficult vowels for English and French speakers in English and French. The work is currently being expanded to include Mandarin and Japanese first language speakers in English.

In this repo you will find: *Pilot* containing the code for the initial pilot study and *Interspeech* containing the code for the above paper. This work is still in progress and as the project continues the repository will grow.

## Interspeech

* Experiment folders - code for GUI to run the experiment and the audio files used
* Revcor avc - this contains the results and the code to calculate and plot the resulting kernels
* Stimulus generation - this contains *CLEESE* and a modified whisper *Whisper*
    * CLEESE makes random transformations on the audio filed to generate our stimuli. The full repo is [here](https://github.com/neuro-team-femto/cleese)
    * [Whisper](https://github.com/openai/whisper) was modified to provide log probabilities of specific words in order to generate mixed vowels for the stimuli