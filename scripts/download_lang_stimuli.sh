# Description: Download the stimuli for the language localizer experiment
wget https://www.dropbox.com/sh/c9jhmsy4l9ly2xx/AACQ41zipSZFj9mFbDfJJ9c4a -O stimuli/language/data.zip
unzip stimuli/language/data.zip -d stimuli/language

# Remove unnecessary files
rm stimuli/language/*.zip
rm stimuli/language/*.mat 
rm stimuli/language/*.m 
rm stimuli/language/*.jpeg
rm -rf stimuli/language/archive_previous_langloc_versions
rm -rf stimuli/language/data