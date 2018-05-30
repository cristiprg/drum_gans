import os
import sys
import xml.dom.minidom
import numpy as np
import pandas as pd
import madmom
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def spec(file):
    """
    This is copy-paste from ADTLib. So the feature-exctraction is the same.
    https://github.com/CarlSouthall/ADTLib/blob/master/ADTLib/utils/__init__.py#L19
    """
    return madmom.audio.spectrogram.Spectrogram(file, frame_size=2048, hop_size=512, fft_size=2048,num_channels=1)


def second_to_spectrogram_frames(second, sample_rate=44100, spectrogram_frame_size=2048, hop_size=512):
    """
    One moment in time in the wav file corresponds to several frames in the spectrogram.
    The parameter "second" is a real number and represents the moment in time, not the duration of one second.
    """

    spectrogram_frames = np.array([], dtype=np.int32)
    first_frame = int(second * sample_rate / hop_size)

    for i in range(spectrogram_frame_size / hop_size):
        spectrogram_frames = np.append(spectrogram_frames, first_frame + i)

    return spectrogram_frames


def binarize_onset_array(onsets, length):
    onsets_binary = np.zeros(length)
    for i in range(len(onsets)):
        onsets_binary[onsets[i]] = 1

    return onsets_binary


def get_onsets(xmlAnnotationFile):
    """
    Reads an xml annotation file (as provided by the SMT Drums dataset) and returns for each onset the
    equivalent spectrogram frames where it can be found.
    """
    onsets = {
        "HH": np.array([], dtype=np.int32),
        "KD": np.array([], dtype=np.int32),
        "SD": np.array([], dtype=np.int32)
    }

    doc = xml.dom.minidom.parse(xmlAnnotationFile)
    events = doc.getElementsByTagName("event")

    for event in events:
        sec = event.getElementsByTagName("onsetSec")[0].firstChild.data  # extract the annotated time
        instrument = event.getElementsByTagName("instrument")[0].firstChild.data  # extract the instrument

        try:
            onsets[instrument] = np.append(onsets[instrument], second_to_spectrogram_frames(float(sec)))
        except KeyError as e:
            sys.stderr.write("Error reading xml file: unkown instrument '" + instrument + "'.\n")
            raise type(e)(e.message + ", unkown instrument '" + instrument + "'.")

    return onsets["HH"], onsets["KD"], onsets["SD"]


def build_smt_drums_dataset(path, hdf5_path=None):
    """
    Reads the XML and WAV files in the specified path and constructs an array pd dataframes, one for each wav file.
    :param path: Path to the SMT root folder
    :param hdf5_path: Path where to save the hdf5 file
    :return: an array of pandas dataframes, each frame is basically the spectrogram + annotations for each frame
    """

    xmlAnnotationsFolder = os.path.join(path, "annotation_xml/")
    wavesFolder = os.path.join(path, "audio/")
    xmlFiles = os.listdir(xmlAnnotationsFolder)

    spectrograms = []


    for xmlFile in xmlFiles:

        # Get the spectrogram of the MIX wav file
        wavFile = os.path.join(wavesFolder, os.path.splitext(xmlFile)[0] + ".wav")
        if not os.path.isfile(wavFile):
            raise "Error: could not find the wav file for xml " + xmlFile

        try:
            spectrogram = spec(wavFile)
        #             spectrogram /= spectrogram.max() # TODO: is this OK? NO - do subtract mean and divide by stddev, for each freq (dimension/variable), after the split
        except madmom.audio.signal.LoadAudioFileError:
            print "Warning: could not get spectrogram for ", wavFile
            continue

            # Get the onsets for each instrument
        HH, KD, SD = get_onsets(os.path.join(xmlAnnotationsFolder, xmlFile))

        # Small sanity check:
        spectrogram_num_frames = spectrogram.shape[0]
        if max(HH.max(), KD.max(), SD.max()) > spectrogram_num_frames:
            raise Exception("Onset detected at non-existing frame!")

        # One hot encoding the labels
        HH = binarize_onset_array(HH, spectrogram_num_frames)
        KD = binarize_onset_array(KD, spectrogram_num_frames)
        SD = binarize_onset_array(SD, spectrogram_num_frames)

        # Now, HH, KD, SD are the labels for training and spectrogram contains the features.

        # Create new pandas dataframe with the spectrogram and the annotations
        df = pd.DataFrame(spectrogram)
        df['HH'] = HH
        df['KD'] = KD
        df['SD'] = SD

        spectrograms.append(df)

        if hdf5_path is not None:
            df.to_hdf(hdf5_path, wavFile.replace("#MIX.wav", ""))

    return spectrograms

def load_smt_dataset(hd5_path):
    """
    :param hd5_path:
    :return: (an array of pandas dataframes, total number of spectrogram frames)
    """
    try:
        total_len = 0
        store = pd.HDFStore(hd5_path)
        spectrograms = [store.get(key) for key in store.keys()] #pd.read_hdf(hd5_path, "/*")
        store.close()
    except Exception as e:
        raise e

    for spec in spectrograms:
        total_len += spec.shape[0]

    return spectrograms, total_len


# Usage examples:
# spectrograms = build_smt_drums_dataset("/mnt/antares_raid/home/cristiprg/notebooks/SMT_DRUMS", "./smt_spectrograms.h5")
# spectrograms = load_smt_dataset("./smt_spectrograms.h5")
