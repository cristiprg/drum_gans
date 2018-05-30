import numpy as np
import pandas as pd
import gc

def get_spectrograms(path="/mnt/antares_raid/home/cristiprg/notebooks/smt_drums.csv", target="HH", rebalance_method="nothing"):
    """
    :param path:
    :param target:
    :param rebalance_method:
    :return: X, Y from the specified file
    """


    # Takes about 40 seconds ...
    df = pd.read_csv(path)
    df.rename(columns={'# HH': 'HH'}, inplace=True)

    if target is 'HH':
        df.drop(columns=['KD', 'SD'], inplace=True)
    elif target is 'KD':
        df.drop(columns=['HH', 'SD'], inplace=True)
    elif target is 'SD':
        df.drop(columns=['HH', 'KD'], inplace=True)
    else:
        raise ValueError("Unkown label")

    class_imbalance = df[df[target] == 1].shape[0] * 1.0 / df.shape[0]
    print("class_imbalance: ", class_imbalance)

    if rebalance_method is "downsampling":
        df.groupby(target).count()
        ones = df[df[target] == 1]
        zeros = df[df[target] == 0]

        keep_0s = zeros.sample(frac=ones.shape[0] * 1.0 / zeros.shape[0])
        keep_0s.describe()
        df = pd.concat([keep_0s, ones], axis=0)
        print(df.describe())

        # Y = df[target]
        # X = df.iloc[:, 1:]
    elif rebalance_method is "upsampling":
      ones = df[df[target] == 1]
      zeros = df[df[target] == 0]

    #   del df
    #   gc.collect()

      ratio = zeros.shape[0] * 1.0 / ones.shape[0]
      print("ratio = ", ratio)
      rep_ones = [ones.iloc[:,1:].apply(lambda x: x + np.random.normal(scale=0.1, size=len(x)))
                  for _ in range(int(round(ratio)))]

      rep_ones = pd.concat(rep_ones)
      rep_ones.insert(0, target, np.ones(shape=rep_ones.shape[0], dtype=np.int32))

      df = pd.concat([rep_ones, zeros])

    #   df = new_data
      print(df.describe())

      # Y = df[target]
      # X = df.iloc[:,1:]

    Y = df[target]
    X = df.iloc[:,1:] # discard first column, assuming that that one is the target

    return X, Y



