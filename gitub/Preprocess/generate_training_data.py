import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd


from basicts.data.transform import standard_transform
from sklearn.preprocessing import MinMaxScaler

def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    add_day_of_month = args.dom
    add_day_of_year = args.doy
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    steps_per_day = args.steps_per_day
    steps_per_week = args.steps_per_week
    steps_per_month = args.steps_per_month
    steps_per_year = args.steps_per_year
    norm_each_channel = args.norm_each_channel
    rescale_data = args.rescale_data
    num_batches = args.num_batches
    batch_size = args.batch_size
    if_rescale = not norm_each_channel # if evaluate on rescaled data. see `basicts.runner.base_tsf_runner.BaseTimeSeriesForecastingRunner.build_train_dataset` for details.

    # read data
    df = pd.read_csv(data_file_path)
    df_index = df["date"].values
    df = df[df.columns[1:]]
    if rescale_data:
        scaler = MinMaxScaler()
        columns_to_normalize = df.columns
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    df.index = df_index

    data = np.expand_dims(df.values, axis=-1)

    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))

    # split data
    l, n, f = data.shape
    num_samples = batch_size - (history_seq_len + future_seq_len) + 1
    train_batch_num = round(num_batches * train_ratio)
    valid_batch_num = round(num_batches * valid_ratio)
    test_batch_num = num_batches - train_batch_num - valid_batch_num
    train_num = train_batch_num * num_samples
    valid_num = valid_batch_num * num_samples
    test_num = test_batch_num * num_samples

    print("number of sample in a batch:{0}".format(num_samples))
    print("number of training batches:{0}".format(train_batch_num))
    print("number of validation batches:{0}".format(valid_batch_num))
    print("number of test batches:{0}".format(test_batch_num))

    # in batch mask
    index_in_batch_mask = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_in_batch_mask.append(index)
    # every batch indices
    indices_of_batches = []
    index_list = []
    index_batch_start = [i*batch_size for i in range(num_batches)]
    for batch in range(num_batches):
        batch_start = index_batch_start[batch]
        indices_in_batch = [(index[0] + batch_start, index[1] + batch_start, index[2] + batch_start) for index in index_in_batch_mask]
        indices_of_batches.append(indices_in_batch)
        index_list.extend(indices_in_batch)

    # No shuttle
    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]

    # normalize data
    scaler = standard_transform
    # Following related works (e.g. informer and autoformer), we normalize each channel separately.
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)


    # 可以在这里搞一些花活，生成不同尺度的时间特征数据
    # day week month year 四个尺度
    # [5,10,15,20] [2,4,8,16] [10,20,50,100]

    # add temporal feature
    # 时间特征
    # tod：time of day;一天中的哪一点（几点钟）
    # dow：day of week;一周中的哪一天（星期几）
    # dom：day of month;一月中的哪一天（第几号）
    # doy：day of year;一年中的哪一天（第几天）
    # 计算得到的是周期进度 在这里第一周期为5的话 
    # 第3个采样 则是 (3%5)/5=0.6
    # 第12个采样 则是 (12%5)/5=0.4
    feature_list = [data_norm]
    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day / steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = [i % steps_per_week / steps_per_week for i in range(data_norm.shape[0])]
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    if add_day_of_month:
        # numerical day_of_month
        dom = [i % steps_per_month / steps_per_month for i in range(data_norm.shape[0])]
        dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dom_tiled)

    if add_day_of_year:
        # numerical day_of_year
        doy = [i % steps_per_year / steps_per_year for i in range(data_norm.shape[0])]
        doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(doy_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # save data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # dataset name
    DATASET_NAME = "IndPenisim_year"        # sampling frequency: every 1 hour
    RESCALE_DATA = True
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12

    # batch of dataset
    NUM_BATCHES = 10
    BATCH_SIZE = 980

    # train test valid split
    TRAIN_RATIO = 4/10
    VALID_RATIO = 3/10
    TARGET_CHANNEL = [0]        # target channel(s)

    # temporal feature period，计算过程在语雀文档
    STEPS_PER_DAY = 12
    STEPS_PER_WEEK = 20
    STEPS_PER_MONTH = 40
    STEPS_PER_YEAR = 81

    TOD = True             # if add time_of_day feature
    DOW = True                # if add day_of_week feature
    DOM = True                 # if add day_of_month feature
    DOY = False                # if add day_of_year feature

    # It is recommended to set norm_each_channel to False when evaluating rescaled data, especially when the magnitudes of different variables differ significantly.
    # Thus, because larger values of the loss function will be obtained when the magnitudes of different variables differ significantly,
    # the model will be trained to focus on the variable with the largest magnitude. Then, the loss is more likely to be reduced.
    # If downstream tasks have other requirements, you can set norm_each_channel to according to your needs.
    NORM_EACH_CHANNEL = False   # if normalize each channel of data separately.

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.csv".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="History Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Future Sequence Length.")
    parser.add_argument("--num_batches", type=int,
                        default=NUM_BATCHES, help="Number of all batches.")
    parser.add_argument("--batch_size", type=int,
                        default=BATCH_SIZE, help="Size of per batch.")
    parser.add_argument("--steps_per_day", type=int,
                        default=STEPS_PER_DAY, help="Steps per day.")
    parser.add_argument("--steps_per_week", type=int,
                        default=STEPS_PER_WEEK, help="Steps per week.")
    parser.add_argument("--steps_per_month", type=int,
                        default=STEPS_PER_MONTH, help="Steps per month.")
    parser.add_argument("--steps_per_year", type=int,
                        default=STEPS_PER_YEAR, help="Steps per year.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--dom", type=bool, default=DOM,
                        help="Add feature day_of_month.")
    parser.add_argument("--doy", type=bool, default=DOY,
                        help="Add feature day_of_year.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--rescale_data", type=list,
                        default=RESCALE_DATA, help="Rescale raw data")
    parser.add_argument("--norm_each_channel", type=float,
                        default=NORM_EACH_CHANNEL, help="Normalize each channel of data separately.")
    args = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)
