import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_csv_file_accel = "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
single_file_accel_pd = pd.read_csv(single_csv_file_accel)

single_csv_file_gyroscope = "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
single_file_gyroscope_pd = pd.read_csv(single_csv_file_gyroscope)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
name_of_files = glob("../../data/raw/MetaMotion/*.csv")
len(name_of_files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
participant_list = [name.split("\\")[1][0] for name in name_of_files]
# participant_df = pd.DataFrame(participant_list, columns=["Participant"])

label_list = [name.split("\\")[1].split("-")[1] for name in name_of_files]
# label_df = pd.DataFrame(label_list, columns=["Label"])

category_list = [
    name.split("\\")[1].split("-")[2].split("_")[0].rstrip("123")
    for name in name_of_files
]
# category_df = pd.DataFrame(category_list, columns=["Category"])

# Create a Data Frame for the first csv file with added columns from the ame of the files
single_file_accel_pd_Added_Columns = single_file_accel_pd
single_file_accel_pd_Added_Columns["Participant"] = participant_list[0]
single_file_accel_pd_Added_Columns["Label"] = label_list[0]
single_file_accel_pd_Added_Columns["Category"] = category_list[0]

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
acceleration_df = pd.DataFrame()
gyroscope_df = pd.DataFrame()

# The following two counters are used as identifiers to separate the data based on the input files
acc_set_counter = 0
gyro_set_counter = 0

for i, name in enumerate(name_of_files):

    AccORGyro_df = pd.read_csv(name)
    AccORGyro_df["Participant"] = participant_list[i]
    AccORGyro_df["Label"] = label_list[i]
    AccORGyro_df["Category"] = category_list[i]

    # ignore_index=True parameter below will recreate the index ever time it adds a new row instead of using the index from the original Data frames
    if "Accelerometer" in name:
        acc_set_counter += 1
        AccORGyro_df["set"] = acc_set_counter
        acceleration_df = pd.concat(
            [acceleration_df, AccORGyro_df], axis=0, ignore_index=True
        )
    else:
        gyro_set_counter += 1
        AccORGyro_df["set"] = gyro_set_counter
        gyroscope_df = pd.concat(
            [gyroscope_df, AccORGyro_df], axis=0, ignore_index=True
        )

acceleration_df.shape
gyroscope_df.shape

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
acceleration_df.info()
# from the info() function on the dataframes, you can see that Panda doesn't recognize either epoch or time column as time format.

# We will replace the index with the unix time to create a time series
# We need to do this because time series indexing allows for resampling down the line
acceleration_df.index = pd.to_datetime(acceleration_df["epoch (ms)"], unit="ms")
gyroscope_df.index = pd.to_datetime(gyroscope_df["epoch (ms)"], unit="ms")

# Delete All the time columns now
del acceleration_df["epoch (ms)"]
del acceleration_df["time (01:00)"]
del acceleration_df["elapsed (s)"]

del gyroscope_df["epoch (ms)"]
del gyroscope_df["time (01:00)"]
del gyroscope_df["elapsed (s)"]


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
name_of_files = glob("../../data/raw/MetaMotion/*.csv")


def read_data_from_csv_files(file_names: list):
    participant_list = [name.split("\\")[1][0] for name in file_names]
    # participant_df = pd.DataFrame(participant_list, columns=["Participant"])

    label_list = [name.split("\\")[1].split("-")[1] for name in file_names]
    # label_df = pd.DataFrame(label_list, columns=["Label"])

    category_list = [
        name.split("\\")[1].split("-")[2].split("_")[0].rstrip("123")
        for name in file_names
    ]

    acceleration_df = pd.DataFrame()
    gyroscope_df = pd.DataFrame()

    # The following two counters are used as identifiers to separate the data based on the input files
    acc_set_counter = 0
    gyro_set_counter = 0

    for i, name in enumerate(file_names):

        AccORGyro_df = pd.read_csv(name)
        AccORGyro_df["Participant"] = participant_list[i]
        AccORGyro_df["Label"] = label_list[i]
        AccORGyro_df["Category"] = category_list[i]

        # ignore_index=True parameter below will recreate the index ever time it adds a new row instead of using the index from the original Data frames
        if "Accelerometer" in name:
            acc_set_counter += 1
            AccORGyro_df["set"] = acc_set_counter
            acceleration_df = pd.concat(
                [acceleration_df, AccORGyro_df],
                axis=0,  # ignore_index=True
            )
        if "Gyroscope" in name:
            gyro_set_counter += 1
            AccORGyro_df["set"] = gyro_set_counter
            gyroscope_df = pd.concat(
                [gyroscope_df, AccORGyro_df],
                axis=0,  # ignore_index=True
            )

    # We will replace the index with the unix time to create a time series
    # We need to do this because time series indexing allows for resampling down the line
    acceleration_df.index = pd.to_datetime(acceleration_df["epoch (ms)"], unit="ms")
    gyroscope_df.index = pd.to_datetime(gyroscope_df["epoch (ms)"], unit="ms")

    # Delete All the time columns now
    del acceleration_df["epoch (ms)"]
    del acceleration_df["time (01:00)"]
    del acceleration_df["elapsed (s)"]

    del gyroscope_df["epoch (ms)"]
    del gyroscope_df["time (01:00)"]
    del gyroscope_df["elapsed (s)"]

    return acceleration_df, gyroscope_df


acc_df, gyro_df = read_data_from_csv_files(name_of_files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]
# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

# the csv data was prepared over a week. if resampling is used for the entire data at once, it generates samples from the first secodn of the first day to the last second of the last day
# To avoid that, let's separate the dta by day, resample it and then concat it together after.
# Splitting by Day
merged_data_by_days = [d for i, d in data_merged.groupby(pd.Grouper(freq="D"))]
merged_data_by_days[1]

data_resampled = pd.concat(
    [
        day_df.resample(rule="200ms").apply(sampling).dropna()
        for day_df in merged_data_by_days
    ]
)

# Changing set type to integer
data_resampled["set"] = data_resampled["set"].astype("int")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled.to_pickle("../../data/interim/processed_data_01.pkl")
