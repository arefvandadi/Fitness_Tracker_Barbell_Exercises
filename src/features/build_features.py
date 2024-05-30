import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim//02_outliers_removed_chauvenets.pkl")

# check what columns have have missing values: either of the following two methods work:
# Method-1
# df.info()
# Method-2
df.isna().any()  # Therefore first 6 columns have missing values

# Create a list of column titles:
predictor_columns = list(df.columns[:6])

# plot Settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
    df[col] = df[col].interpolate()

# check for missing values again
df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
# ***** Let's Visualize Sets First ******
# Let's look at Participant A, Squat, Heavy Reps for each set
Participant = "A"
label = "squat"
category = "heavy"
sensor_data = "acc_z"
Part_Label_Cat_sets = list(
    df[
        (df["participant"] == Participant)
        & (df["label"] == label)
        & (df["category"] == category)
    ]["set"].unique()
)

for s in Part_Label_Cat_sets:
    fig, ax = plt.subplots()
    df[df["set"] == s][sensor_data].plot()


# ***** Let's Measure Duration for Each Set *******
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    end = df[df["set"] == s].index[-1]
    duration = end - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

# Comparing heavy (5 rep) vs medium (10 rep) set duration
mean_cat_dura_df = df.groupby(["category"])["duration"].mean()
heavy_duration = mean_cat_dura_df[0] / 5  # divided by 5 reps
medium_duration = mean_cat_dura_df[1] / 10  # divided by 10 reps


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

# calcualting sampling freq and define cut-off freq
sampling_time = (df.index[1] - df.index[0]).total_seconds()
sampling_freq = 1 / sampling_time
# cutoff will be defined by visually looking at the plot (the higher the cutoff the closer the result to the raw data)
cutoff_freq = 1.2

# the following cretes a low pass column in the data frame
df_lowpass = LowPass.low_pass_filter(
    df_lowpass, "acc_y", sampling_freq, cutoff_freq, order=5
)
subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# let's create low pass for all acc and gyr columns and overwrite the columns with the new data
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(
        df_lowpass, col, sampling_freq, cutoff_freq, order=5
    )
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
# ******* Is it reasonable to run PCa for the whole Data?
# ******* shouldn't we run it separately for each label/exercise?
df_pca = df_lowpass.copy()

PCA = PrincipalComponentAnalysis()
pca_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pca_values)
plt.xlabel("principle component number")
plt.ylabel("explained variance")
plt.show()

# we will keep both PCA and original columns to see which one perform better for clustering
df_pca = PCA.apply_pca(
    df_pca, predictor_columns, 3
)  # three comes from the plot from the previous section, it basically looks at the variance and see where it elbows

# plot PCA columns
df_pca[df_pca["set"] == 30][["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

# Plotting acc_r and gyr_r
df_squared[df_squared["set"] == 30][["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()

NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

window_size = 5

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, "std")

df_temporal

# What we did above should be applied to each set not the whole data since there are times that you average over two exercises' data when you move from one label to another
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)
# Now for each set there are 4 missing values for the new columns
# let's plot one acc and gyr column with its corresponding mean and std columns
df_temporal[df_temporal["set"] == 5][
    ["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]
].plot()
df_temporal[df_temporal["set"] == 5][
    ["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]
].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()

Fourier_Trans = FourierTransformation()

sampling_time = (df.index[1] - df.index[0]).total_seconds()
sampling_freq_FFT = int(1 / sampling_time)
# window size for fourier is the average duration of each rep (2800ms) divided by sampling time
window_size_FFT = int(2800 / (sampling_time * 1000))

# df_freq = Fourier_Trans.abstract_frequency(df_freq, ["acc_y"], window_size_FFT, sampling_freq_FFT)


# Again we need to apply DFT/FFT to each set separately rather than all together
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"applying Fourier Transformation to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = Fourier_Trans.abstract_frequency(
        subset, predictor_columns, window_size_FFT, sampling_freq_FFT
    )
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


df_freq.columns

# let's drop the NAN values
df_freq = df_freq.dropna()

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# Because of window size, the neighboring data generated through rolling average or rolling std or fourier are overlapped and correlated which can result in overfitting
# in order to removve overfitting, you can skip every other data and keep only 50% of the data
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
intertia = []

# we are going to determine the optimum K value for KMEANS
for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_label = kmeans.fit_predict(subset)
    intertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, intertia)
plt.xlabel("k")
plt.ylabel("Sum of Squared Distance")
plt.show()

# we will choose K=5 based on the plot above
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# plotting the clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# plotting data based on labels/exercise
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Looking at the label plots, Bench and OHP are very similar, Deadlift and Row are also similar, Squat and Rest are very different
# Clustering seem to have captured some but has a lot of issues


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
