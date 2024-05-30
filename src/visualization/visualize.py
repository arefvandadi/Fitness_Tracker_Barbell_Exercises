import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/processed_data_01.pkl")
df.iloc[:10]
# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_def = df[df["set"] == 1]
plt.plot(set_def["acc_y"])
plt.plot(set_def["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
df["label"].unique()
for label in df["label"].unique():
    sub_df = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(sub_df["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
# we will define rcParams and then all the pots after would follow these parameters
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

for label in df["label"].unique():
    sub_df = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(sub_df[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
squat_category = df.query("label == 'squat'").query("participant == 'A'").reset_index()
fig, ax = plt.subplots()
squat_category.groupby(["category"])["acc_y"].plot()
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
plt.legend()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participant_category = (
    df.query("label == 'bench'").sort_values("participant").reset_index()
)
fig, ax = plt.subplots()
participant_category.groupby(["participant"])["acc_y"].plot()
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
All_coordinates_df = (
    df.query("label == 'squat'").query("participant == 'A'").reset_index()
)

fig, ax = plt.subplots()
All_coordinates_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_xlabel("samples")
ax.set_ylabel("acc_x, acc_y, acc_z")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels = df["label"].unique()
participants = df["participant"].unique()

# Acceleration Plots
for label in labels:
    for participant in participants:
        All_coordin_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        # since some of the participants didn't perform all the moves: let's make sure we only plot dataframes that have data
        if len(All_coordin_df) > 0:

            fig, ax = plt.subplots()
            All_coordin_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_xlabel("samples")
            ax.set_ylabel("acc_x, acc_y, acc_z")
            plt.title(f"{label} ({participant})".title())
            plt.legend()


# Gyroscope Plots
for label in labels:
    for participant in participants:
        All_coordin_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        # since some of the participants didn't perform all the moves: let's make sure we only plot dataframes that have data
        if len(All_coordin_df) > 0:

            fig, ax = plt.subplots()
            All_coordin_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_xlabel("samples")
            ax.set_ylabel("gyr_x, gyr_y, gyr_z")
            plt.title(f"{label} ({participant})".title())
            plt.legend()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label = "row"
participant = "A"
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper center", bbox_to_anchor=[0.5, 1.15], ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=[0.5, 1.15], ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("samples")

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

# Combined Acceleration and Gyroscope Plots
for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        # since some of the participants didn't perform all the moves: let's make sure we only plot dataframes that have data
        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=[0.5, 1.15],
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=[0.5, 1.15],
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("samples")
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
