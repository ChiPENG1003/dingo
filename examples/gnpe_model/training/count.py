import pandas as pd
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

MODE_LIST = ["main", "init"]
def main(mode):
    if mode in MODE_LIST and mode == "main":
        # FILE_NAME_1 = "main_train_dir_1M_2000train_500val_200SVD_6params"
        # FILE_NAME_2 = "main_train_dir_1M_2000train_500val_200SVD"
        # FILE_NAME_3 = "main_train_dir_2000train_500val_200SVD_6params"
        # FILE_NAME_4 = "main_train_dir_1M_20000train_5000val_200SVD_7params"
        #FILE_NAME_5 = "main_train_dir_2000train_500val_200SVD"
        FILE_NAME_6 = "main_train_dir_ET_5M_sky_localization"
        #FILE_NAME_7 = "main_train_dir_ET"

        FILE_LIST = [
            value
            for name, value in locals().items()
            if name.startswith("FILE_NAME_")
        ]

        plt.figure()
        for file in FILE_LIST:
            # if file == FILE_NAME_2:
            #     file_name = "main_training_1M"
            # elif file == FILE_NAME_3:
            #     file_name = "main_training_500K_6params"
            # elif file == FILE_NAME_4:
            #     file_name = "main_training_1M_7params"
            # elif file == FILE_NAME_5:
            #     file_name = "main_training_500K"
            if file == FILE_NAME_6:
                file_name = "sky_localization"
            # if file == FILE_NAME_7:
            #     file_name = "main_ET"
            DATA_FILE = os.path.join(HERE, file, "history.txt")

            df = pd.read_csv(
                DATA_FILE,
                sep='\t',
                header=None,
                names=['epoch','training_loss','validation_loss','learning_rate']
            )
            # df = df.iloc[:270]

            train_loss = df['training_loss']
            val_loss = df['validation_loss']
            epoch = df['epoch']

            plt.plot(epoch, train_loss,label=f"Train Loss{file_name}")
            plt.plot(epoch, val_loss,label=f"Val Loss{file_name}")

        plt.xlabel("Epoch")
        plt.legend()
        plt.ylabel("Validation Loss")
        plt.title("training process")
        plt.savefig(fname='train_val_loss.png')
        print("Saved the plot to the current file.")
        plt.show()

    elif mode in MODE_LIST and mode == "init":
        FILE_NAME_1 = "init_train_dir_1M_20000train_5000val_200SVD"
        FILE_NAME_2 = "init_train_dir_1M_20000train_5000val_200SVD_6params"
        FILE_NAME_3 = "init_train_dir_7params_O2"
        FILE_NAME_4 = "init_train_dir_20000train_5000val_200SVD"
        FILE_NAME_5 = "init_train_dir_20000train_5000val_300SVD_6params"

        FILE_LIST = [
            value
            for name, value in locals().items()
            if name.startswith("FILE_NAME_")
        ]

        plt.figure()
        for file in FILE_LIST:
            if file == FILE_NAME_1:
                file_name = "init_training_1M"
            elif file == FILE_NAME_2:
                file_name = "init_training_1M_6params"
            elif file == FILE_NAME_3:
                file_name = "init_training_7params_O2"
            elif file == FILE_NAME_4:
                file_name = "init_training_500K"
            elif file == FILE_NAME_5:
                file_name = "init_training_500K_6params"
            DATA_FILE = os.path.join(HERE, file, "history.txt")

            df = pd.read_csv(
                DATA_FILE,
                sep='\t',
                header=None,
                names=['epoch','training_loss','validation_loss','learning_rate']
            )
            # df = df.iloc[:270]

            # train_loss = df['training_loss']
            val_loss = df['validation_loss'][200:300]
            epoch = df['epoch'][200:300]

            # plt.plot(epoch, train_loss,label="Train Loss")
            plt.plot(epoch, val_loss,label=f"Val Loss{file_name}")

        plt.xlabel("Epoch")
        plt.legend()
        plt.ylabel("Validation Loss")
        plt.title("training process")
        plt.savefig(fname='init_train_loss.png')
        print("Saved the plot to the current file.")
        plt.show()

if __name__ == "__main__":
    main("main")
    # main("init")
