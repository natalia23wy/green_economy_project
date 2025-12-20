from src.data_loader import build_dataset_france, train_test_split

def main():
    # 1) Build merged dataset for France
    df = build_dataset_france()
    # print(df.head())
    # print(df.tail())


    # 2) Train/test split (time based)
    X_train, X_test, y_train, y_test, train_df, test_df = (
        train_test_split(df)
    )
    # print("Train years:", train_df["year"].min(), "-", train_df["year"].max())
    # print("Test years:", test_df["year"].min(), "-", test_df["year"].max())
    # print("Train size:", X_train.shape)
    # print("Test size:", X_test.shape)

if __name__ == "__main__":
    main()