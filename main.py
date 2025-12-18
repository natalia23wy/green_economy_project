from src.data_loader import build_dataset_france

def main():
    df = build_dataset_france()
    print(df.head())
    print(df.tail())

if __name__ == "__main__":
    main()
