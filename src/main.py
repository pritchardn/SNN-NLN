from data import load_data, process_into_dataset

if __name__ == "__main__":
    train_x, train_y, test_x, test_y, rfi_models = load_data()
    print(rfi_models)
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    train_x, train_y, test_x, test_y, rfi_models = load_data(excluded_rfi='rfi_stations')
    print(rfi_models)
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    train_dataset = process_into_dataset(train_x, train_y, batch_size=32)
    test_dataset = process_into_dataset(test_x, test_y, batch_size=32)
