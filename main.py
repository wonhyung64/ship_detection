#%%
from models.faster_rcnn.utils import (
    initialize_process,
    run_process,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
)
from utils import (
    load_dataset,
    build_dataset,
)


def main():
    args, run, weights_dir = initialize_process(NEPTUNE_API_KEY, NEPTUNE_PROJECT)

    datasets, labels, train_num, valid_num, test_num = load_dataset(name=args.name, data_dir=args.data_dir, img_size=args.img_size)
    train_set, valid_set, test_set = build_dataset(datasets, args.batch_size)

    run_process(args, labels, train_num, valid_num, test_num, run, train_set, valid_set, test_set, weights_dir)


if __name__ == "__main__":
    main()
