#%%
from models.faster_rcnn.utils import (
    initialize_process,
    run_process,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
)
from utils.voucher import build_dataset

#%%

def main():
    args, run, weights_dir = initialize_process(NEPTUNE_API_KEY, NEPTUNE_PROJECT)
    args.data_dir = "/Users/wonhyung64/data/aihub_512_512"

    # datasets, labels, train_num, valid_num, test_num = load_dataset(
    #     name=args.name, data_dir=args.data_dir, img_size=args.img_size
    # )
    # train_set, valid_set, test_set = build_dataset(datasets, args.batch_size)
    train_set, valid_set, test_set = build_dataset(args)
    train_num , valid_num, test_num = 8000, 100, 1000
    labels = ["bg", "ship"]

    run_process(
        args,
        labels,
        train_num,
        valid_num,
        test_num,
        run,
        train_set,
        valid_set,
        test_set,
        weights_dir,
    )


if __name__ == "__main__":
    main()
