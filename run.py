#%%
import cv2
import numpy as np
import tensorflow as tf
from module.load import load_dataset


def fit_frcnn():
    from models.faster_rcnn.utils.process_utils import initialize_process, run_process
    from models.faster_rcnn.utils.variable import NEPTUNE_API_KEY, NEPTUNE_PROJECT
    from models.faster_rcnn.utils.data_utils import load_data_num
    from frcnn_utils import build_dataset

    args, run, weights_dir = initialize_process(
        NEPTUNE_API_KEY, NEPTUNE_PROJECT
    )

    datasets, labels = load_dataset(data_dir=args.data_dir)
    train_num, valid_num, test_num = load_data_num(
        args.name, args.data_dir, datasets[0], datasets[1], datasets[2]
        )
    train_set, valid_set, test_set = build_dataset(datasets, args.batch_size, 50.)
    
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


def fit_retina():
    from retina_utils import build_dataset
    from models.retinanet.module.model import build_model, DecodePredictions
    from models.retinanet.module.neptune import record_result
    from models.retinanet.module.optimize import build_optimizer
    from models.retinanet.module.loss import RetinaNetBoxLoss, RetinaNetClassificationLoss
    from models.retinanet.module.utils import initialize_process, train, evaluate
    from models.retinanet.module.variable import NEPTUNE_API_KEY, NEPTUNE_PROJECT
    from models.retinanet.module.dataset import load_data_num

    args, run, weights_dir = initialize_process(NEPTUNE_API_KEY, NEPTUNE_PROJECT)
    datasets, labels = load_dataset(data_dir=args.data_dir)
    train_num, valid_num, test_num = load_data_num(
        args.name, args.data_dir, datasets[0], datasets[1], datasets[2]
        )
    train_set, valid_set, test_set = build_dataset(datasets, args.batch_size, -1.)
    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)

    model = build_model(len(labels))
    decoder = DecodePredictions(confidence_threshold=0.5)
    box_loss_fn = RetinaNetBoxLoss(args.delta)
    clf_loss_fn = RetinaNetClassificationLoss(args.alpha, args.gamma)

    optimizer = build_optimizer(args.batch_size, train_num, args.momentum)

    train_time = train(run, args.epochs, args.batch_size,
        train_num, valid_num, train_set, valid_set, labels,
        model, decoder, box_loss_fn, clf_loss_fn, optimizer, weights_dir)

    model.load_weights(f"{weights_dir}.h5")
    mean_ap, mean_evaltime = evaluate(run, test_set, test_num, model, decoder, labels, "test", colors)
    record_result(run, weights_dir, train_time, mean_ap, mean_evaltime)

#%%
if __name__ == "__main__":
    fit_retina()
    # fit_frcnn()
