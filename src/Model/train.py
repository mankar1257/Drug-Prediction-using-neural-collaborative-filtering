
import argparse
from model import Get_model, Get_train_instances
from .Utilities.Evaluation import Evaluator
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
from keras.models import load_model


def args_parser():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--learner', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float32, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--verbose', type=int, default=64)

    # data directories
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str,
                        default=os.environ.get('SM_CHANNEL_TEST'))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()


def get_data():

    train = np.load(os.path.join(train_dir, 'train.npy'))
    test = np.load(os.path.join(train_dir, 'test.npy'))

    return train, test


if __name__ == '__main__':

    args, _ = args_parser()

    layers = [256, 128, 64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0, 0, 0]
    num_negatives = 6
    learner = args.learner
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 5
    model_out_file = 'Pretrain_new.h5'

    # Loading data
    t1 = time()
    train, test = get_data()

    num_medical_conditions, num_drugs = train.shape

    # Build model
    model = Get_model(num_medical_conditions, num_drugs, layers, reg_layers)

    # compile model
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])

    evaluator = Evaluator()

    # Check Init performance
    t1 = time()
    (hr, ndcg) = evaluator.evaluate(model, topK)
    HR, NDCG = np.array(hr).mean(), np.array(ndcg).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (HR, NDCG, time()-t1))

    # Train model
    best_hr, best_ndcg, best_iter = HR, NDCG, -1

    # Generate training instances
    medical_condition_input, drug_input, labels = Get_train_instances(
        train, num_negatives)

    for epoch in range(epochs):
        t1 = time()

        # Training
        hist = model.fit([np.array(medical_condition_input), np.array(drug_input)],
                         np.array(labels), batch_size=batch_size, nb_epoch=20, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:

            (hr, ndcg) = evaluator.evaluate(model, topK)
            HR, NDCG, loss = np.array(hr).mean(), np.array(
                ndcg).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch,  t2-t1, HR, NDCG, loss, time()-t2))

            if HR >= best_hr and NDCG >= best_ndcg:

                best_hr, best_ndcg, best_iter = HR, NDCG, epoch
                model.save(model_out_file)

    print("End. Best Iteration %d: HR = %.4f, NDCG = %.4f. " %
          (best_iter, best_hr, best_ndcg))

    model = load_model('my_model.h5')
    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)
