import gensim.downloader
from easydict import EasyDict
from mlp_lm import run_mlp_lm, load_data_mlp_lm, sample_from_mlp_lm, visualize_epochs
from mlp import run_mlp, load_data_mlp, visualize_configs
from typing import List, Tuple, Dict, Union

EMBEDDING_TYPES = ["glove-twitter-50", "glove-twitter-100", "glove-twitter-200", "word2vec-google-news-300"]
HIDDEN_DIMS = [[], [512], [512, 512], [512, 512, 512]]
HIDDEN_DIMS_NAMES = ["None", "512", "512 -> 512", "512 -> 512 -> 512"]
LEARING_RATES = [0.025, 0.02, 0.01, 0.001]


def single_run_mlp_lm(train_d, dev_d):
    # TODO: once you have completed the backprop.py, you can run this function to train and evaluate your model,
    #  and visualize the training process with a plot
    train_config = EasyDict({
        # model configuration
        'embed_dim': 128,  # the dimension of the word embeddings
        'hidden_dim': 512,  # the dimension of the hidden layer
        'num_blocks': 2,  # the number of transformer blocks
        'dropout_p': 0.2,  # the probability of dropout
        'local_window_size': 6,  # the size of the local window
        # training configuration
        'batch_size': 4096,  # batch size
        'lr': 2e-6,  # learning rate
        'decay': 1.0,
        'num_epochs': 5,  # the total number of times all the training data is iterated over
        'save_path': 'model.pth',  # path where to save the model
    })

    epoch_train_losses, epoch_train_ppls, epoch_dev_losses, epoch_dev_ppls = run_mlp_lm(train_config, train_d, dev_d)
    visualize_epochs(epoch_train_losses, epoch_dev_losses, "Loss", "mlp_lm_loss.png")
    visualize_epochs(epoch_train_ppls, epoch_dev_ppls, "Perplexity", "mlp_lm_ppl.png")


def sample_from_trained_mlp_lm(dev_d):
    pretrained_config = EasyDict({
        # model configuration
        'embed_dim': 256,  # the dimension of the word embeddings
        'hidden_dim': 1048,  # the dimension of the hidden layer
        'num_blocks': 4,  # the number of transformer blocks
        'dropout_p': 0.2,  # the probability of dropout
        'local_window_size': 6,  # the size of the local window
        'save_path': 'pretrained_fixed_window_lm.dat',  # path where to save the model
        # evaluation configuration
        'batch_size': 4096,  # batch size
    })
    sample_from_mlp_lm(pretrained_config, dev_d)


def explore_mlp_structures(dev_d: Dict[str, List[Union[str, int]]],
                           train_d: Dict[str, List[Union[str, int]]],
                           test_d: Dict[str, List[Union[str, int]]]):
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    print(f"{'-' * 10} Load Pre-trained Embeddings: {EMBEDDING_TYPES[0]} {'-' * 10}")
    embeddings = gensim.downloader.load(EMBEDDING_TYPES[0])

    for hidden_dims, hidden_dim_names, lr in zip(HIDDEN_DIMS, HIDDEN_DIMS_NAMES, LEARING_RATES):
        train_config = EasyDict({
            'batch_size': 64,  # we use batching for
            'lr': lr,  # if embedding_type != "None" else 0.01,  # learning rate
            'num_epochs': 20,  # the total number of times all the training data is iterated over
            'hidden_dims': hidden_dims,
            'save_path': f'model_hidden_{hidden_dim_names}.pth',  # path where to save the model
            'embeddings': EMBEDDING_TYPES[0],
            'num_classes': 2,
        })

        epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run_mlp(train_config, embeddings, dev_d, train_d,
                                                                              test_d)
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_loss)
        visualize_epochs(epoch_train_losses, epoch_dev_loss, "Loss", f"mlp_{hidden_dim_names}_loss.png")

    visualize_configs(all_emb_epoch_dev_accs, HIDDEN_DIMS_NAMES, "Accuracy", "./all_mlp_acc.png")
    visualize_configs(all_emb_epoch_dev_losses, HIDDEN_DIMS_NAMES, "Loss", "./all_mlp_loss.png")


if __name__ == '__main__':
    # Load raw data for mlp
    # uncomment the following line to run
    dev_data, train_data, test_data = load_data_mlp()

    # Explore different hidden dimensions
    # uncomment the following line to run
    explore_mlp_structures(dev_data, train_data, test_data)

    # load raw data for lm
    # uncomment the following line to run
    # train_data, dev_data = load_data_mlp_lm()

    # Run a single training run
    # uncomment the following line to run
    # single_run_mlp_lm(train_data, dev_data)

    # Sample from the pretrained model
    # uncomment the following line to run
    # sample_from_trained_mlp_lm(dev_data)

