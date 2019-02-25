from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from model.fw_lstm import FW_lstm
from model.bw_lstm import BW_lstm
from model.win_lstm import Win_LSTM
from model.ripple_model import RippleModel


def main():
    # create instance of config，这里的config实现了load data的作用
    #拥有词表、glove训练好的embeddings矩阵、str->id的function
    config = Config()


    # build model
    model = RippleModel(config)
    model.build("train")


    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets [(char_ids), word_id]
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)




if __name__ == "__main__":
    main()
