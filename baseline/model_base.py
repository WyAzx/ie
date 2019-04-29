import json
import numpy as np
from random import choice
from tqdm import tqdm

train_data = json.load(open('../data_processed/train_data_me.json'))
dev_data = json.load(open('../data_processed/dev_data_me.json'))
#test_data = json.load(open('../data_processed/test_data_me.json', 'r', encoding='utf8'))
id2predicate, predicate2id = json.load(open('../data_processed/all_50_schemas_me.json'))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open('../data_processed/all_chars_me.json'))

char_size = 128
num_classes = len(id2predicate)


def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []
            # T: text char tokenize
            # S1: subject starts 多个subject标记多个 标记为1
            # S2: subject ends+1
            # K1：subject start id 随机选择一个subject
            # K2：subject end id
            # O1: object starts 多个subject 标记多个 标记为predicate id
            # O2：object ends
            for i in idxs:
                d = self.data[i]
                text = d['text']
                items = {}
                for sp in np.random.permutation(d['spo_list']):
                    subjectid = text.find(sp[0])
                    objectid = text.find(sp[2])
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid + len(sp[0]))
                        # 这里相同subject做key时只会取第一个spo
                        # TODO 获取同个subject不同object
                        if key not in items:
                            items[key] = []
                        items[key].append((objectid,
                                           objectid + len(sp[2]),
                                           predicate2id[sp[1]]))
                if items:
                    T.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding
                    s1, s2 = [0] * len(text), [0] * len(text)
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1] - 1] = 1
                    k1, k2 = choice(list(items.keys()))
                    o1, o2 = [0] * len(text), [0] * len(text)  # 0是unk类（共49+1个类）
                    for j in items[(k1, k2)]:
                        o1[j[0]] = j[2]
                        o2[j[1] - 1] = j[2]
                    S1.append(s1)
                    S2.append(s2)
                    K1.append([k1])
                    K2.append([k2 - 1])
                    O1.append(o1)
                    O2.append(o2)
                    if len(T) == self.batch_size or i == idxs[-1]:
                        T = np.array(seq_padding(T))
                        S1 = np.array(seq_padding(S1))
                        S2 = np.array(seq_padding(S2))
                        O1 = np.array(seq_padding(O1))
                        O2 = np.array(seq_padding(O2))
                        K1, K2 = np.array(K1), np.array(K2)
                        yield [T, S1, S2, K1, K2, O1, O2], None
                        T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []


from keras.layers import Conv1D, Dense, Bidirectional, CuDNNLSTM, Concatenate, Embedding, Dropout, Lambda, Input
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    # batch_idxs 为row index [[0],[1],[2]....]
    # idxs 为 标记col index [[10],[4],[5]...]
    # 这个api哪来的
    return K.tf.gather_nd(seq, idxs)


def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    # mask处减成负值，避免参与maxpool
    # 在text_length 处pool
    # return [batch_size, char_size]
    return K.max(seq, 1)


t_in = Input(shape=(None,))
s1_in = Input(shape=(None,))
s2_in = Input(shape=(None,))
k1_in = Input(shape=(1,))
k2_in = Input(shape=(1,))
o1_in = Input(shape=(None,))
o2_in = Input(shape=(None,))

t, s1, s2, k1, k2, o1, o2 = t_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in

# mask Text padding 0, padding处为0，非padding处为1
mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t)
# char embedding
# TODO 考虑加入字向量
t = Embedding(len(char2id) + 2, char_size)(t)  # 0: padding, 1: unk
t = Dropout(0.25)(t)
t = Lambda(lambda x: x[0] * x[1])([t, mask])
t = Bidirectional(CuDNNLSTM(char_size // 2, return_sequences=True))(t)
t = Bidirectional(CuDNNLSTM(char_size // 2, return_sequences=True))(t)

# （batch_size, text_length, char_size）
t_max = Lambda(seq_maxpool)([t, mask])
# t_dim: char_size
t_dim = K.int_shape(t)[-1]

# concat maxpool 和 seq_hidden_state [batch_size, text_length, 2 * char_size]
h = Lambda(seq_and_vec, output_shape=(None, t_dim * 2))([t, t_max])
# out shape: [batch_size, text_length, char_size]
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)

# 用来预测subject位置
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)

subject_model = Model(t_in, [ps1, ps2])  # 预测subject的模型

# 选取subject起始和终止位置的hidden state [batch_size, char_size]
k1 = Lambda(seq_gather, output_shape=(t_dim,))([t, k1])
k2 = Lambda(seq_gather, output_shape=(t_dim,))([t, k2])
# out shape: [batch_size, 2*char_size]
k = Concatenate()([k1, k2])
# 同上做concat
h = Lambda(seq_and_vec, output_shape=(None, t_dim * 2))([t, t_max])
# 在concat上上面的hidden state
h = Lambda(seq_and_vec, output_shape=(None, t_dim * 4))([h, k])
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)

po1 = Dense(num_classes + 1, activation='softmax')(h)
po2 = Dense(num_classes + 1, activation='softmax')(h)

object_model = Model([t_in, k1_in, k2_in], [po1, po2])  # 输入text和subject，预测object及其关系

train_model = Model([t_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in],
                    [ps1, ps2, po1, po2])

s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * mask) / K.sum(mask)

o1_loss = K.sparse_categorical_crossentropy(o1, po1)
o1_loss = K.sum(o1_loss * mask[:, :, 0]) / K.sum(mask)
o2_loss = K.sparse_categorical_crossentropy(o2, po2)
o2_loss = K.sum(o2_loss * mask[:, :, 0]) / K.sum(mask)

# 2.5 超参
loss = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)

train_model.add_loss(loss)
train_model.compile(optimizer='adam')
train_model.summary()


def extract_items(text_in):
    R = []
    _s = [char2id.get(c, 1) for c in text_in]
    _s = np.array([_s])
    _k1, _k2 = subject_model.predict(_s)
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    for i, _kk1 in enumerate(_k1):
        if _kk1 > 0.5:
            _subject = ''
            for j, _kk2 in enumerate(_k2[i:]):
                if _kk2 > 0.5:
                    _subject = text_in[i: i + j + 1]
                    break
            if _subject:
                _k1, _k2 = np.array([i]), np.array([i + j])
                _o1, _o2 = object_model.predict([_s, _k1, _k2])
                _o1, _o2 = np.argmax(_o1[0], 1), np.argmax(_o2[0], 1)
                for i, _oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j, _oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = text_in[i: i + j + 1]
                                _predicate = id2predicate[_oo1]
                                R.append((_subject, _predicate, _object))
                                break
    return list(set(R))


class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('../save_models/best_model_01.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10
        for d in tqdm(iter(dev_data)):
            R = set(extract_items(d['text']))
            T = set([tuple(i) for i in d['spo_list']])
            A += len(R & T)
            B += len(R)
            C += len(T)
        return 2 * A / (B + C), A / B, A / C


train_D = data_generator(train_data)
evaluator = Evaluate()
train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=len(train_D),
                          epochs=150,
                          callbacks=[evaluator]
                          )

# best f1 0.7525

#train_model.load_weights('../save_models/best_model.weights')
# for i in range(10):
#     d = dev_data[-i-1]
#     print(d, end=' ')
#     print(u'预测结果：', extract_items(d['text']))


############
#  get result
############
# result_f = open('../results/result.json', 'w', encoding='utf8')
# schemas_dict = json.load(open('../data_processed/schemas_dict'))
# from tqdm import tqdm
# for data in tqdm(test_data):
#     spo_list = extract_items(data['text'])
#     for spo in spo_list:
#         s = {
#             'subject': spo[0],
#             'predicate': spo[1],
#             'object': spo[2],
#             'subject_type': schemas_dict[spo[1]][1],
#             'object_type': schemas_dict[spo[1]][0]
#         }
#         data['spo_list'].append(s)
#     result_f.write('{}\n'.format(json.dumps(data,ensure_ascii=False)))
#
# import pickle
# with open('../results/result.pkl', 'wb') as f:
#     pickle.dump(test_data, f)
# result_f.close()