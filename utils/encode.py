import numpy as np
from sklearn.preprocessing import OneHotEncoder


def load_encoder():
    arr = [chr(x) for x in range(65, 65+26)]\
            + [chr(x) for x in range(97, 97+26)]\
            + [str(x) for x in range(0, 10)]
    # arr = np.array(
    #     arr
    # ).reshape(-1, 1)

    # word_encoder = OneHotEncoder().fit(arr)
    # return word_encoder
    word2vec = {'sos':0,'eos':1, ' ': 2 }
    vec2word = ['sos','eos', ' ']
    for item in arr:
        word2vec[item] = len(word2vec)
        vec2word.append(item)
    return word2vec, vec2word

if __name__ == '__main__':
    # word_encoder = load_encoder()
    # label = '32adD'
    # trg = word_encoder.transform([[x for x in label]])
    # print(''.join([word_encoder.categories_[0][xx] for xx in [x for x in np.argmax(trg, 1).flatten().tolist()[0]]]))
    print(load_encoder())