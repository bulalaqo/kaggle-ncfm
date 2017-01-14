from keras.models import load_model

from data import load_data


def testing(test_file, ckpt_file, result_file):
    model = load_model(ckpt_file)
    x_test, names = load_data(test_file)
    probs = model.predict(x_test, batch_size=100)

    with open(result_file, 'w') as f:
        print('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT', file=f)
        for name, prob in zip(names, probs):
            prob_str = ','.join('{:.8f}'.format(p) for p in prob)
            print('{},{}'.format(name, prob_str), file=f)