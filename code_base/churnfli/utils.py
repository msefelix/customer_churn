import pickle
import joblib


def load_pickle(path):
    with open(path, 'rb') as handle:
        res = pickle.load(handle)
    return res


def save_pickle(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


def save_model(model, name):
    print("Saving model: ", name)
    joblib.dump(model, name)


def load_model(name):
    print("Loading model: ", name)
    return joblib.load(name)