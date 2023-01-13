import pickle

def load(path):
    fp = open(path, 'rb')
    pickle.load(fp)
    fp.close()


def save(file, path):
    fp = open(path, 'wb')
    pickle.dump(file, fp)
    fp.close()