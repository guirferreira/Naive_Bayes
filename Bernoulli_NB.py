import pandas as pd
from math import log

def argmax(seq):
    return max(range(len(seq)), key=seq.__getitem__)

class BernoulliClassifier(object):
    
    def train_BernoulliNB(self, label, data):
        '''Training'''
        train = [[0]*785 for _ in range(10)]
        priors = [0]*10
        
        data = (data >= 120).astype(int)
        for i in range(len(label)):
            priors[label[i]] = priors[label[i]] + 1
            for j in range(784):                
                train[label[i]][j] = train[label[i]][j] + data[i][j]
        
        for i in range(10):
            for j in range(784):
                train[i][j] = (train[i][j] + 1)/(priors[i]+2)
            priors[i] = priors[i]/len(label)
           
        return train, priors

    def predict_BernoulliNB(self, predict, train, priors):
        '''Classification between 0-9'''
        predict = (predict >= 120).astype(int)
        line = [[0]*2 for i in range(28000)]
        score = [0]*10
        print(priors)
        for i in range(len(predict)):
            for l in range(len(priors)):
                score[l] = log(priors[l])
                for f in range(784):
                    score[l] += log(train[l][f] if predict[i][f]==1 else 1. - train[l][f])
            line[i] = [i+1, argmax(score)] 
            
        df = pd.DataFrame(data = line, columns = ['ImageId', 'Label'])
        df.to_csv("saida.csv", sep = ',', index=False)
    
def main():
    train_csv = pd.read_csv("train.csv").as_matrix()
    test_csv = pd.read_csv("test.csv").as_matrix()
    
    data_train = train_csv[0:,1:]
    label_train = train_csv[0:,0]
    
    train = [[0]*785 for _ in range(10)]
    priors = [0]*10
    
    nb = BernoulliClassifier()
    train, priors = nb.train_BernoulliNB(label_train, data_train)
    
    data_test = test_csv[0:,0:]
    
    nb.predict_BernoulliNB(data_test, train, priors)

def __init__():
    main()