def ids_eval(model, x_test, y_test):
    print("Evaluation: starting validation of the model")
    print()
    TP, FP, TN, FN = 0,0,0,0
    env = IdsEnv(images_per_episode=1, dataset=(x_test, y_test), random=False)
    obs, done = env.reset(), False
    try:
        while True:
            obs, done = env.reset(), False
            while not done:
                obs, rew, done, info = env.step(model.predict(obs)[0])
                label = info['label']
                if label == 0 and rew > 0:
                    TP += 1
                if label == 0 and rew == 0:
                    FP += 1
                if label == 1 and rew > 0:
                    TN += 1
                if label == 1 and rew == 0:
                    FN += 1

    except StopIteration:
        accuracy = (float(TP + TN) / (TP + FP + FN + TN)) 
        precision = (float(TP) / (TP + FP))
        recall = (float(TP) / (TP + FN)) # = TPR = Sensitivity
        try:
            FPR = (float(FP) / (TN + FP)) # 1 - specificity
        except:
            FPR = 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        print()
        print('Evaluation: validation done...')
        print('Accuracy: {0}%'.format(accuracy * 100))
        print('Precision: {0}%'.format(precision * 100))
        print('Recall/TPR/Sensitivity: {0}%'.format(recall * 100))
        print('FPR: {0}%'.format(FPR * 100))
        print('F1 score: {0}'.format(f1_score))
    return [accuracy, precision, recall, FPR, f1_score]