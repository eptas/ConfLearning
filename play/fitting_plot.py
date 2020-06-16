import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_models = 7
n_subjects = 52

AIC, BIC, alpha, beta, alpha_c, gamma = None, None, None, None, None, None

for n in range(n_models):

    fittingData = pd.read_pickle('C:/Users/esthe/PycharmProjects/MetaCognition/play/fittingDataM' + str(n) + '.pkl', compression=None)

    locals()["AIC_M" + str(n)] = np.mean(fittingData.AIC)
    AIC = np.append(AIC, eval("AIC_M" + str(n)))

    locals()["BIC_M" + str(n)] = np.mean(fittingData.BIC)
    BIC = np.append(BIC, eval("BIC_M" + str(n)))

    locals()["alpha_M" + str(n)] = np.mean(fittingData.ALPHA)
    alpha = np.append(alpha, eval("alpha_M" + str(n)))

    locals()["beta_M" + str(n)] = np.mean(fittingData.BETA)
    beta = np.append(beta, eval("beta_M" + str(n)))

    locals()["alpha_c_M" + str(n)] = np.mean(fittingData.ALPHA_C)
    alpha_c = np.append(alpha_c, eval("alpha_c_M" + str(n)))

    locals()["gamma_M" + str(n)] = np.mean(fittingData.GAMMA)
    gamma = np.append(gamma, eval("gamma_M" + str(n)))

subjects = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
colors = ['r', 'b', 'g', 'y', 'm', 'c', 'k']

plt.figure(0)

plt.bar(range(n_models), BIC[1:len(BIC)])   # CHANGE HERE
plt.title('model fit')
plt.xlabel('model')
plt.xticks(np.arange(n_models), ['RLModel', 'RLModel\nWithoutFeedback', 'ConfidencePE', 'ConfidencePE\ngeneric', 'Confidence\nIdealObserver', 'BayesModel', 'BayesIdealObserver'], fontsize=6)
plt.ylabel('BIC')
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('Models_BIC.png', bbox_inches='tight')  # CHANGE HERE

plt.figure(1)

plt.bar(range(n_models), alpha_c[1:len(alpha_c)])
plt.xticks(np.arange(n_models), ['RLModel', 'RLModel\nWithoutFeedback', 'ConfidencePE', 'ConfidencePE\ngeneric', 'Confidence\nIdealObserver', 'BayesModel', 'BayesIdealObserver'], fontsize=6)
plt.title('alpha_c / alpha_n / phi')
plt.xlabel('model')
plt.ylabel('mean alpha_c / alpha_n / phi')
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('Alpha_c.png', bbox_inches='tight')
