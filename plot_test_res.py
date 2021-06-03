import matplotlib.pyplot as plt
import json

with open('test_res.json') as f:
    scores_load = json.load(f)

alphas = [0,.01,.05,.08,.1,.13,.15,.2,.25,.3]
scores = {}

for agent in ['advh', 'dfa', 'rla']:
    if agent in ['dfa', 'rla']:
        for df in ['kss', 'em']:
            agent_str = '%s_%s' % (agent, df)
            scores[agent_str] = []
            for alpha in alphas:
                loc = '%s_%s_0-%02d' % (agent, df, int(100 * alpha))
                scores[agent_str].append(scores_load[loc][0])

    else:
        scores[agent] = []
        for alpha in alphas:
            loc = '%s_0-%02d' % (agent, int(100 * alpha))
            scores[agent].append(scores_load[loc][0])

names = {'advh': 'Adv Human', 'dfa_kss': 'K-Means', 'dfa_em': 'EM', 'rla_kss': 'RL K-Means', 'rla_em': 'RL EM'}

for i in scores:
    plt.plot(alphas, scores[i], label=names[i])

plt.xlabel('beta')
plt.ylabel('Average Score')

plt.legend()
plt.show()

