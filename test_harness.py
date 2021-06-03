import numpy as np
import json
from run_mdk_game import *

params = {'foo': 'run_mdk_game.py', 'players': 2, 'colors': 5, 'rank': 5, 'hand_size': 5, 'max_information_tokens': 8,
          'max_life_tokens': 3, 'seed': -1, 'verbose': 1, 'num_rounds': 100, 'training_rounds': 0, 'alpha': 0.2,
          'alpha2': 0, 'alpha3': 0, 'decision_func': 'kss', 'agents': ['advhm', 'advh']}

res = {}

for alpha in [0,.01,.05,.08,.1,.13,.15,.2,.25,.3]:
    params['alpha'] = alpha
    for agent in ['advh', 'dfa', 'rla']:
        params['agents'][1] = agent
        if agent in ['dfa', 'rla']:
            for df in ['kss', 'em']:
                params['decision_func'] = df
                if agent == 'rla':
                    main(params)
                print()
                print('%s_%s_0-%02d' % (agent, df, int(100*alpha)))
                score_list = np.array(main(params))
                avg, stdev = np.average(score_list), np.std(score_list)
                res['%s_%s_0-%02d' % (agent, df, int(100*alpha))] = (avg, stdev)
        else:
            print()
            print('%s_0-%02d' % (agent, int(100 * alpha)))
            score_list = np.array(main(params))
            avg, stdev = np.average(score_list), np.std(score_list)
            res['%s_0-%02d' % (agent, int(100 * alpha))] = (avg, stdev)
    with open('test_res_0-%02d.json' % (int(100 * alpha),), 'w') as f:
        json.dump(res, f)

with open('test_res.json', 'w') as f:
    json.dump(res, f)
