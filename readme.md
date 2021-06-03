<h2>Matt Riedman's CS 229 Final Project</h2>
<h3>Files:</h3>
<ul>
    <li>
        gen_model_data.py: Running 
        python3 gen_model_data.py -a advhm advhm -n [games] -v 1 -al 0.5
        generates training/validation data for [games] simulated games and
        saves the data to train_data.txt and val_data.txt
    </li>
    <li>
        run_mdk_game.py: Running
        python3 gen_model_data.py -a [agent1] [agent2] -n [games] -v 1 -al [alpha] -df [func]
        runs [games] games with agents [agent1] and [agent2] using mistake rate [alpha] and prediction function [func].
        Valid agents are advh (Adv Human), advhm (Mistake Agent), dfa (Agent using Decision Function), and 
        rla (Reinforcement Learning). Valid functions are kss (K-Means Semi-Sup) and em (EM).
    </li>
    <li>
        test_harness.py: Running this file automatically conducts the tests in my paper and saves the average scores and 
        standard deviations to test_res.json
    </li>
    <li>
        train_k_means.py, train_em.py: Trains K-Means and EM on train_data.txt
    </li>
    <li>
        eval_k_means.py, eval_em.py: Evaluate K-Means and EM on val_data.py
    </li>
    <li>
        adv_human.py, adv_human_mistake.py, decision_func_agent.py, rl_agent.py: Contain code for their namesake agents
    </li>
    <li>
        train_rl.py: Used in rl_agent.py to train the RL agents
    </li>
    <li>
        plot_test_res.py: Plotting code for the results of test_harness.py
    </li>
</ul>
