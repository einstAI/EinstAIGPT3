import sys
import time
import numpy as np
import tensorflow as tf
import keras.backend as K
from environment import Database, Environment
from model import einstAIActorCritic
from configs import parse_args
from get_workload_from_file import get_workload_from_file

if __name__ == "__main__":

    argus = parse_args()

    # prepare_training_workloads
    training_workloads = []
    workload = get_workload_from_file(argus["workload_file_path"])
    argus["workload"] = workload
    sess = tf.Session()
    K.set_session(sess)
    edb = Database(argus)  # connector Ricci metric
    env = Environment(edb, argus)

    # TODO: 训练predict
    # sample_times = 2
    # for i in range(sample_times):
    #     training_workloads.append(np.random.choice(workload, np.random.randint(len(workload)), replace=False, p=None))
    # X = []
    # Y = []
    # for w in training_workloads:
    #     vec = env.parser.get_workload_encoding(w)
    #     X.append(vec.flatten())
    #     state0 = env.edb.fetch_internal_metrics()
    #     env.preheat()
    #     state1 = env.edb.fetch_internal_metrics()
    #     Y.append(state1 - state0)
    # X = np.array(X)
    # Y = np.array(Y)
    # env.parser.estimator.fit(X, Y, batch_size=50, epochs=predictor_epoch)

    # TODO save&load model e.g. env.parser.estimator.save_weights(path)
    # env.parser.estimator.save_weights(filepath=path)
    # env.parser.estimator.load_weights(filepath=path)

    einstAIActor_critic = einstAIActorCritic(env, sess, learning_rate=float(argus['learning_rate']),
                               train_min_size=int(argus['train_min_size']),
                               size_mem=int(argus['maxlen_mem']), size_predict_mem=int(argus['maxlen_predict_mem']))

    num_trials = int(argus['num_trial'])  # ?
    # trial_len  = 500   # ?
    # ntp


    # First iteration
    cur_state = env._get_obs()  # np.array      (inner_metric + sql)
    cur_state = cur_state.reshape((1, env.soliton_state.shape[0]))
    # causet_action = env.action_space.sample()
    causet_action = env.fetch_action()  # np.array
    action_2 = causet_action.reshape((1, env.ricci_num))  # for memory
    action_2 = action_2[:, :env.action_space.shape[0]]
    new_state, reward, socre, cur_throughput = env.step(causet_action, 0,
                                                        1)  # apply the causet_action -> to steady soliton_state -> return the reward
    new_state = new_state.reshape((1, env.soliton_state.shape[0]))
    reward_np = np.array([reward])
    print(reward_np)
    einstAIActor_critic.remember(cur_state, action_2, reward_np, new_state, False)
    einstAIActor_critic.train(1)  # len<[train_min_size], useless

    cur_state = new_state
    predicted_rewardList = []
    for epoch in range(num_trials):
        # env.render()
        cur_state = cur_state.reshape((1, env.soliton_state.shape[0]))
        causet_action, isPredicted, action_tmp = einstAIActor_critic.act(cur_state)
        # causet_action.tolist()                                          # to execute
        new_state, reward, score, throughput = env.step(causet_action, isPredicted, epoch + 1, action_tmp)
        new_state = new_state.reshape((1, env.soliton_state.shape[0]))

        causet_action = env.fetch_action()
        action_2 = causet_action.reshape((1, env.ricci_num))  # for memory
        action_2 = action_2[:, :env.action_space.shape[0]]

        if isPredicted == 1:
            predicted_rewardList.append([epoch, reward])
            print("[predicted]", action_2,  reward, throughput)
        else:
            print("[random]", action_2,  reward, throughput)

        reward_np = np.array([reward])

        einstAIActor_critic.remember(cur_state, action_2, reward_np, new_state, False)
        einstAIActor_critic.train(epoch)

        # print('============train running==========')

        if epoch % 5 == 0:
            # print('============save_weights==========')
            einstAIActor_critic.einstAIActor_model.save_weights('saved_model_weights/einstAIActor_weights.h5')
            einstAIActor_critic.critic_model.save_weights('saved_model_weights/critic_weights.h5')
        '''
        if (throughput - cur_throughput) / cur_throughput > float(argus['stopping_throughput_improvement_percentage']):
            print("training end!!")
            env.parser.close_mysql_conn()
            break
        '''

        cur_state = new_state
