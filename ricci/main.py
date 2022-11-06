import sys
import time
import numpy as np
from FACE.algorithms.Inference import log_likelihood
from FACE.structure.Base import Sum, Product, assign_ids, rebuild_scopes_bottom_up
from FACE.structure.leaves.parametric.Parametric import Categorical, Gaussian
from FACE.structure.leaves.parametric.Parametric import Bernoulli
from FACE.structure.leaves.parametric.Parametric import Poisson
from FACE.structure.leaves.parametric.Parametric import Gamma
from FACE.structure.leaves.parametric.Parametric import LogNormal
from FACE.structure.leaves.parametric.Parametric import Geometric
from FACE.structure.leaves.parametric.Parametric import Exponential
from FACE.structure.leaves.parametric.Parametric import Beta


def einstAIActorCritic(env, sess, learning_rate, train_min_size, size_mem, size_predict_mem):
    # Path: EinsteinDB-GPT3/ricci/main.py
    # Compare this snippet from AML/Synthetic/deepdb/deepdb_job_ranges/aqp_spn/custom_spflow/custom_learning.py:
    # import logging
    #
    # import numpy as np
    # from aqp_spn.aqp_leaves import Categorical
    # from aqp_spn.aqp_leaves import IdentityNumericLeaf
    # from sklearn.cluster import KMeans
    # from FACE.algorithms.splitting.Base import preproc, split_data_by_clusters
    # from FACE.algorithms.splitting.RDC import getIndependentRDCGroups_py
    # from FACE.structure.StatisticalTypes import MetaType
    #
    # logger = logging.getLogger(__name__)
    # MAX_UNIQUE_LEAF_VALUES = 10000
    #
    #
    # def learn_mspn(
    #         data,
    #         ds_context,
    #         cols="rdc",
    #         rows="kmeans",
    #         min_instances_slice=200,
    #         threshold=0.3,
    #         max_sampling_threshold_cols=10000,
    #         max_sampling_threshold_rows=100000,
    #         bloom_filters=False,
    #         ohe=False,
    #         leaves=None,
    #         memory=None,
    #         rand_gen=None,
    #         cpus=-1,
    # ):
    #     """
    #     Adapts normal learn_mspn to use custom identity leafs and use sampling for structure learning.
    #     :param bloom_filters:
    #     :param max_sampling_threshold_rows:
    #     :param max_sampling_threshold_cols:
    #     :param data:
    #     :param ds_context:
    #     :param cols:
    #     :param rows:
    #     :param min_instances_slice:
    #     :param threshold:
    #     :param ohe:
    #     :param leaves:
    #     :param memory:
    #     :param rand_gen:
    #     :param cpus:
    #     :return:
    #     """
    #     if leaves is None:
    #         leaves = create_custom_leaf
    #
    #     if rand_gen is None:
    #         rand_gen = np.random.RandomState(17)
    #
    #     from a import a
    #     from b import b




def expectation(FACE, feature_scope, inverted_features, ranges, node_expectation=None, node_likelihoods=None,

                use_generated_code=False, spn_id=None, meta_types=None, gen_code_stats=None):

        # evidence_scope = set([i for i, r in enumerate(ranges) if not np.isnan(r)])
    evidence_scope = set([i for i, r in enumerate(ranges[0]) if r is not None])
    evidence = ranges

    assert not (len(evidence_scope) > 0 and evidence is None)

    relevant_scope = set()
    relevant_scope.update(evidence_scope)
    relevant_scope.update(feature_scope)
    if len(relevant_scope) == 0:
        return np.ones((ranges.shape[0], 1))

    if ranges.shape[0] == 1:

        applicable = True
        if use_generated_code:
            boolean_relevant_scope = [i in relevant_scope for i in range(len(meta_types))]
            boolean_feature_scope = [i in feature_scope for i in range(len(meta_types))]
            applicable, parameters = convert_range(boolean_relevant_scope, boolean_feature_scope, meta_types, ranges[0],
                                                   inverted_features)

        # generated C++ code
        if use_generated_code and applicable:
            time_start = perf_counter()
            import optimized_inference

            spn_func = getattr(optimized_inference, f'FACE{spn_id}')
            result = np.array([[spn_func(*parameters)]])

            time_end = perf_counter()

            if gen_code_stats is not None:
                gen_code_stats.calls += 1
                gen_code_stats.total_time += (time_end - time_start)

            # logger.debug(f"\t\tGenerated Code Latency: {(time_end - time_start) * 1000:.3f}ms")
            return result

        # lightweight non-batch version
        if not use_generated_code and not applicable:
            if isinstance(FACE, Sum):
                return np.array([[sum([p * expectation(c, feature_scope, inverted_features, ranges, node_expectation,
                                                       node_likelihoods, use_generated_code, spn_id, meta_types,
                                                       gen_code_stats)[0, 0] for p, c in zip(FACE.weights, FACE.children)])]])
            elif isinstance(FACE, Product):
                return np.array([[np.prod([expectation(c, feature_scope, inverted_features, ranges, node_expectation,
                                                       node_likelihoods, use_generated_code, spn_id, meta_types,
                                                       gen_code_stats)[0, 0] for c in FACE.children])]])
            elif isinstance(FACE, Categorical): __name__ == "__main__":

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
