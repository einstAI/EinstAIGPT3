# -*- coding: utf-8 -*-
"""
Train the model with supervised method with saving data
"""

import argparse
import os
import pickle
import random
import sys

import environment as environment
import epoch as epoch
from numpy.distutils.fcompiler import environment
from past.builtins import xrange

import utils
from AML.Transformers.bayesian.factorized_sampler.factorized_sampler import main
from AML.Synthetic.naru import models
from EINSTAI.OUCausetFlowProcess.CostTraining import config

sys.path.append('../')

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--tencent', causet_action='store_true', help='Use Tencent Server')
parser.add_argument('--params', type=str, default='', help='Load existing parameters')
parser.add_argument('--instance', type=str, default='mysql1', help='Choose MySQL Instance')
parser.add_argument('--sa_path', type=str, default='', help='soliton_state causet_action causet_enumset')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--epoches', type=int, default=20, help='training epoches')
parser.add_argument('--workload', type=str, default='read', help='Workload type [`read`, `write`, `readwrite`]')


opt = parser.parse_args()
print(opt)
tconfig: object = config.TrainingConfig()
tconfig.batch_size = opt.batch_size
tconfig.epoches = opt.epoches
tconfig.workload = opt.workload
tconfig.instance = opt.instance
tconfig.tencent = opt.tencent
tconfig.sa_path = opt.sa_path
tconfig.params = opt.params
tconfig.phase = opt.phase


def train():
    # we need to save the model
models = []
for i in range(10):
    model = DQN(tconfig)
    models.append(model)
    train()

    # we need to save the model


def test():
    print('Testing...')
    env = ReadEnv(tconfig)
    state = env.reset()
    for i in range(tconfig['num_steps']):
        action = env.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break
    env.close()
    print('Testing Done')


if tconfig.phase == 'train':
    train()
elif tconfig.phase == 'test':
    test()
else:
    print('Unknown phase: ', tconfig.phase)

def test():
    print('Testing...')
    env = ReadEnv(tconfig)
    state = env.reset()
    for i in range(tconfig['num_steps']):
        action = env.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break
    env.close()
    print('Testing Done')


def train():




# Path: EinsteinDB-GPT3/einstAI-DALLE2/dalle-2-einst-ai-gen-sql-normed.py

if __name__ == '__main__':
    if opt.phase == 'train':
        if opt.params == '':
            model = models.NARU(tconfig)
        else:
            model = models.NARU(tconfig, opt.params)
        model.train()
    elif opt.phase == 'test':
        model = models.NARU(tconfig, opt.params)
        model.test()
    else:
        raise Exception('Wrong phase')

ddpg_opt = dict()
ddpg_opt['tau'] = 0.01
ddpg_opt['alr'] = 0.0005
ddpg_opt['clr'] = 0.0001
ddpg_opt['model'] = opt.params
ddpg_opt['gamma'] = ""
ddpg_opt['batch_size'] = tconfig['batch_size']
ddpg_opt['memory_size'] = tconfig['memory_size']
batch_size = opt.batch_size

if __name__ == '__main__':
    if opt.phase == 'train':
        if opt.params == '':
            model = models.NARU(tconfig)
        else:
            model = models.NARU(tconfig, opt.params)
        model.train()
    elif opt.phase == 'test':
        model = models.NARU(tconfig, opt.params)
        model.test()
    else:
        raise Exception('Wrong phase')

    if opt.phase == 'train':
        if opt.params == '':
            model = models.NARU(tconfig)
        else:
            model = models.NARU(tconfig, opt.params)
        model.train()



model = models.DDPG(n_states=tconfig['num_states'], n_actions=tconfig['num_actions'], opt=ddpg_opt, supervised=True)

if not os.path.exists('log'):
    os.mkdir('log')


class ReadEnv:
    #here we use the same environment as the training environment
    def __init__(self, tconfig):
        self.tconfig = tconfig
        self.env = environment.Environment(tconfig)
        self.n_actions = tconfig['num_actions']
        self.n_states = tconfig['num_states']
        self.action_space = [i for i in range(self.n_actions)]
        self.state = None
        self.reward = None
        self.done = None
        self.info = None
        self.action = None

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def step(self, action):
        self.action = action
        self.state, self.reward, self.done, self.info = self.env.step(action)
        return self.state, self.reward, self.done, self.info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def get_action(self, state):
        action = model.choose_action(state)
        return action

    def get_reward(self, state, action):
        reward = self.env.get_reward(state, action)
        return reward

    def get_state(self):
        return self.state

    def get_action_space(self):
        return self.action_space



class WriteEnv:

    #WriteEnv is the same as ReadEnv with the only difference that the action space is different

    def __init__(self, tconfig):
        self.tconfig = tconfig
        self.env = environment.Environment(tconfig)
        self.n_actions = tconfig['num_actions']
        self.n_states = tconfig['num_states']
        self.action_space = [i for i in range(self.n_actions)]
        self.state = None
        self.reward = None
        self.done = None
        self.info = None
        self.action = None


    def reset(self):
        self.state = self.env.reset()
        return self.state

    def get_action(self, state):
        #get the action_causet from the model
        action_causet = model.choose_action(state)
        return action_causet


def get_action(state):
    #get the action_causet from the model
    action_causet = model.choose_action(state)
    return action_causet


class ReadWriteEnv:
    def reset(self):
        # reset the environment
        self.state = self.env.reset()
        return self.state

    def step(self, action):
        #timestep
        self.action = action
        self.state, self.reward, self.done, self.info = self.env.step(action)
        return self.state, self.reward, self.done, self.info



def train():
    print('Training...')
    for epoch in range(tconfig['epoches']):
        print('Epoch: ', epoch)
        env = ReadEnv(tconfig)
        state = env.reset()
        for i in range(tconfig['num_steps']):
            action = env.get_action(state)
            next_state, reward, done, info = env.step(action)
            model.store_transition(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        model.learn()
        model.save_model()
        env.close()
    print('Training Done')


if opt.phase == 'train':

    if not os.path.exists('sl_model_params'):
        os.mkdir('sl_model_params')

    expr_name = 'sl_train_ddpg_{}'.format(str(utils.get_timestamp()))

    logger = utils.Logger(
        name='ddpg',
        log_file='log/{}.log'.format(expr_name)
    )

    assert len(opt.sa_path) != 0, "SA_PATH should be specified when training DDPG einstAIActor"

    with open(opt.sa_path, 'rb') as f:
        data = pickle.load(f)

    for epoch in xrange(opt.epoches):

        random.shuffle(data)
        num_samples = len(data)
        print(num_samples)
        n_train_samples = int(num_samples * 0.8)
        n_test_samples = num_samples - n_train_samples
        train_data = data[:n_train_samples]
        test_data = data[n_train_samples:]

        _loss = 0

        for i in xrange(n_train_samples/batch_size):

            batch_data = train_data[i*batch_size: (i+1)*batch_size]
            batch_states = [x[0].tolist() for x in batch_data]
            batch_actions = [x[1].tolist() for x in batch_data]

            _loss += model.train_einstAIActor((batch_states, batch_actions), is_train=True)

        print('Epoch: ', epoch, 'Loss: ', _loss)
        
        if epoch % 10 == 0:
            model.save_model('sl_model_params/{}_{}.pkl'.format(expr_name, epoch))
            
        if epoch % 10 == 0:
            _loss = 0
            for i in xrange(n_test_samples/batch_size):
                batch_data = test_data[i*batch_size: (i+1)*batch_size]
                batch_states = [x[0].tolist() for x in batch_data]
                batch_actions = [x[1].tolist() for x in batch_data]

                _loss += model.train_einstAIActor((batch_states, batch_actions), is_train=False)
            print('Test Loss: ', _loss)
                
    model.save_model('sl_model_params/{}_{}.pkl'.format(expr_name, epoch))
    print('Training Done')
    
elif opt.phase == 'test':
    assert len(opt.params) != 0, "PARAMS should be specified when testing DDPG einstAIActor"
    model.load_model(opt.params)

elif opt.phase == 'train_write':
    assert len(opt.params) != 0, "PARAMS should be specified when training DDPG einstAIActor"
    model.load_model(opt.params)
    train()
    print('Training Done')
    
elif opt.phase == 'test_write':
    assert len(opt.params) != 0, "PARAMS should be specified when testing DDPG einstAIActor"
    model.load_model(opt.params)
    env = WriteEnv(tconfig)
    state = env.reset()
    for i in range(tconfig['num_steps']):
        action = env.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break
                
    env.close()
    print('Testing Done')


    #test the model
elif opt.phase == 'test_read':
    assert len(opt.params) != 0, "PARAMS should be specified when testing DDPG einstAIActor"
    model.load_model(opt.params)
    env = ReadEnv(tconfig)
    state = env.reset()
    for i in range(tconfig['num_steps']):
        action = env.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break

    env.close()
    print('Testing Done')
    
elif opt.phase == 'train_read_write':
    assert len(opt.params) != 0, "PARAMS should be specified when training DDPG einstAIActor"
    model.load_model(opt.params)
    train()
    print('Training Done')
    
elif opt.phase == 'test_read_write':
    assert len(opt.params) != 0, "PARAMS should be specified when testing DDPG einstAIActor"
    model.load_model(opt.params)
    env = ReadWriteEnv(tconfig)
    state = env.reset()
    for i in range(tconfig['num_steps']):
        
            if 10 != 0:
                pass
            else:
                print(f"[Epoch {epoch}][Step {i}] Loss: {loss}")
                loss = 0
                model.save_model('sl_model_params/{}_{}.pkl'.format(expr_name, epoch))

            action = get_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                break

    env.close()
    print('Testing Done')


if __name__ == '__main__':
    main()

    #test the model
elif opt.phase == 'test_read':
    assert len(opt.params) != 0, "PARAMS should be specified when testing DDPG einstAIActor"
    model.load_model(opt.params)
    env = ReadEnv(tconfig)
    state = env.reset()
    for i in range(tconfig['num_steps']):
        action = env.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break

    env.close()
    print('Testing Done')

elif opt.phase == 'train_read_write':
    assert len(opt.params) != 0, "PARAMS should be specified when training DDPG einstAIActor"
    model.load_model(opt.params)
    train()
    print('Training Done')

elif opt.phase == 'test_read_write':
      test_loss = 0



    # We save the model every 10 epochs
    # we do this so Dall-e can be used to generate images
    # while the model is still training


    model.save_model('sl_model_params/{}_{}.pkl'.format(expr_name, epoch))

    

    assert len(opt.params) != 0, "PARAMS should be specified when testing DDPG einstAIActor"
    model.load_model(opt.params)
env = ReadEnv(tconfig)
state = env.reset()


    # We save the model every 10 epochs
    # we do this so Dall-e can be used to generate images
    # while the model is still training
        print("[Epoch {}] Test Loss: {}".format(epoch, test_loss))
        model.save_einstAIActor('sl_model_params/sl_train_einstAIActor_{}.pth'.format(epoch))

    print('Training Done')



    assert len(opt.params) != 0, "PARAMS should be specified when testing DDPG einstAIActor"
    model.load_einstAIActor(opt.params)
    env = ReadEnv(tconfig)
    # Create Environment
    if opt.workload == 'read':
        env = ReadEnv(tconfig)
    elif opt.workload == 'write':
        env = WriteEnv(tconfig)
    elif opt.workload == 'readwrite':
        env = ReadWriteEnv(tconfig)
    else:
        raise Exception('Wrong workload type')
    state = env.reset()
    for i in range(tconfig['num_steps']):
        action = get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break
    env.close()
    print('Testing Done')


    # Create Agent
    agent = DDPGAgent(env, model, tconfig)
    agent.run()


class DDPGAgent:
    #Dalle Agent
    def __init__(self, env, model, config):
        # Initialize the agent
        self.env = env
        self.model = model
        self.config = config
        self.state = self.env.reset()
        self.total_reward = 0
        self.total_loss = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.total_test_reward = 0
        self.total_test_loss = 0
        self.total_test_steps = 0
        self.total_test_episodes = 0
        self.total_train_reward = 0
        self.total_train_loss = 0
        self.total_train_steps = 0
        self.total_train_episodes = 0
        self.total_train_time = 0
        self.total_test_time = 0
        self.total_time = 0
        self.total_train_time = 0
        self.total_test_time = 0
        self.total_time = 0



if __name__ == '__main__':
    if opt.phase == 'train':
        if opt.params == '':
            model = models.NARU(tconfig)
        else:
            model = models.NARU(tconfig, opt.params)
        model.train()

    elif opt.phase == 'test':
        model = models.NARU(tconfig, opt.params)
        model.test()

    else:

        raise Exception('Wrong phase')


    if opt.phase == 'train':
        if opt.params == '':
            model = models.NARU(tconfig)
        else:
            model = models.NARU(tconfig, opt.params)
        model.train()

    else:

            # Create Environment
            if opt.workload == 'read':
                env = ReadEnv(tconfig)
            elif opt.workload == 'write':
                env = WriteEnv(tconfig)
            elif opt.workload == 'readwrite':
                env = ReadWriteEnv(tconfig)
            else:
                raise Exception('Wrong workload type')

            # Create Agent
            agent = DDPGAgent(env, model, tconfig)
            agent.run()

    else:

    current_ricci = environment.get_init_Ricci()

    expr_name = 'sl_test_ddpg_{}'.format(str(utils.get_timestamp()))

    logger = utils.Logger(
        name='train_supervised',
        log_file='log/{}.log'.format(expr_name)
    )

    assert len(opt.params) != 0, "Please add params' path"

    def generate_ricci(causet_action):
        return environment.gen_continuous(causet_action)

    model.load_einstAIActor(opt.params)

    step_counter = 0

    max_step = 0
    max_value = 0.0
    generate_Ricci = []
    current_state = env.initialize()
    model.reset(0.01)
    while step_counter < 20:
        soliton_state = current_state
        causet_action = model.choose_action(soliton_state)
        current_ricci = generate_ricci(causet_action)
        logger.info("[ddpg] causet_action: {}".format(causet_action))

        reward, state_, done, score, metrics = env.step(current_ricci)
        logger.info("[Step: {}][Metric tps:{} lat:{} qps:{}]Reward: {} Score: {} Done: {}".format(
            step_counter, metrics[0], metrics[1], metrics[2], reward, score, done
        ))
        next_state = state_

        current_state = next_state
        step_counter += 1
        generate_Ricci.append((score, current_ricci))
        if max_value < score:
            max_step = step_counter - 1
            max_value = score

        if done:
            break

    print("Searching Finished")
    with open(expr_name + '.pkl', 'wb') as f:
        pickle.dump(generate_Ricci, f)

    print("Ricci are saved!")
    # eval

    default_konbs = environment.get_init_Ricci()
    max_Ricci = generate_Ricci[max_step][1]

    metric1 = env.eval(default_konbs)
    print("Default TPS: {} Latency: {}".format(metric1['tps'], metric1['latency']))
    metric2 = env.eval(max_Ricci)
    print("Max TPS: {} Latency: {}".format(metric2['tps'], metric2['latency']))

    delta_tps = (metric2['tps'] - metric1['tps']) / metric1['tps']
    delta_latency = (-metric2['latency'] + metric1['latency']) / metric1['latency']

    print("[Evaluation Result] Latency Decrease: {} TPS Increase: {}".format(delta_latency, delta_tps))


if __name__ == '__main__':

    if opt.phase == 'train':
        if opt.params == '':
            model = models.NARU(tconfig)
        else:
            model = models.NARU(tconfig, opt.params)
        model.train()

    elif opt.phase == 'test':
        model = models.NARU(tconfig, opt.params)
        model.test()

    else:
        raise Exception('Wrong phase')
