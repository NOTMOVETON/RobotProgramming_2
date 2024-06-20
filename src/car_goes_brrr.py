import argparse
from Params import Params
from Agent import Agent

if __name__ == '__main__':

    agent = Agent()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', required=True,
                        choices=['train', 'evaluate_human', 'evaluate_record'],
                        help='Which model to run / train.')
    args = vars(parser.parse_args())

    if args['action'] == 'train':
        agent.train(Params('../params/train.yaml'))

    elif args['action'] == 'evaluate_human':
        agent.eval(Params('../params/eval.yaml'), 'human')

    elif args['action'] == 'evaluate_record':
        agent.eval(Params('../params/eval.yaml'), 'video')