import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart contract vulnerability detection based on graph neural network')
    parser.add_argument('-M', '--model', type=str, default='transformer',
                        choices=[
                                 'gat_lstm', 'gat_lstm_contract', 
                                 ], help="选择使用的模型")
    parser.add_argument('-DT', '--data_type', type=str, default='vectortype',
                        choices=['vectortype', 'vector',
                                 'type', 'graph_vectortype',
                                 'graph_vector', 'graph_type'], help="选择使用的数据类型")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('-C', '--type', type=str, default='access_control',
                        choices=['access_control', 'arithmetic',
                                 'denial_of_service', 'front_running',
                                 'reentrancy', 'time_manipulation',
                                 'unchecked_low_level_calls'],
                        help='categories of vulnerabilities')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--model_dim', type=int, default=640, help='number of transformers model_dim')

    parser.add_argument('--pca', type=str, default='False')

    return parser.parse_args()
