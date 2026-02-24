#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 20/Jul/2024  22:31
import torch
import argparse
from src.environments import load
from src.game import Game
from src.player import Human, AlphaZeroPlayer, NetworkPlayer

parser = argparse.ArgumentParser(description='Play connect four against AlphaZero!')
parser.add_argument('-x', action='store_true', help='Play as X')
parser.add_argument('-o', action='store_true', help='Play as O')
parser.add_argument('-n', type=int, default=500,
                    help='Number of simulations before AlphaZero make an action')
parser.add_argument('--sp', action='store_true',
                    help='AlphaZero play against itself')
parser.add_argument('--model', type=str, default='current', help='Model type')
parser.add_argument('--network', type=str, default='CNN', help='Network type')
parser.add_argument('--env', type=str, default='Connect4', help='env name')
parser.add_argument('--name', type=str, default='AZ', help='Model name')
parser.add_argument('-c', '--c_init', type=float, default=4, help='C_puct init')
parser.add_argument('-a', '--alpha', type=float, default=0.1, help='Dirichlet alpha')
parser.add_argument('--no_symmetry', action='store_true', help='Disable random symmetry augmentation during MCTS search')
parser.add_argument('--lambda_s', type=float, default=0.1,
                    help='Steps-value mixing weight (default=0.1)')
parser.add_argument('--mlh_factor', type=float, default=0.0,
                    help='Moves Left Head factor (0=disabled, recommended 0.2-0.3)')
parser.add_argument('--mlh_threshold', type=float, default=0.85,
                    help='MLH activation threshold')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    module = load(args.env)
    try:
        env = module.Env()
        game = Game(env)

        if args.network == 'CNN':
            net = module.CNN(0, device=device, lambda_s=args.lambda_s)
        elif args.network == 'ViT':
            net = module.ViT(0, device=device, lambda_s=args.lambda_s)
        else:
            raise ValueError(f"Unknown network type: {args.network}")

        net.load(f'./params/{args.name}_{args.env}_{args.network}_{args.model}.pt')

        if args.n == 0:
            az_player = NetworkPlayer(net)
        else:
            az_player = AlphaZeroPlayer(net, c_init=args.c_init,
                                        n_playout=args.n, discount=0.99, alpha=args.alpha, is_selfplay=0,
                                        use_symmetry=not args.no_symmetry,
                                        mlh_factor=args.mlh_factor, mlh_threshold=args.mlh_threshold)
        az_player.eval()

        human = Human()
        
        if args.x and args.o:
            game.play(human, human)
        elif args.x:
            game.play(human, az_player)
        elif args.o:
            game.play(az_player, human)
        elif args.sp and not (args.x or args.o):
            game.play(az_player, az_player)
        else:
            raise AttributeError("Invalid argument(s).\nType 'python3 ./play.py -h' for help")
    except KeyboardInterrupt:
        print('\n\rquit')
        