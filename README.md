# AlphaZero-AL

AlphaZero Actor-Learner framework with transposition table
Based on https://github.com/Sunshine-718/AlphaZero

<hr>

## Keywords

`AlphaGo Zero` `AlphaZero` `Monte Carlo Tree Search` `Reinforcement Learning (RL)` `Model-based RL` `Tree Search` `Heuristic Search` `Zero-sum Game`

<hr>

## Environment

[`Python 3.13`](https://www.python.org)
[`torch 2.7.0`](https://pytorch.org)

<hr>

## Get started

Prerequisite: [Pytorch](https://pytorch.org) is installed in your device.
Run `build.bat` on Windows or `build.sh` on UNIX-like systems.

<hr>

## Play with AlphaZero

### Play in terminal

Type one of the command lines below in terminal:

```shell
python3 play.py -x	# play as X
```

```shell
python3 play.py -o	# play as O
```

and input 0-6 for each column, i.e., 0 for the 1st column, 1 for the 2nd column.
Optional arguments:
`-x`: Play as X
`-o`: Play as O
`-n`: Number of simulation before AlphaZero make an action, set higher for more powerful policy (theoretically), default: `500`.
`--self_play`: AlphaZero will play against itself.
`--model`: current model or best model, default: `current`.
`--name`: model name, default: `AZ`.

## Play in GUI

Run [`gui_play.py`](gui_play.py) to play Connect 4 with AlphaZero in GUI.

<hr>

## How to train your own AlphaZero?

### Launch server:

```shell
python3 server.py --host 0.0.0.0	# any host ip address you want
```

### Launch client:

```shell
chmod +x ./client.sh
./client.sh 5 --host 127.0.0.1 -n 100
```

`5`: number of clients/actor.
`-n`: n_playout.

#### Other arguments details:

see: [`server.py`](server.py) and [`client.py`](client.py) for more arguments.

#### How to monitor the training procedure?

The training procedure is monitored using tensorboard, you can open tensorboard by typing the command below:

```shell
tensorboard --logdir runs --port 6006
```

After launched tensorboard service, open the browser and type `http://localhost:6006` in the URL bar.

<hr>

## References

[Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017).](https://doi.org/10.1038/nature24270)

[David Silver et al. ,A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.](https://doi.org/10.1126/science.aar6404)
