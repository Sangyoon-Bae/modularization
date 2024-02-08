# argparse example file

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='Stella')
parser.add_argument('--favorite_animal', type=str, default='dog')
parser.add_argument('--today', action='store_true') # 파이썬 실행할 때 python greetings.py --today를 하면 args.today가 True가 됨
parser.add_argument('--yesterday', action='store_false') # 파이썬 실행할 때 python greetings.py --yesterday를 하면 args.yesterday가 False가 됨
args = parser.parse_args()

print(f'Hello {args.name}. Your favorite animal is {args.favorite_animal}. Value of today is {args.today}. Value of yesterday is {args.yesterday}')