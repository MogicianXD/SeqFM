
nohup python -u main.py --epochs 2400 --cuda 3 --data Toys > Toys 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 5 --data Beauty > Beauty 2>&1 &
nohup python -u main.py --epochs 1200 --cuda 2 --data Foursquare > Foursquare 2>&1 &
nohup python -u main.py --epochs 1200 --cuda 2 --data Gowalla > Gowalla 2>&1 &
nohup python -u main.py --epochs 1200 --cuda 5 --data Trivago > Trivago 2>&1 &
nohup python -u main.py --epochs 1200 --cuda 4 --data Taobao > Taobao 2>&1 &