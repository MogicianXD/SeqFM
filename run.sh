
nohup python -u main.py --epochs 2400 --cuda 3 --data Toys > Toys 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 5 --data Beauty > Beauty 2>&1 &

nohup python -u main.py --epochs 2400 --cuda 0 --data Trivago --reload > Trivago 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 3 --data Foursquare --reload > Foursquare 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 5 --data Gowalla --reload  > Gowalla 2>&1 &
nohup python -u main.py --epochs 1200 --cuda 1 --data Taobao --reload  > Taobao 2>&1 &

nohup python -u main.py --epochs 2400 --cuda 5 --data Trivago --n_head 6 --reload > Trivago6 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 0 --data Foursquare --n_head 6 --reload > Foursquare6 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 3 --data Gowalla --n_head 6 --reload > Gowalla6 2>&1 &
nohup python -u main.py --epochs 1200 --cuda 5 --data Taobao --n_head 6 --reload > Taobao6 2>&1 &

nohup python -u main.py --epochs 2400 --cuda 4 --data Trivago --unshared --reload > Trivago2 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 3 --data Foursquare --unshared --reload > Foursquare2 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 0 --data Gowalla --unshared --reload > Gowalla2 2>&1 &
nohup python -u main.py --epochs 1200 --cuda 0 --data Taobao --unshared --reload > Taobao2 2>&1 &

nohup python -u main.py --epochs 2400 --cuda 4 --data Trivago --gru > Trivago4 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 3 --data Foursquare --gru > Foursquare4 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 0 --data Gowalla --gru > Gowalla4 2>&1 &
nohup python -u main.py --epochs 1200 --cuda 5 --data Taobao --gru > Taobao4 2>&1 &

nohup python -u main.py --epochs 2400 --cuda 3 --data Trivago --position > Trivago5 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 1 --data Foursquare --position > Foursquare5 2>&1 &
nohup python -u main.py --epochs 2400 --cuda 0 --data Gowalla --position > Gowalla5 2>&1 &
nohup python -u main.py --epochs 1200 --cuda 5 --data Taobao --position > Taobao5 2>&1 &
