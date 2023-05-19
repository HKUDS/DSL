python3 main.py --data Yelp --save_name yelp_our \
--epoch 200 --lr 1e-3 --batch 4096 \
--latdim 256 --gnn_layer 2 \
--sBatch 4096 --uuPre_reg 1e0 --uugnn_layer 2 \
--sal_reg 1e-5 --reg 1e-6 \
--gpu 0