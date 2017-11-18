wget -O model93.ckpt.data-00000-of-00001 https://www.dropbox.com/s/sxjk6ioe35a2m2t/model93.ckpt.data-00000-of-00001?dl=1
wget -O model93.ckpt.index https://www.dropbox.com/s/vqee5etqr18hqrc/model93.ckpt.index?dl=1
wget -O model93.ckpt.meta https://www.dropbox.com/s/eduln9k7ha7k4zs/model93.ckpt.meta?dl=1
python3 ./s2vtPredict-Copy1.py $1 $2
