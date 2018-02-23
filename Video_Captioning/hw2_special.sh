wget -O model200.ckpt.data-00000-of-00001 https://www.dropbox.com/s/8b3zdy0bks31ed9/model200.ckpt.data-00000-of-00001?dl=1
wget -O model200.ckpt.index https://www.dropbox.com/s/9ty1tt8xrw10e9b/model200.ckpt.index?dl=1
wget -O model200.ckpt.meta https://www.dropbox.com/s/jpfbz4yeiwpsk9m/model200.ckpt.meta?dl=1
python ./hw2special.py $1 $2
