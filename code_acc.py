parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'Pavia', 'Salinas', 'KSC', 'Botswana', 'Houston'],
                    default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')

parser.add_argument('--epoches', type=int, default=200, help='epoch number')
parser.add_argument('--patches', type=int, default=9, help='number of patches')#奇数
parser.add_argument('--band_patches', type=int, default=3, help='number of related band')#奇数
parser.add_argument('--n_gcn', type=int, default=21, help='number of related pix')
parser.add_argument('--pca_band', type=int, default=70, help='pca_components')

parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')
# Final result:
# OA: 0.9287 | AA: 0.9552 | Kappa: 0.9186
# [0.96774194 0.86051502 0.90375    0.98550725 0.92715232 0.97714286
#  1.         1.         1.         0.93524416 0.87092784 0.93428064
#  1.         0.99271255 0.96067416 0.96825397]
# **************************************************


parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--patches', type=int, default=11, help='number of patches')#奇数
parser.add_argument('--band_patches', type=int, default=3, help='number of related band')#奇数
parser.add_argument('--n_gcn', type=int, default=21, help='number of related pix')
parser.add_argument('--pca_band', type=int, default=70, help='pca_components')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight_decay')

# **************************************************
# Final result:
# OA: 0.9314 | AA: 0.9551 | Kappa: 0.9217
# [0.96774194 0.91917024 0.91375    0.97101449 0.92935982 0.96428571
#  0.92307692 1.         1.         0.89278132 0.85814433 0.88987567
#  0.97714286 0.99109312 0.99719101 0.98412698]
# **************************************************


parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--patches', type=int, default=9, help='number of patches')#奇数
parser.add_argument('--band_patches', type=int, default=3, help='number of related band')#奇数
parser.add_argument('--n_gcn', type=int, default=21, help='number of related pix')
parser.add_argument('--pca_band', type=int, default=70, help='pca_components')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
# **************************************************
# Final result:
# OA: 0.9379 | AA: 0.9652 | Kappa: 0.9291
# [1.         0.88841202 0.90375    1.         0.93818985 0.97571429
#  1.         0.99776786 1.         0.93949045 0.9142268  0.89165187
#  1.         0.99757085 0.99719101 1.        ]
# **************************************************



parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--patches', type=int, default=9, help='number of patches')#奇数
parser.add_argument('--band_patches', type=int, default=3, help='number of related band')#奇数
parser.add_argument('--n_gcn', type=int, default=21, help='number of related pix')
parser.add_argument('--pca_band', type=int, default=70, help='pca_components')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

#######GCN为四层##############

# **************************************************
# Final result:
# OA: 0.9416 | AA: 0.9651 | Kappa: 0.9333
# [1.         0.83690987 0.9075     0.99516908 0.92273731 0.97285714
#  0.84615385 1.         1.         0.9596603  0.88412371 0.90586146
#  0.98857143 0.99757085 0.99438202 0.95238095]
# **************************************************