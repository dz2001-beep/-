import argparse

parser = argparse.ArgumentParser()

# ---------------------------File--------------------------- #
parser.add_argument('--data_path',          default='./data')
parser.add_argument('--mobility_adj',       default='/mobility_adj.npy')
parser.add_argument('--poi_similarity',     default='/poi_similarity.npy')
parser.add_argument('--source_adj',         default='/source_adj.npy')
parser.add_argument('--destination_adj',    default='/destination_adj.npy')
parser.add_argument('--mh_cd',              default='/mh_cd.json')
parser.add_argument('--crime_counts',       default='/crime_counts.npy')
parser.add_argument('--check_counts',       default='/check_counts.npy')
parser.add_argument('--neighbor',           default='/neighbor.npy')

# ---------------------------model--------------------------- #
parser.add_argument('--device', default='cuda')
parser.add_argument('--epochs', default = 579) # 300
parser.add_argument('--wd', type=int, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float,  default=0.001)
parser.add_argument('--embedding_size', default=144)
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--TopK', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--regions_num', type=int, default=180)
parser.add_argument('--importance_k', type=int, default=10)

args = parser.parse_args()
