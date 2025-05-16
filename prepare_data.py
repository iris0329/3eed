import argparse

from src.joint_det_dataset import Joint3DDataset


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True)
args, _ = parser.parse_known_args()
Joint3DDataset(split='train', data_path=args.data_root)
Joint3DDataset(split='val', data_path=args.data_root)
