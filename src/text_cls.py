import argparse
import csv
import json
import os
import os.path as osp

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizerFast
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer:
    """Train/test models on manipulation."""

    def __init__(self, model, data_loaders, args):
        self.model = model
        self.data_loaders = data_loaders
        self.args = args

        self.optimizer = Adam(
            model.parameters(), lr=args.lr, betas=(0.0, 0.9), eps=1e-8
        )

    def run(self):
        # Set
        start_epoch = 0
        val_acc_prev_best = -1.0

        # Load
        if osp.exists(self.args.ckpnt):
            start_epoch, val_acc_prev_best = self._load_ckpnt()

        # Eval?
        if self.args.eval or start_epoch >= self.args.epochs:
            self.model.eval()
            self.train_test_loop('val')
            return self.model

        # Go!
        for epoch in range(start_epoch, self.args.epochs):
            print("Epoch: %d/%d" % (epoch + 1, self.args.epochs))
            self.model.train()
            # Train
            self.train_test_loop('train', epoch)
            # Validate
            print("\nValidation")
            self.model.eval()
            with torch.no_grad():
                val_acc = self.train_test_loop('val', epoch)

            # Store
            if val_acc >= val_acc_prev_best:
                print("Saving Checkpoint")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_acc": val_acc
                }, self.args.ckpnt)
                val_acc_prev_best = val_acc
            else:
                print("Updating Checkpoint")
                checkpoint = torch.load(self.args.ckpnt)
                checkpoint["epoch"] += 1
                torch.save(checkpoint, self.args.ckpnt)

        return self.model

    def _load_ckpnt(self):
        ckpnt = torch.load(self.args.ckpnt)
        self.model.load_state_dict(ckpnt["model_state_dict"], strict=False)
        # self.optimizer.load_state_dict(ckpnt["optimizer_state_dict"])
        start_epoch = ckpnt["epoch"]
        val_acc_prev_best = ckpnt['best_acc']
        return start_epoch, val_acc_prev_best

    def train_test_loop(self, mode='train', epoch=1000):
        n_correct, n_samples = 0, 0
        total_scores = []
        all_utterances = []
        for step, ex in tqdm(enumerate(self.data_loaders[mode])):

            # Forward pass
            scores = self.model(ex['utterance'])

            # Losses
            loss = F.binary_cross_entropy_with_logits(
                scores, ex['positive_map'][:, :scores.size(1)].to(DEVICE)
            )

            # Update
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging
            if not self.args.store:
                n_samples += len(scores)
                n_correct += (
                    (scores.detach().cpu() > 0).float()
                    == ex['positive_map'][:, :scores.size(1)]
                ).all(1).sum().item()
            else:
                pad_scores = np.zeros((len(scores), 256))
                pad_scores[:, :scores.size(1)] = (
                    (scores.detach().cpu() > 0).float().numpy()
                )
                argmaxes = scores.detach().cpu().argmax(1)
                argmaxes = F.one_hot(argmaxes, 256)
                is_zero = pad_scores.sum(1) < 1
                pad_scores[is_zero] = argmaxes[is_zero]
                total_scores.append(pad_scores / pad_scores.sum(1)[:, None])
                all_utterances.extend(ex['orig_utterance'])
        if not self.args.store:
            acc = n_correct / n_samples
            print(acc)
        else:
            acc = 0
            total_scores = np.concatenate(total_scores).tolist()
            save_obj = [
                {'utterance': utt, 'span': span}
                for utt, span in zip(all_utterances, total_scores)
            ]
            with open(f'{self.args.dataset}_pred_spans.json', 'w') as fid:
                json.dump(save_obj, fid)
        return acc


class Joint3DDataset(Dataset):
    """Dataset utilities for ReferIt3D."""

    def __init__(self, dataset='sr3d',
                 split='train',
                 data_path='./', store=False):
        """Initialize dataset (here for ReferIt3D utterances)."""
        self.split = split
        self.data_path = data_path
        self.store = store
        self.annos = self.load_annos(dataset)
        self.tokenizer = RobertaTokenizerFast.from_pretrained("./data/roberta_base/")

    def load_annos(self, dset):
        """Load annotations of given dataset."""
        loaders = {
            'nr3d': self.load_nr3d_annos,
            'sr3d': self.load_sr3d_annos,
            'sr3d+': self.load_sr3dplus_annos,
            'scanrefer': self.load_scanrefer_annos
        }
        annos = loaders[dset]()
        return annos

    def load_sr3dplus_annos(self):
        """Load annotations of sr3d/sr3d+."""
        return self.load_sr3d_annos(dset='sr3d+')

    def load_sr3d_annos(self, dset='sr3d'):
        """Load annotations of sr3d/sr3d+."""
        split = self.split
        if split == 'val':
            split = 'test'
        if self.store:
            with open('data/meta_data/sr3d_train_scans.txt') as f:
                scan_ids = set(eval(f.read()))
            with open('data/meta_data/sr3d_test_scans.txt') as f:
                scan_ids = scan_ids.union(set(eval(f.read())))
        else:
            with open('data/meta_data/sr3d_%s_scans.txt' % split) as f:
                scan_ids = set(eval(f.read()))
        with open(self.data_path + 'refer_it_3d/%s.csv' % dset) as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            annos = [
                {
                    'scan_id': line[headers['scan_id']],
                    'target_id': int(line[headers['target_id']]),
                    'distractor_ids': eval(line[headers['distractor_ids']]),
                    'utterance': line[headers['utterance']],
                    'target': line[headers['instance_type']],
                    'anchors': eval(line[headers['anchors_types']]),
                    'anchor_ids': eval(line[headers['anchor_ids']]),
                    'dataset': dset
                }
                for line in csv_reader
                if line[headers['scan_id']] in scan_ids
                and
                str(line[headers['mentions_target_class']]).lower() == 'true'
            ]
        return annos

    def load_nr3d_annos(self):
        """Load annotations of nr3d."""
        split = self.split
        if split == 'val':
            split = 'test'
        if self.store:
            with open('data/meta_data/nr3d_train_scans.txt') as f:
                train_scan_ids = set(eval(f.read()))
            with open('data/meta_data/nr3d_test_scans.txt') as f:
                test_scan_ids = set(eval(f.read()))
            scan_ids = train_scan_ids.union(test_scan_ids)
        else:
            with open('data/meta_data/nr3d_%s_scans.txt' % split) as f:
                scan_ids = set(eval(f.read()))
        with open(self.data_path + 'refer_it_3d/nr3d.csv') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            annos = [
                {
                    'scan_id': line[headers['scan_id']],
                    'target_id': int(line[headers['target_id']]),
                    'target': line[headers['instance_type']],
                    'utterance': line[headers['utterance']],
                    'anchor_ids': [],
                    'anchors': [],
                    'dataset': 'nr3d'
                }
                for line in csv_reader
                if line[headers['scan_id']] in scan_ids
                and (
                    str(line[headers['mentions_target_class']]).lower()
                    == 'true'
                    or self.store
                )
                and
                (
                    str(line[headers['correct_guess']]).lower() == 'true'
                    or line[headers['scan_id']] in train_scan_ids
                    or self.store
                )
            ]
        if not self.store:  # train only on sentences that contain the target
            annos = [
                anno for anno in annos if anno['target'] in anno['utterance']
            ]
        else:  # all, assign a fake target just to run batched version
            for anno in annos:
                if anno['target'] not in anno['utterance']:
                    anno['target'] = anno['utterance'].split()[0].strip(',')
        return annos

    def load_scanrefer_annos(self):
        """Load annotations of ScanRefer."""
        _path = self.data_path + 'scanrefer/ScanRefer_filtered'
        split = self.split
        if split in ('val', 'test'):
            split = 'val'
        with open(_path + '_%s.txt' % split) as f:
            scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]
        with open(_path + '_%s.json' % split) as f:
            reader = json.load(f)
        annos = [
            {
                'scan_id': anno['scene_id'],
                'target_id': int(anno['object_id']),
                'distractor_ids': [],
                'utterance': ' '.join(anno['token']),
                'target': ' '.join(str(anno['object_name']).split('_')),
                'anchors': [],
                'anchor_ids': [],
                'dataset': 'scanrefer'
            }
            for anno in reader
            if anno['scene_id'] in scan_ids
        ]
        # Fix missing target reference
        for anno in annos:
            if anno['target'] not in anno['utterance']:
                if anno['target'].split()[-1] in anno['utterance']:
                    anno['target'] = anno['target'].split()[-1]
        print(len(annos))
        if not self.store:  # train only on sentences that contain the target
            annos = [
                anno for anno in annos if anno['target'] in anno['utterance']
            ]
        else:  # assign fake target for batching
            for anno in annos:
                if anno['target'] not in anno['utterance']:
                    anno['target'] = anno['utterance'].split()[0].strip(',')
        print(len(annos))
        return annos

    def _get_token_positive_map(self, anno):
        """Return correspondence of boxes to tokens."""
        # Token start-end span in characters
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())
        caption = ' ' + caption + ' '
        tokens_positive = torch.zeros((1, 2))
        if isinstance(anno['target'], list):
            cat_names = anno['target']
        else:
            cat_names = [anno['target']]
        for c, cat_name in enumerate(cat_names):
            start_span = caption.find(' ' + cat_name + ' ')
            len_ = len(cat_name)
            if start_span < 0:
                start_span = caption.find(' ' + cat_name)
                len_ = len(caption[start_span + 1:].split()[0])
            if start_span < 0:
                start_span = caption.find(cat_name)
                orig_start_span = start_span
                while caption[start_span - 1] != ' ':
                    start_span -= 1
                len_ = len(cat_name) + orig_start_span - start_span
                while caption[len_ + start_span] != ' ':
                    len_ += 1
            end_span = start_span + len_
            assert start_span > -1, caption
            assert end_span > 0, caption
            tokens_positive[c][0] = start_span
            tokens_positive[c][1] = end_span

        # Positive map (for soft token prediction)
        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )
        # positive_map = torch.zeros((128, 256))
        gt_map = get_positive_map(tokenized, tokens_positive[:len(cat_names)])
        # positive_map[:len(cat_names)] = gt_map
        return tokens_positive, gt_map[0]

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = self.annos[index]
        _, positive_map = self._get_token_positive_map(anno)
        tokenized = ' '.join(self.tokenizer.tokenize(
            ' '.join(anno['utterance'].replace(',', ' ,').split()),
        ))
        return {
            'utterance': (
                ' '.join(anno['utterance'].replace(',', ' ,').split())
                + ' . not mentioned'
            ),
            "positive_map": positive_map.float(),
            'orig_utterance': anno['utterance'],
            "tokenized": tokenized
        }


def get_positive_map(tokenized, tokens_positive):
    """Construct a map of box-token associations."""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)

    return positive_map


class TextClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        t_type = "./data/roberta_base/"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
        self.text_encoder = RobertaModel.from_pretrained(t_type)
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, utterances):
        tokenized = self.tokenizer.batch_encode_plus(
            utterances, padding="longest", return_tensors="pt"
        ).to(DEVICE)
        encoded_text = self.text_encoder(**tokenized)
        return self.text_projector(encoded_text.last_hidden_state).squeeze(-1)


def main():
    """Run main training/test pipeline."""
    data_path = '/projects/katefgroup/language_grounding/'

    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--checkpoint_path", default="checkpoints/")
    argparser.add_argument("--checkpoint", default="sr3d.pt")
    argparser.add_argument("--dataset", default="sr3d")
    argparser.add_argument("--epochs", default=20, type=int)
    argparser.add_argument("--batch_size", default=128, type=int)
    argparser.add_argument("--lr", default=1e-4, type=float)
    argparser.add_argument("--eval", action='store_true')
    argparser.add_argument("--store", action='store_true')

    args = argparser.parse_args()
    args.ckpnt = osp.join(args.checkpoint_path, args.checkpoint)

    # Other variables
    args.device = DEVICE
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Loaders
    datasets = {
        split: Joint3DDataset(args.dataset, split, data_path, args.store)
        for split in ('train', 'val')
    }
    print(len(datasets['train']), len(datasets['val']))
    data_loaders = {
        mode: DataLoader(
            datasets[mode],
            batch_size=args.batch_size,
            shuffle=mode == 'train',
            drop_last=mode == 'train',
            num_workers=4
        )
        for mode in ('train', 'val')
    }

    # Models
    model = TextClassifier()
    trainer = Trainer(model.to(args.device), data_loaders, args)
    trainer.run()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
