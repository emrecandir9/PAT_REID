import os
import csv
import glob
import torch
import argparse

import numpy as np
from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking
from data.build_DG_dataloader import build_reid_test_loader
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from processor.part_attention_vit_processor import do_inference as do_inf_pat

#from torch.backends import cudnn


def find_best_weight(log_root, log_name, model_name):
    """Auto-detect the latest/best checkpoint from the training output directory.
    
    Looks in LOG_ROOT/LOG_NAME for files matching {model_name}_*.pth
    and returns the one with the highest epoch number.
    """
    checkpoint_dir = os.path.join(log_root, log_name)
    pattern = os.path.join(checkpoint_dir, f'{model_name}_*.pth')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found matching '{pattern}'. "
            f"Make sure training has completed and checkpoints are saved in '{checkpoint_dir}'."
        )
    
    # Extract epoch number from filename (e.g., part_attention_vit_40.pth -> 40)
    def get_epoch(path):
        basename = os.path.basename(path)
        # Remove model_name prefix and .pth suffix, get the epoch number
        epoch_str = basename.replace(f'{model_name}_', '').replace('.pth', '')
        try:
            return int(epoch_str)
        except ValueError:
            return -1
    
    best = max(checkpoints, key=get_epoch)
    print(f'[Auto-Weight] Found {len(checkpoints)} checkpoint(s). Using: {best}')
    return best


def extract_feature(model, dataloaders, num_query):
    features = []
    count = 0
    img_path = []

    for data in dataloaders:
        img, a, b,_,_ = data.values()
        #obtain values form dict data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, 768).zero_().cuda()  # 2048 is pool5 of resnet
        for i in range(2):
            input_img = img.cuda()
            outputs = model(input_img)
            f = outputs.float()
            ff = ff + f
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features.append(ff)
    features = torch.cat(features, 0)

    # query
    qf = features[:num_query]
    # gallery
    gf = features[num_query:]
    return qf, gf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--track", default="/kaggle/working/submission.txt", help="path to store the submission files", type=str
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # Use only first GPU for inference (handles '0,1' from DDP training config)
    device_id = cfg.MODEL.DEVICE_ID
    if ',' in device_id:
        device_id = device_id.split(',')[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    # Auto-detect weight if set to 'auto'
    weight_path = cfg.TEST.WEIGHT
    if weight_path.lower() == 'auto':
        weight_path = find_best_weight(cfg.LOG_ROOT, cfg.LOG_NAME, cfg.MODEL.NAME)
    logger.info(f"Loading model weights from: {weight_path}")

    model = make_model(cfg, cfg.MODEL.NAME, 0,0,0)
    model.load_param(weight_path)

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query, imgpath_to_class = build_reid_test_loader(cfg, testname)
        if cfg.MODEL.NAME == 'part_attention_vit':
            do_inf_pat(cfg, model, val_loader, num_query, imgpath_to_class=imgpath_to_class)
        else:
            do_inf(cfg, model, val_loader, num_query, imgpath_to_class=imgpath_to_class)
    with torch.no_grad():
        qf, gf = extract_feature(model, val_loader, num_query)

    # save features to the working directory
    feat_dir = os.path.dirname(args.track) or '/kaggle/working'
    qf_np = qf.cpu().numpy()
    gf_np = gf.cpu().numpy()
    np.save(os.path.join(feat_dir, "qf.npy"), qf_np)
    np.save(os.path.join(feat_dir, "gf.npy"), gf_np)

    q_g_dist = np.dot(qf_np, np.transpose(gf_np))
    q_q_dist = np.dot(qf_np, np.transpose(qf_np))
    g_g_dist = np.dot(gf_np, np.transpose(gf_np))

    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    indices = np.argsort(re_rank_dist, axis=1)[:, :100]

    m, n = indices.shape
    # Write the raw track file
    with open(args.track, 'wb') as f_w:
        for i in range(m):
            write_line = indices[i] + 1
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f_w.write(write_line.encode())

    # Build query image name list from the actual dataset (not sequential)
    # The query images come from val_loader's dataset in order
    query_img_paths = []
    dataset_items = val_loader.dataset.img_items[:num_query]
    for item in dataset_items:
        img_path = item[0] if isinstance(item, (list, tuple)) else item['img_path']
        query_img_paths.append(os.path.basename(img_path))

    # Write the submission CSV
    output_path = args.track.replace('.txt', '') + "_submission.csv"
    with open(output_path, 'w', newline='') as archivo_csv:
        csv_writter = csv.writer(archivo_csv)
        csv_writter.writerow(['imageName', 'Corresponding Indexes'])
        for q_name, track in zip(query_img_paths, indices):
            track_str = ' '.join(map(str, track + 1))
            csv_writter.writerow([q_name, track_str])

    logger.info(f"Track file saved to: {args.track}")
    logger.info(f"Submission CSV saved to: {output_path}")
    logger.info("Done!")
