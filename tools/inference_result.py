import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)[
                :, :self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['NUM_POINT_FEATURES']
            ]
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        # index 대신 실제 파일명(확장자 제외)을 frame_id로 사용 -> 결과 txt에 000010 형태로 기록됨
        frame_id = Path(self.sample_file_list[index]).stem

        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kriso_models/kriso_voxelnext.yaml',
                         help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                         help='specify the point cloud data file or directory (예: 0번 폴더)')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--score_thresh', type=float, default=0.3,
                         help='score threshold for filtering predictions (default: 0.3)')
    parser.add_argument('--output_name', type=str, default=None,
                         help="저장할 txt 파일명. 지정 안 하면 데이터 폴더 이름 기준으로 '<폴더이름>.txt' 로 저장")
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def filter_by_score(pred_dict, score_thresh):
    mask = pred_dict['pred_scores'] >= score_thresh
    return {
        'pred_boxes': pred_dict['pred_boxes'][mask],
        'pred_scores': pred_dict['pred_scores'][mask],
        'pred_labels': pred_dict['pred_labels'][mask],
    }


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------OpenPCDet Inference -> txt 저장-------------------------')

    data_path = Path(args.data_path)

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=data_path, ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # 출력 txt는 data_path(예: 0번 폴더) 안에 그대로 저장
    output_name = args.output_name if args.output_name else f"{data_path.name}.txt"
    output_path = data_path / output_name

    lines = []

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            frame_id = data_dict['frame_id']  # collate_batch 호출 전에 미리 빼둠 (문자열 그대로 유지)
            logger.info(f'Inference sample index: \t{idx + 1} (frame_id={frame_id})')

            batch_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)

            start_time = time.time()
            pred_dicts, _ = model.forward(batch_dict)
            logger.info(f'Inference time for sample {idx + 1}: {time.time() - start_time:.4f} seconds')

            filtered = filter_by_score(pred_dicts[0], args.score_thresh)
            logger.info(f'Predictions after score_thresh={args.score_thresh}: '
                        f'{len(filtered["pred_scores"])} / {len(pred_dicts[0]["pred_scores"])}')

            pred_boxes = filtered['pred_boxes'].cpu().numpy()
            pred_labels = filtered['pred_labels'].cpu().numpy()

            for box, label in zip(pred_boxes, pred_labels):
                x, y, z, dx, dy, dz, yaw = box[:7]
                class_name = cfg.CLASS_NAMES[int(label) - 1]  # pred_labels는 1-indexed
                lines.append(
                    f"{frame_id} {class_name} {x:.4f} {y:.4f} {z:.4f} {dx:.4f} {dy:.4f} {dz:.4f} {yaw:.4f}"
                )

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
        if lines:
            f.write('\n')

    logger.info(f'결과 저장 완료: {output_path} (총 {len(lines)}개 박스)')
    logger.info('Demo done.')


if __name__ == '__main__':
    main()